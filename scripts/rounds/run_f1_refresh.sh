#!/bin/bash
# F1（+可选 F2）采纳刷新轮：把已过 A/B 的融合候选开成默认后，重测**所有**已落盘的
# graph 格。模板 = run_p2j_refresh.sh（同样的三步：归档 → 重跑 → 金丝雀/自检）。
#
# 用法（GPU 机仓库根目录）：
#   bash scripts/rounds/run_f1_refresh.sh             # F1 三项（rope/route/mask）
#   F2=1 bash scripts/rounds/run_f1_refresh.sh        # F1 + packed q/k/v（F2 须先有合规 A/B）
#   F2=1 KV=1 …                                       # 再叠加 F4 KV 直写（须先有合规 A/B；
#                                                        作用域=attn=kernel 的格，与 rope/mask 同）
#   F2=1 KV=1 LMHEAD=1 …                              # 再叠加 F5 lm_head w8（须先有合规 A/B；
#                                                        作用域=量化格；EXP-p1-06 门行须已落表）
#
# 前提（缺一条就不要跑）：
#   1) F1 三项：同轮 A/B 已测（8.39% / 3.0% / 1.33%）+ 自然文本检查 PASS（2026-10-23）；
#      若开 F2：F2 的**合规** A/B（host_share ≤5%）已过；
#   2) 机器安静：首尾同格 graph 金丝雀漂移 >5% ⇒ 整轮 VOID；
#   3) 覆盖前归档（本脚本幂等：已刷新的格会跳过，可断点续跑）。
#
# 产出：每格新 payload 的 config 里 fused_rope/fused_route/skip_mask_fill/packed_qkv
#       都是**有效状态**（采纳后为 true）；`result.json` 覆盖前进 archive/。
set -u
cd "${REPO_ROOT:-/root/LeanMoE-Quant}" || exit 1
PY=${PY:-/root/miniconda3/bin/python}
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-f1-refresh
DRY=${DRY:-0}

# 候选开关：整轮固定，一次一个变量（F1 三项是同轮的三个变量 —— 各自 A/B 已独立测过，
# 本轮的标签必须是 combined）。
export LEMOQ_FUSED_ROPE=1 LEMOQ_FUSED_ROUTE=1 LEMOQ_SKIP_MASK_FILL=1
[ "${F2:-0}" = "1" ] && export LEMOQ_PACKED_QKV=1
[ "${KV:-0}" = "1" ] && export LEMOQ_KV_DIRECT_WRITE=1
[ "${LMHEAD:-0}" = "1" ] && export LEMOQ_LMHEAD_W8=1

# ---- 1) 生成命令清单（从每个 payload 的 config 反推；跳过已刷新的格）----
# ★ 2026-10-24：清单先进临时文件 —— DRY 模式**零副作用**。旧写法先建 archive/、清空
#   f1.log/contention.log 再判断 DRY，Mac 自检和 GPU 上的预检都会把上一轮证据抹掉；
#   坏 payload ⇒ 非零退出中止，不以残缺清单续跑。
LIST=$(mktemp /tmp/f1_cells.XXXXXX) || exit 1
trap 'rm -f "$LIST"' EXIT
$PY - > "$LIST" <<'PYEOF'
import glob, json, os, sys

bad = []          # 读不出 / 缺 config ⇒ 清单中止（fail-closed）
not_done_cg = []  # cudagraph 形状但 status != done ⇒ 跳过，但点名（失败残留 / p0-03 by-design）
for p in sorted(glob.glob("results/EXP-*__*/result.json")):
    try:
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
    except Exception as exc:
        bad.append("%s: %s: %s" % (p, type(exc).__name__, exc))
        continue
    if d.get("status") != "done":
        if c.get("execution") == "cudagraph":
            not_done_cg.append("%s status=%s" % (os.path.dirname(p), d.get("status")))
        continue
    if c.get("execution") != "cudagraph":
        continue  # 只刷图格：三项融合都改图内执行
    cell = os.path.dirname(p)
    exp = os.path.basename(cell).split("__")[0]
    args = ["--ctx", str(c.get("ctx") or 1024)]
    if c.get("bit_width"):
        args += ["--bit-width", c["bit_width"]]
    if c.get("attn"):
        args += ["--attn", c["attn"]]
    if c.get("execution"):
        args += ["--execution", c["execution"]]
    if c.get("moe"):
        args += ["--moe", c["moe"]]
    if c.get("dequant") and c["dequant"] != "none":
        args += ["--dequant", c["dequant"]]
    if c.get("group_size"):
        args += ["--group-size", str(c["group_size"])]
    if c.get("batch"):
        args += ["--batch", str(c["batch"])]
    joined = " ".join(args)
    kernel = "1" if "--attn kernel" in joined else "0"
    # F2 还要求格是**量化**的：fp16 的 q/k/v 是 nn.Linear，装不了。
    quantized = (c.get("bit_width") or "fp16") != "fp16"
    packed = "1" if (os.environ.get("LEMOQ_PACKED_QKV") == "1" and kernel == "1" and quantized) else "0"
    # ★ 断点续跑要按**本格的期望值**判断（2026-10-24 修）：旧写法要求四个开关全 True，
    #   但 F2 只作用于 2 格，其余 21 格永远满足不了 ⇒ 重跑会整轮重来。
    # F4 期望值（2026-10-24）：KV=1 时 kernel 格必须带 kv_direct_write=True、
    #   非 kernel 格必须带 False（接线在 _decode_forward_impl ⇒ 非 kernel 格 no-op）；
    #   KV=0（旧轮）不检查——历史 payload 无该键属正常。
    kv_on = os.environ.get("LEMOQ_KV_DIRECT_WRITE") == "1"
    if kv_on:
        kv_ok = (c.get("kv_direct_write") is True) if kernel == "1" else (
            c.get("kv_direct_write") is False
        )
    else:
        kv_ok = True
    # ★ F5 期望值：LMHEAD=1 时量化格必须带 lm_head_w8_effective=True、fp16 格必须 False
    #   （作用域=量化格，与 run_one 的 args 判据一致）；LMHEAD=0 不检查——历史 payload 无该键属正常。
    lh_on = os.environ.get("LEMOQ_LMHEAD_W8") == "1"
    if lh_on:
        lh_ok = (c.get("lm_head_w8_effective") is True) if quantized else (
            c.get("lm_head_w8_effective") is False
        )
    else:
        lh_ok = True
    refreshed = (
        c.get("fused_route") is True
        and (kernel != "1" or (c.get("fused_rope") is True and c.get("skip_mask_fill") is True))
        and ((c.get("packed_qkv_effective") is True) if packed == "1"
             else (c.get("packed_qkv_effective") in (False, None)))
        and kv_ok
        and lh_ok
    )
    if refreshed:
        continue
    print("%s|%s|%s|%s|%s" % (exp, joined, cell, kernel, packed))
if bad:
    print("清单中止：以下 payload 读不出 / 缺 config：", file=sys.stderr)
    for b in bad:
        print("  " + b, file=sys.stderr)
    sys.exit(1)
if not_done_cg:
    print("跳过的非 done cudagraph 格（失败残留 / p0-03 by-design）：", file=sys.stderr)
    for x in not_done_cg:
        print("  " + x, file=sys.stderr)
PYEOF
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "拒绝：清单生成失败（坏 payload / IO）——不以残缺清单续跑" >&2
  exit 2
fi
echo "--- 清单（exp | args | cell | kernel | packed）---"
cat "$LIST"
N_CELLS=$(wc -l < "$LIST" | tr -d ' ')
echo "清单 $N_CELLS 格"
if [ "$DRY" = "1" ]; then echo "--- DRY：只列清单，未写 results/ ---"; exit 0; fi
[ "$N_CELLS" -gt 0 ] || { echo "拒绝：清单为空（results/ 下没有待刷新的 cudagraph done 格）"; exit 2; }

# ---- 非 DRY：从这里才开始写 results/（其上零副作用）----
mkdir -p "$OUT/archive"
LOG="$OUT/f1.log"; CONT="$OUT/contention.log"; CELLS="$OUT/cells.txt"
: > "$LOG"; : > "$CONT"
cp -p "$LIST" "$CELLS"
FAILS="$OUT/failed.txt"; : > "$FAILS"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }
echo "env: rope=$LEMOQ_FUSED_ROPE route=$LEMOQ_FUSED_ROUTE mask=$LEMOQ_SKIP_MASK_FILL packed=${LEMOQ_PACKED_QKV:-0} kv=${LEMOQ_KV_DIRECT_WRITE:-0} lmhead=${LEMOQ_LMHEAD_W8:-0}" >> "$LOG"
echo "清单 $N_CELLS 格" >> "$LOG"; cat "$LIST" >> "$LOG"

# ---- 2) 归档 + 重跑 + 逐格自检 ----
run_one () {
  local exp=$1 args=$2 cell=$3 kernel=${4:-0} packed=${5:-0} label
  label=$(basename "$cell")
  # ★ 四个开关的生效范围不同（2026-10-23 实测）：
  #   route        —— moe_static/moe_grouped 的路由，所有格都生效；
  #   rope / mask  —— 只在 `_decode_forward_impl`（attn=kernel）里；在别的格上开 mask
  #                   会**显式拒绝**（不是 no-op），所以非 kernel 格必须显式写 0；
  #   packed_qkv   —— 两个前置：attn=kernel **且** 该格已量化（否则 no-op）。
  # ★ 全部**显式写 0/1**（不只 unset）：采纳把默认翻成 True 之后，`unset` 会变成"开"，
  #   本轮的开关组合就不再由脚本决定（2026-10-24）。
  export LEMOQ_FUSED_ROUTE=1
  if [ "$kernel" = "1" ]; then
    export LEMOQ_FUSED_ROPE=1 LEMOQ_SKIP_MASK_FILL=1
  else
    export LEMOQ_FUSED_ROPE=0 LEMOQ_SKIP_MASK_FILL=0
  fi
  if [ "$packed" = "1" ]; then export LEMOQ_PACKED_QKV=1; else export LEMOQ_PACKED_QKV=0; fi
  # F4 KV 直写：接线在 _decode_forward_impl ⇒ 作用域=kernel 格；非 kernel 格显式写 0。
  if [ "$kernel" = "1" ] && [ "${KV:-0}" = "1" ]; then export LEMOQ_KV_DIRECT_WRITE=1; else export LEMOQ_KV_DIRECT_WRITE=0; fi
  # F5 lm_head w8：作用域=**量化格**（fp16 没有量化语义）。从 args 判量化 —— 注意
  #   fp16 格的 args 里同样有 `--bit-width`（值是 fp16），所以必须匹配**非 fp16 位宽**，
  #   不能只查键名。fp16 格显式写 0："本轮组合由脚本决定"对每个开关都成立。
  if [ "${LMHEAD:-0}" = "1" ] && printf '%s' "$args" | grep -Eq -- '--bit-width (w[0-9]+a[0-9]+|mxfp4)'; then
    export LEMOQ_LMHEAD_W8=1
  else
    export LEMOQ_LMHEAD_W8=0
  fi
  [ -f "$OUT/archive/$label.json" ] || cp -p "$cell/result.json" "$OUT/archive/$label.json" 2>/dev/null
  snap "$label-pre"
  echo "=== [$label] $exp $args start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp "$exp" --model "$M" $args --force > "$OUT/$label.log" 2>&1
  # ★ 2026-10-24：必须同行声明 —— 旧写法 `local rc; rc=$?` 里 `local rc` 会把 $?
  #   重置为 0，runner 的真实退出码从未进过日志/判定（实测 bash 语义）。
  local rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  local check rc2=0
  check=$($PY - "$label" "$cell/result.json" <<'PYEOF'
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)); lat, att, cfg = d["latency"], d["attribution"], d["config"]
ev = (d.get("_measurement") or {}).get("gate_failed_evidence") or {}
p50 = lat["per_token_p50_ms"] if lat["per_token_p50_ms"] is not None else ev.get("per_token_p50_ms")
warn = []
if d["status"] != "done": warn.append("status=%s" % d["status"])
if (att["host_share_pct"] or 0) > 5: warn.append("host_share=%.1f%%>5%%" % att["host_share_pct"])
if cfg.get("fused_route") is not True: warn.append("fused_route=%r（应为 true）" % (cfg.get("fused_route"),))
kernel = str(cfg.get("attn")) == "kernel"
if kernel and cfg.get("fused_rope") is not True: warn.append("fused_rope=%r（应为 true）" % (cfg.get("fused_rope"),))
if kernel and cfg.get("skip_mask_fill") is not True: warn.append("skip_mask_fill=%r（应为 true）" % (cfg.get("skip_mask_fill"),))
import os
# ★ F2 有两个前置（attn=kernel 且量化）⇒ 自检读**生效**字段，不读"请求"。
#   旧写法读 config.packed_qkv（现在是请求状态），采纳后 21 格会是 True 而实际没装。
eff = cfg.get("packed_qkv_effective")
if kernel and os.environ.get("LEMOQ_PACKED_QKV") == "1":
    if eff is not True:
        warn.append("packed_qkv_effective=%r（应为 true）" % (eff,))
elif eff:
    warn.append("packed_qkv_effective=%r（本格不该装上）" % (eff,))
# ★ F5 生效检查：量化格（bit_width 非 fp16/None）且 LMHEAD=1 ⇒ lm_head_w8_effective
#   必须 True；其余（fp16 格 / 本轮未开 LMHEAD）必须 False/None。
eff_lh = cfg.get("lm_head_w8_effective")
lm_quantized = cfg.get("bit_width") not in (None, "fp16")
if os.environ.get("LEMOQ_LMHEAD_W8") == "1" and lm_quantized:
    if eff_lh is not True:
        warn.append("lm_head_w8_effective=%r（应为 true）" % (eff_lh,))
elif eff_lh:
    warn.append("lm_head_w8_effective=%r（本格不该装上）" % (eff_lh,))
print("    [%s] bw=%-6s attn=%-6s ctx=%-5s exec=%-9s dq=%-8s norm=%-5s gate=%-4s p50=%-9s host=%-5s kernels=%-4s %s" % (
    lbl, cfg.get("bit_width"), cfg.get("attn"), cfg.get("ctx"), cfg.get("execution"), cfg.get("dequant"),
    cfg.get("rmsnorm"),
    d["correctness"]["gate"], p50, att["host_share_pct"], (d.get("bottleneck") or {}).get("kernels_per_step"),
    ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
) || rc2=$?
  printf '%s\n' "$check" >> "$LOG"
  # ★ 2026-10-24：失败格不再静默 —— runner 非零退出、自检崩溃或自检有警告 ⇒ 记入
  #   failed.txt，轮末对账（清单数 = 干净刷新数 + 失败数），不干净就不许回填数字。
  if [ "$rc" -ne 0 ] || [ "$rc2" -ne 0 ] || printf '%s' "$check" | grep -q '!! '; then
    echo "$label" >> "$FAILS"
  fi
  snap "$label-post"
}

# ---- 3) 首尾同格金丝雀 + 23 格 ----
CANARY_CELL=results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024
CANARY_EXP=EXP-p4b-01-truncation
CANARY_ARGS="--ctx 1024 --bit-width w8a16 --execution cudagraph --moe grouped --dequant fused"
run_one "$CANARY_EXP" "$CANARY_ARGS" "$CANARY_CELL" 0 0
cp -p "$CANARY_CELL/result.json" "$OUT/canary_head.json" 2>/dev/null || true

while IFS='|' read -r exp args cell kernel packed; do
  [ -z "${exp:-}" ] && continue
  [ "$cell" = "$CANARY_CELL" ] && continue   # 金丝雀已跑过；尾金丝雀统一放最后
  run_one "$exp" "$args" "$cell" "${kernel:-0}" "${packed:-0}"
done < "$LIST"

run_one "$CANARY_EXP" "$CANARY_ARGS" "$CANARY_CELL" 0 0
cp -p "$CANARY_CELL/result.json" "$OUT/canary_tail.json" 2>/dev/null || true

# ---- 4) 判定：金丝雀漂移（fail-closed，非零退出 = 整轮 VOID）----
$PY -m tools.round_canary "$OUT/canary_head.json" "$OUT/canary_tail.json" --graph \
  --out "$OUT/canary.json" >> "$LOG" 2>&1
RC=$?
echo "=== round_canary exit=${RC}（非 0 ⇒ 整轮 VOID，不要回填任何数字）===" >> "$LOG"
# ---- 5) 对账（2026-10-24）：清单数 = 干净刷新数 + 失败数；失败格不静默 ----
N_FAILED=$(wc -l < "$FAILS" | tr -d ' ')
echo "=== 对账：清单 $N_CELLS 格，失败/自检警告 $N_FAILED 格 ===" >> "$LOG"
if [ "$N_FAILED" -gt 0 ]; then
  echo "对账：以下格未干净刷新（不要据此轮回填它们的数字；修因后可断点续跑）：" >&2
  cat "$FAILS" >&2
fi
if [ "$RC" -ne 0 ]; then exit "$RC"; fi
if [ "$N_FAILED" -gt 0 ]; then exit 3; fi
exit 0
