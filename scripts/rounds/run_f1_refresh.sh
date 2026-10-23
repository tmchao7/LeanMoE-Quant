#!/bin/bash
# F1（+可选 F2）采纳刷新轮：把已过 A/B 的融合候选开成默认后，重测**所有**已落盘的
# graph 格。模板 = run_p2j_refresh.sh（同样的三步：归档 → 重跑 → 金丝雀/自检）。
#
# 用法（GPU 机仓库根目录）：
#   bash scripts/rounds/run_f1_refresh.sh             # F1 三项（rope/route/mask）
#   F2=1 bash scripts/rounds/run_f1_refresh.sh        # F1 + packed q/k/v（F2 须先有合规 A/B）
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
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-f1-refresh
mkdir -p "$OUT/archive"
LOG="$OUT/f1.log"; CONT="$OUT/contention.log"; LIST="$OUT/cells.txt"
: > "$LOG"; : > "$CONT"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }

# 候选开关：整轮固定，一次一个变量（F1 三项是同轮的三个变量 —— 各自 A/B 已独立测过，
# 本轮的标签必须是 combined）。
export LEMOQ_FUSED_ROPE=1 LEMOQ_FUSED_ROUTE=1 LEMOQ_SKIP_MASK_FILL=1
[ "${F2:-0}" = "1" ] && export LEMOQ_PACKED_QKV=1
echo "env: rope=$LEMOQ_FUSED_ROPE route=$LEMOQ_FUSED_ROUTE mask=$LEMOQ_SKIP_MASK_FILL packed=${LEMOQ_PACKED_QKV:-0}" >> "$LOG"

# ---- 1) 生成命令清单（从每个 payload 的 config 反推；跳过已刷新的格）----
$PY - > "$LIST" <<'PYEOF'
import glob, json, os, sys
want = ("fused_rope", "fused_route", "skip_mask_fill", "packed_qkv")
for p in sorted(glob.glob("results/EXP-*__*/result.json")):
    d = json.load(open(p))
    if d.get("status") != "done":
        continue
    c = d["config"]
    if c.get("execution") != "cudagraph":
        continue  # 只刷图格：三项融合都改图内执行
    if all(c.get(k) is True for k in want):
        continue  # 已刷新（断点续跑）
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
    # F2 还要求格是**量化**的：fp16 的 q/k/v 是 nn.Linear，install 会显式拒绝。
    quantized = (c.get("bit_width") or "fp16") != "fp16"
    packed = "1" if (os.environ.get("LEMOQ_PACKED_QKV") == "1" and kernel == "1" and quantized) else "0"
    print("%s|%s|%s|%s|%s" % (exp, joined, cell, kernel, packed))
PYEOF
echo "清单 $(wc -l < "$LIST") 格" >> "$LOG"; cat "$LIST" >> "$LOG"
if [ "${DRY:-0}" = "1" ]; then
  echo "--- DRY：只列清单（exp | args | cell | kernel | packed）---"
  cat "$LIST"
  exit 0
fi

# ---- 2) 归档 + 重跑 + 逐格自检 ----
run_one () {
  local exp=$1 args=$2 cell=$3 kernel=${4:-0} packed=${5:-0} label
  label=$(basename "$cell")
  # ★ 三个开关的生效范围不同（2026-10-23 实测）：
  #   route        —— moe_static/moe_grouped 的路由，所有格都生效；
  #   rope / mask  —— 只在 `_decode_forward_impl`（attn=kernel）里；
  #                   在别的格上开 mask 会**显式拒绝**（不是 no-op）。
  #   packed_qkv   —— 同样只在 attn=kernel 的 forward 上。
  export LEMOQ_FUSED_ROUTE=1
  if [ "$kernel" = "1" ]; then
    export LEMOQ_FUSED_ROPE=1 LEMOQ_SKIP_MASK_FILL=1
  else
    unset LEMOQ_FUSED_ROPE LEMOQ_SKIP_MASK_FILL
  fi
  if [ "$packed" = "1" ]; then export LEMOQ_PACKED_QKV=1; else unset LEMOQ_PACKED_QKV; fi
  [ -f "$OUT/archive/$label.json" ] || cp -p "$cell/result.json" "$OUT/archive/$label.json" 2>/dev/null
  snap "$label-pre"
  echo "=== [$label] $exp $args start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp "$exp" --model "$M" $args --force > "$OUT/$label.log" 2>&1
  local rc; rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  $PY - "$label" "$cell/result.json" <<'PYEOF' >> "$LOG" 2>&1
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
if kernel and os.environ.get("LEMOQ_PACKED_QKV") == "1" and cfg.get("packed_qkv") is not True:
    warn.append("packed_qkv=%r（应为 true）" % (cfg.get("packed_qkv"),))
print("    [%s] bw=%-6s attn=%-6s ctx=%-5s exec=%-9s dq=%-8s gate=%-4s p50=%-9s host=%-5s kernels=%-4s %s" % (
    lbl, cfg.get("bit_width"), cfg.get("attn"), cfg.get("ctx"), cfg.get("execution"), cfg.get("dequant"),
    d["correctness"]["gate"], p50, att["host_share_pct"], (d.get("bottleneck") or {}).get("kernels_per_step"),
    ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
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
echo "=== round_canary exit=$RC（非 0 ⇒ 整轮 VOID，不要回填任何数字）===" >> "$LOG"
exit $RC
