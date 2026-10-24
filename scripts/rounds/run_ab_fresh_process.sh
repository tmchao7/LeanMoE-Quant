#!/bin/bash
# 独立进程 · 同轮相邻 A/B（2026-10-23 协议固化）。
#
# 为什么需要它：`tools/profile_graph.py --ab-*` 在**一个进程里**连续跑多次完整 session，
# 实测同一格的 host_share 会从 3.2%（进程内第一次）爬到 7.1% ⇒ 长轮次被判 VOID。
# 本脚本每臂起**独立进程**（同轮、相邻、交替顺序），canary 用同臂首尾，满足项目的
# "同轮相邻 + 每对胜负 + MAD" 判据（2026-10-23 用它拿到 F2 的合规轮：
# off 1.9536 → on 1.8597 ms，−4.81%，3/3 对，host 3.5–4.2%）。
#
# 用法（GPU 机仓库根目录）：
#   CELL=results/EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024 \
#   ARGS="--ctx 1024 --bit-width w8a16 --attn kernel --execution cudagraph --moe grouped --dequant fused" \
#   ENV_ON="LEMOQ_PACKED_QKV=1" PAIRS=3 OUT=results/EXP-f2-ab-run \
#   bash scripts/rounds/run_ab_fresh_process.sh
#
#   DRY=1 ...  只打印计划（可在 Mac 上跑，不加载模型）
#
# ★ 采纳之后（开关默认变 on）必须给 off 臂一个显式关：
#   ENV_OFF="LEMOQ_PACKED_QKV=0" PAIRS=3 ...
#   不给 ⇒ 脚本 fail-closed 拒绝（否则两臂跑同一实现，会量出一个假的 0% 收益）。
#
# 纪律：只写 OUT（新目录）；canonical payload 先归档、结束（含出错）恢复；不写 result.json。
set -u
cd "${REPO_ROOT:-/root/LeanMoE-Quant}" || exit 1
PY=${PY:-/root/miniconda3/bin/python}
M=${MODEL:-/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master}
CELL=${CELL:?需要 CELL=results/EXP-...__...}
ARGS=${ARGS:?需要 ARGS="--ctx ... --bit-width ... --execution cudagraph ..."}
ENV_ON=${ENV_ON:?需要 ENV_ON，例如 "LEMOQ_PACKED_QKV=1"}
ENV_OFF=${ENV_OFF:-}
PAIRS=${PAIRS:-3}
OUT=${OUT:?需要 OUT=新目录}
EXP=$(basename "$CELL" | sed 's/__.*//')

case "$PAIRS" in ''|*[!0-9]*) echo "PAIRS 必须是正整数"; exit 2;; esac
[ "$PAIRS" -ge 2 ] || { echo "PAIRS >= 2（否则没有 per-pair 胜负）"; exit 2; }
[ -d "$CELL" ] || { echo "CELL 不存在：$CELL"; exit 2; }
[ -e "$OUT" ] && { echo "OUT 已存在（保留旧轮次证据）：$OUT"; exit 2; }

# ★ fail-closed（2026-10-24）：采纳会把候选翻成**默认开**。那时 off 臂若不显式写 0，
#   两臂跑的是同一实现 ⇒ 静默量出"0% 收益"。这比 VOID 更坏，所以直接拒绝。
# ★ 2026-10-25：探针本身失败（PY 不存在 / import 错）也一律拒绝 —— 旧写法探针失败
#   得到空串后**静默放行**（fail-open），与本守卫的存在理由相反。
SWITCH_VAR=${ENV_ON%%=*}
# `PYTHONPATH=src`：Mac 上 lemoq 只以源码目录形式可用（没 pip install -e），
# 而 GPU 机上它是可导入的 —— 两条路都覆盖，别让守卫因为导入失败而放行。
DEFAULT_STATE=$(PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$SWITCH_VAR" <<'PYEOF'
import sys
from lemoq.kernels.fusion_switches import ENV_VARS, FUSION_DEFAULTS

var = sys.argv[1]
key = next((k for k, v in ENV_VARS.items() if v == var), None)
if key is None:
    print("unknown -")
else:
    print("on" if FUSION_DEFAULTS[key] else "off", key)
PYEOF
) || { echo "拒绝：默认状态探针失败（PY / import 问题）—— fail-closed"; exit 2; }
read -r DEFAULT_STATE SWITCH_KEY <<<"$DEFAULT_STATE"
if [ "$DEFAULT_STATE" = "unknown" ] || [ -z "$DEFAULT_STATE" ]; then
  echo "拒绝：$SWITCH_VAR 不是已声明的融合开关（见 fusion_switches.ENV_VARS）"; exit 2
fi
if [ "$DEFAULT_STATE" = "on" ] && [ -z "$ENV_OFF" ]; then
  echo "拒绝：$SWITCH_VAR 的**代码默认已经是 on**，off 臂会与 on 臂跑同一实现。"
  echo "      显式给 off 臂：ENV_OFF=\"$SWITCH_VAR=0\""
  exit 2
fi
if [ -n "$ENV_OFF" ]; then
  OFF_VAR=${ENV_OFF%%=*}
  if [ "$OFF_VAR" != "$SWITCH_VAR" ]; then
    echo "拒绝：ENV_OFF 关的是 ${OFF_VAR}，与 ENV_ON 的 ${SWITCH_VAR} 不是同一开关（两臂会同实现）"; exit 2
  fi
  case "${ENV_OFF#*=}" in 0) ;; *) echo "拒绝：ENV_OFF 必须把 $SWITCH_VAR 显式置 0"; exit 2 ;; esac
fi

# ★ F2 的两个前置（attn=kernel 且量化）与 profile_graph --ab-packed-qkv 同口径
#   （profile_graph.py:103-111）：不适用的格上 on 臂是静默 no-op ⇒ 量出假 0%。
#   显式请求的 A/B 必须拒绝，不能靠"测出来收益是 0"来发现。
if [ "$SWITCH_KEY" = "packed_qkv" ]; then
  "$PY" - "$CELL/result.json" <<'PYEOF' || exit 2
import json, sys

d = json.load(open(sys.argv[1]))
c = d.get("config") or {}
if c.get("attn") != "kernel" or (c.get("bit_width") or "fp16") == "fp16":
    print("拒绝：packed_qkv 只对 量化 × attn=kernel 生效；本格 attn=%r bit_width=%r"
          " —— on 臂会静默 no-op，量出假 0%%" % (c.get("attn"), c.get("bit_width")))
    sys.exit(1)
PYEOF
fi

# 计划：head(off) + PAIRS 对（奇偶交替顺序）+ tail(off)  ⇒ 2 + 2*PAIRS 次
SEQ=("off")
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then SEQ+=("off" "on"); else SEQ+=("on" "off"); fi
done
SEQ+=("off")

echo "exp=$EXP  cell=$CELL"
echo "env_on=$ENV_ON  env_off=${ENV_OFF:-<空>}  pairs=$PAIRS  runs=${#SEQ[@]}  out=$OUT"
for idx in "${!SEQ[@]}"; do printf '  run %d: %s\n' "$idx" "${SEQ[$idx]}"; done
if [ "${DRY:-0}" = "1" ]; then echo "--- DRY：未执行 ---"; exit 0; fi

mkdir -p "$OUT"
cp -p "$CELL/result.json" "$OUT/canonical.json"
restore () { cp -p "$OUT/canonical.json" "$CELL/result.json" 2>/dev/null || true; }
trap restore EXIT INT TERM

snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$OUT/contention.log"; }

i=0
for arm in "${SEQ[@]}"; do
  label=$(printf "%s%02d" "$arm" "$i")
  snap "$label-pre"
  echo "=== $label arm=$arm start $(date -Is)" >> "$OUT/ab.log"
  rc=0
  if [ "$arm" = "on" ]; then
    env $ENV_ON "$PY" -m benchmarks.runner --exp "$EXP" $ARGS --model "$M" --force \
      > "$OUT/$label.log" 2>&1 || rc=$?
  else
    env $ENV_OFF "$PY" -m benchmarks.runner --exp "$EXP" $ARGS --model "$M" --force \
      > "$OUT/$label.log" 2>&1 || rc=$?
  fi
  echo "=== $label exit=$rc end $(date -Is)" >> "$OUT/ab.log"
  # ★ 2026-10-25：runner 失败时不落新 payload —— 旧写法无条件 `cp` 会把**上一臂的
  #   旧 payload**（status=done）存成当前臂的结果，status 门查不出这种错配。
  #   失败就留 FAILED 标记、不采 payload，让"有效对不足"把轮判 INVALID。
  if [ "$rc" -eq 0 ] && [ -f "$CELL/result.json" ]; then
    cp -p "$CELL/result.json" "$OUT/$label.json"
  else
    echo "=== ${label} runner 失败（exit=${rc}）——不采 payload，留 FAILED 标记" >> "$OUT/ab.log"
    : > "$OUT/$label.FAILED"
  fi
  snap "$label-post"
  i=$((i + 1))
done

"$PY" - "$OUT" "$PAIRS" "$SWITCH_KEY" <<'PYEOF'
import glob, json, os, statistics, sys

out, pairs, switch_key = sys.argv[1], int(sys.argv[2]), sys.argv[3]
runs = []
# 标签是 `<arm><%02d>`（off00 / on01 / ...）：按**运行序号**排序，不能按字典序，
# 否则首尾金丝雀与配对都会错位。
def _run_label(path):
    """`off07.json` -> ('off', 7)，只看去扩展名后的运行标签；其它 json 返回 None。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    arm = "on" if stem.startswith("on") else "off" if stem.startswith("off") else None
    if arm is None or not stem[len(arm):].isdigit():
        return None
    return arm, int(stem[len(arm):])


labelled = [(pth, _run_label(pth)) for pth in glob.glob(os.path.join(out, "*.json"))]
labelled = [(pth, lab) for pth, lab in labelled if lab is not None]
labelled.sort(key=lambda item: item[1][1])
names = [os.path.splitext(os.path.basename(pth))[0] for pth, _ in labelled]
for name in names:
    path = os.path.join(out, name + ".json")
    d = json.load(open(path))
    runs.append({
        "file": name,
        "arm": "on" if name.startswith("on") else "off",
        "status": d.get("status"),
        "p50": d["latency"]["per_token_p50_ms"],
        "busy": (d.get("attribution") or {}).get("gpu_busy_ms"),
        "host": (d.get("attribution") or {}).get("host_share_pct"),
        "gate": d["correctness"]["gate"],
        "kernels": (d.get("bottleneck") or {}).get("kernels_per_step"),
        "tokens": (d["correctness"] or {}).get("graph_output_tokens"),
        # ★ 从 ENV_VARS 动态派生，不得硬编码开关名——硬编码会让新注册的开关
        #   在这里读到 None，把 VALID 轮误判成 INVALID（2026-10-25 F4 实测踩中）。
        "switches": {k: d["config"].get(k) for k in ENV_VARS},
    })

head, tail = runs[0], runs[-1]
body = runs[1:-1]
off = [r for r in body if r["arm"] == "off" and r["status"] == "done"]
on = [r for r in body if r["arm"] == "on" and r["status"] == "done"]
ratios = [o["p50"] / f["p50"] for f, o in zip(off, on)] if len(off) == len(on) else []
wins = sum(1 for f, o in zip(off, on) if o["p50"] < f["p50"])
mad = statistics.median([abs(r - statistics.median(ratios)) for r in ratios]) if len(ratios) >= 2 else None
drift = None
if head["p50"] and tail["p50"]:
    drift = 100.0 * (tail["p50"] / head["p50"] - 1.0)
# ★ runner 的 payload **不记录** greedy token 序列（只有 KL/top-1 与图内自检），
#   所以这里如实报 "not recorded"，而不是把空集判成"相等"。
token_rows = [tuple(r["tokens"]) for r in runs if r["tokens"]]
tokens_equal = (len(set(token_rows)) <= 1) if token_rows else None
hosts = [r["host"] for r in runs]

print("%-6s %-7s %-9s %-9s %-7s %-8s %s" % ("run", "arm", "p50_ms", "busy_ms", "host%", "kernels", "gate"))
for r in runs:
    print("%-6s %-7s %-9s %-9s %-7s %-8s %s" % (
        r["file"][:5], r["arm"], r["p50"], r["busy"], r["host"], r["kernels"], r["gate"]))
print()
if off and on:
    print("off mean %.4f ms  on mean %.4f ms  delta %+.4f ms (%+.2f%%)  wins %d/%d" % (
        statistics.mean([r["p50"] for r in off]), statistics.mean([r["p50"] for r in on]),
        statistics.mean([r["p50"] for r in on]) - statistics.mean([r["p50"] for r in off]),
        100 * (statistics.mean([r["p50"] for r in on]) / statistics.mean([r["p50"] for r in off]) - 1),
        wins, len(off)))
print("canary drift %s%%  tokens_equal=%s  ratio MAD %s" % (
    drift, "not recorded" if tokens_equal is None else tokens_equal, mad))
reasons = []
if any(r["status"] != "done" for r in runs): reasons.append("有 run 不是 done")
if any((r["host"] or 0) > 5 for r in runs): reasons.append("host_share > 5%")
if drift is None or abs(drift) > 5: reasons.append("canary 漂移 >5% 或缺失")
if tokens_equal is False: reasons.append("跨臂 greedy token 不同")
if len(off) != len(on) or len(off) < pairs: reasons.append("有效对不足 %d" % pairs)
if mad is None or mad > 0.02: reasons.append("比值 MAD 过大（>2%）")
# ★ 2026-10-25：逐 run 校验开关状态与臂一致 —— 堵两个静默假结果路径：
#   (a) 环境残留（默认翻 on 后 off 臂忘了显式 0 ⇒ 两臂同实现量出假 0%）；
#   (b) runner 失败后误采旧 payload（status=done 门查不出错配）。
arm_mismatch = []
for r in runs:
    want = r["arm"] == "on"
    got = r["switches"].get(switch_key)
    if got is not want:
        arm_mismatch.append("%s %s=%r（%s 臂应为 %r）" % (r["file"], switch_key, got, r["arm"], want))
if arm_mismatch:
    reasons.append("开关状态与臂不一致（" + "; ".join(arm_mismatch[:4]) + "）")
other_drift = []
for k in ENV_VARS:
    if k == switch_key:
        continue
    vals = {r["switches"].get(k) for r in runs}
    if len(vals) > 1:
        other_drift.append("%s 漂移 %r" % (k, sorted(vals, key=str)))
if other_drift:
    reasons.append("非 A/B 开关在 runs 间漂移（" + "; ".join(other_drift) + "）")
verdict = "VALID" if not reasons else "INVALID"
print("verdict:", verdict, ("; ".join(reasons) if reasons else "host<=5% / canary<=5% / tokens_equal / pairs ok"))
json.dump({"runs": runs, "off_mean_ms": statistics.mean([r["p50"] for r in off]) if off else None,
           "on_mean_ms": statistics.mean([r["p50"] for r in on]) if on else None,
           "wins": wins, "pairs": len(off), "canary_drift_pct": drift, "ratio_mad": mad,
           "tokens_equal": tokens_equal, "verdict": verdict, "reasons": reasons,
           "switch_key": switch_key, "arm_mismatch": arm_mismatch, "other_switch_drift": other_drift},
          open(os.path.join(out, "ab_summary.json"), "w"), indent=2, ensure_ascii=False)
print("wrote", os.path.join(out, "ab_summary.json"))
PYEOF
