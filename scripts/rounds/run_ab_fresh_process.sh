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
# 纪律：只写 OUT（新目录）；canonical payload 先归档、结束（含出错）恢复；不写 result.json。
set -u
cd "${REPO_ROOT:-/root/LeanMoE-Quant}" || exit 1
PY=${PY:-/root/miniconda3/bin/python}
M=${MODEL:-/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master}
CELL=${CELL:?需要 CELL=results/EXP-...__...}
ARGS=${ARGS:?需要 ARGS="--ctx ... --bit-width ... --execution cudagraph ..."}
ENV_ON=${ENV_ON:?需要 ENV_ON，例如 "LEMOQ_PACKED_QKV=1"}
PAIRS=${PAIRS:-3}
OUT=${OUT:?需要 OUT=新目录}
EXP=$(basename "$CELL" | sed 's/__.*//')

case "$PAIRS" in ''|*[!0-9]*) echo "PAIRS 必须是正整数"; exit 2;; esac
[ "$PAIRS" -ge 2 ] || { echo "PAIRS >= 2（否则没有 per-pair 胜负）"; exit 2; }
[ -d "$CELL" ] || { echo "CELL 不存在：$CELL"; exit 2; }
[ -e "$OUT" ] && { echo "OUT 已存在（保留旧轮次证据）：$OUT"; exit 2; }

# 计划：head(off) + PAIRS 对（奇偶交替顺序）+ tail(off)  ⇒ 2 + 2*PAIRS 次
SEQ=("off")
for i in $(seq 1 "$PAIRS"); do
  if [ $((i % 2)) -eq 1 ]; then SEQ+=("off" "on"); else SEQ+=("on" "off"); fi
done
SEQ+=("off")

echo "exp=$EXP  cell=$CELL"
echo "env_on=$ENV_ON  pairs=$PAIRS  runs=${#SEQ[@]}  out=$OUT"
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
  if [ "$arm" = "on" ]; then
    env $ENV_ON "$PY" -m benchmarks.runner --exp "$EXP" $ARGS --model "$M" --force \
      > "$OUT/$label.log" 2>&1
  else
    env "$PY" -m benchmarks.runner --exp "$EXP" $ARGS --model "$M" --force \
      > "$OUT/$label.log" 2>&1
  fi
  echo "=== $label exit=$? end $(date -Is)" >> "$OUT/ab.log"
  cp -p "$CELL/result.json" "$OUT/$label.json"
  snap "$label-post"
  i=$((i + 1))
done

"$PY" - "$OUT" "$PAIRS" <<'PYEOF'
import glob, json, os, statistics, sys

out, pairs = sys.argv[1], int(sys.argv[2])
runs = []
for path in sorted(glob.glob(os.path.join(out, "[a-z][a-z][0-9][0-9].json"))):
    d = json.load(open(path))
    runs.append({
        "file": os.path.basename(path),
        "arm": "on" if os.path.basename(path).startswith("on") else "off",
        "status": d.get("status"),
        "p50": d["latency"]["per_token_p50_ms"],
        "busy": (d.get("attribution") or {}).get("gpu_busy_ms"),
        "host": (d.get("attribution") or {}).get("host_share_pct"),
        "gate": d["correctness"]["gate"],
        "kernels": (d.get("bottleneck") or {}).get("kernels_per_step"),
        "tokens": (d["correctness"] or {}).get("graph_output_tokens"),
        "switches": {k: d["config"].get(k) for k in
                     ("fused_rope", "fused_route", "skip_mask_fill", "packed_qkv")},
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
verdict = "VALID" if not reasons else "INVALID"
print("verdict:", verdict, ("; ".join(reasons) if reasons else "host<=5% / canary<=5% / tokens_equal / pairs ok"))
json.dump({"runs": runs, "off_mean_ms": statistics.mean([r["p50"] for r in off]) if off else None,
           "on_mean_ms": statistics.mean([r["p50"] for r in on]) if on else None,
           "wins": wins, "pairs": len(off), "canary_drift_pct": drift, "ratio_mad": mad,
           "tokens_equal": tokens_equal, "verdict": verdict, "reasons": reasons},
          open(os.path.join(out, "ab_summary.json"), "w"), indent=2, ensure_ascii=False)
print("wrote", os.path.join(out, "ab_summary.json"))
PYEOF
