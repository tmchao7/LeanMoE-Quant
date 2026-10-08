#!/bin/bash
# P2d：kernel 7d1f92b（投影按组广播 scale）之后，重跑**受影响的格子**并归档 v1。
# 受影响 = 量化 且 dequant=fused（separate 臂不经过这个 kernel，fp16 格不过量化）。
#   p2-01：3 个 fused 格 + 3 个 separate 格（后者是**对照**：应当不变 ⇒ 顺便检验这轮没跑偏）
#   p4b  ：w8a16 × cudagraph × {1024,4096} 的 fused 变体（E7 头号结论的分子）
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/P2d-run; LOG="$OUT/p2d.log"; CONT="$OUT/p2d-contention.log"
mkdir -p "$OUT"; : > "$LOG"; : > "$CONT"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }
show () {
  $PY - "$1" "$2" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)); a = d["attribution"]; c = d["config"]
ev = (d.get("_measurement") or {}).get("gate_failed_evidence") or {}
p50 = d["latency"]["per_token_p50_ms"] or ev.get("per_token_p50_ms")
warn = []
if (a["host_share_pct"] or 0) > 95: warn.append("host=%.1f%%" % a["host_share_pct"])
print("    [%s] bw=%-6s deq=%-8s exec=%-9s ctx=%-5s gate=%-4s p50=%-10s host=%-6s%s" % (
    lbl, c["bit_width"], c["dequant"], c["execution"], c["ctx"], d["correctness"]["gate"], p50,
    a["host_share_pct"], ("  " + "; ".join(warn)) if warn else ""))
PYEOF
}
run () {  # label exp args...
  local label=$1 exp=$2; shift 2
  snap "$label-pre"
  echo "=== [$label] $* start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp "$exp" --model "$M" --force "$@" > "$OUT/${label}.log" 2>&1
  echo "=== [$label] exit=$? end $(date -Is)" >> "$LOG"
  snap "$label-post"
}

# ---- p4b：E7 的 fused 变体（canary = 同格首尾各一次）----
run p4b_fused_1024_A EXP-p4b-01-truncation --bit-width w8a16 --execution cudagraph --ctx 1024 --dequant fused
cp results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024/result.json "$OUT/p4b_fused_1024_A.json"
show p4b_fused_1024_A "$OUT/p4b_fused_1024_A.json"
run p4b_fused_4096 EXP-p4b-01-truncation --bit-width w8a16 --execution cudagraph --ctx 4096 --dequant fused
cp results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=4096/result.json "$OUT/p4b_fused_4096.json"
show p4b_fused_4096 "$OUT/p4b_fused_4096.json"
run p4b_fused_1024_B EXP-p4b-01-truncation --bit-width w8a16 --execution cudagraph --ctx 1024 --dequant fused
cp results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024/result.json "$OUT/p4b_fused_1024_B.json"
show p4b_fused_1024_B "$OUT/p4b_fused_1024_B.json"

# ---- p2-01：6 格全刷（fused 受影响、separate 是对照）----
for bw in w8a16 w8a8 w4a16; do
  for dq in separate fused; do
    run "p2-01_${bw}_${dq}" EXP-p2-01-dequant-fusion --bit-width "$bw" --dequant "$dq"
    cp "results/EXP-p2-01-dequant-fusion__bit_width=${bw}_dequant=${dq}/result.json" "$OUT/p2-01_${bw}_${dq}.json"
    show "p2-01_${bw}_${dq}" "$OUT/p2-01_${bw}_${dq}.json"
  done
done
echo "ALL DONE $(date -Is)" >> "$LOG"
