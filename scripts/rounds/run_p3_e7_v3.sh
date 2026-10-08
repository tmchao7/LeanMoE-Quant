#!/bin/bash
# P3/E7 轮 v3：补齐 evidence 格（w4a16 的 4 格 + w8a8 eager 的 2 格），带两张金丝雀与占用自检。
# ★ w4a16 / w8a8 的引用门 FAIL → 延迟按硬规则 2 只进 gate_failed_evidence（latency 段为 null）。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4b-01-run
LOG="$OUT/p3-e7-v3.log"; CONT="$OUT/p3-v3-contention.log"
: > "$LOG"; : > "$CONT"

snap () { { echo "--- $1 $(date -Is)";
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }

check () {
  $PY - "$1" "$2" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(path))
except Exception as exc:
    print("    [%s] !! payload 读不到: %s" % (lbl, exc)); raise SystemExit(0)
lat, att = d["latency"], d["attribution"]
ev = (d.get("_measurement") or {}).get("gate_failed_evidence") or {}
p50 = lat["per_token_p50_ms"] if lat["per_token_p50_ms"] is not None else ev.get("per_token_p50_ms")
warn = []
if d["status"] != "done":
    warn.append("status=%s" % d["status"])
if d["config"]["execution"] == "cudagraph" and (att["host_share_pct"] or 0) > 5:
    warn.append("host_share=%.1f%%>5%%" % att["host_share_pct"])
if p50 is None:
    warn.append("p50=None")
print("    [%s] status=%s gate=%s p50=%s host=%s%% W=%s %s" % (
    lbl, d["status"], d["correctness"]["gate"], p50, att["host_share_pct"],
    d["bytes"]["W_total_bytes"], ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
}

cell () {
  local bw=$1 ex=$2 cx=$3 label=$4 dq=${5:-none} pre
  pre=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  snap "$label-pre"
  [ -n "$pre" ] && echo "    [$label] !! PRE 占用: $pre" >> "$LOG"
  echo "=== [$label] bw=$bw exec=$ex ctx=$cx dequant=$dq start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp EXP-p4b-01-truncation --model "$M" \
      --bit-width "$bw" --execution "$ex" --ctx "$cx" --dequant "$dq" --force \
      > "$OUT/v3_${label}.log" 2>&1
  local rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  cp "results/EXP-p4b-01-truncation__bit_width=${bw}_execution=${ex}_ctx=${cx}/result.json" \
     "$OUT/v3_${label}_result.json" 2>/dev/null
  check "$label" "$OUT/v3_${label}_result.json"
  snap "$label-post"
}

cell fp16  eager     1024 v3_canaryA_fp16_eager_1024
cell w8a16 cudagraph 1024 v3_canaryA_w8a16_cg_1024
cell w4a16 eager     1024 v3_w4a16_eager_1024
cell w4a16 cudagraph 1024 v3_w4a16_cg_fused_1024 fused
cell w4a16 cudagraph 1024 v3_w4a16_cg_1024
cell w8a8  eager     1024 v3_w8a8_eager_1024
cell w4a16 eager     4096 v3_w4a16_eager_4096
cell w4a16 cudagraph 4096 v3_w4a16_cg_4096
cell w8a8  eager     4096 v3_w8a8_eager_4096
cell fp16  eager     1024 v3_canaryB_fp16_eager_1024
cell w8a16 cudagraph 1024 v3_canaryB_w8a16_cg_1024
echo "ALL DONE $(date -Is)" >> "$LOG"
