#!/bin/bash
# EXP-p4c-01 第二轮（v2）：投影 kernel 在 7d1f92b 改成"按组广播 scale"后重测同样 8 格。
# v1 的工件已归档到 results/EXP-p4c-01-run/v1_archive/（规则 9：覆盖前先归档、记录里标 v1/v2）。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4c-01-run
LOG="$OUT/p4c01-v2.log"; CONT="$OUT/p4c01-v2-contention.log"
: > "$LOG"; : > "$CONT"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }

check () {
  $PY - "$1" "$2" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)); lat, att, cfg = d["latency"], d["attribution"], d["config"]
ev = (d.get("_measurement") or {}).get("gate_failed_evidence") or {}
p50 = lat["per_token_p50_ms"] if lat["per_token_p50_ms"] is not None else ev.get("per_token_p50_ms")
tk = d["bottleneck"]["top_kernels"] or []
proj = sum(k["self_ms_per_token"] or 0 for k in tk if "_dequant_gemm_kernel" in k["name"])
warn = []
if d["status"] != "done": warn.append("status=%s" % d["status"])
if (att["host_share_pct"] or 0) > 5: warn.append("host_share=%.1f%%>5%%" % att["host_share_pct"])
if p50 is None: warn.append("p50=None")
print("    [%s] attn=%-6s bw=%-6s ctx=%-5s gate=%-4s p50=%-9s host=%-5s busy=%-7s proj=%.4fms %s" % (
    lbl, cfg["attn"], cfg["bit_width"], cfg["ctx"], d["correctness"]["gate"], p50,
    att["host_share_pct"], round(att["gpu_busy_ms"] or -1, 3), proj,
    ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
}

cell () {
  local bw=$1 at=$2 cx=$3 label=$4 dq=() pre
  [ "$bw" != "fp16" ] && dq=(--dequant fused)
  pre=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  snap "$label-pre"
  [ -n "$pre" ] && echo "    [$label] !! PRE 占用: $pre" >> "$LOG"
  echo "=== [$label] bw=$bw attn=$at ctx=$cx start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp EXP-p4c-01-in-graph-attention --model "$M" \
      --bit-width "$bw" --attn "$at" --ctx "$cx" "${dq[@]}" --force \
      > "$OUT/v5_${label}.log" 2>&1
  echo "=== [$label] exit=$? end $(date -Is)" >> "$LOG"
  cp "results/EXP-p4c-01-in-graph-attention__bit_width=${bw}_attn=${at}_ctx=${cx}/result.json" \
     "$OUT/v5_${label}_result.json" 2>/dev/null
  check "$label" "$OUT/v5_${label}_result.json"
  snap "$label-post"
}

cell fp16  sdpa   1024 canaryA_fp16_sdpa_1024
cell w8a16 sdpa   1024 w8a16_sdpa_1024
cell w8a16 kernel 1024 w8a16_kernel_1024
cell fp16  kernel 1024 fp16_kernel_1024
cell w8a16 kernel 4096 w8a16_kernel_4096
cell fp16  kernel 4096 fp16_kernel_4096
cell fp16  sdpa   4096 fp16_sdpa_4096
cell w8a16 sdpa   4096 w8a16_sdpa_4096
cell fp16  sdpa   1024 canaryB_fp16_sdpa_1024
echo "ALL DONE $(date -Is)" >> "$LOG"
