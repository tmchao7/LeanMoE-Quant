#!/bin/bash
# P3 / E7 轮 v2（干净卡重跑）：第 1 轮在 19:17:03–19:22:10 被外人进程 31929（峰值 14 GB）污染，
# 出现一次 capture OOM 与一个 3× 离群（fp16×cudagraph×4096 = 22.06 ms vs v1 的 7.31 ms）。
# 协议：记录不杀、等它退出后重跑。本轮加两件事：
#   (a) cudagraph 臂自己的**同格首尾金丝雀**（w8a16×cudagraph×1024 跑两次）——
#       eager 金丝雀对这种污染不敏感（第 1 轮的 eager 金丝雀只漂 1.16%）；
#   (b) 每格后做一次 payload 自检：graph 格的 host_share >5% 就是"有共租户"的硬信号。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4b-01-run
LOG="$OUT/p3-e7-v2.log"
CONT="$OUT/p3-v2-contention.log"
: > "$LOG"; : > "$CONT"
W8=w8a16_execution=cudagraph_ctx=1024

snap () { { echo "--- $1 $(date -Is)";
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } \
          >> "$CONT" 2>&1; }

check () {  # $1=label  $2=result.json 路径
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
    warn.append("host_share=%.1f%%>5%%（graph 格可疑：共租户）" % att["host_share_pct"])
if p50 is None:
    warn.append("p50=None")
print("    [%s] status=%s gate=%s p50=%s host=%s%% W=%s %s" % (
    lbl, d["status"], d["correctness"]["gate"], p50, att["host_share_pct"],
    d["bytes"]["W_total_bytes"], ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
}

cell () {  # $1=bw $2=execution $3=ctx $4=label [$5=dequant]
  local bw=$1 ex=$2 cx=$3 label=$4 dq=${5:-none} pre
  pre=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  snap "$label-pre"
  [ -n "$pre" ] && echo "    [$label] !! PRE 占用: $pre" >> "$LOG"
  echo "=== [$label] bw=$bw exec=$ex ctx=$cx dequant=$dq start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp EXP-p4b-01-truncation --model "$M" \
      --bit-width "$bw" --execution "$ex" --ctx "$cx" --dequant "$dq" --force \
      > "$OUT/v2_${label}.log" 2>&1
  local rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  cp "results/EXP-p4b-01-truncation__bit_width=${bw}_execution=${ex}_ctx=${cx}/result.json" \
     "$OUT/v2_${label}_result.json" 2>/dev/null
  check "$label" "$OUT/v2_${label}_result.json"
  snap "$label-post"
}

cell fp16  eager     1024 v2_canaryA_fp16_eager_1024
cell w8a16 cudagraph 1024 v2_canaryA_w8a16_cg_1024
cell fp16  cudagraph 1024 v2_fp16_cg_1024
cell w8a16 eager     1024 v2_w8a16_eager_1024
cell w8a16 cudagraph 1024 v2_w8a16_cg_fused_1024 fused
cell fp16  eager     4096 v2_fp16_eager_4096
cell fp16  cudagraph 4096 v2_fp16_cg_4096
cell w8a16 eager     4096 v2_w8a16_eager_4096
cell w8a16 cudagraph 4096 v2_w8a16_cg_fused_4096 fused
cell w8a16 cudagraph 4096 v2_w8a16_cg_4096
cell w8a8  cudagraph 1024 v2_w8a8_cg_1024
cell w8a8  cudagraph 4096 v2_w8a8_cg_4096
cell fp16  eager     1024 v2_canaryB_fp16_eager_1024
cell w8a16 cudagraph 1024 v2_canaryB_w8a16_cg_1024
echo "ALL DONE $(date -Is)" >> "$LOG"
