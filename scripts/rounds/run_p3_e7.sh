#!/bin/bash
# P3 / E7 轮（EXP-p4b-01）：同一轮内相邻执行 eager/cudagraph，首尾金丝雀，逐格占用快照。
# 协议（EXP-p4b-01 记录 §5 K2）：K2.1 相邻执行 + 首尾金丝雀；K2.2 计时前查占用、记录不杀。
# ★ 本轮会 --force 重跑 6 个 v1 已落盘的格子（协议升级：同轮相邻 + 量化格交错 A/B），
#   先把 v1 的 result.json 归档到 results/EXP-p4b-01-run/v1_archive/。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4b-01-run
mkdir -p "$OUT/v1_archive"
LOG="$OUT/p3-e7.log"
: > "$OUT/p3-contention.log"

for cell in fp16_execution=eager_ctx=1024 fp16_execution=cudagraph_ctx=1024 \
            fp16_execution=cudagraph_ctx=4096 w8a16_execution=eager_ctx=1024 \
            w8a16_execution=eager_ctx=4096 w8a16_execution=cudagraph_ctx=1024; do
  src="results/EXP-p4b-01-truncation__bit_width=$cell/result.json"
  [ -f "$src" ] && cp "$src" "$OUT/v1_archive/$(echo "$cell" | tr '=' '_').json"
done

snap () { { echo "--- $1 $(date -Is)";
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } \
          >> "$OUT/p3-contention.log" 2>&1; }

cell () {  # $1=bw $2=execution $3=ctx $4=label [$5=dequant]
  local bw=$1 ex=$2 cx=$3 label=$4 dq=${5:-none}
  snap "$label-pre"
  echo "=== [$label] bw=$bw exec=$ex ctx=$cx dequant=$dq start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp EXP-p4b-01-truncation --model "$M" \
      --bit-width "$bw" --execution "$ex" --ctx "$cx" --dequant "$dq" --force \
      > "$OUT/p3_${label}.log" 2>&1
  local rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  cp "results/EXP-p4b-01-truncation__bit_width=${bw}_execution=${ex}_ctx=${cx}/result.json" \
     "$OUT/p3_${label}_result.json" 2>/dev/null
  snap "$label-post"
}

cell fp16  eager     1024 canaryA_fp16_eager_1024
cell fp16  cudagraph 1024 fp16_cg_1024
cell w8a16 eager     1024 w8a16_eager_1024
cell w8a16 cudagraph 1024 w8a16_cg_fused_1024 fused
cell w8a16 cudagraph 1024 w8a16_cg_1024
cell fp16  eager     4096 fp16_eager_4096
cell fp16  cudagraph 4096 fp16_cg_4096
cell w8a16 eager     4096 w8a16_eager_4096
cell w8a16 cudagraph 4096 w8a16_cg_fused_4096 fused
cell w8a16 cudagraph 4096 w8a16_cg_4096
cell w8a8  cudagraph 1024 w8a8_cg_1024
cell w8a8  cudagraph 4096 w8a8_cg_4096
cell fp16  eager     1024 canaryB_fp16_eager_1024
echo "ALL DONE $(date -Is)" >> "$LOG"
