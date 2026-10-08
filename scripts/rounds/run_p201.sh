#!/bin/bash
# EXP-p2-01 的 6 格：首尾金丝雀 + 逐格占用快照；两臂同进程交错由 runner 内部完成。
# 协议（EXP-p2-01 记录 §2 K3）：
#   - 计时前查 nvidia-smi --query-compute-apps；有外人进程只记录、绝不杀；
#   - 首尾金丝雀 >5% 则整轮作废（只记录占用）。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p2-01-run
mkdir -p "$OUT"
LOG="$OUT/p201.log"
: > "$OUT/contention.log"

snap () { { echo "--- $1 $(date -Is)";
            nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } \
          >> "$OUT/contention.log" 2>&1; }

cell () {  # $1=bit_width $2=dequant $3=label
  snap "$3-pre"
  echo "=== [$3] bit_width=$1 dequant=$2 start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp EXP-p2-01-dequant-fusion --model "$M" \
      --bit-width "$1" --dequant "$2" --ctx 1024 --force > "$OUT/p201_${3}.log" 2>&1
  rc=$?
  echo "=== [$3] exit=$rc end $(date -Is)" >> "$LOG"
  cp "results/EXP-p2-01-dequant-fusion__bit_width=$1_dequant=$2/result.json" \
     "$OUT/p201_${3}_result.json" 2>/dev/null
  snap "$3-post"
}

cell w8a16 separate canaryA
cell w8a16 fused    w8a16_fused
cell w8a8  separate w8a8_separate
cell w8a8  fused    w8a8_fused
cell w4a16 separate w4a16_separate
cell w4a16 fused    w4a16_fused
cell w8a16 separate canaryB
echo "ALL DONE $(date -Is)" >> "$LOG"
