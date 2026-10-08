#!/bin/bash
# EXP-p4-01（E4 路由开销）申报的**唯一一格**：fp16 / eager / hf 路径 / ctx=1024。
# 无轴 → cell id 就是 base id。为拿同一轮的对照，另跑 fp16 × grouped 与 w8a16 × grouped
# 两格（它们是别的实验的 cell，只作**跨实验参照**并显式标注；本实验自己的数字只取申报格）。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4-01-run
LOG="$OUT/p401-run.log"; CONT="$OUT/p401-contention.log"
: > "$LOG"; : > "$CONT"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }

show () {
  $PY - "$1" "$2" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)); a = d["attribution"]; b = d["bottleneck"]
print("    [%s] status=%s gate=%s p50=%s host=%s%% busy=%.3f routing_share=%s%% bucket_ms=%s" % (
    lbl, d["status"], d["correctness"]["gate"], d["latency"]["per_token_p50_ms"],
    a["host_share_pct"], a["gpu_busy_ms"] or 0, a["routing_share_pct"],
    json.dumps(b["bucket_ms"], ensure_ascii=False)))
PYEOF
}

run_cell () {   # label exp extra-args...
  local label=$1 exp=$2; shift 2
  snap "$label-pre"
  echo "=== [$label] exp=$exp $* start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp "$exp" --model "$M" --ctx 1024 --force "$@" \
      > "$OUT/p401_${label}.log" 2>&1
  echo "=== [$label] exit=$? end $(date -Is)" >> "$LOG"
  snap "$label-post"
}

# 申报格（A/B 两次 = 同格金丝雀）
run_cell declared_A EXP-p4-01-moe-routing-overhead
cp results/EXP-p4-01-moe-routing-overhead/result.json "$OUT/p401_declared_A_result.json" 2>/dev/null
show declared_A "$OUT/p401_declared_A_result.json"
run_cell declared_B EXP-p4-01-moe-routing-overhead
cp results/EXP-p4-01-moe-routing-overhead/result.json "$OUT/p401_declared_B_result.json" 2>/dev/null
show declared_B "$OUT/p401_declared_B_result.json"
# 参照：grouped fp16 / grouped w8a16（同一轮、同一进程边界条件）
run_cell grouped_fp16 EXP-p4b-01-truncation --bit-width fp16 --execution eager
cp "results/EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=1024/result.json" "$OUT/p401_grouped_fp16_result.json" 2>/dev/null
show grouped_fp16 "$OUT/p401_grouped_fp16_result.json"
run_cell grouped_w8a16 EXP-p4b-01-truncation --bit-width w8a16 --execution eager
cp "results/EXP-p4b-01-truncation__bit_width=w8a16_execution=eager_ctx=1024/result.json" "$OUT/p401_grouped_w8a16_result.json" 2>/dev/null
show grouped_w8a16 "$OUT/p401_grouped_w8a16_result.json"
echo "ALL DONE $(date -Is)" >> "$LOG"
