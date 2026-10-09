#!/bin/bash
# EXP-p4b-01 克隆机 gate 轮（clone1）：新实例上的最小正确性门 + 金丝雀。
# ★ 目的：克隆实例复现性 + 本分支改动（dtype 守卫 / ppl_note / schema）在真路径上的回归门。
#   覆盖 4 个主目录 result.json 之前，已整目录归档到 clone1_archive_pre/（规则 9）。
#   带宽数据以 results/machine_probe.json（本实例 1529.5 GB/s）为准，不跨轮。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p4b-01-run
ARC="$OUT/clone1_archive_pre"
LOG="$OUT/p4b-clone1.log"; CONT="$OUT/p4b-clone1-contention.log"
mkdir -p "$ARC"
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
if d["status"] != "done": warn.append("status=%s" % d["status"])
if d["config"]["execution"] == "cudagraph" and (att["host_share_pct"] or 0) > 5:
    warn.append("host_share=%.1f%%>5%%" % att["host_share_pct"])
if p50 is None: warn.append("p50=None")
# PPL 声明（G5 前置）：ppl_* 全 null 必须有声明
corr = d["correctness"]
if all(corr.get(k) is None for k in ("ppl_fp16","ppl_this","ppl_delta")) and not corr.get("ppl_note"):
    warn.append("ppl_* 全 null 且无 ppl_note")
print("    [%s] status=%s gate=%s kl=%s top1=%s p50=%s host=%s%% %s" % (
    lbl, d["status"], d["correctness"]["gate"], d["correctness"]["kl"],
    d["correctness"]["top1_agreement"], p50, att["host_share_pct"],
    ("!! " + "; ".join(warn)) if warn else "OK"))
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
      > "$OUT/clone1_${label}.log" 2>&1
  local rc=$?
  echo "=== [$label] exit=$rc end $(date -Is)" >> "$LOG"
  cp "results/EXP-p4b-01-truncation__bit_width=${bw}_execution=${ex}_ctx=${cx}/result.json" \
     "$OUT/clone1_${label}_result.json" 2>/dev/null
  check "$label" "$OUT/clone1_${label}_result.json"
  snap "$label-post"
}

# ---- 归档将被覆盖的主目录（规则 9）----
for d in \
  "EXP-p4b-01-truncation__bit_width=fp16_execution=eager_ctx=1024" \
  "EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024" \
  "EXP-p4b-01-truncation__bit_width=w8a16_execution=eager_ctx=1024"; do
  if [ -d "results/$d" ] && [ ! -d "$ARC/$d" ]; then
    cp -r "results/$d" "$ARC/$d" && echo "archived: $d" >> "$LOG"
  fi
done

cell fp16  eager     1024 canaryA_fp16_eager_1024
cell w8a16 cudagraph 1024 w8a16_cg_fused_1024 fused
cell w8a16 eager     1024 w8a16_eager_1024
cell fp16  eager     1024 canaryB_fp16_eager_1024

# ---- 金丝雀漂移验收（≤5% 否则整轮 VOID）----
$PY - "$OUT" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
out = sys.argv[1]
a = json.load(open(out + "/clone1_canaryA_fp16_eager_1024_result.json"))
b = json.load(open(out + "/clone1_canaryB_fp16_eager_1024_result.json"))
pa = a["latency"]["per_token_p50_ms"]; pb = b["latency"]["per_token_p50_ms"]
if pa is None or pb is None:
    print("    [canary] !! 金丝雀 p50 缺失 A=%s B=%s -> 整轮 VOID" % (pa, pb))
else:
    drift = abs(pb - pa) / pa * 100
    print("    [canary] A=%.4f ms B=%.4f ms drift=%.2f%% -> %s" % (
        pa, pb, drift, "PASS" if drift <= 5 else "VOID (>5%)"))
PYEOF
echo "ALL DONE $(date -Is)" >> "$LOG"
