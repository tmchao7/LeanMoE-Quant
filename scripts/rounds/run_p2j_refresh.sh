#!/bin/bash
# P2j（2026-10-08，用户批准）：decode 的 RMSNorm 换成自研融合 kernel 后，把**所有已落盘的
# runner 格**按 P2d 的先例刷新一遍（归档改前 payload、重跑、记录里标 v-P2i 前后）。
#
# 范围：p2-01(6) + p2-02(1) + p2-04(2) + p2-05(2) + p4b(16) = 27 格。
#   - p4c-01 的 8 格**已**是 fused（v3），跳过；
#   - p1-01 的 5 格是**精度协议**（KL，不是延迟），且该协议已被 p1-04 取代 —— 不在本轮，
#     其 KL 数字的 pre-P2i 状态写进记录；
#   - 命令**从每个 payload 的 config 反推**（bit_width/attn/ctx/execution/moe/dequant），
#     不靠猜；跑完核对 cell id 与目录名一致。
set -u
cd /root/LeanMoE-Quant || exit 1
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
OUT=results/EXP-p2j-refresh
mkdir -p "$OUT/archive"
LOG="$OUT/p2j.log"; CONT="$OUT/p2j-contention.log"; LIST="$OUT/cells.txt"
: > "$LOG"; : > "$CONT"
snap () { { echo "--- $1 $(date -Is)"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader;
            nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; } >> "$CONT" 2>&1; }

# ---- 1) 生成命令清单（从 payload 反推）----
$PY - > "$LIST" <<'PYEOF'
import glob, json, os
SKIP_EXP = ("EXP-p4c-01", "EXP-p1-01")   # 已是 fused / 精度协议（见脚本头）
for p in sorted(glob.glob("results/EXP-*__*/result.json")):
    d = json.load(open(p))
    if d.get("status") != "done":
        continue
    cell = os.path.dirname(p)
    exp = os.path.basename(cell).split("__")[0]
    if any(exp.startswith(s) for s in SKIP_EXP):
        continue
    c = d["config"]
    if c.get("rmsnorm") == "fused":
        continue  # 已刷新（断点续跑；见脚本头）
    args = ["--ctx", str(c.get("ctx") or 1024)]
    if c.get("bit_width"):
        args += ["--bit-width", c["bit_width"]]
    if c.get("attn"):
        args += ["--attn", c["attn"]]
    if c.get("execution"):
        args += ["--execution", c["execution"]]
    if c.get("moe"):
        args += ["--moe", c["moe"]]
    if c.get("dequant") and c["dequant"] != "none":
        args += ["--dequant", c["dequant"]]
    if c.get("group_size"):
        args += ["--group-size", str(c["group_size"])]
    if c.get("batch"):
        args += ["--batch", str(c["batch"])]
    print("%s|%s|%s" % (exp, " ".join(args), cell))
PYEOF
echo "清单 $(wc -l < "$LIST") 格" >> "$LOG"; cat "$LIST" >> "$LOG"

# ---- 2) 归档 + 重跑 ----
run_one () {
  local exp=$1 args=$2 cell=$3 label
  # ★ 标签必须是**完整目录名**：早先只取 `ctx=1024` 这种后缀，跨实验会重名 ——
  #   日志文件互相覆盖、p2j.log 里的自检行也分不清属于哪一格（实测第 1 格就撞了）。
  label=$(basename "$cell")
  # ★ 幂等：重启轮次时**不能**用已刷新的 payload 覆盖改前归档（否则丢掉 pre-P2i 基线）
  [ -f "$OUT/archive/$(basename "$cell").json" ] || cp -p "$cell/result.json" "$OUT/archive/$(basename "$cell").json"
  snap "$label-pre"
  echo "=== [$label] $exp $args start $(date -Is)" >> "$LOG"
  $PY -m benchmarks.runner --exp "$exp" --model "$M" $args --force > "$OUT/$label.log" 2>&1
  echo "=== [$label] exit=$? end $(date -Is)" >> "$LOG"
  $PY - "$label" "$cell/result.json" <<'PYEOF' >> "$LOG" 2>&1
import json, sys
lbl, path = sys.argv[1], sys.argv[2]
d = json.load(open(path)); lat, att, cfg = d["latency"], d["attribution"], d["config"]
ev = (d.get("_measurement") or {}).get("gate_failed_evidence") or {}
p50 = lat["per_token_p50_ms"] if lat["per_token_p50_ms"] is not None else ev.get("per_token_p50_ms")
warn = []
if d["status"] != "done": warn.append("status=%s" % d["status"])
if (att["host_share_pct"] or 0) > 5: warn.append("host_share=%.1f%%>5%%" % att["host_share_pct"])
if cfg.get("rmsnorm") != "fused" or (cfg.get("rms_norms_replaced") or 0) < 65:
    warn.append("rmsnorm=%r replaced=%r（应为 fused/65）" % (cfg.get("rmsnorm"), cfg.get("rms_norms_replaced")))
print("    [%s] bw=%-6s attn=%-6s ctx=%-5s exec=%-9s dq=%-8s gate=%-4s p50=%-9s host=%-5s %s" % (
    lbl, cfg.get("bit_width"), cfg.get("attn"), cfg.get("ctx"), cfg.get("execution"), cfg.get("dequant"),
    d["correctness"]["gate"], p50, att["host_share_pct"], ("!! " + "; ".join(warn)) if warn else "OK"))
PYEOF
  snap "$label-post"
}

while IFS='|' read -r exp args cell; do
  [ -z "${exp:-}" ] && continue
  run_one "$exp" "$args" "$cell"
done < "$LIST"

# ---- 3) 金丝雀 B：p4b 的 w8a16×cudagraph×1024（清单里也会出现，首尾同格）----
CANARY=results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024
CANARY_EXP=EXP-p4b-01-truncation
CANARY_ARGS="--ctx 1024 --bit-width w8a16 --execution cudagraph --moe grouped --dequant fused"
run_one "$CANARY_EXP" "$CANARY_ARGS" "$CANARY"
echo "ALL DONE $(date -Is)" >> "$LOG"
