#!/usr/bin/env bash
# run_cudagraph_gate.sh — 在 GPU 机器上一键跑完 CUDA Graph 可行性门
#
#   bash scripts/run_cudagraph_gate.sh              # 全部：自检 -> 离线测试 -> GPU smoke -> p0-03 -> p0-04
#   bash scripts/run_cudagraph_gate.sh --check      # 只自检，不跑实验
#   bash scripts/run_cudagraph_gate.sh --skip-smoke # 跳过 GPU smoke（smoke 已在本次会话过了）
#   bash scripts/run_cudagraph_gate.sh --only p0-04 # 只跑一条
#
# 为什么要有这个脚本：
#   这台机器 ¥2.88/h 且按量计费。手工一步步敲会在"忘了设 HF_HOME"、
#   "跑错了 EXP id"、"重复跑已经跑过的"这些地方烧钱。把顺序固化成脚本，
#   出错时也能从断点接着跑（runner 默认跳过 status=done 的实验）。
#
# ★ 顺序是刻意的：p0-03（HF 原版）先跑，p0-04（固定形状）后跑。
#   如果 p0-03 直接通过，说明我离线写的预判错了 —— 那也是结果，且 p0-04 仍然值得跑
#   （它更快，因为不再循环 64 个专家）。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# ---------- 参数 ----------
DO_CHECK=0; DO_SMOKE=1; ONLY=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check)      DO_CHECK=1 ;;
    --skip-smoke) DO_SMOKE=0 ;;
    --only)       ONLY="${2:-}"; shift ;;
    --only=*)     ONLY="${1#--only=}" ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
  shift
done

# ---------- 环境 ----------
# ★ 必须在任何下载/缓存访问之前设好：否则 OLMoE 的 12.89 GiB 会落到 30 GB 系统盘
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
# ★ nvcc / ncu 不在默认 PATH 上，cpp_extension 与工具探针都需要
export PATH="/root/miniconda3/bin:/usr/local/cuda/bin:${PATH}"

PY="${LEMOQ_PYTHON:-/root/miniconda3/bin/python3}"
MODEL="${LEMOQ_MODEL:-/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master}"
CTX="${LEMOQ_CTX:-1024}"
RESULTS="${LEMOQ_RESULTS:-results}"

RC=0
say() { printf '\n=== %s ===\n' "$*"; }
fail() { printf '  ✗ %s\n' "$*" >&2; RC=1; }

# ---------- 0) 自检 ----------
say "0. 自检"
echo "  python   : ${PY}"
if [ ! -x "${PY}" ]; then
  echo "  error: 找不到解释器 ${PY}。先跑 scripts/bootstrap.sh" >&2
  exit 3
fi
"${PY}" - <<'PYEOF' || exit 3
import sys
try:
    import torch
except Exception as exc:
    print("  error: import torch 失败: %s" % exc); sys.exit(3)
print("  torch    : %s (cuda %s)" % (torch.__version__, torch.version.cuda))
if not torch.cuda.is_available():
    print("  error: CUDA 不可用。本脚本只能在有 GPU 的机器上跑。"); sys.exit(3)
p = torch.cuda.get_device_properties(0)
print("  gpu      : %s sm_%d%d (%d SMs, %.1f GiB)"
      % (p.name, p.major, p.minor, p.multi_processor_count, p.total_memory / 1024**3))
if (p.major, p.minor) != (12, 0):
    print("  warning: 预期 sm_120（RTX 5090），实测 sm_%d%d —— 结果不要跨架构引用"
          % (p.major, p.minor))
try:
    import transformers
    print("  transformers: %s" % transformers.__version__)
except Exception as exc:
    print("  error: import transformers 失败: %s" % exc); sys.exit(3)
PYEOF

echo "  repo     : $(git log --oneline -1 2>/dev/null || echo 'not a git repo')"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
  echo "  warning: 工作树不干净 —— 结果将无法对应到确定提交"
  git status --short | head -5
fi

echo "  model    : ${MODEL}"
if [ ! -d "${MODEL}" ]; then
  fail "模型目录不存在。先跑 scripts/fetch_models.sh（或设 LEMOQ_MODEL）"
else
  total=0
  for f in "${MODEL}"/*.safetensors; do
    [ -e "$f" ] || continue
    total=$(( total + $(stat -c%s "$f" 2>/dev/null || stat -f%z "$f") ))
  done
  echo "  bytes    : ${total}"
  if [ "${total}" != "13838721960" ]; then
    fail "字节数不是 13,838,721,960 —— 模型可能不完整（检查 .incomplete）"
  else
    echo "  ✓ 字节数匹配"
  fi
fi

echo "  HF_HOME  : ${HF_HOME}"
echo "  磁盘     : $(df -Ph "${HF_HOME}" 2>/dev/null | awk 'NR==2{print $4" avail on "$6}')"

if [ "${DO_CHECK}" = "1" ]; then
  say "仅自检（--check），退出码 ${RC}"
  exit "${RC}"
fi
[ "${RC}" = "0" ] || { echo "自检失败，停止。" >&2; exit "${RC}"; }

# ---------- 1) 离线回归 ----------
say "1. 离线回归（不需要 GPU 的部分，先把明显的错挡住）"
"${PY}" -m pytest -m "not gpu" -q || { echo "离线测试失败，停止。" >&2; exit 4; }

# ---------- 2) GPU smoke ----------
if [ "${DO_SMOKE}" = "1" ]; then
  say "2. GPU smoke：小图能不能 capture（失败就停，不浪费时间加载 13 GB 模型）"
  if ! "${PY}" -m pytest tests/test_cudagraph.py -m gpu -q; then
    echo "  ✗ GPU smoke 失败 —— 说明连最小场景都捕获不了。" >&2
    echo "    先修这个；不要在 smoke 失败的情况下跑 p0-03/p0-04。" >&2
    exit 5
  fi
  echo "  ✓ smoke 通过"
else
  say "2. GPU smoke（--skip-smoke，跳过）"
fi

# ---------- 3) 两条 gate ----------
run_exp() { # $1 = exp id, $2 = 额外参数
  local exp="$1" extra="${2:-}"
  say "3. ${exp}"
  # shellcheck disable=SC2086
  "${PY}" -m benchmarks.runner \
    --exp "${exp}" \
    --model "${MODEL}" \
    --ctx "${CTX}" \
    --batch 1 \
    --max-new-tokens 32 \
    --warmup 5 \
    --timed 20 \
    --dtype float16 \
    --results-dir "${RESULTS}" \
    ${extra}
  local rc=$?
  # runner 的退出码：0=正常（含 blocked/failed 记录），1=schema 问题
  # blocked/failed 是**合法结果**，不当成脚本失败，但要在总结里显示出来。
  if [ "${rc}" != "0" ]; then
    fail "${exp} 退出码 ${rc}（见上）"
  fi
}

if [ -z "${ONLY}" ] || [ "${ONLY}" = "p0-03" ]; then
  run_exp "EXP-p0-03-baseline-cudagraph" ""
fi
if [ -z "${ONLY}" ] || [ "${ONLY}" = "p0-04" ]; then
  run_exp "EXP-p0-04-cudagraph-static-moe" "--static-moe"
fi

# ---------- 4) 总结 ----------
say "4. 结果总结"
"${PY}" - "${RESULTS}" <<'PYEOF'
import glob, json, os, sys

root = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(root, "EXP-p0-0[34]*", "result.json"))):
    with open(path, encoding="utf-8") as fh:
        r = json.load(fh)
    lat = r.get("latency") or {}
    attr = r.get("attribution") or {}
    corr = r.get("correctness") or {}
    cap = r.get("capture") or {}
    rows.append({
        "exp": r.get("exp_id", "?"),
        "status": r.get("status"),
        "verdict": (r.get("verdict") or {}).get("verdict"),
        "capture": cap.get("capture_success"),
        "replay": cap.get("replay_success"),
        "kl": corr.get("kl"),
        "top1": corr.get("top1_agreement"),
        "p50": lat.get("per_token_p50_ms"),
        "host%": attr.get("host_share_pct"),
        "kernels": attr.get("kernels_per_call"),
    })

if not rows:
    print("  （没有找到 p0-03 / p0-04 的结果文件）")
    sys.exit(0)

print("  %-42s %-9s %-7s %-8s %-9s %-9s %-10s %-7s" % (
    "EXP", "status", "capture", "replay", "KL", "top1", "p50(ms)", "host%"))
for r in rows:
    def f(v, fmt="%.4g"):
        return "-" if v is None else (fmt % v if isinstance(v, (int, float)) else str(v))
    print("  %-42s %-9s %-7s %-8s %-9s %-9s %-10s %-7s" % (
        r["exp"][:42], r["status"], r["capture"], r["replay"],
        f(r["kl"]), f(r["top1"]), f(r["p50"], "%.2f"), f(r["host%"], "%.1f")))

print()
print("  判据（见 docs/MOE_CUDAGRAPH.md §4）：")
print("    p0-03 blocked + p0-04 通过 -> 预判成立，routing 的动态形状是阻塞点")
print("    p0-03 也通过            -> 预判错了，HF 原版就够（但务必确认 KL/top1 真的过了）")
print("    p0-04 也 blocked        -> 阻塞点不止 routing，看 failure 的 stage")
print("    capture 成功但 KL 失败  -> 静默错，这是最有价值的结果之一")
PYEOF

say "退出码 ${RC}"
echo
echo "关机前记得把结果拉回 Mac（results/ 是 gitignored，不会自动保留）："
echo "  rsync -az <server>:$(pwd)/${RESULTS}/ ./${RESULTS}/"
exit "${RC}"
