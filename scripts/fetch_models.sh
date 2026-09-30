#!/usr/bin/env bash
# fetch_models.sh — 从国内源拉取模型并**校验字节数**
#
# 用法：
#   bash scripts/fetch_models.sh                  # 主模型 + dev 模型（自动选源）
#   bash scripts/fetch_models.sh --primary        # 只拉 OLMoE
#   bash scripts/fetch_models.sh --dev            # 只拉 Qwen2.5-0.5B
#   bash scripts/fetch_models.sh --check          # 只校验已下载的，不下载
#   bash scripts/fetch_models.sh --source ms      # 强制走 ModelScope
#   bash scripts/fetch_models.sh --source hf      # 强制走 hf-mirror
#
# ★ 两个源，实测差距很大（RTX 5090 / AutoDL，2026-09）：
#     ModelScope（8 路并行）: ~15 MiB/s  → 13.9 GiB 约 15 min
#     hf-mirror（单流）      : ~2.2 MiB/s → 同量约 100 min
#   所以默认 auto 会优先 ModelScope，失败再退回 hf-mirror。
#
# ★ 为什么不走 huggingface.co：本环境不可达（curl 返回 000）。即使挂上 AutoDL 的
#   /etc/network_turbo 学术加速，实测也只有 0.45 MiB/s —— 比 hf-mirror 还慢 5 倍。
#   「有代理就用代理」在这里是错的，所以 `--source hf` 明确指向 hf-mirror 而非 hf.co。
#
# ★ 为什么必须校验字节数：
#   `allenai/OLMoE-1B-7B` 这个仓库**不存在**（hf-mirror 对不存在的仓库返回 401，
#   与"无权限"同码，极易误判）。正确仓库是 `allenai/OLMoE-1B-7B-0924`。
#   字节数校验能在下载残缺 / 仓库名写错时立刻发现。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
# 数据盘优先：系统盘常常只有 30 GB，而模型就有 13 GB
MODELSCOPE_DIR="${MODELSCOPE_DIR:-$(dirname "${HF_HOME}")/ms}"

PRIMARY_REPO="allenai/OLMoE-1B-7B-0924"
PRIMARY_BYTES=13838721960          # 3 shard 实测总字节（2026-09 由 hf-mirror HEAD 取得）
DEV_REPO="Qwen/Qwen2.5-0.5B"

DO_PRIMARY=0; DO_DEV=0; CHECK_ONLY=0; SOURCE="auto"
if [ "$#" -eq 0 ]; then DO_PRIMARY=1; DO_DEV=1; fi
for arg in "$@"; do
  case "${arg}" in
    --primary) DO_PRIMARY=1 ;;
    --dev)     DO_DEV=1 ;;
    --check)   CHECK_ONLY=1; DO_PRIMARY=1; DO_DEV=1 ;;
    --source)  SOURCE="__next__" ;;
    --source=*) SOURCE="${arg#--source=}" ;;
    ms|modelscope)               SOURCE="ms" ;;
    hf|hf-mirror|huggingface)    SOURCE="hf" ;;
    *) [ "${SOURCE}" = "__next__" ] && SOURCE="${arg}" || { echo "unknown arg: ${arg}" >&2; exit 64; } ;;
  esac
done
case "${SOURCE}" in
  auto|ms|hf) ;;
  *) echo "unknown --source: ${SOURCE}（可选 auto|ms|hf）" >&2; exit 64 ;;
esac

echo "=== fetch_models ==="
echo "  source      : ${SOURCE}"
echo "  HF_ENDPOINT : ${HF_ENDPOINT}"
echo "  HF_HOME     : ${HF_HOME}"
echo "  MODELSCOPE  : ${MODELSCOPE_DIR}"
echo

# ---------- 磁盘预检 ----------
NEED_GB=20
if [ "${DO_PRIMARY}" = "1" ]; then NEED_GB=25; fi
# ★ 必须先建目录再 df：HF_HOME 默认指向 ~/.cache/huggingface，首次运行时它不存在，
#   df 会失败 → 可用空间读成 0 → 误报"磁盘不足"。这个顺序不能反。
mkdir -p "${HF_HOME}" "${MODELSCOPE_DIR}" 2>/dev/null || true
AVAIL_GB="$(df -Pk "${HF_HOME}" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}' || echo 0)"
[ -n "${AVAIL_GB}" ] || AVAIL_GB=0
echo "磁盘可用：约 ${AVAIL_GB} GiB（需要约 ${NEED_GB} GiB）"
if [ "${CHECK_ONLY}" != "1" ] && [ "${AVAIL_GB}" -lt "${NEED_GB}" ] 2>/dev/null; then
  echo "  error: 磁盘不足（HF_HOME=${HF_HOME}）" >&2
  echo "  提示：把 HF_HOME 指到数据盘，例如 export HF_HOME=/root/autodl-tmp/hf" >&2
  exit 5
fi

# ---------- 解释器 ----------
PY=".venv/bin/python"
[ -x "${PY}" ] || PY="$(command -v python3 || command -v python || true)"
if [ -z "${PY}" ]; then echo "error: 找不到 python。先跑 scripts/bootstrap.sh" >&2; exit 3; fi

has_ms() { "${PY}" -c "import modelscope" >/dev/null 2>&1; }

# ---------- 选源 ----------
if [ "${SOURCE}" = "auto" ]; then
  if has_ms; then
    SOURCE="ms"
    echo "自动选源：ModelScope（已安装 modelscope SDK）"
  else
    SOURCE="hf"
    echo "自动选源：hf-mirror（未安装 modelscope；如需加速见下方提示）"
    echo "  提示：${PY} -m pip install modelscope  之后可获约 7 倍加速"
  fi
fi
if [ "${SOURCE}" = "ms" ] && ! has_ms; then
  echo "error: --source ms 但 ${PY} 里没有 modelscope。" >&2
  echo "  ${PY} -m pip install -i https://mirrors.aliyun.com/pypi/simple/ modelscope" >&2
  exit 3
fi
echo "下载器：$([ "${SOURCE}" = "ms" ] && echo "modelscope (8 路并行)" || echo "hf-mirror")"
echo

download_ms() { # $1 = repo, $2 = 本地目录
  "${PY}" - "$1" "$2" <<'PYEOF'
import sys, time
from modelscope import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
p = snapshot_download(repo, cache_dir=dest, max_workers=8)
print("  done in %.1f s -> %s" % (time.time() - t0, p))
PYEOF
}

download_hf() { # $1 = repo
  if command -v hf >/dev/null 2>&1; then hf download "$1"
  elif command -v huggingface-cli >/dev/null 2>&1; then huggingface-cli download "$1"
  else "${PY}" -c "from huggingface_hub import snapshot_download; snapshot_download('$1')"
  fi
}

download() { # $1 = repo
  local repo="$1"
  echo "-- 拉取 ${repo}"
  if [ "${CHECK_ONLY}" = "1" ]; then echo "   (--check：跳过下载)"; return 0; fi
  if [ "${SOURCE}" = "ms" ]; then
    download_ms "${repo}" "${MODELSCOPE_DIR}" || return 1
  else
    download_hf "${repo}" || return 1
  fi
}

# ---------- 定位本地快照目录（两种布局都认） ----------
snapshot_dir() { # $1 = repo
  local repo="$1" d
  # HF 布局：HF_HOME/hub/models--org--name/snapshots/<rev>/
  d="${HF_HOME}/hub/models--$(printf '%s' "${repo}" | tr '/' '-')"
  if [ -d "${d}/snapshots" ]; then ls -1d "${d}/snapshots/"*/ 2>/dev/null | head -1; return 0; fi
  # ModelScope 布局：MODELSCOPE_DIR/models/org--name/snapshots/<rev>/
  d="${MODELSCOPE_DIR}/models/$(printf '%s' "${repo}" | tr '/' '-')"
  if [ -d "${d}/snapshots" ]; then ls -1d "${d}/snapshots/"*/ 2>/dev/null | head -1; return 0; fi
  # ModelScope 的另一种布局（local_dir 形式）
  d="${MODELSCOPE_DIR}/${repo}"
  [ -d "${d}" ] && printf '%s/\n' "${d}"
}

file_size() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo 0; }

check_primary() {
  local repo="$1" dir total=0 f
  dir="$(snapshot_dir "${repo}")"
  if [ -z "${dir}" ]; then echo "  ✗ 未找到本地快照：${repo}"; return 1; fi
  for f in "${dir}"*.safetensors; do
    [ -e "${f}" ] || continue
    total=$(( total + $(file_size "${f}") ))
  done
  # 未完成的下载会留下 .incomplete，必须显式检查：只统计分片会"看起来缺一点"，
  # 但如果三个分片碰巧都在，字节数也可能凑巧接近，所以要单独报出来。
  local partial
  partial="$(find "${MODELSCOPE_DIR}" "${HF_HOME}" -name '*.incomplete' 2>/dev/null | wc -l | tr -d ' ')"
  echo "  repo      : ${repo}"
  echo "  snapshot  : ${dir}"
  echo "  bytes     : ${total}"
  echo "  expected  : ${PRIMARY_BYTES}"
  if [ "${partial}" != "0" ]; then
    echo "  ✗ 存在 ${partial} 个 .incomplete 文件 —— 下载未完成，重跑本脚本即可续传"
    return 1
  fi
  if [ "${total}" = "${PRIMARY_BYTES}" ]; then
    echo "  ✓ 字节数匹配"
    return 0
  fi
  echo "  ✗ 字节数不匹配（差 $(( total - PRIMARY_BYTES ))）"
  echo "    常见原因：仓库名写错（必须是 -0924 后缀）、下载未完成、HF_HOME 指向了旧缓存"
  return 1
}

RC=0
if [ "${DO_PRIMARY}" = "1" ]; then
  download "${PRIMARY_REPO}" || RC=1
  echo
  echo "-- 校验 ${PRIMARY_REPO}"
  check_primary "${PRIMARY_REPO}" || RC=1
fi
if [ "${DO_DEV}" = "1" ]; then
  echo
  download "${DEV_REPO}" || echo "  warn: dev 模型拉取失败（不影响主模型）" >&2
fi

echo
if [ "${RC}" = "0" ]; then
  echo "=== 完成 ==="
  echo "把 snapshot 路径传给 runner（不要硬编码进代码）："
  echo "  python -m benchmarks.runner --model \"$(snapshot_dir "${PRIMARY_REPO}")\""
else
  echo "=== 有校验失败，见上 ===" >&2
fi
exit "${RC}"
