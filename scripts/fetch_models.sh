#!/usr/bin/env bash
# fetch_models.sh — 从国内镜像拉取模型并**校验字节数**
#
# 用法：
#   bash scripts/fetch_models.sh              # 拉主模型 + dev 模型
#   bash scripts/fetch_models.sh --primary    # 只拉 OLMoE
#   bash scripts/fetch_models.sh --dev        # 只拉 Qwen2.5-0.5B
#   bash scripts/fetch_models.sh --check      # 只校验已下载的，不下载
#
# ★ 不走 huggingface.co（本环境不可达），走 HF_ENDPOINT=https://hf-mirror.com
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

PRIMARY_REPO="allenai/OLMoE-1B-7B-0924"
PRIMARY_BYTES=13838721960          # 3 shard 实测总字节（2026-09 由 hf-mirror HEAD 取得）
DEV_REPO="Qwen/Qwen2.5-0.5B"

DO_PRIMARY=0; DO_DEV=0; CHECK_ONLY=0
if [ "$#" -eq 0 ]; then DO_PRIMARY=1; DO_DEV=1; fi
for arg in "$@"; do
  case "${arg}" in
    --primary) DO_PRIMARY=1 ;;
    --dev)     DO_DEV=1 ;;
    --check)   CHECK_ONLY=1; DO_PRIMARY=1; DO_DEV=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 64 ;;
  esac
done

echo "=== fetch_models ==="
echo "  HF_ENDPOINT : ${HF_ENDPOINT}"
echo "  HF_HOME     : ${HF_HOME}"
echo

# ---------- 磁盘预检 ----------
NEED_GB=20
if [ "${DO_PRIMARY}" = "1" ]; then NEED_GB=25; fi
AVAIL_GB="$(df -Pk "${HF_HOME}" 2>/dev/null | awk 'NR==2{printf "%d", $4/1024/1024}' || echo 0)"
echo "磁盘可用：约 ${AVAIL_GB} GiB（需要约 ${NEED_GB} GiB）"
if [ "${AVAIL_GB}" -lt "${NEED_GB}" ] 2>/dev/null; then
  echo "  error: 磁盘不足" >&2; exit 5
fi

# ---------- 选择下载器 ----------
PY=".venv/bin/python"
[ -x "${PY}" ] || PY="$(command -v python3)"
DOWNLOADER=""
if command -v hf >/dev/null 2>&1; then DOWNLOADER="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then DOWNLOADER="huggingface-cli"
elif [ -x "${PY}" ]; then DOWNLOADER="python"; fi

if [ -z "${DOWNLOADER}" ]; then
  echo "error: 找不到 hf / huggingface-cli / python。先跑 scripts/bootstrap.sh" >&2; exit 3
fi
echo "下载器：${DOWNLOADER}"
echo

download() { # $1 = repo
  local repo="$1"
  echo "-- 拉取 ${repo}"
  if [ "${CHECK_ONLY}" = "1" ]; then echo "   (--check：跳过下载)"; return 0; fi
  case "${DOWNLOADER}" in
    hf)              hf download "${repo}" ;;
    huggingface-cli) huggingface-cli download "${repo}" ;;
    python)          "${PY}" -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}')" ;;
  esac
}

# ---------- 定位本地快照目录 ----------
snapshot_dir() { # $1 = repo
  local repo="$1" d
  d="${HF_HOME}/hub/models--$(printf '%s' "${repo}" | tr '/' '-')"
  [ -d "${d}/snapshots" ] && ls -1d "${d}/snapshots/"*/ 2>/dev/null | head -1 || true
}

check_primary() {
  local repo="$1" dir total=0 f
  dir="$(snapshot_dir "${repo}")"
  if [ -z "${dir}" ]; then echo "  ✗ 未找到本地快照：${repo}"; return 1; fi
  for f in "${dir}"*.safetensors; do
    [ -e "${f}" ] || continue
    # macOS 与 Linux 的 stat 参数不同
    total=$(( total + $(stat -f%z "${f}" 2>/dev/null || stat -c%s "${f}" 2>/dev/null || echo 0) ))
  done
  echo "  repo      : ${repo}"
  echo "  snapshot  : ${dir}"
  echo "  bytes     : ${total}"
  echo "  expected  : ${PRIMARY_BYTES}"
  if [ "${total}" = "${PRIMARY_BYTES}" ]; then
    echo "  ✓ 字节数匹配"
    return 0
  else
    echo "  ✗ 字节数不匹配（差 $(( total - PRIMARY_BYTES ))）"
    echo "    常见原因：仓库名写错（必须是 -0924 后缀）、下载未完成、HF_HOME 指向了旧缓存"
    return 1
  fi
}

RC=0
if [ "${DO_PRIMARY}" = "1" ]; then
  download "${PRIMARY_REPO}"
  echo
  echo "-- 校验 ${PRIMARY_REPO}"
  check_primary "${PRIMARY_REPO}" || RC=1
fi
if [ "${DO_DEV}" = "1" ]; then
  echo
  download "${DEV_REPO}"
fi

echo
if [ "${RC}" = "0" ]; then
  echo "=== 完成。把 snapshot 路径写进 ENVIRONMENT.md §5 本机附录（不要硬编码进代码）==="
else
  echo "=== 有校验失败，见上 ===" >&2
fi
exit "${RC}"
