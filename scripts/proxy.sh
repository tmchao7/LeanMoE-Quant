#!/usr/bin/env bash
# proxy.sh — 代理开关与连通性验证
#
# 用法：
#   scripts/proxy.sh on        # 从 .env 读 PROXY_URL 并导出到当前 shell 的环境
#   scripts/proxy.sh off       # 清除代理变量
#   scripts/proxy.sh status    # 显示当前代理相关变量
#   scripts/proxy.sh test      # 直连 vs 代理 分别探测关键站点
#
# 注意：
#   - `on` 只能影响**当前 shell 的环境**。要让子进程（pytest/uv/python）继承，
#     必须 `source scripts/proxy.sh on`（用 source，不是执行）。
#     执行方式下会提示你把 export 行写到 shell 配置里。
#   - 凭据只来自 .env（已 gitignored）。绝不把值打印出来。
#
# 兼容 macOS 自带的 bash 3.2：不使用关联数组等 bash 4+ 特性。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
ENV_EXAMPLE="${REPO_ROOT}/configs/env.env.example"

# ---------- 读取 .env ----------
load_env() {
  if [ -f "${ENV_FILE}" ]; then
    # shellcheck disable=SC1090
    set -a; . "${ENV_FILE}"; set +a
    echo "loaded: .env"
  else
    echo "warn: 未找到 .env（参照 ${ENV_EXAMPLE} 创建）" >&2
  fi
}

# ---------- 子命令 ----------
cmd_on() {
  load_env
  if [ -z "${PROXY_URL:-}" ]; then
    echo "error: PROXY_URL 为空。请在 .env 里填写，或用 scripts/proxy.sh test 走直连。" >&2
    exit 2
  fi
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export all_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export ALL_PROXY="${PROXY_URL}"
  export no_proxy="${NO_PROXY_LIST:-localhost,127.0.0.1,::1}"
  export NO_PROXY="${no_proxy}"
  echo "proxy enabled (value not printed)."
  echo "如果这是被执行的（而不是 source 的），子进程不会继承。改用："
  echo "    source scripts/proxy.sh on"
}

cmd_off() {
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY 2>/dev/null || true
  echo "proxy disabled."
}

cmd_status() {
  for v in http_proxy https_proxy all_proxy NO_PROXY no_proxy HF_ENDPOINT PIP_INDEX_URL TORCH_CUDA_INDEX; do
    eval "val=\${${v}:-<unset>}"
    # 代理类变量只报是否设置，避免把凭据打进日志
    case "${v}" in
      http_proxy|https_proxy|all_proxy)
        if [ "${val}" = "<unset>" ]; then echo "  ${v} = <unset>"
        else echo "  ${v} = <set, hidden>"; fi ;;
      *) echo "  ${v} = ${val}" ;;
    esac
  done
}

probe() {
  # $1 = label, $2 = url
  local label="$1" url="$2" code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "${url}" 2>/dev/null || echo "000")"
  if [ "${code}" = "000" ]; then printf '  %-28s FAIL\n' "${label}"
  else printf '  %-28s %s\n' "${label}" "${code}"; fi
}

cmd_test() {
  load_env
  echo "== 直连 =="
  ( unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
    probe "hf-mirror.com"     "${HF_PROBE:-https://hf-mirror.com/}"
    probe "aliyun pypi"       "${PYPI_PROBE:-https://mirrors.aliyun.com/pypi/simple/}"
    probe "github.com"        "${GITHUB_PROBE:-https://github.com/}"
    probe "registry.npmjs.org" "${NPM_PROBE:-https://registry.npmjs.org/}"
    probe "cdn.jsdelivr.net"  "${JSDELIVR_PROBE:-https://cdn.jsdelivr.net/}" )
  if [ -n "${PROXY_URL:-}" ]; then
    echo "== 走代理 =="
    ( export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}" all_proxy="${PROXY_URL}"
      export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}" ALL_PROXY="${PROXY_URL}"
      probe "hf-mirror.com"     "${HF_PROBE:-https://hf-mirror.com/}"
      probe "aliyun pypi"       "${PYPI_PROBE:-https://mirrors.aliyun.com/pypi/simple/}"
      probe "github.com"        "${GITHUB_PROBE:-https://github.com/}"
      probe "registry.npmjs.org" "${NPM_PROBE:-https://registry.npmjs.org/}"
      probe "cdn.jsdelivr.net"  "${JSDELIVR_PROBE:-https://cdn.jsdelivr.net/}" )
  else
    echo "（PROXY_URL 未设置，跳过代理探测）"
  fi
  echo
  echo "策略提醒：模型走 hf-mirror、包走国内镜像 —— 这两条**不需要代理**。"
  echo "          代理只留给 github.com 与 registry.npmjs.org。"
}

case "${1:-}" in
  on)     cmd_on ;;
  off)    cmd_off ;;
  status) cmd_status ;;
  test)   cmd_test ;;
  *) echo "usage: scripts/proxy.sh {on|off|status|test}" >&2; exit 64 ;;
esac
