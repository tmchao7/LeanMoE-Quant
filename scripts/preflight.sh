#!/usr/bin/env bash
# preflight.sh — 上机前的只读检查：硬件存在性 + 网络路由
#
# 用法：
#   bash scripts/preflight.sh                 # 人读输出 + 写 results/preflight.json
#   bash scripts/preflight.sh --no-json       # 只打印
#
# 设计原则：
#   - **只读**：除了写 results/preflight.json 之外不改动任何东西；
#   - 在 Mac 上也能跑（没有 nvidia-smi 时如实报告，不报错退出）；
#   - 网络分「直连」与「走代理」两条路径分别探测，给出路由建议。
#
# 兼容 macOS bash 3.2。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WRITE_JSON=1
[ "${1:-}" = "--no-json" ] && WRITE_JSON=0

# ---------- .env ----------
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
fi

HF_PROBE="${HF_PROBE:-https://hf-mirror.com/}"
PYPI_PROBE="${PYPI_PROBE:-https://mirrors.aliyun.com/pypi/simple/}"
GITHUB_PROBE="${GITHUB_PROBE:-https://github.com/}"
NPM_PROBE="${NPM_PROBE:-https://registry.npmjs.org/}"
JSDELIVR_PROBE="${JSDELIVR_PROBE:-https://cdn.jsdelivr.net/}"

# ---------- 硬件 ----------
echo "=== 硬件 ==="
GPU_PRESENT=0
SMI_OUT=""
if command -v nvidia-smi >/dev/null 2>&1; then
  SMI_OUT="$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
              --format=csv,noheader 2>/dev/null || true)"
  if [ -n "${SMI_OUT}" ]; then
    GPU_PRESENT=1
    echo "${SMI_OUT}" | sed 's/^/  /'
  else
    echo "  nvidia-smi 存在但无输出（无可见 GPU）"
  fi
else
  echo "  nvidia-smi: 不存在 → 这台机器没有 NVIDIA GPU（Mac 上属预期）"
fi

HOST_CPU="unknown"
if command -v sysctl >/dev/null 2>&1; then
  HOST_CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
elif [ -r /proc/cpuinfo ]; then
  HOST_CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ //')"
fi
echo "  host CPU: ${HOST_CPU}"

# ---------- 网络 ----------
probe_one() { # $1=url -> http code (000 = fail)
  curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$1" 2>/dev/null || echo "000"
}

collect() { # $1=mode  -> echoes "label=code;..."
  local mode="$1" out=""
  if [ "${mode}" = "proxy" ]; then
    export http_proxy="${PROXY_URL}" https_proxy="${PROXY_URL}" all_proxy="${PROXY_URL}"
    export HTTP_PROXY="${PROXY_URL}" HTTPS_PROXY="${PROXY_URL}" ALL_PROXY="${PROXY_URL}"
  else
    unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
  fi
  local l u c
  for pair in "hf-mirror:${HF_PROBE}" "aliyun_pypi:${PYPI_PROBE}" "github:${GITHUB_PROBE}" \
              "npm:${NPM_PROBE}" "jsdelivr:${JSDELIVR_PROBE}"; do
    l="${pair%%:*}"; u="${pair#*:}"
    c="$(probe_one "${u}")"
    out="${out}${l}=${c};"
  done
  printf '%s' "${out}"
}

echo
echo "=== 网络（直连）==="
DIRECT="$(collect direct)"
echo "${DIRECT}" | tr ';' '\n' | grep '=' | sed 's/^/  /'

PROXY_RESULT=""
if [ -n "${PROXY_URL:-}" ]; then
  echo
  echo "=== 网络（走代理）==="
  PROXY_RESULT="$(collect proxy)"
  echo "${PROXY_RESULT}" | tr ';' '\n' | grep '=' | sed 's/^/  /'
else
  echo
  echo "=== 网络（走代理）==="
  echo "  PROXY_URL 未设置 —— 跳过"
fi

# ---------- 路由建议 ----------
echo
echo "=== 路由建议 ==="
code_of() { printf '%s' "$1" | tr ';' '\n' | grep "^$2=" | cut -d= -f2; }
for k in hf-mirror aliyun_pypi github npm jsdelivr; do
  d="$(code_of "${DIRECT}" "${k}")"
  if [ "${d}" != "000" ]; then echo "  ${k}: 直连可用（${d}）"
  elif [ -n "${PROXY_RESULT}" ] && [ "$(code_of "${PROXY_RESULT}" "${k}")" != "000" ]; then
    echo "  ${k}: 需要代理"
  else
    echo "  ${k}: ⚠️ 不可达（直连与代理都失败）"
  fi
done

# ---------- JSON ----------
if [ "${WRITE_JSON}" = "1" ]; then
  mkdir -p results
  json_esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
  {
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "kind": "preflight",\n'
    printf '  "host_cpu": "%s",\n' "$(json_esc "${HOST_CPU}")"
    printf '  "gpu_present": %s,\n' "$([ "${GPU_PRESENT}" = "1" ] && echo true || echo false)"
    printf '  "nvidia_smi": "%s",\n' "$(json_esc "${SMI_OUT}")"
    printf '  "proxy_configured": %s,\n' "$([ -n "${PROXY_URL:-}" ] && echo true || echo false)"
    printf '  "direct": "%s",\n' "$(json_esc "${DIRECT}")"
    printf '  "via_proxy": "%s"\n' "$(json_esc "${PROXY_RESULT}")"
    printf '}\n'
  } > results/preflight.json
  echo
  echo "wrote results/preflight.json"
fi

# 在无 GPU 的机器上不视为失败（Mac 是合法的开发环境）
exit 0
