#!/usr/bin/env bash
# bootstrap.sh — 幂等的环境搭建：从裸机到「能 import torch 且能用 GPU」
#
# 用法：
#   bash scripts/bootstrap.sh              # 自动判断 CPU/GPU（无 nvidia-smi 则装 CPU 轮子）
#   bash scripts/bootstrap.sh --cpu        # 强制 CPU（Mac 上验证流程用）
#   bash scripts/bootstrap.sh --no-torch   # 不装 torch（torch 已就绪时）
#   bash scripts/bootstrap.sh --python     # 复用已有解释器（镜像自带 conda python 等）
#   bash scripts/bootstrap.sh --python=/root/miniconda3/bin/python
#   bash scripts/bootstrap.sh --dry-run    # 只打印将要执行的命令
#
# 幂等：重复执行不破坏已有 .venv。
#
# ★ 云镜像上的推荐组合（例如 AutoDL 的 "PyTorch 2.8.0 / CUDA 12.8" 镜像）：
#       bash scripts/bootstrap.sh --no-torch --python
#   理由：镜像自带的 torch 已由平台验证过能在目标 arch 上跑，且与镜像的 CUDA
#   toolkit 匹配（自写 kernel 要用那个 nvcc 编译）。重新装一个别的版本既慢又可能
#   引入构建不匹配。`--no-torch` 会把 torch 从依赖列表里剔除以免覆盖它。
#
# ★ 为什么 torch 必须单独一步、且用 --index-url：
#   `uv pip install --index-url X` 会**替换** PyPI。
#   所以先装 torch（走 CUDA 索引），再用国内镜像装其余包。顺序不能反。
#
# ★ 为什么最后一定要做「死亡测试」：
#   sm_120 上 torch < 2.7 的轮子没有对应 kernel，会报
#   `CUDA error: no kernel image is available for execution on the device`。
#   在计费的机器上，这个错误必须**在这里就炸出来**，而不是等到跑实验时。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

FORCE_CPU=0
SKIP_TORCH=0
DRY_RUN=0
USE_EXISTING_PY=0
EXPLICIT_PY=""
SKIP_PREFLIGHT=0
for arg in "$@"; do
  case "${arg}" in
    --cpu)      FORCE_CPU=1 ;;
    --no-torch) SKIP_TORCH=1 ;;
    --dry-run)  DRY_RUN=1 ;;
    --no-preflight) SKIP_PREFLIGHT=1 ;;
    --python)   USE_EXISTING_PY=1 ;;
    --python=*) USE_EXISTING_PY=1; EXPLICIT_PY="${arg#--python=}" ;;
    *) echo "unknown arg: ${arg}" >&2; exit 64 ;;
  esac
done

# ---------- .env ----------
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
fi

PY_VERSION="$(cat .python-version 2>/dev/null || echo 3.12)"
VENV="${VENV_DIR:-.venv}"
INDEX="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu130}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 目标解释器：默认自建 venv；--python 时复用已有的解释器（例如镜像自带的 conda python）
EXISTING_PY="${EXPLICIT_PY:-${LEMOQ_PYTHON:-}}"

# 自动判断 CPU / GPU
if [ "${FORCE_CPU}" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  MODE="cpu"
else
  TORCH_INDEX="${CUDA_INDEX}"
  MODE="cuda"
fi

run() {
  if [ "${DRY_RUN}" = "1" ]; then echo "  [dry-run] $*"; else echo "  + $*"; "$@"; fi
}

echo "=== LeanMoE-Quant bootstrap ==="
echo "  mode          : ${MODE}"
if [ "${USE_EXISTING_PY}" = "1" ]; then
  echo "  python        : 复用已有解释器 (--python) —— 不新建 venv"
else
  echo "  python        : ${PY_VERSION}  (新建 venv: ${VENV})"
fi
echo "  torch         : $([ "${SKIP_TORCH}" = "1" ] && echo "跳过（复用已有）" || echo "${TORCH_VERSION} <- ${TORCH_INDEX}")"
echo "  other pkgs    : ${INDEX}"
echo "  HF_ENDPOINT   : ${HF_ENDPOINT}"
echo

# ---------- 0. preflight ----------
if [ "${SKIP_PREFLIGHT}" = "1" ]; then
  echo "[0/5] preflight —— 跳过 (--no-preflight)"
else
  echo "[0/5] preflight（只读）"
  run bash scripts/preflight.sh --no-json || true
fi

# ---------- 1. uv ----------
echo
echo "[1/5] uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "  error: 找不到 uv。安装：curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "         或（国内镜像已可用时）: python3 -m pip install uv" >&2
  exit 3
fi
uv --version

# ---------- 2. 目标解释器 ----------
echo
if [ "${USE_EXISTING_PY}" = "1" ]; then
  echo "[2/5] 复用已有解释器（--python）"
  if [ -n "${EXISTING_PY}" ]; then
    PY="${EXISTING_PY}"
  else
    PY="$(command -v python3 || command -v python)"
  fi
  if [ -z "${PY}" ] || [ ! -x "${PY}" ]; then
    echo "  error: 找不到可用的解释器（尝试了 --python= / LEMOQ_PYTHON / python3 / python）" >&2
    exit 3
  fi
  echo "  interpreter: ${PY}"
  "${PY}" -c 'import sys;print("  version    :", sys.version.split()[0])'
  # 复用已有解释器时，必须确认它已经有 torch；否则后面所有 GPU 测试都会失败
  if ! "${PY}" -c 'import torch' 2>/dev/null; then
    echo "  error: ${PY} 里没有 torch。要么去掉 --python 自建 venv，要么先在该解释器里装 torch。" >&2
    exit 3
  fi
  "${PY}" -c 'import torch;print("  torch      :", torch.__version__, "| cuda", torch.version.cuda)'
else
  echo "[2/5] venv（幂等）"
  run uv venv -p "${PY_VERSION}" "${VENV}"
  PY="${VENV}/bin/python"
fi

# ---------- 3. torch（单独一步！）----------
if [ "${SKIP_TORCH}" = "0" ]; then
  echo
  echo "[3/5] torch（单独一步：--index-url 会替换 PyPI）"
  run uv pip install --python "${PY}" --index-url "${TORCH_INDEX}" "torch==${TORCH_VERSION}"
else
  echo
  echo "[3/5] torch —— 跳过 (--no-torch)"
fi

# ---------- 4. 其余依赖 ----------
echo
echo "[4/5] 其余依赖（走国内镜像）"
REQS="configs/constraints.txt"
if [ "${SKIP_TORCH}" = "1" ] && [ -f "${REQS}" ]; then
  # ★ 必须把 torch 从依赖列表里剔掉：constraints.txt 含 `torch==<pin>`，
  #   若原样安装，会在 `--no-torch`（意图是复用镜像自带 torch）时把那个能用的
  #   torch 覆盖成一个来自 PyPI 镜像的、可能不匹配 CUDA 构建的版本。
  #   注意只剔除 `torch`，不要误伤 `transformers`。
  REQS_FILTERED="${TMPDIR:-/tmp}/lemoq-constraints-notorch.txt"
  grep -vE '^[[:space:]]*torch([=<>!~ ]|$)' configs/constraints.txt > "${REQS_FILTERED}"
  echo "  (--no-torch: 已从依赖中剔除 torch，保留镜像自带版本)"
  echo "  剔除的行:"; grep -E '^[[:space:]]*torch([=<>!~ ]|$)' configs/constraints.txt | sed 's/^/    /'
  REQS="${REQS_FILTERED}"
fi
if [ -f "${REQS}" ]; then
  run uv pip install --python "${PY}" --index-url "${INDEX}" -r "${REQS}"
else
  run uv pip install --python "${PY}" --index-url "${INDEX}" \
      "transformers==4.57.6" numpy safetensors pyyaml pytest
fi
run uv pip install --python "${PY}" --no-deps -e .

# ---------- 5. 死亡测试 ----------
echo
echo "[5/5] 死亡测试"
if [ "${MODE}" = "cpu" ]; then
  TEST_SNIPPET='import torch; import torch.nn as nn; m=nn.Linear(8,8); x=torch.ones(2,8); print("CPU forward ok:", m(x).shape); print("torch", torch.__version__)'
  if [ "${DRY_RUN}" = "1" ]; then echo "  [dry-run] ${PY} -c '<cpu forward test>'"
  else
    "${PY}" -c "${TEST_SNIPPET}" || { echo "  error: CPU 前向失败" >&2; exit 4; }
  fi
else
  TEST_SNIPPET='import torch,sys
assert torch.cuda.is_available(), "torch.cuda.is_available() == False"
x = torch.ones(2, device="cuda")
y = x + x
torch.cuda.synchronize()
p = torch.cuda.get_device_properties(0)
print("GPU ok:", p.name, "sm_%d%d" % (p.major, p.minor), "| torch", torch.__version__)'
  if [ "${DRY_RUN}" = "1" ]; then echo "  [dry-run] ${PY} -c '<cuda death test>'"
  else
    if ! "${PY}" -c "${TEST_SNIPPET}"; then
      echo >&2
      echo "  ✗ 死亡测试失败。" >&2
      echo "    若报 'no kernel image is available'：torch 版本对该 arch 无 kernel。" >&2
      echo "    对 sm_120 需要 torch >= 2.7（当前 pin ${TORCH_VERSION}）。" >&2
      echo "    也可能是 CUDA 索引与驱动不匹配 —— 看 nvidia-smi 的驱动版本重选。" >&2
      exit 4
    fi
  fi
fi

echo
echo "=== bootstrap 完成 ==="
echo "下一步："
echo "  uv run python -m tools.probe_machine --json results/machine_probe.json"
echo "  uv run pytest -m \"not gpu\""
