#!/usr/bin/env bash
# run_server_gate0.sh — 服务器到手后的第一条命令
#
# 用法：
#   bash scripts/run_server_gate0.sh
#
# 做什么（全部只读，除写 results/ 外不改机器）：
#   1. preflight  → 硬件存在性 + 网络路由
#   2. probe_machine → results/machine_probe.json（arch / 实测带宽 / 特性 / 工具可用性）
#   3. 汇总裁决：这条路线走不走得通，以及要挂载哪些实验臂
#
# 为什么叫 Gate-0：它决定后续所有实验的适用区间，也可能推翻技术路线
# （例如 Triton 在该 arch 上跑不通 → Phase 2 全盘返工）。
# 成本约 1 GPU 小时（¥1–3），必须在其它任何事之前做完。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; . ./.env; set +a
fi

PY="${LEMOQ_PYTHON:-}"
if [ -z "${PY}" ] && [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
fi
if [ -z "${PY}" ] && [ -x "/root/miniconda3/bin/python" ]; then
  PY="/root/miniconda3/bin/python"
fi
if [ -z "${PY}" ] || [ ! -x "${PY}" ]; then
  echo "error: 找不到可用 Python。设置 LEMOQ_PYTHON，或先跑 scripts/bootstrap.sh" >&2
  exit 3
fi

echo "############ Gate-0 ############"
echo

# ---------- 1. preflight ----------
echo "### 1/3 preflight"
bash scripts/preflight.sh || true
echo

# ---------- 2. probe ----------
echo "### 2/3 machine probe"
mkdir -p results
"${PY}" -m tools.probe_machine --json results/machine_probe.json
PROBE_RC=$?
echo

# ---------- 3. 裁决 ----------
echo "### 3/3 裁决"
"${PY}" - <<'PYEOF'
import json
import os
import sys

path = "results/machine_probe.json"
if not os.path.exists(path):
    print("  ✗ 没有产出 machine_probe.json —— Gate-0 未通过")
    sys.exit(1)

with open(path, encoding="utf-8") as fh:
    r = json.load(fh)

torch_info = r.get("torch", {}) or {}
devices = torch_info.get("devices") or []
if not torch_info.get("cuda_available") or not devices:
    print("  ✗ 没有可见的 CUDA 设备。Gate-0 未通过（这是预期内的：Mac 上就是这样）")
    print("    注意：无 GPU 时 machine_probe 会以退出码 2 结束，结果不得用于任何结论。")
    sys.exit(2)

dev = devices[0]
sm = dev.get("compute_capability")
print("  GPU        : %s" % dev.get("name"))
print("  arch       : %s" % sm)
print("  VRAM       : %s GiB" % dev.get("vram_total_gb"))

arch_features = r.get("arch_features", {}) or {}
print("  features   :")
for k in ("cp_async_supported", "tma_supported", "fp8_conversion_supported",
          "mx_block_scale_supported", "bf16_tensor_core_available", "wgmma_supported"):
    if k in arch_features:
        print("      %-28s %s" % (k, arch_features[k]))

bw = r.get("bandwidth", {}) or {}
print("  bandwidth  : %s GB/s (copy) / %s GB/s (read-only)"
      % (bw.get("measured_gbps"), bw.get("read_only_gbps")))

# 决定要挂载哪些实验臂
arch_arms = {
    "sm_120": "全臂：cp.async + TMA + FP8 + MXFP4(block-scale)",
    "sm_89":  "去掉 TMA 与 MXFP4 臂；保留 cp.async + FP8",
    "sm_86":  "再去掉 FP8 臂（Ampere 无 FP8）；保留 cp.async + bf16",
    "sm_70":  "仅作跨带宽第二数据点；去掉 FP8 / cp.async / bf16 臂",
}
print("  分支       : %s" % arch_arms.get(sm, "未知 arch —— 需要人工判断，不要猜"))
print()

blockers = []
if not arch_features.get("triton_works"):
    blockers.append("Triton 未跑通（Phase 2 可能全盘返工）")
profiler = r.get("profiler", {}) or {}
if not profiler.get("ncu_available"):
    blockers.append("ncu 不可用（kernel 级证据是本项目的关键加分项）")
compile_info = r.get("compile", {}) or {}
if not compile_info.get("cpp_extension_compiles"):
    blockers.append("cpp_extension 编译失败（自写 CUDA kernel 路线受阻，改用 Triton 优先）")

if blockers:
    print("  ⚠️ 前置风险：")
    for b in blockers:
        print("      - %s" % b)
else:
    print("  ✓ 无前置阻塞项")

print()
print("  下一步：把上面的 arch / 实测带宽 / 特性回填进 docs/reference/SPECS.md §1，")
print("         并把 stable 事实摘录进提交（results/ 本身是 gitignored 的）。")
PYEOF
GATE_RC=$?

echo
if [ "${PROBE_RC}" = "0" ] && [ "${GATE_RC}" = "0" ]; then
  echo "############ Gate-0 通过 ############"
else
  echo "############ Gate-0 未通过（见上）############" >&2
fi
exit $(( PROBE_RC != 0 ? PROBE_RC : GATE_RC ))
