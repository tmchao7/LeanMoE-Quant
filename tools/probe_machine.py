#!/usr/bin/env python3
"""probe_machine — 落盘本机 GPU/编译环境的**事实**，供全项目引用。

为什么这个工具是第一个要写的东西：

    本项目所有性能结论都建立在「实测可达带宽」之上（见 docs/METRICS.md §3）。
    规格带宽（datasheet）**不得**用于任何理论下限的计算。

因此这里产出的 JSON 是后续实验的**输入依赖**：`results/machine_probe.json`。
没有它，Phase 1 之后任何"带宽达成率"都不允许计算。

本工具在 Mac（无 GPU）上运行时会以 `--dry-run` 产出占位结构，便于先写消费端代码。

用法：

    python -m tools.probe_machine --json results/machine_probe.json
    python -m tools.probe_machine --dry-run            # 无 GPU 也可跑，产出占位

检测项（每一项都要带测量方法与不确定性）：

    - GPU 名称 / compute capability / sm 数 / 架构
    - 显存总量、物理卡数
    - 驱动 / CUDA runtime / torch 版本
    - **实测可达带宽**（>=3 次重复，报中位数与离散度）
    - fp16 张量核实测吞吐
    - cp.async 是否可用（编译探针）
    - shared memory 上限
    - bf16 张量核是否可用（V100 预期为 false）

状态：**骨架**。带宽标定与编译探针尚未实现（见 TODO），当前只落盘可无成本获取的字段。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from typing import Any

# 本项目主开发机的预期：V100 32GB, sm_70。任何偏差都必须在报告里显式说明。
EXPECTED_ARCH = "sm_70"
EXPECTED_NO_BF16 = True  # V100 无 bf16 张量核
EXPECTED_NO_CP_ASYNC = True  # cp.async 自 Ampere (sm_80) 起


def _safe(cmd: list[str]) -> dict[str, Any]:
    """Run a command and capture its outcome without ever raising."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return {"ok": proc.returncode == 0, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except Exception as exc:  # noqa: BLE001 - probe must never crash the caller
        return {"ok": False, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def probe_torch() -> dict[str, Any]:
    """Collect what PyTorch reports. Returns availability=False without torch."""
    try:
        import torch  # noqa: PLC0415 - optional heavy import, kept local on purpose
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": (torch.backends.cudnn.version() if torch.cuda.is_available() else None),
        "device_count": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "devices": [],
    }
    if not torch.cuda.is_available():
        return info

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info["devices"].append(
            {
                "index": idx,
                "name": props.name,
                "compute_capability": f"sm_{props.major}{props.minor}",
                "sm_count": props.multi_processor_count,
                "vram_total_bytes": props.total_memory,
                "vram_total_gb": round(props.total_memory / 1024**3, 2),
            }
        )
    return info


def probe_driver() -> dict[str, Any]:
    """Driver / toolkit facts from nvidia-smi (absent on a Mac)."""
    smi = _safe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    return {
        "nvidia_smi": smi,
        "nvcc": _safe(["nvcc", "--version"]),
    }


def probe_host() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }


def probe_measured_bandwidth(*, runs: int, size_mb: int) -> dict[str, Any]:
    """Measure achievable HBM bandwidth with a large contiguous device memcpy.

    TODO(phase-0): implement. Requirements before this is allowed to feed any number:
      1. allocate >= 2x L2-sized buffers (V100 L2 = 6 MB; use >= 512 MB each side);
      2. warm up >= 5 iterations, time >= 20 iterations, report median + dispersion;
      3. also measure a *read-only* pattern (reduction), since decode is read-dominated —
         a pure copy measures read+write and will understate the read ceiling;
      4. record the achieved bytes/sec AND the access pattern used.
    """
    return {
        "status": "not_implemented",
        "planned": {
            "runs": runs,
            "buffer_size_mb": size_mb,
            "method": "device copy + read-only reduction, median of >=20 timed iterations",
            "why_two_patterns": "decode is read-dominated; copy measures read+write",
        },
        "note": "Do NOT substitute the datasheet number (V100 spec ~900 GB/s) for this value.",
    }


def probe_sm70_features() -> dict[str, Any]:
    """Compile probes for architecture features that change the whole design.

    TODO(phase-0): compile (not just parse) two tiny kernels via nvcc/torch.utils.cpp_extension:
      - `__pipeline_memcpy_async` (cp.async) -> expected unavailable on sm_70;
      - bf16 tensor-core path -> expected unavailable on sm_70;
      plus query shared memory per block limit.
    """
    return {
        "status": "not_implemented",
        "expected": {
            "arch": EXPECTED_ARCH,
            "bf16_tensor_core_available": not EXPECTED_NO_BF16,
            "cp_async_supported": not EXPECTED_NO_CP_ASYNC,
            "shared_mem_per_block_max_bytes": "expected ~96 KiB (unverified)",
        },
        "note": "Expected values are PRIORS from the architecture, not measurements.",
    }


def build_report(*, dry_run: bool, runs: int, size_mb: int) -> dict[str, Any]:
    torch_info = probe_torch()
    report: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": dry_run,
        "host": probe_host(),
        "driver": probe_driver(),
        "torch": torch_info,
        "bandwidth": probe_measured_bandwidth(runs=runs, size_mb=size_mb),
        "sm70_features": probe_sm70_features(),
    }

    if not dry_run and not torch_info.get("cuda_available", False):
        report["warnings"] = report.get("warnings", []) + [
            "No CUDA device visible: this report must not be used for any measurement."
        ]

    # Deviations from the project's assumed hardware are called out, not silently accepted.
    deviations: list[str] = []
    for dev in torch_info.get("devices", []):
        if dev.get("compute_capability") != EXPECTED_ARCH:
            deviations.append(
                f"device {dev['index']} is {dev.get('compute_capability')}, expected {EXPECTED_ARCH}"
            )
        if dev.get("vram_total_gb", 0) < 30:
            deviations.append(
                f"device {dev['index']} has {dev.get('vram_total_gb')} GB, expected ~32 GB"
            )
    if deviations:
        report["deviations"] = deviations
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record machine facts (GPU, driver, measurable bandwidth, sm_70 features).",
    )
    parser.add_argument("--json", metavar="PATH", help="write the report to this path (default: stdout)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="produce the report structure without requiring a GPU (safe on the Mac)",
    )
    parser.add_argument("--runs", type=int, default=3, help="bandwidth measurement repeats (>=3)")
    parser.add_argument("--size-mb", type=int, default=512, help="bandwidth buffer size per side, MiB")
    args = parser.parse_args(argv)

    report = build_report(dry_run=args.dry_run, runs=args.runs, size_mb=args.size_mb)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        from pathlib import Path

        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)

    # Exit non-zero when a real probe could not see the expected device: a silent
    # success here would let downstream phases build on an unverified machine.
    if not args.dry_run and not report["torch"].get("cuda_available", False):
        print("ERROR: --dry-run not set but no CUDA device is visible", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
