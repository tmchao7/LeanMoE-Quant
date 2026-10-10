#!/usr/bin/env python3
"""probe_machine — 落盘本机 GPU/编译环境的**事实**，供全项目引用。

为什么这个工具是第一个要写的东西：

    本项目所有性能结论都建立在「实测可达带宽」之上（见 docs/reference/METRICS.md §3）。
    规格带宽（datasheet）**不得**用于任何理论下限的计算。

因此这里产出的 JSON 是后续实验的**输入依赖**：`results/machine_probe.json`。
没有它，任何"带宽达成率"都不允许计算。

用法：

    python -m tools.probe_machine --json results/machine_probe.json
    python -m tools.probe_machine --dry-run          # 无 GPU 也可跑，产出**完整**占位结构
    python -m tools.probe_machine --skip-heavy       # 跳过带宽/编译/Triton 探针（快速）

设计约定（消费端代码依赖这一点）：

    - **无 GPU 时也产出完整结构**：每个探针返回 `status` + `value` + `method` +
      `n_runs` + `dispersion`。未测的写 `status: "not_measured"`、`value: null`。
      **绝不把"没测"写成 `false`**——`false` 是一个结论，`null` 才是"没测"。
    - 架构特性同时给 **expected**（来自架构表）与 **verified**（来自编译探针）。
      架构表来自本地 LLVM NVPTX 后端的指令谓词（可核对、可复现），但它**不是测量**。

状态：**已实现**（schema v2）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# 架构特性表
#
# 来源：本地 `llvm-project/llvm/lib/Target/NVPTX/NVPTXInstrInfo.td` 与
#       `NVPTXIntrinsics.td`（2026-09 checkout）的指令谓词，逐条读出：
#         cp.async        -> Requires<[SM80]>
#         wgmma           -> Predicates = [SM90a]  （'a' 变体不被任何后代架构隐含）
#         TMA             -> sm_90 基础架构；Blackwell 扩展 TMA 为 [SM100f,SM110f]
#         FP8 转换        -> (SM90 | (SM89 & PTX81))
#         MX block-scale  -> hasMMABlockScale = [SM120f]
#         tcgen05         -> [SM100a, SM103a, SM110a]
#
# ★ 这是**先验**，不是测量。只有编译探针成功的项才可标 verified=True。
# ---------------------------------------------------------------------------
_SRC_NOTE = "本地 LLVM checkout: NVPTXInstrInfo.td / NVPTXIntrinsics.td 指令谓词"

_SM_ORDER = {
    "sm_70": 70,
    "sm_72": 72,
    "sm_75": 75,
    "sm_80": 80,
    "sm_86": 86,
    "sm_87": 87,
    "sm_88": 88,
    "sm_89": 89,
    "sm_90": 90,
    "sm_90a": 90,
    "sm_100": 100,
    "sm_103": 103,
    "sm_107": 107,
    "sm_110": 110,
    "sm_120": 120,
    "sm_121": 121,
}


def arch_expectations(sm: str) -> dict[str, Any]:
    """Expected feature availability for a compute-capability string."""
    n = _SM_ORDER.get(sm, 0)
    if n == 0:
        return {"known": False, "source": _SRC_NOTE}
    return {
        "known": True,
        "source": _SRC_NOTE,
        "cp_async_supported": n >= 80,
        "bf16_tensor_core_available": n >= 80,
        "fp8_conversion_supported": (n >= 90) or (n == 89),
        "tma_supported": n >= 90,
        # 门在 SM90a；消费级 40/50 系都拿不到
        "wgmma_supported": False,
        "mx_block_scale_supported": n >= 120,
        "tcgen05_supported": n >= 100,
    }


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------
def _unmeasured(method: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "not_measured",
        "value": None,
        "method": method,
        "n_runs": 0,
        "dispersion": None,
    }
    out.update(extra)
    return out


def _safe(cmd: list[str]) -> dict[str, Any]:
    """Run a command and capture its outcome without ever raising."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001 - probe must never crash the caller
        return {"ok": False, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _host_cpu() -> str:
    try:
        if platform.system() == "Darwin":
            return _safe(["sysctl", "-n", "machdep.cpu.brand_string"]).get("stdout") or "unknown"
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------
def _find_tool(name: str) -> str | None:
    """Locate an executable on PATH **or in the usual CUDA locations**.

    为什么不能只用 `shutil.which`：在容器镜像里 `/usr/local/cuda/bin` 常常不在
    PATH 上（AutoDL 的镜像就是如此），于是 `which nvcc` 返回空，而实际上
    nvcc 就在 `/usr/local/cuda/bin/nvcc`。第一版探针因此把"有 nvcc"误报成
    "没有 nvcc"，并顺带把 cpp_extension 的结论也带偏了。
    """
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        "/usr/local/cuda/bin/%s" % name,
        "/usr/local/cuda-12.8/bin/%s" % name,
        "/usr/local/cuda-12/bin/%s" % name,
        "/usr/local/cuda-13/bin/%s" % name,
        # ★ 必须按 name 展开：这里曾经硬编码成 "ncu"，导致查 nsys 时返回了 ncu 的
        #   路径，于是探针报告"nsys 可用"——而它其实根本没装。
        "/opt/nvidia/nsight-compute/*/%s" % name,
        "/opt/nvidia/nsight-systems/*/bin/%s" % name,
        "/opt/nvidia/nsight-systems/*/%s" % name,
        "/usr/local/bin/%s" % name,
        "/opt/conda/bin/%s" % name,
    ]
    for pat in candidates:
        if "*" in pat:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[-1]
            continue
        if os.path.isfile(pat) and os.access(pat, os.X_OK):
            return pat
    return None


def probe_host() -> dict[str, Any]:
    import os

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "host_cpu": _host_cpu(),
        "cpu_count": os.cpu_count(),
    }


def probe_driver() -> dict[str, Any]:
    """Driver / toolkit facts from nvidia-smi (absent on a Mac)."""
    return {
        "nvidia_smi": _safe(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap,clocks.sm,clocks.mem",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": _safe(["nvcc", "--version"]),
        "ptxas": _safe(["ptxas", "--version"]),
    }


def probe_torch() -> dict[str, Any]:
    """Collect what PyTorch reports. Returns available=False without torch."""
    try:
        import torch  # noqa: PLC0415 - optional heavy import, kept local on purpose
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    info: dict[str, Any] = {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": getattr(torch.version, "cuda", None),
        "cudnn_version": None,
        "device_count": 0,
        "devices": [],
    }
    if not torch.cuda.is_available():
        return info

    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001
        pass
    info["device_count"] = torch.cuda.device_count()

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        dev: dict[str, Any] = {
            "index": idx,
            "name": props.name,
            "compute_capability": "sm_%d%d" % (props.major, props.minor),
            "sm_count": props.multi_processor_count,
            "vram_total_bytes": props.total_memory,
            "vram_total_gb": round(props.total_memory / 1024**3, 2),
        }
        # 这些属性在不同 torch 版本上名字不一，取到就用
        for attr, key in (
            ("shared_memory_per_block", "shared_mem_per_block_bytes"),
            ("shared_memory_per_multiprocessor", "shared_mem_per_sm_bytes"),
            ("regs_per_multiprocessor", "regs_per_sm"),
            ("max_threads_per_multi_processor", "max_threads_per_sm"),
        ):
            if hasattr(props, attr):
                dev[key] = getattr(props, attr)
        info["devices"].append(dev)
    return info


# 仅供**合理性告警**使用的规格带宽表。
# ★ 绝不用于任何理论下限计算（AGENTS.md 规则 4：roofline 必须用实测可达带宽）。
# 它唯一的作用是：当实测值离规格差一个数量级时**大声报警**，而不是安静地写进结果。
_SPEC_BW_GBPS = {
    "V100": 900,
    "A100": 2039,
    "4090": 1008,
    "L40S": 864,
    "5090": 1792,
    "A6000": 768,
    "H100": 3350,
}


def _plausibility_check(gpu_name: str, measured: float | None) -> dict[str, Any]:
    """Warn (never silently accept) when a measurement is off by an order of magnitude.

    这个检查是有来历的，而且它真的救了一次：第一版探针漏了"除以迭代数"这一步，
    在 RTX 5090 上报出 76 GB/s（同卡真值约 1520 GB/s，恰好差 timed=20 倍）。
    结果**看起来完全合理**——数值稳定、CoV 0.1%、三次重复一致——只有跟规格一比
    才会露馅。若不加这个告警，那个数字会被当成"实测可达带宽"写进结果，
    而依据它的每一个带宽达成率都会 >100%（docs/reference/METRICS.md §3 要求解释这种情况）。
    """
    if not measured:
        return {"warning": None}
    for key, spec in _SPEC_BW_GBPS.items():
        if key.lower() in (gpu_name or "").lower():
            ratio = measured / spec
            warn = None
            if ratio < 0.25:
                warn = (
                    "measured %.1f GB/s is only %.0f%% of the %s datasheet value "
                    "%.0f GB/s -- investigate BEFORE using it for any roofline "
                    "calculation" % (measured, ratio * 100, key, spec)
                )
            return {
                "warning": warn,
                "measured_over_spec": round(ratio, 4),
                "spec_is_informational_only": True,
            }
    return {"warning": None}


def bandwidth_gbps(bytes_moved: int, total_ms: float, iters: int = 1) -> float:
    """Loop-timed bandwidth. `total_ms` is the time for **all** `iters` iterations.

    ★ 这个函数存在的唯一理由是：把那次的 bug 变成可测的。
    第一版直接 `bytes / (total_ms/1000)`，把整段循环的耗时当成一次迭代，
    于是带宽被低估了 `iters` 倍（`iters=20` 时在 5090 上报出 76 GB/s，真值 ~1520）。
    数值稳定、CoV 极低，**看起来完全可信**——只有跟规格一比才露馅。

    所以换算只此一处，且必须有测试。
    """
    if iters <= 0:
        raise ValueError("iters 必须 > 0")
    if total_ms <= 0:
        return 0.0
    per_iter_ms = total_ms / iters
    return bytes_moved / (per_iter_ms / 1000.0) / 1e9


def _bandwidth_method(size_mb: int, warmup: int, timed: int, runs: int) -> str:
    """Method string for both the measured and the skipped case.

    Sharing it means the dry-run structure describes the *same* procedure it would run —
    consumers can rely on the wording (and on the fact that it says how it is summarized).
    """
    return (
        "cuda events; **float32** buffers of %d MiB x2 (vectorized access); warmup %d, "
        "timed %d iters x %d runs; median + dispersion (copy and read-only reduction)"
        % (size_mb, warmup, timed, runs)
    )


def probe_bandwidth(*, runs: int, size_mb: int, timed: int, warmup: int) -> dict[str, Any]:
    """Measure achievable HBM bandwidth with two access patterns.

    Decode is read-dominated, so a pure device copy (read+write) **understates** the
    read ceiling. Both are therefore measured and reported.

    ★ 计时必须换算到**单次迭代**。第一版返回整段循环的总耗时却按单次字节数相除，
    在 5090 上报出 76 GB/s（真值 ~1520 GB/s，恰差 timed 倍）。用 >=4 字节的元素类型
    也重要（float32，便于向量化），但它不是那次偏差的原因。
    """
    method = _bandwidth_method(size_mb, warmup, timed, runs)
    out = _unmeasured(method, measured_gbps=None, read_only_gbps=None, pattern={})
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if not torch.cuda.is_available():
        out["error"] = "no CUDA device"
        return out

    def _time_ms(fn, iters: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        # 返回**整段循环**的总毫秒。换算到单次迭代由 bandwidth_gbps() 负责，
        # 只此一处，避免重复除法。
        return start.elapsed_time(end)

    n_bytes = size_mb * 1024 * 1024
    n_elems = n_bytes // 4  # float32 -> 4-byte elements keep the access vectorized
    bytes_copy = 2 * n_bytes  # read src + write dst
    bytes_read = n_bytes  # read src only

    # 缓冲区与闭包定义在 try 之外，这样 ruff 能静态看到绑定；
    # 分配本身单独包在 try 里（OOM 是预期内的失败模式）。
    src = dst = None
    try:
        src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
        dst = torch.empty(n_elems, dtype=torch.float32, device="cuda")
    except Exception as exc:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, exc)
        return out

    def do_copy() -> None:
        dst.copy_(src)

    def do_read() -> None:
        # 只读 pattern：求和会完整读一遍 src
        src.sum()

    try:
        for fn in (do_copy, do_read):
            for _ in range(warmup):
                fn()
        torch.cuda.synchronize()

        copy_gbps: list[float] = []
        read_gbps: list[float] = []
        for _ in range(runs):
            copy_gbps.append(bandwidth_gbps(bytes_copy, _time_ms(do_copy, timed), timed))
            read_gbps.append(bandwidth_gbps(bytes_read, _time_ms(do_read, timed), timed))

        def _med(xs: list[float]) -> float:
            s = sorted(xs)
            return s[len(s) // 2]

        def _cov(xs: list[float]) -> float:
            m = sum(xs) / len(xs)
            var = sum((x - m) ** 2 for x in xs) / len(xs)
            return (var**0.5) / m if m else 0.0

        read_med = round(_med(read_gbps), 1)
        out.update(
            {
                "status": "measured",
                "measured_gbps": round(_med(copy_gbps), 1),
                "read_only_gbps": read_med,
                # read-only 才是 decode 的相关上限
                "value": read_med,
                "n_runs": runs,
                "dispersion": {
                    "copy_cov": round(_cov(copy_gbps), 4),
                    "read_cov": round(_cov(read_gbps), 4),
                },
                "pattern": {
                    "copy": "fp32 dst.copy_(src) -> bytes = 2 x N",
                    "read_only": "fp32 src.sum() -> bytes = N",
                    "note": (
                        "decode 是读主导；用 copy 会低估上限，故两者都报。"
                        "★ 用 >=4 字节的元素类型（float32）以便向量化访问"
                    ),
                    "dtype": "float32",
                },
                "raw_gbps": {
                    "copy": [round(x, 1) for x in copy_gbps],
                    "read_only": [round(x, 1) for x in read_gbps],
                },
                "buffers": {"size_mb_each": size_mb, "count": 2, "dtype": "float32"},
                "plausibility": _plausibility_check(torch.cuda.get_device_name(0), read_med),
            }
        )
        # 用重绑定代替 del：del 会让 lint 认为这些名字在闭包里未绑定（F821）
        src = dst = None
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def probe_arch_features() -> dict[str, Any]:
    """Architecture features: expected from the LLVM-derived table, verified by probes.

    ★ `verified` stays False unless a compile/launch probe actually succeeded.
    A prior is never silently upgraded to a measurement.
    """
    out: dict[str, Any] = _unmeasured(
        "expected from NVPTX instruction predicates; verified only by compile probe"
    )
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if not torch.cuda.is_available():
        out["error"] = "no CUDA device"
        return out

    props = torch.cuda.get_device_properties(0)
    sm = "sm_%d%d" % (props.major, props.minor)
    out.update(arch_expectations(sm))
    out["compute_capability"] = sm
    out["status"] = "expected_only"
    out["verified"] = False
    out["shared_mem_per_block_bytes"] = getattr(props, "shared_memory_per_block", None)
    out["shared_mem_per_sm_bytes"] = getattr(props, "shared_memory_per_multiprocessor", None)
    out["sm_count"] = props.multi_processor_count
    return out


def probe_compile() -> dict[str, Any]:
    """Can we actually compile a kernel for this device? The core deliverable needs it."""
    out = _unmeasured("torch.utils.cpp_extension.load_inline on a trivial kernel")
    out.update(
        {
            "nvcc_present": bool(_find_tool("nvcc")),
            "ptxas_present": bool(_find_tool("ptxas")),
            "cpp_extension_compiles": None,
            "torch_cuda_build": None,
        }
    )
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["torch_cuda_build"] = getattr(torch.version, "cuda", None)

    if not torch.cuda.is_available():
        out["error"] = "no CUDA device"
        return out
    if not out["nvcc_present"]:
        out.update(
            {
                "cpp_extension_compiles": False,
                "status": "measured",
                "value": False,
                "note": (
                    "没有 nvcc：自写 CUDA kernel 不可行，改用 Triton 优先"
                    "（Triton 自带 ptxas，不需要系统 toolkit）"
                ),
            }
        )
        return out

    # ★ 让 torch 的构建系统能找到 nvcc：容器镜像里 /usr/local/cuda/bin 常常不在 PATH，
    #   而 cpp_extension 依赖 CUDA_HOME 或 PATH 来定位 nvcc。父进程环境也会被设置，
    #   这样后续任何 cpp_extension 调用都自动受益。
    nvcc_path = _find_tool("nvcc") or ""
    cuda_home = os.path.dirname(os.path.dirname(nvcc_path)) if nvcc_path else ""
    if cuda_home and os.path.isdir(cuda_home):
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["CUDA_PATH"] = cuda_home
        cuda_bin = os.path.join(cuda_home, "bin")
        if cuda_bin not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        out["cuda_home"] = cuda_home
        out["cuda_home_source"] = "derived from nvcc location (was not on PATH)"

    # ★ `cpp_sources` 必须给出**声明**：load_inline 会生成一个 main.cpp 去调用 f，
    #   如果 f 只在 cuda_sources 里定义，main.cpp 看不到它，报
    #   "'f' was not declared in this scope"。第一版传了空 cpp_sources，于是把
    #   "环境里缺 ninja / nvcc" 与"我的源码分层写错"混在了一起。
    decl = "at::Tensor f(at::Tensor x);"
    defn = "at::Tensor f(at::Tensor x){ return x + 1; }"
    try:
        from torch.utils.cpp_extension import load_inline

        mod = load_inline(
            name="lemoq_probe_compile",
            cpp_sources=decl,
            cuda_sources=defn,
            functions=["f"],
            verbose=False,
            with_cuda=True,
        )
        y = mod.f(torch.ones(4, device="cuda"))
        torch.cuda.synchronize()
        ok = bool(y[0].item() == 2.0)
        out.update({"status": "measured", "value": ok, "cpp_extension_compiles": ok, "n_runs": 1})
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "cpp_extension_compiles": False,
                "value": False,
                "status": "measured",
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            }
        )
    return out


def probe_triton() -> dict[str, Any]:
    """Does a minimal Triton kernel actually launch? Biggest single front-loaded risk."""
    out = _unmeasured("define and launch a minimal elementwise Triton kernel")
    out.update({"available": False, "triton_works": None, "triton_version": None})
    try:
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    out["available"] = True
    out["triton_version"] = getattr(triton, "__version__", None)

    try:
        import torch

        if not torch.cuda.is_available():
            out["error"] = "no CUDA device"
            return out

        @triton.jit
        def _add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            v = tl.load(x_ptr + offs, mask=mask)
            tl.store(y_ptr + offs, v + 1.0, mask=mask)

        n = 1024
        x = torch.zeros(n, device="cuda", dtype=torch.float32)
        y = torch.empty_like(x)
        _add_one[(triton.cdiv(n, 256),)](x, y, n, BLOCK=256)
        torch.cuda.synchronize()
        ok = bool(y.min().item() == 1.0 and y.max().item() == 1.0)
        out.update(
            {
                "status": "measured",
                "value": ok,
                "triton_works": ok,
                "n_runs": 1,
                "kernel": "elementwise add-one, BLOCK=256",
            }
        )
    except Exception as exc:  # noqa: BLE001
        out.update(
            {
                "triton_works": False,
                "value": False,
                "status": "measured",
                "error": f"{type(exc).__name__}: {str(exc)[:400]}",
            }
        )
    return out


def probe_profiler() -> dict[str, Any]:
    """ncu/nsys availability. Profiling is a hard requirement for the target roles."""
    ncu = _find_tool("ncu")
    nsys = _find_tool("nsys")
    out: dict[str, Any] = {
        "status": "measured",
        "ncu_available": bool(ncu),
        "nsys_available": bool(nsys),
        "ncu_path": ncu,
        "nsys_path": nsys,
        "method": "PATH plus the usual CUDA / Nsight install locations",
        "n_runs": 1,
        "dispersion": None,
    }
    out["value"] = bool(ncu or nsys)
    if ncu:
        out["ncu_version"] = _safe(["ncu", "--version"]).get("stdout", "").split("\n")[0]
    if nsys:
        out["nsys_version"] = _safe(["nsys", "--version"]).get("stdout", "").split("\n")[0]
    return out


def probe_disk(paths: list[str]) -> dict[str, Any]:
    mounts = []
    for p in paths:
        try:
            target = p if Path(p).exists() else "."
            usage = shutil.disk_usage(target)
            mounts.append(
                {
                    "path": p,
                    "exists": Path(p).exists(),
                    "total_gib": round(usage.total / 1024**3, 1),
                    "free_gib": round(usage.free / 1024**3, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001
            mounts.append({"path": p, "error": f"{type(exc).__name__}: {exc}"})
    first_free = mounts[0].get("free_gib") if mounts else None
    return {
        "status": "measured",
        "value": first_free,
        "method": "shutil.disk_usage",
        "n_runs": 1,
        "dispersion": None,
        "mounts": mounts,
        "need_gib": {
            "olmoe_fp16": 13.0,
            "qwen0.5b": 1.0,
            "torch_and_cuda": 15.0,
            "recommended_free": 40.0,
        },
    }


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def build_report(
    *, dry_run: bool, runs: int, size_mb: int, timed: int, warmup: int, skip_heavy: bool
) -> dict[str, Any]:
    torch_info = probe_torch()

    if dry_run or skip_heavy:
        why = "heavy probe skipped" if skip_heavy else "dry-run: no measurement attempted"
        bandwidth = _unmeasured(
            _bandwidth_method(size_mb, warmup, timed, runs),
            measured_gbps=None,
            read_only_gbps=None,
            pattern={"copy": "dst.copy_(src)", "read_only": "float32 view sum"},
            note="Do NOT substitute the datasheet number for this value.",
            skipped_because=why,
        )
        arch = _unmeasured("expected from NVPTX predicates; verified only by compile probe")
        arch.update(
            {
                "known": None,
                "cp_async_supported": None,
                "bf16_tensor_core_available": None,
                "fp8_conversion_supported": None,
                "tma_supported": None,
                "wgmma_supported": None,
                "mx_block_scale_supported": None,
                "tcgen05_supported": None,
                "verified": False,
                "shared_mem_per_block_bytes": None,
                "skipped_because": why,
            }
        )
        compile_info = _unmeasured(
            "torch.utils.cpp_extension.load_inline on a trivial kernel",
            cpp_extension_compiles=None,
            nvcc_present=bool(_find_tool("nvcc")),
            skipped_because=why,
        )
        triton_info = _unmeasured(
            "minimal elementwise Triton kernel launch",
            available=False,
            triton_works=None,
            skipped_because=why,
        )
    else:
        bandwidth = probe_bandwidth(runs=runs, size_mb=size_mb, timed=timed, warmup=warmup)
        arch = probe_arch_features()
        compile_info = probe_compile()
        triton_info = probe_triton()

    # Gate-0 会从 arch_features 里读 triton_works
    arch["triton_works"] = triton_info.get("triton_works")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": bool(dry_run),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": probe_host(),
        "driver": probe_driver(),
        "torch": torch_info,
        "bandwidth": bandwidth,
        "arch_features": arch,
        "triton": triton_info,
        "profiler": probe_profiler(),
        "compile": compile_info,
        "disk": probe_disk(["results", "."]),
    }

    if not dry_run and not torch_info.get("cuda_available", False):
        report.setdefault("warnings", []).append(
            "No CUDA device visible: this report must not be used for any measurement."
        )

    # Deviations from the project's assumed hardware are called out, not silently accepted.
    deviations: list[str] = []
    assumed_sm = "sm_120"
    for dev in torch_info.get("devices", []):
        sm = dev.get("compute_capability")
        if sm != assumed_sm:
            deviations.append(
                "device %s is %s, project assumption is %s "
                "(arms are mounted per arch; see docs/reference/SPECS.md 2.2)"
                % (dev.get("index"), sm, assumed_sm)
            )
        if (dev.get("vram_total_gb") or 0) < 20:
            deviations.append(
                "device %s has %s GB, expected 24-32 GB"
                % (dev.get("index"), dev.get("vram_total_gb"))
            )
    if deviations:
        report["deviations"] = deviations
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record machine facts (GPU, driver, measured bandwidth, arch features).",
    )
    parser.add_argument(
        "--json", metavar="PATH", help="write the report to this path (default: stdout)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="produce the FULL report structure without measuring anything",
    )
    parser.add_argument(
        "--skip-heavy",
        action="store_true",
        help="skip bandwidth/compile/Triton probes (fast, structure only)",
    )
    parser.add_argument("--runs", type=int, default=3, help="bandwidth repeats (>=3)")
    parser.add_argument("--size-mb", type=int, default=512, help="bandwidth buffer MiB per side")
    parser.add_argument("--timed", type=int, default=20, help="timed iterations per run")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    args = parser.parse_args(argv)

    report = build_report(
        dry_run=args.dry_run,
        runs=args.runs,
        size_mb=args.size_mb,
        timed=args.timed,
        warmup=args.warmup,
        skip_heavy=args.skip_heavy,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % path)
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
