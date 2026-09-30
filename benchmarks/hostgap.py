"""hostgap — 把单步时间拆成 **GPU 时间 vs host/launch 开销**。

为什么这是一等指标（`docs/METRICS.md` §2.1）：

    在 batch=1，`T_host` **不随量化下降**，所以量化对 token 延迟的收益被截断：

        speedup_E2E = (T_bytes_fp16 + T_host) / (T_bytes_quant + T_host)

    只报"量化后 GEMM 快 N 倍"而不报 host 占比，等于无法解释端到端结果。

测量方式与**已知局限**：

    - `wall_ms`：perf_counter 包住 N 步（含 host 与 launch），最后 synchronize；
    - `gpu_ms`  ：CUDA event 包住同一段内的 kernel 执行；
    - `host_ms` = wall - gpu。**这是一个残差，不是直接测量**：
      kernel 之间有重叠、event 本身有开销、框架缓冲会摊进 wall。
    - 因此本模块要求 **nsys 交叉验证**（`verify_with_nsys` 字段），
      并在报告里声明该残差的性质。

在无 CUDA 的机器上返回 `status: "not_measured"`，绝不编造数字。
"""

from __future__ import annotations

import time
from typing import Any, Callable

from benchmarks.timing import summarize


def measure_host_gap(
    fn: Callable[[], Any], *, warmup: int = 5, timed: int = 20, repeats: int = 3
) -> dict[str, Any]:
    """Measure wall time and GPU time for the same callable, then split."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "not_measured",
            "reason": "no torch: %s" % type(exc).__name__,
            "method": "wall(perf_counter) vs gpu(cuda events)",
        }

    if not torch.cuda.is_available():
        return {
            "status": "not_measured",
            "reason": "no CUDA device",
            "method": "wall(perf_counter) vs gpu(cuda events)",
        }

    warmup = max(warmup, 5)
    timed = max(timed, 20)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    wall_ms: list[float] = []
    gpu_ms: list[float] = []
    for _ in range(repeats):
        # 先量 GPU：event 只包 kernel
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(timed):
            fn()
        end.record()
        torch.cuda.synchronize()
        gpu_ms.append(start.elapsed_time(end) / timed)

        # 再量 wall：同一段，含 host 与 launch
        t0 = time.perf_counter()
        for _ in range(timed):
            fn()
        torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0 / timed)

    w = summarize(wall_ms)
    g = summarize(gpu_ms)
    host_ms = [max(a - b, 0.0) for a, b in zip(w["samples"], g["samples"])]
    h = summarize(host_ms)
    total = w["p50"]

    return {
        "status": "measured",
        "method": "wall(perf_counter over step) vs gpu(cuda events over same step)",
        "warmup": warmup,
        "timed": timed,
        "repeats": repeats,
        "wall_p50_ms": w["p50"],
        "wall_p99_ms": w["p99"],
        "gpu_p50_ms": g["p50"],
        "gpu_p99_ms": g["p99"],
        "host_p50_ms": h["p50"],
        "host_p99_ms": h["p99"],
        "host_share_pct": round(100.0 * h["p50"] / total, 1) if total else 0.0,
        "gpu_share_pct": round(100.0 * g["p50"] / total, 1) if total else 0.0,
        "stats": {"wall": w, "gpu": g, "host": h},
        "is_residual": True,
        "verify_with_nsys": False,
        "caveats": [
            "host_ms 是 wall - gpu 的**残差**，不是直接测量",
            "kernel 之间的重叠、event 开销、框架缓冲都会摊进它",
            "必须用 nsys 交叉验证后才能写进结论（见 docs/METRICS.md 2）",
        ],
    }


def truncation_prediction(
    *, w_bytes_fp16: int, w_bytes_quant: int, host_ms: float, measured_bw_gbps: float
) -> dict[str, Any]:
    """Given a host overhead, predict the E2E speedup ceiling from quantization.

    This is the quantitative form of E7's hypothesis: the byte reduction is capped by
    the host overhead that quantization does NOT remove.
    """
    if measured_bw_gbps <= 0:
        return {"status": "not_measurable", "reason": "need measured_bw_gbps > 0"}
    t_fp16 = (w_bytes_fp16 / (measured_bw_gbps * 1e9)) * 1000.0
    t_quant = (w_bytes_quant / (measured_bw_gbps * 1e9)) * 1000.0
    denom = t_quant + host_ms
    speedup = (t_fp16 + host_ms) / denom if denom > 0 else float("inf")
    return {
        "status": "derived",
        "note": "推导值，不是实测。必须上机验证（E7）。",
        "t_bytes_fp16_ms": round(t_fp16, 4),
        "t_bytes_quant_ms": round(t_quant, 4),
        "host_ms": host_ms,
        "byte_reduction_x": round(w_bytes_fp16 / w_bytes_quant, 2) if w_bytes_quant else None,
        "predicted_e2e_speedup_x": round(speedup, 3),
        "host_share_at_fp16_pct": round(100.0 * host_ms / (t_fp16 + host_ms), 1),
        "interpretation": (
            "字节降到 1/N，但端到端只有 predicted_e2e_speedup_x —— "
            "差额就是被 host 开销吃掉的部分（E7 要证伪/证实的对象）"
        ),
    }
