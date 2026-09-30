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


def gpu_busy_ms(fn: Callable[[], Any], *, warmup: int = 3, iters: int = 10) -> dict[str, Any]:
    """Sum of per-kernel GPU self time, from the profiler.

    ★ 为什么必须用它，而不是 `wall - gpu_event`：
    在 **launch-bound** 的情况下，GPU 大部分时间在等 host 把下一个 kernel 发过来。
    CUDA event 的窗口是从第一个 kernel 到最后一个 kernel，而 host 的等待被夹在中间，
    于是 event 测到的 ≈ wall —— 相减得 0。
    实测（OLMoE fp16, batch=1, ctx=1024）：

        wall 140.3 ms | event 窗口 99.0 ms | profiler 真实 GPU 忙 10.4 ms

    残差法给出 host_share = 0.0%，而真值是 ~93%。这个错误的方向很危险：
    它会把"launch 开销是主要瓶颈"误判成"没有 host 开销"，从而让人以为
    只要降字节就能线性加速 —— 而实际上量化收益会被 launch 完全吃掉。
    """
    try:
        import torch
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "not_measured",
            "reason": "%s" % type(exc).__name__,
            "method": "torch.profiler self_device_time_total",
        }
    if not torch.cuda.is_available():
        return {
            "status": "not_measured",
            "reason": "no CUDA device",
            "method": "torch.profiler self_device_time_total",
        }
    for _ in range(max(warmup, 3)):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(max(iters, 5)):
            fn()
        torch.cuda.synchronize()
    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    n = max(iters, 5)
    return {
        "status": "measured",
        "method": "torch.profiler CUDA self_device_time_total, summed over kernels",
        "gpu_busy_ms": sum(e.self_device_time_total for e in evts) / n / 1000.0,
        "kernels_per_call": sum(e.count for e in evts) / float(n),
        "note": "这是 GPU 真正在执行 kernel 的时间；与 wall 的差 = host/launch/空闲",
    }


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

    # ★ 残差法在 launch-bound 时会出错（见 gpu_busy_ms 的说明），所以这里
    #   额外用 profiler 直接测 GPU 忙时间，并**以它**为准计算 host 占比。
    busy = gpu_busy_ms(fn, warmup=warmup, iters=timed)
    busy_ms = busy.get("gpu_busy_ms")
    host_pct = None
    if busy_ms is not None and total:
        host_pct = round(100.0 * max(total - busy_ms, 0.0) / total, 1)

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
        # 保留下面的残差口径（供对照），但**结论要用 gpu_busy 那一组**
        "host_share_pct_residual": round(100.0 * h["p50"] / total, 1) if total else 0.0,
        "gpu_share_pct_event": round(100.0 * g["p50"] / total, 1) if total else 0.0,
        # ★ 以 profiler 的 GPU 忙时间为准：这是 launch-bound 下唯一正确的口径
        "gpu_busy_ms": busy_ms,
        "kernels_per_call": busy.get("kernels_per_call"),
        "host_share_pct": host_pct,
        "gpu_share_pct": (
            round(100.0 * busy_ms / total, 1) if (busy_ms is not None and total) else None
        ),
        "host_share_pct_primary_is": "gpu_busy_ms (profiler)",
        "stats": {"wall": w, "gpu_event": g, "host_residual": h},
        "is_residual": True,
        "residual_is_misleading_when_launch_bound": True,
        "verify_with_nsys": False,
        "caveats": [
            "host_share_pct 以 profiler 的 GPU 忙时间为基准 —— 残差法在 launch-bound "
            "时会把 host 占比算成 0（实测：真值 ~93%，残差给出 0.0%）",
            "host_ms 仍是残差，不是直接测量；kernel 间重叠与 event 开销会摊进它",
            "profiler 自身有观测开销，但对'GPU 忙 vs 空闲'的量级判断足够稳健",
            "nsys 不可用（本镜像未安装），故无法做第三方交叉验证（docs/METRICS.md 2）",
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
