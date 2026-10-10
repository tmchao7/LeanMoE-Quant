"""timing — 计时与统计口径（`docs/reference/METRICS.md` §2）。

口径要求（不得妥协）：

    - warmup >= 5，计次 >= 20；
    - 报 **p50 / p99** 与离散度，**不得用 min-of-3** 这类旧做法作为正式数字；
    - **JIT / autotune 的首次编译时间不计入稳态数字，但必须单独记录**；
    - 必须能回答"是否包含 kernel launch 开销"。

统计部分是**纯 Python**（可在 Mac 的最小环境运行）；计时部分需要 torch，做了延迟导入。
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Sequence

MIN_WARMUP = 5
MIN_TIMED = 20


# ---------------------------------------------------------------------------
# 统计（纯 Python）
# ---------------------------------------------------------------------------
def percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default)."""
    if not sorted_xs:
        raise ValueError("empty sample")
    if len(sorted_xs) == 1:
        return float(sorted_xs[0])
    if q <= 0:
        return float(sorted_xs[0])
    if q >= 100:
        return float(sorted_xs[-1])
    pos = (q / 100.0) * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_xs) - 1)
    frac = pos - lo
    return float(sorted_xs[lo]) * (1.0 - frac) + float(sorted_xs[hi]) * frac


def summarize(samples: Sequence[float]) -> dict[str, Any]:
    """Full distribution report. Never report a bare mean for a latency claim."""
    xs = [float(x) for x in samples]
    if not xs:
        raise ValueError("empty sample")
    s = sorted(xs)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    sd = math.sqrt(var)
    return {
        "n": n,
        "mean": mean,
        "stddev": sd,
        "min": s[0],
        "max": s[-1],
        "p5": percentile(s, 5),
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "cov": (sd / mean) if mean else 0.0,
        "samples": xs,
        "summarization": (
            "report median/mean/stddev/min/max/p5/p95/p99 + coefficient of variation; "
            "do NOT use min-of-N as the headline number"
        ),
    }


def coefficient_of_variation(samples: Sequence[float]) -> float:
    st = summarize(samples)
    return float(st["cov"])


def is_stable(samples: Sequence[float], max_cov: float = 0.05) -> bool:
    """A cheap 'is this measurement trustworthy?' gate. High CoV means investigate."""
    return coefficient_of_variation(samples) <= max_cov


# ---------------------------------------------------------------------------
# 计时（需要 torch；无 torch 时退化为 perf_counter）
# ---------------------------------------------------------------------------
def _device_synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass


def measure(
    fn: Callable[[], Any], *, warmup: int = MIN_WARMUP, timed: int = MIN_TIMED, repeats: int = 3
) -> dict[str, Any]:
    """Time `fn` with warmup, then `timed` iterations x `repeats`, returning per-repeat
    means (so repeats give us dispersion across repeats, not just within one).

    Records whether CUDA events were used and whether launch overhead is included.
    """
    if warmup < MIN_WARMUP:
        warmup = MIN_WARMUP
    if timed < MIN_TIMED:
        timed = MIN_TIMED

    use_cuda_events = False
    torch = None
    try:
        import torch as _torch

        torch = _torch
        use_cuda_events = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        torch = None

    for _ in range(warmup):
        fn()
    _device_synchronize()

    per_repeat_ms: list[float] = []
    for _ in range(repeats):
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timed):
                fn()
            end.record()
            torch.cuda.synchronize()
            per_repeat_ms.append(start.elapsed_time(end) / timed)
        else:
            t0 = time.perf_counter()
            for _ in range(timed):
                fn()
            _device_synchronize()
            per_repeat_ms.append((time.perf_counter() - t0) * 1000.0 / timed)

    return {
        "warmup": warmup,
        "timed": timed,
        "repeats": repeats,
        "timing_method": "cuda_events" if use_cuda_events else "perf_counter",
        "includes_launch_overhead": True,
        "per_repeat_ms": per_repeat_ms,
        "stats": summarize(per_repeat_ms),
    }


def measure_wall(
    fn: Callable[[], Any], *, warmup: int = MIN_WARMUP, timed: int = MIN_TIMED, repeats: int = 3
) -> dict[str, Any]:
    """Wall-clock timing of the whole step, i.e. **including host/launch overhead**.

    This is the number a user experiences; `hostgap.py` splits it into GPU vs host.
    """
    for _ in range(max(warmup, MIN_WARMUP)):
        fn()
    _device_synchronize()

    per_repeat_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(max(timed, MIN_TIMED)):
            fn()
        _device_synchronize()
        per_repeat_ms.append((time.perf_counter() - t0) * 1000.0 / max(timed, MIN_TIMED))

    return {
        "warmup": max(warmup, MIN_WARMUP),
        "timed": max(timed, MIN_TIMED),
        "repeats": repeats,
        "timing_method": "perf_counter_wall",
        "includes_launch_overhead": True,
        "per_repeat_ms": per_repeat_ms,
        "stats": summarize(per_repeat_ms),
    }


def record_compile_time(fn: Callable[[], Any]) -> dict[str, Any]:
    """Time a one-off compile/autotune. MUST NOT be mixed into steady-state numbers."""
    t0 = time.perf_counter()
    result = fn()
    _device_synchronize()
    return {
        "compile_or_autotune_ms": (time.perf_counter() - t0) * 1000.0,
        "excluded_from_steady_state": True,
        "note": "docs/reference/METRICS.md 2: JIT/autotune first-compile time is recorded separately",
        "result_repr": repr(result)[:200],
    }
