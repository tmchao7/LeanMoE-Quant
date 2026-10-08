"""test_stats — 统计口径（docs/METRICS.md §2）。

要点：
    - 必须报 p50/p99，**不得用 min-of-N** 当正式数字；
    - 必须能识别"不稳定"的测量（高 CoV 要触发复采）。
"""

from __future__ import annotations

import pytest

from benchmarks.timing import (
    MIN_TIMED,
    MIN_WARMUP,
    coefficient_of_variation,
    is_stable,
    percentile,
    summarize,
)


def test_percentile_matches_numpy_convention():
    """★ 容差必须窄到能**区分** percentile 的几种约定。

    旧写法是 `abs=0.51`，对 1..100 而言 numpy 的线性插值(50.5) / lower(50) /
    higher(51) / nearest(50) 全部落在 ±0.51 内 —— 把实现改成 `nearest` 或
    `lower` 这条测试也全绿，"matches numpy convention" 名不副实。
    钉住线性插值需要 1e-9 量级。
    """
    xs = [float(i) for i in range(1, 101)]  # 1..100
    # pos = q/100 * (n-1) = 0.99*99 … 线性插值
    assert percentile(xs, 50) == pytest.approx(50.5, abs=1e-9)
    assert percentile(xs, 5) == pytest.approx(5.95, abs=1e-9)
    assert percentile(xs, 95) == pytest.approx(95.05, abs=1e-9)
    assert percentile(xs, 0) == 1.0
    assert percentile(xs, 100) == 100.0
    # 判别力：其它约定必须**落在外面**，否则这条断言没有意义
    assert abs(percentile(xs, 50) - 50.0) > 1e-9, "lower 约定"
    assert abs(percentile(xs, 50) - 51.0) > 1e-9, "higher 约定"


def test_percentile_single_sample():
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 99) == 7.0


def test_percentile_empty_raises():
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarize_reports_full_distribution():
    st = summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    for key in ("n", "mean", "stddev", "min", "max", "p5", "p50", "p95", "p99", "cov"):
        assert key in st
    assert st["n"] == 5
    assert st["mean"] == pytest.approx(3.0)
    assert st["min"] == 1.0 and st["max"] == 5.0
    assert st["p50"] == 3.0
    assert "min-of-N" in st["summarization"]


def test_cov_zero_for_constant_samples():
    assert coefficient_of_variation([2.0, 2.0, 2.0]) == 0.0
    assert is_stable([2.0, 2.0, 2.0])


def test_high_variance_is_flagged_unstable():
    noisy = [1.0, 10.0, 1.0, 10.0, 1.0]
    assert not is_stable(noisy, max_cov=0.05)
    assert coefficient_of_variation(noisy) > 0.05


def test_minimum_warmup_and_timed_are_enforced_constant():
    """口径下限写死在模块里，任何调用点都不能偷偷降低。"""
    assert MIN_WARMUP >= 5
    assert MIN_TIMED >= 20


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        summarize([])
