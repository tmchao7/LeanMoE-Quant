"""test_ab_stats — 配对 A/B 显著性口径（benchmarks/ab_stats.py；R1）。

钉住的口径：
    - 符号检验是**精确二项**（不是正态近似）——n=8 全胜的 p 必须等于 2/256；
    - Wilcoxon 小样本走精确分布，测试里用暴力枚举 2^n 符号翻转交叉验证 DP；
    - bootstrap CI **确定性**（同种子同结果），这是"可复核"的底线；
    - 小样本（n_eff<5）不装作显著，note 如实写。
"""

from __future__ import annotations

import itertools
import random
import statistics

import pytest

from benchmarks.ab_stats import (
    _average_ranks,
    _wilcoxon_exact_p,
    median_delta_ci,
    paired_significance,
    sign_test,
    wilcoxon_signed_rank,
)

# 模块内 p 值 round 到 6 位小数,断言容差放宽到 1e-6 覆盖舍入。
P_TOL = 1e-6


def test_sign_test_exact_binomial_all_wins():
    st = sign_test([0.1] * 8)
    assert st["n_eff"] == 8
    assert st["p_two_sided"] == pytest.approx(2.0 / 256.0, abs=P_TOL)


def test_sign_test_exact_binomial_seven_and_one():
    # 7 胜 1 负：p = 2 * (C(8,0)+C(8,1)) / 256 = 18/256
    st = sign_test([0.1] * 7 + [-0.1])
    assert st["p_two_sided"] == pytest.approx(18.0 / 256.0, abs=P_TOL)


def test_sign_test_drops_ties_and_reports_them():
    st = sign_test([1.0, -2.0, 0.0, 0.0, 3.0])
    assert st["ties"] == 2
    assert st["n_eff"] == 3
    # 2 胜 1 负：p = 2 * (C(3,0)+C(3,1)) / 8 = 1.0
    assert st["p_two_sided"] == pytest.approx(1.0, abs=P_TOL)


def test_sign_test_all_ties_returns_none():
    st = sign_test([0.0, 0.0, 0.0])
    assert st["p_two_sided"] is None
    assert st["n_eff"] == 0
    assert "note" in st


def test_sign_test_caps_at_one():
    st = sign_test([1.0, -1.0])
    assert st["p_two_sided"] == 1.0


def _brute_wilcoxon_two_sided(ds: list[float]) -> float:
    """暴力枚举 2^n 个"哪些秩取正号"的子集（= W+ 的零分布;与 DP 相互独立的第二实现,防自证）。"""
    ranks = _average_ranks([abs(d) for d in ds])
    w_obs = sum(r for r, d in zip(ranks, ds) if d > 0)
    sums = [
        sum(b * r for b, r in zip(mask, ranks))
        for mask in itertools.product((0, 1), repeat=len(ranks))
    ]
    p_ge = sum(1 for s in sums if s >= w_obs) / len(sums)
    p_le = sum(1 for s in sums if s <= w_obs) / len(sums)
    return min(1.0, 2.0 * min(p_ge, p_le))


def test_wilcoxon_exact_matches_brute_force():
    rng = random.Random(7)
    checked = 0
    for n in range(1, 11):
        for _ in range(20):
            vec = [
                round(rng.uniform(-2.0, 2.0), 3) if rng.random() > 0.2 else 1.5  # 造出并列秩
                for _ in range(n)
            ]
            ds = [d for d in vec if d != 0]
            if not ds:
                continue
            ranks = _average_ranks([abs(d) for d in ds])
            w_plus = sum(r for r, d in zip(ranks, ds) if d > 0)
            assert _wilcoxon_exact_p(w_plus, ranks) == pytest.approx(
                _brute_wilcoxon_two_sided(ds), abs=P_TOL
            )
            checked += 1
    assert checked >= 150, "交叉验证样本量不足,测试可能空转"


def test_wilcoxon_all_positive_small_n():
    w = wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0])
    assert w["method"] == "exact"
    # 全胜：p = 2 * 1/32
    assert w["p_two_sided"] == pytest.approx(2.0 / 32.0, abs=P_TOL)


def test_wilcoxon_switches_to_normal_approx():
    rng = random.Random(11)
    ds = [rng.uniform(-1.0, 1.0) for _ in range(25)]
    w = wilcoxon_signed_rank(ds)
    assert w["method"] == "normal_approx"
    assert w["p_two_sided"] is not None
    assert 0.0 <= w["p_two_sided"] <= 1.0


def test_wilcoxon_empty_returns_none():
    assert wilcoxon_signed_rank([])["p_two_sided"] is None
    assert wilcoxon_signed_rank([0.0, 0.0])["p_two_sided"] is None


def test_bootstrap_ci_deterministic_and_degenerate():
    a = median_delta_ci([1.0, 2.0, 3.0])
    b = median_delta_ci([1.0, 2.0, 3.0])
    assert a["ci"] == b["ci"], "同种子必须同区间"
    c = median_delta_ci([2.5, 2.5, 2.5, 2.5])
    assert c["ci"] == [2.5, 2.5]


def test_bootstrap_ci_brackets_sample_median():
    ds = [1.0, 2.0, 3.0, 4.0, 10.0]
    ci = median_delta_ci(ds)["ci"]
    med = statistics.median(ds)
    assert ci[0] <= med <= ci[1]


def test_bootstrap_ci_requires_two_pairs():
    assert median_delta_ci([1.0])["ci"] is None
    assert median_delta_ci([])["ci"] is None


def test_paired_significance_shape_and_direction():
    out = paired_significance(
        [0.2, 0.3, 0.25, 0.1, 0.4, 0.35, 0.22, 0.31], label_a="off", label_b="on"
    )
    assert out["n_pairs"] == 8
    assert out["wins"]["on_faster"] == 8
    assert out["median_delta_ms"] > 0  # 正 = on 更快
    assert out["significant_at_0_05"] is True  # 8/8 全胜 p=0.0078
    assert out["delta_def"] == "delta = off - on；正 = on 更快"
    assert out["median_delta_ci95_ms"][0] > 0
    assert "MAD" in out["caveat"]


def test_paired_significance_small_n_note_is_honest():
    out = paired_significance([0.2, 0.3], label_a="off", label_b="on")
    assert out["n_pairs"] == 2
    assert out["significant_at_0_05"] is False  # 2/2 胜 p=0.5
    assert "n_eff=2 < 5" in out["note"]


def test_paired_significance_empty():
    out = paired_significance([], label_a="a", label_b="b")
    assert out["n_pairs"] == 0
    assert out["sign_test_p"] is None
    assert out["median_delta_ci95_ms"] is None
