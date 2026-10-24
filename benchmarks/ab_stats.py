"""ab_stats — 配对 A/B 的显著性口径（MATURITY_REVIEW_2026-10-24 R1；METRICS.md §2 的延伸）。

为什么需要：同轮交错 A/B 已经输出 per-pair wins 与 MAD 警示，但没有正式检验——
"3/3 pairs 更快"到底有多强，靠眼睛数不可复核。torch.utils.benchmark 的 significant
标记、google/benchmark 的随机交错是同一诉求的工业版；本模块给出可复核的
p 值与区间（纯标准库，Mac/CI 均可运行）。

口径（不得妥协）：

    - 配对差 delta_i = arm_a_i - arm_b_i，与既有 `wins` 的方向一致（正 = arm_b 更快）；
    - 符号检验：丢并列（ties），n_eff = wins+losses，精确二项双侧 p；
    - Wilcoxon 符号秩：n_eff<=20 用精确零分布（对秩的 2^n 符号翻转做子集和 DP），
      否则正态近似（含并列秩校正与连续性校正）；
    - 中位差 CI：成对 bootstrap 的百分位区间，**固定种子 ⇒ 结果可复现**；
    - p 值不替代 MAD 口径：|median_delta| < MAD 时仍按"不可分辨"处理（METRICS.md §2），
      显著块里的 `significant_at_0_05` 只是 p 值的机械转述，不是采纳判据。
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Any, Sequence

from benchmarks.timing import percentile

# 默认 bootstrap 种子：固定值 ⇒ 同一份数据在任何机器上重算得到同一区间。
DEFAULT_BOOT_SEED = 20261024
DEFAULT_N_BOOT = 10000
WILCOXON_EXACT_MAX_N = 20


def _clean(deltas: Sequence[float]) -> list[float]:
    """丢掉非有限值（与 tools/round_canary 的 fail-closed 口径一致：脏数据不静默变 0）。"""
    return [float(d) for d in deltas if math.isfinite(float(d))]


def sign_test(deltas: Sequence[float]) -> dict[str, Any]:
    """精确二项符号检验（双侧）。ties 丢弃并如实报告。"""
    ds = _clean(deltas)
    wins_b = sum(1 for d in ds if d > 0)  # arm_b 更快
    wins_a = sum(1 for d in ds if d < 0)
    ties = len(ds) - wins_b - wins_a
    n_eff = wins_b + wins_a
    out: dict[str, Any] = {
        "wins_b_faster": wins_b,
        "wins_a_faster": wins_a,
        "ties": ties,
        "n_eff": n_eff,
        "p_two_sided": None,
    }
    if n_eff == 0:
        out["note"] = "全部并列或无有效对，无方向信息"
        return out
    k = min(wins_b, wins_a)
    p_le = sum(math.comb(n_eff, i) for i in range(k + 1)) / 2**n_eff
    out["p_two_sided"] = round(min(1.0, 2.0 * p_le), 6)
    return out


def _average_ranks(abs_vals: Sequence[float]) -> list[float]:
    """并列取平均秩（Wilcoxon 的标准处理）。"""
    order = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i])
    ranks = [0.0] * len(abs_vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs_vals[order[j + 1]] == abs_vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 秩从 1 开始
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_exact_p(w_plus: float, ranks: Sequence[float]) -> float:
    """W+ 的精确零分布：对每条秩做 +/- 翻转，子集和 DP 数出分布（n<=20 可行）。"""
    counts: dict[float, int] = {0.0: 1}
    for r in ranks:
        nxt: dict[float, int] = {}
        for s, c in counts.items():
            nxt[s] = nxt.get(s, 0) + c
            nxt[s + r] = nxt.get(s + r, 0) + c
        counts = nxt
    total = sum(counts.values())
    p_ge = sum(c for s, c in counts.items() if s >= w_plus) / total
    p_le = sum(c for s, c in counts.items() if s <= w_plus) / total
    return min(1.0, 2.0 * min(p_ge, p_le))


def _wilcoxon_normal_p(w_plus: float, ranks: Sequence[float]) -> float:
    """正态近似（含并列秩校正与连续性校正）。"""
    n = len(ranks)
    mean = n * (n + 1) / 4.0
    tie_groups: dict[float, int] = {}
    for r in ranks:
        tie_groups[r] = tie_groups.get(r, 0) + 1
    tie_corr = sum(t**3 - t for t in tie_groups.values()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        return float("nan")
    sd = math.sqrt(var)
    z = (w_plus - mean) / sd
    z -= math.copysign(0.5 / sd, z)  # 连续性校正：向零收缩
    return min(1.0, math.erfc(abs(z) / math.sqrt(2.0)))


def wilcoxon_signed_rank(deltas: Sequence[float]) -> dict[str, Any]:
    """Wilcoxon 符号秩检验（双侧）。ties（差为 0）丢弃。"""
    ds = [d for d in _clean(deltas) if d != 0]
    n_eff = len(ds)
    out: dict[str, Any] = {"n_eff": n_eff, "p_two_sided": None, "method": None}
    if n_eff == 0:
        out["note"] = "无有效非零对"
        return out
    ranks = _average_ranks([abs(d) for d in ds])
    w_plus = sum(r for r, d in zip(ranks, ds) if d > 0)
    out["w_plus"] = w_plus
    if n_eff <= WILCOXON_EXACT_MAX_N:
        out["method"] = "exact"
        out["p_two_sided"] = round(_wilcoxon_exact_p(w_plus, ranks), 6)
    else:
        out["method"] = "normal_approx"
        out["p_two_sided"] = round(_wilcoxon_normal_p(w_plus, ranks), 6)
    return out


def median_delta_ci(
    deltas: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, Any]:
    """中位差 bootstrap 百分位 CI。固定种子 ⇒ 确定性；样本不足 2 对时返回 None。"""
    ds = _clean(deltas)
    if len(ds) < 2:
        return {"ci": None, "n_boot": n_boot, "seed": seed, "note": "n<2 无法 bootstrap"}
    rng = random.Random(seed)
    meds: list[float] = []
    for _ in range(n_boot):
        sample = [ds[rng.randrange(len(ds))] for _ in range(len(ds))]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = percentile(meds, 100.0 * (alpha / 2.0))
    hi = percentile(meds, 100.0 * (1.0 - alpha / 2.0))
    return {"ci": [round(lo, 4), round(hi, 4)], "n_boot": n_boot, "seed": seed}


def paired_significance(
    deltas: Sequence[float],
    *,
    label_a: str,
    label_b: str,
    alpha: float = 0.05,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_BOOT_SEED,
) -> dict[str, Any]:
    """一个配对 A/B 的完整显著性块（进 payload 用；含方向与口径声明）。"""
    ds = _clean(deltas)
    n = len(ds)
    sign = sign_test(ds)
    wilx = wilcoxon_signed_rank(ds)
    ci = median_delta_ci(ds, alpha=alpha, n_boot=n_boot, seed=seed)
    median = statistics.median(ds) if ds else None
    sig = sign["p_two_sided"]
    out: dict[str, Any] = {
        "n_pairs": n,
        "delta_def": "delta = %s - %s；正 = %s 更快" % (label_a, label_b, label_b),
        "wins": {
            "%s_faster" % label_b: sign["wins_b_faster"],
            "%s_faster" % label_a: sign["wins_a_faster"],
            "ties": sign["ties"],
        },
        "median_delta_ms": round(median, 4) if median is not None else None,
        "sign_test_p": sig,
        "wilcoxon_p": wilx["p_two_sided"],
        "wilcoxon_method": wilx["method"],
        "median_delta_ci95_ms": ci["ci"],
        "bootstrap": {"n_boot": ci["n_boot"], "seed": ci["seed"]},
        "significant_at_0_05": bool(sig is not None and sig < 0.05),
    }
    if n == 0:
        out["note"] = "无有效配对"
    elif sign["n_eff"] < 5:
        out["note"] = (
            "有效对 n_eff=%d < 5：任何 p 值都几乎不可能显著，如实报告而不是隐藏" % sign["n_eff"]
        )
    out["caveat"] = "p 值不替代 MAD 口径：|median_delta| < MAD 时仍按不可分辨处理（METRICS.md §2）"
    return out
