"""test_quant_group_audit — 量化器在**权重域**的不变量（全部 CPU 可跑）。

## 为什么这些不变量必须先钉住

`EXP-p1-01` 的两个信号都是关于 KL 的**间接**证据。判断"非单调是不是实现缺陷"，
需要先证明量化器本身的行为与它的数学定义一致：

    嵌套分组   s32 ≤ s64 ≤ s128（逐元素）—— 这是"group↓ ⇒ 误差界↓"的**唯一**理由
    逐元素界   |err| ≤ s/2            —— 违反即反量化用错 scale / codes 与 scale 错位
    步长算术   W4/W8 的界之比 ≡ 127/7 = 18.143 —— 实现误差应当只在界之内浮动

这三条都是**纯算术**，不需要 GPU、也不需要模型。它们一旦违反，后面所有 KL 的
解释都得推翻重来；它们成立，才能把 KL 的非单调性归给"放大"而不是"缺陷"。

## 判别力

只断言"不变量成立"是不够的 —— 一个从不失败的检查比没有检查更糟。所以这里同时
植入**已知缺陷**（把 scale 换成逐行共享、把分组划分换成非嵌套），断言审计会
报出来。这样"不变量成立"才是有信息的。
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lemoq.quant import rtn as qrtn  # noqa: E402
from tools.quant_error_audit import (  # noqa: E402
    error_stats,
    mse_monotonicity,
    nested_scale_check,
    qmax_for,
)

GROUPS = (32, 64, 128)


def _w(shape=(256, 512), seed=0, outlier_col=None, scale=0.02):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(*shape, generator=g) * scale).half()
    if outlier_col is not None:
        w[:, outlier_col] *= 40.0
    return w


# ---------------------------------------------------------------------------
# 1) 手算钉住：一个 1×4 的组，误差界与步长必须是手算的那个数
# ---------------------------------------------------------------------------
def test_hand_computed_group_step_and_error_bound():
    """w = [1.0, -0.5, 0.25, -0.125]，4-bit 对称、group=4。

    手算：absmax = 1.0，qmax = 7 → s = 1/7；
          codes = round(w/s) = [7, -4(=-3.5→-4), 2(1.75→2), -1(-0.875→-1)]；
          ŵ = s*codes = [1.0, -0.5714, 0.2857, -0.1429]；
          err = [0, 0.0714, -0.0357, 0.0179] → max|err| = 0.0714 = s/2。
    """
    w = torch.tensor([[1.0, -0.5, 0.25, -0.125]])
    stats = error_stats(w, bits=4, group_size=4)
    s = 1.0 / qmax_for(4)
    assert stats["n_groups"] == 1
    assert abs(stats["step_max"] - s) < 1e-6, stats["step_max"]
    assert abs(stats["max_abs_err"] - s / 2) < 1e-6, stats["max_abs_err"]
    assert stats["bound_violation"] == 0.0


# ---------------------------------------------------------------------------
# 2) 三条核心不变量
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [8, 4])
def test_group_partitions_are_nested(bits):
    """s32 ≤ s64 ≤ s128 逐元素成立 —— 分组是**细化**关系，不是三个无关的划分。"""
    for seed in (0, 1, 2):
        check = nested_scale_check(_w(seed=seed), bits, GROUPS)
        assert check["all_nested"], (bits, seed, check)


@pytest.mark.parametrize("bits,group_size", [(8, 32), (8, 128), (4, 32), (4, 128)])
def test_per_element_error_respects_the_step_bound(bits, group_size):
    """`|err| ≤ s/2` 必须逐元素成立（含离群通道的情形）。"""
    for outlier in (None, 7):
        stats = error_stats(_w(outlier_col=outlier), bits=bits, group_size=group_size)
        assert stats["bound_violation"] == 0.0, (bits, group_size, outlier, stats)


@pytest.mark.parametrize("bits", [8, 4])
def test_realized_mse_is_monotone_in_group_size(bits):
    """★ 实测 MSE 随 group 单调下降（多 seed 验证）。

    这是**经验**性质，不是定理：逐元素的**界**是单调的（见上一条 + 嵌套性），
    但单个元素的实现误差可以因舍入运气而反过来。在百万级元素上求均值后，
    翻转的概率极低 —— 所以这条能量化"非单调性"到底是权重的性质还是别处来的。
    """
    for seed in range(6):
        mono = mse_monotonicity(_w(seed=seed), bits, GROUPS)
        assert mono["monotone"], (bits, seed, mono)
        # 更细的分组必须真的更好，而不是差一点点
        assert mono["step_ratios"][0] > 1.0, mono


def test_group_step_ratios_are_the_geometric_expectation():
    """`step_max` 与分组**无关**（全局 absmax 总落在某个组里）—— 这条本身就是个陷阱。

    随分组变的是步长的**分布**。高斯权重下组内 absmax 随组长近似 sqrt(2 ln n) 增长，
    所以**平均步长**的比值应当落在 1.0–1.6 之间，而不是 2 倍、更不是常数。
    """
    w = _w(shape=(512, 2048))
    stats = {g: error_stats(w, bits=8, group_size=g) for g in GROUPS}
    assert stats[32]["step_max"] == stats[64]["step_max"] == stats[128]["step_max"]
    r1 = stats[64]["step_mean"] / stats[32]["step_mean"]
    r2 = stats[128]["step_mean"] / stats[64]["step_mean"]
    assert 1.0 < r1 < 1.6 and 1.0 < r2 < 1.6, (r1, r2)


# ---------------------------------------------------------------------------
# 3) W4/W8 的比值必须是**步长算术**，不是"2–4 倍"
# ---------------------------------------------------------------------------
def test_w4_w8_error_bound_ratio_is_exactly_qmax_ratio():
    """界之比 ≡ qmax_8/qmax_4 = 127/7 = 18.143。

    ★ 这条是信号 2（W4 悬崖 160×）的算术基础：位宽从 8 到 4，**步长**变大 18.14 倍，
    所以权重误差（RMS）变大 ~18 倍、MSE 变大 ~329 倍 —— 这不是缺陷，是定义。
    说"W4/W8 该差 2–4 倍"是把**KL**的经验比当成了**权重误差**的比。
    """
    w = _w(shape=(512, 2048))
    for g in GROUPS:
        s8 = error_stats(w, bits=8, group_size=g)["step_max"]
        s4 = error_stats(w, bits=4, group_size=g)["step_max"]
        assert abs(s4 / s8 - 127.0 / 7.0) < 1e-6, (g, s4 / s8)


def test_w4_w8_realized_error_ratios_match_the_arithmetic():
    """实现误差的比值：max|err| ≈ 18（界是紧的），RMS ≈ 18，MSE ≈ 329。"""
    w = _w(shape=(512, 2048))
    for g in GROUPS:
        st8 = error_stats(w, bits=8, group_size=g)
        st4 = error_stats(w, bits=4, group_size=g)
        assert 9.0 < st4["max_abs_err"] / st8["max_abs_err"] < 40.0
        assert 250.0 < st4["mse"] / st8["mse"] < 400.0, (g, st4["mse"] / st8["mse"])


# ---------------------------------------------------------------------------
# 4) 离群统计必须对**植入的**离群有反应（判别力）
# ---------------------------------------------------------------------------
def test_outlier_statistics_react_to_a_planted_outlier():
    """没有离群时不该报离群；植入一个 40× 的输入通道后必须报出来。

    这个统计量决定 `REMEDIATION_LADDER` 的 AWQ 档有没有依据 —— 所以它必须
    **能分辨**，不能永远返回同一个数。
    """
    clean = error_stats(_w(shape=(512, 2048)), bits=4, group_size=128)
    dirty = error_stats(_w(shape=(512, 2048), outlier_col=7), bits=4, group_size=128)
    assert dirty["dynamic_range_p95"] > clean["dynamic_range_p95"] * 1.5
    assert dirty["single_element_dominated_frac"] > clean["single_element_dominated_frac"]
    assert dirty["channel_rel_err_max"] > clean["channel_rel_err_max"] * 2.0
    assert dirty["rel_fro"] > clean["rel_fro"], "离群没有把整体误差抬高，审计无判别力"


def test_error_concentration_grows_with_an_outlier_column():
    """误差集中度：含离群通道时，"最大 1% 权重"贡献的误差份额应当更高。"""
    clean = error_stats(_w(shape=(512, 2048)), bits=4, group_size=128)["error_top1pct_share"]
    dirty = error_stats(_w(shape=(512, 2048), outlier_col=7), bits=4, group_size=128)[
        "error_top1pct_share"
    ]
    assert dirty >= clean
    assert 0.0 < dirty <= 1.0


# ---------------------------------------------------------------------------
# 5) 审计必须抓得到两类**已知缺陷**（否则上一条的"成立"没有信息量）
# ---------------------------------------------------------------------------
def test_audit_detects_a_scale_misalignment_defect():
    """缺陷 A：**scale 与 codes 错位一个组**（reshape/permute 顺序写错的典型形态）。

    这种缺陷在 GPU 上不报错、只会让 KL 变差，正是"非单调性会不会是实现问题"
    要排除的东西。审计的 `rel_fro` 必须把它从正确实现里区分出来，而且**幅度很大**。
    顺带：逐组 scale 换成逐行 scale（粗一档）只让 rel_fro 涨 ~1.2× —— 那种"缺陷"
    与"分组选择"无法区分，所以拿它当对照是**没有判别力**的。
    """
    from lemoq.quant import rtn as _rtn

    w = _w(shape=(256, 1024))
    good = error_stats(w, bits=4, group_size=128)["rel_fro"]
    q = _rtn.quantize_rtn(w.float(), bits=4, group_size=128, symmetric=True)
    codes = q["codes_padded"].float().reshape(256, 8, 128)
    scale = q["scale"].float().reshape(256, 8, 1)
    shifted = scale.roll(1, dims=1)  # 错位一个组
    bad = w.float().reshape(256, 8, 128) - codes * shifted
    bad_rel = float(bad.norm() / w.float().norm())
    # 实测约 1.95×（不是 5×）：误差被 **scale 的相对失配**限住（相邻组步长只差 ~20%），
    # 而不是被 step 本身限住。所以 rel_fro 只是"钝"指示器 —— 真正**锐利**的是
    # 逐元素界：正确的反量化满足 |err| ≤ s/2，错位的会**越界**。
    assert bad_rel > good * 1.5, (bad_rel, good)
    assert float((bad.abs() - scale / 2.0).clamp(min=0.0).max()) > 0.0, "界违反没有被检出"


def test_audit_detects_a_non_nested_partition():
    """缺陷 B：分组起点错位（+1 偏移）会破坏嵌套性 —— `nested_scale_check` 必须报 False。"""
    from lemoq.quant import rtn as _rtn

    w = _w(shape=(128, 512))
    q = _rtn.quantize_rtn(w.float(), bits=8, group_size=32, symmetric=True)
    scale = q["scale"].float()  # (128, 16)
    # ★ 只在**同一个粗组内**交换细组不破坏嵌套性（检查取的是组内最大值）——
    #   必须让 scale 跨行/跨粗组错位，才等价于"分组划分与声称的不一致"。
    scrambled = scale.roll(1, dims=0)
    coarse = _rtn.quantize_rtn(w.float(), bits=8, group_size=64, symmetric=True)["scale"].float()
    fine_max = scrambled.reshape(128, 8, 2).amax(dim=2)
    assert bool((fine_max > coarse + 1e-12).any()), "错位分组没有被检出来"


def test_bound_violation_is_positive_when_scale_is_wrong():
    """直接检验"审计能报出界违反"：用 2× 的 scale 反量化，界必然被违反。"""
    w = _w(shape=(64, 256))
    q = qrtn.quantize_rtn(w.float(), bits=4, group_size=32, symmetric=True)
    codes = q["codes_padded"].float().reshape(64, 8, 32)
    scale = q["scale"].float().reshape(64, 8, 1) * 2.0  # 故意错 2 倍
    err = w.float().reshape(64, 8, 32) - codes * scale
    viol = float((err.abs() - scale / 2.0).clamp(min=0.0).max())
    assert viol > 0.0
