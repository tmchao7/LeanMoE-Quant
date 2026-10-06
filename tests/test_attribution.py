"""test_attribution — 归因算术与判据的离线回归（**不需要 GPU，也不产生可引用数字**）。

为什么这些断言值得写：

    `EXP-p0-04` 的残差（1.59 ms）决定了下一个 GPU 小时花在哪 —— grouped GEMM
    还是融合小 kernel。而做出这个决定的算术全部是**纯 Python**，所以可以在这里
    用手算值钉住。单位错（µs↔ms）、口径错（每层↔每 token、总计↔每个调用）
    在 GPU 上只会表现为"数字看起来不太对"，没人能发现；
    在这里会当场红。

    上一轮的教训：prefill 的 64 GiB 与 128× 重复读，小模型 parity 测试一个都没抓到，
    最后是靠**纯算术断言**抓到的（见 `tests/test_moe_static.py`）。
"""

from __future__ import annotations

import pytest
from conftest import requires_torch

from benchmarks.attribution import (
    HYPO_THRESHOLDS,
    expert_path_accounting,
    group_kernels,
    resolve_hypothesis,
)
from benchmarks.bytes import ModelSpec, byte_budget, weight_budget

# 实测带宽（results/machine_probe.json，RTX 5090 / driver 595.71.05）。
# ★ 这是**实测值**，不是 datasheet 的 1792 —— AGENTS.md 硬规则 4。
BW = 1559.2

# OLMoE-1B-7B-0924 的**已核实** config（docs/MODELS.md §3.1）。
# ★ `tie_word_embeddings: False` 不是细节：untied 的 lm_head 占 [W] 的 8.7%。
#   把它写成 True 会让专家占比从 68.3% 变成 74.9% —— 这次就是被下面那条
#   交叉核对测试抓出来的。所以这里走 `from_hf_config`，与 runner 同一条路径。
VERIFIED_OLMOE = {
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "num_hidden_layers": 16,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "intermediate_size": 1024,
    "vocab_size": 50304,
    "tie_word_embeddings": False,
    "max_position_embeddings": 4096,
}


def olmoe_spec() -> ModelSpec:
    """The **verified** OLMoE-1B-7B-0924 shape (docs/MODELS.md §3.1)."""
    return ModelSpec.from_hf_config(VERIFIED_OLMOE, name="allenai/OLMoE-1B-7B-0924")


def test_spec_matches_the_conftest_verified_config(olmoe_config):
    """本文件的 spec 必须与 `tests/conftest.py` 的已核实 config 一致。

    ★ 为什么值得单独一条测试：整份归因算术都建立在这些字段上，
    而 `tie_word_embeddings` 一个人就能让专家占比偏 6.6 个百分点。
    """
    assert VERIFIED_OLMOE == olmoe_config


# ---------------------------------------------------------------------------
# 专家路径的字节账
# ---------------------------------------------------------------------------
def test_expert_path_bytes_match_the_p0_04_appendix():
    """p0-04 记录附录里的每个数字都必须能被重算出来。

    如果这些数字变了，说明模型形状或 dtype 变了 —— 那时**结论**也要重算，
    不能只改注释。
    """
    acc = expert_path_accounting(olmoe_spec(), bw_gbps=BW)
    per_layer = acc["per_layer"]
    assert per_layer["gate_up_bytes"] == 67_108_864  # (8, 2048, 2048) fp16
    assert per_layer["down_bytes"] == 33_554_432  # (8, 1024, 2048) fp16
    assert per_layer["ideal_bytes"] == 100_663_296

    per_token = acc["per_token"]
    assert per_token["ideal_bytes"] == 1_610_612_736
    assert per_token["tax_bytes"] == 3_221_225_472
    assert per_token["driven_bytes"] == 4_831_838_208


def test_gather_tax_is_exactly_twice_the_ideal():
    """gather 路径的机制：读一次 + 写一份副本 + bmm 再读副本 = 理想量的 3 倍。

    所以税 = 2 × 理想量。这条要成立，否则 gather 的实现假设被改过。
    """
    acc = expert_path_accounting(olmoe_spec(), bw_gbps=BW)
    per_token = acc["per_token"]
    assert per_token["driven_bytes"] == 3 * per_token["ideal_bytes"]
    assert per_token["tax_bytes"] == 2 * per_token["ideal_bytes"]
    assert (
        per_token["gather_read_bytes"]
        == per_token["gather_write_bytes"]
        == per_token["gather_reread_bytes"]
        == per_token["ideal_bytes"]
    )


def test_expert_path_ms_at_measured_bandwidth():
    """2.066 ms / 3.099 ms —— p0-04 记录的 2.07 ms tax 与 3.099 ms 下界。"""
    acc = expert_path_accounting(olmoe_spec(), bw_gbps=BW)
    assert acc["ms"]["tax"] == pytest.approx(2.065953, abs=1e-5)
    assert acc["ms"]["driven"] == pytest.approx(3.098928, abs=1e-5)
    assert acc["ms"]["ideal"] == pytest.approx(1.032971, abs=1e-5)


def test_four_groups_sum_to_the_driven_total():
    """四组（两个 gather + 两个 bmm）的下界之和必须等于 driven 总量 —— 没有漏项。"""
    acc = expert_path_accounting(olmoe_spec(), bw_gbps=BW)
    total = sum(g["floor_ms_per_token"] for g in acc["four_groups"])
    assert total == pytest.approx(acc["ms"]["driven"], abs=1e-6)
    by_name = {g["name"]: g["floor_ms_per_token"] for g in acc["four_groups"]}
    assert by_name["gate_up_gather"] == pytest.approx(1.377304, abs=1e-5)
    assert by_name["gate_up_bmm"] == pytest.approx(0.688652, abs=1e-5)
    assert by_name["down_gather"] == pytest.approx(0.688652, abs=1e-5)
    assert by_name["down_bmm"] == pytest.approx(0.344326, abs=1e-5)


def test_expert_share_matches_the_documented_68_percent():
    """独立交叉核对：专家路径的理想量应当 ≈ [W] 的 68.3%（docs/METRICS.md §3.2）。

    这条把 `attribution.py` 与 `bytes.py` 两个模块钉在一起 —— 如果哪天
    `[W]` 的口径变了，这里会断，而不是让两份文档各说各话。
    """
    spec = olmoe_spec()
    acc = expert_path_accounting(spec, bw_gbps=BW)
    w_bytes = weight_budget(spec)["total_bytes"]
    share = 100.0 * acc["per_token"]["ideal_bytes"] / w_bytes
    assert share == pytest.approx(68.3, abs=0.1)
    assert byte_budget(spec, ctx=1024)["W"]["total_bytes"] == w_bytes


def test_per_token_accounting_scales_with_layers():
    """层数翻倍 → 字节与时间翻倍（防止把 per-layer 当成 per-token）。"""
    a = expert_path_accounting(olmoe_spec(), bw_gbps=BW, layers=8)
    b = expert_path_accounting(olmoe_spec(), bw_gbps=BW, layers=16)
    assert b["per_token"]["tax_bytes"] == 2 * a["per_token"]["tax_bytes"]
    assert b["ms"]["tax"] == pytest.approx(2 * a["ms"]["tax"], abs=1e-6)


def test_accounting_rejects_spec_bandwidth_and_empty_spec():
    """★ 必须拒绝：用规格书带宽算 roofline，或 spec 缺字段。

    `AGENTS.md` 硬规则 4：任何理论下界都必须引用**实测**带宽。
    这里让"忘了传带宽"变成异常，而不是变成 1792。
    """
    with pytest.raises(ValueError):
        expert_path_accounting(olmoe_spec(), bw_gbps=0)
    with pytest.raises(ValueError):
        expert_path_accounting(olmoe_spec(), bw_gbps=-1.0)
    with pytest.raises(ValueError):
        expert_path_accounting({"layers": 16}, bw_gbps=BW)


# ---------------------------------------------------------------------------
# 实测 kernel 表 → 分组 → 判据
# ---------------------------------------------------------------------------
def _row(name: str, ms: float, occ: float = 1.0) -> dict:
    return {
        "name": name,
        "self_ms_per_token": ms,
        "occurrences_per_token": occ,
        "mean_us_per_occurrence": ms / occ * 1000.0 if occ else None,
    }


def test_group_kernels_uses_per_token_contribution_not_mean():
    """分组累加必须用 `self_ms_per_token`（每 token 贡献），不是单次均值。

    OLMoE 一层里 attention 相关 kernel 会出现多次；用均值会把总量低估
    几十倍，而低估恰恰会把判决推向"残差是小 kernel 地板"。
    """
    rows = [
        _row("indexFuncSmallIndexKernel", 0.02, 16.0),  # 每层一次 × 16 层
        _row("MULTI_LEVEL_TREE", 0.30, 16.0),
    ]
    groups = group_kernels(rows)
    assert groups["by_pattern"]["gather"]["self_ms_per_token"] == pytest.approx(0.02, abs=1e-6)
    assert groups["sums"]["pattern_total_ms_per_token"] == pytest.approx(0.32, abs=1e-6)


def test_group_kernels_flags_unmatched_instead_of_guessing():
    """匹配不中必须报出来，不能默默归到某个组里。"""
    rows = [_row("my_total_mystery_kernel_xyz", 0.5)]
    groups = group_kernels(rows)
    assert groups["by_pattern"]["unmatched"]["self_ms_per_token"] == pytest.approx(0.5)
    assert any("mystery" in n for n in groups["unmatched_names"])


def test_latency_floor_bucket_uses_the_frozen_threshold():
    """均值 < 4 µs 的 kernel 进地板桶；>= 4 µs 的进另一桶。两桶之和 = 总量。"""
    rows = [
        _row("elementwise_kernel_a", 0.8, 400.0),  # 2 µs → 地板
        _row("elementwise_kernel_b", 0.7, 1.0),  # 700 µs → 不是地板
    ]
    groups = group_kernels(rows)
    assert groups["latency_floor"]["threshold_us"] == HYPO_THRESHOLDS["latency_floor_us"]
    assert groups["latency_floor"]["self_ms_per_token"] == pytest.approx(0.8)
    assert groups["latency_floor"]["n_kernels"] == 1
    assert groups["above_floor_ms_per_token"] == pytest.approx(0.7, abs=1e-6)


def test_resolve_hypothesis_ha_when_the_moe_group_dominates():
    """H-A：MoE 四组之和 ≥ 1.6 ms → 下一步是 grouped GEMM。"""
    rows = [
        _row("indexFuncSmallIndexKernel", 1.38, 16.0),
        _row("ampere_sgemm_128x64", 0.69, 32.0),
        _row("elementwise_kernel_b", 0.30, 500.0),
    ]
    out = resolve_hypothesis(rows)
    assert out["verdict"] == "H-A"
    assert out["dram_group_ms_per_token"] == pytest.approx(2.07, abs=1e-6)
    assert "grouped GEMM" in out["action"]


def test_resolve_hypothesis_hb_when_the_latency_floor_dominates():
    """H-B：小 kernel 之和 ≥ 1.0 ms 且 MoE 四组 < 0.5 ms → 主攻融合/减 kernel。"""
    rows = [_row("elementwise_kernel_%d" % i, 0.06, 300.0) for i in range(20)]
    out = resolve_hypothesis(rows)
    assert out["verdict"] == "H-B"
    assert out["latency_bucket_ms_per_token"] == pytest.approx(1.2, abs=1e-6)
    assert "融合" in out["action"]


def test_resolve_hypothesis_reports_both_and_neither():
    """两条假设**相互独立**，同时成立要报 H-A+H-B；都不成立要报 neither。

    ★ 关键：`neither` 时**不得**默认去写 grouped GEMM —— 那是把"判据没成立"
    当成"假设成立"来用。`dominant` 只决定先做哪个，不改变 verdict。
    """
    both = [_row("indexFuncSmallIndexKernel", 1.7, 16.0)] + [
        _row("elementwise_kernel_%d" % i, 0.05, 300.0) for i in range(20)
    ]
    out = resolve_hypothesis(both)
    assert out["verdict"] == "H-A+H-B"
    assert out["dominant"] == "H-A"  # 1.7 ms > 1.0 ms，先做 grouped GEMM
    assert out["action"].startswith("Stage C 与 D-2 都要做")

    neither = [_row("ampere_sgemm_128x64", 0.2, 1.0)]
    out = resolve_hypothesis(neither)
    assert out["verdict"] == "neither"
    assert out["dominant"] == "neither"
    assert "否证" in out["action"]


def test_thresholds_are_frozen_constants_not_arguments():
    """判据必须是模块级常量 —— 不允许在调用点临时给一个更好看的阈值。"""
    assert HYPO_THRESHOLDS["dram_group_ms_min"] == 1.6
    assert HYPO_THRESHOLDS["latency_bucket_ms_min"] == 1.0
    assert HYPO_THRESHOLDS["latency_floor_us"] == 4.0
    # 覆盖只能显式传入，且默认值不受影响（防止某人用 thresholds= 把门槛调低后不记录）
    relaxed = resolve_hypothesis(
        [_row("ampere_sgemm_128x64", 0.2, 1.0)], thresholds={"dram_group_ms_min": 0.1}
    )
    assert relaxed["verdict"] == "H-A"
    assert HYPO_THRESHOLDS["dram_group_ms_min"] == 1.6


@requires_torch
def test_kernel_table_units_and_ordering():
    """`kernel_table` 的 µs→ms 与"每 token"换算必须正确，并按贡献降序。

    这是纯函数（不依赖 torch），放在这里是因为它决定 GPU 预算去向，
    所以必须能用手算值钉住。
    """
    from benchmarks.cudagraph import kernel_table

    rows = [
        {"name": "big", "count": 40, "self_us_total": 4000.0},  # 40 次 / 20 次调用 = 2 次/token
        {"name": "small", "count": 20, "self_us_total": 200.0},
        {"name": "zero", "count": 0, "self_us_total": 0.0},
    ]
    table = kernel_table(rows, timed=20, busy_ms=4.2)
    assert [r["name"] for r in table] == ["big", "small"]
    big = table[0]
    assert big["self_ms_per_token"] == pytest.approx(0.2, abs=1e-9)  # 4000 µs / 20 / 1000
    assert big["occurrences_per_token"] == pytest.approx(2.0)
    assert big["mean_us_per_occurrence"] == pytest.approx(100.0)  # 4000 / 40
    assert big["share_pct"] == pytest.approx(100 * 0.2 / 4.2, abs=0.01)
    # 每 token 贡献之和 = busy（这就是这张表能解释时间的原因）
    assert sum(r["self_ms_per_token"] for r in table) == pytest.approx(0.21, abs=1e-9)
