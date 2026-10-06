"""test_matrix — `configs/experiments.yaml` 的展开与 arch 门控。

这是"蹲到 4090 还是 5090 不用改代码"这个说法的**回归测试**。
"""

from __future__ import annotations

import os

import pytest
from conftest import requires_yaml  # noqa: E402

from benchmarks.harness import arms_for_arch, expand, load_matrix

pytestmark = requires_yaml

MATRIX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "experiments.yaml",
)


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(MATRIX)


def test_arch_arms_come_from_verified_llvm_predicates(matrix):
    assert set(arms_for_arch(matrix, "sm_120")) == {
        "cp_async",
        "tma",
        "fp8",
        "mx_block_scale",
        "bf16",
    }
    # 4090: 有 cp.async 与 fp8，没有 tma / mx_block_scale
    assert set(arms_for_arch(matrix, "sm_89")) == {"cp_async", "fp8", "bf16"}
    # A6000 (Ampere)：没有 fp8
    assert set(arms_for_arch(matrix, "sm_86")) == {"cp_async", "bf16"}
    # V100：什么都没有
    assert arms_for_arch(matrix, "sm_70") == []


def test_e7_truncation_matrix_is_complete(matrix):
    """E7 必须是 {fp16,w8a8,w4a16} x {eager,cudagraph} x ctx{1024,4096} = 12 格。"""
    runs = expand(matrix, arch="sm_120")
    e7 = [
        r
        for r in runs
        if r.base_id == "EXP-p4b-01-truncation" and not r.exp_id.endswith("__SKIPPED-arch")
    ]
    assert len(e7) == 12
    bit_widths = {r.config.get("bit_width") for r in e7}
    executions = {r.config.get("execution") for r in e7}
    assert bit_widths == {"fp16", "w8a8", "w4a16"}
    assert executions == {"eager", "cudagraph"}


def test_e7_is_marked_combined(matrix):
    """E7 同时改 bit-width 与 execution —— AGENTS.md 规则 3 要求显式标注。"""
    e7 = next(e for e in matrix["experiments"] if e["id"] == "EXP-p4b-01-truncation")
    assert e7.get("combined") is True


def test_arch_gating_skips_mxfp4_on_sm89(matrix):
    runs = expand(matrix, arch="sm_89")
    skipped = [r for r in runs if r.exp_id.endswith("__SKIPPED-arch")]
    assert any("mxfp4" in r.base_id for r in skipped)
    # sm_89 保留 fp8 臂
    assert not any(r.base_id == "EXP-p3-01-fp8-w8a8" for r in skipped)


def test_arch_gating_skips_fp8_and_mxfp4_on_sm86(matrix):
    runs = expand(matrix, arch="sm_86")
    skipped_ids = {r.base_id for r in runs if r.exp_id.endswith("__SKIPPED-arch")}
    assert "EXP-p3-01-fp8-w8a8" in skipped_ids
    assert "EXP-p3-02-mxfp4-block-scale" in skipped_ids


def test_arch_gating_skips_cpasync_work_on_v100(matrix):
    runs = expand(matrix, arch="sm_70")
    skipped_ids = {r.base_id for r in runs if r.exp_id.endswith("__SKIPPED-arch")}
    assert "EXP-p2-01-dequant-fusion" in skipped_ids


def test_sm120_expands_to_expected_total(matrix):
    runs = expand(matrix, arch="sm_120")
    real = [r for r in runs if not r.exp_id.endswith("__SKIPPED-arch")]
    # 12 (E7) + 9 (p1) + 5 (p2) + 1 (fp8) + 2 (mxfp4) + 5 (p0) + 1+1+3 (p4/p5) + 1 (p8)
    # p0 从 3 变成 4：新增 EXP-p0-04（固定形状 MoE 的 CUDA Graph 门）——
    # 它与 p0-03 是**不同假设**，不是同一次实验的另一个臂，所以各占一个 run。
    # p0 再从 4 变成 5：p0-04 的 ctx 轴从 [1024] 扩到 [1024, 4096]。
    # 这不是隐藏的变量变更 —— ctx 是声明的轴（同 p0-02 的 [1024, 4096]），
    # 而且 4096 是 attention 与 kernel 数地板的判别点。
    # p2 从 4 变成 5：新增 EXP-p2-02（grouped GEMM 去 gather）。它排在 p2-01
    # **之前**，因为 dequant 要融进的结构就是它。
    # p2 从 5 变成 7：EXP-p2-02（grouped，1）+ EXP-p2-04（eager attention，ctx 轴 2 个）
    assert len(real) == 43


def test_exp_ids_are_unique(matrix):
    runs = expand(matrix, arch="sm_120")
    ids = [r.exp_id for r in runs]
    assert len(ids) == len(set(ids)), "展开出了重复的 EXP-ID"


def test_scope_minimal_credible_is_a_subset(matrix):
    full = {r.exp_id for r in expand(matrix, arch="sm_120")}
    minimal = {r.exp_id for r in expand(matrix, arch="sm_120", scope="minimal_credible")}
    assert minimal < full
    assert not any("p3-" in e for e in minimal)


def test_unknown_scope_raises(matrix):
    with pytest.raises(KeyError):
        expand(matrix, arch="sm_120", scope="nope")


def test_every_experiment_has_hypothesis_and_refutation(matrix):
    """EXPERIMENTS.md §3：假设必须在跑之前写下。矩阵里不许有空假设。"""
    for e in matrix["experiments"]:
        assert e.get("hypothesis"), "实验 %s 缺少 hypothesis" % e["id"]
        assert e.get("refutation_signal"), "实验 %s 缺少 refutation_signal" % e["id"]


def test_olmoe_ctx_never_exceeds_model_limit(matrix):
    """OLMoE 的 max_position_embeddings = 4096 —— 矩阵里不得出现更大的扫描点。"""
    for e in matrix["experiments"]:
        limit = e.get("ctx_hard_limit")
        ctxs = e.get("ctx")
        if limit and isinstance(ctxs, list):
            assert max(ctxs) <= limit, "%s 的 ctx 扫描超过 %d" % (e["id"], limit)


def test_model_ids_have_required_0924_suffix():
    """防止 `allenai/OLMoE-1B-7B`（不存在的仓库）再次溜回来。"""
    raw = open(MATRIX, encoding="utf-8").read()
    assert "allenai/OLMoE-1B-7B-0924" in raw
    assert 'allenai/OLMoE-1B-7B"' not in raw


def test_p0_cudagraph_gate_is_an_independent_ctx_1024_run(matrix):
    runs = [r for r in expand(matrix, arch="sm_120") if r.base_id == "EXP-p0-03-baseline-cudagraph"]
    assert len(runs) == 1
    assert runs[0].exp_id == "EXP-p0-03-baseline-cudagraph__ctx=1024"
    assert runs[0].config == {"execution": "cudagraph", "bit_width": "fp16", "ctx": 1024}
