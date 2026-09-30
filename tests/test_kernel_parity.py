"""test_kernel_parity — 真 kernel 与参考实现的数值对照（**骨架，需 GPU**）。

为什么先写好骨架：

    Triton/CUDA kernel 的错误只有在真机上才能发现。把对照测试**提前写好**，
    上机后填进真 kernel 就能立刻得到 PASS/FAIL，而不是现写测试再调试。

当前状态：只测参考实现自身（fused 与 separate 的**数学等价性**），这是没有 GPU 也能做的
部分。真正的 kernel 一旦落地，把 `_triton_fused_gemm` 的桩替换即可。
"""

from __future__ import annotations

import pytest
from conftest import requires_cuda, requires_torch

pytestmark = requires_torch

import torch  # noqa: E402

from lemoq.kernels.reference import (  # noqa: E402
    dequant_gemm_fused_reference,
    dequant_gemm_separate_reference,
    tolerance_report,
)
from lemoq.quant import rtn as qrtn  # noqa: E402


def _case(m=4, k=256, n=512, bits=4, group_size=128, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    x = torch.randn(m, k, generator=g)
    q = qrtn.quantize_rtn(w, bits=bits, group_size=group_size)
    return (
        x.to(device),
        q["codes"].to(device),
        q["scale"].to(device),
        q["zero"].to(device),
        group_size,
    )


def test_fused_and_separate_are_mathematically_equal():
    """两条路径的**数值必须相同** —— 差别只在是否物化中间张量（E2 的全部内容）。"""
    x, codes, scale, zero, gs = _case()
    a = dequant_gemm_separate_reference(x, codes, scale, zero, group_size=gs)
    b = dequant_gemm_fused_reference(x, codes, scale, zero, group_size=gs)
    rep = tolerance_report(a, b)
    assert rep["max_abs_err"] == 0.0


def test_reference_matches_dense_matmul_of_dequantized_weight():
    """参考实现必须等于"先反量化、再用 fp32 做普通 matmul"，否则 oracle 本身是错的。"""
    x, codes, scale, zero, gs = _case()
    y_ref = dequant_gemm_separate_reference(x, codes, scale, zero, group_size=gs)
    q = {
        "codes": codes,
        "codes_padded": codes,
        "scale": scale,
        "zero": zero,
        "bits": 4,
        "group_size": gs,
        "symmetric": True,
        "shape": tuple(codes.shape),
        "n_groups": scale.shape[1],
        "padded_in": scale.shape[1] * gs,
    }
    w_hat = qrtn.dequantize_rtn(q)
    y_dense = x.float() @ w_hat.t()
    assert torch.allclose(y_ref, y_dense, atol=1e-4, rtol=1e-4)


def test_reference_handles_non_divisible_k():
    x, codes, scale, zero, gs = _case(k=500, group_size=128)
    y = dequant_gemm_separate_reference(x, codes, scale, zero, group_size=gs)
    assert y.shape == (4, 512)


def test_tolerance_report_flags_mismatch():
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4)
    rep = tolerance_report(a, b)
    assert rep["allclose_atol_1e-3"] is False
    assert rep["max_abs_err"] == 1.0


# ---------------------------------------------------------------------------
# 以下需要 GPU —— 真 kernel 落地后把桩替换掉
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@requires_cuda
def test_triton_fused_gemm_matches_reference():
    """TODO(phase-c): 实现 Triton fused dequant-GEMM 后取消 skip。

    上机时的步骤：
        1. 在 src/lemoq/kernels/ 落地 triton_dequant_gemm.py
        2. 用与 `_case()` 相同的输入跑两侧
        3. 断言 tolerance_report(...)["allclose_atol_1e-3"] 为 True
        4. 同时记录 ncu 的 dram__throughput（见 lemoq.kernels.NCU_METRICS）

    容差不得为了通过而放宽；放宽必须写明理由并记录在实验记录里。
    """
    pytest.skip("Triton kernel 尚未实现（按计划留到有 GPU 时写）")


@pytest.mark.gpu
@requires_cuda
def test_fused_saves_bytes_versus_separate():
    """E2 的机制验证：separate 物化中间张量，fused 不物化。

    上机后用实测的 DRAM 流量差值确认这一点（ncu 的 dram__bytes.sum）。
    """
    pytest.skip("需要 ncu 实测 DRAM 流量（按计划留到有 GPU 时做）")
