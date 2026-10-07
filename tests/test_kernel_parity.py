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
    """★ `EXP-p2-01` 的 **K1 门**：真 Triton fused kernel vs separate 参考实现。

    两侧吃**同一份**量化结果（packed codes + scale），所以差异只能来自"读取时解压"
    的实现，不能来自量化本身。容差是 `moe_grouped.PARITY_GATE`（atol/rtol 1e-3，
    slack 1.5）—— **不得为了通过而放宽**；放宽必须写明理由并进实验记录。

    ★ 部署口径：`QuantLinear` 按字节账把 scale 存成 **fp16**，所以主用例用 fp16 scale；
      另有一组 fp32 scale 用来排除"scale 舍入"这一项（参考实现与 kernel 同 dtype）。

    覆盖：8-bit / 4-bit（nibble 寄存器展开是独立代码路径）、m=1（decode）与 m=4
    （kernel 的 M 行循环）、k 不被 group_size 整除（尾部补零由 padded_in 的 stride 表达）。
    """
    import torch

    from lemoq.kernels._dequant_triton import dequant_gemm_fused
    from lemoq.kernels.moe_grouped import PARITY_GATE
    from lemoq.quant import pack as qpack

    def _one(m, k, n, bits, group_size, *, scale_dtype=torch.float16, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(m, k, device="cuda", dtype=torch.float16) * 0.5
        w = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.05
        q = qrtn.quantize_rtn(w, bits=bits, group_size=group_size)
        codes = q["codes"]
        # ★ `codes_padded` 是 (out, n_groups, group_size) 的**分组**形状（rtn 的约定），
        #   打包要的是展平后的 (out, padded_in)。
        capped = q["codes_padded"].reshape(n, -1)
        scale, zero = q["scale"].to(scale_dtype), q["zero"].to(scale_dtype)
        # 打包：与 `QuantLinear` 完全同一条路径（8-bit 直接 int8；4-bit 两个一组进 uint8）
        packed = capped.to(torch.int8) if bits == 8 else qpack.pack_4bit(capped)
        sym = bool(q["symmetric"])
        ref = dequant_gemm_separate_reference(
            x, codes, scale, zero, group_size=group_size, out_dtype=torch.float32
        )
        got = dequant_gemm_fused(
            x,
            packed,
            scale,
            None if sym else zero,
            group_size=group_size,
            bits=bits,
            symmetric=sym,
            out_dtype=torch.float32,
        )
        rep = tolerance_report(ref, got)
        assert rep["allclose_atol_1e-3"], {
            "case": (m, k, n, bits, group_size, str(scale_dtype)),
            "report": rep,
            "gate": PARITY_GATE,
        }
        # 打包必须无损（否则上面比较的是两组不同的数）：解包回来后逐位等于 padded codes。
        # 对称量化的 nibble 是有符号的（-8..7）：pack 取低 4 位、unpack 做符号扩展，
        # 与 `dequantize_rtn` 用的是同一组 code 值。
        if bits == 4:
            assert torch.equal(qpack.unpack_4bit(packed, signed=True).to(torch.int32), capped)
        else:
            assert torch.equal(packed.to(torch.int32), capped)
        return rep

    # decode 的最常见形状：m=1、k=2048、n=2048（OLMoE 的 q_proj），g=128
    _one(1, 2048, 2048, 8, 128)
    _one(1, 2048, 2048, 4, 128)
    # m>1：kernel 的 M 行循环（生产路径不用 M>1，但 K1 要交叉验证）
    _one(4, 256, 512, 8, 128)
    _one(4, 256, 512, 4, 128)
    # fp32 scale：与参考实现完全同 dtype，排除 scale 舍入
    _one(1, 512, 256, 8, 128, scale_dtype=torch.float32)
    # k 不被 group_size 整除：参考实现补零，kernel 用 padded_in 的 stride 表达同一件事
    _one(1, 500, 256, 8, 128)
    _one(1, 500, 256, 4, 128)
    # 非对称（有 zero）不在 K1 里测：部署路径（QuantLinear + SchemeSpec）只用对称量化
    # （`symmetric=True` 是默认），而非对称 code 的按位存储在大位宽下另有约定。
    # 另一个 group_size：只验证 stride 算术（门的配置是 g=128，部署路径只用 128）
    _one(1, 1024, 512, 8, 64)


@pytest.mark.gpu
@requires_cuda
def test_triton_quantized_grouped_matches_oracle():
    """★ P1 的 K1 门：**量化 grouped kernel** vs 同权重下的 fp32 oracle。

    这里比的是"kernel 在寄存器里解包（code*scale）"与"先把权重量化再反量化、
    用 fp32 做同一件事"——两者的数学完全相同，所以差异只能来自 fp32 累加顺序。

    `EXP-p4b-01-truncation` §3 的 P1；容差用 `moe_grouped.PARITY_GATE`
    （1e-3 / 1e-3），不为通过而放宽。
    """
    from lemoq.kernels.moe_grouped_quant import (
        grouped_moe_forward_quant,
        quantize_fused_experts,
    )
    from lemoq.kernels.moe_static import FusedExperts

    torch.manual_seed(0)
    n_experts, ffn, hidden = 4, 256, 512
    fused = FusedExperts(
        gate_up=torch.randn(n_experts, 2 * ffn, hidden) * 0.05,
        down=torch.randn(n_experts, hidden, ffn) * 0.05,
    )
    x = torch.randn(hidden, dtype=torch.float16, device="cuda") * 0.5
    ids = torch.tensor([0, 2], dtype=torch.int64, device="cuda")
    weights = torch.tensor([0.7, 0.3], dtype=torch.float16, device="cuda")

    # down 的 ffn=256：默认 SPLIT_K=4 要求 256 % (4*128) == 0（不成立），
    # 所以显式用 SPLIT_K=2 —— 顺便覆盖 out_reduce kernel。
    config = {"down_sum": {"SPLIT_K": 2}}

    # ★ 8-bit 与 4-bit 都要过：4-bit 走的是**另一条代码路径**（`_expand_int4` 的
    #   nibble 展开），K1 门对它同样不放宽。
    for bits in (8, 4):
        quantized = quantize_fused_experts(fused, bits=bits, group_size=128).to("cuda")
        got = grouped_moe_forward_quant(x, quantized, ids, weights, config=config, backend="triton")
        ref = grouped_moe_forward_quant(x, quantized, ids, weights, config=config, backend="oracle")
        report = tolerance_report(ref.float(), got.float())
        assert report["allclose_atol_1e-3"], {"bits": bits, "report": report}


@pytest.mark.gpu
@requires_cuda
def test_fused_saves_bytes_versus_separate():
    """E2 的机制验证：separate 物化中间张量，fused 不物化。

    上机后用实测的 DRAM 流量差值确认这一点（ncu 的 dram__bytes.sum）。
    """
    pytest.skip("需要 ncu 实测 DRAM 流量（按计划留到有 GPU 时做）")
