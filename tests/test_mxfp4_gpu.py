"""test_mxfp4_gpu — MXFP4 fused kernel 的 K1 门（GPU；EXP-p3-02 的工程前置）。

钉住的口径：
    - kernel 的 E2M1 解码 + E8M0 尺度必须与主机侧参照（`mxfp4.dequantize_mxfp4`）
      **allclose@1e-3**（PARITY_GATE；事实上应逐位一致——解码值全是二次幂精确数）；
    - A6 合取：err_kernel ≤ 1.5 × err_baseline（双方对 fp64 oracle；权重是精确二次幂
      值 ⇒ 基线误差只含 fp32 累加顺序噪声，合取退化但不缺席——防别名半边仍在）；
    - 守卫显式拒绝：scale 非 uint8 / group_size≠32 / symmetric=False+zero。
    覆盖：m=1（decode）与 m=4；K 整除 32 与 K=50（pad 到 64、尾组真实列）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

from lemoq.kernels._dequant_triton import dequant_gemm_fused  # noqa: E402
from lemoq.quant.mxfp4 import dequantize_mxfp4, quantize_mxfp4  # noqa: E402

pytestmark = pytest.mark.gpu

TOL = {"atol": 1e-3, "rtol": 1e-3}


def _requires_gpu():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")


def _case(m, k, n, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, dtype=torch.float32) * 0.05
    x = torch.randn(m, k, dtype=torch.float16) * 0.5
    q = quantize_mxfp4(w)
    return x.cuda(), q, w


def test_triton_mxfp4_matches_host_reference():
    """K 门：fused kernel（E2M1+exp2 算术解码）vs 主机侧反量化参照。"""
    _requires_gpu()
    for m, k, n, seed in [(1, 256, 128, 0), (4, 256, 128, 1), (1, 50, 128, 2), (1, 2048, 512, 3)]:
        x, q, _ = _case(m, k, n, seed)
        w_ref = dequantize_mxfp4(q, dtype=torch.float32).cuda()  # (N, K)（pad 列已切回）
        ref = x.float() @ w_ref.T
        got = dequant_gemm_fused(
            x,
            q["codes_packed"].cuda(),
            q["scales"].cuda(),
            None,
            group_size=32,
            bits=4,
            symmetric=True,
            mxfp4=True,
        )
        assert torch.allclose(got, ref, **TOL), (m, k, n, seed)
        # ★ 逐位一致**不成立**且不该要求：单个乘积（二次幂 × 二次幂）确实精确，
        #   但 K 维**归约顺序**不同（tl.sum 树形 vs torch mm 的分块序）⇒ 末位 ULP
        #   必有差。K1 的门是 PARITY_GATE（1e-3），不是位相等——2026-10-25 实测记录。


def test_triton_mxfp4_conjunction_half_vs_fp64_oracle():
    """A6 合取：err_kernel ≤ 1.5 × err_baseline（双方对 fp64 oracle）。"""
    _requires_gpu()
    x, q, _ = _case(1, 256, 128, 7)
    # 参照数学统一留 CPU（权重是精确二次幂值），只有 kernel 输入上卡。
    w64 = dequantize_mxfp4(q).double()
    oracle = x.cpu().double() @ w64.T
    ref = x.cpu().float() @ dequantize_mxfp4(q, dtype=torch.float32).T
    got = (
        dequant_gemm_fused(
            x,
            q["codes_packed"].cuda(),
            q["scales"].cuda(),
            None,
            group_size=32,
            bits=4,
            symmetric=True,
            mxfp4=True,
        )
        .cpu()
        .double()
    )
    err_ref = float((ref.double() - oracle).abs().max())
    err_got = float((got - oracle).abs().max())
    assert err_got <= 1.5 * err_ref + 1e-9, (err_ref, err_got)


def test_wrapper_rejects_bad_mxfp4_arguments():
    """守卫显式拒绝：scale 非 uint8 / group_size≠32 / 带 zero。"""
    _requires_gpu()
    x, q, _ = _case(1, 256, 64, 9)
    with pytest.raises(ValueError, match="uint8"):
        dequant_gemm_fused(
            x,
            q["codes_packed"].cuda(),
            q["scales"].cuda().half(),
            None,
            group_size=32,
            bits=4,
            symmetric=True,
            mxfp4=True,
        )
    with pytest.raises(ValueError, match="32"):
        dequant_gemm_fused(
            x,
            q["codes_packed"].cuda(),
            q["scales"].cuda(),
            None,
            group_size=128,
            bits=4,
            symmetric=True,
            mxfp4=True,
        )
    with pytest.raises(ValueError, match="zero"):
        zero = torch.zeros_like(q["scales"].cuda())
        dequant_gemm_fused(
            x,
            q["codes_packed"].cuda(),
            q["scales"].cuda(),
            zero,
            group_size=32,
            bits=4,
            symmetric=False,
            mxfp4=True,
        )
