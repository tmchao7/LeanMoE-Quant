"""test_rmsnorm_parity — 自研融合 RMSNorm 的 K1 门（GPU）。

判据：与 **HF 的 `OlmoeRMSNorm` 数学**（在测试里逐字复刻，包括"先舍入到 fp16、再用 fp16 的
weight 乘"这一步）在 `PARITY_GATE`（atol/rtol 1e-3）内一致。**不为通过而放宽。**

★ 为什么不拿 fp64 当唯一参照：本 kernel 的语义就是 HF 那条链（含它自己的 fp16 舍入点），
  换实现对不齐 fp64 是**预期的**；fp64 距离作为信息一并报出来，不是门槛。
"""

from __future__ import annotations

import pytest
from conftest import requires_cuda

torch = pytest.importorskip("torch")

from lemoq.kernels.reference import tolerance_report  # noqa: E402


def _hf_formula(x, weight, eps):
    """逐字复刻 `transformers.models.olmoe.modeling_olmoe.OlmoeRMSNorm.forward`。"""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


@pytest.mark.gpu
@requires_cuda
def test_triton_rms_norm_matches_hf_formula():
    from lemoq.kernels.rmsnorm import fused_available, rms_norm_fused

    if not fused_available():
        pytest.skip("需要 triton + CUDA")

    torch.manual_seed(0)
    shapes = [(1, 2048), (16, 128), (1, 128), (4, 2048), (2, 3, 128)]
    for shape in shapes:
        H = shape[-1]
        x = (torch.randn(*shape, device="cuda", dtype=torch.float16) * 1.7).contiguous()
        w = (torch.randn(H, device="cuda", dtype=torch.float16) * 0.5 + 1.0).contiguous()
        eps = 1e-5
        got = rms_norm_fused(x, w, eps)
        ref = _hf_formula(x, w, eps)
        assert tuple(got.shape) == tuple(x.shape), (shape, tuple(got.shape))
        rep = tolerance_report(ref, got)
        assert rep["allclose_atol_1e-3"], {"shape": shape, "report": rep}
        # 信息项：与 fp64 理想值的距离（不作为门槛）
        ideal = (
            x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + eps)
        ) * w.double()
        fp64_gap = float((got.double() - ideal).abs().max().item())
        hf_gap = float((ref.double() - ideal).abs().max().item())
        assert abs(fp64_gap - hf_gap) < 1e-3, (shape, fp64_gap, hf_gap)
        # ★ A6 合取另一半（PARITY_GATE note：err_kernel ≤ err_baseline × 1.5，双方对 fp64 oracle）：
        #   融合核不得比 HF 基线'显著更准'——比基线还准意味着它没走真计算（防作弊半边）。
        assert fp64_gap <= 1.5 * hf_gap + 1e-6, (shape, fp64_gap, hf_gap)


@pytest.mark.gpu
@requires_cuda
def test_triton_rms_norm_rejects_bad_contract():
    """契约违反必须**响亮报错**（不静默算错）。"""
    from lemoq.kernels.rmsnorm import fused_available, rms_norm_fused

    if not fused_available():
        pytest.skip("需要 triton + CUDA")

    x = torch.randn(1, 2048, device="cuda", dtype=torch.float16)
    w = torch.randn(2048, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="weight"):
        rms_norm_fused(x, w[:1024], 1e-5)
    with pytest.raises(NotImplementedError, match="2 的幂"):
        rms_norm_fused(
            torch.randn(1, 1000, device="cuda", dtype=torch.float16),
            torch.randn(1000, device="cuda", dtype=torch.float16),
            1e-5,
        )
    # ★ 用**有非 1 维**的张量构造非连续：`(1, 2048).t().contiguous().t()` 会被
    #   PyTorch 判成"连续"（它跳过 size==1 的维度），第一版就是这么写错的。
    nc = torch.randn(2, 3, 128, device="cuda", dtype=torch.float16).transpose(0, 1)
    assert not nc.is_contiguous(), "构造非连续张量失败"
    with pytest.raises(ValueError, match="连续"):
        rms_norm_fused(nc, torch.randn(128, device="cuda", dtype=torch.float16), 1e-5)
