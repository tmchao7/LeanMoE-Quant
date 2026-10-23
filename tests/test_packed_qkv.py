"""F2 candidate — packed q/k/v projection contracts.

What is CPU-testable here: the switch default, the install/remove path, the
refusals (nothing silently falls back), the concatenated buffer shapes, and the
**packing math** through the eager reference (one call on the concatenated
buffers must equal three separate calls).

What is not: the Triton kernel parity — that is the `gpu`-marked test at the
bottom, and it is the K1 gate for this candidate (bitwise, because the grouped
w8 layout is row-local).
"""

from __future__ import annotations

import dataclasses

import pytest
from conftest import requires_cuda, requires_torch

pytestmark = requires_torch

import torch  # noqa: E402

from lemoq.kernels.packed_qkv import (  # noqa: E402
    PackedQKV,
    install_packed_qkv,
    packed_qkv_enabled,
    packed_qkv_installed,
    remove_packed_qkv,
)
from lemoq.kernels.reference import dequant_gemm_fused_reference  # noqa: E402
from lemoq.quant.linear import QuantLinear, SchemeSpec  # noqa: E402

IN, OUT, GROUP = 32, 16, 16


def _spec(*, bits=8, symmetric=True, dequant="fused"):
    name = "w8a16" if bits == 8 else "w4a16"
    spec = SchemeSpec.from_flags(name, GROUP, targets=("attention",), dequant=dequant)
    return dataclasses.replace(spec, symmetric=symmetric)


def _qlin(*, out=OUT, bits=8, bias=False, symmetric=True, seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(IN, out, bias=bias)
    return QuantLinear(lin, _spec(bits=bits, symmetric=symmetric), name="qkv")


class _FakeAttention(torch.nn.Module):
    """Duck-typed so the capability-based selector accepts it (see `_is_olmoe_attention`)."""

    def __init__(self, q, k, v):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = q, k, v
        self.o_proj = torch.nn.Identity()
        self.q_norm = torch.nn.Identity()
        self.k_norm = torch.nn.Identity()
        self.num_heads = 1
        self.head_dim = OUT


def _fake_model(q, k, v):
    layer = torch.nn.Module()
    layer.self_attn = _FakeAttention(q, k, v)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([layer])
    return model


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("LEMOQ_PACKED_QKV", raising=False)
    assert packed_qkv_enabled() is False
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "1")
    assert packed_qkv_enabled() is True
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "0")
    assert packed_qkv_enabled() is False


def test_install_rejects_unquantized_projections():
    model = _fake_model(
        torch.nn.Linear(IN, OUT, bias=False),
        torch.nn.Linear(IN, OUT, bias=False),
        torch.nn.Linear(IN, OUT, bias=False),
    )
    with pytest.raises(ValueError, match="已量化"):
        install_packed_qkv(model)


def test_install_rejects_bias():
    model = _fake_model(_qlin(bias=True), _qlin(), _qlin())
    with pytest.raises(NotImplementedError, match="bias"):
        install_packed_qkv(model)


def test_install_rejects_asymmetric():
    model = _fake_model(_qlin(symmetric=False), _qlin(), _qlin())
    with pytest.raises(NotImplementedError, match="对称"):
        install_packed_qkv(model)


def test_install_rejects_4bit_first_round():
    model = _fake_model(_qlin(bits=4), _qlin(bits=4), _qlin(bits=4))
    with pytest.raises(NotImplementedError, match="w8"):
        install_packed_qkv(model)


def test_install_rejects_mismatched_shapes():
    model = _fake_model(_qlin(out=OUT), _qlin(out=OUT * 2), _qlin(out=OUT))
    with pytest.raises(ValueError, match="完全一致"):
        install_packed_qkv(model)


def test_install_shapes_and_remove():
    q, k, v = _qlin(seed=1), _qlin(seed=2), _qlin(seed=3)
    model = _fake_model(q, k, v)
    assert packed_qkv_installed(model) == 0
    assert install_packed_qkv(model) == 1
    assert packed_qkv_installed(model) == 1
    packed = model.model.layers[0].self_attn._lemoq_packed_qkv
    assert isinstance(packed, PackedQKV)
    assert packed.packed.shape == (3 * OUT, IN)
    assert packed.scale.shape == (3 * OUT, IN // GROUP)
    # 行序 = q | k | v，逐位来自各自的 buffer
    assert torch.equal(packed.packed[:OUT], q.packed)
    assert torch.equal(packed.packed[OUT : 2 * OUT], k.packed)
    assert torch.equal(packed.packed[2 * OUT :], v.packed)
    assert remove_packed_qkv(model) == 1
    assert packed_qkv_installed(model) == 0


def test_packed_reference_equals_three_separate_calls():
    """The packing math: one call on the concatenated buffers == three calls (K1 1e-3)."""
    q, k, v = _qlin(seed=7), _qlin(seed=8), _qlin(seed=9)
    model = _fake_model(q, k, v)
    install_packed_qkv(model)
    packed = model.model.layers[0].self_attn._lemoq_packed_qkv
    torch.manual_seed(11)
    x = torch.randn(4, IN)
    kw = {"group_size": GROUP, "symmetric": True}
    y = dequant_gemm_fused_reference(x, packed.packed, packed.scale, None, **kw)
    for start, module in ((0, q), (OUT, k), (2 * OUT, v)):
        ref = dequant_gemm_fused_reference(x, module.packed, module.scale, None, **kw)
        torch.testing.assert_close(y[:, start : start + OUT], ref, atol=1e-3, rtol=1e-3)


@pytest.mark.gpu
@requires_cuda
def test_gpu_packed_kernel_is_bitwise_equal_to_three_calls():
    """K1 for F2: one packed launch must reproduce the three single-projection rows **exactly**.

    Row-local claim: codes and scale are indexed per output row and input group, so
    changing N only changes which program computes which row — not the arithmetic.
    A non-zero diff here means the layout assumption is wrong; do not relax it.
    """
    from lemoq.kernels._dequant_triton import dequant_gemm_fused

    q = _qlin(seed=21).cuda()
    k = _qlin(seed=22).cuda()
    v = _qlin(seed=23).cuda()
    model = _fake_model(q, k, v)
    install_packed_qkv(model)
    packed = model.model.layers[0].self_attn._lemoq_packed_qkv
    torch.manual_seed(5)
    x = torch.randn(1, IN, device="cuda", dtype=torch.float16)
    kw = {"group_size": GROUP, "symmetric": True, "bits": 8, "out_dtype": torch.float16}
    y = dequant_gemm_fused(x, packed.packed, packed.scale, None, **kw)
    for start, module in ((0, q), (OUT, k), (2 * OUT, v)):
        ref = dequant_gemm_fused(x, module.packed, module.scale, None, **kw)
        err = (y[:, start : start + OUT] - ref).abs().max().item()
        assert err == 0.0, "packed kernel differs from single projection by %g" % err
