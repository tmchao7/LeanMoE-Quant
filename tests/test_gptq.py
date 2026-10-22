"""Independent small-matrix oracle for the opt-in calibration algorithm."""

import numpy as np
import pytest
import torch

from lemoq.quant import dequantize_reference, quantize_linear
from lemoq.quant.gptq import CalibrationError, HessianStats


def _stats(x, cap=2048):
    s = HessianStats.empty(x.shape[-1], max_samples=cap)
    s.add(x)
    return s


def _numpy_gptq(weight, inputs, group, damp=0.01):
    """Unblocked, float64 derivation: q_i then eliminate its error from ALL later columns."""
    w = weight.astype(np.float64).copy()
    h = inputs.astype(np.float64).T @ inputs.astype(np.float64) * (2 / len(inputs))
    h += np.eye(h.shape[0]) * damp * np.diag(h).mean()
    u = np.linalg.cholesky(np.linalg.inv(h)).T
    scales = np.abs(w.reshape(w.shape[0], -1, group)).max(-1) / 7
    scales = np.where(scales > 0, scales, 1).astype(np.float16)
    codes = np.zeros_like(w, dtype=np.int32)
    for i in range(w.shape[1]):
        s = scales[:, i // group].astype(np.float64)
        codes[:, i] = np.clip(np.rint(w[:, i] / s), -8, 7)
        error = (w[:, i] - codes[:, i] * s) / u[i, i]
        w[:, i:] -= error[:, None] * u[i, i:]
    return codes, scales


@pytest.mark.parametrize("block", [1, 3, 4, 128])
def test_blocked_feedback_matches_independent_unblocked_numpy(block):
    torch.manual_seed(112)
    x = torch.randn(320, 16)
    x[:, 3] += x[:, 0] * 0.7
    w = torch.randn(7, 16)
    q = quantize_linear(
        w, bits=4, group_size=4, scheme="gptq", calibration=_stats(x), block_size=block
    )
    codes, scales = _numpy_gptq(w.numpy(), x.numpy(), 4)
    assert np.array_equal(q.codes.numpy(), codes)
    assert np.array_equal(q.scale.numpy(), scales)
    assert q.scale.dtype == torch.float16
    reference = torch.from_numpy(codes).reshape(7, 4, 4) * torch.from_numpy(
        scales
    ).float().unsqueeze(-1)
    assert torch.equal(dequantize_reference(q), reference.reshape_as(w))


def test_hessian_chunking_cap_and_shared_input_statistics():
    torch.manual_seed(5)
    x = torch.randn(300, 8)
    a, b = _stats(x, 256), HessianStats.empty(8, max_samples=256)
    b.add(x[:100])
    b.add(x[100:])
    assert a.samples == b.samples == 256 and a.seen == b.seen == 300
    torch.testing.assert_close(a.hessian(), b.hessian())
    xd = x[:256].double()
    torch.testing.assert_close(a.hessian().double(), 2 * xd.T @ xd / 256, atol=5e-7, rtol=1e-6)


def test_reconstruction_improves_a_correlated_output_objective():
    torch.manual_seed(112)
    x = torch.randn(512, 16)
    x[:, 1::2] = x[:, ::2] * 0.98 + x[:, 1::2] * 0.02
    w = torch.randn(8, 16)
    q = quantize_linear(w, group_size=4, scheme="gptq", calibration=_stats(x))
    rtn = quantize_linear(w, group_size=4)
    rtn.scale = rtn.scale.half()

    def error(candidate):
        return ((x @ (w - dequantize_reference(candidate)).T) ** 2).sum()

    assert error(q) < error(rtn)


def test_rank_deficient_hessian_is_damped_without_zeroing_dead_columns():
    x = torch.zeros(256, 8)
    x[:, 0] = 1
    w = torch.arange(32).reshape(4, 8).float() * 0.01
    q = quantize_linear(w, group_size=4, scheme="gptq", calibration=_stats(x))
    assert bool(torch.isfinite(dequantize_reference(q)).all())
    assert torch.count_nonzero(q.codes[:, 1:]) > 0


@pytest.mark.parametrize(
    "case", ["missing", "samples", "zero", "nan", "shape", "negative", "asymmetric"]
)
def test_invalid_statistics_fail_without_falling_back(case):
    x = torch.randn(256, 8)
    s = _stats(x)
    if case == "missing":
        s = None
    elif case == "samples":
        s.samples = 12
    elif case == "zero":
        s.gram.zero_()
    elif case == "nan":
        s.gram[0, 0] = float("nan")
    elif case == "shape":
        s.gram = torch.eye(6)
    elif case == "negative":
        s.gram = -torch.eye(8)
    elif case == "asymmetric":
        s.gram[0, 1] += 50
    with pytest.raises(CalibrationError):
        quantize_linear(torch.randn(4, 8), group_size=4, scheme="gptq", calibration=s)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 8},
        {"group_size": 3},
        {"damp": float("nan")},
        {"damp": 0},
        {"block_size": 0},
        {"symmetric": False},
    ],
)
def test_unsupported_quantizer_options_rejected(kwargs):
    opts = {"group_size": 4, "scheme": "gptq", "calibration": _stats(torch.randn(256, 8)), **kwargs}
    with pytest.raises(ValueError):
        quantize_linear(torch.randn(4, 8), **opts)


def test_rtn_contract_remains_default_and_does_not_consume_hessians():
    w = torch.randn(4, 8)
    assert torch.equal(
        quantize_linear(w, group_size=4).codes, quantize_linear(w, group_size=4, scheme="rtn").codes
    )
    with pytest.raises(ValueError, match="RTN"):
        quantize_linear(w, group_size=4, calibration=_stats(torch.randn(256, 8)))


def test_nonfinite_activations_do_not_increment_coverage():
    s = HessianStats.empty(4)
    with pytest.raises(CalibrationError):
        s.add(torch.full((256, 4), float("nan")))
    assert s.samples == s.seen == 0
