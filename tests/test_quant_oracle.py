"""test_quant_oracle — oracle 与 torch 快路径必须**逐元素一致**。

没有这个断言，快路径悄悄偏离 oracle 时没人会发现，而真 kernel 又是照着快路径写的——
错误会一路传到结论里。

同时锁死打包/解包的往返正确性与 scale 开销的量级。
"""

from __future__ import annotations

import pytest
from conftest import requires_numpy, requires_torch

pytestmark = requires_numpy

import numpy as np  # noqa: E402

from lemoq.quant import numpy_oracle as no  # noqa: E402

CASES = [
    (4, 128, True),
    (4, 32, True),
    (4, 128, False),
    (8, 128, True),
    (8, 64, False),
]


def _weights(shape=(128, 256), seed=0):
    import numpy as np

    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


@pytest.mark.parametrize("bits,group_size,symmetric", CASES)
def test_oracle_roundtrip_reduces_error(bits, group_size, symmetric):
    w = _weights()
    q = no.quantize_rtn(w, bits=bits, group_size=group_size, symmetric=symmetric)
    w_hat = no.dequantize_rtn(q)
    assert w_hat.shape == w.shape
    assert q["codes"].shape == w.shape
    err = no.quantization_error(w, q)
    assert err["max_abs_err"] > 0
    # 位宽越高、分组越细，误差越小
    assert 5.0 < err["snr_db"] < 80.0


def test_more_bits_gives_lower_error():
    w = _weights()
    e4 = no.quantization_error(w, no.quantize_rtn(w, bits=4, group_size=128))["snr_db"]
    e8 = no.quantization_error(w, no.quantize_rtn(w, bits=8, group_size=128))["snr_db"]
    assert e8 > e4, "8-bit 的 SNR 必须高于 4-bit"


def test_smaller_group_gives_lower_error():
    w = _weights()
    coarse = no.quantization_error(w, no.quantize_rtn(w, bits=4, group_size=256))["snr_db"]
    fine = no.quantization_error(w, no.quantize_rtn(w, bits=4, group_size=32))["snr_db"]
    assert fine > coarse, "分组更细的 SNR 必须更高"


def test_codes_respect_bit_range():
    for bits, sym in ((4, True), (4, False), (8, True), (8, False)):
        q = no.quantize_rtn(_weights(), bits=bits, group_size=64, symmetric=sym)
        if sym:
            qmax = no.qmax_for(bits, True)
            lo, hi = -(qmax + 1), qmax  # int4: -8..7 ; int8: -128..127
        else:
            hi = no.qmax_for(bits, False)
            lo = 0  # uint4: 0..15 ; uint8: 0..255
        assert q["codes"].min() >= lo, (bits, sym, int(q["codes"].min()))
        assert q["codes"].max() <= hi, (bits, sym, int(q["codes"].max()))


def test_non_divisible_in_features_is_padded_correctly():
    """in_features 不被 group_size 整除时，padding 不能污染输出。"""
    w = _weights(shape=(64, 500))
    q = no.quantize_rtn(w, bits=4, group_size=128)
    out = no.dequantize_rtn(q)
    assert out.shape == (64, 500)
    assert q["padded_in"] == 512
    assert q["n_groups"] == 4


def test_rejects_bad_inputs():
    w = _weights()
    with pytest.raises(ValueError):
        no.quantize_rtn(w[0], bits=4, group_size=128)  # 1-D
    with pytest.raises(ValueError):
        no.quantize_rtn(w, bits=4, group_size=0)  # 非法分组
    with pytest.raises(ValueError):
        no.quantize_rtn(w, bits=3, group_size=128)  # 只支持 4/8


def test_bit_cost_counts_scale_overhead():
    """scale 开销**不可忽略**（g=128 时约 3%），它必须进字节账。"""
    cost = no.bit_cost(
        {"codes": _weights(shape=(128, 256)), "bits": 4, "scale": _weights(shape=(128, 2))}
    )
    assert cost["packed_bytes"] == 128 * 256 * 4 // 8
    assert cost["scale_bytes"] == 128 * 2 * 2
    assert cost["total_bytes"] == cost["packed_bytes"] + cost["scale_bytes"]


@requires_torch
@pytest.mark.parametrize("bits,group_size,symmetric", CASES)
def test_torch_path_matches_oracle_exactly(bits, group_size, symmetric):
    """**核心断言**：快路径与 oracle 必须 bit-identical（不是 allclose）。"""
    import numpy as np
    import torch

    from lemoq.quant import rtn as tr

    w = _weights()
    qn = no.quantize_rtn(w, bits=bits, group_size=group_size, symmetric=symmetric)
    qt = tr.quantize_rtn(torch.from_numpy(w), bits=bits, group_size=group_size, symmetric=symmetric)
    assert np.array_equal(qn["codes"], qt["codes"].numpy())
    assert np.array_equal(qn["scale"], qt["scale"].numpy())
    assert np.allclose(no.dequantize_rtn(qn), tr.dequantize_rtn(qt).numpy(), atol=0.0, rtol=0.0)


@requires_torch
def test_api_layer_matches_oracle():
    import numpy as np
    import torch

    from lemoq.quant import dequantize_reference, quantize_linear

    w = _weights()
    ql = quantize_linear(torch.from_numpy(w), bits=4, group_size=128)
    got = dequantize_reference(ql).numpy()
    want = no.dequantize_rtn(no.quantize_rtn(w, bits=4, group_size=128))
    assert np.allclose(got, want, atol=0.0, rtol=0.0)


@requires_torch
def test_api_rejects_unimplemented_scheme():
    import torch

    from lemoq.quant import quantize_linear

    with pytest.raises(NotImplementedError):
        quantize_linear(torch.zeros(4, 128), scheme="gptq")


def test_remediation_ladder_is_documented():
    """精度不过门槛时的兜底阶梯必须存在，否则项目会卡住而没有出路。"""
    from lemoq.quant import REMEDIATION_LADDER

    steps = [s["step"] for s in REMEDIATION_LADDER]
    assert "1_rtn" in steps
    assert any("gptq" in s for s in steps)
    assert any("awq" in s for s in steps)
    assert any("mixed_bits" in s for s in steps)


@pytest.mark.parametrize("bits", [4, 8])
def test_pack_roundtrip_is_exact(bits):
    from lemoq.quant import pack

    w = _weights(shape=(64, 256))
    q = no.quantize_rtn(w, bits=bits, group_size=64)
    if bits == 4:
        packed = pack.pack_4bit(q["codes"])
        un = pack.unpack_4bit(packed, signed=True)
        assert packed.shape == (64, 128)
        assert packed.dtype.kind == "u"
        assert (un.astype("int32") == q["codes"]).all(), "4-bit 打包往返必须精确"
    else:
        # ★ 8-bit 这一格以前执行 0 条断言：参数化写了 [4, 8]，函数体却只有
        #   `if bits == 4:` 而没有 else —— 于是 8-bit 的打包往返从来没被测过。
        #   8-bit 的"打包"就是 int8 化，对称量化的 code 域是 -128..127，必须无损。
        packed = q["codes"].astype("int8")
        assert packed.dtype == np.int8
        assert (packed.astype("int32") == q["codes"]).all(), "8-bit 打包往返必须精确"
        # 判别力：真出现越界（例如 code 落到 int8 之外）必须被抓到
        assert int(np.abs(q["codes"]).max()) <= 127, "对称 8-bit 的 code 必须在 int8 域内"


def test_pack_rejects_odd_last_dim():
    from lemoq.quant import pack

    with pytest.raises(ValueError):
        pack.pack_4bit(_weights(shape=(8, 33)))


def test_pack_size_matches_bytes_module():
    """打包字节数必须与 `benchmarks.bytes` 的账一致，否则 [W] 会对不上。"""
    from benchmarks.bytes import _quantized_bytes
    from lemoq.quant import pack

    n, bits, g = 805_306_368, 4, 128
    assert pack.packed_nbytes(n, bits) == n * bits // 8
    assert pack.bit_cost(n, bits, g)["total_bytes"] == _quantized_bytes(n, bits, g, 2)
