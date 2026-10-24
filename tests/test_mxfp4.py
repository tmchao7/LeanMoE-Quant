"""test_mxfp4 — MXFP4 主机侧量化器的规范对拍（OCP MX v1.0；侦察 §2 坑清单的机器化）。

钉住的口径：
    - E2M1 码表（16 码全查）与 E8M0 尺度（bias 127）；
    - 共享指数 X = floor(log2 amax) − 2；全零块尺度字节 0、码全 0；
    - RTN 就近（平局取较小幅值码）；饱和到 ±6（格式 wart，如实保留）；
    - 误差界：幅值 ≤ 6s 区间 err ≤ 1s，饱和区 err < 2s ⇒ 全局 ≤ 2s；
    - 打包复用 pack_4bit（&0xF 掩码，unsigned 0..15）；pad 列回读为 0。
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lemoq.quant.mxfp4 import (  # noqa: E402
    _E2M1_MAG,
    BLOCK,
    dequantize_mxfp4,
    e2m1_decode,
    e8m0_to_scale,
    quantize_mxfp4,
)


def test_e2m1_decode_full_table():
    """16 个码逐一核对（bit3 符号 + bit2..0 幅值）。"""
    for code, want in enumerate(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    ):
        got = e2m1_decode(code)
        assert got == pytest.approx(want, abs=0.0), (code, got, want)


def test_e8m0_scale_bias_127():
    assert e8m0_to_scale(127) == 1.0
    assert e8m0_to_scale(128) == 2.0
    assert e8m0_to_scale(126) == 0.5
    assert e8m0_to_scale(0) == 2.0**-127


def test_amax_6_block_encodes_the_ladder():
    """amax=6 ⇒ X = floor(log2 6) − 2 = 0 ⇒ scale 1；阶梯值各自落码。"""
    w = torch.tensor([[_E2M1_MAG[i] for i in range(8)] + [0.0] * (BLOCK - 8)])
    q = quantize_mxfp4(w)
    assert q["scales"][0, 0].item() == 127  # X=0
    dq = dequantize_mxfp4(q)
    assert torch.equal(dq, w)


def test_value_above_6s_saturates_and_error_bound_holds():
    """amax=7.9 ⇒ 7.9 超出 6·scale ⇒ 饱和到 6（格式 wart）；误差 < 2s。"""
    w = torch.tensor([[7.9] + [0.0] * (BLOCK - 1)])
    q = quantize_mxfp4(w)
    s = e8m0_to_scale(q["scales"][0, 0].item())
    assert s == pytest.approx(2.0 ** (math.floor(math.log2(7.9)) - 2))
    dq = dequantize_mxfp4(q)[0, 0].item()
    assert dq == pytest.approx(6.0 * s)
    assert abs(7.9 - dq) < 2.0 * s


def test_small_block_picks_fine_scale():
    """amax=0.1 ⇒ floor(log2 0.1)=−4，X=−6，scale=2^-6；0.1/s=6.4 → 饱和到 6 ⇒ 0.09375。

    （手算警告：log2(0.1)=−3.32 的 floor 是 **−4**，不是 −3——第一版测试在这里算错。）
    """
    w = torch.tensor([[0.1] + [0.0] * (BLOCK - 1)])
    q = quantize_mxfp4(w)
    assert q["scales"][0, 0].item() == 127 - 6
    dq = dequantize_mxfp4(q)[0, 0].item()
    assert dq == pytest.approx(6.0 * 2.0**-6)  # 6.4 饱和到 6
    assert abs(0.1 - dq) <= 2.0 * 2.0**-6


def test_negative_values_carry_sign_bit():
    w = torch.tensor([[-3.0, 3.0] + [0.0] * (BLOCK - 2)])  # amax=3 ⇒ X=−1, s=0.5
    q = quantize_mxfp4(w)
    dq = dequantize_mxfp4(q)
    # 3/0.5 = 6 → 恰好落在最大码上（无饱和）；符号位随码走。
    assert dq[0, 0].item() == pytest.approx(-3.0)
    assert dq[0, 1].item() == pytest.approx(3.0)


def test_all_zero_block_gets_scale_byte_zero():
    w = torch.zeros(1, BLOCK)
    q = quantize_mxfp4(w)
    assert q["scales"][0, 0].item() == 0
    assert dequantize_mxfp4(q).abs().max().item() == 0.0


def test_random_tensor_error_bounded_by_two_scales():
    """全局误差界 ≤ 2·scale（逐元素，按所在块的 scale）。"""
    torch.manual_seed(0)
    w = torch.randn(8, BLOCK * 4) * 0.3
    q = quantize_mxfp4(w)
    dq = dequantize_mxfp4(q)
    s = q["scales"].float().apply_(e8m0_to_scale).repeat_interleave(BLOCK, dim=1)
    assert (dq - w).abs().max().item() <= 2.0 * s.max().item() + 1e-12
    err = (dq - w).abs()
    assert torch.all(err <= 2.0 * s + 1e-12), "逐元素界被打破"


def test_padding_columns_round_trip_to_zero_and_are_excluded():
    """K=50 → pad 到 64：pad 列不进 absmax（全零块语义），回读切片回 50。"""
    torch.manual_seed(3)
    w = torch.randn(4, 50) * 0.2
    q = quantize_mxfp4(w)
    assert q["padded_in"] == 64
    assert q["codes_packed"].shape == (4, 32)  # 64 半字节 → 32 B
    assert q["scales"].shape == (4, 2)
    dq = dequantize_mxfp4(q)
    assert dq.shape == (4, 50)
    # pad 区不污染最后一块的尺度：末块 amax 只看真实列
    tail_true_amax = w[:, 32:50].abs().max().item()
    got_x = q["scales"][0, 1].item() - 127
    assert 2**got_x <= max(tail_true_amax, 1e-30) * 2.0 or tail_true_amax == 0.0


def test_pack_round_trip_preserves_unsigned_codes():
    """码 0..15（含负号位）经 pack/unpack(signed=False) 必须无损。"""
    torch.manual_seed(5)
    codes = torch.randint(0, 16, (2, BLOCK * 4), dtype=torch.uint8)
    q = {
        "codes_packed": __import__("lemoq.quant.pack", fromlist=["pack_4bit"]).pack_4bit(codes),
        "scales": torch.full((2, 4), 127, dtype=torch.uint8),
        "orig_in": BLOCK * 4,
        "padded_in": BLOCK * 4,
        "block": BLOCK,
    }
    from lemoq.quant.pack import unpack_4bit  # noqa: PLC0415

    back = unpack_4bit(q["codes_packed"], signed=False).to(torch.uint8)
    assert torch.equal(back.reshape(2, -1), codes)
    # 反量化在 X=0（scale 1）下逐码解码
    dq = dequantize_mxfp4(q)
    for i in range(BLOCK * 4):
        assert dq[0, i].item() == pytest.approx(e2m1_decode(int(codes[0, i])), abs=1e-6)


def test_quant_linear_mxfp4_end_to_end_tiny():
    """QuantLinear 的 mxfp4 分支端到端（CPU）：构造/forward/参照一致/组大小强制 32。"""
    from lemoq.quant.linear import QuantLinear, SchemeSpec

    torch.manual_seed(11)
    lin = torch.nn.Linear(64, 32, dtype=torch.float32)
    spec = SchemeSpec.from_flags("mxfp4", 128, targets=("attention",))  # 传 128 也必须强制 32
    assert spec.group_size == 32
    ql = QuantLinear(lin, spec, name="tiny")
    assert ql.group_size == 32 and ql.n_groups == 2 and ql.scale.dtype == torch.uint8
    x = torch.randn(1, 64, dtype=torch.float16)
    y = ql(x)
    w = ql.dequantized_weight(dtype=torch.float16)
    y_ref = torch.nn.functional.linear(x, w, ql.bias.to(x.dtype))
    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2), "separate 臂与参照权重不一致"
    # 误差界：|w - lin.weight| ≤ 2·scale（逐元素，所在块）
    s = (
        ql.scale.float()
        .apply_(__import__("lemoq.quant.mxfp4", fromlist=["e8m0_to_scale"]).e8m0_to_scale)
        .repeat_interleave(32, dim=1)[:, :64]
    )
    assert torch.all((w.float() - lin.weight).abs() <= 2.0 * s + 1e-12)
