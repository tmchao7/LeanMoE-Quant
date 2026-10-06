"""test_quant_linear — 把量化装到模型上的离线回归（全部 CPU 可跑）。

## 为什么先写这些再上机

`EXP-p1-01` 的结论是"某个位宽/分组能不能过 KL 门"。在那之前，必须先证明
**量化确实被装上了**，否则"W4 的 KL 不好"可能只是"W4 根本没生效"或
"装到了错的层上"。这两类错误在 GPU 上都不报错，只会给出一个看起来合理的 KL。

所以这里钉住四件事：

1. 打包是真的（不是留 int32 codes 假装量化）；
2. 反量化与 `api.dequantize_reference` 一致；
3. 层的**归类**正确（attention / experts / router / lm_head / 未知）；
4. **实数 vs 账**：`realized_bytes` 与 `bytes._quantized_bytes` 的对照。
   项目的 `[W]` 全部建立在那条公式上，这是它第一次被一个真实实现核对。
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lemoq.quant import api as qapi  # noqa: E402
from lemoq.quant.linear import (  # noqa: E402
    QuantLinear,
    SchemeSpec,
    classify_linear,
    parse_scheme,
    quantize_model,
    realized_bytes,
)


def _spec(name="w4a16", group_size=128, **kw):
    return SchemeSpec.from_flags(name, group_size, **kw)


# ---------------------------------------------------------------------------
# 1) scheme 解析与层归类
# ---------------------------------------------------------------------------
def test_parse_scheme_separates_weight_and_activation_bits():
    """★ a16 与 a8 必须分开 —— 它们是两个不同的实验臂。

    混为一谈会让"W8A8 的精度损失"被错误归因给权重位宽，而其中一部分来自
    激活量化。
    """
    assert parse_scheme("w8a16") == (8, None)
    assert parse_scheme("w4a16") == (4, None)
    assert parse_scheme("w8a8") == (8, 8)
    assert parse_scheme("fp16") == (None, None)
    with pytest.raises(ValueError):
        parse_scheme("w3a16")


@pytest.mark.parametrize(
    "module_name,expected",
    [
        ("lm_head", "lm_head"),
        ("model.layers.0.self_attn.q_proj", "attention"),
        ("model.layers.15.self_attn.o_proj", "attention"),
        ("model.layers.3.mlp.gate", "router"),
        ("model.layers.3.mlp.experts.17.gate_proj", "experts"),
        ("model.layers.3.mlp.experts.17.up_proj", "experts"),
        ("model.layers.3.mlp.experts.17.down_proj", "experts"),
        ("model.layers.3.mlp.shared_expert.gate_proj", "shared"),
        ("model.embed_tokens", None),  # 不是 Linear，但归类也不该瞎猜
        ("some.unknown.proj", None),
    ],
)
def test_classify_linear_maps_olmoe_names(module_name, expected):
    """归类错 = 量化了错的层，而 KL 不会告诉你这件事。"""
    assert classify_linear(module_name) == expected


def test_classify_linear_does_not_confuse_router_with_expert_gate():
    """★ `mlp.gate`（router, 0.18% of [W]）与 `experts.i.gate_proj`（68.3%）必须分开。

    正则写松一点就会把两者混起来 —— 而它们占 [W] 的比例差 380 倍。
    """
    assert classify_linear("model.layers.0.mlp.gate") == "router"
    assert classify_linear("model.layers.0.mlp.experts.0.gate_proj") == "experts"
    assert classify_linear("model.layers.0.mlp.experts.63.down_proj") == "experts"


# ---------------------------------------------------------------------------
# 2) 打包是真的，且反量化正确
# ---------------------------------------------------------------------------
def test_packing_is_real_not_an_int32_tensor_in_disguise():
    """★ `[W]` 说的是**打包后**的字节。留 int32 codes 会让实数比账大 8 倍。"""
    linear = nn.Linear(256, 512, bias=False)
    ql = QuantLinear(linear, _spec("w4a16", 128))
    # (512, 256) 4-bit -> 512*256/2 = 65536 B
    assert ql.packed.dtype == torch.uint8, ql.packed.dtype
    assert ql.packed.numel() == 512 * 256 // 2

    q8 = QuantLinear(linear, _spec("w8a16", 128))
    assert q8.packed.dtype == torch.int8
    assert q8.packed.numel() == 512 * 256


def test_dequantized_weight_matches_a_fp16_scale_reference_exactly():
    """★ 契约是"与**同样存 fp16 scale** 的参考一致"，不是"与 fp32 scale 参考逐位一致"。

    `api.dequantize_reference` 用 fp32 scale；`QuantLinear` 按
    `bytes.QuantSpec.scale_dtype_bytes=2` 存 fp16。所以两者必然有 fp16 量级的差 ——
    那是记账口径决定的，不是 bug。这里把契约写成可检验的形式：
    参考也用 fp16 scale 时，必须**逐位相等**。
    """
    torch.manual_seed(0)
    linear = nn.Linear(128, 64, bias=False)
    for name in ("w4a16", "w8a16"):
        ql = QuantLinear(linear, _spec(name, 64))
        q = qapi.quantize_linear(
            linear.weight.detach(), bits=ql.weight_bits, group_size=64, symmetric=True
        )
        codes = qapi._repad(q).float().reshape(64, q.meta["n_groups"], 64)
        scale16 = q.scale.half().float().reshape(64, q.meta["n_groups"], 1)
        ref16 = (codes * scale16).reshape(64, q.meta["padded_in"])[:, :128]
        assert torch.equal(ql.dequantized_weight(), ref16), name


def test_fp16_scale_storage_stays_within_fp16_rounding_of_the_fp32_reference():
    """同时钉住"差多少"：必须只在 fp16 scale 舍入的量级内。"""
    torch.manual_seed(0)
    linear = nn.Linear(128, 64, bias=False)
    ql = QuantLinear(linear, _spec("w4a16", 64))
    fp32_ref = qapi.dequantize_reference(
        qapi.quantize_linear(linear.weight.detach(), bits=4, group_size=64, symmetric=True)
    )
    diff = (ql.dequantized_weight() - fp32_ref).abs().max()
    scale = fp32_ref.abs().max()
    assert float(diff) <= float(scale) * 1e-3, (float(diff), float(scale))


def test_forward_equals_linear_of_dequantized_weight():
    """`forward` 的契约：反量化后再走 fp16 `F.linear`。"""
    torch.manual_seed(1)
    linear = nn.Linear(128, 64)
    ql = QuantLinear(linear, _spec("w8a16", 32))
    x = torch.randn(4, 128)
    expect = torch.nn.functional.linear(x, ql.dequantized_weight(dtype=x.dtype), ql.bias)
    assert torch.allclose(ql(x), expect, atol=1e-6)


def test_scale_is_fp16_and_symmetric_quantization_stores_no_zero():
    """★ scale 存 fp16（2 B）；对称量化**不存 zero**。

    存 fp32 scale 会让实数比 `[W]` 的账大一倍；对称量化还存 zero 会白占
    `out * n_groups * 2` 字节（实测 (2048,2048)/group=32 时多 262,144 B），
    并且让"实数 vs 账"这条核对永远对不上 —— 那核对就失去了意义。
    """
    ql = QuantLinear(nn.Linear(256, 256, bias=False), _spec("w4a16", 64))
    assert ql.scale.dtype == torch.float16
    assert ql.zero is None
    assert realized_bytes(ql)["zero_bytes"] == 0

    # 非对称量化才需要 zero，且必须是 fp16
    asym = SchemeSpec.from_flags("w4a16", 64)
    asym = SchemeSpec(**{**asym.__dict__, "symmetric": False})
    qa = QuantLinear(nn.Linear(256, 256, bias=False), asym)
    assert qa.zero is not None and qa.zero.dtype == torch.float16


# ---------------------------------------------------------------------------
# 3) ★ 实数 vs 账 —— 项目 [W] 公式的第一次对照
# ---------------------------------------------------------------------------
def test_realized_bytes_match_the_w_formula_when_dimensions_divide():
    """OLMoE 的形状全部整除，所以实数应当与 `_quantized_bytes` **完全相等**。

    `_quantized_bytes` 是 `docs/METRICS.md` §3.2 里 `[W]` 的公式实现。
    """
    from benchmarks.bytes import _quantized_bytes

    linear = nn.Linear(2048, 2048, bias=False)
    for name, bits in (("w8a16", 8), ("w4a16", 4)):
        for gs in (32, 64, 128):
            ql = QuantLinear(linear, _spec(name, gs))
            realized = realized_bytes(ql)["total_bytes"]
            predicted = _quantized_bytes(2048 * 2048, bits, gs, 2)
            assert realized == predicted, (name, gs, realized, predicted)


def test_w_formula_undercounts_scales_when_the_input_dim_does_not_divide():
    """★ 发现：`_quantized_bytes` 的 `n_groups = ceil(n_params / group_size)`
    只在**输入维能被 group_size 整除**时等于真实的 `out * ceil(in / group_size)`。

    实测反例：`in=300, out=8, group_size=128`
        真实 scale 数 = 8 * ceil(300/128) = 8 * 3 = 24
        公式给出     = ceil(8*300/128)    = ceil(18.75) = 19

    所以对维度不整除的模型，公式会**低估** scale 开销。OLMoE 不受影响
    （2048 / 1024 / 50304 对 32/64/128 都整除），但这条限制应当写下来而不是
    等别人踩。这里断言**实测值**，并把差值记在测试里。
    """
    from benchmarks.bytes import _quantized_bytes

    linear = nn.Linear(300, 8, bias=False)
    ql = QuantLinear(linear, _spec("w8a16", 128))
    realized = realized_bytes(ql)["scale_bytes"]
    true_scales = 8 * ((300 + 127) // 128) * 2  # fp16
    predicted = _quantized_bytes(8 * 300, 8, 128, 2) - (8 * 300 * 8 // 8)
    assert realized == true_scales
    assert predicted != true_scales, "公式若哪天修好了，这条测试会断 —— 那是好事"


def test_padding_inflation_is_measured_not_assumed():
    """padding 的字节也是真占的。OLMoE 整除 → 0%，但必须是**测出来的**。"""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = OlmoeForCausalLM(cfg).eval()
    _, report = quantize_model(model, _spec("w4a16", 32))
    assert report.padding["inflation_pct"] == 0.0
    assert report.padding["n_params_padded"] == report.padding["n_params_unpadded"]

    # 换一个不整除的 group_size：hidden=64 / gs=48 -> 64 % 48 != 0 -> 有膨胀
    torch.manual_seed(0)
    model2 = OlmoeForCausalLM(cfg).eval()
    _, report2 = quantize_model(model2, _spec("w4a16", 48))
    assert report2.padding["inflation_pct"] > 0.0


# ---------------------------------------------------------------------------
# 4) 层选择：不静默跳过
# ---------------------------------------------------------------------------
def _tiny_model():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return OlmoeForCausalLM(cfg).eval()


def test_quantize_model_replaces_only_declared_targets():
    model = _tiny_model()
    _, report = quantize_model(model, _spec("w4a16", 32, targets=("attention", "experts")))
    assert report.replaced["attention"] == 4
    assert report.replaced["experts"] == 4 * 3  # 每个专家 3 个投影
    assert "router" not in report.replaced
    # router 与 lm_head 必须被**显式列出**为跳过，不是静默略过
    assert any("mlp.gate" in s for s in report.skipped)
    assert any(s.startswith("lm_head") for s in report.skipped)
    # 真的换成了 QuantLinear
    assert isinstance(model.model.layers[0].self_attn.q_proj, QuantLinear)
    assert not isinstance(model.model.layers[0].mlp.gate, QuantLinear)


def test_keep_lm_head_fp16_is_honoured_and_recorded():
    model = _tiny_model()
    _, keep = quantize_model(model, _spec("w4a16", 32, targets=("attention", "experts", "lm_head")))
    assert "lm_head" not in keep.replaced
    assert any("keep_lm_head_fp16" in s for s in keep.skipped)

    model2 = _tiny_model()
    _, drop = quantize_model(
        model2,
        _spec("w4a16", 32, targets=("attention", "experts", "lm_head"), keep_lm_head_fp16=False),
    )
    assert drop.replaced["lm_head"] == 1


def test_fp16_scheme_is_a_no_op_and_is_reported_as_such():
    model = _tiny_model()
    before = sum(p.numel() for p in model.parameters())
    _, report = quantize_model(model, _spec("fp16", 32))
    assert report.replaced == {}
    assert sum(p.numel() for p in model.parameters()) == before


# ---------------------------------------------------------------------------
# 5) 数值：位宽单调性（比一个绝对 KL 阈值更能说明"量化真的生效了"）
# ---------------------------------------------------------------------------
def test_quantization_error_grows_as_bits_shrink():
    """★ 量化必须**真的**改变输出，且位宽越低差得越多。

    如果 W4 与 W8 的输出一样，那说明量化没生效 —— 那时 KL 门测的是别的东西。
    """
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        ref = model(ids).logits

    diffs = {}
    for name in ("w8a16", "w4a16"):
        m = _tiny_model()  # 同样的种子 -> 同样的 fp16 权重
        _, _ = quantize_model(m, _spec(name, 32, targets=("attention", "experts")))
        with torch.no_grad():
            diffs[name] = float((m(ids).logits - ref).abs().max())

    assert diffs["w8a16"] > 0.0, "W8 没有改变输出 -> 量化没生效"
    assert diffs["w4a16"] > diffs["w8a16"], diffs
    assert torch.isfinite(torch.tensor(list(diffs.values()))).all()


def test_w8a8_also_quantizes_activations_and_says_so():
    """★ `w8a8` 比 `w8a16` 多一个变量（激活量化），必须在 spec 里标注它是 fake-quant。

    否则"W8A8 比 W8A16 差"会被错误归因给权重位宽。
    """
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        ref = model(ids).logits

    m16 = _tiny_model()
    _, r16 = quantize_model(m16, _spec("w8a16", 32, targets=("attention",)))
    m8 = _tiny_model()
    _, r8 = quantize_model(m8, _spec("w8a8", 32, targets=("attention",)))
    with torch.no_grad():
        d16 = float((m16(ids).logits - ref).abs().max())
        d8 = float((m8(ids).logits - ref).abs().max())

    assert d8 > d16, (d8, d16)
    assert r8.spec["act_bits"] == 8 and r16.spec["act_bits"] is None
    assert "fake-quant" in r8.spec["activation_quantization"]
    assert r16.spec["activation_quantization"] == "none"


def test_report_carries_everything_needed_to_check_the_byte_account():
    """报告里缺 padding / 实数 / targets 任一项，`[W]` 就不可核对。"""
    model = _tiny_model()
    _, report = quantize_model(model, _spec("w4a16", 32, targets=("attention", "experts")))
    d = report.to_dict()
    assert d["realized_bytes_total"] > 0
    assert set(d["realized_bytes_by_target"]) == {"attention", "experts"}
    assert "inflation_pct" in d["padding"]
    assert d["spec"]["targets"] == ["attention", "experts"]
    # 每个 target 的实数必须小于它 fp16 时的字节数（否则量化没省东西）
    for target, bucket in d["realized_bytes_by_target"].items():
        fp16_bytes = bucket["n_params"] * 2
        assert bucket["total_bytes"] < fp16_bytes, target
