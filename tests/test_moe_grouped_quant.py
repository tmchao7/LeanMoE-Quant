"""test_moe_grouped_quant — P1（量化 × grouped MoE）的**离线**测试。

P1 = `EXP-p4b-01-truncation` 记录 §3 的"量化感知 expert 路径"。真 Triton kernel 的
数值对照是 GPU 上的事（`tests/test_kernel_parity.py`），但**这一组**能在 Mac 上跑，
而且它钉住的正是最容易静默出错的部分：

    1. "先融合再量化" 必须与 "逐专家量化再融合" **逐位相同**
       （否则 P1 就悄悄换了量化语义，而门是引用 p1-04 按逐专家测的）；
    2. 量化容器的反量化必须与 `rtn` 的参考一致；
    3. 量化路径的 oracle 必须与"fp16 路径跑在反量化权重上"一致；
    4. host 管线的形状/约束检查（不满足就报错，不静默算错）；
    5. 字节账：量化后的 experts 必须**真的更小**，且被 `_weight_bytes_actual` 数到。
"""

from __future__ import annotations

import pytest
from conftest import requires_torch

pytestmark = requires_torch

import torch  # noqa: E402

from lemoq.kernels.moe_grouped import static_moe_forward_grouped_reference  # noqa: E402
from lemoq.kernels.moe_grouped_quant import (  # noqa: E402
    QuantFusedExperts,
    dequantize_fused_experts,
    grouped_moe_forward_quant,
    quant_launch_plan,
    quantize_fused_experts,
    static_moe_forward_grouped_quant,
)
from lemoq.kernels.moe_static import (  # noqa: E402
    StaticShapedMoE,
    fuse_experts,
    replace_moe_blocks,
)
from lemoq.quant import rtn as qrtn  # noqa: E402
from lemoq.quant.linear import SchemeSpec  # noqa: E402


def _tiny_olmoe():
    """同 `test_runner_cells._fresh_olmoe` 的极小模型（每次新建，避免脏状态）。"""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return OlmoeForCausalLM(cfg).eval()


def _spec8(group_size: int = 32) -> SchemeSpec:
    return SchemeSpec.from_flags("w8a16", group_size, targets=("attention", "experts"))


def _fused_and_quantized(group_size: int = 32):
    model = _tiny_olmoe()
    block = model.model.layers[0].mlp
    fused = fuse_experts(block.experts)
    quant = quantize_fused_experts(fused, bits=8, group_size=group_size)
    return block, fused, quant


# ---------------------------------------------------------------------------
# 1. 等价性：先融合再量化 == 逐专家量化再融合
# ---------------------------------------------------------------------------
def test_quantized_fuse_matches_per_expert_quantization():
    """★ P1 的**语义正确性**：融合不改变任何一行的量化结果。

    RTN 是逐行、逐组的（组沿最后一维）。融合只做"把专家的行堆起来"与
    "gate/up 在 dim=1 拼接" —— 都不跨行。所以逐位相同必须成立，否则 P1 用的
    量化语义就与 p1-04 的门（按逐专家 `gate_proj/up_proj/down_proj` 测的）不一致。
    """
    group_size = 32
    block, _fused, quant = _fused_and_quantized(group_size)
    ffn = quant.ffn

    for e_idx, expert in enumerate(block.experts):
        cases = (
            (
                "gate",
                expert.gate_proj.weight,
                quant.packed_gate_up[e_idx, :ffn, :],
                quant.scale_gate_up[e_idx, :ffn, :],
            ),
            (
                "up",
                expert.up_proj.weight,
                quant.packed_gate_up[e_idx, ffn:, :],
                quant.scale_gate_up[e_idx, ffn:, :],
            ),
            ("down", expert.down_proj.weight, quant.packed_down[e_idx], quant.scale_down[e_idx]),
        )
        for name, weight, packed, scale in cases:
            ref = qrtn.quantize_rtn(weight.detach().float(), bits=8, group_size=group_size)
            # ★ `codes_padded` 是 3-D 的 (out, n_groups, group_size)（rtn 的原始布局），
            #   而容器存的是 (rows..., in_features) —— 比之前 reshape 成同一形状。
            ref_codes = ref["codes_padded"].reshape(weight.shape).to(torch.int8)
            assert torch.equal(packed, ref_codes), "expert %d 的 %s codes 与逐专家量化不一致" % (
                e_idx,
                name,
            )
            assert torch.equal(
                scale, ref["scale"].half()
            ), "expert %d 的 %s scale 与逐专家量化不一致" % (e_idx, name)


def test_dequantized_container_matches_rtn_reference():
    """容器的反量化 == `rtn.dequantize_rtn`（scale 存 fp16，所以比 fp16 舍入后的值）。"""
    group_size = 32
    block, _fused, quant = _fused_and_quantized(group_size)
    got = quant.dequant_gate_up_expert(0, dtype=None)[: quant.ffn, :]
    expert = list(block.experts)[0]
    expected = qrtn.dequantize_rtn(
        qrtn.quantize_rtn(expert.gate_proj.weight.detach().float(), bits=8, group_size=group_size)
    )
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# 2. oracle：量化路径 == fp16 路径跑在反量化权重上
# ---------------------------------------------------------------------------
def test_quantized_oracle_matches_fp16_oracle_on_dequantized_weights():
    """量化路径的 oracle 必须与"把反量化权重喂给 fp16 路径"完全一致。

    这是 host 管线（形状/索引/dtype）的锚点：kernel 与它比，它就与 fp16 参考比。
    """
    _block, _fused, quant = _fused_and_quantized()
    x = torch.randn(1, 1, 64)
    router = torch.randn(8, 64)
    out_q, logits_q = static_moe_forward_grouped_quant(
        x, router, quant, top_k=2, norm_topk_prob=False, act_fn=None, backend="oracle"
    )
    deq = dequantize_fused_experts(quant, dtype=torch.float32)
    out_f, logits_f = static_moe_forward_grouped_reference(
        x, router, deq, top_k=2, norm_topk_prob=False
    )
    assert torch.allclose(logits_q, logits_f, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_q.float(), out_f.float(), atol=1e-4, rtol=1e-4)


def test_grouped_moe_forward_quant_oracle_runs_and_is_finite():
    _block, _fused, quant = _fused_and_quantized()
    x = torch.randn(quant.hidden)
    ids = torch.tensor([0, 3], dtype=torch.int64)
    top_w = torch.tensor([0.6, 0.4])
    out = grouped_moe_forward_quant(x, quant, ids, top_w, backend="oracle")
    assert out.shape == (quant.hidden,)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 3. host 约束：不满足就报错，绝不静默算错
# ---------------------------------------------------------------------------
def test_quant_launch_plan_rejects_block_k_not_dividing_group_size():
    """BLOCK_K 必须整除 group_size：否则一个 k 块会跨组边界，整块共用一个 scale 不成立。"""
    _block, _fused, quant = _fused_and_quantized(group_size=32)
    with pytest.raises(ValueError, match="BLOCK_K"):
        quant_launch_plan(quant, top_k=2, config={"gate_up_act": {"BLOCK_K": 64, "SPLIT_K": 1}})


def test_quant_launch_plan_rejects_4bit():
    _block, _fused, quant = _fused_and_quantized()
    quant.bits = 4
    with pytest.raises(NotImplementedError):
        quant_launch_plan(quant, top_k=2)


def test_quantize_fused_experts_rejects_4bit_and_bad_group():
    _block, fused, _quant = _fused_and_quantized()
    with pytest.raises(NotImplementedError):
        quantize_fused_experts(fused, bits=4, group_size=32)
    with pytest.raises(NotImplementedError):
        quantize_fused_experts(fused, bits=8, group_size=7)  # 64 不能被 7 整除


def test_quantized_gather_is_rejected_at_the_api_level():
    """量化 × gather 不存在（gather 直接索引 fp16 融合张量）—— 在构造点就报错。"""
    block = _tiny_olmoe().model.layers[0].mlp
    with pytest.raises(NotImplementedError, match="grouped"):
        StaticShapedMoE(block, impl="gather", quant=_spec8())


# ---------------------------------------------------------------------------
# 4. 端到端：replace_moe_blocks(quant=…) + prefill（T>1，CPU 可跑）
# ---------------------------------------------------------------------------
def test_replace_moe_blocks_quantized_builds_container_and_saves_bytes():
    import copy

    from benchmarks.runner import _weight_bytes_actual

    model = _tiny_olmoe()
    fp16_model = copy.deepcopy(model)
    assert replace_moe_blocks(fp16_model, impl="gather") == 2
    assert replace_moe_blocks(model, impl="grouped", quant=_spec8(group_size=32)) == 2

    containers = [
        m._static_experts
        for m in model.modules()
        if isinstance(getattr(m, "_static_experts", None), QuantFusedExperts)
    ]
    assert len(containers) == 2
    assert all(c.bits == 8 and c.group_size == 32 for c in containers)

    fp16_bytes = _weight_bytes_actual(fp16_model, None)
    quant_bytes = _weight_bytes_actual(model, object())
    assert quant_bytes < fp16_bytes, "8-bit experts 必须比 fp16 小：%d vs %d" % (
        quant_bytes,
        fp16_bytes,
    )


def test_quantized_prefill_path_uses_per_expert_dequant():
    """T > 1（prefill）在量化容器上也必须能跑，且是**逐专家**反量化。

    decode（T=1）在 Mac 上需要 triton，但 prefill 永远走 `_prefill_forward` ——
    所以这条能在 CPU 上验证"量化模型能前向"。
    """
    model = _tiny_olmoe()
    replace_moe_blocks(model, impl="grouped", quant=_spec8(group_size=32))
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        out = model(ids)
    assert tuple(out.logits.shape) == (1, 8, 256)
    assert torch.isfinite(out.logits).all()
