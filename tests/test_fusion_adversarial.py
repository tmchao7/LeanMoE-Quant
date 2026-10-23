"""Discriminating numerical fixtures; GPU tests here are prepared, not run on Mac."""

import pytest
import torch


def test_equal_boundary_weights_do_not_make_experts_interchangeable():
    # Equal softmax weights permit multiple top-k sets, but different expert outputs.
    weights = torch.softmax(torch.zeros(64), dim=-1)
    experts = torch.arange(64, dtype=torch.float32)
    low_index = torch.arange(8)
    alternative = torch.arange(1, 9)
    assert torch.equal(weights[low_index], weights[alternative])
    assert (weights[low_index] * experts[low_index]).sum() != (
        weights[alternative] * experts[alternative]
    ).sum()


def test_residual_epilogue_must_preserve_intermediate_fp16_rounding():
    acc = torch.tensor([1.0 + 2.0**-11], dtype=torch.float32)
    residual = torch.tensor([-1.0], dtype=torch.float16)
    original = (acc.half() + residual).half()
    fused_without_intermediate_rounding = (acc + residual.float()).half()
    assert original.item() == 0.0
    assert fused_without_intermediate_rounding.item() == 2.0**-11


def test_stale_mask_fixture_changes_attention_output():
    q = torch.zeros(1, 1, 1, 8)
    k = torch.zeros(1, 1, 4, 8)
    v = torch.zeros_like(k)
    v[..., 2, :] = 12.0  # the newly visible KV entry must matter
    stale = torch.tensor([0.0, 0.0, float("-inf"), float("-inf")]).view(1, 1, 1, 4)
    fresh = torch.tensor([0.0, 0.0, 0.0, float("-inf")]).view(1, 1, 1, 4)
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=stale)
    b = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=fresh)
    assert (a - b).abs().max().item() > 1.0


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["all_equal", "kth_boundary", "fp16_rounding"])
def test_gpu_routing_boundary_matches_aten_moe_output(case):
    """An actual difference FAILS correctness; equal probabilities alone do not pass."""
    from lemoq.kernels._routing_triton import routing_softmax_topk

    logits = torch.zeros(1, 64, device="cuda", dtype=torch.float32)
    if case == "kth_boundary":
        logits[:, :7] = torch.arange(8, 1, -1, device="cuda").float()
    elif case == "fp16_rounding":
        logits = (torch.ones_like(logits) + torch.arange(64, device="cuda") * 1e-6).half()
    ref_w, ref_i = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), 8)
    got_w, got_i = routing_softmax_topk(logits, top_k=8)
    expert_outputs = torch.arange(64, device="cuda", dtype=torch.float32).square()
    ref = (ref_w * expert_outputs[ref_i]).sum(-1)
    got = (got_w * expert_outputs[got_i]).sum(-1)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3), (case, ref_i, got_i, ref, got)


@pytest.mark.gpu
def test_gpu_rope_different_sin_stride_falls_back_bitwise():
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb

    from lemoq.kernels.rope import apply_rotary_pos_emb_fused

    q = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16)
    k = q[:, :1].clone()
    c = torch.randn(1, 1, 128, device="cuda", dtype=q.dtype)
    s = torch.randn(1, 1, 256, device="cuda", dtype=q.dtype)[..., ::2]
    got = apply_rotary_pos_emb_fused(q, k, c, s)
    ref = apply_rotary_pos_emb(q, k, c, s)
    assert all(torch.equal(a, b) for a, b in zip(got, ref))


@pytest.mark.gpu
@pytest.mark.parametrize("filled", [0, 1, 17])
def test_gpu_attention_valid_device_length_boundaries(filled):
    from lemoq.kernels.attn_decode import attn_decode_forward, dense_attention_oracle

    q = torch.randn(1, 2, 1, 128, device="cuda", dtype=torch.float16) * 0.1
    k = torch.randn(1, 2, 17, 128, device="cuda", dtype=q.dtype) * 0.1
    v = torch.randn_like(k)
    n = torch.tensor([filled], device="cuda", dtype=torch.int32)
    got = attn_decode_forward(q, k, v, n, config={"num_splits": 4})
    ref = dense_attention_oracle(q, k, v, filled).to(q.dtype).reshape_as(got)
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-3)
