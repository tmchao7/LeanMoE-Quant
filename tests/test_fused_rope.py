"""test_fused_rope — fused rope 候选的契约（回退矩阵 + GPU parity）。

CPU 上钉：env 默认关、签名与 HF 一致、无 CUDA/形状不满足时**逐位退回 HF**、
call site 源码分支。GPU 上钉 K1（fused vs HF ≤ PARITY_GATE，见文件尾）。
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lemoq.kernels import rope  # noqa: E402


def _hf_apply(q, k, cos, sin, **kw):
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb

    return apply_rotary_pos_emb(q, k, cos, sin, **kw)


def _inputs(batch=1, heads=16, kv=4, seq=1, d=128, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(batch, heads, seq, d, generator=g).to(dtype=dtype, device=device)
    k = torch.randn(batch, kv, seq, d, generator=g).to(dtype=dtype, device=device)
    cos = torch.randn(batch, seq, d, generator=g).to(dtype=dtype, device=device)
    sin = torch.randn(batch, seq, d, generator=g).to(dtype=dtype, device=device)
    return q, k, cos, sin


# ---------------------------------------------------------------------------
# 开关与签名
# ---------------------------------------------------------------------------
def test_env_default_off_and_honors_flag(monkeypatch):
    monkeypatch.delenv("LEMOQ_FUSED_ROPE", raising=False)
    assert rope._fused_rope_enabled() is False
    monkeypatch.setenv("LEMOQ_FUSED_ROPE", "1")
    assert rope._fused_rope_enabled() is True


def test_signature_matches_hf():
    ours = inspect.signature(rope.apply_rotary_pos_emb_fused)
    from transformers.models.olmoe.modeling_olmoe import apply_rotary_pos_emb

    hf = inspect.signature(apply_rotary_pos_emb)
    assert list(ours.parameters) == list(hf.parameters)
    for name in ours.parameters:
        assert ours.parameters[name].default == hf.parameters[name].default


def test_attn_decode_call_site_branches_on_env():
    from lemoq.kernels.attn_decode import _decode_forward_impl

    src = inspect.getsource(_decode_forward_impl)
    assert "_fused_rope_enabled()" in src
    assert "apply_rotary_pos_emb_fused(query_states, key_states, cos, sin)" in src
    # else 分支必须保留 HF 原调用（默认路径）
    assert "apply_rotary_pos_emb(query_states, key_states, cos, sin)" in src


# ---------------------------------------------------------------------------
# 回退矩阵（Mac / 无 CUDA：必须与 HF 逐位相同）
# ---------------------------------------------------------------------------
def test_cpu_fallback_is_bitwise_hf():
    q, k, cos, sin = _inputs()
    got_q, got_k = rope.apply_rotary_pos_emb_fused(q, k, cos, sin)
    want_q, want_k = _hf_apply(q, k, cos, sin)
    assert torch.equal(got_q, want_q) and torch.equal(got_k, want_k)


def test_shape_guard_rejects_non_decode_shapes():
    # CPU ⇒ 守卫第一步就 False；decode 形状之外同样 False
    q, k, cos, sin = _inputs()
    assert rope._decode_shape_guard(q, k, q, q) is False
    q2, k2, cos2, sin2 = _inputs(batch=2)
    assert rope._decode_shape_guard(q2, k2, cos2, sin2) is False
    # cos 经 unsqueeze 后必须是 (1,1,1,D)：L=2 的 prefill 形状不满足
    q3, k3, cos3, sin3 = _inputs(seq=2)
    assert rope._decode_shape_guard(q3, k3, cos3.unsqueeze(1), sin3.unsqueeze(1)) is False


def test_odd_head_dim_falls_back_to_hf():
    q, k, cos, sin = _inputs(d=64)
    q_odd = torch.randn(1, 16, 1, 63)
    k_odd = torch.randn(1, 4, 1, 63)
    cos_odd = torch.randn(1, 1, 63)
    sin_odd = torch.randn(1, 1, 63)
    got = rope.apply_rotary_pos_emb_fused(q_odd, k_odd, cos_odd, sin_odd)
    want = _hf_apply(q_odd, k_odd, cos_odd, sin_odd)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    # 对照：正常偶数维也走回退（CPU）但结果仍与 HF 相同
    got2 = rope.apply_rotary_pos_emb_fused(q, k, cos, sin)
    want2 = _hf_apply(q, k, cos, sin)
    assert torch.equal(got2[0], want2[0])


def test_warm_is_noop_without_triton():
    if rope.fused_rope_available():
        pytest.skip("本机有 triton，warm 走真实 launch（GPU 测试覆盖）")
    rope.warm()  # 不抛即通过


# ---------------------------------------------------------------------------
# GPU：K1 parity（fused vs HF，PARITY_GATE 同口径）
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("d,dtype", [(128, torch.float16), (64, torch.float32)])
def test_gpu_parity_fused_vs_hf(d, dtype):
    if not rope.fused_rope_available():
        pytest.skip("triton 不可用")
    device = "cuda"
    q, k, cos, sin = _inputs(heads=16, kv=4, d=d, dtype=dtype, device=device, seed=3)
    got_q, got_k = rope.apply_rotary_pos_emb_fused(q, k, cos, sin)
    want_q, want_k = _hf_apply(q, k, cos, sin)

    # fp64 oracle（同一数学），两路分别与它比
    def oracle(x, c, s):
        xf = x.double()
        x1, x2 = xf.chunk(2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return (xf * c.double().unsqueeze(1) + rot * s.double().unsqueeze(1)).to(dtype)

    tol = {
        "tol_atol": 1e-3,
        "tol_rtol": 1e-3,
        "tol_slack": 1.5,
    }  # 与 attn_decode.PARITY_GATE 同口径
    for x, got, want in ((q, got_q, want_q), (k, got_k, want_k)):
        err_k = (got - oracle(x, cos, sin)).abs().max().item()
        err_b = (want - oracle(x, cos, sin)).abs().max().item()
        assert err_k <= err_b * tol["tol_slack"] + 1e-12, "err_kernel=%g err_baseline=%g" % (
            err_k,
            err_b,
        )
    assert torch.allclose(got_q, want_q, atol=tol["tol_atol"], rtol=tol["tol_rtol"])
    assert torch.allclose(got_k, want_k, atol=tol["tol_atol"], rtol=tol["tol_rtol"])


@pytest.mark.gpu
def test_gpu_non_decode_shape_falls_back_bitwise():
    if not rope.fused_rope_available():
        pytest.skip("triton 不可用")
    q, k, cos, sin = _inputs(batch=2, device="cuda", dtype=torch.float16)
    got = rope.apply_rotary_pos_emb_fused(q, k, cos, sin)
    want = _hf_apply(q, k, cos, sin)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


@pytest.mark.gpu
def test_gpu_bf16_falls_back_bitwise():
    """bf16 不在 kernel 臂支持列表（1 ulp 过不了 K1）——必须逐位退回 HF。"""
    if not rope.fused_rope_available():
        pytest.skip("triton 不可用")
    q, k, cos, sin = _inputs(device="cuda", dtype=torch.bfloat16)
    got = rope.apply_rotary_pos_emb_fused(q, k, cos, sin)
    want = _hf_apply(q, k, cos, sin)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


# ---------------------------------------------------------------------------
# review 2026-10-22（P4/P5）：position_ids 守卫 + 死参数
# ---------------------------------------------------------------------------


def test_position_ids_falls_back_to_hf_before_guard(monkeypatch):
    """P4: position_ids 非 None ⇒ 直接回退 HF（不进 kernel 路径，不静默算错）。"""
    calls = []

    def rec(q, k, cos, sin, **kw):
        calls.append(kw.get("position_ids"))
        return q, k

    def guard_bomb(*a, **k):
        raise AssertionError("position_ids 非 None 时不应走到 kernel 守卫")

    monkeypatch.setattr(rope, "_hf_apply", rec)
    monkeypatch.setattr(rope, "_decode_shape_guard", guard_bomb)
    q, k, cos, sin = _inputs(d=64, dtype=torch.float32)
    pos = torch.zeros(1, 1, dtype=torch.long)
    rope.apply_rotary_pos_emb_fused(q, k, cos, sin, position_ids=pos)
    assert calls and calls[0] is pos


def test_kernel_has_no_dead_params():
    """P5: `_rope_one_kernel` 签名里不留未使用的参数（曾有 H 传入但从不用）。"""
    from pathlib import Path

    src = Path(rope.__file__).read_text(encoding="utf-8")
    start = src.index("def _rope_one_kernel(")
    end = src.index("):", start)
    params = [p.strip().rstrip(":").split(":")[0].strip() for p in src[start:end].split(",")]
    params = [p for p in params if p and "(" not in p]
    assert "H" not in params, params
    # 调用处也不能再传死参数（launch 参数个数 = 签名参数个数）
    assert "            heads,\n            d," not in src
