"""test_routing_fusion — 融合路由 kernel 的语义与 parity。

分层：
    - **CPU**（Mac 可跑）：钉死"目标语义"= fp32 softmax + 降序 top-K +（可选）归一化，
      numpy oracle 与 torch 参考互证；无 CUDA 时**直接调用 kernel** 必须显式报错
      （不许静默降级）。注意：`moe_static._route` 的候选分支另有 rope 同款的
      **可用性回退**（CPU 张量 ⇒ ATen 原路，2026-10-24）—— 那是调用方的回退，
      不是 kernel 静默降级，两层都有测试钉住。
    - **GPU**（`-m gpu`，服务器）：triton kernel vs ATen 参考的 1e-3 parity（无平局样本
      ids 精确相等）、`_route` env 开关的端到端一致性、fp16 logits 与非 2 幂 E。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lemoq.kernels.moe_static import _route

GATE = 1e-3  # 与 moe_grouped.PARITY_GATE 同口径


def _numpy_route(logits: np.ndarray, *, top_k: int, norm: bool):
    """目标语义的 numpy oracle（fp32 softmax → 降序 top-K → 可选归一化）。"""
    z = logits.astype(np.float32)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    w = e / e.sum(axis=-1, keepdims=True)
    order = np.argsort(-w, axis=-1, kind="stable")
    idx = order[..., :top_k]
    top_w = np.take_along_axis(w, idx, axis=-1)
    if norm:
        top_w = top_w / top_w.sum(axis=-1, keepdims=True)
    return top_w, idx


def test_numpy_oracle_matches_torch_reference():
    """oracle 与 torch 参考一致 ⇒ '目标语义'本身没有歧义（CPU 钉死）。"""
    rng = np.random.default_rng(2026)
    for e, k, t in [(64, 8, 1), (12, 3, 5), (8, 8, 2), (48, 4, 3)]:
        logits = rng.standard_normal((t, e)).astype(np.float32)
        # 连续随机值 ⇒ 无平局，ids 允许逐位比较
        ref_w, ref_i = _numpy_route(logits, top_k=k, norm=False)
        torch_logits = torch.from_numpy(logits)
        w_t, i_t = torch.topk(torch.softmax(torch_logits, dim=-1, dtype=torch.float32), k, dim=-1)
        assert np.array_equal(ref_i, i_t.numpy()), (e, k, t)
        assert np.allclose(ref_w, w_t.numpy(), atol=1e-6, rtol=0), (e, k, t)
        # norm 分支
        rn_w, _ = _numpy_route(logits, top_k=k, norm=True)
        tn_w = w_t / w_t.sum(dim=-1, keepdim=True)
        assert np.allclose(rn_w, tn_w.numpy(), atol=1e-6, rtol=0)


def test_kernel_requires_cuda_without_silent_fallback():
    if torch.cuda.is_available():
        pytest.skip("本机有 CUDA —— 报错路径在无 GPU 机上才有意义")
    from lemoq.kernels._routing_triton import routing_softmax_topk

    with pytest.raises(RuntimeError, match="CUDA"):
        routing_softmax_topk(torch.zeros(1, 64), top_k=8)


# ---------------------------------------------------------------------------
# GPU：kernel 与 _route 开关的 parity（服务器 `-m gpu` 跑）
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("e,k", [(64, 8), (12, 3), (8, 8)])
def test_kernel_matches_aten_on_fp32_logits(norm, e, k):
    """fp32 logits、无平局：ids 逐位相等，weights 收敛到 ATen。"""
    from lemoq.kernels._routing_triton import routing_softmax_topk

    g = torch.Generator(device="cpu").manual_seed(20261010)
    logits = torch.randn(7, e, dtype=torch.float32, generator=g).cuda()
    ref_w, ref_i = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), k, dim=-1)
    if norm:
        ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
    got_w, got_i = routing_softmax_topk(logits, top_k=k, norm_topk_prob=norm)
    assert torch.equal(got_i, ref_i), (e, k, norm, got_i, ref_i)
    assert torch.allclose(got_w, ref_w, atol=GATE, rtol=0), (got_w - ref_w).abs().max()


@pytest.mark.gpu
def test_kernel_handles_fp16_logits_and_noncontiguous():
    from lemoq.kernels._routing_triton import routing_softmax_topk

    g = torch.Generator(device="cpu").manual_seed(7)
    logits = torch.randn(4, 64, dtype=torch.float32, generator=g).cuda().half()
    ref_w, ref_i = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), 8, dim=-1)
    got_w, got_i = routing_softmax_topk(logits, top_k=8)
    # fp16 输入可能出现等值平局：比集合与权重，不比顺序
    assert torch.equal(got_i.sort(dim=-1).values, ref_i.sort(dim=-1).values)
    assert torch.allclose(got_w, ref_w, atol=GATE, rtol=0)
    # 非连续
    noncontig = logits.transpose(0, 1).contiguous().transpose(0, 1)
    assert not noncontig.is_contiguous()
    w2, i2 = routing_softmax_topk(noncontig, top_k=8)
    assert torch.equal(i2, got_i)
    assert torch.allclose(w2, got_w, atol=0, rtol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("norm", [False, True])
def test_route_env_switch_matches_aten(norm, monkeypatch):
    """`LEMOQ_FUSED_ROUTE=1` 的 `_route` 输出与 ATen 路径一致（端到端含 F.linear）。"""
    x = torch.randn(1, 2048, dtype=torch.float16, device="cuda")
    rw = torch.randn(64, 2048, dtype=torch.float16, device="cuda")
    lg0, tw0, ti0 = _route(x, rw, top_k=8, norm_topk_prob=norm)
    monkeypatch.setenv("LEMOQ_FUSED_ROUTE", "1")
    lg1, tw1, ti1 = _route(x, rw, top_k=8, norm_topk_prob=norm)
    assert torch.equal(lg1, lg0), "logits 必须与 ATen 逐位一致（同一 F.linear）"
    assert torch.equal(ti1.sort(-1).values, ti0.sort(-1).values)
    assert torch.allclose(tw1.float(), tw0.float(), atol=GATE, rtol=0)
    # dtype 契约：top_w 跟 x 的 dtype（fp16）
    assert tw1.dtype == torch.float16 and ti1.dtype == torch.int64


@pytest.mark.gpu
def test_route_default_matches_switches(monkeypatch):
    """无 env ⇒ 生效值 = `FUSION_DEFAULTS`（翻默认后此测试不需要改动，2026-10-24）。"""
    monkeypatch.delenv("LEMOQ_FUSED_ROUTE", raising=False)
    from lemoq.kernels import moe_static
    from lemoq.kernels.fusion_switches import FUSION_DEFAULTS

    assert moe_static._fused_route_enabled() is FUSION_DEFAULTS["fused_route"]


def test_route_cpu_fallback_bit_exact_when_env_on(monkeypatch):
    """翻默认后的可用性回退（2026-10-24）：CPU 张量上即使开关开也走 ATen 原路，
    且与开关关**逐位相同**（rope `_decode_shape_guard` 同款）。kernel 本身的显式
    报错不放松 —— 直接调 `routing_softmax_topk` 仍 raise（见上面的无静默回退测试）。"""
    monkeypatch.delenv("LEMOQ_FUSED_ROUTE", raising=False)
    x = torch.randn(3, 64)
    rw = torch.randn(64, 64)
    lg0, tw0, ti0 = _route(x, rw, top_k=8, norm_topk_prob=True)
    monkeypatch.setenv("LEMOQ_FUSED_ROUTE", "1")
    lg1, tw1, ti1 = _route(x, rw, top_k=8, norm_topk_prob=True)
    assert torch.equal(lg1, lg0)
    assert torch.equal(ti1, ti0)
    assert torch.equal(tw1, tw0)
