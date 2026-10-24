"""routing_softmax_topk — 把 MoE 路由的 softmax + top-K 融成一个 triton kernel。

动机（2026-10-10 `tools/profile_graph` 实测，图内最佳路径 wall 1.94 ms）：
`moe_static._route` 的 `F.softmax(fp32) → torch.topk(sorted)` 在 decode 里逐层打
**3 个 ATen 小 kernel**（softmax_warp ×16 + gatherTopK ×16 + bitonicSortKV ×16），
合计 **0.110 ms/step ≈ gpu_busy 的 5.9%** —— 最大的非权重、非注意力、可融合桶。

实现要点：
    - 每 token 一个 triton program（grid = T），`E`/`K`/`NORM` 全是 `tl.constexpr`
      ⇒ 形状静态、无 autotune ⇒ **CUDA Graph capture 安全**；
    - softmax 按与 ATen 相同的 fp32 语义（max-subtract + exp-sum）；
    - top-K 用 **K 轮 argmax + one-hot 置 -inf**（O(K·E)，E/K constexpr 时被展开），
      **不依赖 `tl.sort`**；平局时 argmax 取最低 lane 序号。若平局跨越 K 边界，
      可能与 torch.topk 选择不同的专家，等权不能推出专家输出相同。候选尚待独立门验证；
    - `E` 非 2 的幂时用 `E_POW` 装载 + mask（OLMoE 是 64，但测试覆盖非幂）。

    **默认启用（2026-10-24 采纳）**：`_route` 默认走这里，`LEMOQ_FUSED_ROUTE=0` 回 ATen（见
    `moe_static._route`）。这是**候选**路径 —— 采纳需要重测 grouped 路径的全部
    已录格（P2d/P2j 同款流程），不在本 feature 范围内。
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["routing_softmax_topk"]


def _require_cuda_triton():
    if not torch.cuda.is_available():
        raise RuntimeError("routing_softmax_topk 需要 CUDA（只在 LEMOQ_FUSED_ROUTE=1 时应被调用）")
    import triton  # noqa: PLC0415

    return triton


def _triton_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _routing_softmax_topk_kernel(
        logits_ptr,
        top_w_ptr,
        top_i_ptr,
        stride_logits_t,
        E: tl.constexpr,
        E_POW: tl.constexpr,
        K: tl.constexpr,
        K_POW: tl.constexpr,
        NORM: tl.constexpr,
    ):
        """logits (T, E) → top_w (T, K) fp32 + top_i (T, K) int64。"""
        t = tl.program_id(0)
        lanes = tl.arange(0, E_POW)
        offs = t * stride_logits_t + lanes
        logit = tl.load(logits_ptr + offs, mask=lanes < E).to(tl.float32)
        # 无效 lane 置 -inf：max 不受污染、exp(-inf-m)=0 不进分母
        logit = tl.where(lanes < E, logit, float("-inf"))
        # ---- fp32 softmax（与 ATen 同语义：max-subtract + exp-sum）----
        m = tl.max(logit, axis=0)
        p = tl.exp(logit - m)
        p = p / tl.sum(p, axis=0)
        p = tl.where(lanes < E, p, -1.0e30)
        # ---- K 轮 argmax：选一个、记分、置 -inf ----
        cur = p
        k_lanes = tl.arange(0, K_POW)
        k_valid = k_lanes < K
        local_w = tl.zeros([K_POW], tl.float32)
        for k in range(K):
            kk = tl.argmax(cur, axis=0)
            val = tl.sum(cur * (lanes == kk))  # one-hot 抽取当前 lane 值
            local_w = tl.where(k_lanes == k, val, local_w)
            tl.store(top_i_ptr + t * K + k, kk.to(tl.int64))
            cur = tl.where(lanes == kk, -1.0e30, cur)
        if NORM:
            local_w = local_w / tl.sum(tl.where(k_valid, local_w, 0.0))
        tl.store(top_w_ptr + t * K + k_lanes, local_w, mask=k_valid)

    return _routing_softmax_topk_kernel


_KERNEL = None


def routing_softmax_topk(
    logits: Any, *, top_k: int, norm_topk_prob: bool = False
) -> tuple[Any, Any]:
    """`(T, E) logits` → `(top_w (T, K) fp32, top_i (T, K) int64)`。

    语义对齐 `moe_static._route` 的后半段：fp32 softmax → 降序 top-K →（可选）按
    top-K 权重和归一化。**不含** router 线性层（`F.linear` 仍走 ATen）。
    """
    global _KERNEL
    if not torch.is_tensor(logits):
        raise TypeError("logits 必须为张量")
    if logits.ndim != 2:
        raise ValueError("logits 必须是 (T, E)，得到 shape %s" % (tuple(logits.shape),))
    t_tokens, e = int(logits.shape[0]), int(logits.shape[1])
    if type(top_k) is not int or not (0 < top_k <= e):
        raise ValueError("top_k=%r 必须是 (0, %d] 内的整数" % (top_k, e))
    if not logits.is_floating_point():
        raise ValueError("logits 必须为浮点张量")
    contiguous = logits if logits.is_contiguous() else logits.contiguous()
    top_w = torch.empty((t_tokens, top_k), dtype=torch.float32, device=logits.device)
    top_i = torch.empty((t_tokens, top_k), dtype=torch.int64, device=logits.device)
    if t_tokens == 0:
        return top_w, top_i  # no CUDA initialization or launch for empty input
    triton = _require_cuda_triton()
    if not logits.is_cuda:
        raise RuntimeError("routing_softmax_topk 需要 CUDA 张量")
    if _KERNEL is None:
        _KERNEL = _triton_kernel()
    e_pow = triton.next_power_of_2(e)
    k_pow = triton.next_power_of_2(top_k)
    _KERNEL[(t_tokens,)](
        contiguous,
        top_w,
        top_i,
        contiguous.stride(0),
        E=e,
        E_POW=e_pow,
        K=top_k,
        K_POW=k_pow,
        NORM=bool(norm_topk_prob),
        num_warps=1,
    )
    return top_w, top_i
