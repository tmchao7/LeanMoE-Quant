"""moe_static — 固定形状（CUDA Graph 安全）的 MoE decode 前向。

## 为什么需要这个模块

HF 的 `OlmoeSparseMoeBlock.forward`（transformers 4.57.6）对全部 64 个专家做
Python 循环，每次调用 `torch.where(expert_mask[e])`。`torch.where` → `nonzero`，
输出长度**取决于数据**，所以必须先把 count 从 device 拷回 host 才能分配输出张量。

实测证据（EXP-p0-02 的 profile，RTX 5090 / OLMoE / batch=1）：

    单步 Memcpy DtoH = 1024 次 = 64 experts x 16 layers   <- 完全吻合

后果有两层：

1. **CUDA Graph capture 大概率直接失败。** capture 期间不允许 host 同步，
   报 `operation not permitted when stream is capturing`。
   HF 自己在 modeling_olmoe.py 里写了：
   `_can_compile_fullgraph = False  # MoE models don't work with torch.compile
   (torch.where(condition) not supported)` —— torch.compile 与 CUDA Graph
   在这个模式上受同一条约束。
2. **即使侥幸捕获成功，也会静默算错。** 输出缓冲大小在 capture 时被冻结；
   replay 时路由分布不同 → 越界写或漏算。**这类错误不会被异常捕获**，
   只能靠 KL / top-1 门来发现。

所以本模块的目标是把那条数据依赖路径**整体消掉**。

## 怎么消掉

关键观察：decode 时 `top_k` 是**固定**的（OLMoE = 8），batch=1、q_len=1。
所以每 token 恰好有 `top_k` 个 (token, expert) 对，形状完全静态：

    ids   = topk(...)                 # (T*K,)         形状固定
    xr    = x.repeat_interleave(K)    # (T*K, hidden)  形状固定
    gu    = bmm(gate_up[ids], xr)     # (T*K, 2*ffn)   形状固定

`gate_up[ids]` 是**用张量索引**，不是 Python int，因此可以进图。

代价要说清楚：`gate_up[ids]` 是一次真实的 gather 拷贝
（OLMoE fp16：约 1.0 GiB/token 的读 + 写，再加 bmm 读一遍）。
也就是说这条路径**多花约 2 GB/token 的显存流量**换掉 8000 次 kernel launch。
在 launch 占 89.5% 时间的前提下这笔交易是划算的，但它不是终局 ——
真正的解法是 Phase 2 的 grouped GEMM（Triton，kernel 内直接算指针，无 gather 拷贝）。
本模块是**可行性门**用的正确性参照，不是最终性能形态。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "FusedExperts",
    "fuse_experts",
    "static_moe_forward",
    "replace_moe_blocks",
]


@dataclass
class FusedExperts:
    """把 num_experts 个独立 MLP 堆成两个可被张量索引的参数。

    gate_up: (E, 2*ffn, hidden)   —— gate_proj 与 up_proj 拼在 dim=1
    down:    (E, hidden, ffn)
    """

    gate_up: Any
    down: Any

    @property
    def num_experts(self) -> int:
        return int(self.gate_up.shape[0])

    def to(self, device: Any, dtype: Any = None) -> "FusedExperts":
        g = self.gate_up.to(device=device, dtype=dtype or self.gate_up.dtype)
        d = self.down.to(device=device, dtype=dtype or self.down.dtype)
        return FusedExperts(gate_up=g.contiguous(), down=d.contiguous())

    def nbytes(self) -> int:
        return int(self.gate_up.numel() * self.gate_up.element_size()) + int(
            self.down.numel() * self.down.element_size()
        )


def fuse_experts(experts: Any) -> FusedExperts:
    """把 `OlmoeSparseMoeBlock.experts`（nn.ModuleList of OlmoeMLP）堆成张量。

    只在**图外调用一次**。调用方负责在之后释放原来的 ModuleList（见
    `replace_moe_blocks`），否则显存会翻倍。
    """
    import torch

    layers = list(experts)
    if not layers:
        raise ValueError("experts 是空的")
    gate = torch.stack([e.gate_proj.weight.detach() for e in layers], dim=0)
    up = torch.stack([e.up_proj.weight.detach() for e in layers], dim=0)
    down = torch.stack([e.down_proj.weight.detach() for e in layers], dim=0)
    # gate: (E, ffn, hidden) -> cat(dim=1) -> (E, 2*ffn, hidden)
    gate_up = torch.cat([gate, up], dim=1)
    return FusedExperts(gate_up=gate_up.contiguous(), down=down.contiguous())


def static_moe_forward(
    hidden_states: Any,
    router_weight: Any,
    fused: FusedExperts,
    *,
    top_k: int,
    norm_topk_prob: bool = False,
    act_fn: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any]:
    """固定形状的 MoE 前向。返回 (output, router_logits)。

    与 `OlmoeSparseMoeBlock.forward` 数值等价（见 tests/test_moe_static.py 的
    CPU parity 测试），但**全程没有数据依赖的形状**：

        - `torch.topk` 的输出是 (T, K)，K 固定；
        - 没有任何 `torch.where` / `nonzero` / `.item()` / `.cpu()`；
        - 没有 `for expert in range(num_experts)` 这种随数据变化的循环。

    因此可以放进 CUDA Graph。
    """
    import torch
    import torch.nn.functional as F

    if act_fn is None:
        act_fn = F.silu

    shape = tuple(hidden_states.shape)
    hidden = shape[-1]
    tokens = 1
    for s in shape[:-1]:
        tokens *= s
    x = hidden_states.reshape(tokens, hidden)

    logits = F.linear(x, router_weight)  # (T, E)
    weights = F.softmax(logits, dim=-1, dtype=torch.float)
    top_w, top_i = torch.topk(weights, top_k, dim=-1)  # (T, K) —— K 固定
    if norm_topk_prob:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    top_w = top_w.to(x.dtype)

    ids = top_i.reshape(-1)  # (T*K,)
    # 每个 token 的 K 个专家共用同一份 hidden state。repeat_interleave 让
    # `xr[t*K + j]` 正好对应 `ids[t*K + j]`，配对关系不会错位。
    xr = x.repeat_interleave(top_k, dim=0)  # (T*K, hidden)

    gate_up = fused.gate_up[ids]  # (T*K, 2*ffn, hidden) —— 张量索引，可进图
    gu = torch.bmm(gate_up, xr.unsqueeze(-1)).squeeze(-1)  # (T*K, 2*ffn)
    gate, up = gu.chunk(2, dim=-1)
    act = act_fn(gate) * up  # (T*K, ffn)

    down = fused.down[ids]  # (T*K, hidden, ffn)
    y = torch.bmm(down, act.unsqueeze(-1)).squeeze(-1)  # (T*K, hidden)
    y = y * top_w.reshape(-1, 1)

    out = y.reshape(tokens, top_k, hidden).sum(dim=1)  # (T, hidden)
    return out.reshape(*shape), logits


class StaticShapedMoE:
    """`OlmoeSparseMoeBlock` 的 drop-in 替代，使用固定形状前向。

    刻意**不是** nn.Module 的子类：它复用原 block 的 `gate` 权重与配置，
    并把 experts 换成 fused 张量，这样显存不翻倍。
    """

    def __init__(self, block: Any, *, device: Any = None, dtype: Any = None) -> None:
        self.top_k = int(block.top_k)
        self.norm_topk_prob = bool(getattr(block, "norm_topk_prob", False))
        self.act_fn = getattr(block, "act_fn", None)
        self.router_weight = block.gate.weight.detach()
        self.fused = fuse_experts(block.experts)
        if device is not None or dtype is not None:
            self.fused = self.fused.to(device or self.fused.gate_up.device, dtype=dtype)

    def __call__(self, hidden_states: Any) -> tuple[Any, Any]:
        return static_moe_forward(
            hidden_states,
            self.router_weight,
            self.fused,
            top_k=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
            act_fn=self.act_fn,
        )


def replace_moe_blocks(model: Any, *, verbose: bool = False) -> int:
    """把模型里每个 `OlmoeSparseMoeBlock` 就地换成固定形状版本。

    返回被替换的 block 数。替换后原来的 `experts` ModuleList 会被释放，
    所以**显存不会翻倍**（峰值短暂 +1 层的量）。

    ★ 必须在 CUDA Graph capture **之前**调用，且之后不要再改权重。
    """
    import gc

    import torch

    replaced = 0
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "experts"):
            continue
        if not hasattr(block, "gate") or not hasattr(block, "top_k"):
            continue  # 不是 sparse MoE（例如 dense 层）

        static = StaticShapedMoE(block)

        # 释放原始 experts，避免显存翻倍
        del block.experts
        block._static_experts = static.fused
        block._static_router = static.router_weight
        block._static_top_k = static.top_k
        block._static_norm = static.norm_topk_prob
        block._static_act = static.act_fn

        def _forward(hidden_states, _b=block):
            return static_moe_forward(
                hidden_states,
                _b._static_router,
                _b._static_experts,
                top_k=_b._static_top_k,
                norm_topk_prob=_b._static_norm,
                act_fn=_b._static_act,
            )

        block.forward = _forward
        replaced += 1
        if verbose:
            print("  replaced MoE block %d" % replaced)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced
