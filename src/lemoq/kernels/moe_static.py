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
    "VALID_MOE_IMPLS",
    "fuse_experts",
    "static_moe_forward",
    "replace_moe_blocks",
]

# decode 路径的两种实现（**同一个假设的不同实现，不是两个假设**）：
#   "gather"  固定形状 gather + bmm —— 可行性参照；每 token 多 3.22 GB 流量
#   "grouped" Triton grouped GEMV —— kernel 内算指针，无 gather 拷贝（moe_grouped）
# prefill（T > 1）与 impl 无关，永远走 `_prefill_forward`。
VALID_MOE_IMPLS = ("gather", "grouped")


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

    def to(self, device: Any = None, dtype: Any = None) -> "FusedExperts":
        """与 `Tensor.to` 同样的调用习惯：`to(torch.float32)` 也是合法写法。

        ★ 为什么要做这个兼容：`Tensor.to(x)` 在 x 是 dtype 时是**类型转换**，
        而这里的第一参数是 device。于是 `fused.to(torch.float32)` 会抛出
        "to() received an invalid combination of arguments" —— 一个与真实
        原因毫无关系的报错（实测踩到）。宁可在这里识别 dtype。
        """
        import torch

        if isinstance(device, torch.dtype):
            device, dtype = None, device
        dtype = dtype or self.gate_up.dtype
        if device is None:
            device = self.gate_up.device
        g = self.gate_up.to(device=device, dtype=dtype)
        d = self.down.to(device=device, dtype=dtype)
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
    impl: str = "gather",
) -> tuple[Any, Any]:
    """MoE 前向。返回 (output, router_logits)。

    ## 两条路径，按 token 数分派

        T == 1  -> `_decode_forward`（impl="gather"）或 grouped kernel
                   （impl="grouped"）：**固定形状，可进 CUDA Graph**
        T >  1  -> `_prefill_forward`：逐专家循环，**只用于 eager**，与 impl 无关

    ★ `impl` **只影响 decode**。prefill 永远走 `_prefill_forward`，因为 grouped
    路径的 `for slot in range(top_k)` 是"每 token 只读 top-k 个专家"的写法 ——
    在 T=1024 时它会对每个 token 重复读一遍专家权重，正是下面事故 2 的错误。
    这一点有测试钉住（`test_prefill_never_uses_the_grouped_kernel`）。

    ## ★ 为什么必须分开（两次实测事故）

    **事故 1：分块 gather 在 prefill 时 OOM。**
    最直接的写法是 `gate_up[ids]`，`ids` 有 `T*K` 个元素，会物化
    `(T*K, 2*ffn, hidden)`。decode 时 T=1（8 个专家，67 MiB）没事；prefill 时
    T=1024 → (8192, 2048, 2048) fp16 = **64 GiB**，报错原文就是这个数字。

    **事故 2：分块之后 prefill 慢 20 倍。**
    按 token 分块确实压住了显存，但算法本身是错的 —— 它对**每个 (token, expert)
    对**都读一遍专家权重：

        HF 逐专家循环 : 64 专家 x 8.4 MB          =   537 MB/层
        分块 gather   : 8192 对 x 8.4 MB          = 68000 MB/层   (多读 128 倍)

    实测单层 prefill：HF 7.60 ms vs 分块 155.77 ms（加大预算只降到 152 ms，
    说明瓶颈是重复读权重，不是块数）。

    所以 prefill 必须回到"每个专家的权重只读一次、作用于所有路由到它的 token"。
    那条路径需要数据依赖的索引（`nonzero`），**但它不进图**，因此安全。
    真正的固定形状约束只作用于 decode。
    """
    import torch.nn.functional as F

    if impl not in VALID_MOE_IMPLS:
        raise ValueError("unknown impl %r; expected one of %s" % (impl, VALID_MOE_IMPLS))

    if act_fn is None:
        act_fn = F.silu

    shape = tuple(hidden_states.shape)
    hidden = shape[-1]
    tokens = 1
    for s in shape[:-1]:
        tokens *= s
    x = hidden_states.reshape(tokens, hidden)

    if tokens == 1:
        if impl == "grouped":
            # 惰性 import：grouped 需要 triton，而 Mac 上装不了。只在真正走到
            # decode 时才 import，这样纯 CPU 的 prefill 路径与 gather 路径
            # 在没有 triton 的机器上依然可用。
            from lemoq.kernels.moe_grouped import static_moe_forward_grouped

            return static_moe_forward_grouped(
                hidden_states,
                router_weight,
                fused,
                top_k=top_k,
                norm_topk_prob=norm_topk_prob,
                act_fn=act_fn,
            )
        return _decode_forward(
            x,
            router_weight,
            fused,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            act_fn=act_fn,
            out_shape=shape,
            hidden=hidden,
        )
    # ★ T > 1 与 impl 无关：永远走逐专家循环。grouped kernel 是"每 token 读
    #   top-k 个专家"的写法，在 T=1024 上会对每个 token 重读一遍权重（128 倍）。
    out, logits = _prefill_forward(
        x,
        router_weight,
        fused,
        top_k=top_k,
        norm_topk_prob=norm_topk_prob,
        act_fn=act_fn,
    )
    return out.reshape(*shape), logits


def _route(x: Any, router_weight: Any, *, top_k: int, norm_topk_prob: bool):
    """公共路由计算：返回 (logits, top_w, top_i)。两种形状约定下的输出都固定。"""
    import torch
    import torch.nn.functional as F

    logits = F.linear(x, router_weight)  # (T, E)
    weights = F.softmax(logits, dim=-1, dtype=torch.float)
    top_w, top_i = torch.topk(weights, top_k, dim=-1)  # (T, K) —— K 固定
    if norm_topk_prob:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    return logits, top_w.to(x.dtype), top_i


def _decode_forward(
    x: Any,
    router_weight: Any,
    fused: FusedExperts,
    *,
    top_k: int,
    norm_topk_prob: bool,
    act_fn: Callable[[Any], Any],
    out_shape: tuple,
    hidden: int,
) -> tuple[Any, Any]:
    """**固定形状**路径 —— 唯一允许进 CUDA Graph 的一条。

    全程没有数据依赖的形状：

        - `torch.topk` 的输出是 (T, K)，K 固定；
        - 没有任何 `torch.where` / `nonzero` / `.item()` / `.cpu()`；
        - 没有随模型规模变化的 Python 循环。

    T 恒为 1，所以 gather 的规模就是 K 个专家（OLMoE 下 67 MiB），与 ctx 无关。
    """
    import torch

    logits, top_w, top_i = _route(x, router_weight, top_k=top_k, norm_topk_prob=norm_topk_prob)
    tokens = int(x.shape[0])
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
    return out.reshape(*out_shape), logits


def _prefill_forward(
    x: Any,
    router_weight: Any,
    fused: FusedExperts,
    *,
    top_k: int,
    norm_topk_prob: bool,
    act_fn: Callable[[Any], Any],
) -> tuple[Any, Any]:
    """**prefill 专用**路径：逐专家循环，每个专家的权重只读一次。

    ⚠️ 这条路径**含数据依赖的算子**（`nonzero`），因此**绝不能进 CUDA Graph**。
    它只应在 T > 1（prefill）时被调用，而 prefill 是 eager 的。
    下面静态源码测试会断言这些算子只出现在本函数里。
    """
    import torch

    logits, top_w, top_i = _route(x, router_weight, top_k=top_k, norm_topk_prob=norm_topk_prob)
    out = torch.zeros_like(x)

    for e in range(fused.num_experts):  # trip count = num_experts，与数据无关
        selected = top_i == e  # (T, K) bool
        # 数据依赖：这个专家被哪些 token 选中。仅在 eager prefill 路径出现。
        tok, slot = selected.nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w = top_w[tok, slot].unsqueeze(-1)  # (n, 1)
        xe = x[tok]  # (n, hidden)
        gu = xe @ fused.gate_up[e].t()  # (n, 2*ffn)
        gate, up = gu.chunk(2, dim=-1)
        act = act_fn(gate) * up
        y = act @ fused.down[e].t()  # (n, hidden)
        out.index_add_(0, tok, y * w)

    return out, logits


class StaticShapedMoE:
    """`OlmoeSparseMoeBlock` 的 drop-in 替代，使用固定形状前向。

    刻意**不是** nn.Module 的子类：它复用原 block 的 `gate` 权重与配置，
    并把 experts 换成 fused 张量，这样显存不翻倍。
    """

    def __init__(
        self,
        block: Any,
        *,
        device: Any = None,
        dtype: Any = None,
        impl: str = "gather",
    ) -> None:
        if impl not in VALID_MOE_IMPLS:
            raise ValueError("unknown impl %r; expected one of %s" % (impl, VALID_MOE_IMPLS))
        self.impl = impl
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
            impl=self.impl,
        )


def replace_moe_blocks(model: Any, *, verbose: bool = False, impl: str = "gather") -> int:
    """把模型里每个 `OlmoeSparseMoeBlock` 就地换成固定形状版本。

    返回被替换的 block 数。替换后原来的 `experts` ModuleList 会被释放，
    所以**显存不会翻倍**（峰值短暂 +1 层的量）。

    `impl="grouped"` 只在 decode（T=1）生效；prefill 仍然走逐专家循环，
    原因见 `static_moe_forward` 的说明。

    ★ 必须在 CUDA Graph capture **之前**调用，且之后不要再改权重。
    """
    import gc

    import torch

    if impl not in VALID_MOE_IMPLS:
        raise ValueError("unknown impl %r; expected one of %s" % (impl, VALID_MOE_IMPLS))

    replaced = 0
    for layer in model.model.layers:
        block = getattr(layer, "mlp", None)
        if block is None or not hasattr(block, "experts"):
            continue
        if not hasattr(block, "gate") or not hasattr(block, "top_k"):
            continue  # 不是 sparse MoE（例如 dense 层）

        static = StaticShapedMoE(block, impl=impl)

        # 释放原始 experts，避免显存翻倍
        del block.experts
        block._static_experts = static.fused
        block._static_router = static.router_weight
        block._static_top_k = static.top_k
        block._static_norm = static.norm_topk_prob
        block._static_act = static.act_fn
        block._static_impl = static.impl

        def _forward(hidden_states, _b=block):
            return static_moe_forward(
                hidden_states,
                _b._static_router,
                _b._static_experts,
                top_k=_b._static_top_k,
                norm_topk_prob=_b._static_norm,
                act_fn=_b._static_act,
                impl=_b._static_impl,
            )

        block.forward = _forward
        replaced += 1
        if verbose:
            print("  replaced MoE block %d" % replaced)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return replaced
