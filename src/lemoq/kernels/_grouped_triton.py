"""_grouped_triton — grouped GEMM 的 Triton kernel 本体（**只在有 Triton 时导入**）。

## 这个文件为什么单独存在

`triton` 在 Mac 上装不了（没有 CUDA 后端）。把 `@triton.jit` 放在
`moe_grouped.py` 里会让整个模块在 Mac 上 import 失败，于是**离线测试也跑不了** ——
而离线测试正是发现索引/形状错误的地方。所以这里把 kernel 与 host 逻辑分开：

    moc_grouped.py      纯 PyTorch oracle + host 逻辑 + 惰性 import 本模块
    _grouped_triton.py  @triton.jit 的 kernel（本文件）—— 只被 GPU 路径 import

## 为什么是 GEMV 而不是 tl.dot

decode 时 T=1、top_k=K，展开成 K 个 (token, expert) 对。每个对用的是**不同的**
权重矩阵，而 x 是同一个向量：

    out[i, n] = sum_k W[ids[i], n, k] * x[k]

所以这里不存在共享的 A 矩阵 —— 它是一个 **batched GEMV**，不是 GEMM。
用 `tl.dot` 必须把 M 补到 16，那会让每个 program 多读 16/K 倍的权重，
在一个**纯带宽算子**上正好是反优化。这也解释了为什么 cuBLAS 的
batched-GEMV 在这个形状上跑不好（它的 tile 是按 GEMM 设计的）。

于是 kernel 的形状是：`tl.sum(w * x[None, :], axis=1)` ——
广播相乘 + 沿 K 维归约，每个 program 只读它自己那一片专家权重。

## 布局约定

`gate_up` 是 `(E, 2F, H)` 连续张量（`fuse_experts` 的产物，gate 与 up 在 dim=1 拼接）：

    gate 第 n 行 : gate_up[e, n, :]       偏移 e*2F*H + n*H
    up   第 n 行 : gate_up[e, F + n, :]   偏移 e*2F*H + (F+n)*H

**刻意不在 host 侧把 gate/up 切开**：切片要么复制一份 1.6 GB 的权重，
要么让 kernel 处理非连续视图。两者都不必要 —— 只要多传一个 `F` 偏移，
kernel 就能在同一个 buffer 里同时读两段。

## CUDA Graph 约束（**违反会导致 capture 失败或静默算错**）

1. **不得用 `@triton.autotune`。** autotune 会真的跑多个 config 来计时，
   在 capture 期间等于在图里启动一堆计划外的 kernel。
   配置由 host 显式传入，调参在隔离微基准里完成。
2. **循环边界必须与数据无关。** 本文件里所有 `range(...)` 的边界只来自
   `program_id` 与 constexpr，没有一个来自 `load` 的结果。
   （`load` 出来的 expert id 只用在**指针算术**里，不控制控制流。）
3. 首次调用会 JIT 编译 —— 必须在 capture **之前** warm 过。
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _gate_up_act_kernel(
    x_ptr,  # (H,) fp16  —— 本 token 的 hidden state
    w_ptr,  # (E, 2F, H) fp16  —— fuse_experts 的 gate_up
    ids_ptr,  # (K,) int64  —— 本 token 选中的专家 id
    act_ptr,  # (K, F) fp16  —— SPLIT_K == 1 时直接写这里
    part_g_ptr,  # (SPLIT_K, K, F) fp32  —— SPLIT_K > 1 时的 gate 部分和
    part_u_ptr,  # (SPLIT_K, K, F) fp32  —— SPLIT_K > 1 时的 up 部分和
    F: tl.constexpr,  # expert ffn dim（gate 段的行数）
    H: tl.constexpr,  # hidden
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_TOKENS: tl.constexpr,  # top_k，仅用于 SPLIT_K > 1 的部分和寻址
):
    """`act[i, n] = silu(W_gate[e_i, n, :] . x) * (W_up[e_i, n, :] . x)`。

    激活在 epilogue 融合 —— 它不需要额外读任何权重，所以白拿，
    而且省掉一个 (K, F) 的中间张量往返。
    """
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # ★ 唯一的图内"数据依赖"：expert id。它只进指针算术，不进控制流。
    e = tl.load(ids_ptr + slot)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # 输出行（gate/up 共用同一组 n）
    n_mask = n < F

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    rows_per_split = tl.cdiv(H, SPLIT_K)
    k_start = pid_k * rows_per_split
    k_end = tl.minimum(k_start + rows_per_split, H)

    base = w_ptr + e * (2 * F * H)
    for k0 in range(k_start, k_end, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < k_end
        xv = tl.load(x_ptr + k, mask=k_mask, other=0.0).to(tl.float32)
        m2 = n_mask[:, None] & k_mask[None, :]
        g = tl.load(base + n[:, None] * H + k[None, :], mask=m2, other=0.0).to(tl.float32)
        u = tl.load(base + (F + n[:, None]) * H + k[None, :], mask=m2, other=0.0).to(tl.float32)
        acc_g += tl.sum(g * xv[None, :], axis=1)
        acc_u += tl.sum(u * xv[None, :], axis=1)

    if SPLIT_K == 1:
        act = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(act_ptr + slot * F + n, act.to(tl.float16), mask=n_mask)
    else:
        off = (pid_k * K_TOKENS + slot) * F
        tl.store(part_g_ptr + off + n, acc_g, mask=n_mask)
        tl.store(part_u_ptr + off + n, acc_u, mask=n_mask)


@triton.jit
def _act_combine_kernel(
    part_g_ptr,  # (SPLIT_K, K, F) fp32
    part_u_ptr,  # (SPLIT_K, K, F) fp32
    act_ptr,  # (K, F) fp16
    F: tl.constexpr,
    K_TOKENS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """把 SPLIT_K 个部分和相加，再施加 silu 门 —— 只有 SPLIT_K > 1 时才存在。

    元素数只有 K*F（OLMoE: 8 KB 的一半），所以这个 kernel 是纯为了
    "非线性不能跨 K 分片归约"而存在的。
    """
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = K_TOKENS * F
    m = idx < total
    acc_g = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc_g += tl.load(part_g_ptr + s * total + idx, mask=m, other=0.0)
        acc_u += tl.load(part_u_ptr + s * total + idx, mask=m, other=0.0)
    act = acc_g * tl.sigmoid(acc_g) * acc_u
    tl.store(act_ptr + idx, act.to(tl.float16), mask=m)


@triton.jit
def _down_sum_kernel(
    act_ptr,  # (K, F) fp16
    w_ptr,  # (E, H, F) fp16  —— fuse_experts 的 down
    ids_ptr,  # (K,) int64
    wsum_ptr,  # (K,) fp16  —— top_k 的 routing 权重
    out_ptr,  # (H,) fp16  —— SPLIT_K == 1
    part_ptr,  # (SPLIT_K, H) fp32  —— SPLIT_K > 1
    H: tl.constexpr,
    F: tl.constexpr,
    K_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """`out[n] = sum_i top_w[i] * (W_down[e_i, n, :] . act[i, :])`。

    ★ routing 权重在这里乘，而不是像 gather 路径那样先写一个 (K, H) 的 fp16
    中间张量再乘、再 `sum(dim=1)`。省掉两件事：一次 (K, H) 的往返，
    以及 fp16 中间张量的舍入。代价是与 gather 路径**不再逐位相同** ——
    本 kernel 在 fp32 里累加，比 FP16 中间张量更准，这正是 K1 门要拿
    fp64 oracle 比、而不是拿 gather 路径比的原因。

    top_k 维在这里被**完全吃掉**（for slot in range(K_TOKENS) 是静态展开），
    所以输出直接就是 (H,)，不需要再做 `sum(dim=1)`。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < H
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    cols_per_split = tl.cdiv(F, SPLIT_K)
    k_start = pid_k * cols_per_split
    k_end = tl.minimum(k_start + cols_per_split, F)

    for slot in range(K_TOKENS):
        e = tl.load(ids_ptr + slot)
        w = tl.load(wsum_ptr + slot).to(tl.float32)
        base = w_ptr + e * (H * F)
        for k0 in range(k_start, k_end, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            k_mask = k < k_end
            a = tl.load(act_ptr + slot * F + k, mask=k_mask, other=0.0).to(tl.float32)
            wv = tl.load(
                base + n[:, None] * F + k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv * a[None, :], axis=1) * w

    if SPLIT_K == 1:
        tl.store(out_ptr + n, acc.to(tl.float16), mask=n_mask)
    else:
        tl.store(part_ptr + pid_k * H + n, acc, mask=n_mask)


@triton.jit
def _out_reduce_kernel(
    part_ptr,  # (SPLIT_K, H) fp32
    out_ptr,  # (H,) fp16
    H: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """把 SPLIT_K 个输出分片相加。只有 SPLIT_K > 1 时才存在。"""
    n = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = n < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * H + n, mask=m, other=0.0)
    tl.store(out_ptr + n, acc.to(tl.float16), mask=m)
