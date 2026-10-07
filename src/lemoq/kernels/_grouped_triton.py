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


# ===========================================================================
# ★ 量化版（P1：量化 × 可 capture 的 MoE 路径）—— 8-bit per-group 对称 RTN
#
# 与上面两个 kernel 的唯一区别：权重不是 fp16，而是
#     q_ptr   int8 codes，布局与 fp16 权重**逐元素对应**（(E, 2F, H) / (E, H, F) 连续）
#     s_ptr   fp16 per-group scale，(E, 2F, H//GROUP) / (E, H, F//GROUP)
# 于是 kernel 在读权重时当场做 `code * scale`，**不物化 fp16 权重**。
# 这正是 E7 要的：把 experts 的 HBM 流量按位宽砍半，而不是"先反量化再算"。
#
# 三条硬约束（host 侧 `quant_launch_plan` 断言，违反直接报错而不是静默算错）：
#   1. 输入维必须被 group_size 整除（不补零）—— OLMoE 的 2048/1024 对 128 整除；
#   2. `BLOCK_K` 必须整除 group_size：这样一个 k 块**不会跨组边界**，
#      整块的 scale 就是同一个标量（省掉一次 (BLOCK_N, BLOCK_K) 的 gather）；
#   3. k 循环无掩码：要求 `H % (SPLIT_K * BLOCK_K) == 0`（同上，OLMoE 满足）。
#
# CUDA Graph 约束与 fp16 版完全一致（无 autotune、循环边界只来自 program_id/constexpr、
# expert id 只进指针算术）—— 这就是它能当 P3 的前提的原因。
# ===========================================================================


@triton.jit
def _expand_int4(packed, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """uint8 packed（低半字节在前）-> (BLOCK_N, BLOCK_K) 的 fp32 int4 值。

    `k = 2*j + which`：byte j 的低半字节是 code[2j]、高半字节是 code[2j+1]。
    符号扩展与 `lemoq.quant.pack.unpack_4bit(signed=True)` 逐位一致。
    """
    # 符号扩展用纯算术 `((x & 0xF) ^ 8) - 8`（0..15 -> -8..7），而不是 `tl.where` ——
    # `_grouped_triton.py` 有一条 AST 守卫禁 device/host 的 where（见
    # tests/test_moe_grouped.py::test_triton_kernels_avoid_everything_that_breaks_capture）。
    lo = ((packed & 0xF) ^ 8) - 8
    hi = (((packed >> 4) & 0xF) ^ 8) - 8
    return tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K)).to(tl.float32)


@triton.jit
def _gate_up_act_kernel_q(
    x_ptr,  # (H,) fp16     —— 本 token 的 hidden state
    q_ptr,  # (E, 2F, H) int8 [8-bit] / (E, 2F, H//2) uint8 [4-bit] —— packed codes
    s_ptr,  # (E, 2F, H//GROUP) fp16 —— per-group scale
    ids_ptr,  # (K,) int64
    act_ptr,  # (K, F) fp16  —— SPLIT_K == 1
    part_g_ptr,  # (SPLIT_K, K, F) fp32
    part_u_ptr,  # (SPLIT_K, K, F) fp32
    F: tl.constexpr,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    K_TOKENS: tl.constexpr,
    BITS: tl.constexpr,
):
    """`act[i, n] = silu((W_gate[e_i,n,:]*s).x) * ((W_up[e_i,n,:]*s).x)`，权重在寄存器里解包。

    `BITS=4` 时 packed 是 uint8（两个 4-bit code 一个字节，低半字节在前），
    在寄存器里展开成 int4 —— 与 dense 侧 `_dequant_triton` 同一个把戏。
    """
    slot = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    e = tl.load(ids_ptr + slot)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < F
    ng = H // GROUP  # 每行的组数

    # ★ 字节/行：8-bit 是 H，4-bit 是 H//2（`H * BITS // 8` 把两者写成同一个表达式）。
    #   不写 `: tl.constexpr` 注解：H/BITS 都是 constexpr，这个表达式在 trace 时就是常量。
    packed_row = H * BITS // 8
    qbase = q_ptr + e * (2 * F * packed_row)
    sbase = s_ptr + e * (2 * F * ng)

    acc_g = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_N,), dtype=tl.float32)

    rows_per_split = tl.cdiv(H, SPLIT_K)
    k_start = pid_k * rows_per_split
    k_end = tl.minimum(k_start + rows_per_split, H)

    for k0 in range(k_start, k_end, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        # ★ BLOCK_K 整除 GROUP ⇒ 本块整体落在第 g0 组，scale 是 (BLOCK_N,) 的广播
        g0 = k0 // GROUP
        xv = tl.load(x_ptr + k).to(tl.float32)
        sg = tl.load(sbase + n * ng + g0, mask=n_mask, other=0.0).to(tl.float32)
        su = tl.load(sbase + (F + n) * ng + g0, mask=n_mask, other=0.0).to(tl.float32)
        if BITS == 8:
            qg = tl.load(
                qbase + n[:, None] * packed_row + k[None, :], mask=n_mask[:, None], other=0
            ).to(tl.float32)
            qu = tl.load(
                qbase + (F + n)[:, None] * packed_row + k[None, :], mask=n_mask[:, None], other=0
            ).to(tl.float32)
        else:
            kb = k // 2  # 两个 k 共用一个 uint8
            pg = tl.load(
                qbase + n[:, None] * packed_row + kb[None, :], mask=n_mask[:, None], other=0
            ).to(tl.int32)
            pu = tl.load(
                qbase + (F + n)[:, None] * packed_row + kb[None, :], mask=n_mask[:, None], other=0
            ).to(tl.int32)
            qg = _expand_int4(pg, BLOCK_N, BLOCK_K)
            qu = _expand_int4(pu, BLOCK_N, BLOCK_K)
        acc_g += tl.sum(qg * sg[:, None] * xv[None, :], axis=1)
        acc_u += tl.sum(qu * su[:, None] * xv[None, :], axis=1)

    if SPLIT_K == 1:
        act = acc_g * tl.sigmoid(acc_g) * acc_u
        tl.store(act_ptr + slot * F + n, act.to(tl.float16), mask=n_mask)
    else:
        off = (pid_k * K_TOKENS + slot) * F
        tl.store(part_g_ptr + off + n, acc_g, mask=n_mask)
        tl.store(part_u_ptr + off + n, acc_u, mask=n_mask)


@triton.jit
def _down_sum_kernel_q(
    act_ptr,  # (K, F) fp16
    q_ptr,  # (E, H, F) int8 [8-bit] / (E, H, F//2) uint8 [4-bit]
    s_ptr,  # (E, H, F//GROUP) fp16
    ids_ptr,  # (K,) int64
    wsum_ptr,  # (K,) fp16
    out_ptr,  # (H,) fp16  —— SPLIT_K == 1
    part_ptr,  # (SPLIT_K, H) fp32
    H: tl.constexpr,
    F: tl.constexpr,
    GROUP: tl.constexpr,
    K_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BITS: tl.constexpr,
):
    """`out[n] = sum_i top_w[i] * ((W_down[e_i,n,:]*s).act[i,:])`，权重在寄存器里解包。

    `BITS=4` 的 nibble 展开同 `_gate_up_act_kernel_q`。
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n < H
    ng = F // GROUP

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    cols_per_split = tl.cdiv(F, SPLIT_K)
    k_start = pid_k * cols_per_split
    k_end = tl.minimum(k_start + cols_per_split, F)

    for slot in range(K_TOKENS):
        e = tl.load(ids_ptr + slot)
        w = tl.load(wsum_ptr + slot).to(tl.float32)
        packed_row = F * BITS // 8
        qbase = q_ptr + e * (H * packed_row)
        sbase = s_ptr + e * (H * ng)
        for k0 in range(k_start, k_end, BLOCK_K):
            k = k0 + tl.arange(0, BLOCK_K)
            g0 = k0 // GROUP
            a = tl.load(act_ptr + slot * F + k).to(tl.float32)
            s = tl.load(sbase + n * ng + g0, mask=n_mask, other=0.0).to(tl.float32)
            if BITS == 8:
                wq = tl.load(
                    qbase + n[:, None] * packed_row + k[None, :], mask=n_mask[:, None], other=0
                ).to(tl.float32)
            else:
                kb = k // 2
                pw = tl.load(
                    qbase + n[:, None] * packed_row + kb[None, :], mask=n_mask[:, None], other=0
                ).to(tl.int32)
                wq = _expand_int4(pw, BLOCK_N, BLOCK_K)
            acc += tl.sum(wq * s[:, None] * a[None, :], axis=1) * w

    if SPLIT_K == 1:
        tl.store(out_ptr + n, acc.to(tl.float16), mask=n_mask)
    else:
        tl.store(part_ptr + pid_k * H + n, acc, mask=n_mask)
