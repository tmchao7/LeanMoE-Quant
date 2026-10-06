"""_attn_triton — decode attention（split-KV flash-decoding）的 Triton kernel 本体。

## 这个文件为什么单独存在

`triton` 在 Mac 上装不了（没有 CUDA 后端）。把 `@triton.jit` 放在 `attn_decode.py` 里
会让整个模块在 Mac 上 import 失败，于是**离线测试也跑不了** —— 而离线测试正是发现
索引/形状错误的地方。所以 kernel 与 host 逻辑分开：

    attn_decode.py      纯 PyTorch oracle + host 逻辑 + 惰性 import 本模块
    _attn_triton.py     @triton.jit 的 kernel（本文件）—— 只被 GPU 路径 import

## 结构（两段式，见 attn_decode.py 的模块 docstring）

    kernel 1  grid = (NUM_SPLITS, heads)   每片在自己的 KV 区间上做在线 softmax
    kernel 2  grid = (heads,)              按 log-sum-exp 合并各片

## CUDA Graph 约束（**违反会 capture 失败或静默算错**）

1. **不得用 `@triton.autotune`。** 它会真的跑多个 config 来计时，capture 期间等于
   往图里塞一堆计划外的 kernel。配置由 host 显式传入。
2. **不得用 `tl.make_tensor_descriptor` / TMA。** descriptor 路径需要 global scratch，
   未设 `triton.set_allocator` 时会分配 → capture 隐患。这里全是普通指针算术。
3. **循环上界来自设备端 `tl.load(filled_ptr)`** —— 这是本 kernel 的立身之本：
   launch 形状（grid）静态，运行时的循环次数由设备内存决定。
   ★ 注意这与"形状依赖数据"**不是**一回事：grid 是 host 侧常量，图记录的是 launch
   参数。把长度读成 host 变量才会静默算错（实测 err=10 vs 0.004）。
4. 首次调用会 JIT 编译 —— 必须在 capture **之前** warm 过（`attn_decode.warm`）。

## 数值

`m_i/l_i/acc` 全在 fp32；`p` 走 `exp2`（`QK_SCALE = scale·1/ln2` 在 host 侧算好，
作为 constexpr 传入）。合并用固定顺序的循环，**不用 atomic_add**：非确定性会让
"同路径 KL 恰好为 0"这个门不可复现。

`l_i` 初值取 **0.0**（官方教程取 1.0，两者等价：首块 `alpha = exp2(-inf − m) = 0`
会把初值乘没）。取 0.0 的理由是它才是 l 的真值 —— 于是"这个分片一个元素都没读到"
时写出的就是 `(m=-inf, l=0)`，合并端可以据此把该片当成不贡献。
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _attn_split_kernel(
    q_ptr,  # (heads, D) fp16        —— decode 的 q（q_len=1）
    k_ptr,  # (heads, S, D) fp16     —— 静态 cache 的 k（S = max_cache_len，整段）
    v_ptr,  # (heads, S, D) fp16     —— 同形状
    filled_ptr,  # (1,) int32        —— ★ 设备端已填充长度：唯一的循环上界来源
    m_ptr,  # (heads, NUM_SPLITS) fp32
    l_ptr,  # (heads, NUM_SPLITS) fp32
    acc_ptr,  # (heads, NUM_SPLITS, D) fp32
    kv_stride_h,  # k/v 的 head stride（= S*D；切成 [0,filled) 会让它非连续）
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QK_SCALE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)

    # ★ 唯一的"数据依赖"：长度。它只进循环上界与掩码，不进 grid。
    n_filled = tl.load(filled_ptr).to(tl.int32)

    # 分片区间：与 `split_kv_partials_reference` 的 `cdiv` 划分**逐字对应**。
    # 两边必须一致，否则 K1 门量的是"两种分片方式"，不是 kernel 的实现误差。
    chunk = tl.cdiv(n_filled, NUM_SPLITS)
    n_start = pid_s * chunk
    n_end = tl.minimum(n_start + chunk, n_filled)

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + pid_h * D + offs_d).to(tl.float32)  # (D,)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((D,), dtype=tl.float32)

    for n0 in range(n_start, n_end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_end
        base = pid_h * kv_stride_h + offs_n[:, None] * D + offs_d[None, :]
        k = tl.load(k_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1) * QK_SCALE  # (BLOCK_N,)
        # 越界槽位必须在 max/exp2 之前变成 -inf，否则 exp2(0 − m) 会被算进来
        qk = tl.where(mask_n, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new)
        v = tl.load(v_ptr + base, mask=mask_n[:, None], other=0.0).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # 空分片（n_start >= n_end）写出 m=-inf / l=0：合并端权重 exp2(-inf−M)=0 → 不贡献。
    tl.store(m_ptr + pid_h * NUM_SPLITS + pid_s, m_i)
    tl.store(l_ptr + pid_h * NUM_SPLITS + pid_s, l_i)
    tl.store(acc_ptr + (pid_h * NUM_SPLITS + pid_s) * D + offs_d, acc)


@triton.jit
def _attn_combine_kernel(
    m_ptr,  # (heads, NUM_SPLITS) fp32
    l_ptr,  # (heads, NUM_SPLITS) fp32
    acc_ptr,  # (heads, NUM_SPLITS, D) fp32
    out_ptr,  # (heads, D) fp16
    D: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """`out = Σ_j acc_j·exp2(m_j−M) / Σ_j l_j·exp2(m_j−M)`，`M = max_j m_j`。

    ★ 为什么是**确定性**归约：`atomic_add` 的加法顺序由调度决定，同一条路径两次
    运行的 KL 不保证为 0，而 K2 门要求恰好 0。这里的 16 个分片用固定顺序的循环合并。
    """
    pid_h = tl.program_id(0)

    # 第一遍：取各分片运行最大值的最大值
    big_m = float("-inf")
    for j in range(NUM_SPLITS):
        big_m = tl.maximum(big_m, tl.load(m_ptr + pid_h * NUM_SPLITS + j))

    # 全空（filled_len=0）时 big_m=-inf，exp2(-inf − -inf)=NaN。换成 0：权重全 0、
    # 分母 0 → 输出 0。响亮不了，至少不能是 NaN（NaN 会污染整条下游链）。
    big_m = tl.where(big_m > float("-inf"), big_m, 0.0)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    num = tl.zeros((BLOCK_D,), dtype=tl.float32)
    den = 0.0
    for j in range(NUM_SPLITS):
        w = tl.math.exp2(tl.load(m_ptr + pid_h * NUM_SPLITS + j) - big_m)
        l_j = tl.load(l_ptr + pid_h * NUM_SPLITS + j)
        a_j = tl.load(acc_ptr + (pid_h * NUM_SPLITS + j) * D + offs_d, mask=mask_d, other=0.0)
        num += a_j * w
        den += l_j * w

    out = tl.where(den > 0.0, num / den, num)
    tl.store(out_ptr + pid_h * D + offs_d, out.to(tl.float16), mask=mask_d)
