"""attn_decode — decode attention（split-KV flash-decoding）的 host 逻辑 + fp32 oracle。

## 它要消掉的东西

`EXP-p2-04` 的图内归因（ctx=1024，per-token 3.1051 ms）里，attention 核心是四到五条
kernel 堆出来的：

    gemvNSP_kernel (q·kᵀ)              0.0928 ms/token   occ 16
    cunn_SoftMaxForward                0.0393
    gemvx::kernel  (w·v)               0.0373
    softmax_warp_forward               0.0145
    elementwise add（+mask）            0.0282
                                      ────────
                                      0.2121 ms/token  ≈ 6.8% of per-token

`tools/bench_decode_attention.py` 的隔离测量显示，把 k/v 裁到已填充长度、**不要 mask**
只要 0.03265 ms（ctx=1024）。但那种实现把长度烘进了 **shape**，换长度重放会**静默算错**
（实测 err=10.0，而正确答案是 O(1)）。本模块拿它的收益，同时保住换长度后仍然正确 ——
办法是让 kernel 从**设备内存**读已填充长度当循环上界，launch 形状保持静态。

## 为什么必须切 KV（split-KV）

decode 时 `batch=1, q_len=1, heads=16` → 朴素实现的 grid 只有 16 个 program，而卡上有
170 个 SM：**91% 的 SM 是空的**。把 KV 维切成 `num_splits` 片，每片算部分在线 softmax，
再用 log-sum-exp 合并（Flash-Decoding）。默认 `num_splits = ceil(SM / heads)`，
但参数敏感性本身是结果，扫描在隔离微基准里做（`tools/bench_attn_in_graph.py`）。

## 两段式结构

    kernel 1  grid = (num_splits, heads)   每片在自己 KV 区间上做在线 softmax
              写出 (m_i, l_i, acc[D]) -> (heads, num_splits, ·) 的 fp32 部分结果
    kernel 2  grid = (heads,)              按 LSE 合并：
              M = max_j m_j ; out = Σ_j acc_j·exp2(m_j−M) / Σ_j l_j·exp2(m_j−M)

合并用**确定性归约**（固定顺序的循环），不用 `atomic_add`：后者非确定性，
会让"同路径 KL 恰好为 0"这个门不可复现。

## 图安全（违反会 capture 失败或静默算错）

1. **禁止 `@triton.autotune`** —— 它会真的跑多个 config 来计时，capture 期间等于往图里
   塞一堆计划外的 kernel。配置在图**外**扫好、显式传参。
2. **不用 `tl.make_tensor_descriptor` / TMA** —— descriptor 路径需要 global scratch，
   未设 `triton.set_allocator` 时会分配 → capture 隐患。用普通指针算术。
3. **循环上界只能来自设备端 load**（`filled_len`），不能来自 host。launch 形状（grid）
   保持静态，所以图记录的是 launch 参数，运行时的循环次数由设备内存决定。
4. 首次调用前必须 `warm()`：Triton 首次调用会 JIT 编译，在 capture 期间做这件事
   报错信息会指向别处（`moe_grouped.warm` 是现成范例）。

## 数值骨架（照 Triton 官方 FA2 教程，走 exp2）

    m_i = -inf (fp32)  运行最大值
    l_i = 0.0  (fp32)  运行和        <- 与教程的 1.0 初值等价：alpha=exp2(-inf−m)=0
                                      会把初值乘没；这里用 0.0 是因为它才是 l 的真值
    acc = 0    (fp32)
    qk_scale = scale · 1.44269504    # 1/ln2，为走 exp2
    qk = dot(q, kᵀ) · qk_scale
    m_ij = max(m_i, max(qk))
    p    = exp2(qk − m_ij)
    alpha= exp2(m_i − m_ij)
    l_i  = l_i·alpha + sum(p)
    acc  = acc·alpha + p @ v
    m_i  = m_ij

**exp2 而不是 exp**：快，且 `1/ln2` 的缩放是精确可复现的写法（教程同款）。

## 与 HF 的关系（**必须说清楚**）

- **不再需要 4D mask**：kernel 只读 `[0, filled_len)`，decode 时 `q_len=1`，
  所有已填充的 key 都可见，没有因果结构要表达。mask 是 HF 屏蔽未填充槽位的唯一机制，
  不再需要它是因为 kernel 自己就不读那些槽位了。
- **K1 门（kernel 级）必须比 fp64 oracle**，不能比 eager 路径：累加顺序不同。
- **K2 门（端到端同路径）仍要求 KL 恰好为 0**：graph 与同路径 eager 都跑这个 kernel，
  确定性一致。
- 集成是**最小侵入**的：只换 attention core，保留 HF 的全部前后处理
  （q_norm/k_norm、RoPE、`past_key_values.update`、最后的 transpose/reshape/o_proj）。
  参见 `replace_attention_cores`。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "PARITY_GATE",
    "LENGTH_CHANGE_GATE",
    "ATTN_CONFIGS",
    "LOG2E",
    "attn_decode_available",
    "merged_config",
    "resolve_num_splits",
    "launch_plan",
    "split_kv_partials_reference",
    "merge_splits_reference",
    "split_kv_oracle",
    "dense_attention_oracle",
    "attn_decode_forward",
    "judge_length_change",
    "warm",
    "replace_attention_cores",
    "restore_attention_cores",
    "attention_cores_replaced",
    "decode_call_count",
]

# 1/ln2：为走 exp2 的精确可复现缩放（Triton FA2 教程同款写法）。
LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# ★ 门在**上机之前**冻结，与测得的数字无关。放宽必须单独成一个 commit 并写明理由
#   （AGENTS.md 规则 2）。与 `moe_grouped.PARITY_GATE` 同口径，便于横向比较。
# ---------------------------------------------------------------------------
PARITY_GATE = {
    "tol_atol": 1e-3,
    "tol_rtol": 1e-3,
    "tol_slack": 1.5,
    "note": (
        "err_kernel <= err_baseline * tol_slack 且 allclose(atol=tol_atol, rtol=tol_rtol)；"
        "err_* 都是相对 **fp64 oracle** 的最大绝对误差。baseline = HF eager 数学实现（fp16）。"
    ),
}

# K4（换长度）：这是本任务的**核心判据**，也是"一个不可能失败的测试比没有测试更糟"
# 那件事的防线 —— 所以除了"正确答案的误差要小"，还必须要求"陈旧答案离正确答案足够远"，
# 否则这个判据没有判别力（第一版 bench 就因此永不失败）。
LENGTH_CHANGE_GATE = {
    "correct_err_max": 1e-2,
    "discriminant_min": 1.0,
    "note": (
        "在长度 N 捕获 → 改成 M 重放：必须给出 M 的答案（err < correct_err_max），"
        "同时把 N 的答案与 M 的答案的差记为 discriminant；若它 < discriminant_min，"
        "说明这个测试分不出两种长度，判 INCONCLUSIVE 而不是 PASS。"
    ),
}

# 默认 kernel 配置。**刻意不用 @triton.autotune**（见模块 docstring 第 1 条）。
#   num_splits = None → ceil(SM / heads)：170/16 ≈ 10.6 → 11。
#   为什么不是别的值：compute 在这个形状上无关紧要，grid 覆盖 SM 数才是延迟的关键；
#   但**参数敏感性本身就是结果**，扫描要报全部（见 tools/bench_attn_in_graph.py）。
ATTN_CONFIGS: dict[str, Any] = {
    "split": {"BLOCK_N": 128, "num_warps": 4},
    "combine": {"BLOCK_D": 128, "num_warps": 4},
    "num_splits": None,
}

_MAX_OFFSET = 2**31


def _torch():
    import torch  # noqa: PLC0415

    return torch


def attn_decode_available() -> bool:
    """Triton kernel 是否可用。Mac 上永远是 False —— 那时只有 oracle 能跑。"""
    try:
        from lemoq.kernels import _attn_triton  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def _kernels():
    try:
        import triton  # noqa: PLC0415

        from lemoq.kernels import _attn_triton as k  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "attn_decode 的 kernel 路径需要 triton（本机不可用：%s）。"
            "oracle 路径（backend='oracle'）不需要 triton，可以在 CPU 上跑。" % type(exc).__name__
        ) from exc
    return triton, k


def merged_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """`ATTN_CONFIGS` 的浅拷贝，按调用点覆盖合并（缺项的键保持默认）。"""
    cfg = {
        name: dict(value) if isinstance(value, dict) else value
        for name, value in ATTN_CONFIGS.items()
    }
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def resolve_num_splits(num_splits: int | None, *, heads: int, device: Any = None) -> int:
    """`num_splits=None` → `ceil(SM / heads)`（本机实测 SM 数，不写死）。

    写死 170 会让这个默认值在别的卡上悄悄变成错的覆盖度；而 `torch.cuda`
    拿到的 SM 数是 host 侧常量，图安全没有影响。
    """
    if num_splits is not None:
        if int(num_splits) < 1:
            raise ValueError("num_splits 必须 >= 1")
        return int(num_splits)
    sm = None
    try:
        torch = _torch()

        if device is not None and torch.cuda.is_available():
            sm = int(torch.cuda.get_device_properties(device).multi_processor_count)
        elif torch.cuda.is_available():
            sm = int(torch.cuda.get_device_properties(0).multi_processor_count)
    except Exception:  # noqa: BLE001
        sm = None
    if not sm:
        return 1
    return max(1, -(-sm // max(1, int(heads))))


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def launch_plan(
    *,
    heads: int,
    head_dim: int,
    num_splits: int,
    block_n: int,
    block_d: int | None = None,
) -> dict[str, Any]:
    """把 grid / 缓冲区形状**算清楚**，不碰 GPU。

    ★ 为什么把这段算术单独拎出来（纯 Python、可离线测试）：CUDA Graph 只约束
    "形状静态"，不约束"形状对不对"。grid 少算一维、部分和缓冲区写错大小，
    都不会报错 —— 只会**静默算错**，而这类错误在 GPU 上要靠 KL 门才能发现，
    代价是加载 13 GB 模型 + 几分钟。这里把它变成 Mac 上可以手算核对的东西。

    `kernel_count` = 2（split + combine）：对照 p2-04 图内 attention 核心的 5 条。
    """
    if heads < 1 or head_dim < 1:
        raise ValueError("heads 与 head_dim 必须为正")
    if block_n < 1 or block_n & (block_n - 1):
        raise ValueError("BLOCK_N 必须是 2 的幂（tl.arange 的要求），得到 %r" % (block_n,))
    if head_dim & (head_dim - 1):
        raise ValueError("head_dim 必须是 2 的幂（tl.arange 的要求），得到 %r" % (head_dim,))
    if num_splits < 1:
        raise ValueError("num_splits 必须 >= 1")
    block_d = int(block_d or head_dim)
    if block_d < head_dim:
        raise ValueError("BLOCK_D 不能小于 head_dim")
    return {
        "heads": int(heads),
        "head_dim": int(head_dim),
        "num_splits": int(num_splits),
        "BLOCK_N": int(block_n),
        "BLOCK_D": block_d,
        "kernel_count": 2,
        "split": {
            "grid": (int(num_splits), int(heads)),
            "m_shape": (int(heads), int(num_splits)),
            "l_shape": (int(heads), int(num_splits)),
            "acc_shape": (int(heads), int(num_splits), int(head_dim)),
            "dtype": "float32",
        },
        "combine": {
            "grid": (int(heads),),
            "out_shape": (1, int(heads), 1, int(head_dim)),
        },
    }


# ---------------------------------------------------------------------------
# 参考实现（纯 PyTorch，**不需要 triton / GPU**）—— 这是 K1 门的 oracle
# ---------------------------------------------------------------------------
def _canonical(q: Any, k: Any, v: Any) -> tuple[Any, Any, Any]:
    """(1, H, 1, D) / (1, H, S, D) → (H, D) / (H, S, D) fp32。"""
    torch = _torch()
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维：q (1,H,1,D)、k/v (1,H,S,D)")
    if int(q.shape[0]) != 1 or int(q.shape[2]) != 1:
        raise ValueError("attn_decode 只处理 batch=1 / q_len=1：得到 q %s" % (tuple(q.shape),))
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("k 与 v 形状必须相同")
    if int(k.shape[1]) != int(q.shape[1]):
        raise NotImplementedError(
            "本 kernel 不做 GQA：k 的 head 数 %d != q 的 %d。"
            "OLMoE 的 num_key_value_heads == num_attention_heads，所以这里直接拒绝，"
            "而不是静默算出错的结果。" % (int(k.shape[1]), int(q.shape[1]))
        )
    qf = q.to(torch.float32).reshape(int(q.shape[1]), int(q.shape[3])).contiguous()
    kf = k.to(torch.float32).reshape(int(k.shape[1]), int(k.shape[2]), int(k.shape[3]))
    vf = v.to(torch.float32).reshape(int(v.shape[1]), int(v.shape[2]), int(v.shape[3]))
    return qf, kf, vf


def split_kv_partials_reference(
    q: Any,
    k: Any,
    v: Any,
    filled_len: int,
    *,
    num_splits: int,
    scale: float | None = None,
    block_n: int = 128,
) -> tuple[Any, Any, Any]:
    """在线 softmax 的逐分片部分结果 `(m, l, acc)`，**块序与 kernel 一致**。

    形状：`m, l: (H, num_splits) fp32`、`acc: (H, num_splits, D) fp32`。

    ★ 为什么先写它、再写合并（而不是直接写真值）：`m/l/acc` 是 split-KV 唯一
    容易写错的地方（运行最大值与运行和的更新顺序），把它暴露出来就能**分别**测
    "分片内部的在线 softmax"与"分片之间的 LSE 合并"两件事。
    两块各自对了，合起来才对。
    """
    torch = _torch()
    qf, kf, vf = _canonical(q, k, v)
    heads, dim = int(qf.shape[0]), int(qf.shape[1])
    seq = int(kf.shape[1])
    n = int(filled_len)
    if n < 0 or n > seq:
        raise ValueError("filled_len=%d 必须在 [0, %d] 内" % (n, seq))
    if num_splits < 1:
        raise ValueError("num_splits 必须 >= 1")
    scale = (1.0 / math.sqrt(dim)) if scale is None else float(scale)
    qk_scale = scale * LOG2E

    m = torch.full((heads, num_splits), float("-inf"), dtype=torch.float32, device=qf.device)
    l = torch.zeros((heads, num_splits), dtype=torch.float32, device=qf.device)
    acc = torch.zeros((heads, num_splits, dim), dtype=torch.float32, device=qf.device)
    if n == 0:
        return m, l, acc

    chunk = _cdiv(n, num_splits)
    for s in range(num_splits):
        start = s * chunk
        end = min(start + chunk, n)
        if start >= end:
            # 空分片：m=-inf / l=0 / acc=0。合并时权重 exp2(-inf−M)=0，贡献恰好为 0，
            # 但**前提**是 M 来自别的非空分片 —— 全空的退化情形在 merge 里显式挡住。
            continue
        m_i = torch.full((heads,), float("-inf"), dtype=torch.float32, device=qf.device)
        l_i = torch.zeros((heads,), dtype=torch.float32, device=qf.device)
        a_i = torch.zeros((heads, dim), dtype=torch.float32, device=qf.device)
        for n0 in range(start, end, block_n):
            hi = min(n0 + block_n, end)
            kk = kf[:, n0:hi, :]
            vv = vf[:, n0:hi, :]
            qk = torch.matmul(kk, qf.unsqueeze(-1)).squeeze(-1) * qk_scale  # (H, b)
            m_new = torch.maximum(m_i, qk.max(dim=1).values)
            alpha = torch.exp2(m_i - m_new)  # m_i=-inf → 0，把初值乘没
            p = torch.exp2(qk - m_new.unsqueeze(1))
            l_i = l_i * alpha + p.sum(dim=1)
            a_i = a_i * alpha.unsqueeze(1) + torch.matmul(p.unsqueeze(1), vv).squeeze(1)
            m_i = m_new
        m[:, s] = m_i
        l[:, s] = l_i
        acc[:, s] = a_i
    return m, l, acc


def merge_splits_reference(m: Any, l: Any, acc: Any) -> Any:
    """LSE 合并：`out = Σ_j acc_j·exp2(m_j−M) / Σ_j l_j·exp2(m_j−M)`，`M = max_j m_j`。

    返回 `(H, D)` fp32。**确定性**：按 j 的固定顺序求和，没有 `atomic_add`。
    """
    torch = _torch()
    if m.shape != l.shape or m.dim() != 2:
        raise ValueError("m/l 必须是同形状的 (H, num_splits)")
    m = m.to(torch.float32)
    l = l.to(torch.float32)
    acc = acc.to(torch.float32)
    big_m = m.max(dim=1, keepdim=True).values
    # 所有分片都空时 M=-inf，exp2(-inf − -inf)=NaN。把 M 换成 0：所有权重变 0，
    # den=0 → 输出 0。**响亮不了就至少不能是 NaN**（NaN 会污染整条链下游）。
    big_m = torch.where(torch.isfinite(big_m), big_m, torch.zeros_like(big_m))
    w = torch.exp2(m - big_m)  # (H, S)
    num = (acc * w.unsqueeze(-1)).sum(dim=1)
    den = (l * w).sum(dim=1)
    safe_den = den.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(1)
    out = torch.where(den.unsqueeze(1) > 0, num / safe_den, num)
    return out


def split_kv_oracle(
    q: Any,
    k: Any,
    v: Any,
    filled_len: int,
    *,
    num_splits: int,
    scale: float | None = None,
    block_n: int = 128,
) -> Any:
    """split-KV + LSE 合并的 fp32 oracle（K1 的对照物之一）。返回 `(H, D)` fp32。"""
    m, l, acc = split_kv_partials_reference(
        q, k, v, filled_len, num_splits=num_splits, scale=scale, block_n=block_n
    )
    return merge_splits_reference(m, l, acc)


def dense_attention_oracle(
    q: Any, k: Any, v: Any, filled_len: int, *, scale: float | None = None, dtype: Any = None
) -> Any:
    """不分片的稠密 attention，默认 **fp64**。返回 `(H, D)`。

    这是 K1 的判据来源：split-KV 的数学必须等于它，差多少要看 fp64 而不是 fp16 的
    eager 路径（eager 路径的误差本身就在 1e-3 量级，拿它当 oracle 量不出东西）。
    """
    torch = _torch()
    dtype = torch.float64 if dtype is None else dtype
    qf, kf, vf = _canonical(q, k, v)
    n = int(filled_len)
    if n < 0 or n > int(kf.shape[1]):
        raise ValueError("filled_len 越界")
    dim = int(qf.shape[1])
    scale = (1.0 / math.sqrt(dim)) if scale is None else float(scale)
    qq = qf.to(dtype)
    kk = kf[:, :n, :].to(dtype)
    vv = vf[:, :n, :].to(dtype)
    if n == 0:
        return torch.zeros((int(qf.shape[0]), dim), dtype=dtype, device=qf.device)
    scores = torch.matmul(kk, qq.unsqueeze(-1)).squeeze(-1) * scale  # (H, n)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights.unsqueeze(1), vv).squeeze(1)


# ---------------------------------------------------------------------------
# GPU 路径
# ---------------------------------------------------------------------------
def judge_length_change(
    *, correct_err: float, stale_err: float, gate: dict[str, Any] | None = None
) -> dict[str, Any]:
    """K4 的判据：**先问这个测试有没有判别力，再问它过没过**。

    `correct_err` = 换长度重放后的输出与"新长度的正确答案"的误差。
    `stale_err`   = "旧长度的答案"与"新长度的正确答案"之间的距离。

    为什么必须有 `stale_err` 这一项：`tools/bench_decode_attention.py` 的第一版
    换长度检查有两个 bug（只改 host 变量没改设备 mask；softmax 在约 1000 个随机 key 上
    近似均匀，让两种长度的答案趋同），结果**永不失败**。一个不可能失败的测试比没有
    测试更糟，因为它会被引用。所以这里显式要求 `stale_err >= discriminant_min`；
    达不到就判 `INCONCLUSIVE` —— 既不是 PASS 也不是 FAIL，而是"这次测量没资格下结论"。
    """
    gate = LENGTH_CHANGE_GATE if gate is None else gate
    correct_err = float(correct_err)
    stale_err = float(stale_err)
    discriminating = stale_err >= float(gate["discriminant_min"])
    correct = correct_err < float(gate["correct_err_max"])
    if not discriminating:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS" if correct else "FAIL"
    return {
        "verdict": verdict,
        "correct_err": correct_err,
        "stale_err": stale_err,
        "discriminating": bool(discriminating),
        "gate": dict(gate),
        "note": (
            "correct_err < %s 且 stale_err >= %s"
            % (gate["correct_err_max"], gate["discriminant_min"])
        ),
    }


def _filled_tensor(filled_len: Any, *, device: Any, seq_len: int) -> Any:
    """把 host 长度或设备 scalar 归一成 `(1,) int32` **设备**张量。

    ★ 设备 scalar 是首选：它是图内唯一允许的长度来源（见模块 docstring 第 3 条）。
    host int 只服务于离线/隔离测试 —— 传 host int 时**不能**进图，这里不做任何
    额外拦截（进图时 kernel 读到的仍是设备内存），但会在推理里显式记录。
    """
    torch = _torch()
    if torch.is_tensor(filled_len):
        if filled_len.numel() != 1:
            raise ValueError("filled_len 张量必须是 1 个元素，得到 %d" % int(filled_len.numel()))
        if filled_len.dtype != torch.int32:
            raise ValueError("filled_len 张量必须是 int32，得到 %s" % filled_len.dtype)
        if filled_len.device != device:
            raise ValueError(
                "filled_len 与 k/v 不在同一设备：%s vs %s" % (filled_len.device, device)
            )
        if filled_len.device.type == "cpu":
            n = int(filled_len.item())
            if n < 0 or n > seq_len:
                raise ValueError("filled_len=%d 必须在 [0, %d] 内" % (n, seq_len))
        # CUDA scalar bounds remain the caller's contract; do not add a host .item() sync.
        return filled_len.reshape(1)
    if type(filled_len) is not int:
        raise ValueError("filled_len 必须为 int 或单元素 int32 张量")
    n = int(filled_len)
    if n < 0 or n > seq_len:
        raise ValueError("filled_len=%d 必须在 [0, %d] 内" % (n, seq_len))
    return torch.full((1,), n, dtype=torch.int32, device=device)


def attn_decode_forward(
    q: Any,
    k: Any,
    v: Any,
    filled_len: Any,
    *,
    config: dict[str, Any] | None = None,
    backend: str = "auto",
    scale: float | None = None,
) -> Any:
    """`(1, H, 1, D)` 的 q + 静态 cache 的 k/v → `(1, H, 1, D)` 的 fp16 输出。

    `k`/`v` 是 HF `StaticCache.update()` 返回的**整段** cache（形状 `(1,H,max_len,D)`）：
    kernel 只读 `[0, filled_len)`，所以既不需要 mask，也不受 `max_cache_len` 影响。
    长度来自 `filled_len`（设备 scalar 优先）。

    `backend`：
        "triton"  真 kernel（需要 GPU + triton）
        "oracle"  纯 PyTorch 参考实现跑**同一条 host 管线**（同形状检查、同输出契约）
                  —— 这样 host 侧的 reshape/dtype/契约在没有 GPU 的机器上也能被测到
        "auto"    triton，不可用时**报错**（像 `moe_grouped` 一样：静默换慢路径
                  会让基准测量在一个没被声明的实现上跑）
    """
    torch = _torch()
    cfg = merged_config(config)
    if backend not in ("auto", "triton", "oracle"):
        raise ValueError("unknown backend %r" % (backend,))
    if backend == "auto" and not attn_decode_available():
        raise RuntimeError(
            "attn_decode 的 kernel 路径需要 triton；本机不可用。"
            "用 backend='oracle' 可以只跑 host 管线 + 参考实现。"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not torch.is_tensor(tensor):
            raise TypeError("%s 必须是张量" % name)
        if tensor.dim() != 4:
            raise ValueError("%s 必须是 4 维，得到 %s" % (name, tuple(tensor.shape)))
        if tensor.stride(-1) != 1:
            raise ValueError("%s 的最后一维必须连续（kernel 里的偏移按 stride=1 算）" % name)
        if tensor.numel() >= _MAX_OFFSET:
            raise ValueError("%s 的元素数超过 int32 偏移上限" % name)
    if int(q.shape[2]) != 1 or int(q.shape[0]) != 1:
        raise NotImplementedError("attn_decode 只处理 batch=1 / q_len=1")
    if tuple(k.shape) != tuple(v.shape):
        raise ValueError("k 与 v 形状必须相同")
    if int(k.shape[0]) != 1 or int(k.shape[3]) != int(q.shape[3]):
        raise ValueError("k/v 必须为 batch=1 且 head_dim 与 q 相同")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q/k/v 必须在同一设备")
    if int(q.shape[1]) < 1 or int(k.shape[2]) < 1:
        raise ValueError("head 数与 cache 容量必须为正")
    if k.stride(2) != k.shape[3] or v.stride(2) != v.shape[3]:
        raise ValueError("k/v 的序列 stride 必须等于 head_dim")
    if k.stride(1) != v.stride(1) or k.stride(1) < k.shape[2] * k.shape[3]:
        raise ValueError("k/v 必须有相同且不重叠的 head stride")
    if int(k.shape[1]) != int(q.shape[1]):
        raise NotImplementedError("attn_decode 不做 GQA（见 docstring）")
    if k.dtype != v.dtype:
        raise ValueError("k/v dtype 必须相同")
    if q.dtype != k.dtype:
        raise ValueError("q 与 k/v dtype 必须相同（%s vs %s）" % (q.dtype, k.dtype))
    if backend != "oracle" and q.dtype != torch.float16:
        raise ValueError("triton attention 仅支持 fp16；oracle 可使用其他浮点类型")

    heads = int(q.shape[1])
    head_dim = int(q.shape[3])
    seq_len = int(k.shape[2])
    if (heads - 1) * k.stride(1) + seq_len * head_dim >= _MAX_OFFSET:
        raise ValueError("k/v 的实际 stride 偏移超过 int32 上限")
    filled = _filled_tensor(filled_len, device=q.device, seq_len=seq_len)
    num_splits = resolve_num_splits(cfg.get("num_splits"), heads=heads, device=k.device)
    plan = launch_plan(
        heads=heads,
        head_dim=head_dim,
        num_splits=num_splits,
        block_n=int(cfg["split"]["BLOCK_N"]),
        block_d=int(cfg["combine"]["BLOCK_D"]),
    )

    qc = q.reshape(heads, head_dim)
    if not qc.is_contiguous():
        qc = qc.contiguous()
    kc = k.reshape(heads, seq_len, head_dim)
    vc = v.reshape(heads, seq_len, head_dim)

    if backend == "oracle":
        n = int(filled.item())
        acc = split_kv_oracle(
            q, k, v, n, num_splits=num_splits, scale=scale, block_n=plan["BLOCK_N"]
        )
        return acc.to(q.dtype).reshape(*plan["combine"]["out_shape"])

    triton, kernels = _kernels()
    _ = triton  # 只为导入副作用（编译后端注册）
    dev = q.device
    if not q.is_cuda:
        raise RuntimeError("triton backend 需要 CUDA 张量")
    m = torch.empty(plan["split"]["m_shape"], dtype=torch.float32, device=dev)
    l = torch.empty(plan["split"]["l_shape"], dtype=torch.float32, device=dev)
    acc = torch.empty(plan["split"]["acc_shape"], dtype=torch.float32, device=dev)
    out = torch.empty(plan["combine"]["out_shape"], dtype=q.dtype, device=dev)

    qk_scale = (scale if scale is not None else 1.0 / math.sqrt(head_dim)) * LOG2E
    kernels._attn_split_kernel[plan["split"]["grid"]](
        qc,
        kc,
        vc,
        filled,
        m,
        l,
        acc,
        int(k.stride(1)),
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_N=plan["BLOCK_N"],
        QK_SCALE=float(qk_scale),
        num_warps=int(cfg["split"]["num_warps"]),
    )
    kernels._attn_combine_kernel[plan["combine"]["grid"]](
        m,
        l,
        acc,
        out,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_D=plan["BLOCK_D"],
        num_warps=int(cfg["combine"]["num_warps"]),
    )
    return out


def attn_decode_forward_int8(
    q: Any,
    k_codes: Any,
    k_scale: Any,
    v_codes: Any,
    v_scale: Any,
    filled_len: Any,
    *,
    config: dict[str, Any] | None = None,
    scale: float | None = None,
) -> Any:
    """E6（EXP-p5-01）：int8 镜像 cache 的 decode attention —— **scale 折叠，内层零解包**。

    `k_codes/v_codes`：(1,H,S,D) int8（per-token 对称量化，`kv_int8_bulk_quantize` +
    `kv_int8_cache_write` 维护）；`k_scale/v_scale`：(1,H,S) fp16 per-token scale。
    分数/权重折叠的代数见 `_attn_triton._attn_split_kernel_int8`；与"先反量化再
    attention"代数恒等、结合序不同 ⇒ parity 判据 = fp64 oracle + PARITY_GATE 1e-3。
    combine kernel 与 fp16 路径**共用**（m/l/acc 语义相同）。
    """
    torch = _torch()
    cfg = merged_config(config)
    for name, tensor in (("q", q), ("k_codes", k_codes), ("v_codes", v_codes)):
        if not torch.is_tensor(tensor) or tensor.dim() != 4:
            raise ValueError("%s 必须是 4 维张量" % name)
        if tensor.stride(-1) != 1:
            raise ValueError("%s 的最后一维必须连续" % name)
    for name, tensor in (("k_scale", k_scale), ("v_scale", v_scale)):
        if not torch.is_tensor(tensor) or tensor.dim() != 3:
            raise ValueError("%s 必须是 3 维 (1,H,S)" % name)
    if k_codes.dtype != torch.int8 or v_codes.dtype != torch.int8:
        raise ValueError("codes 必须 int8")
    if k_scale.dtype != torch.float16 or v_scale.dtype != torch.float16:
        raise ValueError("scale 必须 fp16")
    if int(q.shape[2]) != 1 or int(q.shape[0]) != 1:
        raise NotImplementedError("int8 attention 只处理 batch=1 / q_len=1")
    heads, seq_len, head_dim = int(k_codes.shape[1]), int(k_codes.shape[2]), int(k_codes.shape[3])
    if tuple(k_codes.shape) != tuple(v_codes.shape) or int(q.shape[1]) != heads:
        raise ValueError("k/v codes 形状必须一致且头数与 q 相同")
    if tuple(k_scale.shape) != (1, heads, seq_len) or tuple(v_scale.shape) != (1, heads, seq_len):
        raise ValueError("scale 必须是 (1,H,S)")
    if not (k_codes.is_cuda and q.is_cuda):
        raise RuntimeError("int8 attention 需要 CUDA")
    if int(head_dim - 1) & int(head_dim) or head_dim < 2:
        raise ValueError("head_dim 必须是 2 的幂")

    filled = _filled_tensor(filled_len, device=q.device, seq_len=seq_len)
    num_splits = resolve_num_splits(cfg.get("num_splits"), heads=heads, device=q.device)
    plan = launch_plan(
        heads=heads,
        head_dim=head_dim,
        num_splits=num_splits,
        block_n=int(cfg["split"]["BLOCK_N"]),
        block_d=int(cfg["combine"]["BLOCK_D"]),
    )
    qc = q.reshape(heads, head_dim)
    if not qc.is_contiguous():
        qc = qc.contiguous()
    kc = k_codes.reshape(heads, seq_len, head_dim)
    vc = v_codes.reshape(heads, seq_len, head_dim)
    ksc = k_scale.reshape(heads, seq_len)
    vsc = v_scale.reshape(heads, seq_len)

    triton, kernels = _kernels()
    _ = triton
    dev = q.device
    m = torch.empty(plan["split"]["m_shape"], dtype=torch.float32, device=dev)
    l = torch.empty(plan["split"]["l_shape"], dtype=torch.float32, device=dev)
    acc = torch.empty(plan["split"]["acc_shape"], dtype=torch.float32, device=dev)
    out = torch.empty(plan["combine"]["out_shape"], dtype=q.dtype, device=dev)
    qk_scale = (scale if scale is not None else 1.0 / math.sqrt(head_dim)) * LOG2E
    # ★ head stride 必须取 **4D 的 stride(1)**（= S·D）：`kc` 已被 reshape 成 (H,S,D)，
    #   它的 stride(1) 是 S 维步长（= D）——传它等于让 head h 读 h·D 偏移（头串数据，
    #   head 0 恰好正确、其余头全错，2026-10-25 K 门首跑抓到）。fp16 路径传 4D 的
    #   stride(1) 所以没踩中。
    head_stride = int(k_codes.stride(1))
    kernels._attn_split_kernel_int8[plan["split"]["grid"]](
        qc,
        kc,
        ksc,
        vc,
        vsc,
        filled,
        m,
        l,
        acc,
        head_stride,
        seq_len,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_N=plan["BLOCK_N"],
        QK_SCALE=float(qk_scale),
        num_warps=int(cfg["split"]["num_warps"]),
    )
    kernels._attn_combine_kernel[plan["combine"]["grid"]](
        m,
        l,
        acc,
        out,
        D=head_dim,
        NUM_SPLITS=num_splits,
        BLOCK_D=plan["BLOCK_D"],
        num_warps=int(cfg["combine"]["num_warps"]),
    )
    return out


def warm(
    *,
    heads: int,
    head_dim: int,
    kv_len: int,
    filled: int | None = None,
    dtype: Any = None,
    device: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在 capture **之前**把 kernel 编译好（Triton 首次调用会 JIT）。

    ★ 为什么必须有这一步：在 CUDA Graph capture 期间做 JIT 编译会直接报错，
    而且那时的报错信息通常指向别处（`moe_grouped.warm` 的注释同款理由）。

    返回被 warm 的配置，便于记录进结果文件。
    """
    torch = _torch()
    if dtype is None:
        dtype = torch.float16
    if device is None:
        device = "cuda"
    if isinstance(device, str):
        device = torch.device(device)
    q = torch.zeros((1, heads, 1, head_dim), dtype=dtype, device=device)
    k = torch.zeros((1, heads, kv_len, head_dim), dtype=dtype, device=device)
    v = torch.zeros((1, heads, kv_len, head_dim), dtype=dtype, device=device)
    n = int(filled if filled is not None else kv_len)
    filled_len = torch.full((1,), n, dtype=torch.int32, device=device)
    attn_decode_forward(q, k, v, filled_len, config=config, backend="triton")
    if device.type == "cuda":
        torch.cuda.synchronize()
    cfg = merged_config(config)
    return {
        "heads": int(heads),
        "head_dim": int(head_dim),
        "kv_len": int(kv_len),
        "filled": n,
        "dtype": str(dtype),
        "device": str(device),
        "num_splits": resolve_num_splits(cfg.get("num_splits"), heads=int(heads), device=device),
        "BLOCK_N": int(cfg["split"]["BLOCK_N"]),
    }


# ---------------------------------------------------------------------------
# 最小侵入地接进 HF 的 OLMoE
# ---------------------------------------------------------------------------
def _is_olmoe_attention(module: Any) -> bool:
    """按**能力**认 attention 模块，不按类名（同名类在不同版本里可能搬家）。"""
    needed = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    if not all(hasattr(module, name) for name in needed):
        return False
    return hasattr(module, "num_heads") and hasattr(module, "head_dim")


def _attention_modules(model: Any) -> list[Any]:
    """优先按 `model.model.layers[*].self_attn` 取；退化时按能力全树扫描。"""
    found: list[Any] = []
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is not None:
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is not None and _is_olmoe_attention(attn) and attn not in found:
                found.append(attn)
    if found:
        return found
    for module in model.modules():
        if _is_olmoe_attention(module) and module not in found:
            found.append(module)
    return found


def _filled_len_of(past_key_values: Any) -> Any:
    """从 cache 对象上取设备端已填充长度（**不用模块级全局变量**）。

    这个张量由 `StaticCacheDecodeSession._set_step_inputs()` 在图**外**每步 `fill_`，
    所以图里记录的是它的地址，运行时读到的值总是当前步的长度 —— 这就是
    "launch 形状静态 + 长度来自设备内存"的全部机制。
    """
    if past_key_values is None:
        return None
    return getattr(past_key_values, "_lemoq_filled_len", None)


def _decode_forward_impl(
    self: Any,
    hidden_states: Any,
    attention_mask: Any = None,
    position_ids: Any = None,
    past_key_values: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Any = None,
    position_embeddings: Any = None,
    **kwargs: Any,
) -> Any:
    """HF `Olmoe*Attention.forward` 的替换：**只换中间的 attention core**。

    ★ 保留 HF 的全部前后处理（q_norm/k_norm、投影、RoPE、`past_key_values.update`、
    最后的 transpose/reshape + o_proj），理由：投影/RoPE/cache 的数值偏离风险最大，
    而换掉它们对性能没有帮助（那些是 `[W]` 里必要的 attention 权重读量）。

    三条**退回原实现**的路径（都必须是显式的，不能静默算错）：
        1. `q_len != 1` —— prefill 走 HF 原路（prefill 永远逐 token/逐专家，见 §5.7）；
        2. 拿不到 `filled_len` —— 例如动态 cache 的 informational 参照；
        3. `output_attentions=True` —— 本 kernel 不产出 attention 权重。
    """
    original = self._lemoq_original_forward
    filled = _filled_len_of(past_key_values)
    q_len = int(hidden_states.shape[1])
    if output_attentions or q_len != 1 or filled is None or position_embeddings is None:
        if q_len != 1 and getattr(self, "_lemoq_kv_int8_bufs", None) is not None:
            # ★ E6：prefill = 一段新序列开始。int8 镜像必须作废，下一个 decode 步
            #   对新 cache 重新 bulk——否则静默复用上一会话/窗口的 codes（prompt 一换
            #   就是污染）。同 prompt 的重复 prefill（runner 的 reset 循环）数值等价，
            #   作废只是多付一次 bulk。prefill 恒 eager，不经过图，capture 安全。
            self._lemoq_kv_int8_bufs = None
        return original(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    if int(getattr(self, "num_key_value_groups", 1)) != 1:
        raise NotImplementedError(
            "attn_decode 不做 GQA：num_key_value_heads=%s != num_attention_heads=%s。"
            "拒绝静默算错。"
            % (getattr(self, "num_key_value_heads", "?"), getattr(self, "num_heads", "?"))
        )

    from transformers.models.olmoe.modeling_olmoe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    from lemoq.kernels.packed_qkv import packed_qkv_enabled  # noqa: PLC0415
    from lemoq.kernels.rope import _fused_rope_enabled  # noqa: PLC0415

    bsz, _, _ = hidden_states.shape
    packed = getattr(self, "_lemoq_packed_qkv", None)
    if packed is not None and packed_qkv_enabled():
        # F2 候选：q/k/v 一次 launch，逐行结果与三次独立调用逐位一致
        # （行局部布局；默认开（2026-10-24 采纳），见 `lemoq/kernels/packed_qkv.py`）。
        q_states, k_states, v_states = packed.project(hidden_states)
        query_states = self.q_norm(q_states)
        key_states = self.k_norm(k_states)
        value_states = v_states
    else:
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

    clip_qkv = getattr(self.config, "clip_qkv", None)
    if clip_qkv is not None:
        query_states.clamp_(min=-clip_qkv, max=clip_qkv)
        key_states.clamp_(min=-clip_qkv, max=clip_qkv)
        value_states.clamp_(min=-clip_qkv, max=clip_qkv)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
        1, 2
    )
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
        1, 2
    )

    cos, sin = position_embeddings
    if _fused_rope_enabled():
        # 候选路径：rotate_half 链（neg/cat/mul/add ×10/层）融合为 2 kernel/层。
        # Python 相位读 env ⇒ capture 烘焙；默认开走融合，env=0 走 HF 原路。
        # 见 docs/compose/spec/fused-rope.md。
        from lemoq.kernels.rope import apply_rotary_pos_emb_fused  # noqa: PLC0415

        query_states, key_states = apply_rotary_pos_emb_fused(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    new_key, new_value = key_states, value_states  # int8 镜像写要用"新行"（下面会被换成整段 cache）
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        from lemoq.kernels.fusion_switches import fusion_enabled  # noqa: PLC0415

        kv_direct = fusion_enabled("kv_direct_write")
        if kv_direct:
            from lemoq.kernels.kv_direct import (  # noqa: PLC0415
                kv_direct_write,
                kv_direct_write_applicable,
            )

        if kv_direct and kv_direct_write_applicable(
            past_key_values.layers[self.layer_idx], key_states, value_states
        ):
            # F4 候选（S2）：rope 后的 k/v 逐位直写 cache 的 [pos] 槽，替代
            # update() 的 index_copy_ 链（每步 32 次 aten launch）。默认关
            # （`fusion_switches.FUSION_DEFAULTS`），K 门见 tests/test_kv_direct_write.py。
            layer = past_key_values.layers[self.layer_idx]
            kv_direct_write(layer, key_states, value_states, cache_position)
            key_states, value_states = layer.keys, layer.values
        else:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # ★ E6（EXP-p5-01）：int8 镜像 cache 读路径（默认关）。fp16 cache 照常维护
    #   （正确性网）；int8 buffer 是**读源**——attention 从 int8 读，scale 折叠进
    #   分数/权重（内层零解包）。首次调用 bulk 量化整段 cache（eager warmup 相位，
    #   免 host 长度；capture 期调用会显式拒绝——初始化绝不能进图）。
    from lemoq.kernels.fusion_switches import fusion_enabled as _fusion_enabled  # noqa: PLC0415

    if _fusion_enabled("kv_int8"):
        if self._lemoq_attn_backend != "triton":
            raise RuntimeError("kv_int8 需要 attn backend=triton（oracle 只用于单测）")
        bufs = getattr(self, "_lemoq_kv_int8_bufs", None)
        if bufs is None:
            import torch as _t  # noqa: PLC0415

            if _t.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "kv_int8 的 buffer 初始化不能发生在 capture 里——先跑 eager warmup 步"
                )
            from lemoq.kernels.kv_int8 import kv_int8_init_buffers  # noqa: PLC0415

            layer = past_key_values.layers[self.layer_idx]
            bufs = kv_int8_init_buffers(layer.keys, layer.values, filled=None)
            self._lemoq_kv_int8_bufs = bufs
        from lemoq.kernels.kv_int8 import kv_int8_cache_write  # noqa: PLC0415

        kv_int8_cache_write(bufs, new_key, new_value, cache_position)
        attn_output = attn_decode_forward_int8(
            query_states,
            bufs["kc"],
            bufs["ks"],
            bufs["vc"],
            bufs["vs"],
            filled,
            config=self._lemoq_attn_config,
        )
        self._lemoq_kv_int8_runs = getattr(self, "_lemoq_kv_int8_runs", 0) + 1
    else:
        # ★ 核心替换：mask + q·kᵀ + softmax + w·v  ->  一次 kernel（只读 [0, filled)）
        #   注意这里**不再需要 4D mask**：kernel 不读未填充槽位。
        #   key_states/value_states 是整段 cache（(1,H,max_len,D) 的连续张量），
        #   长度由 kernel 从设备内存读，shape 里没有长度。
        attn_output = attn_decode_forward(
            query_states,
            key_states,
            value_states,
            filled,
            config=self._lemoq_attn_config,
            backend=self._lemoq_attn_backend,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    self._lemoq_decode_calls += 1
    return attn_output, None


def replace_attention_cores(
    model: Any, *, backend: str = "triton", config: dict[str, Any] | None = None
) -> int:
    """把每个 OLMoE attention 模块的 `forward` 换成只换 core 的版本，返回替换个数。

    ★ 用**实例属性**覆盖 `forward`（`nn.Module.__call__` 动态查 `self.forward`），
    而不是改类：类级 patch 会影响同进程里所有模型，还会让"退回原实现"的参照对象
    变得难以推理。原方法存在 `self._lemoq_original_forward` 上，可逆
    （`restore_attention_cores`）。
    """
    import types  # noqa: PLC0415

    patched = 0
    for module in _attention_modules(model):
        if not hasattr(module, "_lemoq_original_forward"):
            module._lemoq_original_forward = module.forward
        module._lemoq_attn_backend = backend
        module._lemoq_attn_config = merged_config(config)
        module._lemoq_decode_calls = 0
        module.forward = types.MethodType(_decode_forward_impl, module)
        patched += 1
    if not patched:
        raise RuntimeError("没有找到任何 OLMoE attention 模块；模型结构可能变了")
    return patched


def attention_cores_replaced(model: Any) -> int:
    """已被替换的 attention 模块数（供 runner/bench 记录与自检）。"""
    return sum(
        1 for module in _attention_modules(model) if hasattr(module, "_lemoq_original_forward")
    )


def attention_int8_runs(model: Any) -> int:
    """E6：所有 attention 模块的 int8 decode 运行计数之和（payload `kv_int8_effective` 的证据）。"""
    return sum(
        int(getattr(module, "_lemoq_kv_int8_runs", 0)) for module in _attention_modules(model)
    )


def decode_call_count(model: Any) -> int:
    """走 kernel core 的 forward 次数（用于测试断言"确实走了新路径"）。"""
    return sum(int(getattr(m, "_lemoq_decode_calls", 0)) for m in _attention_modules(model))


def restore_attention_cores(model: Any) -> int:
    """还原成 HF 原方法（臂之间切换、以及对照实验都需要）。"""
    restored = 0
    for module in _attention_modules(model):
        if hasattr(module, "_lemoq_original_forward"):
            module.forward = module._lemoq_original_forward
            for attr in ("_lemoq_original_forward", "_lemoq_attn_config", "_lemoq_attn_backend"):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
            restored += 1
    return restored
