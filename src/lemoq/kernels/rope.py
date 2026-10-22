"""fused rope — decode 路径 `apply_rotary_pos_emb` 的单 kernel 融合候选（默认关）。

## 为什么值得做（证据：`results/graph_profile.json`，wall 1.94 ms / busy 1.876 ms）

HF 的 `rotate_half` 链每层 10 个 kernel、全步 160 个，**散在四个 family 里**：

| family                        | 次数/step | ms/step |
|-------------------------------|----------|---------|
| `neg_kernel`（`-x2`）          |       32 |  0.0454 |
| `CatArrayBatchedCopy`（拼接）  |       32 |  0.0545 |
| `BinaryFunctor` mul（q·cos 等）|       64 |  0.0917 |
| add 的 rope 份额               |      ~32 |  ~0.024 |
| **合计**                       |  **160** | **~0.216（≈11.5% busy）** |

全是 q/k 小张量（1×16×1×128 / 1×4×1×128 fp16），纯 launch/延迟 bound。
融合成 **每层 2 个 kernel**（q、k 各一），预期把 ~0.19 ms 收回到 ~0.03 ms。

## 契约

- `apply_rotary_pos_emb_fused(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)`
  与 HF 同签名、同语义（out-of-place，返回新张量）；数学与 HF 逐项一致
  （`rotate_half` 半切：`cat((-x2, x1))`），kernel 内 fp32 计算、按原 dtype 舍入。
- **形状/环境守卫**：只处理 decode 形状 `(1, H, 1, D)`（D 为偶数）且 CUDA +
  triton 可用；其余（prefill、batch>1、CPU/Mac）**退回 HF 原函数**——不静默算错。
- 开关 `LEMOQ_FUSED_ROPE`（默认关）由 call site（`attn_decode`）读取；
  本模块只提供实现。Python 相位读取 ⇒ capture 烘焙所选实现。
- K1（fused vs HF，`PARITY_GATE` 1e-3）由 GPU 测试钉住；回退路径与 HF **逐位相同**。
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["apply_rotary_pos_emb_fused", "fused_rope_available", "warm"]


def _fused_rope_enabled() -> bool:
    """候选开关（默认关 = 走 HF 原路，已录格子零影响）。"""
    return os.environ.get("LEMOQ_FUSED_ROPE", "0") == "1"


try:  # pragma: no cover - triton 只在 GPU 机器上存在
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 — Mac（无 triton）必须能 import 本模块
    _HAS_TRITON = False


def fused_rope_available() -> bool:
    """这台机器上 kernel 臂能不能用（Mac 上 False）。查询不抛异常。"""
    return _HAS_TRITON


def _hf_apply(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    position_ids: Any = None,
    unsqueeze_dim: int = 1,
) -> tuple[Any, Any]:
    """HF 原函数（回退路径；本地 import 保持模块离线可 import）。"""
    from transformers.models.olmoe.modeling_olmoe import (  # noqa: PLC0415
        apply_rotary_pos_emb,
    )

    return apply_rotary_pos_emb(
        q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim
    )


_TORCH_TO_TRITON_DTYPE = None
if _HAS_TRITON:  # pragma: no cover - 仅 GPU 机器
    import torch  # noqa: E402

    _TORCH_TO_TRITON_DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }

    @triton.jit
    def _rope_one_kernel(
        x_ptr,
        out_ptr,
        cos_ptr,
        sin_ptr,
        D,
        sx_h,
        sx_d,
        sc_d,
        so_h,
        so_d,
        DTYPE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """单个张量（q 或 k）的 rope：`out = x*cos + rotate_half(x)*sin`。

        - program = 一个 head；`BLOCK >= D`（2 的幂），`offs < D` 掩掉尾巴；
        - stride 全部从 host 传入（q/k 是 transpose 视图，不假设连续）；
        - B=L=1（host 守卫保证），所以 batch/seq 维偏移恒为 0；
        - fp32 计算、按 `DTYPE` 舍入 store（K1 ≤1e-3 由 parity 测试钉住）。
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        m = offs < D
        d2 = D // 2
        # rotate_half：d<d2 → -x[d+d2]；d≥d2 → +x[d-d2]
        shift = tl.where(offs < d2, offs + d2, offs - d2)
        sign = tl.where(offs < d2, -1.0, 1.0)

        base = pid * sx_h + offs * sx_d
        x = tl.load(x_ptr + base, mask=m, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + offs * sc_d, mask=m, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + offs * sc_d, mask=m, other=0.0).to(tl.float32)
        xs = tl.load(x_ptr + pid * sx_h + shift * sx_d, mask=m, other=0.0).to(tl.float32)

        y = x * c + sign * xs * s
        tl.store(out_ptr + pid * so_h + offs * so_d, y.to(DTYPE), mask=m)


def _decode_shape_guard(q: Any, k: Any, cos_u: Any, sin_u: Any) -> bool:
    """只处理 decode 形状；其余回退 HF（不静默算错）。"""
    import torch  # noqa: PLC0415

    if not (_HAS_TRITON and q.is_cuda and k.is_cuda):
        return False
    if q.ndim != 4 or k.ndim != 4:
        return False
    if q.dtype not in (torch.float16, torch.float32):
        return False  # bf16 的 1 ulp ≈ 8e-3 过不了 K1 的 1e-3 —— 本项目只跑 fp16
    if any(t.dtype != q.dtype or t.device != q.device or not t.is_cuda for t in (k, cos_u, sin_u)):
        return False  # mixed types promote differently in HF; mixed devices cannot launch safely
    if q.shape[0] != 1 or k.shape[0] != 1:
        return False  # batch=1 only
    if q.ndim != 4 or k.ndim != 4 or q.shape[2] != 1 or k.shape[2] != 1:
        return False  # decode q_len=1
    if q.shape[3] != k.shape[3]:
        return False
    d = int(q.shape[3])
    if d % 2 != 0 or d < 2:
        return False
    if tuple(cos_u.shape) != (1, 1, 1, d) or tuple(sin_u.shape) != (1, 1, 1, d):
        return False
    if cos_u.stride(-1) != sin_u.stride(-1) or q.shape[1] < 1 or k.shape[1] < 1:
        return False
    return True


def apply_rotary_pos_emb_fused(
    q: Any,
    k: Any,
    cos: Any,
    sin: Any,
    position_ids: Any = None,
    unsqueeze_dim: int = 1,
) -> tuple[Any, Any]:
    """HF `apply_rotary_pos_emb` 的融合版：每层 2 个 kernel（q、k 各一）。

    任何不满足守卫的输入（CPU、prefill、batch>1、dtype 不支持、cos 形状不是
    decode 的 (1,1,1,D)）都**原样退回 HF 实现**——回退与 HF 逐位相同。

    `position_ids` 非 None 保留保守 HF 回退。固定的 transformers 4.57.6 中该参数
    已弃用且未使用；这项限制不是“修复 HF 按位置索引”的证据。
    """
    if position_ids is not None:
        return _hf_apply(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    if not _decode_shape_guard(q, k, cos_u, sin_u):
        return _hf_apply(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)

    d = int(q.shape[3])
    dtype = _TORCH_TO_TRITON_DTYPE.get(q.dtype)
    if dtype is None:  # 守卫已覆盖；保底不静默算错
        return _hf_apply(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)

    import torch  # noqa: PLC0415

    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    block = triton.next_power_of_2(d)
    for x, out, cos_t, sin_t in ((q, qo, cos_u, sin_u), (k, ko, cos_u, sin_u)):
        heads = int(x.shape[1])
        _rope_one_kernel[(heads,)](
            x,
            out,
            cos_t,
            sin_t,
            d,
            int(x.stride(1)),
            int(x.stride(3)),
            int(cos_t.stride(3)),
            int(out.stride(1)),
            int(out.stride(3)),
            DTYPE=dtype,
            BLOCK=block,
        )
    return qo, ko


def warm(*, heads: int = 16, kv_heads: int = 4, head_dim: int = 128, device: Any = None) -> None:
    """capture 之前 JIT 编译 + 真实 launch（对齐 `moe_grouped.warm` 惯例）。"""
    if not _HAS_TRITON:
        return
    import torch  # noqa: PLC0415

    dev = device if device is not None else torch.device("cuda")
    q = torch.randn(1, heads, 1, head_dim, dtype=torch.float16, device=dev)
    k = torch.randn(1, kv_heads, 1, head_dim, dtype=torch.float16, device=dev)
    # 与 position_embeddings 同形状：(B, L, D) —— unsqueeze(1) 后才是 (1,1,1,D)
    cos = torch.randn(1, 1, head_dim, dtype=torch.float16, device=dev)
    sin = torch.randn(1, 1, head_dim, dtype=torch.float16, device=dev)
    apply_rotary_pos_emb_fused(q, k, cos, sin)
    torch.cuda.synchronize(dev)
