"""reference — 反量化 GEMM 的两条参考路径（**纯 PyTorch，可在 CPU 上验证**）。

E2 的命题是"融合 vs 独立 kernel"。两条路径的**数学结果相同**，区别只在
**是否物化中间张量**：

    separate : W_hat = dequant(W)          # 写一个 [out, in] 的 fp16 中间张量
               y     = x @ W_hat.T         # 再把它整块读回来   ← 多出来的写+读
    fused    : y     = matmul(x, W, scale) # 在读取 packed 权重时当场解压，不落中间张量

`separate` 多出来的流量 ≈ 权重本身的读写量，所以如果它吃掉大部分收益，
"反量化必须融合"这个判断就得到支持（E2 的假设），否则被否证。

本模块是**正确性 oracle**：真 kernel 的输出必须与它一致（容差见测试）。
它也是 Phase 2 在 GPU 上的**基线臂**（separate 一侧），因此不是可有可无的占位代码。
"""

from __future__ import annotations

from typing import Any


def _torch():
    import torch  # noqa: PLC0415

    return torch


def _dequant_tile(qweight, scale, zero, group_size: int, symmetric: bool):
    """Dequantize an [out, in] code tensor with per-group (along `in`) scale/zero."""
    torch = _torch()
    out_features, in_features = qweight.shape
    n_groups = scale.shape[1]
    padded_in = n_groups * group_size
    if padded_in != in_features:
        pad = torch.zeros(
            (out_features, padded_in - in_features), dtype=qweight.dtype, device=qweight.device
        )
        q = torch.cat([qweight, pad], dim=1)
    else:
        q = qweight
    codes = q.reshape(out_features, n_groups, group_size).to(torch.float32)
    sc = scale.reshape(out_features, n_groups, 1)
    if symmetric:
        w = codes * sc
    else:
        w = (codes - zero.reshape(out_features, n_groups, 1)) * sc
    return w.reshape(out_features, padded_in)[:, :in_features]


def dequant_gemm_separate_reference(
    x, qweight, scale, zero, *, group_size: int, symmetric: bool = True, out_dtype=None
):
    """The 'separate kernel' arm: **materialize** the dequantized weight, then GEMM.

    This is the arm a naive implementation produces, and the one E2 is measured against.
    """
    torch = _torch()
    w_hat = _dequant_tile(qweight, scale, zero, group_size, symmetric)
    y = x.to(torch.float32) @ w_hat.t()
    return y.to(out_dtype) if out_dtype is not None else y


def dequant_gemm_fused_reference(
    x, qweight, scale, zero, *, group_size: int, symmetric: bool = True, out_dtype=None
):
    """The 'fused' arm, expressed in eager PyTorch.

    ★ 诚实说明：这只是**参考实现**，它仍然物化了中间张量，因此**不代表真实的融合
    kernel 的性能**。它的用途是：
        (a) 给出融合路径必须匹配的**数值**；
        (b) 作为 Phase 2 的接口契约。
    真正的融合只在 Triton/CUDA 里实现（读 packed 权重 -> 寄存器/shared 内解压 -> MMA）。
    """
    return dequant_gemm_separate_reference(
        x,
        qweight,
        scale,
        zero,
        group_size=group_size,
        symmetric=symmetric,
        out_dtype=out_dtype,
    )


def tolerance_report(y_ref, y_test) -> dict[str, Any]:
    """Numeric comparison used by the parity test. Tolerances are declared, not implicit."""
    torch = _torch()
    a = y_ref.to(torch.float64)
    b = y_test.to(torch.float64)
    diff = (a - b).abs()
    denom = a.abs().max().clamp(min=1e-12)
    return {
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "max_rel_err": float((diff / denom).max().item()),
        "allclose_atol_1e-3": bool(torch.allclose(a, b, atol=1e-3, rtol=1e-3)),
    }
