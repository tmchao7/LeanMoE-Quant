"""numpy_oracle — 量化/反量化的**纯 numpy 参考实现**（无 torch 依赖）。

它有两个用途：

1. **正确性 oracle**：Triton/CUDA kernel 的输出必须与它逐元素一致（容差写死在测试里）。
   没有 oracle，"我的 kernel 对了"无法证伪。
2. **可在无 GPU 的机器上验证数学**：Mac 上 `python3` 是 3.9.6、可能没有 torch，
   但本模块只需 numpy —— 或者连 numpy 都没有时，退化路径见 `_as_array`。

量化方案（RTN，round-to-nearest）：

    权重 W: [out_features, in_features]，沿 **in_features** 分组，group_size = g
    对称（signed，W4 用 -8..7 或对称取 7）
        scale  = max(|W_group|) / qmax
        q      = clamp(round(W / scale), -qmax-1, qmax)
        W_hat  = q * scale
    非对称（asymmetric，uint，0..2^bits-1）
        scale  = (max - min) / (2^bits - 1)
        zero   = round(-min / scale)
        q      = clamp(round(W / scale) + zero, 0, 2^bits-1)
        W_hat  = (q - zero) * scale

**分组是沿输入维度**（每行独立分组），因为 GEMM 的 K 维是内积维——这是 W4A16 的常规做法。
"""

from __future__ import annotations

import math
from typing import Any

try:
    import numpy as np
except Exception:  # noqa: BLE001 - oracle must degrade gracefully
    np = None  # type: ignore


def _require_numpy() -> None:
    if np is None:
        raise RuntimeError("numpy 不可用；numpy_oracle 需要 numpy")


def qmax_for(bits: int, symmetric: bool = True) -> int:
    return (2 ** (bits - 1)) - 1 if symmetric else (2**bits) - 1


def _pad_to_group(w, group_size: int):
    """Pad the last (incomplete) group with zeros; caller slices back."""
    _require_numpy()
    out_features, in_features = w.shape
    if in_features % group_size == 0:
        return w, in_features
    padded = int(math.ceil(in_features / group_size) * group_size)
    pad = np.zeros((out_features, padded), dtype=w.dtype)
    pad[:, :in_features] = w
    return pad, in_features


def quantize_rtn(
    weight, *, bits: int = 4, group_size: int = 128, symmetric: bool = True
) -> dict[str, Any]:
    """Quantize a 2-D weight with per-group RTN.

    Returns a dict holding packed-free integer codes and the per-group scale
    (and zero-point when asymmetric).
    """
    _require_numpy()
    w = np.asarray(weight)
    if w.ndim != 2:
        raise ValueError("numpy_oracle 只支持 2-D 权重，得到 ndim=%d" % w.ndim)
    if group_size <= 0:
        raise ValueError("group_size 必须 > 0")
    if bits not in (4, 8):
        raise ValueError("本 oracle 只实现 4 与 8 bit，得到 %r" % (bits,))

    padded, orig_in = _pad_to_group(w.astype(np.float32), group_size)
    out_features, padded_in = padded.shape
    n_groups = padded_in // group_size
    grouped = padded.reshape(out_features, n_groups, group_size)

    qmax = qmax_for(bits, symmetric=symmetric)
    if symmetric:
        absmax = np.max(np.abs(grouped), axis=2, keepdims=True)
        scale = np.where(absmax > 0, absmax / qmax, 1.0).astype(np.float32)
        zero = np.zeros_like(scale)
        qmin = -(qmax + 1)
        codes = np.clip(np.round(grouped / scale), qmin, qmax)
    else:
        gmax = np.max(grouped, axis=2, keepdims=True)
        gmin = np.min(grouped, axis=2, keepdims=True)
        scale = np.where((gmax - gmin) > 0, (gmax - gmin) / qmax, 1.0).astype(np.float32)
        zero = np.round(-gmin / scale)
        codes = np.clip(np.round(grouped / scale) + zero, 0, qmax)

    return {
        "codes": codes.astype(np.int32).reshape(out_features, padded_in)[:, :orig_in],
        "codes_padded": codes.astype(np.int32),
        "scale": scale.astype(np.float32).reshape(out_features, n_groups),
        "zero": zero.astype(np.float32).reshape(out_features, n_groups),
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "shape": (out_features, orig_in),
        "n_groups": n_groups,
        "padded_in": padded_in,
    }


def dequantize_rtn(q: dict[str, Any]) -> Any:
    """Inverse of `quantize_rtn`. Returns fp32."""
    _require_numpy()
    out_features = q["shape"][0]
    padded_in = q["padded_in"]
    group_size = q["group_size"]
    n_groups = q["n_groups"]

    codes = q["codes_padded"].reshape(out_features, n_groups, group_size).astype(np.float32)
    scale = q["scale"].reshape(out_features, n_groups, 1)
    zero = q["zero"].reshape(out_features, n_groups, 1)
    if q["symmetric"]:
        w = codes * scale
    else:
        w = (codes - zero) * scale
    return w.reshape(out_features, padded_in)[:, : q["shape"][1]]


def quantization_error(weight, q: dict[str, Any]) -> dict[str, float]:
    """Absolute/relative error of the round trip. Used by the correctness gate."""
    _require_numpy()
    w = np.asarray(weight, dtype=np.float32)
    w_hat = dequantize_rtn(q)
    diff = w - w_hat
    denom = float(np.max(np.abs(w))) or 1.0
    return {
        "max_abs_err": float(np.max(np.abs(diff))),
        "mean_abs_err": float(np.mean(np.abs(diff))),
        "max_rel_err_vs_absmax": float(np.max(np.abs(diff)) / denom),
        "snr_db": float(
            10.0 * math.log10((float(np.mean(w**2)) or 1e-12) / (float(np.mean(diff**2)) or 1e-12))
        ),
    }


def bit_cost(q: dict[str, Any], scale_bytes: int = 2) -> dict[str, int]:
    """Packed size + scale overhead. Matches `benchmarks.bytes._quantized_bytes`."""
    n_params = int(q["codes"].size)
    packed = n_params * q["bits"] // 8
    n_scales = int(q["scale"].size)
    return {
        "params": n_params,
        "packed_bytes": packed,
        "scale_bytes": n_scales * scale_bytes,
        "total_bytes": packed + n_scales * scale_bytes,
        "n_scales": n_scales,
    }
