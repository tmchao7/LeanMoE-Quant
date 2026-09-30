"""rtn — RTN 量化的 **torch** 实现，数值必须与 `numpy_oracle` 一致。

为什么要有两份实现：

    - `numpy_oracle` 是**正确性 oracle**（可在任何环境跑，包括没有 torch 的机器）；
    - 本模块是**快路径**（GPU 上跑，供 harness 使用）；
    - `tests/test_quant_oracle.py` 断言两者逐元素一致（容差写死）。
      没有这个断言，快路径悄悄偏离 oracle 时没人会发现。
"""

from __future__ import annotations

import math
from typing import Any


def _torch():
    import torch  # noqa: PLC0415

    return torch


def qmax_for(bits: int, symmetric: bool = True) -> int:
    return (2 ** (bits - 1)) - 1 if symmetric else (2**bits) - 1


def _pad_to_group(t, group_size: int):
    torch = _torch()
    out_features, in_features = t.shape
    if in_features % group_size == 0:
        return t, in_features
    padded = int(math.ceil(in_features / group_size) * group_size)
    pad = torch.zeros((out_features, padded), dtype=t.dtype, device=t.device)
    pad[:, :in_features] = t
    return pad, in_features


def quantize_rtn(
    weight, *, bits: int = 4, group_size: int = 128, symmetric: bool = True
) -> dict[str, Any]:
    """Per-group RTN. Same semantics as `numpy_oracle.quantize_rtn`."""
    torch = _torch()
    w = weight
    if w.ndim != 2:
        raise ValueError("只支持 2-D 权重，得到 ndim=%d" % w.ndim)
    if group_size <= 0:
        raise ValueError("group_size 必须 > 0")
    if bits not in (4, 8):
        raise ValueError("只实现 4 与 8 bit，得到 %r" % (bits,))

    padded, orig_in = _pad_to_group(w.float(), group_size)
    out_features, padded_in = padded.shape
    n_groups = padded_in // group_size
    grouped = padded.reshape(out_features, n_groups, group_size)

    qmax = qmax_for(bits, symmetric=symmetric)
    if symmetric:
        absmax = grouped.abs().amax(dim=2, keepdim=True)
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        zero = torch.zeros_like(scale)
        qmin = -(qmax + 1)
        codes = (grouped / scale).round().clamp(qmin, qmax)
    else:
        gmax = grouped.amax(dim=2, keepdim=True)
        gmin = grouped.amin(dim=2, keepdim=True)
        rng = gmax - gmin
        scale = torch.where(rng > 0, rng / qmax, torch.ones_like(rng))
        zero = (-gmin / scale).round()
        codes = ((grouped / scale).round() + zero).clamp(0, qmax)

    return {
        "codes": codes.to(torch.int32).reshape(out_features, padded_in)[:, :orig_in],
        "codes_padded": codes.to(torch.int32),
        "scale": scale.reshape(out_features, n_groups),
        "zero": zero.reshape(out_features, n_groups),
        "bits": bits,
        "group_size": group_size,
        "symmetric": symmetric,
        "shape": (out_features, orig_in),
        "n_groups": n_groups,
        "padded_in": padded_in,
        "device": str(w.device),
    }


def dequantize_rtn(q: dict[str, Any], dtype=None):
    """Inverse of `quantize_rtn`."""
    out_features = q["shape"][0]
    padded_in = q["padded_in"]
    group_size = q["group_size"]
    n_groups = q["n_groups"]

    codes = q["codes_padded"].reshape(out_features, n_groups, group_size).float()
    scale = q["scale"].reshape(out_features, n_groups, 1)
    zero = q["zero"].reshape(out_features, n_groups, 1)
    w = codes * scale if q["symmetric"] else (codes - zero) * scale
    w = w.reshape(out_features, padded_in)[:, : q["shape"][1]]
    return w.to(dtype) if dtype is not None else w


def quantization_error(weight, q: dict[str, Any]) -> dict[str, float]:
    w = weight.float()
    w_hat = dequantize_rtn(q)
    diff = w - w_hat
    denom = float(w.abs().max().item()) or 1.0
    return {
        "max_abs_err": float(diff.abs().max().item()),
        "mean_abs_err": float(diff.abs().mean().item()),
        "max_rel_err_vs_absmax": float(diff.abs().max().item() / denom),
        "snr_db": float(
            10.0
            * math.log10(
                (float((w**2).mean().item()) or 1e-12) / (float((diff**2).mean().item()) or 1e-12)
            )
        ),
    }
