"""pack — 4/8-bit 打包与解包。

为什么需要它：

    `docs/METRICS.md` §3.2 的 `[W]` 是在**打包后**的字节数上算的。若只算
    `params x bits / 8`，就漏掉了 per-group scale 的开销——那会让 W4 的收益被高估。
    本模块负责真实的位打包，`bit_cost()` 与 `benchmarks.bytes._quantized_bytes` 对齐。

4-bit 打包约定：**两个 4-bit code 装进一个 uint8**，低半字节在前（little nibble first）。
    byte = (code0 & 0xF) | ((code1 & 0xF) << 4)
    还原：code0 = byte & 0xF ； code1 = (byte >> 4) & 0xF      (符号扩展见下)

对称量化时 code 是 int4（-8..7），解包后要**符号扩展**回 int32。
"""

from __future__ import annotations

from typing import Any


def pack_4bit(codes) -> Any:
    """Pack int4 codes (last dim even) into uint8. Works for numpy or torch arrays."""
    last = codes.shape[-1]
    if last % 2 != 0:
        raise ValueError("最后一维必须是偶数才能按 4-bit 打包，得到 %d" % last)

    if _is_torch(codes):
        import torch

        c = codes.to(torch.int32)
        lo = (c[..., 0::2] & 0xF).to(torch.uint8)
        hi = ((c[..., 1::2] & 0xF) << 4).to(torch.uint8)
        return lo | hi

    import numpy as np

    c = codes.astype(np.int32)
    lo = (c[..., 0::2] & 0xF).astype(np.uint8)
    hi = ((c[..., 1::2] & 0xF) << 4).astype(np.uint8)
    return lo | hi


def unpack_4bit(packed, *, signed: bool = True) -> Any:
    """Inverse of `pack_4bit`. `signed=True` sign-extends to int8 range."""
    if _is_torch(packed):
        import torch

        p = packed.to(torch.int32)
        lo = p & 0xF
        hi = (p >> 4) & 0xF
        out = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
        if signed:
            out = torch.where(out >= 8, out - 16, out)
        return out.to(torch.int8 if signed else torch.uint8)

    import numpy as np

    p = packed.astype(np.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    out = np.stack((lo, hi), axis=-1).reshape(packed.shape[:-1] + (packed.shape[-1] * 2,))
    if signed:
        out = np.where(out >= 8, out - 16, out)
    return out.astype(np.int8 if signed else np.uint8)


def _is_torch(x) -> bool:
    mod = type(x).__module__
    return mod.startswith("torch")


def packed_nbytes(n_params: int, bits: int) -> int:
    """Packed weight bytes for `n_params` at `bits`. Rejects non-multiples of 8 packs."""
    if bits not in (4, 8):
        raise ValueError("只支持 4 或 8 bit")
    return n_params * bits // 8


def scale_nbytes(
    n_params: int, group_size: int, scale_bytes: int = 2, *, in_features: int | None = None
) -> int:
    """Per-group scale overhead. Groups are along the input dim, **per output row**.

    ★ `in_features`（每行的输入维）必须给，否则算出来的组数是错的：

        正确   n_groups = rows × ceil(in / group_size)       rows = n_params / in
        旧写法 n_groups = ceil(n_params / group_size)

    两者只在"每个矩阵的输入维都被 group_size 整除"时相等。反例
    `out=2, in=130, group_size=128`：真实 4 组，旧写法给 3 组 —— **低估** scale 开销。

    `in_features=None` 保留旧行为只是为了兼容不了解形状的调用点；它会在下面
    `bit_cost` 的文档里被明确标为"仅当整除时成立"，不静默假装正确。
    """
    if group_size <= 0:
        return 0
    if in_features is None:
        n_groups = -(-n_params // group_size)
    else:
        in_features = int(in_features)
        if in_features <= 0:
            raise ValueError("in_features 必须 > 0，得到 %r" % (in_features,))
        if n_params % in_features != 0:
            raise ValueError(
                "n_params=%d 不能被 in_features=%d 整除：组数按行算，形状不自洽就没法数"
                % (n_params, in_features)
            )
        rows = n_params // in_features
        n_groups = rows * -(-in_features // group_size)
    return n_groups * scale_bytes


def bit_cost(
    n_params: int,
    bits: int,
    group_size: int,
    scale_bytes: int = 2,
    *,
    in_features: int | None = None,
) -> dict[str, int]:
    """Total quantized footprint — must agree with `benchmarks.bytes`.

    ★ `in_features=None` 时 scale 开销按"整块参数"近似，**仅在输入维被 group_size
      整除时成立**（OLMoE 的 2048/1024/50304 对 32/64/128 都整除，所以本项目已发布的
      字节账不受影响）。能给形状就必须给 —— 见 `scale_nbytes`。
    """
    p = packed_nbytes(n_params, bits)
    s = scale_nbytes(n_params, group_size, scale_bytes, in_features=in_features)
    return {"params": n_params, "packed_bytes": p, "scale_bytes": s, "total_bytes": p + s}
