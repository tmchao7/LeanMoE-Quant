"""mxfp4 — MXFP4(OCP MX v1.0)主机侧量化：**存储格式**实验的前半段。

为什么：4-bit 诊断（EXP-p1-05）判死的是 **int4-RTN**（fp16 组尺度 + 整数格点）；
MXFP4 是不同的格式（E2M1 码 + 32 元素共享 **E8M0 幂 2 尺度**，4.25 bits/权重），
是 5090（sm_120）的差异化存储格式，也是"4-bit 是否可行"的最后一次裁决
（侦察与实施坑清单：`docs/reference/QUANT_FORMAT_RECON_2026-10-25.md`）。

本模块只做**主机侧**（量化/反量化/打包，torch float32 精度做 RTN），
kernel 侧（load uint8 → nibble 解包 → e2m1 查表 → `exp2(scale−127)`）另起。
规范要点（实施坑清单 §2.3）：

    - 块 = 32 元素共享一个 E8M0 字节尺度，值 = 2^(X−127)（bias 127，无 inf/NaN）；
    - 共享指数 `X = floor(log2(amax)) − emax_elem`，e2m1 的 emax_elem = 2（最大正常值 6）；
    - E2M1 可表示幅值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}，RTN 就近取整；
    - **幅值 > 6·scale 的值饱和到 ±6**（该格式已知 wart，NVFP4 用 fp 尺度正为此因）——
      所以误差界是 2·scale（饱和区），不是 1·scale；
    - 全零块的尺度字节记 0（X=−127），码全 0；
    - E2M1 编码：bit3=符号，bit2..0 = 幅值码（0..7 对应上表）。
"""

from __future__ import annotations

from typing import Any

_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # 幅值码 0..7
_EMAX_ELEM = (
    2  # e2m1：2^emax − 1 = 最大正常值 3? —— 不：规范取 floor(log2 amax) − emax，e2m1 emax=2
)
BLOCK = 32
_E8M0_BIAS = 127


def e2m1_decode(code: int) -> float:
    """4-bit e2m1 码（bit3 符号 + bit2..0 幅值）→ 实值。"""
    mag = _E2M1_MAG[code & 0x7]
    return -mag if (code & 0x8) else mag


def e8m0_to_scale(byte: int) -> float:
    """E8M0 字节 → 2^(byte−127)（无 inf/NaN；0 字节 = 2^-127，全零块的合法占位）。"""
    return 2.0 ** (byte - _E8M0_BIAS)


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


def quantize_mxfp4(weight: Any, *, block: int = BLOCK) -> dict[str, Any]:
    """fp32/fp16 权重 → MXFP4：打包 uint8 码（低 nibble 在前，复用 `pack.pack_4bit`）+ E8M0 尺度。

    返回 {"codes_packed": uint8 (out, in/2), "scales": uint8 (out, in/block),
          "orig_in", "padded_in", "block"}。K 必须 pad 到 block 的倍数（pad 值 0，
    不影响 absmax；e2m1 code 0 解码为 0.0，与现有 padded-K 方案兼容）。
    """
    torch = _torch()
    if torch is None:
        raise RuntimeError("mxfp4 量化需要 torch")
    from lemoq.quant.pack import pack_4bit  # noqa: PLC0415

    if weight.ndim != 2:
        raise ValueError("只支持 2-D 权重，得到 ndim=%d" % weight.ndim)
    orig_in = weight.shape[1]
    pad = (-orig_in) % block
    padded = torch.nn.functional.pad(weight.float(), (0, pad))
    out_features, padded_in = padded.shape
    if padded_in % block != 0:
        raise ValueError("pad 后 K 仍不是 %d 的倍数" % block)
    blocks = padded.reshape(out_features, padded_in // block, block)

    amax = blocks.abs().amax(dim=-1)  # (out, n_blocks)
    # X = floor(log2(amax)) − 2；amax=0 的块 X 记 −127（尺度字节 0，码全 0）。
    x = torch.where(
        amax > 0,
        torch.floor(torch.log2(amax)) - _EMAX_ELEM,
        torch.full_like(amax, -_E8M0_BIAS),
    ).clamp(-_E8M0_BIAS, _E8M0_BIAS)
    scale_bytes = (x + _E8M0_BIAS).to(torch.uint8)  # (out, n_blocks)
    scale = torch.exp2(x)  # 2^(X)，全零块 = 2^-127（下溢为 0 的极小值，码仍全 0）

    # RTN 就近（并列远离零：规范 round-to-nearest-even 的幅值实现里，
    # 从格点中点选偶码——这里用"距离 + 平局选较小幅值码"等价实现）。
    mags = torch.tensor(_E2M1_MAG, dtype=torch.float32)  # (8,)
    norm = blocks / scale.unsqueeze(-1)  # 幅值域（带符号）
    mag = norm.abs().clamp(max=_E2M1_MAG[-1])  # 饱和到 ±6（格式 wart，如实保留）
    # 就近：|mag − mags| 最小的码；平局（如 2.5 介于 2 与 3）取较小码（= 较小幅值）。
    d = (mag.unsqueeze(-1) - mags).abs()  # (out, n_blocks, block, 8)
    code_mag = d.argmin(dim=-1)  # argmin 平局取最小索引 ⇒ 较小幅值码
    codes = ((norm < 0).to(torch.uint8) << 3) | code_mag.to(torch.uint8)
    codes = codes.reshape(out_features, padded_in).to(torch.uint8)

    return {
        "codes_packed": pack_4bit(codes),
        "scales": scale_bytes.reshape(out_features, padded_in // block),
        "orig_in": orig_in,
        "padded_in": padded_in,
        "block": block,
    }


def dequantize_packed(
    packed: Any, scales: Any, orig_in: int, *, block: int = BLOCK, dtype: Any = None
) -> Any:
    """从打包缓冲反量化（`QuantLinear` 的 separate 臂与 CPU 参照共用同一语义）。"""
    torch = _torch()
    from lemoq.quant.pack import unpack_4bit  # noqa: PLC0415

    out_features, n_blocks = scales.shape[0], scales.shape[1]
    padded_in = n_blocks * block
    codes = unpack_4bit(packed, signed=False).to(torch.uint8).reshape(out_features, padded_in)
    mag = torch.tensor(_E2M1_MAG, dtype=torch.float32)[(codes & 0x7).long()]
    signed = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    scale = scales.float().apply_(e8m0_to_scale)
    w = (mag * signed).reshape(out_features, n_blocks, block)
    w = w * scale.unsqueeze(-1)
    w = w.reshape(out_features, padded_in)[:, :orig_in]
    return w.to(dtype) if dtype is not None else w


def dequantize_mxfp4(q: dict[str, Any], *, dtype: Any = None) -> Any:
    """MXFP4 → 浮点权重（CPU 参照；kernel 侧用 exp2 算术解码复刻同一语义，见 K1）。"""
    return dequantize_packed(
        q["codes_packed"], q["scales"], q["orig_in"], block=q["block"], dtype=dtype
    )
