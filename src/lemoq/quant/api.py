"""api — `lemoq.quant` 的对外稳定接口。

    quantize_linear(weight, bits, group_size, scheme) -> QuantizedLinear
    dequantize_reference(qlinear) -> Tensor        # 数值对照用

`REMEDIATION_LADDER` 记录**精度不达标时的兜底阶梯**。

为什么需要这条阶梯：`docs/METRICS.md` §1 的门槛是 W4 的 KL < 0.05。对 7B MoE 做**朴素
per-group RTN W4**，很可能过不了。而项目纪律是"不达标只记 FAIL、不报性能"——若没有补救
方案，项目就会停在第 4 阶段且没有任何可交付的性能结论。所以先把阶梯写下来。

阶梯**不是**"允许放宽门槛"，而是"在不改门槛的前提下，还有哪些合法手段"：

    1. RTN              —— 基线；最快，精度最差
    2. GPTQ 式逐层重建  —— 用校准集最小化逐层输出误差（需要校准数据）
    3. AWQ 式激活感知缩放 —— 按激活幅度保护显著通道（需要校准数据）
    4. 敏感度驱动混合位宽 —— 对量化敏感的层/专家用更高位宽（E5 的直接产物）
    5. lm_head / attention 保持 fp16，只量化专家（`[W]` 的 68.3% 在专家）

每一档都必须**重新过 KL/top-1 门槛**，并标注它改变了哪些变量（规则 3）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REMEDIATION_LADDER: tuple[dict[str, str], ...] = (
    {
        "step": "1_rtn",
        "method": "round-to-nearest per-group",
        "needs_calibration": "no",
        "expected_accuracy": "worst",
        "effort": "lowest",
    },
    {
        "step": "2_gptq",
        "method": "layer-wise reconstruction (GPTQ-style)",
        "needs_calibration": "yes",
        "expected_accuracy": "good",
        "effort": "high",
    },
    {
        "step": "3_awq",
        "method": "activation-aware scaling (AWQ-style)",
        "needs_calibration": "yes",
        "expected_accuracy": "good",
        "effort": "medium",
    },
    {
        "step": "4_mixed_bits",
        "method": "sensitivity-driven mixed bit-width (E5)",
        "needs_calibration": "yes",
        "expected_accuracy": "best",
        "effort": "medium",
    },
    {
        "step": "5_selective",
        "method": "quantize experts only; keep lm_head/attn fp16",
        "needs_calibration": "no",
        "expected_accuracy": "better",
        "effort": "lowest",
    },
)


@dataclass
class QuantizedLinear:
    """A quantized 2-D weight plus everything needed to dequantize it.

    `codes` is un-packed for clarity in this baseline; `pack.pack_4bit` is used when a
    real memory footprint is required. Byte accounting lives in
    `benchmarks.bytes` / `pack.bit_cost`, not here.
    """

    codes: Any
    scale: Any
    zero: Any
    bits: int
    group_size: int
    symmetric: bool
    scheme: str
    shape: tuple[int, int]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return self.shape[0] * self.shape[1]


def quantize_linear(
    weight, *, bits: int = 4, group_size: int = 128, scheme: str = "rtn", symmetric: bool = True
) -> QuantizedLinear:
    """Quantize a 2-D weight. Only `scheme="rtn"` is implemented in this baseline.

    The signature is deliberately the stable public one so later schemes
    (GPTQ / AWQ) can be added without changing callers.
    """
    if scheme != "rtn":
        raise NotImplementedError(
            "scheme=%r 尚未实现。当前基线的合法取值：'rtn'。"
            "兜底阶梯见 lemoq.quant.REMEDIATION_LADDER。" % (scheme,)
        )
    from lemoq.quant import rtn as _rtn

    q = _rtn.quantize_rtn(weight, bits=bits, group_size=group_size, symmetric=symmetric)
    return QuantizedLinear(
        codes=q["codes"],
        scale=q["scale"],
        zero=q["zero"],
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        scheme=scheme,
        shape=q["shape"],
        meta={"n_groups": q["n_groups"], "padded_in": q["padded_in"], "device": q.get("device")},
    )


def dequantize_reference(qlinear: QuantizedLinear):
    """Reference dequantization. Kernels must match this within a declared tolerance."""
    from lemoq.quant import rtn as _rtn

    q = {
        "codes": qlinear.codes,
        "codes_padded": _repad(qlinear),
        "scale": qlinear.scale,
        "zero": qlinear.zero,
        "bits": qlinear.bits,
        "group_size": qlinear.group_size,
        "symmetric": qlinear.symmetric,
        "shape": qlinear.shape,
        "n_groups": qlinear.meta["n_groups"],
        "padded_in": qlinear.meta["padded_in"],
    }
    return _rtn.dequantize_rtn(q)


def _repad(qlinear: QuantizedLinear):
    """Re-create the padded code tensor the dequantizer expects."""
    mod = type(qlinear.codes).__module__
    padded_in = qlinear.meta["padded_in"]
    in_features = qlinear.shape[1]
    if padded_in == in_features:
        return qlinear.codes
    if mod.startswith("torch"):
        import torch

        out = torch.zeros(
            (qlinear.shape[0], padded_in), dtype=qlinear.codes.dtype, device=qlinear.codes.device
        )
        out[:, :in_features] = qlinear.codes
        return out
    import numpy as np

    out = np.zeros((qlinear.shape[0], padded_in), dtype=qlinear.codes.dtype)
    out[:, :in_features] = qlinear.codes
    return out
