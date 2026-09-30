"""lemoq.quant — 量化实现（RTN 起步，兜底阶梯见 `api.REMEDIATION_LADDER`）。

模块分工：

    numpy_oracle    纯 numpy 参考实现 —— **正确性 oracle**，无 torch 依赖
    rtn             torch 快路径 —— 数值必须与 oracle 一致
    pack            4/8-bit 打包与解包，含 scale 开销的字节核算
    api             对外稳定接口

设计原则（来自 `AGENTS.md` 的项目硬规则）：

    - **正确性先于性能**：量化后必须先过 KL / top-1 门槛才允许报性能（规则 2）；
    - **一次一个变量**：bit-width 与 group size 不同时改（规则 3）；
    - **不编造数字**：未测的写 `None` / `not_measured`，不写 `0`。
"""

from __future__ import annotations

from lemoq.quant.api import (  # noqa: F401
    REMEDIATION_LADDER,
    QuantizedLinear,
    dequantize_reference,
    quantize_linear,
)

__all__ = [
    "REMEDIATION_LADDER",
    "QuantizedLinear",
    "dequantize_reference",
    "quantize_linear",
]
