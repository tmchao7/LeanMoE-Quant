"""lemoq.kernels — 反量化融合 GEMM。

**当前状态**：本目录提供

    reference   纯 PyTorch 参考实现（fused 与 separate 两条路径的**数学**等价）
    autotune    Triton/CUDA kernel 的参数空间声明 + ncu 指标清单

**尚未提供**：真正的 Triton / CUDA kernel。它留到有 GPU 时写。

为什么这样安排（而不是现在写好）：

    Triton kernel 的价值全在**性能**，而性能只有在真机上才能验证；离线写出的一大段
    未经编译的 kernel 无法被发现错误，反而会在计费机器上制造调试时间。
    因此离线阶段只交付**能被验证的部分**：参考实现 + oracle + 参数空间 + 测试骨架。

    `tests/test_kernel_parity.py` 已经写好骨架（`@pytest.mark.gpu`），
    上机后填入真 kernel 即可立刻得到数值对照。
"""

from __future__ import annotations

from lemoq.kernels.autotune import DEQUANT_GEMM_AUTOTUNE_SPACE, NCU_METRICS  # noqa: F401
from lemoq.kernels.reference import (  # noqa: F401
    dequant_gemm_fused_reference,
    dequant_gemm_separate_reference,
)

__all__ = [
    "DEQUANT_GEMM_AUTOTUNE_SPACE",
    "NCU_METRICS",
    "dequant_gemm_fused_reference",
    "dequant_gemm_separate_reference",
]
