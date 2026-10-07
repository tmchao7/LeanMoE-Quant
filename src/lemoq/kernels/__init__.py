"""lemoq.kernels — 反量化融合 GEMM / MoE kernel。

**当前状态**：本目录提供

    reference              纯 PyTorch 参考实现（fused 与 separate 两条路径的**数学**等价）
    autotune               Triton/CUDA kernel 的参数空间声明 + ncu 指标清单
    _attn_triton           split-KV decode attention kernel（GPU 路径专属）
    _grouped_triton        fp16 grouped MoE GEMV
    _grouped_triton 的量化版  moe_grouped_quant：**expert 侧**在融合张量上量化、
                            decode 在 kernel 内解包（`EXP-p4b-01` 的 P1，8-bit）
    _dequant_triton        **dense 侧**的反量化融合 GEMV（`EXP-p2-01` 的 fused 臂，4/8-bit）

三条 kernel 族的落地顺序与本项目的依赖链一致（见 `EXP-p4b-01-truncation` §3）：
先 execution（cudagraph）、再 expert 侧量化（P1）、再 attention 侧融合（P2-01）。

为什么 kernel 与 host 逻辑分文件：

    Triton kernel 的价值全在**性能**与**真机语义**，只有在 GPU 上才能验证；
    而 `_*_triton.py` 在没有 CUDA 后端的机器（Mac）上装不了。把它们与 host 逻辑分开，
    离线测试才跑得起来（回退逻辑、形状检查、索引算术都是离线可测的）。

    `tests/test_kernel_parity.py` 里的 `@pytest.mark.gpu` 就是这些 kernel 的数值门
    （PARITY_GATE，1e-3/1e-3，不得为了通过而放宽）。
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
