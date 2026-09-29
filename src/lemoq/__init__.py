"""lemoq — byte-budget-driven quantization and kernel optimization for MoE decode.

本包当前**没有实现任何算法**（Phase 0）。规划中的模块与顺序：

1. `quant/`      — W8A8 / W8A16 / W4A16 量化实现（per-group / per-channel）
2. `kernels/`    — Triton（先）与 CUDA（后）的反量化融合 GEMM
3. `eval/`       — 正确性门槛（KL / top-1 / PPL）与性能测量
4. `models/`     — 模型加载（**强制 fp16**，V100 无 bf16）与逐层统计

见根目录 `PLAN.md` 与 `docs/`。
"""

__version__ = "0.1.0"
