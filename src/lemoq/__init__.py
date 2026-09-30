"""lemoq — byte-budget-driven quantization and kernel optimization for MoE decode.

模块与状态（Phase A 完成后的实况）：

1. `quant/`      **已实现**。RTN W4/W8 per-group 量化，含纯 numpy oracle（正确性基准）
                 与 torch 快路径（两者断言 bit-identical）。精度不达标时的兜底阶梯见
                 `quant.REMEDIATION_LADDER`。
2. `kernels/`    **部分实现**。`reference` 给出 E2 两条臂（separate / fused）的数学等价
                 实现；`autotune` 声明参数空间与 ncu 指标清单。
                 真正的 Triton/CUDA kernel 留到有 GPU 时写——离线写出的未编译 kernel
                 无法验证，只会在计费机器上制造调试时间。
3. `eval/`       **尚未建立**。正确性门槛（KL / top-1 / PPL）与性能测量目前由
                 根目录 `benchmarks/` 承担（bytes / timing / vram / hostgap / schema /
                  harness / report）。
4. `models/`     **尚未建立**。模型加载与逐层统计；目标精度为 **fp16**（V100 无 bf16
                 张量核），注意 checkpoint 自述为 bfloat16，需转换并记录落差。

口径与纪律见根目录 `PLAN.md`、`AGENTS.md` 与 `docs/`（`METRICS.md` §3.2 定义了
`[W]` 与 `[T]` 两个字节分母，两者不得混用）。
"""

__version__ = "0.1.0"
