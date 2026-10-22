# EXP-p1-05-w4-expert-calibration

> 预留实验记录，**尚未注册矩阵、未执行 GPU 实验**。本次只交付离线实现、测试和工作量估算。
> 新增矩阵条目/正式 cell 或 gate 引用之前仍须确认；不借用 p1-04 的 W4 失败行或 W8 通过行。

- **ID**: `EXP-p1-05-w4-expert-calibration`（reserved）
- **Date**: 2026-10-22
- **Operator**: Mac 离线开发；GPU operator / 实例待确认
- **Status**: `planned`
- **执行规格**: [分支规格与命令](../docs/compose/spec/quant-w4-calibration.md)

## Hypothesis（GPU 前冻结）

在专家 W4、attention W8、激活 FP16、g128 的同一精度计划下，GPTQ 式层输出重建可比 RTN
减小 validation 误差，并可能让配置达到 KL <0.05、top-1 ≥0.95、PPL 相对退化 <1%。
以上均为待检验假设，不是已实现收益。W8 RTN 为稳定性控制；RTN W4 与 GPTQ W4 只改算法。

反证：相同数据与精度计划下重建误差未改善，或仍不过模型精度门；若保留最多 25% 专家参数
在整层 W8 的退路也不能同时过门并减少至少 30% packed 专家字节，则停止扩展，保留负结果。

## Model / Hardware / Runtime

- OLMoE-1B-7B-0924，16 层，64 专家，top-8，H=2048，F=1024；单 GPU、batch=1 decode。
- 目标 RTX 5090 / sm_120 / 32 GB。**本轮 GPU、带宽、时钟和资源占用未测**。
- 沿用已有服务器依赖，不安装升级；实际版本、源码 commit/dirty hash、checkpoint 内容哈希随工件记录。

## Configuration

| 臂 | experts | attention | 激活 | 算法 |
| --- | --- | --- | --- | --- |
| `w8_rtn` | W8 | W8 | FP16 | RTN，稳定性控制 |
| `experts_w4_rtn` | W4 | W8 | FP16 | RTN，精度计划控制 |
| `experts_w4_gptq` | W4 | W8 | FP16 | GPTQ，主要候选 |

共同设置：g128、对称 codes、FP16 scale、既有 nibble 布局；router/norm/embedding/lm_head 保持
原精度；decode grouped + split-KV + fused RMSNorm；三个新融合候选默认关。
GPTQ：FP32 Gram/反馈、block128、damp=0.01、act-order=False、static groups、每专家至少 256 个
样本、最多 2048 个进入 Gram。零信息/覆盖不足/数值失败均拒绝，不静默降级。

train 的 128×512 窗口只做校准，validation 的 16×512 窗口做重建评分/选择；固定 seed=42、
dataset commit 与 tokenizer 哈希。test 不参与选择。最终 ctx1024、ctx4096 的五窗口 ×33 位置
使用已有协议；4096 需要同配置 1024 已通过的证据，test 需要冻结的 validation 证据。

## Results

GPU KL / top-1 / PPL / latency / VRAM / speedup：**未测**。
CPU 数学与集成测试结果见分支规格；不得转述为 5090 正确性或精度通过。
本记录不产生正式 `result.json`，不修改 `GATE_CITATIONS`，不更新任何 done 格。

## Verdict

`planned`。无 GPU verdict。profile 的预算上界不能当作实际加速；诊断可用也不等于正式采纳。

## Applicability

仅针对该模型的单卡 batch=1 decode 和指定精度计划。未覆盖 W8A8 真 INT8 GEMM、FP8/MXFP4、
逐专家混合位宽、多卡、并发服务或量化×batch 扩展。
