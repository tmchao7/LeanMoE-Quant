# EXP-p1-05-w4-expert-calibration

> 预留实验记录，**未注册矩阵**。2026-10-23 已在 RTX 5090 上执行一轮完整诊断（W8 控制 → RTN/GPTQ
> 两个 W4 臂 → 25% 整层 W8 退路），结果是**负结果**，见 §GPU 诊断结果。
> 仍不注册矩阵条目、不改 `GATE_CITATIONS`、不借用 p1-04 的 W4 失败行或 W8 通过行。

- **ID**: `EXP-p1-05-w4-expert-calibration`（reserved）
- **Date**: 2026-10-22（设计/离线实现）· 2026-10-23（GPU 诊断执行）
- **Operator**: Mac 离线开发 + 5090 诊断执行
- **Status**: `planned`（矩阵未注册；GPU 诊断已执行，负结果，不进入扩展）
- **执行规格**: [分支规格与命令](../docs/compose/spec/quant-w4-calibration.md)

## Hypothesis（GPU 前冻结）

在专家 W4、attention W8、激活 FP16、g128 的同一精度计划下，GPTQ 式层输出重建可比 RTN
减小 validation 误差，并可能让配置达到 KL <0.05、top-1 ≥0.95、PPL 相对退化 <1%。
以上均为待检验假设，不是已实现收益。W8 RTN 为稳定性控制；RTN W4 与 GPTQ W4 只改算法。

反证：相同数据与精度计划下重建误差未改善，或仍不过模型精度门；若保留最多 25% 专家参数
在整层 W8 的退路也不能同时过门并减少至少 30% packed 专家字节，则停止扩展，保留负结果。

## Model / Hardware / Runtime

- OLMoE-1B-7B-0924，16 层，64 专家，top-8，H=2048，F=1024；单卡、batch=1 decode。
- GPU：NVIDIA GeForce RTX 5090，32607 MiB，sm_120，driver 595.71.05，单卡。
- **带宽本轮未重测**：本轮只做精度判定，不用带宽做任何屋顶/加速结论。此前实例实测见
  `results/machine_probe.json` 与 [HANDOFF_2026-10-10](../docs/handoff/HANDOFF_2026-10-10.md)（1529.5 GB/s，只读 1560.3）。
- Runtime：Python 3.12.3、torch 2.8.0+cu128、triton 3.4.0、transformers 4.57.6（取自工件 `runtime`）。
- 代码 provenance：所有臂的 `source.code_sha256_including_untracked` = `a3918930f952…`。W8 与
  W4-RTN 工件头写 `commit=a216bf7 dirty=true`（未提交的是文档），W4-GPTQ/退路/四个验证写
  `commit=3f79ade dirty=false`；**同一份代码**产生了全部臂。
- 数据：WikiText-2 raw v1，固定 revision `b08601e04326c79dfdd32d625aee71d232d685c3`；
  tokenizer sha256 `f0594b0a…97d93`；train/validation/test 三份缓存哈希互异，seed=42。
  校准 128×512 train 窗口，validation 16×512 窗口只做重建评分，**test 未读取**。

## Configuration

| 臂 | experts | attention | 激活 | 算法 |
| --- | --- | --- | --- | --- |
| `w8_rtn` | W8 | W8 | FP16 | RTN，稳定性控制 |
| `experts_w4_rtn` | W4 | W8 | FP16 | RTN，精度计划控制 |
| `experts_w4_gptq` | W4 | W8 | FP16 | GPTQ，主要候选 |
| `experts_w4_gptq` + 25% 整层 W8 退路 | 4/16 层 W8 RTN + 12/16 层 GPTQ W4 | W8 | FP16 | 反证路径 |

共同设置：g128、对称 codes、FP16 scale、FP16 激活、既有 nibble 布局；router/norm/embedding/lm_head
保持原精度；decode grouped + split-KV + fused RMSNorm；三个新融合候选关闭。
GPTQ：FP32 Gram/反馈、block128、damp=0.01、act_order=False、static groups、每专家 ≥256 样本、
≤2048 样本进 Gram、FP16 scale 参与反馈。

## GPU 诊断结果（2026-10-23，RTX 5090）

### 精度：validation，ctx=1024，5 窗口 × 32 个 greedy 位置

门限（`QUANT_THRESHOLDS`，W4）：KL <0.05、top-1 ≥0.95、PPL 相对退化 <1%；`same_path` 必须为真。

| 臂 | KL | top-1 | PPL（ref → cand） | PPL Δ | KL 窗口 SD | same_path | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W8 RTN（控制） | **0.001154** | **0.981818** | 6.739495 → 6.742482 | **+0.0443%** | 0.000576 | true | **PASS** |
| W4 RTN | 0.044720 | 0.903030 | 6.739495 → 7.035026 | +4.3851% | 0.022153 | true | FAIL |
| W4 GPTQ | 0.054835 | 0.915152 | 6.739495 → 7.221355 | +7.1498% | 0.037748 | true | FAIL |
| W4 GPTQ + 25% 整层 W8 | 0.026455 | 0.933333 | 6.739495 → 7.021895 | +4.1902% | 0.011459 | true | FAIL |

逐窗口（KL / top-1）：

- W8：`0.000367, 0.001919, 0.000960, 0.001436, 0.001089` / `0.9697, 1, 1, 0.9394, 1`
- W4 RTN：`0.023072, 0.045136, 0.040344, 0.081474, 0.033575` / `0.9697, 0.9091, 0.9394, 0.7273, 0.9697`
- W4 GPTQ：`0.025537, 0.071314, 0.048521, 0.111021, 0.017782` / `0.9697, 0.9394, 0.8788, 0.7879, 1`
- 退路：`0.016213, 0.032082, 0.028706, 0.041436, 0.013839` / `0.9697, 0.9394, 0.8788, 0.8788, 1`

### 逐层 validation 相对 MSE（整层输出，跨层传播）

| 层 | W4 RTN | W4 GPTQ | 退路（选层后） | 退路算法 |
| --- | --- | --- | --- | --- |
| 00 | 0.004314 | **0.006367** | 0.000013 | W8 RTN |
| 01 | 0.003329 | 0.003661 | 0.003648 | GPTQ W4 |
| 02 | 0.001418 | 0.000991 | 0.001013 | GPTQ W4 |
| 03 | 0.000187 | 0.000207 | 0.000208 | GPTQ W4 |
| 04 | 0.000276 | 0.000298 | 0.000298 | GPTQ W4 |
| 05 | 0.000308 | 0.000388 | 0.000384 | GPTQ W4 |
| 06 | 0.000462 | 0.000563 | 0.000559 | GPTQ W4 |
| 07 | 0.000550 | 0.000661 | 0.000656 | GPTQ W4 |
| 08 | 0.000727 | 0.000823 | 0.000818 | GPTQ W4 |
| 09 | 0.001101 | 0.001252 | 0.001241 | GPTQ W4 |
| 10 | 0.001629 | 0.001833 | 0.001825 | GPTQ W4 |
| 11 | 0.002515 | 0.002551 | 0.002544 | GPTQ W4 |
| 12 | 0.002994 | 0.003684 | 0.003678 | GPTQ W4 |
| 13 | 0.003698 | **0.004730** | 0.000011 | W8 RTN |
| 14 | 0.004296 | **0.005060** | 0.000015 | W8 RTN |
| 15 | 0.003376 | **0.004054** | 0.000010 | W8 RTN |

### packed 专家字节（实际文件内 tensor 账，不是精度或速度）

| 臂 | packed 字节 | 相对 W8 | 备注 |
| --- | --- | --- | --- |
| W8 RTN | 6,543,114,240 | — | 控制组 |
| W4 RTN | 3,321,888,768 | **−49.23%** | |
| W4 GPTQ | 3,321,888,768 | **−49.23%** | 与 RTN 同布局 |
| 退路 | 4,127,195,136 | **−36.92%** | 4/16 层整层 W8，满足 ≥30% 目标 |

退路选层是机械规则的结果（按完整 GPTQ bundle 的 `validation_relative_mse` 倒序，在 ≤25%
专家参数预算内整层恢复 W8）：**L0、L13、L14、L15**（L0 因为 GPTQ 的首层误差 0.006367 最大而入选）。

### 工件与 provenance

服务端（`/root/autodl-tmp/lemoq-w4-data/`，未入库）：

| 目录 | 状态 | bundle manifest sha256 |
| --- | --- | --- |
| `w8-rtn-v2-20261023` | complete 16/16 | `f1148d3f…b968` |
| `w8-validate-1024-20261023` | validated（PASS） | 同上（被引用的 sha） |
| `w4-rtn-20261023` | complete 16/16 | `78459e62…f5cf` |
| `w4-rtn-validate-1024-20261023` | validated（FAIL） | `78459e62…f5cf` |
| `w4-gptq-20261023` | complete 16/16 | `3c2b353a…36ba` |
| `w4-gptq-validate-1024-20261023` | validated（FAIL） | `3c2b353a…36ba` |
| `w4-gptq-fallback-20261023` | complete 16/16 | `ffafcba9…a45f` |
| `w4-gptq-fallback-validate-1024-20261023` | validated（FAIL） | `ffafcba9…a45f` |
| `w8-rtn-a216bf7-20261023` | **failed**（保留） | 首次 W8 导出，`582e6013` 代码误拒 FP16 scale 下溢；已由 v2 取代 |

Mac 侧镜像：`lemoq-artifacts/EXP-p1-05-2026-10-23/`（与仓库平级、**非历史目录**，不入 git）：
三个 W4 bundle + W8 控制 bundle、全部 `manifest.json` 与运行日志；每层文件可按 manifest 里的
`sha256` 逐个复核。日志在 `logs/`。

### 观测耗时（墙钟，±5 s；不是经过 canary 的正式计时）

W8 导出 19:04:27、验证 19:06:02（≈95 s）；W4 RTN 导出 19:07:56、验证 19:09:37（≈100 s）；
GPTQ 导出 ≈8 min、验证 ≈100 s；退路导出 ≈8 min、验证 ≈110 s。峰值显存**未按 allocator 分项测量**
（`nvidia-smi` 观测不作为 VRAM 结论）。

## Results

**GPU 精度诊断（负结果）：**

1. GPTQ 式重建在本模型上**没有**减小 validation 层重建误差：16 层里 15 层比 RTN 更差，首层
   0.006367 vs 0.004314；模型级 KL 0.0548（比 RTN 的 0.0447 更差）、PPL +7.15%（RTN +4.39%），
   只有 top-1 略好（0.9152 vs 0.9030）。假设的第一条反证（“重建误差未改善”）成立。
2. 两个完整 W4 臂都**未过**模型精度门：RTN 的 KL 0.0447 恰好低于 0.05，但 top-1 0.903 与
   PPL +4.39% 都不过；GPTQ 连 KL 都过不了。
3. 预注册的退路（最多 25% 专家参数整层 W8）减少了 **36.92%** packed 字节（门槛 ≥30%），并把
   KL 拉到 0.0265、top-1 拉到 0.9333、KL 窗口 SD 降到 0.0115，**但 PPL 仍 +4.19%、top-1 仍 <0.95**，
   门仍不过。按假设的第二条反证，**停止扩展**，保留负结果。

本轮不产生正式 `result.json`，不修改 `GATE_CITATIONS`，不更新任何 done 格，不注册矩阵。
没有做 ctx=4096、test split 或性能 A/B：它们都以“ctx=1024 精度过门”为前置条件。

## Verdict

`planned`（矩阵未注册）。**GPU 诊断 verdict：负结果（hypothesis refuted）。** 在 OLMoE-1B-7B-0924、
RTX 5090、attention W8 / 激活 FP16 / g128 的精度计划下，专家 W4（RTN 与 GPTQ）都过不了本项目
的精度门，且 ≤25% 整层 W8 的退路也不能同时满足“过门 + 减少 ≥30% 字节”。按计划书中的反证
条件停止扩展；profile 的预算上界不能当作实际加速，诊断可用也不等于正式采纳。

## Applicability

仅针对该模型的单卡 batch=1 decode 和上述精度计划；5 窗口 ×32 位置是共享诊断协议，不是矩阵门。
未覆盖 W8A8 真 INT8 GEMM、FP8/MXFP4、逐专家混合位宽、>25% 的 W8 退路、多卡、并发服务或
量化×batch 扩展；也不构成“GPTQ 方法普遍无效”的证据（本轮 GPTQ 的样本上限 2048、真实专家
激活分布与 train/validation 迁移未单独归因）。
