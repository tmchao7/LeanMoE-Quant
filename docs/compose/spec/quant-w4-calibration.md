# 专家 W4 校准分支 — 离线交付与上机边界

分支：`codex/quant-w4-calibration`，基于 `review/fusion-2026-10-22` / `70f4610`，完整承接原工作区。
已有修改已按验收工具、入口契约/独立检查、文档三组本地提交；后续分为校准实现与文档提交。
没有推送、连接服务器或启动 GPU；矩阵和历史工件保持不变。

预留记录：[EXP-p1-05](../../../experiments/EXP-p1-05-w4-expert-calibration.md)。它尚未进入矩阵，
不能把本文件的 CPU 验证或估算写成已完成科学实验。

## 实现

- [`gptq.py`](../../../src/lemoq/quant/gptq.py)：显式 `HessianStats`，FP32 Gram 和逐列误差反馈；
  block128、damp 1%、无置换、静态分组，量化反馈使用实际 FP16 scale。
- `quantize_linear(..., scheme="gptq", calibration=stats)` 是新增入口；RTN 默认保持原语义。
  GPTQ v1 只支持对称 INT4、偶数且整除输入宽度的组；统计缺失、零信息和数值失败均报错。
- [`expert_calibration.py`](../../../src/lemoq/quant/expert_calibration.py)：逐层前向，不反复重放全模型；
  gate/up 共用一次统计与重建，量化后重新收集 down 输入。下一层吃实际重建后的输出。
  样本覆盖/层评分/完整层工件持续保存，失败目录不可作为完整 bundle 加载。
- `replace_moe_blocks(..., prequantized=...)` 先校验所有层，再直接安装 packed buffers；禁止同时
  传 `quant`。此路径不调用 RTN、不融合浮点权重，避免校准结果被覆盖；既有 kernel 本体不变。
- 控制组和候选均为 attention W8、激活 FP16。混合配置显式记 `experts4_attention8_activation16`，
  不伪装成旧的统一 W4A16，也不继承旧引用门。

## 数据准备（CPU，可在上机前完成）

[`prepare_w4_data`](../../../tools/prepare_w4_data.py) 从已存在的 WikiText-2 raw split 文本生成缓存，
不下载、不安装依赖。需要本地 tokenizer、固定数据集 40 位 commit、`train.txt` / `validation.txt` /
`test.txt`。相同 tokenizer/文本连接规则适用于全部 split；保存原文、文件及 int64 token 哈希。

```bash
# 默认只打印计划；路径为用户实际准备的目录，不依赖租卡实例。
.venv-mac/bin/python -m tools.prepare_w4_data \
  --model /path/to/local/tokenizer --text-dir /path/to/wikitext-2-raw \
  --revision <DATASET_COMMIT_SHA> --out /path/to/new/w4-data
# 在本地输入齐全后加 --execute，执行 CPU tokenization。
```

校准固定抽取 128×512 train 窗口，validation 抽取 16×512；seed=42，无放回、窗口不重叠。
校准不读取 test tokens。每专家实际 seen 与进入 Gram 的 samples 分开保存；samples ≥256，cap=2048。
出现稀有专家覆盖不足时停止导出，不自动增加数据或将其升级为 W8。

## 工作量与资源估算（非实测）

运行 `.venv-mac/bin/python -m tools.w4_calibrate --phase plan` 可重算下表。

| 项 | 默认模型估算 |
| --- | --- |
| 校准 / 验证 tokens | 65,536 / 8,192 |
| 原始专家矩阵 | 16×64×3 = 3,072；gate/up 合并重建 |
| Cholesky 分解次数 | 16×64×4 = 4,096（每个 gate/up 与 down 均两次） |
| GPTQ train layer forward | 16×128×3 = 6,144（gate 统计、down 统计、下一层传播） |
| validation layer forward / 臂 | 16×16×2 = 512（浮点层参考与重建层输出） |
| 专家权重 / scale 字节 | W8 = 6,543,114,240；W4 = 3,321,888,768；均为形状推算，非测量 |
| 单层最大 Gram | 64×2048²×4 = 1 GiB |
| 单专家分解工作区下限 | 4×2048²×4 = 64 MiB；不含库内部 workspace |
| CPU hidden cache / 份 | 144×512×2048×2 = 288 MiB；传播期间存在多份 |
| 单层 INT32 导出暂存 | 64×3×2048×1024×4 = 1.5 GiB；另加 packed 与 stack 临时副本 |

以上还须加上完整模型驻留、当前层激活、代码/scale 临时张量、CUDA allocator 和库工作区，
**不是 32 GB 能运行的峰值保证**。多个臂串行执行，GPU 只驻留一份模型；CPU/磁盘需容纳完整
packed bundles。GPU 秒数、租卡金额均为 `null`：本轮没有足以换算 wall time 的测量。

## 预算确认后的流程（本轮未执行）

[`w4_calibrate`](../../../tools/w4_calibrate.py) 的 `plan / calibrate / validate` 默认 dry-run，
实际执行必须同时给 `--execute` 和已确认的 `--gpu-budget-seconds`。输出必须是新目录。
预算按执行过程 wall clock 控制，在 forward、专家重建及 profile 间检查；单个库调用无法抢占，
因此属于有检查粒度的停止边界，不是 GPU 硬抢占计时器。

1. 检查实例/软件与外部进程；不杀同租户，不安装升级。先运行默认 W8 和新增 packed GPU parity。
2. `calibrate --arm w8_rtn` 导出控制 bundle；`validate` 在 validation 上核对独立 fp16 HF、
   PPL 和 graph/eager。HF 参考的 RMSNorm 与项目默认 fused 实现一致。
3. 控制通过后，两个 W4 臂分别新目录执行，提供 `--w8-bundle` 和通过的 `--w8-evidence`。
   RTN W4 与 GPTQ W4 使用相同数据/精度计划；不同时扫 group size、非对称或融合开关。
4. W4 validation 失败且 graph/eager 本身通过，才允许给 `--fallback-from` 和 `--parent-bundle`。
   根据完整 W4 bundle 的层输出误差排序，最多 25% 专家参数的整层恢复 W8，重新逐层校准其余 W4。
   不复用不匹配分布的旧 W4 权重；相对实际 W8 控制 bundle 的专家字节须减少至少 30%。
5. 方案冻结后，test 评估必须提供该 bundle 的通过 validation `--freeze-evidence`；
   `--ctx 4096` 还必须有同 bundle、同 split 的 ctx1024 通过证据 `--ctx1024-evidence`。
6. 只有精度门通过才允许 `validate --profile`，同一 session 首尾采集，host-share ≤5%、
   canary 漂移 ≤5%。最多输出 q/k/v 投影融合一条待评估规格；其预算上界不是已实现收益。
   正式 W8/W4 性能 A/B、矩阵注册与 gate 引用仍需独立批准和验收。

上机 CLI 骨架（先 dry-run，不填写未知预算）：

```bash
python -m tools.w4_calibrate --phase calibrate --arm w8_rtn \
  --model <LOCAL_SNAPSHOT> --data-manifest <W4_DATA>/data_manifest.json \
  --train-cache <W4_DATA>/train.pt --validation-cache <W4_DATA>/validation.pt \
  --out <NEW_W8_BUNDLE> --dry-run

python -m tools.w4_calibrate --phase validate --bundle <BUNDLE> \
  --model <LOCAL_SNAPSHOT> --data-manifest <W4_DATA>/data_manifest.json \
  --validation-cache <W4_DATA>/validation.pt --ctx 1024 \
  --out <NEW_VALIDATION_RUN> --dry-run

python -m tools.w4_calibrate --phase calibrate --arm experts_w4_gptq \
  --model <LOCAL_SNAPSHOT> --data-manifest <W4_DATA>/data_manifest.json \
  --train-cache <W4_DATA>/train.pt --validation-cache <W4_DATA>/validation.pt \
  --w8-bundle <W8_BUNDLE> --w8-evidence <W8_VALIDATION_RUN>/manifest.json \
  --out <NEW_GPTQ_BUNDLE> --dry-run
```

执行时将 `--dry-run` 替换为 `--execute --gpu-budget-seconds <CONFIRMED_BUDGET>`；未知预算不能填假值。
test PPL 使用前 8192 token / 8×1024 窗口；ctx4096 的五窗口门至少需 20480 个 test token，
现有 8k 缓存不足时明确报错，不能截短窗口。

## 离线验收与当前边界

独立 NumPy FP64 非分块 oracle 检查分块反馈、FP16 scale 和 codes；小矩阵相关输入验证输出重建
目标，零信息、秩亏与非法统计分别测试。真实 tiny MoE 验证 gate/up 共用输入、down 统计重新采集、
逐层输出传播、非专家参数保持原样、packed 导出/重载及“不再次 RTN”的推理集成。
另覆盖 split/哈希/冻结证据、预算耗尽、部分 bundle、参数上限与防覆盖；GPU 新用例只 collection。

最终 CPU 结果：**656 passed / 32 deselected**（当前仓库 657——后续 packed parity 用例 +1）；
其中新增 66 个 CPU 用例与 1 个 GPU 用例。
ruff、black（103 个文件）、check_docs（57 份 Markdown）和 `git diff --check` 全通过。
新增 GPU 用例只完成 collection。旧极值 oracle 的一条 overflow warning 保留，不放宽任何门。

**2026-10-23 已在 RTX 5090 上按本规格执行一轮诊断**：W8 控制过门（KL 0.001154 / top-1 0.981818 /
PPL +0.0443%）；W4 RTN、W4 GPTQ 与 ≤25% 整层 W8 退路都**未过门**（退路字节 −36.92%、top-1 0.9333、
PPL +4.19%）。按预注册反证条件停止扩展：**未做 ctx=4096、test split、性能 A/B**，峰值显存与正式
速度仍未测。数字、逐层 MSE、provenance 与工件位置见
[EXP-p1-05 记录](../../../experiments/EXP-p1-05-w4-expert-calibration.md) §GPU 诊断结果。
本轮不注册矩阵、不修改 gate 引用，不变更默认融合，不重跑 done cells。此前离线修复状态见
[前序记录](offline-validation-2026-10-22.md（已退役，git 历史）)。
