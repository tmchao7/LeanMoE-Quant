# README — LeanMoE-Quant

> **端侧约束下的 MoE 解码：字节预算（byte budget）驱动的量化与 kernel 优化**

状态：**计划阶段（Phase 0 / 环境与基线准备）**。
本仓库目前只有计划与工程骨架，**没有任何实测数字**。任何未测量的值都以 `planned` /
`target` / `to be evaluated` 标注，绝不写成已达成结果。

---

## 这个项目在问什么

**立论**：端侧 batch=1 解码是纯带宽受限（memory-bound）——每生成一个 token，必须把参与
计算的权重完整流一遍。因此优化的唯一主线是**减少每 token 的字节数**；量化是主杠杆，
反量化与 GEMM 的融合决定这个杠杆是否真兑现。

要验证的三条推论：

1. 量化在 batch=1 收益最大，随 batch 增大被算力开销侵蚀；
2. 反量化必须与 GEMM 融合，否则中间张量的写+读会吃掉大部分收益；
3. MoE 上存在量化收益上限：专家权重字节可降到 1/4，但**路由与打包开销不随量化下降**。

完整计划见 **[PLAN.md](PLAN.md)**（立项理由、硬件约束、阶段划分、实验矩阵、风险）。

## 与其它两个项目的关系（三者互补，不重复）

| 项目 | 层级 | 回答的问题 |
|---|---|---|
| [MegaKernel-vLLM](../MegaKernel-vLLM/README.md) | 系统 / kernel | 整模型融合 vs CUDA Graph 的收益边界在哪 |
| [MoE-Flow](../MoE-Flow/PROJECT.md) | 系统 / 调度 | 专家放哪、缓存谁、何时预取（Host/Device 混合） |
| **LeanMoE-Quant**（本仓库） | 算法 / kernel | 每个权重用几 bit、反量化怎么融进 GEMM |

组合叙事：**表示（量化）→ 放置（专家管理）→ 执行（融合）**。

## 主硬件

- 主开发机：**V100 32 GB（sm_70）**——学校导师提供，单卡课题，边际成本 ≈ 0；
- 对照机：**可租 RTX 5090 / 4090 / 48GB 专业卡**，只在需要跨带宽对照或 30B 单点时开。

**V100 的三个硬约束**（详见 [docs/SPECS.md](docs/SPECS.md)）：无 BF16、无 `cp.async`/TMA、
带宽仅 ~900 GB/s。三者都必须在报告中显式声明适用区间。

## 目录

```text
LeanMoE-Quant/
├── PLAN.md              # 计划书（唯一权威计划；README 只给立论与入口）
├── AGENTS.md            # 协作/agent 规则（含项目硬规则）
├── ENVIRONMENT.md       # 机器、环境、模型获取、上机清单
├── EXPERIMENTS.md       # 实验记录格式与指标定义（强制）
├── docs/
│   ├── SPECS.md         # 机器规格与 sm_70 硬约束
│   ├── MODELS.md        # 模型候选与选型理由
│   └── METRICS.md       # 指标口径（KL/PPL、带宽达成率、显存分列）
├── configs/models.yaml  # 模型清单（唯一来源）
├── experiments/
│   └── TEMPLATE.md      # 每个实验复制此模板
├── results/             # JSON/日志落盘（gitignored；稳定后摘录进提交）
├── src/                 # 量化实现、反量化 kernel、评估入口（待实现）
├── benchmarks/          # 计时与显存口径的统一实现（待实现）
├── tools/               # 设备探测、可达带宽标定、结果汇总
└── tests/               # 正确性门槛（先于性能）
```

## 快速开始

尚未实现。当前可做的只有阶段 0 的环境与基线准备，步骤见
[ENVIRONMENT.md](ENVIRONMENT.md) 与 [PLAN.md](PLAN.md) §5。

## 复现纪律（摘要，完整版见 EXPERIMENTS.md）

- **先正确性、后性能**：与 fp16 参考的 KL 散度 / top-1 一致率不达标，不允许报性能数字；
- **一次一个变量**；跨阶段数字不可混比；
- **实测可达带宽**用于算理论下限，不用规格值；
- 负结果同样记录——本项目的一半价值在"量化在 MoE 上到底能走多远"的边界。
