# README — LeanMoE-Quant

> **batch=1 单流 MoE 解码：字节预算（byte budget）驱动的量化与 kernel 优化**

状态：**Phase 0：环境与基线已完成；fp16 CUDA Graph feasibility gate 已在本地实现，
待 RTX 5090 上做真实 capture/replay 验证**。
仓库已经包含 RTX 5090 上的 Gate-0 与 fp16 eager 基线记录；CUDA Graph、量化和 kernel
优化仍以 `planned` / `target` / `to be evaluated` 标注，绝不把本机 Mac 测试当成 GPU 实测。

---

## 这个项目在问什么

**立论**：batch=1 解码是纯带宽受限（memory-bound）——每生成一个 token，必须把参与计算的
权重完整流一遍。因此优化的唯一主线是**减少每 token 的字节数**；量化是主杠杆，
反量化与 GEMM 的融合决定这个杠杆是否真兑现。

要验证的四条推论：

1. 量化在 batch=1 收益最大，随 batch 增大被算力开销侵蚀；
2. 反量化必须与 GEMM 融合，否则中间张量的写+读会吃掉大部分收益；
3. MoE 上存在量化收益上限：专家权重字节可降到 1/4，但**路由与打包开销不随量化下降**；
4. **量化对 token 延迟的收益被 host/launch 开销截断**——字节降到 1/4 不等于 token 快 4 倍，
   除非同时消除 launch 开销（CUDA Graph）。

完整计划见 **[PLAN.md](PLAN.md)**（立项理由、硬件、阶段划分、实验矩阵、风险）。

### 关于"端侧"

本项目的**立论**能迁移到端侧（端侧是统一内存 UMA，只有一个内存池，decode 是**更纯粹**的
字节受限问题，所以"减少每 token 字节"这个杠杆在端侧更值钱）；但**数字不能**——kernel 是
CUDA、卡是桌面旗舰。详见 [PLAN.md](PLAN.md) §0.1。

## 与其它两个项目的关系（三者互补，不重复）

| 项目 | 层级 | 回答的问题 |
|---|---|---|
| [MegaKernel-vLLM](../MegaKernel-vLLM/README.md) | 系统 / kernel | 整模型融合 vs CUDA Graph 的收益边界在哪 |
| [MoE-Flow](../MoE-Flow/PROJECT.md) | 系统 / 调度 | 专家放哪、缓存谁、何时预取（Host/Device 混合） |
| **LeanMoE-Quant**（本仓库） | 算法 / kernel | 每个权重用几 bit、反量化怎么融进 GEMM |

组合叙事：**表示（量化）→ 放置（专家管理）→ 执行（融合）**。

**三者需要互斥的硬件**：`MoE-Flow` 需要"模型装不下 + 窄的主机↔设备链路"（BF16-only，
V100 上跑不了）；本项目需要"模型装得下 + 带宽越高越好"。互斥意味着不抢资源。

## 主硬件

- **主开发机：租用 RTX 5090 32 GB（sm_120）**，≈¥3/h，按量计费，单卡课题；
- **第二数据点：学校 V100 32 GB（sm_70，免费）**，用于跨带宽对照；
- **选卡铁律：带宽 > 显存**（OLMoE fp16 只有 13.84 GB，24 GB 已足够）。

5090 有 `cp.async` / TMA / FP8 / MXFP4 block-scale，但 **40/50 系消费卡都没有 `wgmma`**
（特性矩阵从本地 LLVM 谓词核实，见 [docs/SPECS.md](docs/SPECS.md) §2.1）。
所有结论必须声明适用区间。

## 目录

```text
LeanMoE-Quant/
├── PLAN.md                  # 计划书（唯一权威计划）
├── AGENTS.md                # 协作/agent 规则（含项目硬规则）
├── ENVIRONMENT.md           # 机器、环境、网络/镜像/代理、模型获取
├── EXPERIMENTS.md           # 实验记录格式与指标定义（强制）
├── docs/
│   ├── SPECS.md             # 机器规格 + 跨代特性矩阵 + 选卡
│   ├── MODELS.md            # 模型选型 + 已核实的 OLMoE 架构与字节预算
│   ├── METRICS.md           # 指标口径（KL/PPL、[W]/[T]、达成率、host 占比）
│   └── SERVER_RUNBOOK.md    # 服务器到手后的命令序列
├── configs/
│   ├── models.yaml          # 模型清单（唯一来源）
│   ├── experiments.yaml     # 声明式实验矩阵（harness 读它展开）
│   └── constraints.txt      # 冻结点版本
├── scripts/                 # preflight / proxy / bootstrap / fetch_models / check_links
├── experiments/TEMPLATE.md  # 每个实验复制此模板
├── results/                 # JSON/日志落盘（gitignored）
├── src/lemoq/               # quant/ 量化实现、kernels/ 反量化 kernel
├── benchmarks/              # timing / vram / hostgap / bytes / harness / report
├── tools/                   # 设备探测、可达带宽标定
└── tests/                   # 正确性门槛（先于性能）
```

## 快速开始

环境与基线准备见 [ENVIRONMENT.md](ENVIRONMENT.md)，服务器上机序列见
[docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)。核心命令：

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
uv run pytest -m "not gpu"          # 在 Mac 上也能跑（无 GPU 的测试）
bash scripts/preflight.sh           # 网络路由 + GPU 存在性
uv run python -m tools.probe_machine --json results/machine_probe.json
```

## 复现纪律（摘要，完整版见 EXPERIMENTS.md）

- **先正确性、后性能**：与 fp16 参考的 KL 散度 / top-1 一致率不达标，不允许报性能数字；
- **不得用"逐 token 严格相等"**作门槛——bf16 greedy 在 logit 平局处是非确定性的；
- **一次一个变量**；跨阶段数字不可混比；
- **实测可达带宽**用于算理论下限，不用规格值；
- **必须报 host/kernel 拆分**——只报加速比无法解释（见 [docs/METRICS.md](docs/METRICS.md) §2.1）；
- 负结果同样记录——本项目的一半价值在"量化在 MoE 上到底能走多远"的边界。
