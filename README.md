# README — LeanMoE-Quant

> **batch=1 单流 MoE 解码：字节预算（byte budget）驱动的量化与 kernel 优化**

**一页纸结果（所有数字带出处）见 [RESULTS.md](RESULTS.md)；图在
[docs/figures/](docs/figures/)。**

状态（2026-10-23）：**阶段 0 / 1（精度门）/ 2（kernel）/ 4b（E7）已执行**；W4 专家量化已判负；头号结论：

- **E2 被否证**：融合反量化不是矩阵预期的 1.3–2×——在预注册的 eager 格里它反而比独立 kernel
  **慢 4.4%**（`host_share` 72.2%、**19,029 kernel/步**，那条路径是 host/launch 受限的）；
- **E7 已判定**：截断效应成立，但**只有融合臂兑现**——同一个 `w8a16` **fused** 实现
  `speedup_eager` 0.78 → `speedup_cudagraph` **1.1974**（ctx=1024）/ **1.0912**（4096）
  （★ 2026-10-24 采纳刷新，F1+F2 默认开），
  `host_share` 77.5% → 1.6%。天花板是**实测**出来的：权重只占 graph 步的 42% / 23%，
  完美权重路径的上限也只有 ~1.74× / ~1.30×。★ 2026-10-08 起**规范实现是融合 RMSNorm**
  （`--rmsnorm fused`），绝对延迟已按新实现重测；
- **E4 已判定**：路由/打包是 hf/gather 基线里最大的桶（**4.478 ms = GPU busy 的 44.2%**），
  且**量化不变**（fp16 → w4a16 只差 1.7%）——它就是计划预测的那个上限；
- **`EXP-p4c-01` 支持**：图内把 HF sdpa 换成自研 split-KV，量化格 **1.4392× / 1.3711×**，
  最好格 **2.27× / 3.73×** vs fp16×sdpa（同上采纳刷新轮）；**`EXP-p4c-02`**：自研融合 RMSNorm 全步
  **1.2358×**，kernel/步 **994 → 539**。

进度、工件与"还没做什么"见 [AGENTS.md](AGENTS.md) 的 §0/§3；权威记录是
`experiments/<EXP-ID>.md`。**未做**：E5 专家敏感度、E6 KV INT8、E8 跨带宽（V100）、
FP8/MXFP4、dense 曲线。（★ 2026-10-10：G2 batch 曲线与 G3 occupancy 已测，见
[RESULTS.md](RESULTS.md) §6。）**ncu 性能计数器在本机被宿主策略阻断**
（`RmProfilingAdminOnly: 1` + 容器无 `CAP_SYS_ADMIN`），已交出替代的 kernel 级证据：
[docs/reference/NCU_EVIDENCE.md](docs/reference/NCU_EVIDENCE.md)。

绝不把本机 Mac 的测试当成 GPU 实测；`results/` 是机器本地工件（gitignored），不进 git。

---

## 30 秒版（点进来先看这里）

**MoE 大模型单卡推理优化**（模型：OLMoE-1B-7B · 显卡：RTX 5090 × 1 · 自研 Triton kernel）

`大模型推理优化 | 算子融合 | 量化(W8A16) | Triton | CUDA Graph | 高性能计算 | MoE`

- 自研 **4 个 Triton kernel**（grouped MoE GEMV / 融合反量化 GEMM / split-KV decode attention /
  融合 RMSNorm），全部通过 `1e-3` 数值 parity 门。
- **融合 RMSNorm：HF 的 7 个 kernel 合成 1 个**，全步 **1.24×**、kernel 数 **994→539**，
  同进程交错 A/B **15/15 轮获胜**；torch 原生 `F.rms_norm` 对照反而慢 5%。
- **无 gather 的 MoE decode kernel**（expert id 只做指针算术，保持 CUDA Graph 可捕获）：
  消除 **3.22 GB/token** 的 gather 复制，per-token **5.26→4.07 ms（1.29×）**。
- **W8A16 量化端到端 1.19×**，并**实测出上限只有 1.74×**（权重仅占一步 42%），
  把 **90%** 收益归因到单个 kernel（专家 GEMV **1.52×**）；
  精度三链闭合：KL **0.00136**、PPL **+0.067%**、MMLU 200 题 **−1.0 个点**。
- **4-bit 被精度门挡住（负结果）**：W4 RTN / GPTQ / ≤25% 整层 W8 退路在 5090 上都过不了门
  （top-1 0.903 / 0.915 / 0.933），所以 4-bit 的延迟收益**不发布**——量化这条杠杆只兑现到 8-bit。
- **已测未采纳（下一步）**：三项融合（RoPE / routing / mask-fill）A/B 已测、代码已写、默认关，
  账面 **≈0.24 ms（12.5%）** 未兑现，见 [FUSION_OPPORTUNITIES](docs/FUSION_OPPORTUNITIES.md) §F1。

每个数字的出处：**[RESULTS.md](RESULTS.md)**（一页纸，逐条带工件路径）；图：
**[docs/figures/](docs/figures/)**；面试必问三题的标准答法与岗位映射：
**[docs/apply/PORTFOLIO.md](docs/apply/PORTFOLIO.md)**。

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
   > 这四条现在各走到哪一步（哪条已答、哪条未答），见 [AGENTS.md](AGENTS.md) §0/§3。

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
（特性矩阵从本地 LLVM 谓词核实，见 [docs/reference/SPECS.md](docs/reference/SPECS.md) §2.1）。
所有结论必须声明适用区间。

## 目录

```text
LeanMoE-Quant/
├── PLAN.md                  # 计划书（唯一权威计划）
├── RESULTS.md               # ★ 一页纸结果（每个数字带出处）
├── AGENTS.md                # 协作/agent 规则（含项目硬规则）
├── ENVIRONMENT.md           # 机器、环境、网络/镜像/代理、模型获取
├── EXPERIMENTS.md           # 实验记录格式与指标定义（强制）
├── docs/
│   ├── README.md             # 文档地图（docs/ 下每份文档给谁看）
│   ├── handoff/              # 交接：当前唯一权威摘要
│   ├── reference/            # 查证：SPECS / MODELS / METRICS / NCU_* / MOE_CUDAGRAPH / RUNBOOK
│   ├── apply/                # 投递：PORTFOLIO（简历 bullet + 面试三题）
│   └── figures/              # ★ 三张图（由 tools/make_figures.py 从工件生成）
├── configs/
│   ├── models.yaml          # 模型清单（唯一来源）
│   ├── experiments.yaml     # 声明式实验矩阵（harness 读它展开）
│   └── constraints.txt      # 冻结点版本
├── scripts/                 # preflight / proxy / bootstrap / fetch_models / check_links
├── experiments/TEMPLATE.md  # 每个实验复制此模板
├── results/                 # JSON/日志落盘（gitignored）
├── src/lemoq/               # quant/ 量化实现、kernels/ 反量化 kernel
├── benchmarks/              # timing / vram / hostgap / bytes / harness / report
├── tools/                    # 设备探测、可达带宽标定、ncu 证据、画图
└── tests/                   # 正确性门槛（先于性能）
```

## 快速开始

环境与基线准备见 [ENVIRONMENT.md](ENVIRONMENT.md)，服务器上机序列见
[docs/reference/SERVER_RUNBOOK.md](docs/reference/SERVER_RUNBOOK.md)。核心命令：

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
- **必须报 host/kernel 拆分**——只报加速比无法解释（见 [docs/reference/METRICS.md](docs/reference/METRICS.md) §2.1）；
- 负结果同样记录——本项目的一半价值在"量化在 MoE 上到底能走多远"的边界。
