# RESULTS — 一页纸（所有数字都有出处）

> 目标读者：**五分钟**。每个数字后面都跟着工件或记录的路径；没有出处的数字一律不写。
> 口径、门槛、以及"哪些数字不可比"见 [EXPERIMENTS.md](EXPERIMENTS.md) 与
> [docs/METRICS.md](docs/METRICS.md) §1。逐格记录见 [`experiments/`](experiments/)。
>
> **前提**：OLMoE-1B-7B-0924（64 专家 / top-8，16 层，hidden 2048），batch=1 单流 decode，
> RTX 5090（sm_120），**实测可达带宽 1561.9 GB/s**（不是规格值）。
> 参考实现为 fp16，性能数字只在**过了正确性门**的配置上报（硬规则 2）。

## 1. 项目在问什么

batch=1 的 decode 是**纯带宽受限**：每生成一个 token，要把参与计算的权重完整流一遍
（fp16 下 **2.36 GB/token**）。所以唯一的主线是减少每 token 的字节数 —— 量化是主杠杆，
而"这个杠杆能不能兑现"取决于反量化、注意力实现与 **launch 开销**。

起点有多远：fp16 eager 单步 **82.6436 ms**（[记录](experiments/EXP-p0-02-baseline-fp16.md) §P2j 的"记录采用"列、
工件 `results/EXP-p0-02-baseline-fp16__ctx=1024/result.json`），而字节下限是 **1.5982 ms** ⇒
**带宽达成率 1.9%**，离带宽墙 **≈52×**。这 52 倍就是本项目全部的空间。

> 口径注：该记录正文里的 100.102 ms / 1.6% 是 2026-09-30 首轮的值，已被 10-06 的重跑
> 覆盖（payload 为权威，硬规则 7）。这个格子是 host 主导的，绝对延迟本身不稳定 ——
> 它的作用是"起点有多远"，不是精确基线。

![离带宽墙有多远](docs/figures/fig2_to_the_wall.svg)

> 空心条是**另一个轮次**的参照，**不属于**那条链 —— 把它接进链里会造出一个假的
> "8.36× 杠杆"，这正是本项目禁止的跨轮混比（硬规则 3）。链从 E7 那一轮的 eager 开始。

## 2. 五条杠杆，每条都有自己的 A/B

| # | 杠杆 | 实测 | 口径 | 出处 |
| --- | --- | --- | --- | --- |
| 1 | **执行轴**（eager → cudagraph，同一实现、**同一轮**） | **2.43×** @1024 / **1.35×** @4096 | fp16：9.8841 → 4.0670 / 9.8716 → 7.3211 ms；`host_share` 67.3% → 1.4%。★ 用归档里那一轮的两个 payload，**不**用"pre-P2i eager vs post-P2j graph"（跨轮，会造出假的 2.75×） | 归档 `results/EXP-p2j-refresh/archive/EXP-p4b-01-*`；[EXP-p4b-01](experiments/EXP-p4b-01-truncation.md) §2.8.2 |
| 2 | **量化**（w8a16 vs fp16，图内 fused 臂） | **1.1911×** @1024 / **1.0888×** @4096 | 同一实现只换 bit-width；canary ±0.13% | 同上 §P2j（采用列 3.5881 / 3.0124） |
| 3 | **注意力实现**（HF sdpa → 自研 split-KV，图内） | **1.3034×** @1024 / **1.2555×** @4096（w8a16 格） | vs **同格 fp16**；最好格 **1.87× / 3.11×** vs fp16×sdpa | [EXP-p4c-01](experiments/EXP-p4c-01-in-graph-attention.md) §5.7 |
| 4 | **融合 RMSNorm**（7 kernel → 1 kernel） | **1.2358×** 全步；kernel/步 **994 → 539** | 同进程交错 A/B，**15/15 轮获胜**；同一个 cudagraph 格在 P2j 前后 **4.0670 → 3.5881 ms（1.13×）** 是它的端到端兑现 | [EXP-p4c-02](experiments/EXP-p4c-02-elementwise-tail.md) §5.6 |
| 5 | **投影 kernel 修复**（每元素 gather → 每 tile 装载+广播） | 每调用 **8.14 → 5.15 µs**（**523 → 826 GB/s**） | 图内配置扫描；模型桶 0.5245 → 0.3254 ms | [EXP-p4c-01](experiments/EXP-p4c-01-in-graph-attention.md) §5.4 |

![每条杠杆各自的同轮 A/B](docs/figures/fig1_per_axis_speedups.svg)

★ **不要把这几行相乘当"总加速"**：它们来自不同轮次/不同上下文，本项目的规定是**只报同轮 A/B**
（硬规则 3）。乘法的意义留给读者，不写进结论。

## 3. 为什么只有 1.19×：上限是实测出来的

这是本项目**最有价值的一条**，也是唯一能解释"量化在 MoE 上到底能走多远"的东西：

| 量 | 值 | 出处 |
| --- | --- | --- |
| E7 步级量化收益（w8a16，图内） | **1.1911×** / 1.0888× | [EXP-p4b-01](experiments/EXP-p4b-01-truncation.md) §P2j |
| 权重流占一个 graph 步的比例 | **42%**（1024）/ **23%**（4096） | 同上 §4 |
| ⇒ 完美权重路径的**结构性上限** | **~1.74×** / **~1.30×** | 同上 |
| 专家 GEMV 实测占一步 GPU 时间 | **1.540 ms / 3.5881 ms = 42.9%** | [docs/NCU_EVIDENCE.md](docs/NCU_EVIDENCE.md) §4.3 |
| kernel 级量化收益（专家 GEMV，隔离测） | 96.3 → 63.5 µs = **1.52×** | 同上 §3 |
| ⇒ 该 kernel 解释了步级收益的 | **90.4%**（16 层 × Δ0.029 ms = 0.524 / 0.580 ms） | 同上 §4.3 |

**读法**：量化确实兑现了，但它能兑现的天花板**不是 4×，是 1.74×** —— 因为权重只占一步的 42%。
而且我们已经把这个上限的**九成**拿到了，并且能定位到具体 kernel。

![kernel 级证据：必要字节 vs 实测时间](docs/figures/fig3_kernel_attainment.svg)

> 这三张图的每一个输入都带来源，见 [`docs/figures/inputs.json`](docs/figures/inputs.json)；
> 生成脚本 `tools/make_figures.py` **读不到工件就报错**，并且自带画布越界自检。

## 4. 负结果（同样是一等公民）

| 结论 | 证据 | 为什么它值钱 |
| --- | --- | --- |
| **融合反量化不是 1.3–2×，反而慢 4.4%** | [EXP-p2-01](experiments/EXP-p2-01-dequant-fusion.md)：eager 格里 `fused/separate` = **1.0442**，`host_share` 72.2%、**19,029 kernel/步** | 那条路径是 host/launch 受限的 ⇒ "融合"本身不是收益来源，"把 launch 消掉"才是 |
| **torch 的 `F.rms_norm` 比 HF 原链慢 5%** | [EXP-p4c-02](experiments/EXP-p4c-02-elementwise-tail.md) §5.5：2.5474 vs 2.4143 ms，**0/15 轮获胜**（虽然数值等价） | 它保留了同一个 `MeanOps` reduce，却多出 65 次 fp32 行乘 ⇒ 只有自研 kernel 能赢 |
| **权重 ÷4 拿不到延迟 ÷4** | [docs/NCU_EVIDENCE.md](docs/NCU_EVIDENCE.md) §4.2：字节 ÷3.88，时间只 ÷2.24，达成率 71% → 44% | 瓶颈从 DRAM 挪到了 kernel 内解包 —— 这才是"字节预算"的真实边界 |
| **`EXP-p0-03`（HF 路径 + cudagraph）** | 记录里是 `blocked` **by design**：动态 cache 不可 capture | 说明"能不能进图"是这条路径的硬约束 |

还有一条**内部的负结果**：`EXP-p4c-02` 的模块级归因被它自己预注册的 K1 判据否掉了
（eager/graph top-8 kernel 重叠只有 5/8）—— 记录里写的是"这条归因 **VOID**"，而不是把它当结论引用。

## 5. 工程量（能跑、能测、能复现）

| 项 | 量 |
| --- | --- |
| 自研 kernel 族（全部过 K1 parity 门 1e-3） | **4**：grouped MoE GEMV（fp16 + 8/4-bit）、dense 融合反量化 GEMM、split-KV decode attention、融合 RMSNorm |
| Python（不含虚拟环境） | 24,874 行 / 75 文件 |
| 测试 | 23 个测试文件；Mac CPU **399 passed / 0 failed**；服务器含 GPU 用例 **392 passed / 7 skipped / 0 failed** |
| 实验矩阵 | 21 条声明 / 64 格展开；**16 条已执行**，`results/**/result.json` 98 份（81 done / 13 planned / 4 blocked） |
| 度量纪律 | 实测带宽而非规格值；KL/top-1 门先于性能；schema 强制"门失败 ⇒ latency 为 null"；首尾同格金丝雀；同租户记录不 kill；覆盖前先归档 |

## 6. 明确的缺口（不含糊）

| 缺口 | 状态 |
| --- | --- |
| **ncu 性能计数器** | **宿主策略阻断**（`RmProfilingAdminOnly: 1` + 容器无 `CAP_SYS_ADMIN`），连 `gpu__time_duration.sum` 都被拒。已交出替代口径（解析字节 / 隔离计时）与可复核的工具；缺的是 stall/occupancy 归因。见 [docs/NCU_EVIDENCE.md](docs/NCU_EVIDENCE.md) §1 |
| FP8 / MXFP4（sm_120 的 block-scale） | 矩阵已声明（`EXP-p3-01/02`），**未跑** |
| E5 专家敏感度 / E6 KV INT8 / E8 跨带宽（V100） | 矩阵已声明，**未跑** |
| batch > 1 的吞吐轴 | 本项目是**有意**的 batch=1 单流课题；E3 在矩阵里没有条目 |
| 18 个 host 主导格子的刷新 | 刷新值 VOID（共租 CPU），规范 payload 保留 pre-P2i 版本；列为 P2k。结论不依赖它 |
| 一处口径不一致（在修） | `AGENTS.md` §0 写的 `1.1927` 用了金丝雀首值 3.0084 做分母；**记录采用值 3.0124 ⇒ 1.1911**。本文用 1.1911 |

## 7. 一句话总结

> 在一个 82.6 ms → 1.6 ms 字节下限的带宽受限 decode 上，本项目用**执行方式、量化、
> 注意力实现、融合 norm、投影 kernel** 五条各自独立的 A/B 把差距收窄，并**实测出**
> "权重只占一步 42% ⇒ 量化上限 1.74×"这个结构性边界；其中九成收益已定位到具体 kernel。
> 四个自研 Triton kernel 全部有 parity 门，负结果与阻断同样留在库里。
