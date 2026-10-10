# PORTFOLIO — 面向实习投递的项目自述（每句话都能追到工件）

> 用途：**投递与面试**。写简历时从这里挑句子，面试前用 §3 的三问自检。
> 规矩不变：本文不引入任何新数字，每个数字都指向本仓库里的记录或工件；
> 没测的东西一律写"未测"（与 [`RESULTS.md`](../../RESULTS.md) §6 同一口径）。
> 生成时间：2026-10-09。

---

## 1. 三十秒版（自我介绍时怎么说）

> 我在单张 RTX 5090 上做一个 **MoE 大模型 decode 的推理优化**课题：OLMoE-1B-7B（64 专家 / top-8）
> 的单请求 decode 起手是 **82.64 ms/token**，而它的字节下限是 **1.60 ms**（实测带宽 1561.9 GB/s），
> 差 **52 倍**。我做了 **4 个自研 Triton kernel**（grouped MoE GEMV、融合反量化 GEMV、
> split-KV decode attention、融合 RMSNorm），并用**五条各自独立、同一轮内 A/B** 的杠杆把差距收窄；
> 其中最有价值的一条是**实测出量化的结构性上限只有 1.74×**（因为权重只占一步的 42%），
> 并把其中九成收益定位到具体 kernel。所有结论都有产物：门禁、金丝雀、负结果都在库里。

**三个数字记忆点**：`82.6 → 3.59 ms`（同一实现链上的实测改进）、`1.74×`（量化上限，实测）、`52×`（起手离带宽墙的距离）。

---

## 2. 简历 bullet（中文版 / 英文版）

**中文（挑 3–4 条放简历）**

1. **自研 4 个 Triton kernel 族**（grouped MoE GEMV、dense 融合反量化 GEMM、split-KV decode attention、
   融合 RMSNorm），全部通过 `atol=rtol=1e-3` 的数值 parity 门；工程上强制"门未过 ⇒ 不报性能数字"。
2. **融合 RMSNorm：把 HuggingFace 的 7 个 kernel 合成 1 个**（OLMoE 每步 65 个 norm），
   全步 **1.2358×**、kernel 数 **994 → 539**，同进程交错 A/B **15/15 轮获胜**；
   对照发现 torch 原生 `F.rms_norm` 反而慢 5%（0/15 轮获胜）。
3. **无 gather 的 MoE decode kernel**：expert id 只参与指针算术（保住 CUDA Graph 可捕获性），
   消除 **3.22 GB/token** 的 gather 税，per-token **5.2637 → 4.0715 ms（1.293×）**，
   且 fp32 累加使数值比原路径更准。
4. **量化端到端**：W8A16 per-group 量化 + 图内执行，步级 **1.1911×**（ctx=1024）；
   并**实测出收益的结构性上限 1.74×**（权重仅占一步 42%），把九成收益归因到具体 kernel
   （专家 GEMV **96.3 → 63.5 µs = 1.52×**，解释步级收益 **90.4%**）。
5. **量化精度三重证据闭合**：分布级 KL **0.001361** / top-1 **0.9758**（门 PASS）、
   token 级 PPL **9.1985 → 9.2047（+0.067%）**、任务级 MMLU 5-shot（200 题子集）
   **53.50% → 52.50%（−1.0 个点**，门槛 <2 点）。
6. **实验方法论**：64 格实验矩阵 + 首尾同格金丝雀（漂移 >5% 判整轮 VOID）；
   三轮实测判 VOID 后**不发布数字**，负结果（融合反量化被否证）与阻断同样入库。

**English（resume-ready）**

1. Built **4 custom Triton kernels** (grouped-MoE GEMV, fused dequant GEMM, split-KV decode attention,
   fused RMSNorm) with enforced `1e-3` numerical parity gates.
2. **Fused RMSNorm fused 7 HF kernels into 1** for OLMoE decode: **1.24×** end-to-end step speedup,
   kernel count **994 → 539**, won **15/15** interleaved A/B rounds.
3. **Gather-free MoE decode kernel** (expert id as pointer arithmetic, CUDA-Graph safe): removed a
   **3.22 GB/token** gather tax, **5.26 → 4.07 ms/token (1.29×)**.
4. **W8A16 quantization**: **1.19×** step-level speedup with CUDA Graph; measured the **structural
   ceiling of 1.74×** (weights are only 42% of a decode step) and attributed **90%** of the gain to a
   single kernel (expert GEMV, **1.52×**).
5. **Closed the quantization-accuracy loop**: KL **0.00136** / top-1 **0.9758**, PPL
   **9.1985 → 9.2047 (+0.067%)**, MMLU 5-shot **53.50% → 52.50%** on a 200-question subset.
6. Ran a **64-cell experiment matrix** with same-round A/B discipline, canary-based VOID rules, and
   first-class negative results.

---

## 2b. 简历定稿（可直接粘贴，**字数很短**）

> 规则：**只写已实测的数字**。上机待跑的两项（batch 曲线、occupancy 上界）在下面留了
> 占位符 `[待测]`——测完再填。**不要**把预估数字写进简历：面试官一追问"这个数怎么来的"
> 就崩，而这个项目最值钱的恰恰是"每个数都能追到工件"。

**中文（4 条，约 170 字）**

> **MoE 大模型单卡推理优化**（OLMoE-1B-7B，RTX 5090，自研 Triton kernel）
> - 自研 **4 个 Triton kernel**（grouped MoE GEMV / 融合反量化 GEMM / split-KV decode attention /
>   融合 RMSNorm），全部通过 `1e-3` 数值 parity 门。
> - **融合 RMSNorm：HF 的 7 个 kernel 合成 1 个**，全步 **1.24×**、kernel 数 **994→539**，
>   同进程交错 A/B **15/15 轮获胜**；torch 原生 `F.rms_norm` 对照反而慢 5%。
> - **无 gather 的 MoE decode kernel**（expert id 只做指针算术，保持 CUDA Graph 可捕获）：
>   消除 **3.22 GB/token** 的 gather 复制，per-token **5.26→4.07 ms（1.29×）**。
> - **W8A16 量化端到端 1.19×**，并**实测出上限只有 1.74×**（权重仅占一步 42%），
>   把 **90%** 收益归因到单个 kernel（专家 GEMV **1.52×**）；
>   精度三链闭合：KL **0.00136**、PPL **+0.067%**、MMLU 200 题 **−1.0 个点**。

**English (4 lines)**

> - Built **4 custom Triton kernels** (grouped-MoE GEMV, fused-dequant GEMM, split-KV decode
>   attention, fused RMSNorm), all gated by `1e-3` numerical parity checks.
> - **Fused 7 HF RMSNorm kernels into 1**: **1.24×** decode-step speedup, kernel count
>   **994→539**, won **15/15** interleaved A/B rounds (torch's native `F.rms_norm` was 5% slower).
> - **Gather-free MoE decode kernel** (expert id as pointer arithmetic, CUDA-Graph safe):
>   removed **3.22 GB/token** of copying, **5.26→4.07 ms/token (1.29×)**.
> - **W8A16 quantization: 1.19×** end-to-end; measured the **ceiling of 1.74×** (weights = 42% of a
>   step) and attributed **90%** of the gain to one kernel; accuracy closed with KL **0.00136**,
>   PPL **+0.067%**, MMLU **−1.0 pt** (200-q subset).

**上机后可以再加的两条（现在留空，别提前写）**

- `batch 曲线：量化收益随 batch 从 1.19× 衰减到 [待测]×（同轮 4 点，金丝雀 ≤5%）`
- `occupancy 上界：融合 RMSNorm / grouped kernel 的 blocks/SM = [待测]（解析上界，非 ncu 实测）`

⚠️ 示例写法（**数值是编的，切勿抄**）：`blocks/SM 上界 = 4（寄存器受限）` —— 若上机测出
不是 4，简历立刻变成假话。**先测，再填。**

**不要写进简历的**：`82.6 ms → 3.59 ms` 这类跨轮合并的"总加速"（那是五条不同轮次的杠杆，
本项目的纪律是只报同轮 A/B）；以及任何 batch>1、FP8、ncu 相关的数字（都没测）。



## 3. 面试必问三题（照这个答）

### Q1「你省了 4 倍内存，为什么延迟只快 19%？」

因为 decode 是**带宽受限**、而权重只是这一步的一部分。实测：权重流占一个 graph 步的
**42%**（ctx=1024）/ **23%**（ctx=4096）⇒ 就算权重路径变成完美、0 字节，步级上限也只有
**1.74× / 1.30×**。我们已经拿到这个上限的**九成**，并且能定位到具体 kernel：
专家 GEMV 隔离测 **96.3 → 63.5 µs（1.52×）**，16 层累计 **0.524 ms**，正好解释步级收益的 **90.4%**。
另一半是 attention 与 elementwise 的固定开销，不随权重比特数变化。

### Q2「你做了融合，融合一定更快吗？」

不一定——**我测过，反例就在库里**。融合反量化（kernel 内解包 vs 先物化 fp16 权重）孤立测
**快 21%**，但在 eager 全模型里**慢 5%**：那条路径每步 **19,029 个 kernel**、host 占 **71.7%**、
GPU 真正干活只有 **57 ms / 199 ms**，省下的字节被每次调用开销吃掉。
三个旁证让它不是"实现坏了"的借口：① 同一实验里 4-bit（字节比更大）**反向赢 3.6%**；
② 同一份 kernel 在图内让投影快 **1.58×**，在 eager 臂里完全看不见；
③ 进图后它成为量化格的**默认路径**（图内 speedup 1.1071 → **1.1703**）。
结论：**融合的收益要等 launch/host 开销被消掉才兑现**，而消掉它的手段是 CUDA Graph。

### Q3「你怎么保证这些数字可信？」

四条机械约束，不是嘴上的：
1. **门先于性能**：正确性门（KL + top-1）不过的格子，`latency` 段由 schema 强制写 `null`，
   实测值只能进 `gate_failed_evidence`；
2. **金丝雀判 VOID**：每轮首尾测同一个格子，漂移 >5% 整轮作废、不发布数字（本项目已因此
   主动作废过三轮，包括本次的 batch 扫描两轮）；
3. **禁止跨轮比较**：不同轮次/不同机器状态的数字不得相除或同表（加速比只报同轮 A/B）；
4. **负结果一等公民**：被否证的假设（融合反量化）、被阻断的路径（ncu 计数被宿主策略拒绝）
   都写进记录与 `RESULTS.md` §4/§6，不藏。

---

## 4. 岗位映射（如实，含缺口）

| 目标岗位 | 能拿出手的 | 缺口（要提前准备） |
|---|---|---|
| **推理优化 / AI Infra** | 端到端 profiling 与归因（host/kernel 拆解、字节账、达成率）；CUDA Graph 可行性与其收益量化；量化部署全链路（打包→反量化→精度门）；同轮 A/B 方法论 | 没有多卡 / 并行策略 / 服务侧调度（本项目是**有意**的 batch=1 单流课题）；FP8 未跑 |
| **算子开发 / Kernel** | 4 个 Triton kernel（含 4-bit nibble 解包、split-KV、行内融合 norm）；访存模式优化（gather → 每 tile 装载+广播，523 → 826 GB/s）；parity 门 + 边界契约（非连续/非整除/超 int32 一律显式拒绝） | 没有 CUDA C++ / 手写 PTX；没有 Tensor Core（MMA）路径（M=1 带宽受限下**刻意**不用，理由已文档化） |
| **AI 编译器** | 融合与图捕获的边界意识（什么能进图、什么导致重编译）；kernel 级 A/B 与 autotune 子集扫描的语义区分 | **最弱一项**：无 TVM/MLIR/`torch.compile` 后端经验。若要投编译器岗，建议另补一个小实验（如 `torch.compile` vs 手工融合的对照） |

---

## 5. 想验证时，一条命令

```bash
# 离线（Mac，无需 GPU）：全部数值断言与形状契约
python -m pytest -m "not gpu" -q

# 上机计划（不测量，只打印命令/时长/显存/验收阈值/回滚）
bash scripts/run_gpu_suite.sh --dry-run
```

代码入口：`src/lemoq/kernels/`（4 个 kernel 族）、`benchmarks/runner.py`（实验驱动）、
`tools/`（精度与工具的 CLI）、`experiments/`（逐条实验记录，含假设先写、结论后写的完整轨迹）。

---

## 6. 边界（不要吹的部分）

- **规模小**：1B 激活 / 7B 总参、单卡、batch=1 单流；不能外推成服务性能（有一整段
  applicability 声明在每份 payload 里）。
- **量化算法简单**：per-group 对称 **RTN**，没有 GPTQ/AWQ/校准集；卖点在**执行与评估闭环**，
  不在算法新颖度。
- **ncu 计数拿不到**：宿主策略阻断（`RmProfilingAdminOnly: 1` + 无 `CAP_SYS_ADMIN`），
  已交出替代口径（解析字节 / 隔离计时）但没有 stall / occupancy 归因（见 `docs/reference/NCU_EVIDENCE.md`）。
- **两条声明了但未跑的**：FP8 / MXFP4（矩阵里有条目，未跑）。
