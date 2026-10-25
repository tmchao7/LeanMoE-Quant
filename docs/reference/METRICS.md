# METRICS — 指标口径

> 记录格式见 [EXPERIMENTS.md](../../EXPERIMENTS.md)。本文定义**每个指标怎么算、为什么用它、
> 什么情况下不能用**。

---

## 1. 正确性指标（先过门槛，才允许报性能）

| 指标 | 定义 | 门槛（Phase 1 定，先给建议值） | **来源** | 注意事项 |
|---|---|---|---|---|
| **KL 散度** | 固定 prompt 集上，本配置与 fp16 参考的 logits 逐位置 KL 平均 | W8 ≈ <0.01；W4 ≈ <0.05 | **自定建议值**（本文件 Phase 1 拍定，未引自文献） | 主指标；不受采样策略影响 |
| **top-1 一致率** | 与 fp16 参考的 greedy 轨迹 teacher forcing 下**逐位置 argmax 一致比例**（多窗口平均；实现：33 位置 × 5 窗口 = 165 位置取均值） | ≥0.95 | **自定建议值**（同上） | 直观；但比 KL 粗糙。注意这不是"只看首 token"——是轨迹上每个位置都比 |
| **PPL** | wikitext-2 困惑度，报绝对值与相对退化 | 相对退化 <1% | **自定建议值**（同上） | 与传统对照可比；对 4bit 区分度弱 |
| **任务级分数** | GSM8K / MMLU 小子集 | 下降 <2 个点 | **自定建议值**（同上） | 最贴"能不能用"，但贵、方差大；★ MMLU **已测**（2026-10-09，n=200，见 §6 G1 与 RESULTS §3.1）；GSM8K 生成式在基座上贴地板、已放弃 |

★ **门槛来源的说明（必须随结论一起读）**：上面四个门槛都是**本项目自定的建议值**，
不是从文献抄的。外部常用的一个参考是"KLD < ~0.007 可接受"，比这里的 W8 门槛
（0.01）**更严**。已实测的 w8a16 是 0.001361，两条口径下都过；但这不改变
"门槛是自定的"这件事 —— 引用本项目结论时必须带上这条说明。

**规则**：KL 或 top-1 不达标 → **只记录正确性失败，不报性能数字**。

**为什么 KL 是主指标**：PPL 对 4bit 的区分度弱，且受解码策略影响；KL 直接量化"分布偏了
多少"，是量化文献的标准语言。

---

## 2. 性能指标（batch=1：延迟与显存优先）

| 指标 | 定义 | 口径要求 |
|---|---|---|
| **TTFT p50 / p99** | 首 token 延迟 | 必须报 p99，不只报均值 |
| **per-token p50 / p99 @ batch=1** | 逐 token 解码延迟 | **主指标** |
| **throughput @ batch ∈ {1,2,4,8}** | tok/s | **fp16 曲线已测（G2，2026-10-10）**：金丝雀 −3.448% ≤5%，同轮 4 点单列表（`results/batch_sweep.json`），**禁止与 batch=1 历史数字跨轮混表**（硬规则 3）；**量化 × batch 的衰减对比未测**（矩阵 E3 未跑） |
| **host/launch 占比** ★ | 主机+launch 开销 ÷ 单步总耗时 | **一等指标。** 它决定量化收益的**上限**（见 §2.1） |
| **dequant 占比** | 反量化耗时 ÷ 单步总耗时 | 融合 vs 不融合的直接证据 |
| **路由占比**（MoE） | 路由/打包耗时 ÷ 单步总耗时 | 量化在 MoE 上的收益上限来源 |

**计时口径**
- warmup ≥ 5，计次 ≥ 20，报 **p50 / p99** 与离散度（stddev 或 bootstrap CI）；
- **样本基底要声明**：cudagraph 格的 p50/p99 算在**每 repeat 的 20 步块均值**上
  （块数少），eager 格算在**逐步 wall 样本**上（n≥20）——p50 是同量级的，
  但两臂的 **p99 与 stddev 不可直接比较**（块统计 ≠ 逐步尾延迟）；
- **不得用 min-of-3** 这类旧做法作为正式数字；
- 明确声明是否包含 kernel launch 开销；
- JIT/autotune 的首次编译时间不能计入稳态数字，但**要单独记录**；
- **必须同时报绝对值与比值**——只给加速比无法解释（Hoefler 规则 #1）；
- **必须报 host/kernel 拆分**（`benchmarks/hostgap.py`），并用 nsys 交叉验证。

### 2.1 为什么 host/launch 占比是一等指标

batch=1 的单步时间可以拆成互不重叠的四项：

```text
T_step = T_bytes + T_compute + T_routing + T_host
           ↑          ↑           ↑          ↑
        权重流字节   张量核     路由/打包   kernel launch + 框架
        ÷ 实测带宽  (b=1 极小)  (MoE 专有)
```

**`T_host` 不随量化下降。** 所以量化对 token 延迟的收益被截断：

```text
speedup_E2E = (T_bytes_fp16 + T_host) / (T_bytes_quant + T_host)
```

**示意**（借 `MegaKernel-vLLM` 在 A100 + Qwen3-0.6B 上实测的 host:kernel ≈ 3:1；**这是推导，
不是实测**）：`T_bytes_fp16 = 1.00`、`T_host = 3.00`、`T_bytes_w4 = 0.25`
→ `speedup = 4.00 / 3.25 = 1.23×`。**字节降到 1/4，token 只快 1.23 倍。**

这个机制同时解释了第三方观测到的
["real-FP4 在 GEMM 赢 4×、在 token 输 3–4×"](https://hf-mirror.com/datasets/witcheer/sm120-field-guide/raw/main/README.md)。

**推论**：
1. 只报"量化后 GGEM 快 N 倍"而没有 host 占比 → **无法解释端到端收益**；
2. 本项目必须包含 **E7：`{fp16,W8A8,W4A16} × {eager,cudagraph}`** 这个 2 维矩阵，
   用来隔离截断效应（假设与反证信号见 `PLAN.md` §7）；
3. 卡越快，`T_bytes` 越小、`T_host` 不变 → **host 占比越大、截断越严重**。
   所以 V100(900 GB/s) 与 5090(1792 GB/s) 的对照本身就是这个效应的实验。

### 2.2 配对 A/B 的显著性（2026-10-24 起，`benchmarks/ab_stats.py`）

同轮交错 A/B 除了 wins/MAD，还必须落盘显著性块（动机与外部依据见
MATURITY_REVIEW_2026-10-24 R1——文档已退役，见 git 历史）：

- **符号检验**：精确二项双侧 p（丢并列，`n_eff = wins+losses`）；
- **Wilcoxon 符号秩**：`n_eff ≤ 20` 用精确零分布（2^n 符号翻转的子集和 DP），
  否则正态近似（含并列秩与连续性校正）；
- **中位差 95% CI**：成对 bootstrap 百分位区间，**固定种子 ⇒ 同数据同区间**；
- 落盘点：runner 的 `interleaved_ab.paired.{delta_ms,significance}`、
  profile_graph v2 的 `paired_wall.significance`；旧工件用
  `python -m tools.ab_significance <payload...> [--out ...]` 只读重算（不新测）。

**两条不妥协**：
1. **p 值不替代 MAD 口径**——`|median_delta| < MAD` 时仍按"不可分辨"处理（§2 既有规则）；
2. **小样本不装显著**——`n_eff < 5` 时显著性块带 note 如实声明（任何 p 都几乎不可能显著），
   2026-10-24 的既有 F1 三项 A/B 重算即属此类：n=2 对、方向一致但 sign p=0.5，
   采纳依据写的是**效应量与方向一致性**，不是 p 值（`results/ab_significance.json`）。


---

## 3. 带宽指标（本项目的"物理口径"）

这是把本项目与 [MegaKernel-vLLM](../../../MegaKernel-vLLM/README.md) 连起来的**物理口径**。

### 3.1 单位约定（先读这条，否则数字必错）

| 量 | 单位 | 说明 |
|---|---|---|
| 带宽 | **GB/s = 10^9 B/s** | 与 datasheet / `nvidia-smi` 一致 |
| 字节预算 | 同时给 **GB (10^9)** 与 **GiB** | 两者差 7.4%；混用会让达成率凭空漂移 |
| 显存 | **GiB** | allocator 报字节，除以 1024³ |

全文沿用此约定；任何表格若用别的单位，必须在表头声明。

### 3.2 两个分母 —— 必须分开报

`bytes-per-token` 有**两个不同定义**，它们随 context 长度分叉，**不得混用**
（这是本项目最容易出错、也最容易被面试官抓住的地方）。

```text
[W] 权重字节（与 context 无关 —— 量化的直接杠杆）
    = attention 权重 + router 权重 + top-k 专家权重及其 scale
      + shared expert（OLMoE 无）+ lm_head（若 untied）
    ※ 不含 embedding 查表：每 token 只取 1 行（2048 × 2 B = 4 KiB），可忽略

[T] 总字节（决定实测单步延迟 —— 与 context 有关）
    = [W] + KV 读
    KV 读 = 2 × 层数 × num_kv_heads × head_dim × ctx × dtype 字节
            （decode 每步要读完整 context 的 K 和 V）
```

两个口径各回答一个不同的问题，**都要报**：

```text
量化杠杆下限  ： floor_W(ms/step) = [W] ÷ 实测可达带宽          # 不含 KV
端到端下限    ： floor_T(ms/step) = [T]@ctx ÷ 实测可达带宽       # 含 KV
带宽达成率    ： floor_T(ctx) ÷ 实测 per-token 延迟             # 必须标注 ctx
量化达成率    ： floor_W ÷ 实测 per-token 延迟                  # 回答"量化兑现了多少"
```

**必报三件套**：任何写进结论的达成率，必须同时给出 **① ctx ② 用的是 [W] 还是 [T]
③ [W] 的逐项拆分**。只写"达成率 78%"而没有这三样 = 未测（见下方硬规则 5）。

**为什么必须分开**：随 ctx 增长，[T] 会显著大于 [W]（OLMoE 上 @4K 时 KV 占 [T] 的 18.5%，
见 [MODELS.md](MODELS.md) §3.2）。若有人用 [W] 当分子、用实测延迟含 KV 当分母，
达成率会**虚高**，这正是"达成率 >100%"最常见的来源之一。

### 3.3 本项目的锚点数字（OLMoE-1B-7B，fp16，batch=1）

| 量 | 值 | 出处 |
|---|---|---|
| `[W]` 权重字节/token | **2.358 GB (2.196 GiB)** | [MODELS.md](MODELS.md) §3.2，逐项推导 |
| `[W]` 中专家占比 | **68.3%** | 同上 |
| KV 读 @1K / @4K | 0.134 GB / 0.537 GB | 同上 |
| KV 池显存 @4K | 0.5 GiB | 同上 |
| 全量权重 fp16 | 13.84 GB (12.89 GiB) | 同上 |

**这些是推导值，不是实测值**；它们用于阶段 2 之后的理论下限，**上机时必须复核**。

**硬规则**

1. **必须用实测可达带宽**，不能用规格值（V100 规格 ~900 GB/s 是上限，实测会低）；
   可达带宽由 `tools/probe_machine.py` 产出，并随结果一起引用。
2. 计算 bytes-per-token 时必须**逐项列出来源**（哪些是权重、哪些是 KV、是否含 lm_head），
   否则别人无法复核。
3. 达成率 **>100%** 说明字节账算错了或缓存命中了（权重可能被 L2 缓存），**必须解释**，
   不能当成绩报。
4. 报告里必须区分两个不同的东西：
   - **规格带宽**（datasheet）；
   - **实测可达带宽**（本机、本访问模式）。
   两者混用是本领域最常见的错误来源。
5. 达成率必须带 **ctx + 分母口径（[W] 还是 [T]）+ [W] 逐项拆分**；
   `[W]` 与 `[T]` 不得混用（见 §3.2）。

---

## 4. 显存指标（必须分列）

| 项 | 测量方式 |
|---|---|
| weights | 加载后、量化后的参数占用 |
| KV | 按 context 长度与 batch 计算 + allocator 实测 |
| activations | 前向峰值增量 |
| workspace | 框架缓冲 + kernel 工作区 |
| **peak total** | allocator 峰值（`torch.cuda.max_memory_allocated`） |

**规则**：不要用 `nvidia-smi` 的总量当唯一数字——它包含缓存与其它进程，无法定位是哪一项爆的。
**必须分列**，否则"为什么 OOM"无法回答。

---

## 5. 报告纪律

1. 每个数字都要带 **硬件 + 精度 + batch + context** 四个限定；
2. 跨卡/跨阶段/跨模型格式的数字**不放进同一张表**；
3. 用三态标注：`已实测` / `待标定`（测了但物理上说不通，需复采）/ `待跑`；
4. **负结果照写**——本项目一半的价值在"量化在 MoE 上能走多远"的边界。
5. **声明了却没测的指标必须显式标注"未测"**，不得让读者把"已声明口径"读成
   "已覆盖"。本文件里所有标了"未测"的行都指向下面这一节。

---

## 6. 待 GPU 执行清单（本文件里所有"未测"的落脚点）

| # | 项 | 关联 | 脚本 / 计划 | 状态 |
|---|---|---|---|---|
| G1 | **任务级精度**（MMLU 5-shot 200 题 + GSM8K 100 题，只针对已过门的 w8a16） | §1「任务级分数」 | `tools/task_accuracy.py` | **部分完成（2026-10-09）**：MMLU 已测 —— fp16 53.50% / w8a16 52.50%（delta **−1.0 个点**，门槛 <2 点 ⇒ PASS）；口径 B teacher forcing 一致率 99.0%。**GSM8K 放弃**：生成式在基座模型上贴地板（fp16 8-shot **0/15**），如实记为 floor 而非"未测"。工件 `results/task_accuracy.json` |
| G2 | **batch 扫描**（batch ∈ {1,2,4,8}，吞吐-延迟曲线） | §2「throughput @ batch」 | `tools/batch_sweep.py` | ✅ **已测（2026-10-10 单轮有效）**：金丝雀 **−3.448% ≤5%**；fp16 单臂、ctx=1024、n=5 丢 1 轮 ⇒ step p50 87.20/85.88/86.82/86.29 ms、tok/s 11.47/23.29/46.07/92.71（工件 `results/batch_sweep.json`）。★ 单列表，禁止与 batch=1 历史数字跨轮混表；**量化衰减对比未测**（本轮仅 fp16） |
| G3 | **occupancy / register 下限估计**（fused RMSNorm + grouped kernel） | §5「融合后是否引入新瓶颈」 | `tools/occupancy_probe.py` | ✅ **已跑（2026-10-10，`status=done`）**：解析上界（非 ncu 实测）blocks/SM ≤12/≤12/≤6，n_regs 38/40/40，occupancy 上界 100%（工件 `results/occupancy.json`）；stall 实测仍待换机（NCU_PLAN） |
| G4 | **ncu 补测**（需 `CAP_SYS_ADMIN` 的机器：stall / occupancy / `gpu__time_duration.sum`） | §3、§5 | `docs/reference/NCU_PLAN.md`（本地文档） | **换机才能做**（当前 blocked，见 [NCU_EVIDENCE.md](NCU_EVIDENCE.md) §1） |
| G5 | **PPL 写入工件**（`ppl_fp16/ppl_this/ppl_delta` 进 `result.json`） | §1「PPL」 | `tools/quant_ppl.py` + `tools/ppl_status.py --merge` | **完成（2026-10-09）**：3 份 cell 已写（fp16 ppl=9.1985；w8a16 g128 = 9.2047，**+0.067%** < 1% 门槛；w4a16 g128 = 10.0025，+8.7%，与它的 KL FAIL 自洽）。`PPL_STATUS` 44 格 = 3 已测 + 41 已声明、0 缺失 |

一键执行：`scripts/run_gpu_suite.sh`（`--dry-run` 先打印完整计划，含每项的命令、
预计时长、显存需求、验收阈值与失败回滚）。

**上机前的硬约束**（写在这里是为了让每个上机的人先看到）：
- 每轮首尾跑**同格金丝雀**，漂移 >5% ⇒ 整轮 `VOID`，保留日志但不发数字；
- 同租户进程**记录，绝不 kill**（`ncu_evidence` 已实现 `foreign_processes`）；
- 未测的值一律写 `planned` / `null`，**不得**写成结果（硬规则 1）；
- 不得跨轮比较（硬规则 3）—— batch 扫描与已有 batch=1 数字分表呈现。
- **长轮次的 A/B 必须用独立进程**：同一个进程里连续跑多次完整 session 会让 `host_share`
  随轮次累积（2026-10-23 实测同一格：进程内第一次 3.2–3.6%，第 5–7 次 5.5–7.1% ⇒ 三轮
  全被判 VOID，与同租户无关）；同轮相邻、每臂独立进程的协议见
  `scripts/rounds/run_ab_fresh_process.sh`（2026-10-23 用它拿到 F2 的合规轮：
  off 1.9536 → on 1.8597 ms，−4.81%，3/3 对，host 3.5–4.2%）。
