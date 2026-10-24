# 完善度评估（外部标准对照）— 2026-10-24

> **性质：外部标准对照评估**（rubric 取自公开论文、基准规则与工具文档），用途是给后续 agent
> 一份"离外部标准还差什么"的可执行清单。与 [`ROADMAP.md`](../ROADMAP.md) §1.4 的**内部口径评分**
> （72.5/100）互补、不重复；结论与数字的权威仍是 [`RESULTS.md`](../../RESULTS.md) 与
> [`experiments/`](../../experiments/)。
> **硬规则 1 声明**：本文不引入任何新测量值——仓内数字全部引自既有记录/工件（随文给出处）；
> 外部数字一律是**公开报告值，非本项目实测**，沿用
> [`FUSION_RESEARCH.md`](../reference/FUSION_RESEARCH.md) 的标注惯例，不可与本仓数字直接比较。
> 评估对象：分支 `opt/packed-qkv` @ `4933b37`（2026-10-24）；机器事实以 `results/machine_probe.json`
> （实测只读带宽 1560.3 GB/s）为准。

## 0. 十分钟结论

| 维度 | 本仓现状 | 对照的外部标准 | 判定 |
| --- | --- | --- | --- |
| 正确性门（KL+top-1+PPL） | 双门合取 + 门 FAIL ⇒ schema 自动 null 化全部性能字段；w8a16 双 ctx PASS | GPTQ/AWQ 只报 PPL+零样本；2024–26 工业验收转向 divergence-based（llm-compressor KL-RFC） | **方向领先，广度落后**（§2） |
| 测量纪律（canary/A-B/污染控制） | 首尾同格 canary >5% 判 VOID、graph 格 host_share>5 判污染、独立进程同轮交错 A/B + per-pair wins | google/benchmark 随机交错（#1051）、torch.utils.benchmark 显著性标记；调研范围内**未见公开协议含 canary** | **超出已发表实践**（§4） |
| Roofline 纪律 | 分母用实测带宽（1560.3），necessary-bytes 口径，attainment≤100% 机器自检 | Williams roofline；多数 kernel 论文只报 speedup、不报达成率 | **领先**（§3.2） |
| 融合方法学 | 每项融合有 K1 parity + 同轮 A/B + 规格与验收判据；E2 否证 = "host-bound 时融合收益被 launch 吃掉"的实测版 | Horace He 融合判据；TRT-LLM/vLLM 的图捕获+融合 epilogue 主线 | **持平偏上**；缺外部基线锚定（§3） |
| 推理指标覆盖 | decode-step per-token p50（图内捕获，= TPOT 同类量）+ TTFT p50/p99 已入 schema；graph 臂 p99 为块均值（METRICS §2 已声明不可比） | MLPerf v5 LLM 场景只收 TTFT+TPOT；vLLM 惯例 P50+P99 并报 | **持平**；尾部口径是主要缺口（§4） |

**一句话总评：科学纪律高于业界常规，评估广度低于业界常规；当前最大的方法论风险不是测错，
而是自参照——所有 speedup 的基线都是本仓自己的另一条实现，缺一个第三方锚点（R7）。**

领先与落后各五条速览（出处见 §2–§4）：

- **领先的五条**：① KL+top-1 门早于并合于工业 divergence-based 转向；② canary/污染控制无公开先例；
  ③ 实测带宽 roofline + 达成率口径；④ 负结果一等公民（E2 否证、W4 三连 FAIL 全档）；
  ⑤ 门 FAIL ⇒ 性能字段自动进 evidence 的 schema 强制。
- **落后的五条**：① 零样本任务套件缺位（仅 MMLU n=200，SE≈3.5）；② PPL 评估集薄
  （wikitext2 8k token/5 窗 vs 标准全集）；③ 无显著性检验（per-pair wins 在手却没做符号检验）；
  ④ graph 臂 p99 口径不可比；⑤ 无任何第三方基线（vLLM/cuBLASLt/llama-bench）锚定。

## 1. 评估方法与声明

- **rubric 来源**：四个领域各取 3–6 个可点回的公开来源（论文/官方规则/工具文档），见 §7 索引；
  每条对照项按"外部标准 → 本仓现状 → 判定"三段给出，不做无来源的印象分。
- **外部 URL 复核**：2026-10-24 对 33 条候选链接做可达性抽查——27 条直接 200；
  ACM DL 与 mlcommons.org 对脚本返回 403（反爬，规范出处，保留）；4 条（huggingface 博客 ×2、
  blog.mlc.ai、jan.ai）在评估网络不可达（DNS/CDN 拦截），**已从引用中移除**，
  其论点均由已验证来源覆盖（§7 末注）。
- **评估是判断不是测量**：所有"领先/持平/落后"是对照公开标准后的定性结论，不构成新的性能声明；
  引用本文时请连同所引记录一起引。

## 2. 量化评估协议对照

### 2.1 逐项对照表

| 协议要素 | 外部标准（公开报告值） | 来源 | 本仓现状 | 判定 |
| --- | --- | --- | --- | --- |
| 校准集 | GPTQ：128 条随机 2048-token C4 段；LLMC 系统扫 校准数据×算法×格式 | [GPTQ](https://arxiv.org/abs/2210.17323)；[LLMC](https://arxiv.org/abs/2405.06001) | GPTQ 已实现并上机（EXP-p1-05 实测比 RTN 更差）；校准样本有 2048 上限（扩样在遗留清单 P3-3） | 略窄 |
| PPL | WikiText2 **全集**切段（公开报告 ≈140×2048≈283k token）；PPL 与下游任务 Pearson r≈0.79，是可靠代理 | GPTQ/AWQ；[2402.16775](https://arxiv.org/abs/2402.16775) | wikitext2 test 8k token 缓存；fp16 9.1985 / w8a16 9.2047（+0.067%，PASS）/ w4a16 10.0025（+8.7%，与 KL FAIL 一致）——**数字质量高，样本量**低于标准 | **落后（量不在质）** |
| 零样本任务 | 3–7 个任务全套 lm-eval（ARC-e/c、HellaSwag、PIQA、WinoGrande、MMLU 5-shot）是标配 | [GPTQ](https://arxiv.org/abs/2210.17323)/[AWQ](https://arxiv.org/abs/2306.00978)/[SmoothQuant](https://arxiv.org/abs/2211.10438) | 仅 MMLU n=200：53.50→52.50（−1.0，PASS）但 SE≈3.5，与 2 点门统计不可分辨（引用须带此句，[`ROADMAP.md`](../ROADMAP.md) §1.3）；GSM8K 因基座太弱放弃 | **落后（最大单项）** |
| 发散度指标 | KL(base‖quant) 是最佳单一压缩质量信号（HF 工程研究公开报告其与 flip 数 Spearman≈0.96–0.97；原始链接本网络不可达，未入索引）；聚合 accuracy 会掩盖 flip | [2407.09141](https://arxiv.org/abs/2407.09141)；[llm-compressor KL-RFC](https://github.com/vllm-project/llm-compressor/issues/2031) | KL+top-1 双门一直是主口径（[`METRICS.md`](../reference/METRICS.md) §1）；比"只报 accuracy"的论文常规更接近 2024–26 的工业验收方向 | **领先** |
| 敏感度/混合位宽 | HAWQ 系：按 Hessian trace 给层排敏感度，指导混合位宽 | [HAWQ](https://arxiv.org/abs/1905.03696) | 混合位宽已判"1% 字节收益，不值得"（EXP-p1-03）；W4 失败（[EXP-p1-05](../../experiments/EXP-p1-05-w4-expert-calibration.md)）只有现象级结论、**无逐层归因**——`tools/quant_error_audit` 是权重域 MSE，不到 Hessian 级 | 落后（R8） |
| 负结果报告 | 论文惯例不报失败配置 | — | W4-RTN/GPTQ/≤25% 退路三连 FAIL 全档且预注册了止损条款（[EXP-p1-05](../../experiments/EXP-p1-05-w4-expert-calibration.md)） | 高于论文惯例 |

### 2.2 小结

量化协议的特征是"**深度超前、广度欠账**"：门的机械性与发散度口径领先，评估面（任务套件、
PPL 集量、敏感度归因）落后于 GPTQ/AWQ 时代的标配。特别指出：W4 负结果目前是"关一扇门"，
配上 R8 的 Hessian 归因才能升级为"**知道门为什么关**"——这是把负结果价值翻倍的最便宜路径。

## 3. 融合与 kernel 评估对照

### 3.1 融合判据核对（Horace He ["brrr" 判据](https://horace.io/brrr_intro.html)，逐条）

| 判据 | 本仓对应实测 | 判定 |
| --- | --- | --- |
| memory-bound elementwise 永远值得融合 | fused RMSNorm：7 kernel→1，全步 1.2358×（15/15 wins，kernels/step 994→539，EXP-p4c-02 §5.6）；且先否证了 `F.rms_norm`（保 fp32 乘法反而慢 5%）——比判据多踩一层"数值口径"的坑并留档 | 判据成立且被本仓独立复现 |
| 融合进 GEMM epilogue 近乎免费 | F3 残差 epilogue 已有规格与验收（[`FUSION_OPPORTUNITIES.md`](../FUSION_OPPORTUNITIES.md) §F3，位精确 + ~32 kernel/step） | 已规划未执行 |
| 两个 compute-bound 融合收益小 | M=1 decode 全链带宽受限，本仓 byte-floor 口径把这条直接量化成达成率表 | 判据成立 |
| （本仓补充）融合省下的字节 < launch 开销时，eager 下白做 | E2 否证：host_share 71.7%、19,029 kernels/step 时 fused 反慢（w8a16 1.0442、w8a8 1.0523）；同一 kernel 进图后才兑现——投影修复 8.14→5.15 µs 只在图内可见 | **超出判据的实测贡献** |

### 3.2 kernel 质量指标对照

| 指标 | 外部标准 | 本仓现状 | 判定 |
| --- | --- | --- | --- |
| 带宽达成率 | SOL/roofline 惯例：报 attained BW ÷ 实测峰值 | ✅ 71.0%（fp16 grouped GEMV）/ 94.9%（fused dequant-GEMM）/ 87.3%（F2 packed）；分母为实测 1560.3 GB/s；`_consistency_gate` 机器自检达成率≤100% | **领先** |
| correctness×speedup 合流 | [KernelBench](https://arxiv.org/abs/2502.10517)：fast_p = 正确概率×加速概率的几何平均 | ✅ K1 parity（1e-3）+ 同轮 A/B 之前不谈速度；门 FAIL ⇒ latency null（schema 强制） | 持平（口径更严） |
| 基线选择 | 锚定外部实现（eager/cuBLAS/vLLM kernel） | ⚠️ 全部基线是本仓自身实现（fused vs separate、kernel vs sdpa、grouped vs gather） | **落后（R7，自参照风险）** |
| 微基准卫生 | L2 flush、时钟锁定、暖机与迭代披露（NVIDIA 指南 / [冷热测量](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/) / [SOL-ExecBench](https://arxiv.org/abs/2603.19173)） | 部分：`tools/ncu_evidence.py` 有 256 MB memset flush + event-floor 扣除；轮次协议**无时钟锁定/温度日志** | 部分（R2） |
| 计数器证据 | ncu occupancy/stall 常规 | ❌ 宿主策略阻断（RmProfilingAdminOnly），替代口径（必要字节÷冷计时÷实测带宽）已交付；[`NCU_PLAN.md`](../reference/NCU_PLAN.md) 是换机清单 | 受限（如实入档） |

[KernelBench-Verified](https://arxiv.org/abs/2607.16241) 揭示的"benchmark hacking 抬高加速比"是自参照风险的
极端形态；本仓的温和形态是：**所有对照臂共享同一套实现栈，共同偏置无法被内部 A/B 排除**——
这正是 R7（一次性外部锚定）的动机。

### 3.3 业界同类工作的参照（公开报告值，仅证方向一致，不可横比）

- vLLM 的 `fused_moe` Triton kernel、[TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html)
  的整图捕获+融合 epilogue、[SGLang 后端研究](https://arxiv.org/abs/2501.01005)（公开报告 ITL 降 29–69%）、
  [MegaBlocks](https://github.com/databricks/megablocks)（公开报告 expert GEMM ~2.4×）：业界 decode 优化主线
  就是"图捕获 + 分组 GEMM + epilogue/norm 融合"，与本仓 F1–F6 方向一致；
- 差异在**规模**（本仓有意定位 batch=1 单流的纯净字节预算问题，见 [`ROADMAP.md`](../ROADMAP.md) Tier D）
  与**证据口径**（本仓每个融合过 K1+同轮 A/B；业界工程报告大多只报端到端比）。

### 3.4 F1–F6 状态速览（评估视角）

状态与验收判据的权威在 [`FUSION_OPPORTUNITIES.md`](../FUSION_OPPORTUNITIES.md)，此处只给评估视角一句话：
F1（三项已测未采纳，+8.39%/+3.0%/+1.33%）与 F2（packed-QKV，−4.81%，3/3 pairs）是"数字在手、
只差翻默认"的最大账面缺口（≈0.34 ms est）——**外部标准视角不改变这个优先级，反而加强它**：
业界同路线工作已把图捕获+融合当默认路径。F3/F4（S2 预注册规则当前选中 F4，≈98 µs 可捕获，以
刷新后账本为准）、F5（需新 `GATE_CITATIONS` 行 ⇒ 需用户批准）、F6（被 W4 精度门封死，
除非 R8 归因后另立新实验）。

## 4. 测量工具链对照

### 4.1 指标维度

| 项 | 外部标准 | 本仓现状 | 判定 |
| --- | --- | --- | --- |
| 主指标 | [MLPerf Inference v5](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc) LLM 场景只收 TTFT+TPOT（TPOT 40 ms = 25 tok/s）；[llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) 的 tg 即 decode tok/s | decode-step per-token p50（CUDA-graph 捕获步）+ TTFT p50/p99 均已入 schema | 同类量、纪律更严 |
| 尾部延迟 | [vLLM bench](https://docs.vllm.ai/en/latest/benchmarking/cli/) 对 TTFT/ITL/TPOT 报 P50+P99（可配分位） | p99 字段存在，但 **graph 臂 p99 是 20-step 块均值**，与 eager 逐步样本不可跨臂比（[`METRICS.md`](../reference/METRICS.md) §2 已声明，p50 差 <1% 不翻案） | **落后（R6）** |
| 吞吐/扫描 | llama-bench 的 batch/长度叉积扫描 | batch sweep 已有（fp16 臂）；**量化×batch 衰减未测**（ROADMAP B2）；服务侧明确不做（Tier D） | 部分 |
| 能耗 | [optimum-benchmark](https://github.com/huggingface/optimum-benchmark) 含 energy | 无（功率/温度连记录都没有，并入 R2） | 落后（低优先） |

### 4.2 统计纪律对照

| 项 | 外部 | 本仓 | 判定 |
| --- | --- | --- | --- |
| 交错 A/B | [google/benchmark #1051](https://github.com/google/benchmark/issues/1051)：随机交错使方差降约 40%，已成默认 | ✅ 独立进程、同轮相邻、交替顺序 + 首尾**同臂** canary（比随机交错更强） | **超出** |
| 显著性 | [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)：显式 significant 标记 + 中位数+IQR | ❌ 只有 per-pair wins 计数与 MAD 警示（"最小可分辨差≈MAD"）；**配对符号检验/Wilcoxon 几乎免费** | **落后（R1，性价比最高）** |
| canary/污染控制 | 调研范围内无公开 kernel 协议含此项 | ✅ >5% 判 VOID（fail-closed，指标缺失也判 VOID）+ graph 格 host_share>5 判污染——实测抓到过 18 格共租污染并判废（P2j） | **超出已发表实践** |
| 时钟/环境 | [NVIDIA 测量指南](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/measuring-performance.html) 与 SOL-ExecBench：锁频或至少记录温度/降频 | ❌ 无时钟锁定、无温度/降频日志；canary 能**发现**漂移但不能**归因**漂移 | 落后（R2） |
| 参数披露 | 暖机/迭代/重复数必须公开 | ✅ defaults（warmup 5 / timed 20 / repeats 3）入 schema，payload 全记录 | 达标 |
| 图模式可比性 | [JIT+CUDA Graph 方差研究](https://arxiv.org/abs/2604.23467)：图同时降时延与方差，图/非图数字不可直接比 | ✅ 两臂分开报告、graph 臂带自检；E2 的 eager/图内结论分离正是这条的实操 | 达标 |

## 5. 差距清单 R1–R10（按性价比排序）

| ID | 差距与建议动作 | 外部依据 | 成本/窗口 | 映射 |
| --- | --- | --- | --- | --- |
| R1 | A/B 统计收尾：`_interleaved_ab_stats` 输出配对符号检验/Wilcoxon + 中位差 CI——per-pair wins 数据已在手，纯本地 | torch.utils.benchmark significant；google/benchmark #1051 | Mac，低 | 新增（本地） |
| R2 | 轮次协议加时钟/温度/降频日志（`tools/probe_machine.py` 扩展）；锁频（`nvidia-smi -lgc`）可选、先记录后干预 | NVIDIA 测量指南；SOL-ExecBench | GPU，低（开机顺跑） | probe_machine 扩展 |
| R3 | KL/top-1 门样本量 5→≥10 窗（内部早已提出；w8a8@4096 的边界翻转正卡在这里） | 发散度评估的样本量惯例（HF 研究公开报告 ~10K prompts；链接本网络不可达，论点由 KL-RFC 覆盖） | GPU，低 | EXP-p1-04 follow-up #2 |
| R4 | PPL 扩到 WikiText2 全集切段（标准 ≈140×2048；现 8k token/5 窗）——`tools/quant_ppl` 加全集模式 | GPTQ/AWQ 全集口径；r≈0.79 代理有效性 | GPU，低 | quant_ppl 扩展 |
| R5 | lm-eval 零样本套件：ARC-e/c、HellaSwag、PIQA、WinoGrande（经 lm-evaluation-harness）+ MMLU 扩量 | GPTQ/AWQ/SmoothQuant 标配 3–7 任务 | GPU，中 | A4 扩展 |
| R6 | graph 臂逐步 p99：单步 replay 计时或块内逐事件切分，让 p99 跨臂可比（改动涉 METRICS 口径） | vLLM P50+P99 惯例；MLPerf | GPU，中 | METRICS §2 残留 |
| R7 | 一次性外部锚定：同机跑 vLLM（bench latency 或其 fused_moe kernel 对照）；**先查 OLMoE 支持与安装成本**；产出进记录、不进 `GATE_CITATIONS` | KernelBench-Verified 自参照警示 | GPU，中 | 新增 |
| R8 | 逐层敏感度归因：权重域 MSE（已有）→ 校准集 Hessian trace（HAWQ 式），解释 W4 为何败、指导 >25% 混合位宽退路 | HAWQ | CPU 部分 + GPU | C2/P3-3 |
| R9 | CI 绿徽章（A5）+ parity 合取补齐（A6） | — | Mac 低 + GPU 半边 | 已列（ROADMAP Tier A） |
| R10 | torch.compile vs 手工融合最小对照（norm/反量化场景） | TRT-LLM piecewise CUDA graphs | GPU，中 | C1 |

验收口径建议（实现各 R 项时逐条对照）：R1 用既有 A/B 工件重算并给出 p 值/CI；R2 以
`machine_probe.json` 新增字段入 schema；R3/R4 新工件 + 记录小节（不动 `GATE_CITATIONS` 阈值，
样本量变化须在记录中声明）；R5 新工件新记录（**是否并入门禁需用户决定**）；R7 只产对照记录、
明示"公开报告值/同机实测值"之别；R8 先 CPU 后 GPU 两段交付。

## 6. 给后续 agent 的执行建议

**第一优先不变：S1（采纳 F1+F2）。** 本评估不改变 [`ROADMAP.md`](../ROADMAP.md) §1.4 的"兑现"判断——
图捕获+融合在业界已是默认路径，本仓"数字在手、默认未翻"是最反常的缺口。R 项是
"把证据口径抬到外部标准"的补强，不与 S1/S2/S3 竞争 GPU 窗口。

- **Mac 离线（不等 GPU）**：R1（显著性检验，含对既有 `results/*_ab.json` 的重算演示）、
  R9 的 A5 半边（CI 先在干净 venv 跑绿再上徽章）、本报告的滚动维护（R 项关闭时更新 §5）。
- **开机顺跑（挂在既有轮次，不新开实验）**：R2（probe_machine 记时钟/温度）、
  R3（门样本 5→≥10 窗，EXP-p1-04 协议内）、R4（quant_ppl 全集模式）。
- **需用户批准或独立窗口**：R5（lm-eval 套件是新测量；并入 `GATE_CITATIONS` 须用户决定）、
  R6（改 METRICS 口径）、R7（装 vLLM ⇒ 触发"装机需用户同意"红线，先做可行性侦察）、
  R8（CPU 部分可先做，GPTQ 校准段需 GPU）、R10（C1）。
- **排序建议**：R1/R9 → R2/R3/R4 → R5 → R6/R7 → R8/R10。R7 若做，排在 S 系列之后
  （它是叙事补强，不是收益）。

## 7. 外部参考索引（2026-10-24 经 curl 可达性抽查）

| 领域 | 来源 | 本文用到的点 | 链接 |
| --- | --- | --- | --- |
| 量化 | GPTQ（ICLR 2023） | 校准 128×2048、全集 PPL 协议 | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| 量化 | AWQ（MLSys 2024） | PPL + 5 任务零样本标配 | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| 量化 | SmoothQuant（ICML 2023） | 多任务平均、W8A8 报告口径 | [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) |
| 量化 | 量化策略综合评估（ACL Findings 2024） | PPL↔下游 r≈0.79 | [arXiv:2402.16775](https://arxiv.org/abs/2402.16775) |
| 量化 | Accuracy is Not All You Need（NeurIPS 2024） | 聚合 accuracy 掩盖 flip | [arXiv:2407.09141](https://arxiv.org/abs/2407.09141) |
| 量化 | vLLM llm-compressor KL-RFC #2031 | KLD 作标准验收 | [GitHub issue](https://github.com/vllm-project/llm-compressor/issues/2031) |
| 量化 | HAWQ（ICCV 2019） | Hessian trace 敏感度排序 | [arXiv:1905.03696](https://arxiv.org/abs/1905.03696) |
| 量化 | LLMC（EMNLP 2024） | 校准×算法×格式系统基准 | [arXiv:2405.06001](https://arxiv.org/abs/2405.06001) |
| 量化 | 量化方法综述 | 背景框架 | [arXiv:2103.13630](https://arxiv.org/abs/2103.13630) |
| 融合 | Horace He "brrr" | 融合判据（§3.1 逐条核对） | [horace.io](https://horace.io/brrr_intro.html) |
| 融合 | Roofline（CACM 2009） | 字节地板/达成率口径 | [ACM DL](https://dl.acm.org/doi/10.1145/1498765.1498785)（对脚本 403，DOI 有效） |
| 融合 | KernelBench | correctness×speedup 合流指标 | [arXiv:2502.10517](https://arxiv.org/abs/2502.10517) |
| 融合 | KernelBench-Verified | 基准自抬/自参照风险 | [arXiv:2607.16241](https://arxiv.org/abs/2607.16241) |
| 融合 | MegaBlocks | 分组 GEMM MoE 先例 | [GitHub](https://github.com/databricks/megablocks) |
| 融合 | SGLang 后端研究 | ITL 降低的报告形态 | [arXiv:2501.01005](https://arxiv.org/abs/2501.01005) |
| 基准 | MLPerf Inference 规则 | LLM 场景 TTFT+TPOT | [rules.adoc](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)（[公告](https://mlcommons.org/2025/04/llm-inference-v5/) 对脚本 403） |
| 基准 | vLLM bench | P50+P99 报告惯例 | [docs](https://docs.vllm.ai/en/latest/benchmarking/cli/) |
| 基准 | llama-bench | pp/tg/暖机+重复口径 | [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md) |
| 基准 | optimum-benchmark | energy 维度 | [GitHub](https://github.com/huggingface/optimum-benchmark) |
| 统计 | torch.utils.benchmark | 中位数+IQR、显著性标记 | [docs](https://docs.pytorch.org/docs/stable/benchmark_utils.html) |
| 统计 | google/benchmark #1051 | 随机交错降方差 | [issue](https://github.com/google/benchmark/issues/1051) |
| 统计 | NVIDIA 测量指南 | 重复/分位/隔离噪声 | [docs](https://docs.nvidia.com/multi-node-nvlink-systems/multi-node-tuning-guide/measuring-performance.html) |
| 统计 | CUDA 冷热测量（Lei Mao） | L2 残留与冷测口径 | [blog](https://leimao.github.io/blog/CUDA-Performance-Hot-Cold-Measurement/) |
| 统计 | SOL-ExecBench | 时钟锁定一等公民 | [arXiv:2603.19173](https://arxiv.org/abs/2603.19173) |
| 统计 | JIT+CUDA Graph 方差研究 | 图模式降时延也降方差 | [arXiv:2604.23467](https://arxiv.org/abs/2604.23467) |
| 参考 | TensorRT-LLM 核心概念 | 整图捕获+融合 epilogue 主线 | [docs](https://nvidia.github.io/TensorRT-LLM/architecture/core-concepts.html) |
| 参考 | CUDA C Best Practices | 事件计时/暖机 | [docs](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) |

> 注：huggingface.co 博客 ×2、blog.mlc.ai、jan.ai 四条在评估网络不可达（DNS/CDN 拦截），
> 已从引用中删除；其论点（KL↔flip 相关、敏感度混合精度、kernel 基准卫生）分别由
> KL-RFC + 2407.09141、HAWQ、NVIDIA 指南 + SOL-ExecBench + Lei Mao 覆盖。
