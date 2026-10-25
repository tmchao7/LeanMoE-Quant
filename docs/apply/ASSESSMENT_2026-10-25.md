# 项目外部对标评估（2026-10-25）——投递 AI infra 实习的自检

> 目的：回答"（1）量化做得如何？（2）算子融合+瓶颈定位做得如何？"——**每条判断都挂一个
> 外部参照**（论文/官方工程博客/社区基准），不打 内部 分。评估对象是 v7 canonical
> （w8a16+lm_head w8+KV int8 × 自研 split-KV × CUDA Graph，1.4602/1.6075 ms @1024/@4096，
> 最好格 2.44×/4.25×）。内部数字的出处全部在 [RESULTS.md](../../RESULTS.md) 与各记录；
> 本文只负责"放到行业坐标系里看"。
>
> 岗位目标：推理优化（云/端）/ 算子优化开发 / AI 编译器 实习。

## 0. 一页结论

| 维度 | 外部参照 | 我们的位置 | 评级 |
| --- | --- | --- | --- |
| KV 量化精度 | vLLM FP8（≤0.7 点，近无损）；llama.cpp q8_0 社区共识"几乎无损"；KIVI 2-bit ~2% 掉点 | int8 门 KL 0.0006/0.0013、top-1 0.982/0.988——**同一结论带** | **A-**（口径窄：轨迹门，无 PPL/MMLU 链） |
| KV 量化性能 | KIVI 2.35–4.3×（**大 batch/长 ctx 容量场景**） | batch=1 带宽口径 **−6.5%**@4096，兑现地板 65%——**口径不同，份额封顶是自测的** | **B+**（正确问题、正确答案；规模小于论文场景） |
| 权重量化 | Marlin W4A16 kernel 3–4×（batch≤32，权重≈全部流量）；LMDeploy 2.4× | w8 步级 1.22×/1.10× + **实测上限 1.69×/1.31×**（权重仅占步 41%/23.5%）——解释了"为什么只有这么多" | **A-**（上限分析是稀缺能力） |
| 4-bit/低比特 | MoE 量化文献（MxMoE ICML'25、GEMQ、ACL'25 Findings）：**均匀 4-bit 专家量化失败是共识**，解法是专家级混合精度 | W4 RTN/GPTQ、MXFP4、≤25% 混合退路**全 FAIL 且如实入库**；止步于文献确认的边界 | **B**（负结果可信、有文献背书；没有自救成功案例） |
| Attention kernel | Flash-Decoding（Meta/Stanford 2023）：split-KV + online softmax + combine，长 ctx 至 ~8× | 我们的 split-KV = **同一三段式**（含 int8 scale 折叠变体）；步级 1.51×/1.46×（attention 只占步一部分） | **B+**（复刻级+一个变体，非首创） |
| 融合清单 | TensorRT-LLM 生产融合集：residual+RMSNorm 单核、QKV+RoPE 进 GEMM epilogue | F1（rope/route/mask）、F2（packed qkv）、F4（KV 直写）、F5、E6 **与生产清单几乎一一对应且各自 A/B 采纳**；F3/F7（残差/o_proj epilogue）是**生产已做、我们未做**的两条 | **B+**（清单命中率高；最深两条未做） |
| 带宽达成 | llama.cpp 密集 8B Q8 有效 ~75–85% 理论带宽（社区口径） | 全步有效 **56–59%**（E6 修正口径、实测 1560.3 分母）；kernel 级最高 **87.3%**（packed GEMM）/94.3%（attn_split） | **B**（kernel 级达标，步级差值=已分类的尾巴） |
| 实验/瓶颈定位方法学 | 工业 perf 团队纪律（同轮 A/B、回归门、fail-closed） | 七条杠杆**每条独立同轮 A/B**（5/5 对×4、3/3×1）+ 金丝雀 + VOID 不发数 + de-publish 先例 + 账本化 gap 分类与"判死" | **A**（这是全项目最强的部分） |

**一句话**：这是一个**方法学 A、单项技术 B/B+** 的项目——它最值钱的不是某个倍数，而是
"每一步都能回答这个数怎么来的、为什么不能更多"，而这恰是 infra 岗位面试的核心考察面。

---

## 1. 量化做得如何？

### 1.1 KV 量化（E6）：结论带与工业一致，口径窄于工业

**外部**：vLLM 官方 FP8 KV 博客报"最多 0.7 点掉点、AIME25 恢复率 ≥99%"；社区（r/LocalLLaMA、
多个评测）对 q8_0 KV 的共识是"几乎无损"（例外：128k needle-in-haystack 类长检索任务 FP8
有崩塌案例——particula.tech 实测 91%→13%）；KIVI（ICML 2024）在 2-bit 下也只掉 ~2%（靠
per-channel K + per-token V + 近窗残差）。

**我们**：per-token 对称 int8（K/V 同粒度），门 KL 0.0006/0.0013、top-1 0.982/0.988
（forced 轨迹 165 位置）。与"8-bit KV 近无损"的工业结论**同带**。设计上我们比 KIVI 简单
（它为 2-bit 必须非对称+分通道；int8 对称已够过门——这与 vLLM FP8 的工程选择同理：
**先做能过的位宽，把复杂度留给需要它的位宽**）。

**差距（面试要主动说）**：
1. 精度链只有轨迹 KL/top-1（预注册口径），**没有 PPL/MMLU/长检索任务链**——vLLM 的教训
   恰恰是常规基准近无损、128k 检索会崩；我们 ctx 上限 4096 没踩到，但口径上没测就是没测；
2. 性能收益 −6.5%@4096 与 KIVI 的 2.35–4.3× **不可比**：它们省的是 KV **容量**（大 batch/
   长 ctx 下 KV 溢出显存），我们 batch=1 省的是**读带宽**（KV 只占步流量的一部分）——
   项目自己把这个份额测出来了（这是加分项，不是弱点）；
3. 无 FP8（5090 原生支持）、无 2-bit/非对称——按预注册阶梯 v2 没启用（v1 已过门）。

### 1.2 权重量化：拿到的收益被结构性份额封顶——且是自己证明的

**外部**：Marlin（FP16×INT4）kernel 级 3–4×、LMDeploy W4A16 端到端至 2.4×——**前提是
decode 里权重≈全部流量、且 4-bit 门能过**。Red Hat/社区的另一半经验：W8 权重只减字节，
收益天然小于 2×；量化"不一定更快"（反量化开销/访存模式）是反复出现的社区主题。

**我们**：w8 权重门近无损（KL 0.0014/PPL +0.0088%/MMLU −0.5 点三链），步级 1.22×/1.10×；
**上限 1.69×/1.31× 不是拍的是测的**（权重流占步 40.9%/23.5%，从刷新 payload 反推）。
对标结论：我们的数字落在"份额封顶"的理论带内，且比多数项目多走了一步——**量化了天花板
本身**。Marlin 类 3–4× 的前提（4-bit 过门 + 权重主导流量）在本模型上第一条就不成立。

### 1.3 低比特：负结果与 2025-26 文献共识一致

**外部**：MxMoE（ICML 2025）/GEMQ/MoEQuant/ACL 2025 Findings 一致指出 MoE 专家权重
敏感性异质、**均匀 4-bit 会崩、错误经路由复合**，解法是专家级/层级混合精度。

**我们**：W4-RTN、W4-GPTQ、MXFP4、≤25% 整层 W8 退路全部 gate FAIL（KL 0.0447/0.0548/
0.0739/0.0265 vs 门 0.01），按预注册反证条款**停止扩张并入库**。当时没有引用这些文献，
事后看结论与文献严格一致——面试可讲："独立的负结果 + 事后文献验证 + 预注册止损纪律"。
**缺口**：没有尝试文献指出的正解（专家级敏感度分层混合精度）——这是最自然的"下一步"
故事（也解释了为什么止步：一轮一变量 + 立项纪律）。

### 1.4 量化线总评

- **做对的**：门先行、三链闭合（权重）、独立门表（lm_head/KV 各一张）、de-publish 撤回
  不稳健的数字、负结果入库、上限实测。这套口径比多数实习项目（甚至部分论文）严。
- **不足的**：KV int8 无任务级链；无 FP8；4-bit 无成功案例（有可信失败）；单模型单卡。
- **对岗位的映射**：云推理岗会追问 FP8/PagedAttention/continuous batching——前两个答
  "硬件在但项目按预注册止步/规则 10 刻意不接"；端侧岗（llama.cpp 同赛道）则几乎全对口。

## 2. 算子融合 + 瓶颈定位做得如何？

### 2.1 融合清单 vs 生产栈：命中率高，深度差两条

**外部**：TensorRT-LLM 的 Kernel & Fusion Casebook 列的生产 decode 融合集——residual+RMSNorm
单核、QKV 投影+RoPE 进 GEMM epilogue、去独立 RoPE/elementwise 核；PyTorch 官方博客也在做
"Normalization 融进 GEMM/Attention"。

**我们**：七开关里 F1（fused rope/routing/mask）、F2（packed qkv，N=6144 单 launch）、
F4（KV 直写进 RoPE 后路径）、F5、E6（scale 折叠进 split-KV，读路径零解包）**与生产清单
几乎一一对应**，且每条都有独立同轮 A/B（F2 −4.81% 3/3、F4 −2.94% 5/5、F5 −4.13% 5/5、
E6 −6.54% 5/5）后翻默认。**未做的 F3（残差进 GEMM epilogue）与 F7（o_proj 合并）正是
生产已做、我们只预注册的两条**——差距明确且自知（F3/F7 都已写验收判据）。

加分项：**E2 负结果**（融合反量化在 eager 反而慢 5%——host/launch 受限）与社区
"融合不一定更快"的经验（及我们引用的 XLA 融合分析）一致；"孤立 kernel 1.21× 更快但
端到端更慢"的归因是很多工程师会栽的坑。

### 2.2 Attention kernel：标准设计的正确实现 + 一个变体

**外部**：Flash-Decoding 三段式（split KV / online-softmax 部分和 / LSE 合并）是长 ctx
decode 的标准答案（原博客报长 ctx 至 ~8×；我们引用的 8 点网格敏感性也做了）。

**我们**：同三段式（确定性合并、无 atomic——保"同路径 KL=0"门可复现），外加
**int8 读变体**（K 的 scale 折进 logits、V 的折进分子不折分母——代数恒等式有 CPU
反证测试）。步级 1.51×/1.46× 是"attention 占步一部分"的诚实结果。定位为
"复刻级实现+一个量化变体"，**不主张首创**——面试讲差异点（图捕获安全约束下的
设备端长度、逐窗 K 门）比讲倍数更有利。

### 2.3 瓶颈定位：方法学是全项目最强的部分

- **分层账本**：family 级"必要字节 ÷ 实测带宽"的地板 + gap 分类（config/latency/exclude）
  + **可捕量排名**——这正是 TRT-LLM casebook/ncu 工作流的简化版；
- **判死纪律**：config 类候选用 7 轮交错 + MAD 判平局（0.11% < MAD）正式关闭，不追
  沉没成本；ncu 被宿主策略阻断后交出替代证据链（字节地板 + 隔离冷测 + L2 flush +
  event 地板扣除），并明确标注"stall 归因缺失"——**工具受限时的诚实替代**本身是能力；
- **端到端利用率账**（外部锚点）：全步有效带宽 56–59%（913/1080 GB/s，E6 修正后
  869/918 GB/s），kernel 级 87.3%/94.3%——与 llama.cpp 类栈的 75–85% 相比，步级差值
  正是本项目自己分类过的 elementwise/routing 尾巴（p4c-02 的 13 family 表）。能同时给出
  "步级 59%"和"kernel 级 94%"并解释差值去向，比单报一个百分比有说服力得多。

### 2.4 融合线总评

- **做对的**：七条独立 A/B 全采纳、kernel/步 994→345、每步收益兑现率诚实账
  （E6 65%/27%）、假 0/复利口径/capture 初始化三个测量坑的定位与修复全档。
- **不足的**：Triton 而非 CUDA C++（传统算子岗偏好；AI 编译器/新栈岗位 Triton 是主流）；
  无 ncu/SASS 级深度；F3/F7 未做；无 GQA/MHA 变体与 paged KV（定位使然）。

## 3. 按岗位的适配判断

| 岗位 | 适配度 | 面试主要风险与答法 |
| --- | --- | --- |
| **推理优化（端侧）** | ★★★★★ | 几乎全对口：batch=1 单流、消费级卡、KV 容量/带宽、llama.cpp 同赛道。答法语料=本文 §1.1/§2.3 |
| **推理优化（云）** | ★★★★ | 强项=执行轴/量化/融合全链+纪律；会被戳 serving（PagedAttention/continuous batching/多卡）——答"规则 10 明确不接 vLLM 是 scope 选择，R7 外部锚点计划在档"；再加"份额封顶分析正是云上 batching 决策的同款思维" |
| **算子开发** | ★★★☆ | 6 个 Triton kernel 族+parity 门+config 扫描+occupancy 解析上界；缺 CUDA C++/PTX/SASS 与 ncu 深度——答"Triton 是项目预注册的工具选择；字节账/地板方法论与后端无关，转 CUDA 路径不变" |
| **AI 编译器** | ★★★☆ | 融合判定=手写 fusion pass；引用过 XLA 融合分析；开关+图捕获约束="pass 的合法性判定"思维。缺 torch.compile/TVM/MLIR 实操——可一句带过"手写版走完了编译器 fusion 要自动化的同一决策树" |

## 4. 如果再补两件事（性价比排序）

1. **KV int8 的任务级链**（PPL 全集 + MMLU 子集，一天窗）：把 §1.1 的口径缺口补上，
   直接对标 vLLM FP8 的评测形状；
2. **torch.compile（reduce-overhead）vs 手写栈的同轮对照**（半天）：一条数据回答
   "为什么不直接用编译器"——这对 AI 编译器岗是杀手级答法。

---

## 来源（外部参照）

- KIVI: [arXiv:2402.02750](https://arxiv.org/html/2402.02750v2) · [PMLR v235](https://proceedings.mlr.press/v235/liu24bz.html) · [GitHub](https://github.com/jy-yuan/KIVI)
- vLLM FP8 KV: [官方博客](https://vllm.ai/blog/2026-04-22-fp8-kvcache) · [TurboQuant](https://vllm.ai/blog/2026-05-11-turboquant) · [SqueezeBits 对比](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079) · [Red Hat](https://developers.redhat.com/articles/2024/07/15/vllm-brings-fp8-inference-open-source-community) · 长检索风险：[particula.tech](https://particula.tech/blog/kv-cache-quantization-accuracy-loss-benchmarks)
- Marlin/W4A16: [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) · [ETH 论文 PDF](https://www.research-collection.ethz.ch/server/api/core/bitstreams/b4908ef6-b203-4218-8bb5-e46d7d4f0dca/content) · [Red Hat 解读](https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference) · [LMDeploy W4A16](https://lmdeploy.readthedocs.io/en/v0.2.4/quantization/w4a16.html) · 反例经验：[Spheron](https://www.spheron.network/blog/myth-quantization-doesn-t-always-make-inference-faster) · [vLLM 论坛](https://discuss.vllm.ai/t/why-i-feel-cuda-kernel-marlin-run-not-fast/2197)
- MoE 量化敏感性: [MxMoE (ICML'25)](https://icml.cc/virtual/2025/poster/43995) · [GEMQ](https://www.alphaxiv.org/abs/2605.23078) · [ACL'25 Findings](https://aclanthology.org/2025.findings-acl.1386.pdf) · [MoEQuant](https://openreview.net/forum?id=0epuNvt5Dj) · [arXiv:2604.06515](https://arxiv.org/html/2604.06515v1)
- Flash-Decoding: [PyTorch 博客](https://pytorch.org/blog/flash-decoding) · [Stanford CRFM](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) · [flash-attention](https://github.com/dao-ailab/flash-attention) · [FlashInfer](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- 生产融合集: [TRT-LLM Kernel & Fusion Casebook](https://github.com/nvidia/tensorrt-llm/blob/main/.claude/skills/perf-optimization-casebook/references/kernel-and-fusion/index.md) · [TRT-LLM GPT Attention](https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html) · [PyTorch: 融 Norm 进 GEMM](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels)
- 社区 decode 基准: [NVIDIA llama.cpp 博客](https://developer.nvidia.com/blog/accelerating-llms-with-llama-cpp-on-nvidia-rtx-systems) · [modelfit 4090](https://modelfit.io/gpu/rtx-4090) · [llama.cpp 量化评测 (arXiv:2601.14277)](https://arxiv.org/html/2601.14277v1)
- OLMoE 本体: [arXiv:2409.02060](https://arxiv.org/html/2409.02060v1) · [模型卡](https://huggingface.co/allenai/OLMoE-1B-7B-0924)
