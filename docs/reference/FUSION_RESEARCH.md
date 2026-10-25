# FUSION_RESEARCH — 算子融合与量化融合：方法调研与避坑点

> **性质**：联网调研记录（2026-10-22，GitHub / arXiv / 官方文档），给融合任务（`FUSION_OPPORTUNITIES.md` 本地文档的 F1–F6）提供外部依据与避坑清单。
> **纪律**：本文所有收益数字都是**公开报告值**（出处见链接），**不是本项目实测值**——本项目实测值的唯一权威是 [RESULTS.md](../../RESULTS.md)，两套口径不得混用（硬规则 1）。链接均经逐一核验可点击；标注 ✓ = 本次 web_fetch 核验，◇ = 已收录于 FUSION_OPPORTUNITIES §3。

> **2026-10-22 二次复核**：承接已有资料，重新打开关键一手来源并对照当前代码；
> 旧表的 ✓/◇ 保留原检索含义，不表示本轮把每篇全文、每个性能数字重新验证过。
> 本轮可用于 review 的结论与适用限制见 **§5**；问题与验证队列见
> `review 二次复核`（本地文档 compose/spec/review-2026-10-22.md）。

---

## 1. 优化方法

### 1.1 执行 / 图捕获（launch 开销消除）

| 方法 | 核心思路 | 适用条件 | 公开报告收益 |
| --- | --- | --- | --- |
| [NVIDIA CUDA Graphs 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | 一次 capture 整段 kernel 序列，重放消除逐 kernel CPU launch | decode 步 shape/控制流静态，需 warm-up | 机制文档 |
| [vLLM CUDA Graphs 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/) ✓ | decode 整步捕获执行图，去 Python/launch 抖动 | 静态 shape decode step | 官方动机即 launch 开销 |
| [PyTorch CUDA Graph trees](https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html) ✓ | `mode="reduce-overhead"` 自动捕获重放 | 静态输入；动态 shape 触发重捕获 | 官方将 launch-overhead 列为动机 |
| [PyTorch: fullgraph=False](https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_false.html) ✓ | graph break 切段、逐段编译/launch | torch.compile 用户 | 文档明确 "graph breaks make performance worse" |
| [arXiv:2605.30571 Memory-Bound but Not Bandwidth-Limited](https://arxiv.org/abs/2605.30571) ✓ | batch=1 decode 跨 4 GPU 实测：launch 侧开销是"缺失项"，卡越快越暴露 | 7–8B GQA、bf16 SDPA、ctx 2048–16384、44 格 | **CUDA Graphs：H100 1.259×（95% CI 1.253–1.267）vs L4 仅 1.028×**；H100 只达内存地板 27%，L4 达 81% |

★ 与本项目的对应：此文是 E2 结论（融合 kernel 单体 1.21× 便宜、eager 端到端反而 5% 慢，host_share 71.7%、~19k kernels/step）的公开同构证据——**kernel 单体便宜 ≠ 端到端便宜，收益只在 graph/compile 下兑现**。同文还发现常见量化路径拿不到 4× 权重流量缩减（L4：bf16 62.32 ms/步、bnb-nf4 59.36、AutoAWQ+Marlin 45.24、GPTQ+ExLlamaV2 17.36）——量化收益是"实现形态"问题，不是"位宽"问题。

### 1.2 Triton 自研融合 kernel 实践

| 方法 | 核心思路 | 对应本项目 |
| --- | --- | --- |
| [Triton fused LayerNorm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) ✓ | 单 kernel 融 mean/var/rsqrt/normalize/affine | P2i fused RMSNorm |
| [Liger Kernel（arXiv:2410.10989）](https://arxiv.org/abs/2410.10989) ✓（[GitHub](https://github.com/linkedin/Liger-Kernel)） | Triton 融合 RMSNorm/RoPE/SwiGLU；训练吞吐 +20%、显存 −60% | F1 的 RoPE/norm 融合同型 |
| [vLLM fused_add_rms_norm](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) | residual-add + RMSNorm 单 kernel 写回 | F3 残差 epilogue |
| [vLLM fused rotary embedding](https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu) | rotate_half 链（cos/sin/split/mul/add）融成单 kernel | F1 fused rope（已测 8.39%） |
| [vLLM fused_moe（fused_topk）](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) | softmax+top-k 路由融成单 kernel | F1 fused route（已测 3.0%） |
| [DeepSpeed PR #8663](https://github.com/deepspeedai/DeepSpeed/pull/8663) | HF split-half 语义的 Triton 融合 RoPE（含数值等价讨论） | F1 fused rope |
| [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) / [Group GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) / [Block-Scaled Matmul](https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html) | 官方三大融合范式；块缩放对应 MXFP4 | grouped MoE kernel；FP8/MXFP4 方向入口 |

系统级自动融合（◇ 均已收录于 FUSION_OPPORTUNITIES §3）：[Blockbuster 2505.07829](https://arxiv.org/abs/2505.07829)（规则式 block 融合，自动"重新发现"FlashAttention，演示 RMSNorm+SwiGLU FFN mega-kernel ⇒ F2/F3）；[ClusterFusion 2508.18850](https://arxiv.org/abs/2508.18850)（**QKV+attention+O 合成单 kernel**，H100 1.61× ⇒ F2）；[Neptune 2510.08726](https://arxiv.org/abs/2510.08726)（融"归约→GEMM"链，平均 1.35×）；[DNNFusion 2108.13342](https://arxiv.org/abs/2108.13342)（先按数学性质改写图再融合，机会 8.8× ⇒ §2 红线⑤）；[XLA 融合分析 2301.13062](https://arxiv.org/abs/2301.13062)（**"融合不一定更快"**）；[GPUOS 2604.17861](https://arxiv.org/abs/2604.17861)（长驻 worker + host 任务队列，小算子最高 15.3×；**不能据此推出所有 persistent kernel 都与 CUDA Graph 互斥**，见 §5）。

### 1.3 MoE 专用 kernel

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [vLLM fused_moe 文档](https://docs.vllm.ai/en/v0.14.0/api/vllm/model_executor/layers/fused_moe/fused_moe/) + [源码](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) | fused_topk 路由 + grouped GEMM + 加权汇总；含 w8a16/gptq_marlin/awq_marlin/moe_wna16 | 生产默认路线 |
| [Marlin（arXiv:2408.11743）](https://arxiv.org/abs/2408.11743) ✓ | W4A16 dequant 融进 GEMM，per-group scale 寄存器加载+组内广播 | batch 16–32 近 4×；vLLM 端到端最高 2.8× |
| [MegaBlocks（arXiv:2211.15841）](https://arxiv.org/abs/2211.15841) ✓ | block-sparse GEMM，dropless | 训练端比 Tutel 快 40%（decode b=1 参考价值低） |
| [TensorRT-LLM moe_kernels.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) | CUTLASS grouped GEMM MoE | 生产级 |
| [FlashInfer（arXiv:2501.01005）](https://arxiv.org/abs/2501.01005) ✓ | gather+grouped GEMM+activation 一步融合；**CUDAGraph 兼容** | ITL 降 29–69%、长上下文 latency 降 28–30% |
| [SGLang（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104) ✓ | RadixAttention + fused MoE 集合 | 最高 6.4× 吞吐 |
| [SonicMoE（arXiv:2512.14080）](https://arxiv.org/abs/2512.14080) | MoE IO/tile-aware 优化；含前后向、训练与 token rounding | 方法参考；不是本项目 batch=1 decode 的收益证据 |

### 1.4 量化 kernel 融合

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [Marlin（arXiv:2408.11743）](https://arxiv.org/abs/2408.11743) ✓ | dequant 融进 GEMM；**per-group scale 加载+广播**（非逐元素 gather） | 近 4×（batch≤32）；端到端 2.8× |
| [GPTQ（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323) ✓ | 二阶近似逐列 PTQ + 融合 dequant GEMM | 端到端 3.25×（A100）/ 4.5×（A6000） |
| [AWQ（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978) ✓ | 激活感知通道缩放 + TinyChat kernel fusion | >3× vs HF FP16（MLSys 2024 Best Paper） |
| [SmoothQuant（arXiv:2211.10438）](https://arxiv.org/abs/2211.10438) ✓ | W8A8：激活离群难度迁移到权重 | 最高 1.56×、显存减半 |
| [Machete（Red Hat/Neural Magic）](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) ✓ | w4a16 dequant 融入 tensor-core 流水 | **Hopper sm90a 专属——sm_120 无 wgmma，不可用** |

★ **scale 布局与复用**：Marlin 明确把权重和 group scales 离线重排以匹配 kernel 访存。本项目 P2d 的 gather→broadcast 已有自己的 A/B（8.14→5.15 µs/call，523→826 GB/s）。可以借鉴“减少冗余装载”的方法，但不能仅凭源码写了 gather 就认定其他 kernel 必慢，还需检查编译结果与真实形状计时。

### 1.5 split-KV / paged attention / KV 直写

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [Flash-Decoding（Stanford CRFM）](https://crfm.stanford.edu/2023/10/12/flashdecoding) ✓ | KV 维 split 并行 + LSE 跨 split 归约 | batch=1 时 FA 只用 <1% GPU；attention 单体最高 50×、**端到端最高 8×**（单体≠端到端的教科书案例） |
| [FlashDecoding++（arXiv:2311.01282）](https://arxiv.org/abs/2311.01282) ✓ | 统一 max 异步 softmax 去跨 split 同步 + 双缓冲 | vs HF 最高 4.86×/2.18×；vs SOTA 平均 1.37× |
| [FlashAttention 2205.14135](https://arxiv.org/abs/2205.14135) / [FA2 2307.08691](https://arxiv.org/abs/2307.08691) | IO-aware tiling + online softmax | FA2 ≈ 2× FA1 |
| [PagedAttention/vLLM（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) ✓ | 分页 KV 的管理与寻址；cache 写入与 attention 是否为同一 kernel 须另查源码，`reshape_and_cache` 这个名字本身不能证明融合 | 吞吐 2–4×（服务系统公开报告值，不能当 F4 单项收益） |

### 1.6 OLMoE / 小 MoE decode 与 sm_120 适配

1. **模型**：[OLMoE（arXiv:2409.02060）](https://arxiv.org/abs/2409.02060) ✓（7B 总参/1B 激活，全开放）。原模型论文不提供本仓自研 decode kernel 的性能证据；下面的 MoE 推理研究也包含 OLMoE，不能泛称“没有相关论文”。
2. **[arXiv:2608.26612 Launch-Bound and Substitutable](https://arxiv.org/abs/2608.26612)** ✓（本仓 skip-mask-fill spec 已引用）：在 OLMoE-1B-7B 等三个 MoE 上实测——融合 Triton kernel **单体 5.6–9.0×、端到端 0.999×**（对比实测天花板 1.07×），因为时间花在 ~1000 次 kernel launch 的等待上；INT4 量化只平均改变 8 个选中专家里的 0.53 个/位置，重放变化路由只复现 2.7% 质量损失（专家可替代）；**去掉全部 23 个 graph break 反而慢 3×**。
3. **sm_120（RTX 5090，无 wgmma）**：[vLLM#36865](https://github.com/vllm-project/vllm/issues/36865) ✓（5090 构建仍注册不支持的 FlashMLA/FA 目标、Marlin 默认配置非 SM120）；[vLLM#43814](https://github.com/vllm-project/vllm/pull/43814) ✓（为 SM120/121 启用 CUTLASS grouped GEMM MoE 的 bugfix）。结论：**Machete 等 Hopper 专属 kernel 不可用；Marlin 可行但默认值有坑；Triton 3.4（本项目现役）不依赖 wgmma，是 5090 上最稳的融合载体**。

---

## 2. 避坑点清单

> 每条：陷阱 → 出处 → 对本项目的落点。

**① 不要在 CUDA Graph capture 内触发首次 autotune/JIT。** autotune 会多次执行候选，可能破坏 capture 或重复修改输出；应在 capture 外完成调优、预热并固定配置。这里要区分“capture 内首次调优”与“预先调好配置后的调用”，不能把前者的问题写成所有 autotune 与 graph 天然不兼容。本项目仍遵守现有固定配置、禁止新增 `@triton.autotune` 的规则。出处：[Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)、[triton#10596](https://github.com/triton-lang/triton/issues/10596)。落点：所有融合 kernel 的 `warm()` 惯例。

**② 图外 in-place 输入更新的语义坑。** CUDA Graph 重放读的是**捕获时绑定的地址**；in-place 更新输入是正确姿势，但 reassign（`x = new_tensor`）会静默失效；`make_graphed_callables` 甚至会静默丢弃 in-place 变更。出处：[pytorch#178586](https://github.com/pytorch/pytorch/issues/178586)、[NVIDIA CUDA graph 数值错误排查](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/numerical-errors.html)。落点：`LEMOQ_SKIP_MASK_FILL`（跳过 mask 填充 = 喂陈旧数据，须保证图内无人读 mask——tokens_equal 门就是干这个的）；F4 KV 直写改造时的 cache 地址固定性。

**③ MoE 路由的数据依赖破坏编译/捕获。** 路由 top-k 是数据依赖形状（gather 规模随专家分布变化），是 torch.compile graph break 的经典来源；业界修法是把路由输出定形（K 固定）+ 求和类算子放 `no_grad` 下。出处：[sglang#31246](https://github.com/sgl-project/sglang/pull/31246)、[arXiv:2608.26612](https://arxiv.org/abs/2608.26612)（graph break 全修反而慢 3×）。落点：本项目 `_decode_forward` 的"固定形状"设计（top_k 固定、无 `.item()`/`nonzero`）是对的，F 系列新 kernel 不得破坏它。

**④ 融合会反噬 occupancy。** live range 变长 → 寄存器压力上升 → 从"两个快的"变成"一个慢的"。出处：XLA 融合分析（[2301.13062](https://arxiv.org/abs/2301.13062)，"融合不一定更快"）；本仓证据是 36 点 config 网格差 3.2×（`moe_grouped.py` 注释）。落点：**每次融合后重扫 config**（F2 验收里已写）。

**⑤ 数值语义必须逐位对齐；冲突时放弃融合。** fp16/bf16 1 ulp、舍入点（先舍入再乘 vs 先乘再舍入）、不同 torch/transformers 版本 aten 实现变化都会让"逐位一致"失效。出处：本仓 e5e0631（bf16 1 ulp≈8e-3 过不了 1e-3 K1 ⇒ 回退 HF）、fused RMSNorm 的舍入契约。落点：F3 的加法顺序、F4 的 cache 内容逐位对齐（`tokens_equal` 硬门）。

**⑥ 测量口径：单体 ≠ 端到端，host ≠ GPU。** Amdahl/launch-bound；profiler（CUPTI）本身有开销、会扰动计时；共租噪声使 host 时间虚高。出处：[arXiv:2608.26612](https://arxiv.org/abs/2608.26612)（0.999×）、Flash-Decoding 博客（单体 50× vs 端到端 8×）、[CUPTI 文档](https://docs.nvidia.com/cuda/archive/11.0_GA/cupti/pdf/Cupti.pdf)（隔离被测上下文）。落点：`tools/round_canary` 金丝雀协议、`host_share_pct>5` 污染判据、A/B 工具的 busy 双口径。

**⑦ sm_120 适配坑。** Marlin 默认配置非 SM120（[vLLM#36865](https://github.com/vllm-project/vllm/issues/36865)）、CUTLASS grouped GEMM 需补丁（[vLLM#43814](https://github.com/vllm-project/vllm/pull/43814)）、Hopper 专属 kernel（Machete/wgmma）不可用。落点：F6 或任何引 Marlin/Machete 的路线前先核对架构；Triton 路线是 5090 的安全区。

**⑧ 量化融合的形态坑。** 检查 scale 的布局与重复装载，本仓 P2d 的修复已有自己的证据（见 §1.4）；不要仅凭源码形态推广性能结论。4-bit 门 FAIL 时，先用同一 packed 权重的独立 oracle 排除实现偏差，再评估 GPTQ/AWQ 等算法路线；算法与 kernel 不能混为一个变量。F6 继续受正确性门阻塞。

**⑨ 融合收益要按"每次 launch 多少 µs"记账。** 0.0544 ms/32 次 = 1.7 µs/次 才有决策价值；"占比 2.8%"会误导。出处：本仓 FUSION_OPPORTUNITIES 红线⑦（与 §1.3.3 的 launch-bound 结论一致）。

**⑩ 融合要能回退、要能被测。** env 开关默认关 + 正确性门 + 金丝雀，使“候选/已采纳”边界可查。`tokens_equal` 只是其中一层，不能代替 logits、独立 oracle 与真实 prompt 集。arXiv:2608.26612 **实际做了端到端测量**；其单体提升而整步无收益的负结果说明这种验收不可省略。

---

## 3. 与 F1–F6 的对应

| 任务 | 外部依据 |
| --- | --- |
| F1 融合 RoPE/routing/mask-fill | Liger 2410.10989、DeepSpeed PR #8663、vLLM fused rotary embedding / fused_topk |
| F2 q/k/v 三合一 | ClusterFusion 2508.18850（QKV+O 合一，H100 1.61×）、Blockbuster mega-kernel |
| F3 残差 epilogue | vLLM fused_add_rms_norm、DNNFusion 2108.13342 |
| F4 KV 直写 | PagedAttention 2309.06180 的 `reshape_and_cache`、FlashInfer cache 写算子 |
| F5 lm_head w8a16 | SmoothQuant 2211.10438（W8A8 先例）、W8A16 dequant-GEMM 同型 |
| F6 专家 w4a16（阻塞） | 算法前置 GPTQ 2210.17323 / AWQ 2306.00978；kernel 形态参照 Marlin/Machete |

---

## 4. 汇总参考链接表

| 类别 | 链接 | 核验 |
| --- | --- | --- |
| 执行/图捕获 | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html | 检索命中 |
| | https://docs.vllm.ai/en/latest/design/cuda_graphs/ | ✓ |
| | https://docs.pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html | ✓ |
| | https://docs.pytorch.org/docs/stable/compile/programming_model.fullgraph_false.html | ✓ |
| | https://docs.vllm.ai/en/latest/design/fusions/ | ✓ |
| | https://arxiv.org/abs/2605.30571 | ✓ |
| Triton 融合 | https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html | ✓ |
| | https://arxiv.org/abs/2410.10989 | ✓ |
| | https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu | 检索命中 |
| | https://github.com/vllm-project/vllm/blob/main/csrc/pos_encoding_kernels.cu | 检索命中 |
| | https://github.com/deepspeedai/DeepSpeed/pull/8663 | 检索命中 |
| 自动/系统融合 | https://arxiv.org/abs/2505.07829 / 2508.18850 / 2510.08726 / 2108.13342 / 2301.13062 / 2604.17861 | ◇ |
| MoE kernel | https://docs.vllm.ai/en/v0.14.0/api/vllm/model_executor/layers/fused_moe/fused_moe/ | 检索命中 |
| | https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe | ✓ |
| | https://arxiv.org/abs/2408.11743 | ✓ |
| | https://arxiv.org/abs/2211.15841 | ✓ |
| | https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu | 检索命中 |
| | https://arxiv.org/abs/2501.01005 | ✓ |
| | https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/fused_moe | 检索命中 |
| | https://github.com/HazyResearch/ThunderKittens | 检索命中 |
| | https://github.com/ROCm/aiter | 检索命中 |
| | https://arxiv.org/abs/2312.07104 | ✓ |
| | https://arxiv.org/abs/2512.14080 | 检索命中 |
| 量化 | https://arxiv.org/abs/2210.17323（GPTQ） | ✓ |
| | https://arxiv.org/abs/2306.00978（AWQ） | ✓ |
| | https://arxiv.org/abs/2211.10438（SmoothQuant） | ✓ |
| | https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel | ✓ |
| | https://github.com/vllm-project/vllm/tree/main/csrc/quantization | 检索命中 |
| Attention | https://crfm.stanford.edu/2023/10/12/flashdecoding | ✓ |
| | https://arxiv.org/abs/2311.01282 | ✓ |
| | https://arxiv.org/abs/2205.14135 / https://arxiv.org/abs/2307.08691 | 经典文献 |
| | https://arxiv.org/abs/2309.06180 | ✓ |
| 避坑点 | https://github.com/triton-lang/triton/issues/10596 | 检索命中 |
| | https://github.com/pytorch/pytorch/issues/178586 | 检索命中 |
| | https://docs.nvidia.com/dl-cuda-graph/troubleshooting/numerical-errors.html | 检索命中 |
| | https://github.com/sgl-project/sglang/pull/31246 | 检索命中 |
| | https://docs.nvidia.com/cuda/archive/11.0_GA/cupti/pdf/Cupti.pdf | 检索命中 |
| 模型/硬件 | https://arxiv.org/abs/2409.02060（OLMoE） | ✓ |
| | https://arxiv.org/abs/2608.26612 | ✓ |
| | https://github.com/vllm-project/vllm/issues/36865 | ✓ |
| | https://github.com/vllm-project/vllm/pull/43814 | ✓ |
| | https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html | ✓ |

---

## 5. 二次复核：怎样把资料变成可执行的 review

### 5.1 本轮重查的一手依据及迁移边界

检索日期：**2026-10-22**。论文核对标题、摘要与适用对象；涉及具体实现时继续读官方源码/API。
未在本仓重现这些外部实现，不把它们的加速比当成本仓目标或结果。

| 一手来源 | 可用的方法或避坑点 | 本仓落点与边界 |
| --- | --- | --- |
| [Marlin 原仓 README](https://github.com/IST-DASLab/marlin#techniques)、[论文](https://arxiv.org/abs/2408.11743) | 离线布局重排、量化与计算流水、寄存器复用、跨 warp 分工 | dense/grouped 的配置与访存检查；原仓架构要求不是当前 vLLM 全部后端的支持矩阵，也不证明 sm_120 上某配置最优 |
| [Triton fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | 小归约+逐点链融合，减少物化及重复提交 | routing/RMSNorm；需同时核对 exp、归约顺序与输出舍入 |
| [Triton autotune API](https://triton-lang.org/main/python-api/generated/triton.autotune.html) | 一个 key 的调优会多次执行 kernel，写入状态要 reset/restore | cache、残差输出不能被试跑重复累积；本项目继续显式固定配置 |
| [CUDA Graph 官方指南](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) | 图重放降低 CPU 提交开销，节点依赖与参数要正确 | 区分 host 提交、GPU 节点间隙、访存成本；图内减少小 kernel 仍可能有收益 |
| [PyTorch CUDA Graph 约束](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs) | 保持捕获张量地址、capture 外预热、管理输出生命周期 | KV 与 filled_len 原位更新；不能用 Python rebind 当成更新图输入 |
| [HF OLMoE v4.57.6 源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/olmoe/modeling_olmoe.py) | 此版本 RoPE 的 `position_ids` **弃用且未使用** | 旧 review P4 的错误前提必须撤回；不要拿不同版本的签名推断当前语义 |
| [PyTorch 2.8 topk API](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html) | 平局索引不保证稳定 | top-k 边界 tie 可能换专家集合；同权重不代表不同专家输出相等 |
| [vLLM residual + RMSNorm 源码](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu) | 把相邻小算子融合，并显式检查 dtype、stride 与布局 | F3 可借鉴接口约束；vLLM 的数值契约不自动等于本仓 HF 参考 |
| [FlashInfer MoE API](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/api/fused_moe.rst)、[oracle 设计](https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md) | capture 前准备 workspace；独立 torch oracle 能发现“两边调用同一错误 kernel”的盲区 | 只迁移单卡测试思想；该资料含 EP，不引入多 GPU 方案，也不升级本仓依赖 |
| [Nsight Compute 指标定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#occupancy)、[Blackwell tuning](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | 区分 grid、waves、资源上界和实际 active warps | F2 的 grid 推导不能直接宣布实测 occupancy；sm_120 的资源不能套用 sm_90/sm_100 |
| [Blockbuster](https://arxiv.org/abs/2505.07829)、[ClusterFusion](https://arxiv.org/abs/2508.18850) | 数据流/片上通信决定融合范围 | 作为 F2/F3 的设计参考；ClusterFusion 的 H100 与 cluster 原语不构成 5090 可直接移植的证明 |
| [SonicMoE v2](https://arxiv.org/abs/2512.14080v2) | IO/tile 利用率与 padding 成本；token rounding 改变路由策略 | 训练/前后向吞吐不能当 batch=1 decode 收益；不把改变路由的算法混进“等价融合” |
| [Memory-Bound but Not Bandwidth-Limited](https://arxiv.org/abs/2605.30571)、[Launch-Bound and Substitutable v2](https://arxiv.org/abs/2608.26612v2) | 分别验证单体、整步和执行方式；路由一致性与质量是不同指标 | 支持审查方法，不重判本仓已完成的 E2/E7，也不把外部卡/模型数值搬过来 |
| [GPUOS](https://arxiv.org/abs/2604.17861)、[Triton persistent matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | 常驻 worker 队列与有限任务的 persistent tile 调度是不同生命周期 | 不能把一个 runtime 的集成限制泛化为所有 persistent kernel 的图兼容性；本轮不引入任何一种 |

GitHub `main` 和 Triton `main` 是检索当日资料，不能视为本仓 torch 2.8.0 / Triton 3.4.0 已支持的功能。
HF 的争议则用服务器同版 **v4.57.6** 的源码核实，并与 Mac 已安装版本互证。

### 5.2 需要纠正的优化推理

1. **收益不只来自减少 DRAM 往返。** 本仓小张量链也可能受节点调度/执行延迟限制。
   应分别记录省掉的中间张量字节、kernel 节点数、wall 和 GPU 时间；不能只选其中一个解释所有收益。
2. **`grid × warps / SM数` 不是实测 occupancy。** 还需寄存器、shared memory、驻留限制、执行波次和尾部行为。
   F2 目前的“根因已确定”和收益区间应降为待验证假设。grid 变大也不保证输入只读一次。
3. **残差融合要保留舍入点。** 当前先 `fp32 acc → fp16 中间量`，再加 residual；
   `fp32(acc + residual) → fp16` 不是同一数值过程。F3 的验收应明确中间 cast，而非只讨论加法交换律。
4. **消费者数量是分析输入，不是融合的充分必要条件。** 多消费者可通过保留一个物化输出处理；
   单消费者也可能因归约同步、寄存器压力而不适合融合。应画实际数据依赖，而不是据此直接批准或禁止 F3。
5. **技术可行性与本仓规则分开。** 本仓禁止新增 autotune、暂不做 persistent runtime，仍照办；
   但理由应是当前可复现与集成边界，不能写成 CUDA 的普遍限制。
6. **“未发现问题”不等于“已证明算法问题”。** W4 门 FAIL 是事实；在同一 packed 权重的独立 oracle、
   多 prompt/多层对照补齐之前，不把所有 W4 误差直接归因给 RTN，也不发布其性能。

### 5.3 优化机会的处理顺序

先修补测量与数值证据链，再验证 F1；F2/F3/F4 均保留为未实测机会。
F5 需要独立精度证据，F6 继续受 W4 正确性门阻塞。每项先单变量、后组合，
以真实 decode 形状在图内测量，报告 median/MAD、逐对胜率、资源与 VRAM 分项。
本节没有新增实验、cell、性能结果或默认开关变更。
