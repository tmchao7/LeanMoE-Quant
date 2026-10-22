# FUSION_RESEARCH — 算子融合与量化融合：方法调研与避坑点

> **性质**：联网调研记录（2026-10-22，GitHub / arXiv / 官方文档），给融合任务（[FUSION_OPPORTUNITIES.md](../FUSION_OPPORTUNITIES.md) 的 F1–F6）提供外部依据与避坑清单。
> **纪律**：本文所有收益数字都是**公开报告值**（出处见链接），**不是本项目实测值**——本项目实测值的唯一权威是 [RESULTS.md](../../RESULTS.md)，两套口径不得混用（硬规则 1）。链接均经逐一核验可点击；标注 ✓ = 本次 web_fetch 核验，◇ = 已收录于 FUSION_OPPORTUNITIES §3。

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

系统级自动融合（◇ 均已收录于 FUSION_OPPORTUNITIES §3）：[Blockbuster 2505.07829](https://arxiv.org/abs/2505.07829)（规则式 block 融合，自动"重新发现"FlashAttention，演示 RMSNorm+SwiGLU FFN mega-kernel ⇒ F2/F3）；[ClusterFusion 2508.18850](https://arxiv.org/abs/2508.18850)（**QKV+attention+O 合成单 kernel**，H100 1.61× ⇒ F2）；[Neptune 2510.08726](https://arxiv.org/abs/2510.08726)（融"归约→GEMM"链，平均 1.35×）；[DNNFusion 2108.13342](https://arxiv.org/abs/2108.13342)（先按数学性质改写图再融合，机会 8.8× ⇒ §2 红线⑤）；[XLA 融合分析 2301.13062](https://arxiv.org/abs/2301.13062)（**"融合不一定更快"**）；[GPUOS 2604.17861](https://arxiv.org/abs/2604.17861)（长驻 kernel 彻底消 launch，小算子最高 15.3×，**与 CUDA Graph 互斥**）。

### 1.3 MoE 专用 kernel

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [vLLM fused_moe 文档](https://docs.vllm.ai/en/v0.14.0/api/vllm/model_executor/layers/fused_moe/fused_moe/) + [源码](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/fused_moe) | fused_topk 路由 + grouped GEMM + 加权汇总；含 w8a16/gptq_marlin/awq_marlin/moe_wna16 | 生产默认路线 |
| [Marlin（arXiv:2408.11743）](https://arxiv.org/abs/2408.11743) ✓ | W4A16 dequant 融进 GEMM，per-group scale 寄存器加载+组内广播 | batch 16–32 近 4×；vLLM 端到端最高 2.8× |
| [MegaBlocks（arXiv:2211.15841）](https://arxiv.org/abs/2211.15841) ✓ | block-sparse GEMM，dropless | 训练端比 Tutel 快 40%（decode b=1 参考价值低） |
| [TensorRT-LLM moe_kernels.cu](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu) | CUTLASS grouped GEMM MoE | 生产级 |
| [FlashInfer（arXiv:2501.01005）](https://arxiv.org/abs/2501.01005) ✓ | gather+grouped GEMM+activation 一步融合；**CUDAGraph 兼容** | ITL 降 29–69%、长上下文 latency 降 28–30% |
| [SGLang（arXiv:2312.07104）](https://arxiv.org/abs/2312.07104) ✓ | RadixAttention + fused MoE 集合 | 最高 6.4× 吞吐 |
| [SonicMoE（arXiv:2512.14080，ICLR 2026）](https://arxiv.org/abs/2512.14080) | MoE IO/tile-aware 优化 | 中小 MoE decode 相关 |

### 1.4 量化 kernel 融合

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [Marlin（arXiv:2408.11743）](https://arxiv.org/abs/2408.11743) ✓ | dequant 融进 GEMM；**per-group scale 加载+广播**（非逐元素 gather） | 近 4×（batch≤32）；端到端 2.8× |
| [GPTQ（arXiv:2210.17323）](https://arxiv.org/abs/2210.17323) ✓ | 二阶近似逐列 PTQ + 融合 dequant GEMM | 端到端 3.25×（A100）/ 4.5×（A6000） |
| [AWQ（arXiv:2306.00978）](https://arxiv.org/abs/2306.00978) ✓ | 激活感知通道缩放 + TinyChat kernel fusion | >3× vs HF FP16（MLSys 2024 Best Paper） |
| [SmoothQuant（arXiv:2211.10438）](https://arxiv.org/abs/2211.10438) ✓ | W8A8：激活离群难度迁移到权重 | 最高 1.56×、显存减半 |
| [Machete（Red Hat/Neural Magic）](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) ✓ | w4a16 dequant 融入 tensor-core 流水 | **Hopper sm90a 专属——sm_120 无 wgmma，不可用** |

★ **scale 广播 vs gather**：Marlin/AWQ/GPTQ 系一致采用 per-group 加载 + 组内广播，与本项目 P2d 修复（gather→broadcast，8.14→5.15 µs/call，523→826 GB/s）同型——外部做法支持"逐元素 gather scale 是错误形态"的判断。

### 1.5 split-KV / paged attention / KV 直写

| 方法 | 核心思路 | 公开报告收益 |
| --- | --- | --- |
| [Flash-Decoding（Stanford CRFM）](https://crfm.stanford.edu/2023/10/12/flashdecoding) ✓ | KV 维 split 并行 + LSE 跨 split 归约 | batch=1 时 FA 只用 <1% GPU；attention 单体最高 50×、**端到端最高 8×**（单体≠端到端的教科书案例） |
| [FlashDecoding++（arXiv:2311.01282）](https://arxiv.org/abs/2311.01282) ✓ | 统一 max 异步 softmax 去跨 split 同步 + 双缓冲 | vs HF 最高 4.86×/2.18×；vs SOTA 平均 1.37× |
| [FlashAttention 2205.14135](https://arxiv.org/abs/2205.14135) / [FA2 2307.08691](https://arxiv.org/abs/2307.08691) | IO-aware tiling + online softmax | FA2 ≈ 2× FA1 |
| [PagedAttention/vLLM（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) ✓ | 新 token KV **attention 路径内直写**（`reshape_and_cache`），免外部 cat/copy | 吞吐 2–4×（⇒ F4 依据） |

### 1.6 OLMoE / 小 MoE decode 与 sm_120 适配

1. **模型**：[OLMoE（arXiv:2409.02060）](https://arxiv.org/abs/2409.02060) ✓（7B 总参/1B 激活，全开放）。公开文献**没有专门针对 OLMoE decode kernel 的优化论文**。
2. **[arXiv:2608.26612 Launch-Bound and Substitutable](https://arxiv.org/abs/2608.26612)** ✓（本仓 skip-mask-fill spec 已引用）：在 OLMoE-1B-7B 等三个 MoE 上实测——融合 Triton kernel **单体 5.6–9.0×、端到端 0.999×**（对比实测天花板 1.07×），因为时间花在 ~1000 次 kernel launch 的等待上；INT4 量化只平均改变 8 个选中专家里的 0.53 个/位置，重放变化路由只复现 2.7% 质量损失（专家可替代）；**去掉全部 23 个 graph break 反而慢 3×**。
3. **sm_120（RTX 5090，无 wgmma）**：[vLLM#36865](https://github.com/vllm-project/vllm/issues/36865) ✓（5090 构建仍注册不支持的 FlashMLA/FA 目标、Marlin 默认配置非 SM120）；[vLLM#43814](https://github.com/vllm-project/vllm/pull/43814) ✓（为 SM120/121 启用 CUTLASS grouped GEMM MoE 的 bugfix）。结论：**Machete 等 Hopper 专属 kernel 不可用；Marlin 可行但默认值有坑；Triton 3.4（本项目现役）不依赖 wgmma，是 5090 上最稳的融合载体**。

---

## 2. 避坑点清单

> 每条：陷阱 → 出处 → 对本项目的落点。

**① `@triton.autotune` 与 CUDA Graph capture 不兼容。** capture 期间 autotune 会真跑多个 config，往图里塞计划外 kernel。grid 必须 host 侧常量；新 kernel 必须 capture 前 warm（JIT 编译完成）。出处：本仓 FUSION_OPPORTUNITIES 红线④、[triton#10596](https://github.com/triton-lang/triton/issues/10596)（capture 中 sync 破坏流）。落点：所有融合 kernel 的 `warm()` 惯例、`_dequant_triton.py` 模块 docstring。

**② 图外 in-place 输入更新的语义坑。** CUDA Graph 重放读的是**捕获时绑定的地址**；in-place 更新输入是正确姿势，但 reassign（`x = new_tensor`）会静默失效；`make_graphed_callables` 甚至会静默丢弃 in-place 变更。出处：[pytorch#178586](https://github.com/pytorch/pytorch/issues/178586)、[NVIDIA CUDA graph 数值错误排查](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/numerical-errors.html)。落点：`LEMOQ_SKIP_MASK_FILL`（跳过 mask 填充 = 喂陈旧数据，须保证图内无人读 mask——tokens_equal 门就是干这个的）；F4 KV 直写改造时的 cache 地址固定性。

**③ MoE 路由的数据依赖破坏编译/捕获。** 路由 top-k 是数据依赖形状（gather 规模随专家分布变化），是 torch.compile graph break 的经典来源；业界修法是把路由输出定形（K 固定）+ 求和类算子放 `no_grad` 下。出处：[sglang#31246](https://github.com/sgl-project/sglang/pull/31246)、[arXiv:2608.26612](https://arxiv.org/abs/2608.26612)（graph break 全修反而慢 3×）。落点：本项目 `_decode_forward` 的"固定形状"设计（top_k 固定、无 `.item()`/`nonzero`）是对的，F 系列新 kernel 不得破坏它。

**④ 融合会反噬 occupancy。** live range 变长 → 寄存器压力上升 → 从"两个快的"变成"一个慢的"。出处：XLA 融合分析（[2301.13062](https://arxiv.org/abs/2301.13062)，"融合不一定更快"）；本仓证据是 36 点 config 网格差 3.2×（`moe_grouped.py` 注释）。落点：**每次融合后重扫 config**（F2 验收里已写）。

**⑤ 数值语义必须逐位对齐；冲突时放弃融合。** fp16/bf16 1 ulp、舍入点（先舍入再乘 vs 先乘再舍入）、不同 torch/transformers 版本 aten 实现变化都会让"逐位一致"失效。出处：本仓 e5e0631（bf16 1 ulp≈8e-3 过不了 1e-3 K1 ⇒ 回退 HF）、fused RMSNorm 的舍入契约。落点：F3 的加法顺序、F4 的 cache 内容逐位对齐（`tokens_equal` 硬门）。

**⑥ 测量口径：单体 ≠ 端到端，host ≠ GPU。** Amdahl/launch-bound；profiler（CUPTI）本身有开销、会扰动计时；共租噪声使 host 时间虚高。出处：[arXiv:2608.26612](https://arxiv.org/abs/2608.26612)（0.999×）、Flash-Decoding 博客（单体 50× vs 端到端 8×）、[CUPTI 文档](https://docs.nvidia.com/cuda/archive/11.0_GA/cupti/pdf/Cupti.pdf)（隔离被测上下文）。落点：`tools/round_canary` 金丝雀协议、`host_share_pct>5` 污染判据、A/B 工具的 busy 双口径。

**⑦ sm_120 适配坑。** Marlin 默认配置非 SM120（[vLLM#36865](https://github.com/vllm-project/vllm/issues/36865)）、CUTLASS grouped GEMM 需补丁（[vLLM#43814](https://github.com/vllm-project/vllm/pull/43814)）、Hopper 专属 kernel（Machete/wgmma）不可用。落点：F6 或任何引 Marlin/Machete 的路线前先核对架构；Triton 路线是 5090 的安全区。

**⑧ 量化融合的形态坑。** 逐元素 gather scale 是错误形态（应 per-group 加载+广播，见 §1.4）；4-bit 的精度瓶颈是**量化算法**（RTN 不够 → GPTQ/AWQ），不是 kernel。落点：P2d 已修；F6 的前置是 ROADMAP C2。

**⑨ 融合收益要按"每次 launch 多少 µs"记账。** 0.0544 ms/32 次 = 1.7 µs/次 才有决策价值；"占比 2.8%"会误导。出处：本仓 FUSION_OPPORTUNITIES 红线⑦（与 §1.3.3 的 launch-bound 结论一致）。

**⑩ 融合要能回退、要能被测。** env 开关默认关 + `tokens_equal` 门 + 金丝雀，使"候选/已采纳"边界机械可查。出处：本仓红线⑧；arXiv:2608.26612 的"单体 5.6–9×/端到端 0.999×"就是没做端到端验收的反面教材。

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
