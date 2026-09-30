# MODELS — 模型候选与选型理由

`configs/models.yaml` 是唯一来源；本文解释**为什么选它们**。

---

## 1. 分层

| 角色 | 模型 | 精度 | 用途 | 32GB 卡 |
|---|---|---|---|---|
| dev | `Qwen/Qwen2.5-0.5B` | fp16 | kernel 迭代、单测、`--dry-run`、**Mac CPU 冒烟测试** | ✅ |
| dev | `Qwen/Qwen2.5-1.5B` | fp16 | dense 曲线主体 | ✅ |
| **primary (MoE)** | **`allenai/OLMoE-1B-7B-0924`** | fp16 | **MoE 主线实验** | ✅（13.84 GB） |
| dense 对照 | `Qwen/Qwen3-4B` | fp16 | 同规模 dense vs MoE 对比 | ✅ |
| dense 对照 | `Qwen/Qwen3-8B` | fp16 | 上界对照 | ✅（紧） |
| stretch | `Qwen/Qwen3-30B-A3B` | INT4 | scale 单点，**需 48GB 卡** | ❌ |

> ⚠️ `-0924` 后缀**必需**：`allenai/OLMoE-1B-7B` 不存在（hf-mirror 对不存在仓库返回 401，
> 2026-09 核实）。
>
> 用 **base** checkpoint，不用 `-Instruct`：本项目的主门槛是 KL / wikitext-2 PPL，
> 以 base 模型为参考是量化文献的标准协议。若日后要 GSM8K/MMLU 任务指标，再单独加
> `allenai/OLMoE-1B-7B-0924-Instruct` 臂，并标注两个 checkpoint **不可混比**。

---

## 2. 为什么 MoE 主线是 OLMoE-1B-7B，而不是 Qwen3-30B-A3B

| 判据 | OLMoE-1B-7B | Qwen3-30B-A3B |
|---|---|---|
| 能否在 32GB V100 跑 fp16 | ✅ 13.8 GB | ❌ 61 GB |
| 机制是否相同（稀疏激活 / 路由 / 专家权重不连续） | ✅ 64 experts, top-8 | ✅ 128 experts, top-8 |
| 单次前向迭代速度 | 快（实验矩阵可多跑 10×） | 慢 |
| 是否被主流框架支持 | ✅ vLLM / SGLang | ✅ vLLM / SGLang |
| 权重/数据/日志是否完全开放 | ✅ 完全开放（OLMo 生态） | 权重开放 |
| 是否"从业者真在部署的规模" | ❌ 小 | ✅ 是 |

**结论**：要研究的**性质**（稀疏激活下的字节预算、路由开销、专家层量化敏感度）在两者上
都成立，而 OLMoE 能在**不 OOM、不烧钱、迭代快**的前提下把这些性质测清楚。
Qwen3-30B-A3B 作为**收尾的单点验证**，用来回答"结论是否能 scale"，不作为主线。

**这条选型必须写进报告**：面试官会问"为什么不用大模型"，答案是
"机制可测性 + 硬件约束"，而不是"我跑不动"。

---

## 3. 架构数字

### 3.1 OLMoE-1B-7B-0924 —— **已从 `config.json` 逐项核实**

`allenai/OLMoE-1B-7B-0924` 的 `config.json` 已通过 `hf-mirror.com` 获取并逐项核对（2026-09）。
下表是从 config **推导**的值：不是实测，但推导过程可复核。

| 项 | 值 |
|---|---|
| layers | 16 |
| hidden_size | 2048 |
| num_experts | 64 |
| num_experts_per_tok | 8 |
| intermediate_size（expert FFN dim） | 1024 |
| num_attention_heads / num_key_value_heads | 16 / **16 → MHA，不是 GQA** |
| vocab_size | 50304 |
| `tie_word_embeddings` | **false → lm_head 独立，每 token 必须完整读一遍** |
| `torch_dtype` | **bfloat16 → 必须转 fp16，并记录精度落差** |
| `max_position_embeddings` | **4096** ⚠️ 见 §3.3 |
| 总参数 | 6,919,028,736 ≈ **6.92 B** |
| 激活/token（不含 embedding） | 1,178,861,568 ≈ **1.18 B** |
| fp16 权重全量 | 13,838,057,472 B = **13.84 GB (12.89 GiB)** |

推导：`(attention 16,777,216 + router 131,072 + 64 × 6,291,456 experts) × 16 层 + 2 × embedding(103,022,592) = 6,919,028,736`。

### 3.2 每-token 字节预算（fp16、batch=1）—— 本项目的核心数字

**这是 `PLAN.md` 中 `~2.25 GB` 的完整口径**：它是**权重字节 [W]，不含 KV**。

| 项 | 参数量 | fp16 字节 | 占 [W] |
|---|---|---|---|
| attention (q,k,v,o) | 268,435,456 | 0.537 GB | 22.8% |
| router (gate) | 2,097,152 | 0.004 GB | 0.18% |
| **top-8 专家（8/64）** | **805,306,368** | **1.611 GB** | **68.3%** |
| lm_head（untied） | 103,022,592 | 0.206 GB | 8.7% |
| **合计 [W]** | **1,178,861,568** | **2.358 GB (2.196 GiB)** | 100% |

KV 每-token 读（MHA：`2 × 16 层 × 2048 × ctx × 2 B = 131,072 × ctx` 字节）：

| ctx | KV 读 | 总字节 [T] = [W]+KV | KV 占 [T] |
|---|---|---|---|
| 1K | 0.134 GB | 2.492 GB | 5.4% |
| 4K（模型上限） | 0.537 GB | 2.895 GB | 18.5% |

口径定义见 [METRICS.md](METRICS.md) §3.2。

**三条直接结论**（都影响实验设计，不是描述性细节）：

1. **专家权重是唯一值得战斗的地方（68.3%）**，attention 次之（22.8%）。router 只占 0.18%——
   这就是 `PLAN.md` 立论里"路由开销不随量化下降，所以构成 MoE 量化收益上限"的量化依据。
2. **lm_head 占 8.7%，而且它是 untied 的**。量化或保留 fp16 是一个**必须显式记录**的决策，
   不能再当成脚注（见 §4.4）。
3. **KV INT8 在主配置下不是主杠杆**：batch=1、ctx ≤ 4096 时 KV 池只有 **0.5 GiB**（占 13.84 GB
   权重的 3.6%），它省的主要是**带宽**（@4K 占总流量 18.5%，INT8 后约省 9%）而不是显存。
   优先级已在 `PLAN.md` §3 下调。

### 3.3 ⚠️ `max_position_embeddings = 4096`

`PLAN.md` §7 的 E6 原计划把 context 扫到 **8K，超出该模型的训练上下文**。
处理方式（二选一，必须显式声明）：

- 把 OLMoE 的扫描上限压到 **4K**（保守，推荐先做这个）；
- 或换到长上下文模型上做长 ctx 实验，并标注模型不同、不可与 OLMoE 混比。

### 3.4 Qwen3-30B-A3B（对照行）

| 模型 | 层 | hidden | experts | top-k | expert FFN dim | 总参 | 激活/token | fp16 权重 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 2048 | 128 | 8 | 768 | 30.52B | 3.34B | 61.0 GB |

已核实（HF 模型卡 + config 推导，见 MegaKernel-vLLM docs/16 §1.2）。

---

## 4. 精度与格式注意（V100 特有）

1. **一律 fp16**：`torch_dtype=torch.float16`。**不要用 bf16**——sm_70 无 bf16 张量核，
   会退化为 fp32 模拟，速度与显存都劣化。
2. **checkpoint 可能以 bf16 存储**：加载时转 fp16，并记录 PPL/KL 落差（这本身是一个数据点：
   "bf16 权重在 fp16 下推理"的精度代价）。
3. **量化 checkpoint 的取舍**：
   - 使用预量化 checkpoint（如 GPTQ/AWQ 产物）→ 快，但**量化方法不是你做的**，
     只能作为 L1 基线；
   - 自己做量化 → 慢，但方法贡献归你。**本项目的主贡献是自己实现**，
     预量化 checkpoint 只用于对照。
4. **`lm_head` 与 embedding**：注意是否 tied（影响每 token 字节数）；量化时通常保留 fp16，
   **必须显式记录**，否则显存账会对不上。
