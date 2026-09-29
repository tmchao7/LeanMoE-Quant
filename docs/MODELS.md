# MODELS — 模型候选与选型理由

`configs/models.yaml` 是唯一来源；本文解释**为什么选它们**。

---

## 1. 分层

| 角色 | 模型 | 精度 | 用途 | 32GB V100 |
|---|---|---|---|---|
| dev | `Qwen/Qwen2.5-0.5B` | fp16 | kernel 迭代、单测、`--dry-run` | ✅ |
| dev | `Qwen/Qwen2.5-1.5B` | fp16 | dense 曲线主体 | ✅ |
| **primary (MoE)** | **`allenai/OLMoE-1B-7B`** | fp16 | **MoE 主线实验** | ✅ |
| dense 对照 | `Qwen/Qwen3-4B` | fp16 | 同规模 dense vs MoE 对比 | ✅ |
| dense 对照 | `Qwen/Qwen3-8B` | fp16 | 上界对照 | ✅（紧） |
| stretch | `Qwen/Qwen3-30B-A3B` | INT4 | scale 单点，**只在租的 48GB 卡上做** | ❌ |

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

## 3. 架构数字（引用来源已标注）

| 模型 | 层 | hidden | experts | top-k | expert FFN dim | 总参 | 激活/token | fp16 权重 |
|---|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 16 | 2048 | 64 | 8 | 1024 | ~6.9B | ~1.3B | ~13.8 GB |
| Qwen3-30B-A3B | 48 | 2048 | 128 | 8 | 768 | 30.52B | 3.34B | 61.0 GB |

Qwen3-30B-A3B 一行已核实（HF 模型卡 + config 推导，见 MegaKernel-vLLM docs/16 §1.2）。
**OLMoE 一列在上机时以 HF `config.json` 为准复核并落盘**（本文标为待复核）。

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
