# EXP-p4c-02-elementwise-tail

> `EXP-p4c-01` §5.4 的后续（用户 2026-10-07 批准的新 id）：量化 graph 步里那个
> **~0.88 ms / ~330 个 kernel 的 elementwise/copy 长尾到底是什么**。
> ★ 这是**归因**实验（`kind: analysis`）：产物是完整 kernel 表 + 逐作用域的模块归因，
> 不是新的性能格 —— 所以**没有轴**，也不重跑任何已落盘格子。

- **ID**: `EXP-p4c-02-elementwise-tail`
- **Date**: 2026-10-07
- **Operator**: `tools/profile_graph_scopes.py`（新工具，GPU）
- **Status**: `done`（kernel 级归因成立；**模块级归因按自己的预注册判据作废**，见 §4）
- **原始数据**: `results/EXP-p4c-02-elementwise-tail/scopes_w8a16_kernel_ctx1024.json`
- **对照**: `EXP-p4c-01` §5.4（同一个配置的 runner payload）、`EXP-p2-02`（MoE 换掉 gather 的那一步）

## 1. Hypothesis（跑之前写 —— 随矩阵在 `75374ee` 提交）

```text
假设：长尾来自可以命名的模块级算子链：(a) RMSNorm 的 mean/reduce 与 cast，
      (b) 残差相加与 fp16/fp32 往返，(c) KV cache 写入 / RoPE，(d) MoE 的 top-k 索引。
      若成立 ⇒ 修法是"把算子链融进已有 kernel 或消除中间张量"。

反证信号：归因表里没有任何一个作用域 >10%（时间落不到具体模块）⇒ 无法据此下手，该方向作废；
          或某个**单一 kernel** >40% ⇒ 它不是"长尾"而是单点，假设表述被否证。
```

## 2. Configuration

```text
配置（= EXP-p4c-01 v2 里最好那一格）：w8a16 / g128 / dequant=fused / moe=grouped /
    attn=kernel / cudagraph / ctx=1024 / batch=1 / OLMoE-1B-7B-0924 / RTX 5090 32GB
口径：eager + record_function 作用域（按模块名聚合 16 层）+ graph 重放（完整 kernel 表）
```

## 3. 仪器自证（这一步花了三次上机，全部记在这里）

工具 `tools/profile_graph_scopes.py` 的三处**必须先自证**，否则量到的是别的东西：

1. **专家没量化**：`replace_moe_blocks(model, impl="grouped")` 不传 `quant` 就保持 fp16 ⇒
   图总时间 2.770 ms vs 单元格 2.348 ms，差 0.42 ms 正好是 `_gate_up_act_kernel`（fp16）与
   `_gate_up_act_kernel_q` 的差。修：传 `quant_spec_for_cfg(cfg)`，并加**响亮 self-check**。
2. **self-check 自己写错**：`_gate_up_act_kernel_q` 也以 `_gate_up_act_kernel` 开头 ⇒
   前缀匹配误报。改成精确名比较。**这条 self-check 在给出任何结论之前先拦住了"量到另一个配置"。**
3. **作用域恒为 0**：`profile(activities=[CUDA])` 时 kineto 没有 CPU 侧区间事件可关联 ⇒
   `record_function` 的 device time 全是 0。用 20 行最小复现确认：开 **CPU+CUDA** 后同一个
   作用域能拿到 `self_device_time_total`（1310 µs / 5 次）。

**修好之后的核对**：graph 侧 GPU **2.350 ms/步** vs 单元格 `busy` **2.348 ms** ✓，
且逐 kernel 对得上（`_gate_up_act_kernel_q` 0.4439 / `_down_sum_kernel_q` 0.2587 /
`_dequant_gemm_kernel` 0.3244 / `_attn_split_kernel` 0.1211 —— 与 `EXP-p4c-01` §5.6 的
0.4446/0.2589/0.3254/0.1210 一致）⇒ **graph 侧的表可信**。

## 4. K1 门：模块级归因**作废**（预注册判据自己没过）

工具预注册的 K1 是"eager 与 graph 两条路径的前 8 kernel 必须高度重合，否则用 eager 归因解释图
不成立"。实测：

```text
前 8 kernel 重合 5/8
eager GPU 3.607 ms/步  vs  graph 2.350 ms/步   （+53%）
```

⇒ **按预设判据，本记录的"逐作用域模块归因"不成立，不引用**（eager 路径在这里的算子组成与图不同，
多出的部分与 host 交错有关）。工具本身已经用自检把这件事报出来，而不是给出一份好看的假归因。

## 5. 结果：kernel 级归因（**这一半可信**）

`w8a16 × fused × grouped × kernel × cudagraph × 1024`，完整表按 self time 前 18：

| kernel | ms/步 | 占比 | ×/步 | 来源（读出来的，不是猜的） |
|---|---|---|---|---|
| `_gate_up_act_kernel_q` | 0.4439 | 18.9% | 16 | MoE 专家 GEMV（本仓库） |
| `_dequant_gemm_kernel` | 0.3244 | 13.8% | 64 | 投影（本仓库，§5.6 已改） |
| `_down_sum_kernel_q` | 0.2587 | 11.0% | 16 | MoE 专家 GEMV（本仓库） |
| `gemvx`（lm_head） | 0.1269 | 5.4% | 1 | 词表投影 |
| `_attn_split_kernel` | 0.1211 | 5.2% | 16 | attention core（本仓库） |
| **`direct_copy`** | **0.1105** | 4.7% | **66** | fp16→fp16 拷贝（≈4/层） |
| **`reduce_kernel<MeanOps>`** | **0.0969** | 4.1% | **65** | **RMSNorm 的 mean(x²)** |
| `elementwise_kernel<128,4,nocast>` | 0.0905 | 3.9% | 64 | 通用 elementwise（≈4/层） |
| `elementwise<BinaryFunctor<Half>>` | 0.0657 | 2.8% | 65 | 二元 op（乘/加） |
| `elementwise_kernel<128,2>` | 0.0631 | 2.7% | 65 | 通用 elementwise |
| `float16_copy_kernel` | 0.0622 | 2.6% | 83 | fp16 拷贝/cast（≈5/层） |
| **`index_copy_kernel`** | **0.0553** | 2.4% | **32** | **KV cache 写入**（2/层） |
| **`CatArrayBatchedCopy`** | **0.0548** | 2.3% | **32** | `cat`（2/层；KV 或 RoPE 的 rotate_half） |
| `rsqrt_kernel` | 0.0528 | 2.2% | 65 | **RMSNorm 的 rsqrt** |
| `gatherTopK` | 0.0509 | 2.2% | 16 | **MoE 路由 top-k** |
| `CUDAFunctor_add` | 0.0502 | 2.1% | 64 | 残差相加 |

**读出来的结构**（这是本记录真正的产出）：

- **"~330 个 kernel"不是一片平的噪声，而是 ~13 个具名家族 × 16–83 次/步**
  （次数 ≈ 每层 1 / 2 / 4 / 5 次 —— 正好对应"每层有几个 norm / 几个投影 / 几次 cache 写"）；
- 没有任何**单个** kernel >20%（最大 18.9%，而它是我们自己的 MoE kernel）⇒
  反证信号里"单一 kernel >40% ⇒ 不是长尾而是单点"**未被触发**；
- 占大头的非自研部分是：**RMSNorm 的 mean+rsqrt+elementwise（≈0.15–0.25 ms）**、
  **拷贝族 `direct_copy`+`float16_copy`（≈0.17 ms）**、
  **KV cache 的 `index_copy_`+`cat`（≈0.11 ms）**、残差 `add`（≈0.05）、路由 `topK`（≈0.05）。

## 6. Verdict

```text
verdict:    PASS
outcome:    supported（**kernel 级**）；模块级归因 inconclusive（K1 未过，已作废，见 §4）
notes:      长尾确实由**可命名的模块级算子链**构成（假设成立），但它不是一个可以"一刀切"的东西：
            13 个家族、每个 0.05–0.11 ms、次数与"每层几个算子"一一对应。
            没有单点 >40% ⇒ 不存在"改一处就赚 0.4 ms"的机会；只能按家族逐个消。
            ★ 反证信号（"落不到具体模块就作废"）在**模块**这一级确实触发了 —— 我按预注册把它写成
            inconclusive，并且没有拿 eager 的作用域数字去解释图。
```

## 7. Follow-ups（按"数值风险 × 收益"排序）

1. **拷贝族 0.17 ms / 149 次**（§5 的 `direct_copy` 66 + `float16_copy` 83）：
   收益 7%、**不动数值**（若确认是无谓的 `.contiguous()`/同 dtype cast，删掉即可）。
   先要**精确指认**它在哪：模块级归因作废了，替代办法是给 `QuantLinear`/`StaticCache` 路径
   加计数器，或在 HF forward 的候选点上打 `record_function` 单独验证（小范围、可判读）。
2. **RMSNorm 链 0.15–0.25 ms**（mean + rsqrt + elementwise，65 次/步）：
   收益 6–11%，但**要改数值**（把 norm 折进投影 kernel 会改归约顺序）⇒ 必须**重测引用门**
   （`tools/quant_probe.py` 的 5 窗口 prompt 集）才能发表 —— 这是**用户决策点**，
   按 `AGENTS.md` §6 先问。
3. **KV cache 写 0.11 ms**（`index_copy_` + `cat`，32 次）：属于 HF 的 StaticCache 路径，
   要动得替换 cache 实现（与 `attn_decode` 的"长度从设备内存读"是同一类侵入）。
4. **路由 topK 0.05 ms**：与 `EXP-p4-01` 的"路由开销"是同一件事在 grouped 路径上的**可见残余**
   （hf 路径上是 4.5 ms 的 cub 机器）——可以用来给 P2c（扩分类器）提供交叉核对。
5. ⚠️ **不要把本记录当"elementwise 长尾很大所以去融合它"的许可**：真正的机会窗口是拷贝族
   （不改数值），其次是需要重测门的 RMSNorm 融合。

## 8. Applicability

```text
本结论适用于：RTX 5090 32GB (sm_120)、fp16、单卡、batch=1、ctx=1024、
              OLMoE-1B-7B-0924、w8a16/g128/dequant=fused/moe=grouped/attn=kernel/cudagraph、
              measured bw 1561.9 GB/s
不适用于：其它 ctx（KV/cat 家族随 ctx 线性增长，权重与 norm 家族不随）；
          其它 bit-width（`_dequant_gemm_kernel` 的占比会变；w4a16 引用门 FAIL）；
          eager 臂（本记录的**模块级**归因明确不适用于图，见 §4）；
          sm_70/V100（无 cp.async、自研 kernel 不可直接搬）
原因：家族与"每层几个算子"绑定（次数是结构量），倍率随模型层数/投影数/ctx 变化。
```
