# EXP-p1-01-w8-w4-accuracy

> Phase 1：量化功能链路，**只谈精度，不谈性能**。这是本项目立论的第一次检验。

- **ID**: `EXP-p1-01-w8-w4-accuracy`（`ctx=1024`，`max_new_tokens=32`）
- **Date**: 2026-10-06
- **Operator**: 自动工具（`tools/quant_accuracy_gate.py`），AutoDL RTX 5090
- **Status**: `done`
- **原始数据**:
  - `results/EXP-p1-01-w8-w4-accuracy/summary.json`（9 格的汇总；逐格另有 `results/EXP-p1-01-...__bit_width=*_group_size=*/result.json`）
  - `results/p1-01b-router-ablation/`（router 消融）
  - `results/p1-01c-target-split/`（attention / experts 拆分）

## Hypothesis（跑之前写）

```text
假设（configs/experiments.yaml 原话）：位宽↓ → KL↑；group↓ → KL↓。

门槛（docs/METRICS.md §1，**建议值**）：W8 KL < 0.01；W4 KL < 0.05；top-1 > 0.95。

反证信号：W4 无法通过 KL 门槛 → 触发 REMEDIATION_LADDER。
```

## 摘要：**9 格全部 FAIL**

| targets | scheme | group 32 | group 64 | group 128 |
|---|---|---|---|---|
| attention, experts | `w8a8` | 0.197 / 0.79 | — | — |
| attention, experts | `w8a16` | **0.0348** / 1.0 | **0.0377** / 1.0 | **0.0268** / 1.0 |
| attention, experts | `w4a16` | **3.897** / 0.0 | **3.861** / 0.0 | **4.839** / 0.0 |

（KL / top-1。`—` 表示那一格的值只在 `summary.json` 里，本文未转录 —— 不要从这里抄。
另外一次 `attention,experts,router` 的运行保存在 `p1-01b-router-ablation/`，见消融。）

**结论一句话：朴素 per-group RTN 在这一模型上过不了任何一档门槛 ——
W8 差 3–4 倍，W4 直接崩（KL ≈ 4，top-1 = 0）。**

## ★ 但有两个信号说明"RTN 不行"这个结论**还不能下**

### 信号 1：group_size 不单调

假设说 `group↓ → KL↓`。实测 W8A16：g32 (0.0348) < g128 (0.0268) < g64 (0.0377)。
**g64 比 g128 差** —— 违反单调性。纯量化误差应当单调（更细的分组 = 更小的误差）。
出现非单调，通常意味着有别的机制在主导，或者实现里有与 group_size 相关的缺陷。

### 信号 2：W4 的注意力悬崖太陡

目标拆分（W4A16, group=128，一次一个变量）：

| targets | KL | top-1 |
|---|---|---|
| **experts only** | **0.408** | 0.0 |
| **attention only** | **4.271** | 0.0 |
| attention + experts | 4.839 | 0.0 |
| attention + experts（W8A16 对照） | 0.0268 | 1.0 |

W8 → W4 是 2× 的位宽变化，而 attention 上的 KL 从 0.027 跳到 4.27 —— **160×**。
典型 RTN 的 W4/W8 比是 2–4×，不是 160×。**这个陡度更像"有放大机制"或"有缺陷"，
而不像单纯的舍入误差。**

**一个可解释的放大机制（假说，未验证）**：attention 的 q/k 投影之后有
`q_norm`/`k_norm`（RMSNorm），它只归一化**幅度**、不修正**方向**；而 softmax 在
1056 个位置上把方向误差放大成注意力权重的重排，再经 16 层累积。
专家 MLP 没有这个通路，所以 experts-only 的 KL（0.408）低一个数量级。
**这条通路是真实的，但它能否定量解释 160× 尚未验证。**

### 因此本条的判定是 `INCONCLUSIVE`，不是"W4 不行"

在把这两条非单调性解释清楚之前，**不应该**调用 `REMEDIATION_LADDER`
（那等于把"实现可能有问题"当成"RTN 精度不够"来处理，会让后续每一档的结论都不可信）。

## 已经**确证**的部分（这些可以引用）

1. **量化真的被装上了。** 每个 cell 的报告都一致：
   `replaced = {attention: 64, router: 16, experts: 3072, ...}`（16 层 × 4 / 1 / 64×3），
   `skipped = 1`（`lm_head`，符合 `keep_lm_head_fp16`）。
2. **`[W]` 的字节账第一次被真实实现核对。** 实数：
   `w4a16` = 3.46–3.78 GB、`w8a16` = 6.82–7.13 GB，对照 fp16 的 13.85 GB
   → **25% / 49%**。且 **padding 膨胀 = 0.0%**（OLMoE 的 2048/1024/50304 对
   32/64/128 都整除），印证了 `bytes._quantized_bytes` 的假设。
3. **判据真的会失败。** 9 格全 FAIL，不是"从不失败"的判据。
4. **router 消融：router 不是主因。** 去掉 router 后 W8A16 为 0.027–0.038
   （含 router 是 0.020–0.043，基本不变），W4A16 仍是 3.86–4.84。**假说被否证。**
   —— 代价是 0.18% 的 `[W]`，收益为零，所以后续默认继续不带 router。

## Method（为什么值得信）

`docs/METRICS.md` 定义的是**逐位置** KL。而 free-running greedy 参照做不到：
一旦某一步 token 不同，后续位置比的就不是同一上下文。

本实验两条臂都在**同一串 token** 上用 **teacher forcing** 收集 logits（`benchmarks/quant_gate.py`），
所以唯一的变量就是量化本身。轨迹由 fp16 臂贪心生成一次，两条臂共用。

## Verdict

```text
verdict:    FAIL（全部 9 格未过 docs/METRICS.md §1 的门槛）
outcome:    INCONCLUSIVE（不建议就此调用 REMEDIATION_LADDER）
notes:
  1. 量化链路可用、字节账被核对、判据会失败 —— 这些已确证。
  2. W8A16 差 3–4 倍、W4A16 崩溃 —— 数值是真的。
  3. **但** group_size 非单调 + W4 注意力悬崖 160× 两个信号说明，
     现在还不能把"过不了门"归因给 RTN 本身。
  4. 精确探测顺序：先解释那两个信号，再谈补救阶梯。
```

## Follow-ups（按顺序）

1. **先解决信号 1（group_size 非单调）** —— 它最便宜、最能定位问题。
   建议：在**单层**上做受控实验（取 `model.layers.0` 的 attention，fp16 vs W8A16
   g32/g64/g128，逐张量比 KL / 相对误差），把"全局管线"这个变量去掉。
   如果单层上也非单调，那是实现问题，不是模型问题。
2. **再解决信号 2** —— 量一个 attention 投影的**逐组 absmax 分布**，
   看是否存在离群通道（RTN 的已知失效模式：一个离群值把整组压成 0）。
   若是，则这是**真实**的 RTN 失效，REMEDIATION_LADDER 的 AWQ 档正是为此设计 ——
   那时调用它是有依据的。
3. 修工具的落盘命名：per-cell 结果写在 `results/<cell>/`，而 `--json` 写进
   `results/EXP-p1-01-w8-w4-accuracy/summary.json`，两者层级不一致。
4. `w8a8` 的 group 64/128 两格值未转录；它们只在 `summary.json` 里。
5. 本文的 `top1 = 0.0` 不等于"一个都没对"—— 32 步里首 token 一致性为 0 是极端的，
   值得确认 `compare_logits` 在 KL 很大时 top-1 的统计口径（是否被 NaN/极值影响）。
   **这条是自检项，不是结论。**

## Applicability

```text
本结论适用于：RTX 5090、fp16 基线、OLMoE-1B-7B-0924（base）、
              transformers 4.57.6、ctx=1024、32 步 teacher forcing、
              朴素 per-group 对称 RTN、scale 存 fp16、targets 见上表
不适用于：AWQ/GPTQ 等补救档（未测）；其它 group_size（只测 32/64/128）；
          非对称量化（未测）；per-token 以外的激活量化口径（w8a8 用的是 per-token 伪量化）
原因：
  - 全部数字来自这一组 targets 与这一套口径；换 target 已实测会改变结论
    （experts-only W4 的 KL 0.408 vs attention-only 4.271）。
  - w8a8 的激活量化是**伪量化**：复现数值影响，不是 int8 GEMM。
```
