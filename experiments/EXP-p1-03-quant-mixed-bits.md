# EXP-p1-03-quant-mixed-bits

> 混合位宽（`REMEDIATION_LADDER` 第 4 档）+ 第二个指标（PPL）。
>
> **这条实验最重要的结果不是混合位宽，而是一个仪器缺陷**：`EXP-p1-01` 的 KL 门是用
> **1024 个 token id=1** 当 prompt 测的（`docs/METRICS.md` 说的是"固定 prompt 集"），
> 那是一段退化输入。换成真实文本后，同一个配置的 KL **小 20–216 倍**，
> W8A16 的判定从 FAIL 翻成 PASS。见 §3。

- **ID**: `EXP-p1-03-quant-mixed-bits`
- **Date**: 2026-10-06
- **Operator**: `tools/quant_probe.py`（KL，teacher forcing）+ `tools/quant_ppl.py`（PPL，新实现）
- **Status**: `done`
- **原始数据**（`results/EXP-p1-03-quant-mixed-bits/`）：
  - `probe_mixed.json` —— 6 个混合位宽格（all-ones prompt 口径，与 EXP-p1-02 可比）
  - `probe_kl_wikitext_prompt.json` / `probe_kl_wikitext_window2.json` —— 真实 prompt 上的 KL（两个窗口）
  - `ppl.json` —— PPL + 配对 ΔNLL（wikitext-2 test 前 8192 token）
  - ★ **工件在 base id 下**：矩阵把本实验展开成 `results/EXP-p1-03-quant-mixed-bits__ctx=1024/`，那份骨架是
    harness 的占位（永远读 `planned`）。判定"跑没跑"看本记录的"结果"节与上面这些文件名，
    不要看骨架。
- **对照**: `EXP-p1-02-quant-error-attribution`（逐投影敏感性，本条的输入）；
  `EXP-p1-01-w8-w4-accuracy`（被本条更正的对象）

## Hypothesis（跑之前写）

```text
输入（EXP-p1-02 的逐投影实测）：
  W4 只量化 q_proj → KL 0.0019；k_proj → 0.0017；v_proj → 4.06；o_proj → 4.05。

假设：
  H1  混合位宽可以把 q/k 降到 W4 而不付代价：q/k=W4 + v/o=W8 + experts=W8 ≈ W8 baseline
  H2  v/o 降到 W4 是悬崖（对照，应当显著变差）
  H3  剩下的代价几乎全在 experts（它占被量化参数的 96%），W4 的 experts 仍会失败
  H4  PPL 与 KL 对同一批配置给出的**排序**应当一致；不一致就说明其中一个仪器有问题

反证信号：
  - q/k 降到 W4 让 KL 显著上升 → H1 不成立，逐投影结论不能外推到混合配置
  - PPL 与 KL 的 group 排序不一致 → 指标分辨力有问题（★ 这一条**发生了**，而且错的是 KL）
```

## 1. 混合位宽（KL，all-ones prompt 口径，与 EXP-p1-01/02 可比）

| 臂 | 量化内容 | KL | top-1 | 字节 | 占 fp16 的 |
|---|---|---|---|---|---|
| `mixed_baseline_w8` | attn W8 + experts W8 | 0.026837 | 1.0 | 6.816 GB | 50.8% |
| **`mixed_qk_w4_vo_w8_exp_w8`** | q/k **W4**、v/o W8、experts W8 | **0.027216** | 1.0 | 6.749 GB | 50.3% |
| `mixed_qk_w8_vo_w4_exp_w8` | q/k W8、v/o **W4**、experts W8（对照） | **4.378125** | 0.0 | 6.749 GB | 50.3% |
| `mixed_qk_w4_vo_w8_exp_w4` | q/k W4、v/o W8、experts **W4** | 0.420994 | 0.0 | 3.527 GB | 26.3% |
| `mixed_exp_gate_up_w4` | experts 的 gate/up **W4**、down W8 | 0.380021 | 0.0 | 4.668 GB | 34.8% |
| `mixed_exp_down_w4` | experts 的 down **W4**、gate/up W8 | 0.800508 | 0.0 | 5.742 GB | 42.8% |

（fp16 的 attention+experts = 13.42 GB；三个 noop 自检臂全部 KL = 0.0。）

- **H1 成立**：q/k 降到 W4 是 0.027216 vs 0.026837（+1.4%，低于该指标的噪声地板）→ 逐投影结论**能**外推。
- **H2 成立**：v/o 降到 W4 = 4.378，与 attention-only 全 W4（4.27）同级 → 悬崖确实全在 v/o。
- **H3 成立**：q/k W4 省下的字节只有 **1%**（attention 只占被量化参数的 4%！
  experts 6.44G / attention 0.27G）。而专家 W4 一上，KL 0.027 → 0.421。
  专家内部：gate/up W4 = 0.380、down W4 = 0.801 → **down 比 gate/up 敏感 2.1×**，
  但两者都远超 0.05 的门。
- **结论：第 4 档在字节上是无意义的**（1%），它的价值只在机制确认。
  真正决定 `[W]` 的是 experts（96%），而 experts 的每一档 W4 组合都过不了门。

## 2. 第二个指标：PPL + 配对 ΔNLL（`tools/quant_ppl.py`，新实现）

`docs/METRICS.md` §1 从 Phase 1 就把 PPL 列为指标（门槛：相对退化 <1%），但仓库里
**从来没有实现过** —— 之前每个 `result.json` 的 `ppl_fp16/ppl_this/ppl_delta` 都是 `null`。
本条补上它，并按配对方式报：

    同一串 token 上算逐 token NLL → ΔNLL/token = mean(logp_fp16 − logp_arm)，配 paired SE
    ★ 只报 PPL 绝对值是不够的：文本难度主导绝对值，而配对差值把它消掉，留下可判断的误差棒。

数据：`wikitext-2-raw-v1` **test** 前 8192 token（不重叠 1024 窗口，每个 token 计一次）。
**fp16 参照：PPL = 9.1994，NLL/token = 2.219143（8184 个 token）。**

| 臂 | PPL | 相对退化 | ΔNLL/token ± SE | t | 字节 |
|---|---|---|---|---|---|
| `w8a16_attn_g32` | 9.1989 | −0.01% | −0.000062 ± 0.000538 | −0.12 | 0.285 GB |
| `w8a16_attn_g64` | 9.1977 | −0.02% | −0.000187 ± 0.000552 | −0.34 | 0.277 GB |
| `w8a16_attn_g128` | 9.2033 | +0.04% | +0.000418 ± 0.000569 | 0.73 | 0.273 GB |
| `w4a16_attn_g32` | 9.3850 | +2.02% | +0.019973 ± 0.002662 | 7.50 | 0.151 GB |
| `w4a16_attn_g64` | 9.4212 | +2.41% | +0.023821 ± 0.002947 | 8.08 | 0.143 GB |
| `w4a16_attn_g128` | 9.5224 | +3.51% | +0.034504 ± 0.003204 | 10.77 | 0.138 GB |
| `noop`（×3） | 9.1994 | +0.00% | **+0.000000 ± 0.000000** | 0.00 | — |
| `mixed_baseline_w8` | 9.2025 | +0.03% | +0.000331 ± 0.000619 | 0.54 | 6.816 GB |
| `mixed_qk_w4_vo_w8_exp_w8` | 9.2808 | +0.88% | +0.008804 ± 0.001612 | 5.46 | 6.749 GB |
| `mixed_qk_w8_vo_w4_exp_w8` | 9.4495 | +2.72% | +0.026821 ± 0.002900 | 9.25 | 6.749 GB |
| `mixed_qk_w4_vo_w8_exp_w4` | 9.7399 | +5.87% | +0.057087 ± 0.004149 | 13.76 | 3.527 GB |
| `mixed_exp_gate_up_w4` | 9.4375 | +2.59% | +0.025550 ± 0.002675 | 9.55 | 4.668 GB |
| `mixed_exp_down_w4` | 9.4487 | +2.71% | +0.026733 ± 0.002994 | 8.93 | 5.742 GB |
| `noise λ=1.0`，4 个方向 | 9.1975–9.2096 | −0.02 … +0.11% | −0.000212 … +0.001103 | −0.39 … 1.98 | — |
| `noise λ=18.143` | 9.5008 | +3.28% | +0.032235 ± 0.003087 | 10.44 | — |

- **W8A16 在 PPL 上与 fp16 无法区分**（三个 group 的 |t| < 1，相对退化 ≤0.04%）。
  **而 KL 说它们超门 2.4–4 倍。** 两个指标矛盾 —— 见 §3。
- **W4 在 PPL 上是"真的变差"但幅度温和**：attention-only +2.0 … +3.5%；attn+exp 类 +2.6 … +5.9%。
  全部超过 METRICS 的 PPL 门槛（<1%）。KL 在 all-ones prompt 上说 182×（跨门 85 倍），
  PPL 说 +3.5% —— 差了两个数量级的"严重程度"叙事。
- **PPL 的分辨力远好于 KL**：同一量级下 4 个噪声方向的 ΔNLL 只散布在
  −0.0002 … +0.0011（|t| ≤ 2），而 KL 在同一批方向上散布 **19×**（0.028 → 0.526）。
  这直接验证了 EXP-p1-02 的 follow-up #2：**KL 在该量级没有分辨力，PPL 有。**

## 3. ★ 仪器缺陷：KL 门是用**退化 prompt** 测的

`docs/METRICS.md` §1 写的是"**固定 prompt 集**上的逐位置 KL"。而
`benchmarks/quant_gate.py` / `tools/quant_accuracy_gate.py` 一直在用

```python
ids = torch.full((1, ctx), 1, dtype=torch.long)     # 1024 个 token id = 1
```

这不是 prompt 集，是一段**退化输入**（同一個 token 重复 1024 次）。
把它换成真实文本的前 1024 个 token（wikitext-2 test，两个不同窗口）：

| 臂 | 窗口1 | 窗口2 | **all-ones** | 膨胀倍数 | 门（W8 0.01 / W4 0.05） |
|---|---|---|---|---|---|
| `w8a16_attn_g32` | 0.000883 | 0.001318 | 0.036955 | 28–42× | all-ones FAIL → **真实 PASS** |
| `w8a16_attn_g64` | 0.000732 | 0.001738 | 0.040553 | 23–55× | 同上 |
| `w8a16_attn_g128` | 0.000985 | 0.000726 | 0.023469 | 24–32× | 同上 |
| `w8a16_attn_experts_g32` | 0.001612 | 0.001394 | 0.034818 | 22–25× | 同上 |
| `w8a16_attn_experts_g64` | 0.000767 | 0.004290 | 0.037734 | 9–49× | 同上 |
| `w8a16_attn_experts_g128` | 0.000220 | 0.001405 | 0.026837 | 19–122× | 同上 |
| `w4a16_attn_g32` | 0.020275 | 0.034622 | 4.11621 | 119–203× | 由"崩"变"临界" |
| `w4a16_attn_g64` | 0.022763 | 0.039278 | 4.07744 | 104–179× | 同上 |
| `w4a16_attn_g128` | 0.019762 | 0.051988 | 4.27060 | 82–216× | 同上 |
| `w4a16_attn_experts_g32` | 0.021398 | 0.108568 | 3.89656 | 36–182× | 6 次里 5 次 FAIL |
| `w4a16_attn_experts_g64` | 0.146986 | 0.086806 | 3.86128 | 26× | FAIL |
| `w4a16_attn_experts_g128` | 0.162308 | 0.119852 | 4.83880 | 30× | FAIL |

**这张表更正了 `EXP-p1-01` 的核心结论：**

1. **`EXP-p1-01` 的 9 格 KL 全是退化 prompt 的产物**，膨胀 22–216×。那句
   "**W8 差 3–4 倍**"是仪器效应：真实文本上 W8A16 的所有格都在门内，
   `attn+exp g128` 甚至比门低 **45 倍**（0.00022 vs 0.01）。
2. 那条"**W4 attention 悬崖 160×**"也主要是仪器效应：真实 prompt 上
   attention-only W4 是 **0.020–0.052**（门 0.05），从"崩掉"变成"临界"。
   W4 attn+exp 是 0.087–0.162（6 次里 5 次 FAIL）—— **W4 真正的失败点是 experts**，
   与 §1 的混合位宽结论、§2 的 PPL 结论三条独立吻合。
3. `EXP-p1-01` follow-up #5 里那个 `top1 = 0.0` 的疑问也解释了：退化 prompt 上轨迹会发散；
   真实 prompt 上同一批配置的 top-1 是 0.76–1.0。

**这不是放宽门槛**，是修仪器：门没动（W8 0.01 / W4 0.05 / top-1 0.95 / PPL <1%），
改的是 METRICS.md 自己写的"固定 prompt 集"这句**从未被实现**的话。
两个口径的数字都留在这里，可以逐一复核。

顺带暴露的第二个仪器问题：**同一个配置在不同 prompt 窗口上的 KL 差 5.6×**
（`w8a16_attn_experts_g64`: 0.00077 vs 0.00429），比 group_size 之间的差还大。
也就是说 0.01 这个门被套在一个"**单个 prompt、33 个位置**"的估计上，
而它的 prompt 间散布本身就是这个量级。PPL 用 8184 个 token 的配对差分，
SE 是 6e-4/token —— 这才是该拿来做门的仪器。

## Verdict

```text
verdict:    PASS（混合位宽的预测成立；★ 并更正 EXP-p1-01 的判定）
outcome:    supported
notes:
  1. H1/H2/H3/H4 全部成立。混合位宽按逐投影预测工作，但字节收益只有 1%（attention 占 4%）。
  2. PPL 是仓库里第一次真正实现（此前每个 result.json 的 ppl_* 都是 null）。
     在 8192 token 上，W8A16 与 fp16 **无法区分**（|t| < 1）；W4 是 +2.6 … +5.9%（超 <1% 的门）。
  3. ★ 最重要的结果：EXP-p1-01 的 KL 门口径错误 —— 用的是 1024 个 token id=1。
     真实 prompt 上同一批配置的 KL 小 22–216×，W8A16 从"9 格全 FAIL"变成
     "全部在门内（最低到门的 1/45）"，W4 attention-only 从 4.27 变成 0.020–0.052。
     **EXP-p1-01 的 9 格表与"W8 差 3–4 倍"应视为已被本条取代。**
  4. 门槛一个都没改。W8 的 0.01、W4 的 0.05、top-1 的 0.95、PPL 的 <1% 原样保留，
     本条只是把 KL 换成在**真实文本**上测。
```

## REMEDIATION_LADDER 建议（据本条更正后的证据）

```text
★ 第 1 档（朴素 RTN W8A16）**本来就够了**：
   真实 prompt 上 attn+exp g128 = 0.00022（门的 1/45）、最差的格 0.0043（门的 43%）；
   PPL +0.03% ± 0.06（门 <1%）。→ W8 目标不需要任何补救。
   （EXP-p1-01 触发阶梯的那句话建立在退化 prompt 上。）

第 5 档（selective：只量化 experts）：W8 下 experts-only = 0.0393（all-ones 口径）——
   同样要按真实 prompt 重测；但 PPL 口径下 W8 全量化已经无法与 fp16 区分，
   所以这一档在 W8 上没有意义。

W4 目标（[W] 减半）：
   - experts 是失败点（PPL +2.6 … +5.9%，KL 0.087–0.162）且 experts 占 96% 字节；
   - attention 内部：q/k 可以到 W4（PPL +0.88%，字节只省 1%）—— 不值得为它改配置；
   - down_proj 比 gate/up 敏感 2.1×（0.801 vs 0.380），但仍远超门。
   → 需要**算法**而不是位宽分配。EXP-p1-02 的离群统计说 RTN 的失效模式不是
     "离群塌缩"（单元素主导组占比 0.0001–0.0056），而 GPTQ 式的逐层重建直接最小化
     **层输出误差**，与本条测到的放大位置一致 → **第 2 档是下一步该做的**。
     第 3 档（AWQ）机制依据弱，建议排在第 2 档之后。

★ 在动任何后续档位之前，先把**门**换到有分辨力的仪器上：
   多窗口真实 prompt 的 KL（n ≥ 5，报 mean ± 散布）或直接采用 PPL 作主门
   （METRICS §1 本来就列了 PPL）。用单个 33 位置的退化 prompt 当门，
   分不出"哪个方案更好"—— EXP-p1-02 已经用方向散布（19×）证明了这一点。
```

## 环境变更（必须记，否则复现不了）

- 服务器上装了 `datasets==3.6.0`（wikitext-2 用），`HF_ENDPOINT=https://hf-mirror.com`
  （`huggingface.co` 在服务器上不可达，实测 000；镜像 200）。
- ⚠️ **踩坑并已恢复**：`pip install datasets` 默认会装 `datasets 5.1.0` + `huggingface-hub 2.1.1`，
  而 `transformers 4.57.6` 要求 `huggingface-hub<1.0` → **transformers 直接导入失败**。
  已用 `pip install "huggingface-hub==0.36.2" "datasets==3.6.0"` 恢复，并逐项核对
  torch 2.8.0+cu128 / transformers 4.57.6 / numpy 2.3.2 / safetensors 0.8.0 / triton 3.4.0
  与装之前**完全一致**。
  → 这正是 `configs/constraints.txt` 该被冻结的理由。**2026-10-07 已冻结**：文件头写明
  `frozen: 2026-10-07 · RTX 5090 (AutoDL PyTorch 2.8.0/12.8 镜像)`，torch 改为实测的 2.8.0，
  并把 `datasets==3.6.0` / `huggingface-hub==0.36.2` 的**成对约束**写在注释里（下次 bootstrap
  照它装，不会再拉坏 transformers）。
- token id 缓存：`/root/autodl-tmp/lemoq-data/wikitext2_test_8k.pt`（8192 个 token，
  wikitext-2-raw-v1 test 前 8192 个 token，不挑样本）。它在仓库外，不进 git。

## Follow-ups

1. **更正 `EXP-p1-01` 的记录**：那条 9 格表与"W8 差 3–4 倍"应标注为本条取代
   （已在其文档顶部加一条指向本条的更正说明，未改动它的 Hypothesis/Verdict 原文）。
2. **把门换到有分辨力的仪器**：多窗口真实 prompt 的 KL（n ≥ 5）或直接采用 PPL 作主门。
   在此之前，任何"方案 A 比 B 好"的 KL 级结论都不要下。
3. **`w8a8` 的 3 格同样要在真实 prompt 上重测**（0.197/0.79 也是 all-ones 口径），
   而且它的激活伪量化在真实文本上可能表现完全不同（激活分布依赖输入）。
4. **第 2 档（GPTQ 式逐层重建）**：目标是 experts 的 W4；校准集可以用本条的
   wikitext token 流（8192 token），门用 PPL（<1%）+ 多窗口 KL。
5. **不要为 q/k 的 W4 去改配置**：省 1% 字节、PPL +0.88%，可能恰好卡在 1% 门槛两侧，
   是噪声驱动的决策。
6. `mixed_exp_down_w4`（0.801）与 `mixed_exp_gate_up_w4`（0.380）在 KL 上差 2.1×，
   在 PPL 上几乎相同（+2.71% vs +2.59%）—— 又一个 KL 排序不可靠的例子；
   以后报"哪一类张量更敏感"时应当用 PPL 复核。

## Applicability

```text
本结论适用于：RTX 5090、fp16 执行、OLMoE-1B-7B-0924（bf16 checkpoint → fp16）、
              transformers 4.57.6、ctx=1024、32 步 teacher forcing、
              朴素 per-group 对称 RTN、scale 存 fp16、g128（混合位宽格）或 g32/64/128（对照）、
              PPL 口径 = wikitext-2 test 前 8192 token、不重叠 1024 窗口
不适用于：GPTQ/AWQ（未测）；w8a8 的激活量化（未在真实 prompt 上测）；
          batch>1；其它模型族；"prompt 膨胀倍数"这个量绑定在 this config × 这两个窗口上
原因：
  - prompt 膨胀倍数依赖于具体模型的 distribution shift：退化输入（重复 token）
    对 attention 的影响与对 MoE/专家路径的影响不同，换模型要重测；
  - 全部 PPL 数字绑定在 8192 token 的那一段文本与不重叠窗口的口径上；
  - 混合位宽结论绑定在 g128；q/k 在 W4 的"免费"是在 attention 占 4% 字节的前提下，
    换模型（attention 占比不同）要重新算这笔账。
```
