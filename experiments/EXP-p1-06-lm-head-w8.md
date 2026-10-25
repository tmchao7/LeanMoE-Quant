# EXP-p1-06-lm-head-w8

> ★ F5 立项（2026-10-24 用户批准）：`lm_head` 也量化成 w8a16 g128 —— 量化叙事的最后一块。
> 预注册：假设与判据**先于**门实测写定（本文写入时 probe 仍在跑、任何结果未读）。

- **ID**: `EXP-p1-06-lm-head-w8`
- **Date**: 2026-10-24
- **Operator**: agent（GPU：租用 RTX 5090 32 GB，sm_120，实测读带宽 1560.3 GB/s）
- **Status**: `done`（★ **采纳完成 2026-10-25**：三链 PASS + A/B VALID + 翻默认（六 True）
  + 22 格刷新干净（金丝雀 −0.058%）+ 数字回填。v6 见 §结果与
  取数地图 ADOPTION_SYNC_F5_2026-10-25，已退役见 git 历史）
- **接线**: `LEMOQ_LMHEAD_W8` 开关（默认关，commit `79f7b88`）；独立门表
  `LMHEAD_GATE_CITATIONS`（接线时为空 ⇒ fail-closed，门实测通过才填）
- **上游规格**: `docs/FUSION_OPPORTUNITIES.md` F5（含 N7 口径修订）；立项批准记录在
  `AGENTS.md` §4 S3 行与 `docs/handoff/HANDOFF_2026-10-25.md` §0

## Hypothesis（跑之前写）

```text
假设：
  lm_head 是 untied 的 (50304, 2048) 密集 GEMV，每 token 必须整读一遍：fp16 = 206 MB
  ≈ 132 µs/步。W8 g128 后 103 MB ≈ 66 µs/步 —— 地板本身减半（FUSION_OPPORTUNITIES
  N7：这不是 S2 意义的"捕获缺口"，实测 0.1270 ms 已在墙上，收益来自把地板减半）。
  精度上 lm_head 直接决定 logits，对 KL/top-1 最敏感（EXP-p1-02 档 5 的保留理由），
  但 W8 g128 的 per-group RTN 颗粒足够细，预期仍近无损（类比 w8a16 全模型的
  KL 0.00136，lm_head 单独量化的增量应当更小 —— 它只占 [W] 的 8.7%）。

预期结果：
  1. 门：`w8a16_lmhead_only_g128` 两 ctx KL < 0.01 且 top-1 均值 ≥ 0.95（5 窗 + ones 对照）；
  2. 三链：PPL 全集 Δ < 1%；MMLU 掉点 < 2 点（F5 规格的采纳红线）；
  3. 延迟：A/B 每步省 ≈ 40–80 µs（ctx 无关；canonical w8a16×kernel @1024 约
     1.5586 → ≈1.50 ms，−3~5%）。

反证信号（出现即视为假设被否证）：
  - lmhead_only 臂任一 ctx KL ≥ 0.01 或 top-1 < 0.95 ⇒ F5 精度否证，负结果关闭
    （开关保持默认关，不发布任何延迟 —— 硬规则 8）；
  - A/B 的 Δ ≤ 0 或 < MAD ⇒ "地板减半没兑现"，同样不采纳（字节账仍然成立，
    但 decode 时间不受益）；
  - 组合臂（full）KL 显著偏离 lmhead_only 与既有 w8a16 引用的叠加 ⇒ 记录为
    "量化表面叠加非线性"，进记录但单独不成否证。
```

## Method

```text
门（精度第一链）: tools/quant_probe.py --arms f5_lmhead --prompt-windows 5
                  --prompt-source wikitext --include-ones-control --rmsnorm fused
                  --ctx {1024,4096} --prompt-cache wikitext2_test_64k.pt
                  → results/EXP-p1-06-lm-head-w8/probe_f5_lmhead{,_ctx4096}.json
  臂: w8a16_lmhead_only_g128（只量 lm_head —— 门行来源，一次一变量）
      w8a16_full_lmhead_g128（组合生产配置 —— 采纳证据，不进门表）
  阈值: 与 METRICS §1 逐字相同（W8 KL<0.01、top-1≥0.95；合取、按窗口均值）。
PPL（第二链）:    tools/quant_ppl.py --arms f5_lmhead（全集口径，配对 ΔNLL）
MMLU（第三链）:   tools/task_accuracy.py（分层子集 200 题；掉点 <2 点）
延迟:            scripts/rounds/run_ab_fresh_process.sh（独立进程相邻 A/B，
                  ENV_ON=LEMOQ_LMHEAD_W8=1 / ENV_OFF==0；金丝雀 ≤5%；
                  CELL = p4c-01 的 w8a16×kernel @1024 canonical 格）
采纳（全过后）:   LMHEAD_GATE_CITATIONS 填行（新测量抄入）→ A/B VALID → 翻默认
                  → run_f1_refresh.sh 23 格刷新 → 数字同步（ADOPTION_SYNC 体例）。
一次一变量:      门臂只量 lm_head（其余 fp16）；组合臂标注 combined。
```

## Model

```text
model:           allenai/OLMoE-1B-7B-0924（本地快照）
arch:            16 层、64 experts top-8、expert FFN 1024、MHA 16/16
dtype:           fp16（bf16 ckpt → fp16）
hidden:          2048
lm_head:         (50304, 2048) **untied**（tie_word_embeddings=false）= 103,022,592 参数
                 = 0.206 GB fp16，占 [W] 8.7%（MODELS.md §3.1）
total params:    6.92B
```

## 结果

### 第一链：门（KL/top-1）——**PASS，且近无损**（2026-10-24 GPU 实测）

`tools/quant_probe.py --arms f5_lmhead --prompt-windows 5 --prompt-source wikitext
--include-ones-control --rmsnorm fused`（缓存 `wikitext2_test_64k.pt`）。
工件：`results/EXP-p1-06-lm-head-w8/probe_f5_lmhead{,_ctx4096}.json`。

| 臂 | ctx=1024 KL（5 窗） | ctx=4096 KL（5 窗） | top-1（两 ctx） | verdict |
| --- | --- | --- | --- | --- |
| `w8a16_lmhead_only_g128`（门行来源） | **0.0000605** | **0.0000661** | **1.000000**（165/165 全对） | **PASS / PASS** |
| `w8a16_full_lmhead_g128`（组合证据臂） | 0.001405 | 0.000883 | 0.981818 | PASS / PASS |

- 门行已落表（`LMHEAD_GATE_CITATIONS`，commit `cd3c97e`；钉死测试读本地工件）。
- **验真**：每窗 `replaced={'lm_head': 1}`、`realized_bytes_by_target.lm_head = 104,632,320`
  （≈103 MB，正是 lm_head W8 的体量）——不是空跑（对照：同日发现的 quant_ppl 空跑臂
  `realized_bytes={}`、ΔNLL 与 fp16 逐位相同）。
- ones 对照 `ones_control_kl = 0.000105`；noop 臂 Δ 恰为 0。
- **组合臂叠加无非线性意外**：full 臂 KL 与既有 w8a16 引用（0.001361/0.000842）几乎
  重合 —— lm_head 的增量淹没在 attention+experts 的既有误差里。
- 假设的"预期结果 1"命中且**强于预期**（预注册写的是"W8 g128 颗粒足够细，预期仍近无损"；
  实测 top-1 一个位置都没动）。

### 第二链：PPL（全集）——**PASS**（2026-10-25 GPU 实测，v2）

首轮 `results/EXP-p1-06-lm-head-w8/ppl_f5_lmhead.json` **作废**（`tools/quant_ppl.py` 的
rtn 臂漏传 `keep_lm_head_fp16` ⇒ `lmhead_only` 臂静默替换 0 个模块、ΔNLL 与 fp16 逐位
相同；缺陷已修 `d055994`，该 JSON 保留作缺陷证据）。重测
`ppl_f5_lmhead_v2.json`（全集 285,831 token，fp16 参照 = R4 的 8.4703 逐位复现）：

| 臂 | PPL | ΔPPL | ΔNLL/token | 自检（realized lm_head） |
| --- | --- | --- | --- | --- |
| `w8a16_lmhead_only_g128` | 8.471081 | **+0.0088%** | +8.77e-5（t=3.92） | **104,632,320 ✓** |
| `w8a16_full_lmhead_g128`（combined） | 8.472258 | +0.0227% | +2.27e-4（t=1.96） | 104,632,320 ✓ |
| noop ×3 | 8.470338 | 0 | **恰为 0** | — |

验收 **Δ < 1% PASS**；上轮缺陷自检点（空跑臂 `realized_bytes={}`）在本轮**不复现**。

### 第三链：MMLU —— **PASS**（2026-10-25 GPU 实测）

`tools/task_accuracy.py --arms fp16,w8a16_lmhead --tasks mmlu`（分层子集 n=200，
先 smoke 验通路——新臂名映射在真模型上替换成功）。
工件 `results/EXP-p1-06-lm-head-w8/mmlu_w8a16_lmhead.json`：
fp16 **53.50%**（与 p1-02 基线同值）→ w8a16_lmhead **53.00%** = **−0.50 点**（门 <2 点，
PASS；样本噪声下与全 w8a16 当年的 −1.0 点不可区分）。

### 采纳：A/B + 翻默认 + 刷新（2026-10-25）

- **A/B**（`run_ab_fresh_process.sh`，独立进程 5 对，canonical 格 w8a16×kernel @1024）：
  **前两轮 fail-closed 作废**（宿主机共租 host 尖峰：off04 6.2%、on07 6.3% > 5%；
  归档 `EXP-f5-ab-20261025-invalid-run{1,2}`），第三轮 **VALID**：off 1.5329 → on 1.4696 ms，
  **−63.3 µs（−4.13%）**、5/5 对胜、金丝雀 0.131%、逐格 host ≤5%（`results/EXP-f5-ab-20261025/`）。
  三轮 Δ（−69.5/−56.2/−63.3 µs）方向量级一致——效应稳健。
- **翻默认**：`FUSION_DEFAULTS["lm_head_w8"]=True`（`b816fc7`，六 True）；
  无 env 金丝雀验证 canonical 格 p50 **1.4706**、`lm_head_w8_effective=True`。
- **刷新 22 格**（LMHEAD=1，`results/EXP-f1-refresh/` F5 轮）：**22/22 干净、0 失败/警告、
  金丝雀漂移 −0.058%**；逐格 `lm_head_w8_effective` 量化格 True / fp16 格 False；
  w8a8@4096 格以 gate-FAIL evidence 形态重跑（R3 裁决）✓。
- **账本**：lm_head 行出榜；剩余榜首 gate_up 29.2 / down_sum 25.8 µs（config 类，已判死）。

## Verdict

**假设成立，F5 采纳（2026-10-25）**。三链精度全过（KL 6.1e-5/6.6e-5 + top-1 165/165；
PPL +0.0088%；MMLU −0.50 点），A/B VALID（−63.3 µs，−4.13%，5/5 对），翻默认后 22 格刷新
干净。**收益来自把地板本身减半**（0.1319 → 0.066 ms）而非捕获缺口——N7 口径的预测兑现：
实测 −63.3 µs/步落在预注册的 40–80 µs 区间内，且两 ctx 刷新格的差（−41/−46 µs）噪声内一致。
`lm_head` W8 g128 是量化叙事的最后一块：**experts + attention + lm_head 全部 W8A16 g128，
router/embedding/KV 刻意保留**（KV INT8 = E6，已批待做）。

## Applicability

```text
适用于：RTX 5090、OLMoE-1B-7B-0924（fp16）、门 = METRICS §1（W8 KL<0.01、top-1≥0.95、
        合取、按窗口均值）、5 窗 wikitext + ones 对照、fused RMSNorm、
        量化 = 对称 RTN g128、scale 存 fp16（lm_head (50304,2048) 对 32/64/128 全整除）
不适用于：其它模型 / 其它 lm_head 形状（tie_word_embeddings=true 的模型会不同——本模型
          untied，lm_head 独立 103M 参数）；W4/更低位宽的 lm_head（未测，预期不过门）；
          采样温度 >0 的下游任务（门测的是 greedy 轨迹 + teacher forcing KL）
原因：  门绑定在这 5 个窗口与 teacher forcing 口径上；lm_head 直接决定 logits，换
        prompt 分布必须重测（与 p1-04 同一条 applicability 逻辑）。
```
