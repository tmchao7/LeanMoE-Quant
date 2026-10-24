# EXP-p1-06-lm-head-w8

> ★ F5 立项（2026-10-25 用户批准）：`lm_head` 也量化成 w8a16 g128 —— 量化叙事的最后一块。
> 预注册：假设与判据**先于**门实测写定（本文写入时 probe 仍在跑、任何结果未读）。

- **ID**: `EXP-p1-06-lm-head-w8`
- **Date**: 2026-10-25
- **Operator**: agent（GPU：租用 RTX 5090 32 GB，sm_120，实测读带宽 1560.3 GB/s）
- **Status**: `running`（门 probe 进行中）
- **接线**: `LEMOQ_LMHEAD_W8` 开关（默认关，commit `79f7b88`）；独立门表
  `LMHEAD_GATE_CITATIONS`（接线时为空 ⇒ fail-closed，门实测通过才填）
- **上游规格**: `docs/FUSION_OPPORTUNITIES.md` F5（含 N7 口径修订）+ `HANDOFF_2026-10-25.md` §5

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

（待门 probe 完成；本节只填实测，不做推导。）

## Verdict

（待三链齐。）

## Applicability

（待收尾时写。）
