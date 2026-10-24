# F5 采纳刷新后的数字同步映射（2026-10-24 夜预写，10-25 采纳轮回填）

> ★ **本文是 F5（`lm_head` w8a16 g128，开关 `LEMOQ_LMHEAD_W8`）采纳轮的取数地图**：
> 翻默认 + `LMHEAD=1` 23 格刷新跑完之后，**照这张表**把记录与 RESULTS 的数字换成**同一轮**
> payload 的取值（硬规则 3：不跨轮混比）。值留空，刷新后回填。
>
> 前置（已完成/待做）：门行已落表（`cd3c97e`，KL 0.0000605/0.0000661、top-1 1.0）；PPL 重测 +
> MMLU 过线；A/B VALID；然后 `FUSION_DEFAULTS["lm_head_w8"]=True` → `F2=1 KV=1 LMHEAD=1
> bash scripts/rounds/run_f1_refresh.sh` → 回填本表 → 同步。

## 1. 哪些数会动（方向与来源）

| 量 | 预期方向 | 来源（刷新 payload 字段） | 新值 |
| --- | --- | --- | --- |
| 全部 16 个量化图格 p50 | 每步 **−40~80 µs**（lm_head GEMV 字节减半：206→103 MB） | `results/EXP-p4b-01-truncation__*/…/result.json` → `latency.per_token_p50_ms` | （回填） |
| fp16 图格 p50 | **不动**（开关对 fp16 无效，`lm_head_w8_effective=False`） | 同上 | （回填，应≈v5 噪声内） |
| `[W]` 字节账 | lm_head 206→103 MB ⇒ 全量化格 `[W]` −103 MB | `weight_bytes_actual`、`quantize_report.realized_bytes_by_target.lm_head` | （回填） |
| 权重占比 / 上限 | 分母变小 ⇒ 份额微降、上限微降 | `bottleneck.top_kernels` 归一 | （回填） |
| E7 `speedup_cudagraph` | 微升（fp16 分母不变、量化分子变小） | p4b payload | （回填） |
| kernel 杠杆（p4c-01） | 微升 | p4c payload kernel/sdpa 比 | （回填） |
| 最好格倍数 | 微升 | 同上 | （回填） |
| kernels/步 | **不变**（lm_head 是替换不是省 kernel；QuantLinear fused 臂 1 次 GEMV） | `bottleneck.kernels_per_step` | （回填，核对用） |

## 2. 逐格检查点（自检不过不回填）

- 每个量化图格 `config.lm_head_w8_effective is True`；fp16 格 `is False`（刷新脚本自检已机械执行）；
- `config.lm_head_w8 = True`（请求状态）；
- `correctness.thresholds.lm_head_gate` 存在且 `gate=PASS`、`source` 指 EXP-p1-06 工件；
- 金丝雀（首尾 w8a16@1024 graph）漂移 ≤5%；逐格 host_share ≤5%。

## 3. 文档同步清单（回填时逐一过）

1. `RESULTS.md`：w8a16 可发表延迟、字节账、（如有）上限表述；
2. `AGENTS.md` §0 / §3 / §4（S3 行标 done 的 F5 半边）；
3. `experiments/EXP-p1-06-lm-head-w8.md`：结果/Verdict/Applicability 三节补全（A/B 数、
   PPL、MMLU、刷新金丝雀）；
4. `experiments/EXP-p4b-01` / `EXP-p4c-01`：★ F5 采纳刷新小节（v6 canonical）；
5. `docs/handoff/HANDOFF_*.md`（若换代）与 `ISSUES_*`（F5 项关闭）；
6. `tools/operator_ledger.py` 重算：**lm_head 行应出榜**（gap 随字节减半消失/大幅缩小）；
7. `docs/figures`：若有图吃 w8a16 p50，`python -m tools.make_figures` 重生成。

## 4. 不变清单（对照防误伤）

- fp16 全部格子与基线记录（EXP-p0-02/03/04）—— 开关不作用；
- E2/E7/p4c-01/p4-01 的**结论**（变化只是分母分子里的常数，不改判定）；
- 门阈值 τ 与 `GATE_CITATIONS` 六行 —— F5 用独立 `LMHEAD_GATE_CITATIONS`，原表一字不动；
- w8a8@4096 的 de-publish 状态（R3 裁决不因 F5 翻案）。
