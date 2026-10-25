# 离线窗口规划（Mac-only，服务器已关）：五件事，按价值排序

## 1. F5 门行落库（= 明天 GPU 清单第 1 项，纯离线可做，先做）

门 probe 工件已在 Mac（`results/EXP-p1-06-lm-head-w8/probe_f5_lmhead{,_ctx4096}.json`，已验真：
每窗 `replaced={'lm_head':1}`、realized 104.6 MB）：

- `benchmarks/runner.py`：`LMHEAD_GATE_CITATIONS` 从空表填两行——ctx=1024 kl 0.0000605 /
  top1 1.0；ctx=4096 kl 0.0000661 / top1 1.0（各带 `source`/`arm` 字段；行注写明"只解锁
  A/B 的 ON 臂，采纳仍以三链+A/B+刷新为准"；容差核对 3.8e-8/1.25e-7 << 5e-7）。
- `tests/test_runner_cells.py`：新增 `test_lmhead_gate_citations_match_the_artifacts`
  （照 GATE_CITATIONS 钉法，读本地工件）；修 `test_lm_head_w8_without_gate_measurement_…`
  （表不再为空 ⇒ 改 `monkeypatch.setattr(R, "LMHEAD_GATE_CITATIONS", {})` 模拟未测状态）。
- Mac 四门 → commit（`quant:`）→ push。

## 2. 刷新脚本加 `LMHEAD=1` 叠加（离线代码 + Mac DRY 验证）

`scripts/rounds/run_f1_refresh.sh` 照 `KV=1` 模板加一行 `[ "${LMHEAD:-0}" = "1" ] && export
LEMOQ_LMHEAD_W8=1`；用例头注/用法补一行。脚本支持 `REPO_ROOT`/`PY` 覆盖 ⇒ **Mac 上
`DRY=1 LMHEAD=1 PY=.venv-mac/bin/python bash scripts/rounds/run_f1_refresh.sh` 验证命令清单
生成**（results/ 的 23 格 payload 本地都在）。commit（`bench:`）。

## 3. 一致性扫尾（几分钟）

grep 撤销后残留：`8.8509|8.8232|8.8189|0.004259` 全仓——确认只出现在"证据/历史/撤销标注"
语境（evidence、archive、记录的判定史），不再有陈述"现发布"的孤例；有就改。

## 4. F5 采纳同步映射（`docs/handoff/ADOPTION_SYNC_F5_2026-10-25.md`，离线文档）

照 ADOPTION_SYNC_2026-10-24 体例，**预写"翻默认后抓哪些数、从哪抓"**（值明天刷新后回填）：
23 格 canonical p50（预期全降 ~40-80 µs）、`[W]` 与 realized_bytes（lm_head 206→103 MB），
E7 / kernel 杠杆 / 最好格 / 上限占比重算、operator_ledger 重算（lm_head 行应出榜）、
payload 检查点（`lm_head_w8_effective=True`、`LMHEAD_GATE_CITATIONS` 引用进 correctness）、
测试/文档同步清单。commit（`docs:`）。

## 5. E6 KV INT8 侦察（离线大头，联网 recon + 设计文档）

`docs/reference/KV_INT8_RECON_2026-10-25.md`（照 QUANT_FORMAT_RECON 体例）：

- **联网调研**：KIVI（per-channel key / per-token value、2-bit 也可行的原因）、vLLM FP8 KV、
  llama.cpp `q8_0`/`q4_0` KV、FlashAttention/KV 量化的避坑（softmax 分母、scale 与量化
  顺序、outlier channel）——工业界与学术界的已有做法。
- **本仓库工程形状**：字节账（ctx=4096 单步 KV 读 = 16 层×16 头×128×2×4096×2B ≈ 512 MB
  fp16 ⇒ INT8 省 ~165 µs/步，比 F5 还大 2-3 倍，是最大已声明杠杆）；F4 的 `kv_direct_write`
  就是天然的"写入时量化"挂点；split-KV kernel 内解包的设计草案（K per-channel + V per-token
  的两种 scale 粒度）；正确性影响 → 需要新 GATE 行（KV 量化直接改 logits）。
- **裁决段**：推荐的最小实验路径（EXP-p5-01 的栅格）、验收判据、预算与风险（工程量大，
  需用户批准立项——本文档为立项材料）。

## 不做（等 GPU / 等批准）

PPL 重测、MMLU、A/B、翻默认、23 格刷新（等开机）；R5 lm-eval、R7 vLLM 锚点（等批准）；
NVFP4（默认放弃）。每件独立 commit，门令不接管道，改前 Read、行号手术不裸写。