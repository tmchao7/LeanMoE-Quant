# HANDOFF — Phase 2：反量化融合 kernel（`EXP-p2-01`）+ E7 收益截断（`EXP-p4b-01`）

> 这份文件是给**下一个会话**的。读完 `AGENTS.md`、`PLAN.md` §3/§5、`docs/METRICS.md` §1–2 之后
> 按 §7 的顺序执行。**不要重跑已经完成的探索**（下面 §9 有可直接引用的数字）。

## 1. 为什么是这两件事

`PLAN.md` 的技术判断是：**batch=1 的 MoE decode 是带宽问题，所以量化必须先过正确性门，
然后收益会被 host/launch 开销截断**。这两句话的验证都在本文件里：

- `EXP-p2-01`：**反量化必须融合进 GEMV**，否则 dequant 的读写把收益吃掉 —— 这是"能不能拿到收益"。
- `EXP-p4b-01`：`{tolerant config} × {eager, cudagraph} × ctx` 的 12 格，给出
  **本项目头号结论**：量化收益在 batch=1 上被 host 开销截断到什么程度、`cudagraph` 消掉多少。

> ⚠️ 目前**所有性能数字都是 fp16 的**（见 §9）。量化的性能数字至今为 0 个 —— 这就是本任务的价值。

## 2. ★ 先读这一节：门现在的样子（它决定了哪一格能报性能数字）

`AGENTS.md` 硬规则 2：**没过正确性门的配置不得报性能数字**。门在 `docs/METRICS.md` §1，
但**测量口径在 `EXP-p1-04` 被更正过**（旧口径用的是 1024 个 token id=1，是退化输入）：

| 配置（attn+experts，g128，ctx=1024，32 步 teacher forcing） | 5 窗口真实 prompt 集 mean KL | 门 | 能报性能吗 |
|---|---|---|---|
| fp16 | 参照 | — | ✅ |
| **`w8a16`** | **0.001273**（最差窗口 0.00429） | 0.01 | ✅ **唯一过门的量化配置** |
| `w8a8` | 0.032753 | 0.01 | ❌（真失败，不是 prompt 效应） |
| `w4a16` | 0.093323 | 0.05 | ❌ |

**因此：`p4b-01` 矩阵里 `{fp16, w8a8, w4a16}` 三档，现在只有 `fp16` 与 `w8a16` 能报性能。**
`w8a8` 那一档要么先修激活量化（见 §8 的可选项），要么跑但**标成 correctness-failed、
明确写"不得用于性能结论"**。这是纪律，不是可以商量的地方。

好消息：`w8a16` 本身就是一次真实的量化（**权重的 49.4%**），足以支撑 E7 故事。
`EXP-p1-04` 已经把它的 mean/sd/min/max 与 noop 自检都落了盘，可以直接引用。

## 3. `EXP-p2-01`（计划里超时 25 GPU 小时，有兜底）

矩阵里已有 4 格：`bit_width ∈ {w8a8, w4a16} × dequant ∈ {fused, separate}`。
**建议再加 2 格 `w8a16 × {fused, separate}`** —— 它是唯一过门的量化配置，
E7 结论要靠它。加格要在记录里写明（这是对计划矩阵的扩展，不是替换）。

要做的事：
1. **`separate` 臂**：`dequant` 成 fp16 权重 → 普通 `F.linear`（两次显存往返）。
2. **`fused` 臂**：Triton kernel 里直接读 int8/int4 + scale，GEMV 输出 fp16。
   目标：达到**实测可达带宽**的 70%（`results/EXP-p0-01-machine-probe/result.json`
   里有实测 **1561.9 GB/s**，170 SM）。**不要用 spec 带宽。**
3. 报告：kernel 级延迟（p50/p99）+ `dequant 占比`（`docs/METRICS.md` §2）+ ncu 证据。
4. **兜底（计划已授权）**：到 25 GPU 小时若没到 70%，就用现成融合 kernel
   （vLLM / torchao）当"已融合"臂，并把贡献转落到"字节预算分析 + E7 截断解释"。

## 4. `EXP-p4b-01`（计划里的头号结论）

> ★ **预注册已完成（2026-10-06 离线会话）**，权威清单见
> `experiments/EXP-p4b-01-truncation.md`：矩阵现为 **16 格**（加了 `w8a16` 轴 +
> `config.moe: grouped`），其 §3 是**依赖链 P0–P3**。关键事实：**量化臂被 P1 阻塞**
> （`replace_moe_blocks` 只吃 fp16 融合权重，hf 又不能 capture），所以**今天可跑的只有
> fp16 的 4 格**；量化 × 非 hf 的组合 runner 已实现第四条拒绝。
>
> ★ **2026-10-07 上卡进展**（详见该记录 §2.5 / §2.6）：
> - **P2 已完成**：ctx=4096 的门实测补上（w8a16 PASS；**w8a8 的 KL 过门但 top-1 不过 →
>   FAIL**；w4a16 FAIL）。`GATE_CITATIONS` 现在存实测 `kl`/`top1`，`gate` 由
>   `gate_from_metrics()` **推导**（合取：KL < 门 且 top-1 ≥ 0.95），有测试钉住。
> - **fp16 4 格已测且有效**（金丝雀漂移 0.10%）：eager 9.73/9.67 ms、cudagraph
>   4.07/7.31 ms（1024/4096），`host_share` 67%/61% → 1.4%/0.7%。
> - **第 1 轮曾整轮作废**：共享卡上另一条工作线的 `pareto_offload.py` 占到 17.8 GB，
>   金丝雀漂移 37.1%、`cudagraph×1024` 被 OOM 阻塞。协议是**记录不杀**，等它退出后重跑。
> - **E7 仍不可判定**：只有分子（fp16），分母（w8a16）要等 **P1**。
>
> 下面这段是当时按 12 格写的规划，保留作背景：

12 格 = `bit_width × execution × ctx`。按 §2 的实际可用性：
- 必跑：`fp16`、`w8a16` × `{eager, cudagraph}` × `ctx ∈ {1024, 4096}` = **8 格**
- 追加（标 correctness-failed，只报延迟不报收益）：`w8a8` × 同上 = 4 格
- 每格都要报：per-token p50/p99、**host/launch 占比**（一等指标）、以及
  "相对 fp16 的收益"是否被 host 开销截断。

**想清楚要问的问题**：`w8a16` 把 `[W]` 从 13.42 GB 降到 6.63 GB（49.4%），
理论带宽收益接近 2×；实测收益是多少、被什么吃掉、`cudagraph` 消掉多少 —— 这就是 E7。

## 5. 环境与命令

```bash
# 服务器（免密）
ssh 5090b

# 必设（否则 13 GB 模型落到 30 GB 系统盘）
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com          # huggingface.co 在这台机器上不可达（000）
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

# ★ 每次 timing 前都查：这台机器上还有别人的进程（见过 pareto_offload.py 占着卡，
#   那次 ctx=4096 测出 10.38 ms / host-share 67.5%，清干净重测是 3.2356 ms）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 不要在别人的项目进程上动手：/root/autodl-tmp/mf-work/、/root/MegaKernel-vLLM、/root/models/Qwen3-0.6B

# 离线门（Mac 或服务器都行）
uv run pytest -m "not gpu"
uv run ruff check . && uv run black --check .
uv run python -m tools.check_docs

# 现有工具（都在 tools/ 下，可直接复用）
#   quant_probe.py   量化精度探针：--arms prompt_set --prompt-source wikitext \
#                    --prompt-cache /root/autodl-tmp/lemoq-data/wikitext2_test_8k.pt \
#                    --prompt-windows 5 --include-ones-control
#   quant_ppl.py     PPL + 配对 ΔNLL（第二个指标）
#   quant_error_audit.py  纯权重域审计
```

## 6. ⚠️ 坑清单（全部是本仓库实测踩过的，不是推测）

1. **`_snapshot` 会把原模块停在 CPU 省显存** → **任何模型前向之前必须先 `_restore`**，
   否则第一次 forward 就 `mat2 is on cpu`。
2. **就地改权重会污染快照**：`w.data.copy_()` 改的是快照里**同一个对象**，恢复变成空操作，
   噪声臂会累积（实测被交错插入的 `noop` 臂抓到）。做法：新建模块挂上去。
3. **名字必须在量化之前收集**：`quantize_model` 把 `nn.Linear` 换成 `QuantLinear`
   （**不是** `nn.Linear` 子类），之后再按类型找就一个也找不到，而统计会静默变成 0。
4. **门的 prompt 口径**：不要在 `ids = full((1, ctx), 1)` 上测 KL 并当门用
   （那是退化输入，膨胀 21–33×）。用 `--prompt-windows 5` 或 `quant_ppl.py`。
5. **计时必须交错**：臂之间 round-robin，报 MAD；不要一条臂跑完再跑下一条
   （机器上可能有别的负载漂移）。
6. **`results/` 是 gitignored**：跑完必须 `rsync -az 5090b:/root/LeanMoE-Quant/results/ ./results/`
   到 Mac，否则数据随机器关停消失。
7. **Mac 上的任何数字不得进 `results/` 或结论**。
8. **依赖**：`configs/constraints.txt` 至今写着"尚未在服务器上验证"（且里面的 torch 是 2.11.0，
   服务器实际是 2.8.0）。**本会话踩过**：`pip install datasets` 默认装 `datasets 5.1.0` +
   `huggingface-hub 2.1.1`，而 `transformers 4.57.6` 要求 `hub<1.0` → **transformers 直接坏了**。
   已用 `"huggingface-hub==0.36.2" "datasets==3.6.0"` 恢复。
   **装任何包之前先记录版本，装完逐项核对**；有空就把 `uv pip freeze` 覆盖 `constraints.txt`。
9. **merge，不要 rebase，不要 force-push**；运行中的 agent 走 feature 分支。
10. **服务器上有一个先于本项目存在的失败测试**：`tests/test_scripts.py::test_bootstrap_dry_run_...`
    （服务器没装 `uv`）。它不是你的回归，但在记录里要写明它是 pre-existing。

## 7. 建议的执行顺序

1. 读 `AGENTS.md` / `PLAN.md` §3.1、§5 / `docs/METRICS.md` §1–2 / 本文件 §2；
   再读 `experiments/EXP-p1-04-prompt-set-gate.md`（门的现状）与 `EXP-p0-04-cudagraph-static-moe.md`
   （已有的 cudagraph 执行路径）。
2. **先快照当前状态**：`nvidia-smi`（有没有别人的进程）、`git log -1`、跑一遍离线测试。
3. `EXP-p2-01`：先写 `separate` 臂并拿到数字（它是 fused 的对照，必须同进程同条件），
   再写 fused kernel。**每加一格都在同一个进程里交错测。**

   > ★ **离线接线已完成**（2026-10-06，本仓库当时的离线会话）：格子 id、
   > `--bit-width/--dequant/--group-size`、三条拒绝（fused 未实现 / 无门实测 /
   > 量化+cudagraph 未接线）、`GATE_CITATIONS` 与 `EXP-p1-04` 的逐字对齐、
   > 量化 hook 与字节账 —— 全部实现并有离线测试（352 passed）。
   > **还差的清单见 `experiments/EXP-p2-01-dequant-fusion.md` §4**（kernel、
   > dequant_share、跑 6 格、记录）。矩阵的 bit_width 轴已加 `w8a16`
   > ——它是 p1-04 之后唯一过门的配置，其余两档只能跑出 verdict=FAIL 的格子。
4. `EXP-p4b-01`：用 §2 判定过的配置跑 8 格（+4 格标 failed）。
5. 每格落 `results/<EXP-ID>/result.json` + `experiments/<EXP-ID>.md` + 一个 commit。
6. 最后：两台机器的测试、`ruff`/`black`/`check_docs`、`rsync` 结果回 Mac、
   **把服务器留在与 Mac 相同的 commit 上**。

## 8. 可选项（如果 $w8a8$ 那一档必须救）

`w8a8` 现在是 Phase 1 唯一真失败的条目（`EXP-p1-04`）。想救它是独立的一个实验：
激活 scale 粒度（per-tensor / per-token / per-channel）× clamp 搜索。
它同时回答 `[T]`（激活带宽）那一半，值得做，但**不要和 p2-01/p4b-01 混在同一个实验里**。

## 9. 现在就能引用的数字（省得重新推导）

- 机器：RTX 5090 32 GB、170 SM、驱动 595.71.05、**实测带宽 1561.9 GB/s**
  （`results/EXP-p0-01-machine-probe/result.json`）。
- 模型：OLMoE-1B-7B-0924，权重 **13.42 GB fp16**（attention 0.54 GB = 4%，experts 12.88 GB = 96%）。
- 字节账（实测，非估算）：`w4a16` = fp16 的 **25%**、`w8a16` = **49%**、padding 膨胀 **0.0%**。
- 已实现并实测的 fp16 执行路径：CUDA Graph（`EXP-p0-04`）、grouped MoE GEMV（`EXP-p2-02`）、
  eager attention（`EXP-p2-04`）、自研 split-KV attention kernel（`EXP-p2-05`）。
  累计 **27.5×**（ctx=1024：82.64 ms → 3.0077 ms，达成率 50.2%；ctx=4096：25.5×）。
  **这些是 fp16 的数字。**
- 正确性（更正后的门，`EXP-p1-04`）：`w8a16` **过门**（mean 0.00127，5 窗口）；
  `w8a8` 0.0328 与 `w4a16` 0.0933 **不过门**。
- PPL（`EXP-p1-03`，wikitext-2 test 前 8192 token，配对 ΔNLL）：fp16 PPL **9.1994**；
  `w8a16` 与 fp16 无法区分（|t| < 1）；`w4a16` +2.6…+5.9%（超 <1% 门）。
- 量化精度取证工具：`tools/quant_probe.py`（KL，多窗口）、`tools/quant_ppl.py`（PPL）、
  `tools/quant_error_audit.py`（权重域）。三者都带 noop 自检。

## 10. 验收标准（做完自检）

- [ ] `EXP-p2-01`：`fused` vs `separate` 在**同一进程、交错**下测得；报了 dequant 占比与
      ncu 证据（或明确声明走了 §3.4 的兜底）；达成率对照**实测** 1561.9 GB/s。
- [ ] `EXP-p4b-01`：≥8 格（fp16/w8a16 × eager/cudagraph × ctx 1024/4096），
      每格报 p50/p99 + host 占比；`w8a8` 那 4 格如跑则明确标 correctness-failed。
- [ ] 每格有 `result.json` + `experiments/<ID>.md`（假设写在结果之前）+ commit。
- [ ] 门槛一个都没改；没过门的配置没有性能结论。
- [ ] 两台机器测试绿（服务器那个 `uv` 的失败要标注 pre-existing）；
      `results/` 已 rsync 回 Mac；服务器停在同一个 commit。
- [ ] 结论里写明适用性：GPU、**实测**带宽、dtype、batch=1、ctx。
