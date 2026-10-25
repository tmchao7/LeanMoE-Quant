# LeanMoE-Quant

**Byte-budget-driven quantization + Triton kernel optimization for MoE decode at batch=1.**
单卡 RTX 5090 上，把 OLMoE-1B-7B（64 专家/top-8）的单请求 decode 从 82.6 ms/token 压到
**1.46 / 1.61 ms**（ctx 1024/4096），最好格对原生基线 **2.44× / 4.25×**——并实测出这条路的
结构性上限。每一个数字都有工件、门禁与同轮 A/B 支撑，负结果同样入库。

> **EN** — This repo asks *how far quantization can actually take batch-1 MoE decode*, then
> measures the answer instead of assuming it: six custom Triton kernel families, seven
> independently A/B-tested levers (all adopted), a measured structural ceiling (~1.69× at
> ctx=1024, because weights are only ~41% of a decode step), and first-class negative results
> (every 4-bit route failed the accuracy gate and is published as such).
>
> **中文** — 本项目回答"量化在 batch=1 的 MoE decode 上到底能走多远"，并且**测出**答案：
> 6 个自研 Triton kernel 族、七条各自独立同轮 A/B 的杠杆（全部采纳）、实测的结构性上限
> （权重只占一步 ~41% ⇒ 完美权重路径也只有 ~1.69×）、以及一等的负结果（4-bit 全系被精度门
> 挡住并如实发布）。**所有数字只在过正确性门后发布**（硬规则 2）。

`LLM inference | quantization (W8 / KV-INT8) | Triton kernels | CUDA Graph | MoE | batch=1 decode`

---

## Key results 关键结果

| | value 数值 | scope 口径 | source 出处 |
| --- | --- | --- | --- |
| Decode step (canonical) 解码步 | **1.4602 / 1.6075 ms** (ctx 1024/4096) = 685 / 622 tok/s | w8a16 + lm_head w8 + KV-INT8 × split-KV × CUDA Graph | [RESULTS.md](RESULTS.md) §2 |
| Best cell vs native 最好格 | **2.44× / 4.25×** vs fp16×sdpa | same round 同轮 | [RESULTS.md](RESULTS.md) §2 |
| Byte floor 字节下限 | 1.60 ms（52× from the 82.6 ms eager start） | measured read BW 1560.3 GB/s，非规格值 | [RESULTS.md](RESULTS.md) §1 |
| Quantization ceiling 量化上限 | **~1.69× / ~1.31×**（weights = 40.9% / 23.5% of a step 权重占步份额） | **measured**，not derived 不是推演 | [RESULTS.md](RESULTS.md) §3 |
| Accuracy 精度 | KL 0.0014 · PPL +0.0088%（全集）· MMLU −0.50 pt；KV-INT8 KL 0.0006/0.0013 | three chains + per-surface gates 三链+分表面门 | [RESULTS.md](RESULTS.md) §3.1 |
| Negative results 负结果 | 4-bit 全系 FAIL（W4 RTN/GPTQ、MXFP4、混合退路）；w8a8@4096 de-published；融合反量化被否证 | evidence-only, no latency published 只进证据不发布延迟 | [RESULTS.md](RESULTS.md) §5 |

![Seven adopted levers — same-round A/B step-time reduction](docs/figures/fig4_lever_ab.svg)

![Per-axis speedups](docs/figures/fig1_per_axis_speedups.svg)

![To the bandwidth wall](docs/figures/fig2_to_the_wall.svg)

![Kernel bandwidth attainment](docs/figures/fig3_kernel_attainment.svg)

> Figures are regenerated from artifacts by `python -m tools.make_figures` — never hand-edited.
> 图由工件重生成，从不手改（`docs/figures/inputs.json` 记录每个数的来源）。

---

## The seven levers 七条杠杆（all adopted 全部采纳）

| # | lever 杠杆 | same-round A/B 步时降幅 | note 备注 |
| --- | --- | --- | --- |
| 1 | CUDA Graph (eager → graph) | 2.43× / 1.35× | host_share 67% → 1.4% |
| 2 | Weight W8A16 (+lm_head W8) | 1.22× / 1.10×（步级） | near-lossless 近无损 |
| 3 | split-KV attention (vs HF sdpa) | 1.51× / 1.46×（w8a16 格） | Flash-Decoding 三段式 |
| 4 | Fused RMSNorm | 1.24×，kernels/步 994→539 | `F.rms_norm` 反而慢 5% |
| 5 | Packed q/k/v + KV direct write | −4.8% / −2.9% | 48→16 调用/步 |
| 6 | lm_head W8 | −4.1%（−63 µs/步） | 三链精度 PASS |
| 7 | **KV-INT8**（scale folding in-kernel） | **−6.5% @4096**（−109 µs/步） | 2026-10-25 采纳 |

Kernels per step: **994 → 345**. ★ 不得跨轮相乘当"总加速"（[RESULTS.md](RESULTS.md) §2 红线）。

## What it asks 项目在问什么

**EN**: batch-1 decode is purely bandwidth-bound — every token streams the active weights once.
So the only lever is *bytes per token*, and the project measures where that lever stops:
quantization pays only to 8-bit on this model (all 4-bit routes fail the accuracy gate);
the remaining step time is routing/elementwise/launch overhead, which is what the fusion
levers attack; and the **ceiling** is measured, not assumed — even a perfect weight path
yields only ~1.69× because weights are just ~41% of a step.

**中文**：batch=1 decode 是纯带宽受限——每个 token 都要把激活权重流一遍，所以唯一的主线是
**每 token 字节数**。四条预注册推论（量化收益随 batch 衰减 / 反量化必须融进 GEMM / MoE 有
量化上限 / launch 开销截断收益）各自被实验回答；4-bit 被精度门挡住、融合反量化在 eager 上
被否证——边界与负结果同样是本项目的一半产出。完整计划见 [PLAN.md](PLAN.md)。

## Reproduction discipline 复现纪律

- **Correctness before performance 先门后测**：KL / top-1 gate per quantization surface
  （权重 / lm_head / KV 各一张门表，fail-closed）；不过门的格 latency 置 null。
- **One variable per round 一轮一变量**；same-round adjacent A/B with per-pair wins + MAD；
  canary drift >5% ⇒ the whole round is VOID（fail-closed，作废轮留档不发布）。
- **No spec bandwidth 禁用规格带宽**：roofline 分母用实测可达带宽（1560.3 GB/s）。
- 负结果同样记录；`results/` 为机器本地工件（gitignored），权威记录是
  [experiments/](experiments/) 下的 markdown。

## Hardware & scope 硬件与适用范围

- 租用 RTX 5090 32 GB（sm_120，有 cp.async/TMA/FP8/MXFP4、无 wgmma）为唯一测量卡；
  学校 V100（sm_70）为跨带宽第二数据点（未跑）。所有结论**不跨卡迁移**。
- **batch=1 单流**课题：零 req/s、无 serving 集成（不接 vLLM/SGLang——定位选择，见
  [AGENTS.md](AGENTS.md) §2 规则 10）；batch>1 的吞吐曲线只测过 fp16 单臂。
- 立论可迁移到端侧（UMA 下 decode 是更纯粹的字节问题），但数字不迁移。

## Quick start 快速开始

```bash
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"
uv run pytest -m "not gpu"      # 733 tests on CPU/Mac
uv run python -m tools.make_figures   # regenerate all figures from artifacts
# GPU box: see ENVIRONMENT.md + docs/reference/SERVER_RUNBOOK.md
```

## Repo map 目录

```text
PLAN.md / RESULTS.md / AGENTS.md / EXPERIMENTS.md   # 计划 / 一页纸结果 / 规则 / 记录格式
experiments/<EXP-ID>.md      # ★ authoritative per-experiment records 权威记录
src/lemoq/kernels/           # 6 Triton kernel families（grouped MoE GEMV、fused dequant、
                             #   split-KV attention、fused RMSNorm、packed qkv、KV-INT8 读写）
benchmarks/runner.py         # harness：五条加载前拒绝 + 门表 + payload schema
tools/                       # probe/ledger/figures/quant-gate 工具链
docs/figures/                # 4 figures from artifacts（docs/README.md 有完整文档地图）
docs/apply/                  # PORTFOLIO（简历/面试语料）+ 外部对标评估
```

Related projects 姊妹项目：MegaKernel-vLLM（整模型融合）、MoE-Flow（专家放置与调度）——
表示（量化）→ 放置 → 执行，三者互补。

---

> Status 状态（2026-10-25）：量化（权重 W8 + KV-INT8）与融合（七开关）两条线**系统性探完并全部
> 采纳**；剩余为未批项（E5/E8 跨带宽、专家级混合精度 4-bit、F7 o_proj 融合——均已有预注册或
> 侦察材料）。进度与红线见 [AGENTS.md](AGENTS.md) §0/§3；一页纸结果见 [RESULTS.md](RESULTS.md)。
