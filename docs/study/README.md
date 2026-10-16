# 两周吃透 LeanMoE-Quant

> 对象：能读基本 CUDA/Triton，但尚未独立设计过完整推理优化实验的你。每天约 1–2 小时。
> 本目录是**学习材料**，数字仍以 [`RESULTS.md`](../../RESULTS.md)、实验记录和原始工件为准。
> 当前源码基于 `opt/fused-rope` 分支；四个默认路径 kernel 已采纳，RoPE 与 routing 融合是默认关闭的候选。

## 怎么学

每节按 **问题与原理 → 源码入口 → 关键代码 → 证据 → Mac 练习 → 自测** 读。先在纸上画数据流和张量形状，再看实现；读完自己回答题目，最后去[答案册](09-answers.md)核对。命令均从仓库根目录运行；Mac 只做功能验证和历史工件分析，**任何 Mac 计时均不得写进 `results/` 或性能结论**。

| 天 | 内容 | 当天交付给自己的东西 |
| --- | --- | --- |
| 1 | [架构与源码地图](01-architecture.md) §1–3 | 画出一个 decode token 的前向、标出 16 层和每层 top-8 专家 |
| 2 | 架构 §4–6 + [简历证据索引](08-interview.md) | 从 runner 追到一个 `result.json`，解释默认/候选状态 |
| 3 | [量化](02-quantization.md) §1–3 | 手算一组 RTN 的 scale、码值、重建值和字节数 |
| 4 | 量化 §4–7 | 画出 `QuantLinear` 的 separate/fused 分派；解释 KL 门与 1.74× 上限 |
| 5 | [grouped MoE](03-grouped-moe.md) §1–3 | 画出 `(E,2F,H)` / `(E,H,F)` 的地址与 grid |
| 6 | grouped MoE §4–7 | 解释为什么 expert id 只进指针、为什么 K1 用 fp64 oracle |
| 7 | [dense dequant 与 attention](04-dense-attention.md) §1–3 | 解释 per-group scale 广播为什么消除多余 gather |
| 8 | dense/attention §4–7 + [norm 与候选](05-norm-candidates.md) | 推导 split-KV 合并式、解释 RMSNorm 与 RoPE 的融合边界 |
| 9 | [CUDA Graph](06-cudagraph.md) §1–3 | 画出 prefill→warmup→capture→replay 的生命周期 |
| 10 | CUDA Graph §4–7 | 指出固定地址、动态图阻塞和图外 greedy token 的位置 |
| 11 | [profiling](07-profiling.md) §1–3 | 用 `graph_profile.json` 重建一个热点桶 |
| 12 | profiling §4–7 | 独立判断 RoPE/routing A/B 与 VOID 条件 |
| 13 | [简历证据索引](08-interview.md) + [练习](09-answers.md)前先自答 | 不看资料，逐条解释数字的分子、分母、机器与轮次 |
| 14 | 模拟面试与错题复盘 | 口述 10 分钟项目、白板 3 题、修正仍讲不清的源码链 |

## 掌握标准

- **能画图**：从 token、StaticCache、routing id 到专家权重与 logits；能区分 capture 与 replay。
- **能手算**：RTN、4-bit 打包、`[W]` 字节账、grouped kernel grid、`1/(1-0.42)`、A/B 加速比。
- **能定位**：给出一次精确调用链，从 CLI 到替换模块、Triton kernel、计时与结果校验。
- **能解释取舍**：为什么不用 `tl.dot`，为什么 fused dequant 在 eager 可能输，为什么图 replay 内仍有约 539 个 kernel。
- **能审数字**：先查 gate、同轮 A/B、金丝雀、host 污染、适用范围；不把候选说成默认路径。

## Mac 准备与安全边界

```bash
.venv-mac/bin/python -V
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_quant_oracle.py tests/test_quant_linear.py tests/test_cudagraph.py tests/test_profile_graph.py
```

这组测试验证功能契约，不证明 GPU kernel 正确或快。仓库的完整 CPU 门是 `ruff check .`、`black --check .`、`pytest -m "not gpu" -q`、`python -m tools.check_docs`；本目录编写完成后也按此检查。**不运行**已有实验 cell，不覆盖 `result.json`，不改 `GATE_CITATIONS` 或门槛。

## 阅读时的证据优先级

1. 当轮原始 JSON + 对应实验记录；`status=void`/`planned`/`blocked` 不能当完成的性能实验。
2. [`RESULTS.md`](../../RESULTS.md) 的采用数字与适用范围；同一实验的 v1/v2/v3 优先读最新采用节。
3. [`PORTFOLIO.md`](../apply/PORTFOLIO.md) 是面试措辞索引。日期为 2026-10-10 的 [handoff](../handoff/HANDOFF_2026-10-10.md) 含旧状态，不能覆盖后续结果。

想上 5090 时，最后看 [可选真机练习](10-gpu-labs.md)。两周主线不依赖 GPU。
