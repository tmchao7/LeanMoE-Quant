# 08｜简历每条话怎样经得住追问

## 1. 先给 90 秒版本

> 我研究的是 RTX 5090 上 OLMoE-1B-7B、batch=1 单流 decode。起点 fp16 eager 约 82.6 ms/token，相对按实测 1561.9 GB/s 算出的 1.60 ms 字节下限有约 52 倍差距，但起点 host 主导，绝对延迟不稳。我的优化按变量分轮验证：先让动态 MoE 变为可 capture 的 grouped 路径，再比较图执行、W8A16、attention 实现、融合 norm 和投影访存。W8A16 图内同轮提速 1.1911×；权重只占 graph 步约 42%，故理想权重路径的结构性上限约 1.74×。kernel 级隔离证据显示约九成步级节省可由专家 GEMV 解释。正确性门、金丝雀和负结果都保留在仓库。RoPE/routing 融合是另外测过的候选，默认关闭。

这段的每个限定词都有用：**单流、ctx、同轮、采用版、候选**。面试官问“总共快多少”，先说明五条杠杆来自不同轮次，不相乘；给每条合法 A/B。不要用 `82.6→3.59 ms` 写成一条同轮总加速。

## 2. Bullet → 源码 → 证据

| 简历主张 | 直接指向的源码 | 采用证据与计算 | 面试必须补充的边界 |
| --- | --- | --- | --- |
| 起手 82.6 ms、1.60 ms 字节下限、约 52× | [`runner.py`](../../benchmarks/runner.py)、[`attribution.py`](../../benchmarks/attribution.py) | [RESULTS §1](../../RESULTS.md)、[p0-02](../../experiments/EXP-p0-02-baseline-fp16.md)；82.6436/1.5982≈51.7 | fp16 eager、ctx=1024、5090；host-bound 起点不稳定，不是同轮优化前臂 |
| grouped MoE 5.26→4.07 ms | [`moe_grouped.py`](../../src/lemoq/kernels/moe_grouped.py)、[`_grouped_triton.py`](../../src/lemoq/kernels/_grouped_triton.py) | [p2-02](../../experiments/EXP-p2-02-moe-grouped-gemm.md) 同轮 A/B，5.2637/4.0715≈1.293 | 后来融合 norm 的单格 3.6828 ms 属另一轮，不可跨轮除 |
| 投影 523→826 GB/s | [`_dequant_triton.py`](../../src/lemoq/kernels/_dequant_triton.py) | [p4c-01 §5.4/§5.6](../../experiments/EXP-p4c-01-in-graph-attention.md)，8.14→5.15 µs/调用 | kernel 级，不是全步 1.58×；旧版逐元素取 scale |
| split-KV 让 W8A16 图内步级 1.30× | [`attn_decode.py`](../../src/lemoq/kernels/attn_decode.py)、[`_attn_triton.py`](../../src/lemoq/kernels/_attn_triton.py) | [p4c-01 v3](../../experiments/EXP-p4c-01-in-graph-attention.md) 的同格 `sdpa→kernel` 1.3034× @1024 | 换 attention 实现，不是量化本身的 1.30×；ctx=4096 为 1.2555× |
| 融合 RMSNorm 全步 1.24×、994→539 | [`rmsnorm.py`](../../src/lemoq/kernels/rmsnorm.py) | [p4c-02 §5.6](../../experiments/EXP-p4c-02-elementwise-tail.md) 同进程交错 2.4143/1.9537=1.2358，15/15 胜 | 记录把 HF 描述为“7 kernel 链”，但实测净减 455=65×7；不要仅用“7→1”推导 994→539，讲 profiler 实数 |
| CUDA Graph 9.88→4.07 ms，2.43× | [`cudagraph.py`](../../benchmarks/cudagraph.py)、[`runner.py`](../../benchmarks/runner.py) | [p4b-01 §2.8](../../experiments/EXP-p4b-01-truncation.md) 同实现同轮 9.8841/4.0670 | “一次 replay”是 host 提交方式；GPU 图里仍有许多 kernel |
| W8A16 1.19×、上限 1.74× | [`quant/linear.py`](../../src/lemoq/quant/linear.py)、[`moe_grouped_quant.py`](../../src/lemoq/kernels/moe_grouped_quant.py) | [p4b-01 §P2j](../../experiments/EXP-p4b-01-truncation.md) 3.5881/3.0124=1.1911；42% 来自归因，上限由 `1/(1−f)` 推导 | 上限不是实跑速度；W4A16 gate FAIL、W8A8@4096 边缘不可泛化 |
| 专家 GEMV 96.3→63.5 µs、约九成收益 | [`_grouped_triton.py`](../../src/lemoq/kernels/_grouped_triton.py)、[`ncu_evidence.py`](../../tools/ncu_evidence.py) | [NCU_EVIDENCE §3/§4.3](../reference/NCU_EVIDENCE.md) 的 isolated cold 测量；16 层约省 0.525 ms | 旧文档 90.4% 用 3.0084；采用 3.0124 时约 91%，口述“约九成” |
| KL/PPL/MMLU | [`runner.GATE_CITATIONS`](../../benchmarks/runner.py)、[`quant_gate.py`](../../benchmarks/quant_gate.py) | [p1-04 v2](../../experiments/EXP-p1-04-prompt-set-gate.md)、[RESULTS §3.1](../../RESULTS.md) | KL/top-1 门阈值项目自定；MMLU 仅 n=200；不可声称普适精度 |
| RoPE +8.4%、routing +3.0% | [`rope.py`](../../src/lemoq/kernels/rope.py)、[`_routing_triton.py`](../../src/lemoq/kernels/_routing_triton.py)、[`profile_graph.py`](../../tools/profile_graph.py) | [RoPE Report](../compose/spec/fused-rope.md)、[routing Report](../compose/spec/routing-fusion.md) 与各自 A/B JSON | 两个都是默认关候选；RoPE 第一轮 VOID、有效轮有 host 噪声 |

“501 个单元测试”以 [RoPE Report](../compose/spec/fused-rope.md) 的当时 Mac `501 passed` 为出处；测试数随分支演进，不当作永久事实。更稳妥的表述是“500+ CPU 测试”并说明真机 GPU parity 另跑。四个**已采纳 kernel 族**的 K1 门是 GPU 测试，Mac 的 CPU 测试只覆盖 oracle、shape 和 fallback。

## 3. 常见追问的答题骨架

**“为什么不用 `tl.dot`？”** 先写当前运算形状 `M=1`、top-8 每个 slot 不同专家；指出读权重主导且 `tl.dot` 的 M tile 填充会浪费。再补边界：换成大 batch/高复用、FP8 Tensor Core 路径时结论要重测。

**“融合为什么有时慢？”** 举 E2：隔离 kernel 少字节，但 eager 全步约 1.9 万 kernel、host/launch 占比高，fused 臂约慢 4–5%；进入图并修 scale gather 才兑现。结论必须限定路径和测量层级。

**“怎么确信 profile 找对了 RoPE？”** 先用算子语义与调用次数拼 160 次/约 0.216 ms 的功能链，随后候选 A/B 的 busy 减约 0.194 ms、方向和量级相符。补充 add 份额是估计、首轮 VOID、有效轮的 host 噪声与默认关闭状态。

**“怎么知道数字可信？”** 正确性门先于性能；gate FAIL 的 `latency=null`；同轮 A/B；同格首尾 canary >5% 则 VOID；graph cell 查 host share；原始 JSON+记录+适用范围。ncu 没权限就说清替代口径和不能推断的 stall。

## 4. 模拟面试题

闭卷作答，每题不超过 2 分钟；答案与评分点见 [09](09-answers.md#模拟面试)。

1. 白板画出一层 grouped MoE 的 `gate/up/down` 权重布局、两个 grid 和数据依赖。
2. 推导 RTN 的 scale/code 与 W8A16 的字节账；为什么 W8 权重约减半却只快 19%？
3. 画出 graph 的 prefill/capture/replay 生命周期。为何 HF 的 `torch.where` 不行，而 `filled_len` 可以？
4. 给你 `wall=2.0 ms,busy=1.8 ms,canary=+6%,tokens_equal=True`，能发表 10% A/B 吗？
5. ncu 计数器拿不到时，你如何区分“接近带宽墙”与“寄存器 stall”？

## 5. 表述底线

不说 req/s 或线上服务吞吐；不说已跑 FP8/MXFP4、量化×batch 或 ncu stall；不把不同轮的数字相乘；不把默认关闭候选说成已采纳；不把 Mac CPU 测试说成 Triton kernel 的 GPU parity。应用范围按 [RESULTS 顶部](../../RESULTS.md) 原话说明。
