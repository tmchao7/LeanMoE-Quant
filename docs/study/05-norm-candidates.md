# 05｜融合 RMSNorm 与两个候选

## 1. RMSNorm：为什么 7 个 kernel 能变成 1 个

HF 的 `OlmoeRMSNorm.forward` 对一次 norm 依次做 cast、平方、均值、rsqrt、乘法、再 cast 与乘权重；本模型一次 decode 约 65 个 norm，重复 launch 与中间张量成本显著。[`rmsnorm.py`](../../src/lemoq/kernels/rmsnorm.py) 实现一行内完成：

```text
r = rsqrt(mean(float(x)^2) + eps)
y = cast_fp16(float(x) × r) × weight
```

**转换位置也属于数值语义**：先按 fp16 舍入，再乘 fp16 weight；不能把所有乘法都留在 fp32 并声称与 HF 相同。[p4c-02 §5.5/§5.6](../../experiments/EXP-p4c-02-elementwise-tail.md) 记录了 `torch.nn.functional.rms_norm` 数值近似但全步反而慢约 5% 的反例，原因是保留 reduce 且多了 fp32 行乘。

Triton `_rms_norm_kernel` 在最后一维归约；host 的 `H=2048,R=1` 或 `H=128,R=16` 让一个 program 约处理 2048 元素，grid 为 `ceil(M/R)`。输入与权重须 fp16 且连续，`H` 为 2 的幂、`H≤8192`；这些是实现边界，非普适 RMSNorm 属性。`replace_rms_norms` 仅实例级覆写 `forward`，可恢复；当前 decode 默认 `fused`，`runner` 会把实现写进结果配置。

同进程交错 A/B 的结果为全步 **1.2358×**、每步 kernel **994→539**、15/15 轮胜，见 [p4c-02 §5.6](../../experiments/EXP-p4c-02-elementwise-tail.md)。这里约少 `455` 个 kernel，接近 `65×(7−1)=390` 但不必完全相等：实际图内还有实现和计数口径差异，不能只凭乘法推出 profiler 总数。

## 2. RoPE：从四种 kernel 名还原功能链

HF 的 `rotate_half` = 切半、前半取负、拼接；随后计算 `x·cos + rotate_half(x)·sin`。其成本分散在 neg、cat、mul、add 四类 kernel 名里。本地历史工件 `results/graph_profile.json` 的功能块归因约 **160 次/步、0.216 ms/步、GPU busy 的 11.5%**；单看 top kernel 名容易漏掉整个链。候选 [`rope.py`](../../src/lemoq/kernels/rope.py) 一个 Triton kernel 分别 launch 给 q/k，每层 2 次，整步从 160 次约降到 32 次。`attn_decode._decode_forward_impl` 读取 `LEMOQ_FUSED_ROPE=1` 后使用；默认关。

它只接管 fp16/fp32、batch=1、q_len=1、偶数 head dim 等被守卫的形状。bf16 因 1 ULP 约 8e-3 过不了固定的 1e-3 K1，回退 HF；不放宽门。第一轮 A/B 金丝雀漂移 5.923% 整轮 VOID，第二轮本地 `results/fused_rope_ab.json` 为 measured：wall 均值 **1.9655→1.8007 ms，+8.39%**，busy 约少 0.194 ms，token 序列一致。但该有效轮 3/4 run 的 `host_share>5%`，应同时说出共享 host 噪声限制；详见 [spec Report](../compose/spec/fused-rope.md)。它**尚未采纳为默认路径**。

读代码时不要把 [`rope.warm`](../../src/lemoq/kernels/rope.py) 的默认 `kv_heads=4` 当成模型配置：那只是独立预热示例的默认参数；[MODELS](../reference/MODELS.md) 核实的 OLMoE 为 q/kv **16/16**，实际前向使用模型张量的真实 shape。

## 3. routing：为什么选择 K 轮 argmax

原 [`moe_static._route`](../../src/lemoq/kernels/moe_static.py) 做 `F.linear → softmax(fp32) → topk`。候选 [`_routing_triton.py`](../../src/lemoq/kernels/_routing_triton.py) 只融合 softmax 与 top-K，router 的线性层仍是 ATen。一个 program 处理一个 token，`E=64,K=8` 静态；用 K 轮 argmax、选中项置为极小值，不使用 `tl.sort`。`E_POW/K_POW` 是为了 `tl.arange` 的 2 的幂要求，mask 排除无效专家。平局时与 `torch.topk` 的 index 顺序可能不同，因此测试检查非平局精确 id、平局集合及最终输出。

`LEMOQ_FUSED_ROUTE=1` 是 Python 相位的候选开关，默认关，capture 时烘焙。独立本地工件 `results/routing_ab.json` 同进程 A/B：**1.9521→1.8935 ms，+3.0%**，金丝雀 2.215%，token 序列一致；见 [routing spec](../compose/spec/routing-fusion.md)。与 RoPE 一样，不能把它接到已录矩阵结果上计算“新版总加速”。

## 4. Mac 练习

1. 对 `x=[3,4]、weight=[2,1]、eps=0` 手算 RMSNorm；再说明“先舍入再乘 weight”与“最后一次性舍入”可能不同。
2. 对 `x=[1,2,3,4]` 手算 `rotate_half(x)=[-3,-4,1,2]`；在 [`rope.py`](../../src/lemoq/kernels/rope.py) 中找到 `shift/sign` 如何生成它。
3. 跑 CPU 回退与开关测试：

```bash
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_rmsnorm_parity.py tests/test_fused_rope.py tests/test_routing_fusion.py
```

## 5. 自测

1. 为什么 `F.rms_norm` 的一次函数调用不保证只启动一个 CUDA kernel？
2. 为什么不能从 `neg_kernel` 单项时间估算整个 RoPE 的优化头寸？
3. 已 capture 的 graph 运行时改 `LEMOQ_FUSED_ROUTE` 会切换内部实现吗？

答案见 [09](09-answers.md#norm-与候选)。
