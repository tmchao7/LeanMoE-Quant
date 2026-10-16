# 04｜dense 反量化与 split-KV attention

## 1. 问题与原理：dense 权重为什么要“边读边反量化”

W8 权重 code 每元素 1 B、scale 每组一份；若每次 forward 先重建完整 fp16 权重，再调用 `F.linear`，会读写一份中间大矩阵。对 decode 的 `M=1` 投影，把 `code × scale` 放进 GEMV 的加载路径，可避免物化。与专家量化不同，这条路径由 [`QuantLinear._fused_forward`](../../src/lemoq/quant/linear.py) 分派给 [`_dequant_triton.py`](../../src/lemoq/kernels/_dequant_triton.py)。

## 2. dense kernel 的 tile 与访存

输入 `x:(M,K)`、权重 code `q:(N,P)`、scale `s:(N,ceil(P/G))`，输出 `y:(M,N)`。主用场景是 `M=1`。[`_dequant_gemm_kernel`](../../src/lemoq/kernels/_dequant_triton.py) 的 `program_id(0)` 选 `BLOCK_N` 个输出行；每次取 `BLOCK_K` 个输入列：

```text
code 地址 = q + n·P + k
scale 地址 = s + n·n_groups + floor(k/G)
acc[n] += Σ_k float(code[n,k] × scale[n,group(k)]) × float(x[k])
```

`n<N`、`k<P`、`k<K` 三种边界分别保护输出行、packed/padded code、原始输入。W4 用低/高 nibble 拆码，`((nibble ^ 8)-8)` 做有符号扩展。对称量化不读 zero；非对称路径按 `(code-zero)×scale`。输出 fp16/fp32 的转换由 `OUT_FP16` 决定。

关键修复是 **scale 每组加载一次后广播**：当 `BLOCK_K` 包含若干完整组，加载 `(BLOCK_N, BLOCK_K/G)` 个 scale；当一组跨多个 tile，每行只加载一个。旧版逐元素 gather scale 的调用耗时 8.14 µs，修复后 5.15 µs，必要字节除以耗时对应 523→826 GB/s，见 [p4c-01 §5.4/§5.6](../../experiments/EXP-p4c-01-in-graph-attention.md)。这不是“量化算法更准”，是地址和指令成本减少。

此 kernel 不用 `tl.dot`：`M=1` 的填充会浪费权重读取，fp32 `tl.dot` 默认 tf32 也可能触及本仓库的 1e-3 parity 门。不要据此说 Tensor Core 永远无益；batch、tile 或精度约束一变，需要重新判断。

## 3. attention 为什么另写 split-KV

decode 的 `q_len=1`，HF sdpa 的通用 tile 对这个形状可能浪费。项目的 [`attn_decode.py`](../../src/lemoq/kernels/attn_decode.py) 只替换 attention core：q/k/v 投影、q/k norm、RoPE、cache update、o_proj 仍沿用原路径，便于隔离变量。core 做 `softmax(qKᵀ/√D)V`，KV 长度随步数变化，但 cache **形状和地址固定**。

[`_attn_split_kernel`](../../src/lemoq/kernels/_attn_triton.py) 的 grid 为 `(NUM_SPLITS, heads)`；一个 program 处理一个 head 的一个 KV 区间。长度来自设备端 `filled_ptr:(1,) int32`，只影响 kernel 内循环与 mask，不改变 grid。每片用在线 softmax 保存 `(m_j,l_j,acc_j)`：运行最大值、归一化分母、未归一化的加权 V。无效位置在 `max/exp` 前必须设为 `−inf`，否则会进入分母。

[`_attn_combine_kernel`](../../src/lemoq/kernels/_attn_triton.py) 的 grid 为 `(heads,)`，取全局 `M=max_j m_j`，计算：

```text
numerator   = Σ_j acc_j · exp2(m_j - M)
denominator = Σ_j   l_j · exp2(m_j - M)
out         = numerator / denominator
```

合并顺序固定，不用 `atomic_add`，避免调度顺序引入不稳定误差。空分片记录 `m=-inf,l=0`，对结果不贡献。`exp2` 的 scale 由 host 预先把自然指数底数转换好，不能漏掉 `1/ln2`。

## 4. 形状与回退边界

`attn_decode._decode_forward_impl` 只在单 token、有 `filled_len`、不请求 attention 权重时接管；prefill 和其他情况退回 HF。此实现显式拒绝 `num_key_value_groups != 1`，所以不要把它宣传为通用 GQA decode kernel。K1 需要与 oracle 的 split 方式一致；换长度 replay 需要确认输出随设备端长度变化，不能仅验证 capture 成功。真机测试在 [`test_attn_decode.py`](../../tests/test_attn_decode.py) 与 [`test_kernel_parity.py`](../../tests/test_kernel_parity.py)。

## 5. 证据：隔离和整步不是同一个数

当前采用的 [p4c-01 v3](../../experiments/EXP-p4c-01-in-graph-attention.md) 比较图内 sdpa 与 split-KV：W8A16 同格 attention 实现收益 **1.3034× @1024、1.2555× @4096**。这是同轮矩阵；不能直接拿孤立 attention 微基准的倍数替代端到端收益。投影 kernel 修复与 RMSNorm 更换产生了 v2/v3，读记录必须看版本标签。

## 6. Mac 练习

1. 设两个 split 的 `(m,l,acc)` 为 `(2,1,[1,3])` 与 `(1,2,[2,4])`，用上面的公式手算合并输出（`exp2` 口径），检查为什么直接平均两个 `acc/l` 不对。
2. 阅读 [`attn_decode.split_kv_oracle`](../../src/lemoq/kernels/attn_decode.py)，写出长度从 5 改成 7 时哪些 host 形状不变、哪些设备运算会变。
3. 跑 CPU oracle/契约测试：

```bash
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_attn_decode.py tests/test_kernel_parity.py tests/test_quant_linear.py
```

## 7. 自测

1. 为什么 `filled_len` 可变而 graph 仍可 replay？
2. 若越界 KV 的 `qk` 留成 0，会怎样污染 softmax？
3. 为什么 8.14→5.15 µs 不能写成完整模型 1.58×？

答案见 [09](09-answers.md#dense-与-attention)。
