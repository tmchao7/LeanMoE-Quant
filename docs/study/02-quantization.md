# 02｜W8A16 从公式、存储到端到端收益

## 1. 问题与原理

W8A16 = 权重以 8-bit **存储**、激活仍是 fp16；不等于整条前向都做 int8 矩阵乘。本项目用 per-group 对称 RTN（round-to-nearest）：每个输出行沿输入维每 `G=128` 个权重共用一个 scale。对组内权重 `w`：

```text
qmax = 127
s = max(|w|)/127            （全零组用 s=1）
q = clamp(round(w/s), -128, 127)
ŵ = q × s
```

实际量化的正负区间略不对称（`-128..127`），公式和 padding 细节以 [`quant/rtn.py`](../../src/lemoq/quant/rtn.py) 为准。对称 RTN 不存 zero；scale 以 fp16 存。W4A16 的两个有符号 int4 code 合成一个字节，低半字节放偶数列，见 [`quant/pack.py`](../../src/lemoq/quant/pack.py)。

**手算例**：为便于计算，把 8-bit 组取 `[-127, -63.5, 0, 127]`。`absmax=127`，`s=1`，码值为 `[-127,-64,0,127]`（PyTorch/NumPy 的 `round` 为 ties-to-even），重建为 `[-127,-64,0,127]`。第二个数误差 `−0.5`。这个例子只教算法，不代表模型精度。

对 `(N,K)`、`K` 可被 `G` 整除的 W8 对称权重，粗字节账是 `N·K` 个 code 字节 + `N·(K/G)·2` 个 scale 字节；fp16 是 `2·N·K`。W4 code 字节约 `N·K/2`，仍需 scale、padding 与可能的元数据。**显存减少比例不等于步延迟减少比例**：还有 attention、路由、launch 和解包成本。

## 2. 源码：两份量化实现与模型接线

[`numpy_oracle.py`](../../src/lemoq/quant/numpy_oracle.py) 是独立正确性参照，[`rtn.py`](../../src/lemoq/quant/rtn.py) 是 torch 路径；[`test_quant_oracle.py`](../../tests/test_quant_oracle.py) 比较它们并覆盖非整除输入宽度。`quant/rtn.py` 先把末组补齐，但 **padding 零不能污染非对称量化的 min/max**；返回原宽 codes 和 padded codes。

[`quant/linear.py`](../../src/lemoq/quant/linear.py) 的 `QuantLinear` 真正把 code 存在 buffer 中（W8 int8、W4 packed），scale 存 fp16。`quantize_model` 按 target 替换线性层；当前 gate 对应的 targets 是 `attention` 与 `experts`，不能擅自加 router。`QuantLinear.forward` 的两臂：

```text
separate: packed code → 每次 forward 解码、还原浮点权重 → F.linear
fused:    单 token + CUDA + 支持 dtype → _dequant_triton.dequant_gemm_fused
          其他情况返回可读 fallback 原因，随后走 separate
```

grouped MoE 专家走单独的 [`moe_grouped_quant.py`](../../src/lemoq/kernels/moe_grouped_quant.py) 与 [`_grouped_triton.py`](../../src/lemoq/kernels/_grouped_triton.py) 量化 kernel；dense 融合反量化主要服务 attention 投影等 `QuantLinear`。这两种 fused 路径不要混为一谈。W8A8 的激活实现是 `_fake_quant_activation`，用于数值影响，不意味着已经用了 int8 Tensor Core。

## 3. 为什么 fused dequant 不是无条件快

独立 kernel 少一次物化 fp16 权重，理论上省 HBM；但 [E2](../../experiments/EXP-p2-01-dequant-fusion.md) 的 eager 全模型路径被 host/launch 主导，fused 臂反而约慢 4–5%。图内修好投影 kernel 后才兑现。不要把“隔离 kernel 快”推成“端到端快”。具体 scale 修复在 [`_dequant_triton.py`](../../src/lemoq/kernels/_dequant_triton.py)：旧实现对每个权重元素 gather scale，新实现每组加载一次并广播，隔离调用从 8.14 降到 5.15 µs（523→826 GB/s）；见 [p4c-01 §5.4/§5.6](../../experiments/EXP-p4c-01-in-graph-attention.md)。

## 4. 精度门：怎么判断“能用”

[`quant_gate.py`](../../benchmarks/quant_gate.py)、[`runner.GATE_CITATIONS`](../../benchmarks/runner.py)、[METRICS §1](../reference/METRICS.md) 是入口。当前 W8A16、group=128、fused RMSNorm 的 5 窗口 prompt gate：ctx=1024 的 KL **0.001361**、top-1 **0.975758**，两个阈值分别是 KL <0.01、top-1 ≥0.95；这是**本项目自定**的阈值。门测的是 teacher forcing 下 165 个位置，不能说成任意任务绝对正确。ctx=4096 要用另一个独立 gate；`w8a8@4096` 曾靠 3/165 个位置骑线 PASS，**2026-10-25 的 10 窗复测（KL 均值 0.0262>0.01）证明 5 窗 PASS 不稳健，已 de-publish**——该配置现在与 w4a16 同语义：照跑、只作 evidence、不发布延迟。这是"门样本量决定结论"的活教材。见 [p1-04 v3](../../experiments/EXP-p1-04-prompt-set-gate.md)。

补充证据：PPL 是 WikiText-2 前 8192 token，fp16 9.1985、W8A16 9.2047（+0.067%）；MMLU 是 200 题分层子集，53.50%→52.50%（−1.0 点），样本小，不能泛化成“任务精度只降 1 点”。GSM8K 基座 0/15 是 floor，已放弃该口径。出处见 [RESULTS §3.1](../../RESULTS.md)。**gate FAIL 的格子 `latency` 必须为 `null`**，不能挑测量内部的快数字写简历。

## 5. 1.19× 与 1.74× 分别是什么

图内同轮 W8A16/fp16 端到端 p50 为 **3.0124/3.5881 ms**，所以 fp16 ÷ W8A16 = **1.1911×**（ctx=1024，采用值；见 [p4b-01 §P2j](../../experiments/EXP-p4b-01-truncation.md)）。权重流占该 graph 步约 **42%**：若把这部分**理想化地变成零耗时**，保留其余 58%，上限约 `1/0.58 = 1.724`，记录按其更精确份额报 **约 1.74×**。因此“上限实测”准确含义是：**比例来自实测，上限由该比例推导**，不是跑出了 1.74×。4096 的权重份额约 23%，上限约 1.30×。

隔离专家 GEMV 96.3→63.5 µs = 1.52×；16 层预测节省约 0.5248 ms。旧 [NCU_EVIDENCE §4.3](../reference/NCU_EVIDENCE.md) 用当轮首个 W8A16 canary **3.0084 ms** 作分母，报 90.4%；[p4b-01 §P2j](../../experiments/EXP-p4b-01-truncation.md) 后来采用末个读数 **3.0124 ms**，差额 `3.5881−3.0124=0.5757 ms`，重算约 **91%**。稳健说法是“约九成收益可由专家 GEMV 解释”。这是跨工件版本口径的审计题，不是另一个端到端速度。

## 6. Mac 练习

先手算上述四元素 RTN 例，再执行；输出应显示 `codes=[-127,-64,0,127]`、第二个重建值为 `-64`。若不是，检查 `round` 的 ties-to-even 和输入 dtype。

```bash
PYTHONPATH=src .venv-mac/bin/python - <<'PY'
import torch
from lemoq.quant.rtn import quantize_rtn, dequantize_rtn
w = torch.tensor([[-127., -63.5, 0., 127.]])
q = quantize_rtn(w, bits=8, group_size=4)
print('scale', q['scale'].tolist())
print('codes', q['codes'].tolist())
print('restored', dequantize_rtn(q).tolist())
PY
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_quant_oracle.py tests/test_quant_linear.py tests/test_quant_group_audit.py
```

再把组宽改成 3、输入列数改成 5，预测 `padded_in` 和 `n_groups` 后运行。**不要**用这次 CPU 执行时间推断 5090 性能。

## 7. 自测

1. W8A16 为什么不能直接用 `fp16 bytes / int8 bytes = 2` 宣称两倍端到端收益？
2. `QuantLinear` 的 fused 与 grouped quant kernel 分别处理什么权重？
3. `w8a8@4096` 的 gate 判定为什么随窗口数翻转（5 窗 PASS → 10 窗 FAIL）？这说明了什么？
4. 42% 对应的 1.74× 上限是直接测出的速度吗？

答案见 [09](09-answers.md#量化)。
