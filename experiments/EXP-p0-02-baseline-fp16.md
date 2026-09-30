# EXP-p0-02-baseline-fp16

> 复制本模板开始一个实验。**跑之前先填 Hypothesis，跑完之后不要修改它**——被否证是结果，
> 不是事故。
>
> 本文件是 **ctx=1024** 的那一条。ctx=4096 是同一个实验的另一条记录（见 Follow-ups）。

- **ID**: `EXP-p0-02-baseline-fp16`（`ctx=1024`）
- **Date**: 2026-09-30
- **Operator**: 自动 runner（`benchmarks/runner.py`），AutoDL RTX 5090 容器
- **Status**: `done`
- **原始数据**: `results/EXP-p0-02-baseline-fp16__ctx=1024/result.json`（不入 git）

## Hypothesis（跑之前写）

```text
假设：
  1. OLMoE-1B-7B-0924 能在这张卡上按 fp16 加载并前向，峰值显存 ≈ 模型 12.89 GiB + KV。
  2. 测量管线（字节账 / 带宽 / 延迟 / 显存 / 归因）自洽：roofline 下限 <= 实测时间。
  3. 因为这是参考实现本身，KL 与 top-1 对本配置没有意义 —— 记录为 null 而不是 0。

预期结果：
  单步延迟落在"带宽下限的若干倍"这个区间内；达成率为个位数百分比。

反证信号（出现即视为假设被否证）：
  - 加载失败 / OOM / dtype 不是 fp16；
  - floor_T > 实测时间，或带宽达成率 > 100%（说明字节账或带宽错了）；
  - 实测显存与"权重字节 + KV"相差一个数量级。
```

## Model

```text
model:             allenai/OLMoE-1B-7B-0924   (base，非 Instruct)
arch:              olmoe (MoE, 64 experts, top-8)
dtype:             float16     (checkpoint 自述 bf16，按 fp16 加载)
layers:            16
hidden:            2048
num experts:       64
top-k:             8
expert FFN dim:    1024
total params:      6,919,161,856   (实测：,named_parameters() 求和)
activated params:  1,178,861,568   (不含 embedding，含 untied lm_head)
```

> **参数量差值的解释**（实测 vs 预算）：`ModelSpec.total_params()` 给 6,919,028,736，
> 实测 6,919,161,856，差 **133,120** = `16 × 4 × 2048 + 2048` ——
> 即 16 层的 `q_norm / k_norm / input_layernorm / post_attention_layernorm` 加最后的
> `model.norm`。`ModelSpec` 只统计**会随 token 流动的权重**（GEMM 的输入），
> 不统计 norm 向量。差额折合 fp16 只有 266,240 B = `[W]` 的 0.011%，不影响任何结论。

## Hardware

```text
GPU:                NVIDIA GeForce RTX 5090
GPU memory:         32607 MiB = 31.36 GiB 可用
GPU count:          1
GPU count used:     1
compute capability: sm_120        (170 SMs)
driver:             580.95.05
measured achievable bandwidth:  1559.2 GB/s (只读，3 次中位数, CoV 0.68%)
                                1526.7 GB/s (copy)
host CPU:           Intel Xeon Platinum 8470Q (容器配额 25 vCPU / 92 GiB)
```

## Runtime

```text
Python:       3.12.3
PyTorch:      2.8.0+cu128   (镜像自带，未重装)
CUDA:         driver 580.95.05 / torch build 12.8 / toolkit 12.8.93
transformers: 4.57.6
triton:       3.4.0
```

## Configuration

```text
bit-width:          fp16
group size:         -
dequant:            none（本阶段没有量化）
fusion:             none
kv cache dtype:     fp16
context length:     1024
batch size:         1
prompt / max new tokens: 合成 prompt（全部 token id = 1），32 个 decode 步
warmup / timed iterations: 5 / 20（另有 3 次 repeats 用于 GPU busy 测量）
```

## Results

```text
# Correctness
KL divergence (vs fp16):  null   <- 本配置**就是**参考实现，这个数字没有意义
top-1 agreement:          null
PPL (wikitext-2):         not measured
gate: PASS —— 但把守的是 consistency（自洽性），不是 KL（见下）
```

**可证伪的门槛（`correctness.consistency`）**：

```text
[OK] floor_T <= measured_p50                1.598 ms <= 100.10 ms
[OK] attainment_T <= 100%                   1.6%
[OK] bandwidth was measured, not assumed    1559.2 GB/s 来自 machine_probe.json
```

> 为什么不用 KL 把守：fp16 与 fp16 比必然相等。写 `kl = 0.0` 会让这一栏"看起来通过了"，
> 实际什么都没检查，而且占用了将来真门槛的位置。真正能证伪的是自洽性。

```text
# Performance
TTFT p50 / p99 (ms):             111.101 / 159.959
per-token p50 / p99 (ms) @ batch=1:  100.102 / 105.657
throughput (tok/s) @ batch=1:    9.99
bytes-per-token [W] (breakdown): attention 536,870,912 (22.8%) + router 4,194,304 (0.2%)
                                 + experts(top-k) 1,610,612,736 (68.3%) + lm_head 206,045,184 (8.7%)
                                 + embedding_row 4,096 (0.0%) = 2,357,723,136
bytes-per-token [T]:             2,491,940,864   = [W] + KV 读(1024) 134,217,728
theoretical floor_W (ms/step) = [W] ÷ measured bw = 1.5121
theoretical floor_T (ms/step) = [T] ÷ measured bw = 1.5982
quantization attainment = floor_W / measured latency = 1.5%    # 量化兑现了多少
bandwidth attainment    = floor_T / measured latency = 1.6%    # 端到端离墙多远

# VRAM (allocator peaks, split)
weights 12.89 GiB / KV 0.25 GiB / peak total 13.3417 GiB
（KV 池 @ctx=1024：2 × 16 层 × 16 kv-head × 128 head_dim × 1024 × 2 B = 0.125 GiB，
  另有 prefill 阶段的激活。分项口径见 benchmarks/vram.py）

# Attribution
gpu_busy (profiler self time):   10.49 ms/token   = 10.5%
host / launch / GPU 空闲:        89.62 ms/token   = 89.5%
kernels per token:               8,277
routing/packing share (of GPU):  44.8%
dequant share:                   -   （本阶段没有反量化）
```

### ★ 本条记录最重要的发现：瓶颈不是带宽

```text
roofline floor_T   1.598 ms
实测 wall         100.10 ms   ->  慢 62.6 倍
但 GPU 真正在跑 kernel 的时间只有 10.49 ms
=> 89.5% 的时间花在 host 发 kernel 上，而不是在读写显存
```

`torch.profiler` 的 kernel 分布证明了这一点：单步 **8,277 个 kernel**，其中
`gemvx` 449 次、`cub::DeviceSelectSweepKernel` 1024 次、`cub::DeviceReduceSingleTileKernel`
1024 次、elementwise 1024 次、`Memcpy DtoH` 1024 次；而 flash attention 只有 **16 次**。
每个 kernel 平均 GPU 时间 1.26 µs，平均 wall 时间 16.95 µs。

根因在 HF 的实现里：`OlmoeSparseMoeBlock.forward` 对 **全部 64 个专家逐个做 Python 循环**，
每次调用 `torch.where` + `index_add_`，而每 token 只激活 8 个。单独测一个 MoE 块：

```text
per call:                 8.996 ms
experts available:        64
experts activated:        8
bytes if all streamed:    805,306,368
bytes if only top-k:      100,663,296    (比值 8.0x)
```

**对本项目论点的直接后果**（PLAN.md 推论 4 在基线阶段就出现了）：

```text
T_bytes(fp16)   = 1.512 ms
T_bytes(W4A16)  = 0.489 ms          (3.09x 更少字节)
host            = 89.62 ms          (量化**不会**减少这一项)

只做量化        : (1.512 + 89.62) / (0.489 + 89.62) = 1.011x   (+1.1%)
只做 CUDA Graph : (1.512 + 10.49) / (0.489 + 10.49) = 1.093x   (+9.3%)
两者都做        : 3.09x                                        (真实上限)
```

即：**3.09 倍的字节缩减，若单独做，端到端只有 1.1% 的提速。**
量化必须先与"消除 launch / routing 开销"一起做，否则测不出任何东西。

## Verdict

```text
verdict:    PASS
outcome:    supported
notes:
  假设 1/2/3 全部成立，反证信号一个都没出现。基线测量管线可信：
  自洽性门槛通过，字节账与实测一致（[W] 从未被误当成 checkpoint 全量）。
  但本条**不构成任何量化结论** —— 它建立的是绝对数字与瓶颈方向。
  最重要的产出是上一条：瓶颈是 launch，不是带宽。
```

## Follow-ups

- **必做**：`EXP-p0-02` 的 `ctx=4096` 一条（KV 占比从 5.4% 升到 18.5%，看 launch 占比是否变化）。
- **必做**：CUDA Graph 消 launch 开销的对照臂（对应矩阵里的 `execution: cudagraph`）。
  在拿到这个对照之前，任何量化实验的端到端数字都无法解释。
- `EXP-p4-01`：MoE routing/packing 开销 —— 上面 `moe_block` 的量就是它的上界证据。
- `EXP-p1-01`：W8/W4 精度门槛（本条的 fp16 数字是它的参考基准）。
- **口径取消**：`verify_with_nsys` 无法满足 —— 本镜像**未安装 nsys**（只有 ncu 2025.1.1）。
  所以 launch 归因改由 `torch.profiler` 的 self_device_time 承担，并在
  `hostgap.py` 的 caveats 里写明它是"量级判断"而非第三方交叉验证。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、ctx=1024、
              allenai/OLMoE-1B-7B-0924、transformers 4.57.6 的 eager 实现
不适用于：任何量化配置（见 EXP-p1-*）；CUDA Graph 执行（见 execution 对照臂）；
          batch > 1；其它架构（sm_70/sm_89 的 launch 特性与带宽都不同）；
          其它 MoE 实现（routing 开销由**实现**决定，不由模型决定）
原因：
  - 全部数字绑定在**实测** 1559.2 GB/s 与该卡的 launch 开销上，两者都不可跨架构搬运；
  - "89.5% 时间在 host" 是 HF `OlmoeSparseMoeBlock` 的逐专家 Python 循环导致的，
    换一个实现（fused MoE kernel）这个比例会剧烈变化 —— 所以它不能当作
    "MoE 模型的固有属性"来引用；
  - ctx=1024 时 KV 只占 [T] 的 5.4%，ctx 变大后比例上升，本条结论的字节账部分不适用。
```
