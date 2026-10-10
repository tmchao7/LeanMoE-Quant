# EXP-<phase>-<nn>-<slug>

> 复制本模板开始一个实验。**跑之前先填 Hypothesis，跑完之后不要修改它**——被否证是结果，
> 不是事故。

- **ID**: `EXP-p0-00-example`
- **Date**: YYYY-MM-DD
- **Operator**: （谁跑的 / 哪台机器）
- **Status**: `planned` / `running` / `done` / `blocked`

## Hypothesis（跑之前写）

```text
假设：
预期结果：
反证信号（出现即视为假设被否证）：
```

## Model

```text
model:
arch:
dtype:            float16        # V100: 一律 fp16
layers:
hidden:
num experts:
top-k:
expert FFN dim:
total params:
activated params:
```

## Hardware

```text
GPU:                Tesla V100 (SXM2 / PCIe)
GPU memory:         32 GB
GPU count:          15 (physical)
GPU count used:     1
measured achievable bandwidth:  <GB/s>    # from tools/probe_machine.py
```

## Runtime

```text
Python:
PyTorch:
CUDA (driver + torch build):
transformers:
triton:
```

## Configuration

```text
bit-width:          fp16 / W8A8 / W8A16 / W4A16
group size:         - / 32 / 64 / 128
dequant:            fused-into-gemm / separate-kernel / library
fusion:             none
kv cache dtype:     fp16 / int8
context length:
batch size:
prompt / max new tokens:
warmup / timed iterations:
```

## Results

```text
# Correctness (must pass the gate before any performance number is reported)
KL divergence (vs fp16):
top-1 agreement:
PPL (wikitext-2):      fp16 = _, this = _, delta = _

# Performance
TTFT p50 / p99 (ms):
per-token p50 / p99 (ms) @ batch=1:
throughput (tok/s) @ batch ∈ {1,2,4,8}:
bytes-per-token [W] (breakdown):  attention _ + router _ + experts+scale _ + shared _ + lm_head _ = _
bytes-per-token [T]:              [W] + KV 读(ctx) = _
theoretical floor_W (ms/step) = [W] ÷ measured bw =
theoretical floor_T (ms/step) = [T] ÷ measured bw =
quantization attainment = floor_W / measured latency = _%    # 量化兑现了多少
bandwidth attainment    = floor_T / measured latency = _%    # 端到端离墙多远（口径见 docs/reference/METRICS.md §3.2）

# VRAM (allocator peaks, split)
weights _ GB / KV _ GB / activations _ GB / workspace _ GB / peak total _ GB

# Attribution
dequant share of step time:  _%
routing/packing share:       _%
```

## Verdict

```text
verdict:    PASS / CASCADE / FAIL / planned
outcome:    supported / refuted / inconclusive
notes:
```

## Follow-ups

- 下一步要做的实验：
- 与其它实验的差异（必须显式说明）：

## Applicability

```text
本结论适用于：V100 32GB (sm_70)、fp16、单卡、batch ___、context ___、模型 ___
不适用于：Ampere/Hopper/Blackwell；bf16；batch > ___；其它模型族
原因（尤其是架构特性差异）：
```
