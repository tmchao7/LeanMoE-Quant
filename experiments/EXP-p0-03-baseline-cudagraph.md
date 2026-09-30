# EXP-p0-03-baseline-cudagraph

> This record is committed before the first server run. It is an experiment
> plan/record, not evidence that capture or replay has succeeded.

- **ID**: `EXP-p0-03-baseline-cudagraph`
- **Date**: 2026-09-30
- **Operator**: Codex; target machine is the rented RTX 5090 server
- **Status**: `planned`

## Hypothesis

```text
假设：在 batch=1、q_len=1、ctx=1024、fp16 下，Transformers StaticCache 的连续
      KV 存储可以使 OLMoE 单步 decode 完整 capture/replay 为一张 CUDA Graph。
预期结果：capture/replay 成功，且与同一 prompt 的 eager fp16 reference 具有
          top-1 agreement = 1.0、KL <= 1e-5。
反证信号：prefill、capture、replay 或 graph/eager 正确性门失败；失败必须记录具体阶段，
          不填充性能数字。
```

## Model

```text
model:             allenai/OLMoE-1B-7B-0924
arch:              olmoe
dtype:             float16
layers:            16
hidden:            2048
num experts:       64
top-k:             8
expert FFN dim:    1024
total params:      to be filled from result.json
activated params:  to be filled from result.json
```

## Hardware

```text
GPU:                NVIDIA GeForce RTX 5090 (sm_120)
GPU memory:         32 GB
GPU count used:     1
measured achievable bandwidth: to be filled from results/machine_probe.json
```

## Runtime

```text
Python:             to be filled from result.json
PyTorch:            to be filled from result.json
CUDA (driver + torch build): to be filled from result.json
transformers:       to be filled from result.json
triton:             to be filled from result.json
```

## Configuration

```text
bit-width:          fp16
group size:         -
dequant:            none
fusion:             none
execution:          cudagraph
KV cache:           Transformers StaticCache, continuous preallocation
StaticCache max length: ctx + max_new_tokens = 1056
context length:     1024
batch size:         1
prompt / max new tokens: synthetic token id 1 / 32
warmup / timed iterations: 5 / 20
```

## Results

```text
# Correctness (must pass before performance is reported)
KL divergence (graph vs eager fp16): planned
top-1 agreement: planned
correctness gate: planned (KL <= 1e-5 and top-1 = 1.0)

# Capture lifecycle
prefill: planned
capture: planned
replay: planned
failure stage/message: —

# Performance — only fill when correctness gate passes
TTFT p50 / p99 (ms): planned
per-token p50 / p99 (ms) @ batch=1: planned
throughput (tok/s): planned
CUDA Graph replay GPU busy / host share: planned

# Bytes and roofline
bytes-per-token [W]: planned
bytes-per-token [T]: planned
theoretical floor_T (ms/step): planned
bandwidth attainment: planned

# VRAM (allocator peaks, split)
weights / KV / activations / workspace / peak total: planned
```

## Verdict

```text
verdict:    planned
outcome:    inconclusive
notes:      No Mac result is reportable. This record is completed only from the RTX 5090 run.
```

## Follow-ups

- If blocked by dynamic HF MoE routing, preserve the blocked result and design the Triton fixed-metadata/grouped-MoE follow-up.
- Do not start W8/W4 or ctx=4096 from this record.

## Applicability

```text
本结论适用于：planned; intended for RTX 5090 (sm_120)、fp16、单卡、batch=1、ctx=1024、OLMoE-1B-7B-0924
不适用于：PagedAttention、多请求 serving、量化配置、ctx=4096、其它 GPU
原因：本实验只验证连续 StaticCache 与固定 q_len=1 CUDA Graph feasibility。
```
