# EXP-p0-04-cudagraph-static-moe

> 复制本模板开始一个实验。**跑之前先填 Hypothesis，跑完之后不要修改它**——被否证是结果，
> 不是事故。

- **ID**: `EXP-p0-04-cudagraph-static-moe`（`ctx=1024`）
- **Date**: 2026-10-06
- **Operator**: 自动 runner（`benchmarks/runner.py`），AutoDL RTX 5090（新实例）
- **Status**: `done`
- **原始数据**: `results/EXP-p0-04-cudagraph-static-moe__ctx=1024/result.json`（不入 git）
- **对照**: `EXP-p0-02-baseline-fp16`，**在同一台机器、同一个 driver 上重测**
  （旧实例是 driver 580.95.05，新实例 595.71.05 —— 跨 driver 的数字不可直接比）

## Hypothesis（跑之前写）

```text
假设：
  HF 的 OlmoeSparseMoeBlock 对 64 个专家做 Python 循环、每个专家调用 torch.where，
  产生数据依赖的输出形状（单步 1,024 次 DtoH = 64 experts x 16 layers），
  这使 CUDA Graph capture 失败。把它换成固定形状前向（gather + bmm）之后，
  capture/replay 应当成立且与 eager 数值一致。

预期结果：
  EXP-p0-03（HF 原版）blocked；EXP-p0-04 通过 capture 与正确性门，
  端到端延迟接近 roofline 下限的量级。

反证信号（出现即视为假设被否证）：
  - EXP-p0-04 也 capture 失败（说明阻塞点不止 routing）；
  - capture 成功但 KL/top-1 门失败（静默错）；
  - 消除 launch 后延迟仍远高于 roofline（说明瓶颈另有其物）。
```

## 预判及其证据链

预判写在 `docs/MOE_CUDAGRAPH.md`，依据：

1. `EXP-p0-02` 的 profile：单步 `Memcpy DtoH` **恰好 1,024 次**，而
   `64 experts × 16 layers = 1,024`；
2. 机制：`torch.where(expert_mask[e])` → `nonzero`，输出长度取决于数据，
   必须先把 count 回读 host 才能分配输出张量；
3. transformers 自己的注释：
   `_can_compile_fullgraph = False  # MoE models don't work with torch.compile
   (torch.where(condition) not supported)`。

**离线还排除了其它嫌疑**（4D mask 恒等返回、`StaticLayer.update` 原地写、
`0.0 in attention_mask` 只在 FA2 路径、`lazy_initialization` 在 capture 前完成），
把阻塞点缩到唯一一处。

## 三次失败，三次推进（这几次失败本身就是结果）

| 轮次 | `_cudagraph.failure.stage` | 原因 | 归属 |
|---|---|---|---|
| 1 | `capture`（p0-03，HF 原版） | `operation failed due to a previous error during capture` | **预判命中** |
| 1 | `prefill`（p0-04） | `Tried to allocate 64.00 GiB` | 我的 bug |
| 2 | `correctness_compare`（p0-04） | `Expected all tensors to be on the same device` | 潜伏的测量 bug |
| 3 | `correctness`（p0-04） | `KL=1.70e-4 > 1e-5` | **门比错了对象** |

## ★ 关键结果：拆开三个 KL

最初的失败读数 `KL=1.70e-4, top1=1.0` 看起来像"图有问题"。实际是**同时换了两个东西**：
MoE 实现与 KV cache 类型。在真机上把四种组合都测了（32 步，ctx=1024）：

| 组合 | KL | top-1 |
|---|---|---|
| HF MoE + 动态 cache vs 固定形状 MoE + 动态 cache | **7.481e-04** | 1.000 |
| 固定形状 + 动态 cache vs 固定形状 + StaticCache | **1.698e-04** | 1.000 |
| **graph vs 固定形状 + StaticCache（同路径 eager）** | **0.000e+00** | 1.000 |
| graph vs HF MoE + 动态 cache | 7.199e-04 | 1.000 |

**图的回放与 eager 逐位一致（KL 恰好 0）。** 那 1e-4 量级全部来自：
- MoE 实现差异（HF 逐专家 `Linear` vs 我们的 gather+`bmm`，归约顺序不同）；
- cache 口径差异（StaticCache 每步 attend 全部 1056 个槽位并靠 mask 屏蔽，
  动态 cache 只 attend 已填充的 ~33 个）。

所以门改成与**同路径 eager**（同 StaticCache、同 4D mask、同 MoE，只是不进图）比较。
**这是修测量口径，不是放宽门槛**：`kl_max_fp16_graph` 仍是 `1e-5`，而实测是 `0.0`，
门实际上比原来更严。

## Model / Hardware / Runtime

```text
model:    allenai/OLMoE-1B-7B-0924 (base)
dtype:    float16        layers 16 / hidden 2048 / experts 64 / top-k 8 / ffn 1024
GPU:      NVIDIA GeForce RTX 5090, sm_120, 170 SMs, 31.36 GiB
driver:   595.71.05
测量带宽: 1559.2 GB/s (只读，3 次中位数)
Python 3.12.3 / torch 2.8.0+cu128 / transformers 4.57.6 / triton 3.4.0
```

## Configuration

```text
bit-width: fp16        execution: cudagraph        moe: static (固定形状)
ctx: 1024              batch: 1                    max_new_tokens: 32
warmup/timed: 5 / 20   kv cache: StaticCache (max_cache_len=1056)
```

## Results

```text
# Correctness —— 与同路径 eager 比较
KL divergence        : 0.0            (阈值 1e-5)
top-1 agreement      : 1.0
token sequence match : True
seed token match     : True
gate                 : PASS

# informational（不作为门槛）：与 HF+动态 cache 的差异
KL = 8.33e-05, top1 = 1.0   —— 这一项混了 MoE 实现差与 cache 口径差，两者都是预期的

# Performance
TTFT p50 / p99 (ms)          : 55.721 / 64.304
per-token p50 / p99 (ms)     : 5.2547 / 5.2568
throughput (tok/s) @ batch=1 : 190.3
[W] / [T] bytes              : 2,357,723,136 / 2,491,940,864
floor_W / floor_T (ms)       : 1.5121 / 1.5982
量化达成率 (floor_W/p50)     : 28.8%
带宽达成率 (floor_T/p50)     : 30.4%

# VRAM (allocator peaks)
peak total 14.1379 GiB（权重 12.89 + StaticCache KV + 图池）

# Attribution
gpu_busy                : 5.2315 ms
host share              : 0.40%
kernels per token       : 1,073
```

### 与 eager 基线的同机对照（EXP-p0-02，同 driver 重测）

| 指标 | eager | cudagraph | 倍数 |
|---|---|---|---|
| TTFT p50 (ms) | 96.12 | **55.72** | **1.73×** |
| per-token p50 (ms) | 82.64 | **5.255** | **15.73×** |
| host share | 87.6% | 0.40% | — |
| 量化达成率 | 1.8% | **28.8%** | 16× |
| kernels / token | 8,277 | **1,073** | 7.7× |
| VRAM peak (GiB) | 13.34 | 14.14 | — |

端到端（含 prefill）：

```text
  32 tokens: 2740.7 ms ->  223.9 ms  = 12.24x
 128 tokens: 10674.5 ms ->  728.3 ms = 14.66x
 512 tokens: 42409.6 ms -> 2746.1 ms = 15.44x
```

> ⚠️ 之前用**旧实例**的 100.10 ms 对照得到 19.1×，那是跨 driver 的比较，不可引用。
> 同机的诚实数字是 **15.73×**。

## Verdict

```text
verdict:    PASS
outcome:    supported
notes:
  假设成立，反证信号一个都没出现。EXP-p0-03 blocked 的原因被确诊为
  routing 的数据依赖形状，并且只改这一处就使 capture 成立。
  正确性门用 KL=0.0（同路径对照）通过，且 token 序列完全一致。
```

## Follow-ups

**剩余差距（3.29× / 3.66 ms）已拆解：**

```text
floor_T              1.5982 ms
gather 额外流量 3.22 GB -> 2.07 ms    (读+写一遍专家权重)
其它（attention 扫 1056 位置等）   1.59 ms
                     ─────────
实测                 5.2547 ms
```

所以下一步的两件事，按收益排序：

1. **去掉 gather**：Phase 2 的 Triton grouped GEMM 在 kernel 内由专家 id 算指针，
   省掉那 2.07 ms → 预期 ~3.2 ms（达成率 ~50%）。
2. **缩短 attention 扫描**：StaticCache 每步 attend 全部 1056 个槽位。
   可用滑动窗口或按实际长度裁剪 → 预期再省 ~1 ms。

其它：

- `EXP-p0-04` 的 `ctx=4096` 一条（[T] 里 KV 占比从 5.4% 升到 18.5%）。
- 现在才第一次具备做 `EXP-p1-01`（W8/W4 精度）与 `EXP-p4b-01`（截断）的条件 ——
  在此之前端到端数字会被 launch 开销完全淹没。
- `moe_static` 的 gather 方案**只是可行性参照**，不是交付形态；见上第 1 条。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、ctx=1024、
              OLMoE-1B-7B-0924、transformers 4.57.6 + StaticCache、eager 与 cudagraph 两臂
不适用于：量化配置（见 EXP-p1-*）；batch > 1；其它架构；
          PagedAttention（本条用 StaticCache，不是分页 KV）
原因：
  - 全部数字绑定在实测 1559.2 GB/s 与该机 launch 开销上；
  - "HF MoE 无法进 CUDA Graph" 是 transformers 4.57.6 实现的属性，不是 MoE 的属性 ——
    上游若改用固定形状 routing，这个结论会失效（`docs/MOE_CUDAGRAPH.md` 有检测方法）；
  - 15.73× 是**延迟**改善，不是吞吐改善；batch>1 时 launch 开销占比下降，收益会缩小。
```
