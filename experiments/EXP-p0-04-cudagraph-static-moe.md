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

★ **2026-10-24 采纳刷新**：同轮重测采用 p50 **4.6936（1024）/ 7.9057（4096）ms**（旧采用值 5.2637@1024 留上文；`results/EXP-f1-refresh/`，kernel/步 587）。

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

预判写在 `docs/reference/MOE_CUDAGRAPH.md`，依据：

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

## ★ P2j 刷新（2026-10-08）：在融合 RMSNorm 实现下重测本条的格子

> **为什么**：`EXP-p4c-02` §5.6 把 decode 的 `OlmoeRMSNorm`（每个 norm 7 个 kernel）换成自研
> 融合 kernel（图内全步 **1.2358×**），P2i 又把它接成 runner 的**默认**实现（`--rmsnorm fused`）。
> 按硬规则 2/7：门要在新实现下重测（已做，见 `EXP-p1-04` 的 v2 小节），**绝对延迟也要重测**。
>
> **协议**：改前 payload 逐格归档到 `results/EXP-p2j-refresh/archive/`；每格前后记占用快照；
> 首尾同格金丝雀（`w8a16×cudagraph×ctx=1024`：**3.0084 → 3.0124 ms，漂移 +0.13%** ⇒ 轮次有效）；
> payload 自检 `config.rmsnorm == "fused"`、`rms_norms_replaced == 65`（**13 格全部通过**）。
>
> ★★ **本轮把结果分成了两类，因为环境不允许全盘替换**：共享卡上全程有**外来作业**
> （另一项目的 `bench_load_30b.py` / `pareto_offload.py`，CPU 满载）。**图内臂**（host ≤5%）
> 不受影响，刷新值被采用；**host 主导的格子**（host 60–96%）的 wall-clock 被压爆 ——
> 判别实验：用**旧的** `--rmsnorm hf` 在同一环境重跑 `w8a8·eager·separate`，
> `kernels_per_call` 精确回到归档值 **24405**（`gpu_busy` 也一致 61.4 vs 61.7），
> 但 `host_p50` 从归档的 **1.52 ms** 变成 **35.2 ms**（融合版那次 206.1 ms）。
> ⇒ **host 才是变量，不是实现**。那 18 格的刷新值存到 `results/EXP-p2j-refresh/contended/`，
> 规范 payload **恢复为改前版本**（记录里采用它）。
>
> **每条实现改变的签名**：所有格子的 `kernels_per_call` 都**恰好 −455**（= 7 kernel/norm × 65 norm/步），
> `gpu_busy_ms` 一律略降 —— 这是"融合 norm 确实生效"的硬证据，与 wall-clock 无关。


| 格 | 改前 p50 (ms) | 刷新实测 (ms) | Δ | **记录采用** | gate 改前 → 采用 | 说明 |
|---|---|---|---|---|---|---|
| `ctx=1024` | 5.2637 | 14.5208 | +175.9% | **5.2637** | PASS → **PASS** | **作废**：刷新时 host 主导（68%，共租 CPU 压住 host） |
| `ctx=4096` | 8.4873 | 19.5673 | +130.5% | **8.4873** | PASS → **PASS** | **作废**：刷新时 host 主导（60%，共租 CPU 压住 host） |

**结论**：图内臂的绝对延迟已按新实现更新（上表"采用"列）；host 主导格子的 pre-P2i 值保留，
其**结论不受影响** —— 融合 norm 省下的是每步 ~0.34–0.48 ms 的常数，在 8–272 ms 的 host 受限
步长里低于噪声（这一点 `EXP-p2d` 已经量过：同一个 kernel 改进在 eager 臂里完全看不见）。
把 host 主导的 18 格在**安静的机器**上重跑，列为 **P2k**。

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
gather 额外流量 3.22 GB -> 2.07 ms    (读+写一遍专家权重)   <-- 见下方实测修正
其它（attention 扫 1056 位置等）   1.59 ms
                     ─────────
实测                 5.2547 ms
```

> ### ⚠️ 实测修正（后续实验推翻了上面两条预测）
>
> **1) `2.07 ms` 偏大约 2 倍。** 那个数字是"3.22 GB ÷ 1559.2 GB/s"，
> 隐含假设这些字节都来自 DRAM。实测 gather 两个 kernel 是 104.6 µs/层，
> 而"读+写"的 DRAM 下界是 129.1 µs/层 —— **达成率 120.8%，跑得比它自己的下界还快**。
> 副本的写与 bmm 的重读基本留在 L2（5090 约 96 MB，`gate_up` 副本 67 MB）。
> 做完 `EXP-p2-02` 后的**实测净收益是 1.293x**（5.2637 → 4.0715 ms），不是预期的 ~1.65x。
>
> **2) "其它 1.59 ms = attention 扫 1056 槽位"方向对，但推理链是错的。**
> 本条当时用"KV 读只有 138 MB = 0.089 ms"去反驳它 —— 那个反驳本身错了，
> 因为它把 attention 的成本当成**字节**问题。实测 attention kernel 是**延迟受限**的：
> `fmha_cutlassF_...` 每层 75.69 µs（ctx=1024）/ 280 µs（ctx=4096），
> 而 grid 只有 `batch × heads = 16` 个 program 落在 170 个 SM 上。
> `ctx=4096` 的差分给出了干净的判据：**槽位 4.000x，总增量 +3.21 ms，
> attention 占 101.5%，MoE 专家路径 0.985x（与 ctx 无关）**。
>
> **3) 两步的先后顺序是对的，但第 2 步比第 1 步大。** 去掉 gather 之后
> （[EXP-p2-02](EXP-p2-02-moe-grouped-gemm.md)），attention 变成最大项
> （1.2110 ms = busy 的 30%），在 ctx=4096 下是 4.4794 ms = **53%**。
> 所以 attention 才是下一件大事。

按**实测**重排的下一步：

1. **attention kernel，循环上界改成从设备内存读出的已填充长度**（D-2b）。
   launch 形状保持静态 → 图安全。ctx=1024 最多省 ~1.2 ms，**ctx=4096 最多省 ~4.4 ms**。
2. **继续减 kernel 数**（D-2a）：`latency_floor` 仍有 1.0635 ms / 977 kernels。
   router 的 `gemv + softmax + topk + bitonic sort` 可合成一个 kernel。
3. **dequant 融进 grouped kernel**（`EXP-p2-01`，E2 立论）—— 结构已就位。

其它：

- `EXP-p0-04` 的 `ctx=4096` 一条已跑（见 `results/…__ctx=4096/result.json`）：
  per-token 8.4873 ms。它是 attention 假设的判别点而非"多一个点"。
- 现在才第一次具备做 `EXP-p1-01`（W8/W4 精度）与 `EXP-p4b-01`（截断）的条件。
- `moe_static` 的 gather 方案**只是可行性参照**，已由 `EXP-p2-02` 取代。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、ctx=1024、
              OLMoE-1B-7B-0924、transformers 4.57.6 + StaticCache、eager 与 cudagraph 两臂
不适用于：量化配置（见 EXP-p1-*）；batch > 1；其它架构；
          PagedAttention（本条用 StaticCache，不是分页 KV）
原因：
  - 全部数字绑定在实测 1559.2 GB/s 与该机 launch 开销上；
  - "HF MoE 无法进 CUDA Graph" 是 transformers 4.57.6 实现的属性，不是 MoE 的属性 ——
    上游若改用固定形状 routing，这个结论会失效（`docs/reference/MOE_CUDAGRAPH.md` 有检测方法）；
  - 15.73× 是**延迟**改善，不是吞吐改善；batch>1 时 launch 开销占比下降，收益会缩小。
```
