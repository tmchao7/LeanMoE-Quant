# EXP-p2-02-moe-grouped-gemm

> 复制本模板开始一个实验。**跑之前先填 Hypothesis，跑完之后不要修改它**——被否证是结果，
> 不是事故。

- **ID**: `EXP-p2-02-moe-grouped-gemm`（`ctx=1024`）
- **Date**: 2026-10-06
- **Operator**: 自动 runner（`benchmarks/runner.py`），AutoDL RTX 5090（与 p0-04 同一实例、同一 driver）
- **Status**: `done`
- **原始数据**:
  - `results/EXP-p2-02-moe-grouped-gemm__ctx=1024/result.json`（模型级，不入 git）
  - `results/EXP-p2-02-grouped-gemm/kernel_microbench.json`（kernel 级闸门）
  - `results/EXP-p2-02-grouped-gemm/kernel_sweep.json` / `kernel_sweep_refine.json` / `kernel_compare.json`
- **对照**: `EXP-p0-04-cudagraph-static-moe`，**同一台机器、同一个 driver** 上重测

## Hypothesis（跑之前写）

```text
假设：
  decode 时每 token 恰好 top_k 个 (token, expert) 对，expert id 只参与**指针算术**。
  所以可以写一个 grouped GEMV（Triton）：kernel 内由 id 算出权重指针，
  不再需要 `gate_up[ids]` 的 gather 拷贝。形状仍然静态，CUDA Graph 依然安全。

预期结果：
  gather 税消失（p0-04 记录估计 3.22 GB/token = 2.07 ms），
  per-token 从 5.2547 ms 降到 ~3.2 ms，量化达成率从 28.8% 升到 ~50%。

反证信号（出现即视为假设被否证）：
  - kernel 级同形状对照不快于 gather+bmm；
  - 调参后仍不快；
  - 同路径 KL / top-1 门失败（说明 grouped 公式与 gather 不等价）。
```

## ★ 结果：假设**部分成立**——方向对，量级被高估了 40%

```text
                       p0-04 (gather)     p2-02 (grouped)      delta
per-token p50 (ms)        5.2637             4.0715          -1.1922   = 1.293x
量化达成率                 28.7%              37.1%
带宽达成率                 30.3%              39.2%
gpu_busy (ms)             5.2392             4.0309          -1.2083
kernels / token            1,073                977            -96
TTFT p50 (ms)             55.721             58.637          +2.916
VRAM peak (GiB)           14.1379            14.1379          0
```

预估 ~3.2 ms，实测 **4.07 ms**。差额来自**计划里那条字节账的算法错误**，见下。

### 正确性（门通过）

```text
gate                 : PASS
kl (同路径 eager)     : 0.0            (阈值 1e-5)
top-1 agreement      : 1.0
token sequence match : True   ← 32 token 与 p0-04 逐 token 相同
seed token match     : True
informational vs HF+动态 cache: KL = 1.41795e-06
```

最后一行值得注意：p0-04 的同一项是 **8.33e-05**，grouped 路径把它降了 **59 倍**。
原因是 grouped 路径在 fp32 里累加、并把 top_k 维一起吃进 down kernel，
而 gather 路径每步都落回 fp16。**它更准**，所以离 HF 更近。

## ★ 两次失败，第二次差点让整条路线被放弃

### 失败 1：默认 kernel 配置落在网格最慢的一角

第一次闸门（只测默认配置）给出：

```text
gather + bmm        0.16007 ms/层
grouped (默认配置)  0.34550 ms/层    -> 0.46x，即**慢 2.17 倍**
```

按 plan 的判据，这应当"记录 null 结果、不要接进模型、转 D-2"。
**如果只有这一个配置，整条路线就被误杀了。**

36 点网格扫描显示 spread 是 **3.17x**，赢家比 gather 基线**快 1.35x**。
原因是可算的：`down_sum` 的 grid 是 `(cdiv(hidden, BLOCK_N), SPLIT_K)`，
默认 `BLOCK_N=128 / SK=1` → **(16, 1) = 16 个 program 落在 170 个 SM 上**（9% occupancy）。

### 失败 2：两轮扫描对同一个 config 给出 29% 不同的结果

精扫里重测第一轮的赢家，得到 0.15293 ms，而第一轮是 0.11872 ms —— 同机同权重。
顺序式扫描让每个 config 继承了前一个留下的 GPU 状态（时钟、L2 残留、allocator），
于是排名里混进了顺序效应。

改成**交错执行**（每轮把所有候选各测一遍）后噪声地板是 **MAD 0.00046 ms**，
比 config 之间的差异小两个数量级：

```text
候选（7 轮交错，MAD 单位 ms）                     median    MAD     wins
refine_winner  BN32x128 w4 | down BN64x128 SK4 w8  0.12044  0.00042  6/7
sweep1_winner  BN64x128 w8 | down BN64x128 SK4 w4  0.12106  0.00029  1/7
down_only_SK4  BN64x64  w4 | down BN64x128 SK4 w8  0.12294  0.00050  0/7
default        BN64x64  w4 | down BN128x64 SK1 w4  0.38683  0.00271  0/7
gather 基线（同一交错循环内）                       0.17850  0.00163    —
```

**结论：kernel 级 1.48x，逐轮排名 6/7。**

> ★ 这两次失败的方法论价值高于结果本身：**单配置的"正确性+性能门"不是合格的门。**
> 而参数扫描之所以在计划里，只是因为 `lemoq.kernels.autotune` 要求"参数敏感性本身
> 是结果" —— 它顺手救了整条路线。

## ★ 字节账错在哪：额外流量 ≠ 额外 DRAM 流量

p0-04 记录与 approved plan 都用同一套算术：

```text
理想（每个激活专家只读一次）   1.6106 GB/token
gather 路径驱动流量           4.8318 GB/token   (读 + 写副本 + bmm 再读)
税                            3.2212 GB/token  -> 2.066 ms @1559.2 GB/s
```

实测下来这个"税"**从来不是 DRAM 成本**：

| 观测 | 数值 | 含义 |
|---|---|---|
| gather 两个 kernel 实测 | 104.6 µs/层 | 而"读+写"的 DRAM 下界是 86+43 = **129.1 µs/层** |
| 即 gather 臂的达成率 | **120.8%** | 跑得比"它自己的下界"**还快** |

只有当副本的写与 bmm 的重读**留在 L2** 时才可能。5090 的 L2 约 96 MB，
而 `gate_up` 副本一份就是 67 MB —— 100 MB 的工作集在 96 MB 的 L2 里打转，
大部分根本没到 DRAM。

**教训（值得单独记一条）**：把"多搬运的字节"直接除以 DRAM 带宽，
隐含假设是这些字节都来自 DRAM。有 L2 的机器上这个假设必须验证，
否则会得到一个**看起来有理有据、但偏大 2 倍**的收益预测。
真正的可省量只能靠**实测的两臂差**（这里 0.17850 − 0.12026 = 0.0582 ms/层），
不能靠流量账。

## 归因：gather 消失后，attention 变成最大项

同一份 per-kernel 表，逐桶对照（`benchmarks/attribution.py` 的六个桶，`unmatched = 0`）：

| bucket (ms/token) | p0-04 | p2-02 | Δ |
|---|---|---|---|
| expert gather | 1.6733 | 0.0600 | **−1.6133** |
| **custom kernel（本项目自己的）** | 0.0000 | 1.1426 | +1.1426 |
| expert gemm | 1.1793 | 0.6361 | −0.5432 |
| attention | 1.2167 | 1.2110 | −0.0058 |
| routing | 0.1128 | 0.1123 | −0.0005 |
| other（elementwise + copy） | 1.1348 | 0.9606 | −0.1742 |
| **busy 合计** | **5.2392** | **4.0309** | **−1.2083** |

对账：−1.6133 − 0.5432 − 0.1742 + 1.1426 = **−1.1881**，与实测 −1.2083 相差 0.02 ms。
这条账**闭合**了（`unmatched = 0.0000`）—— 也就是说之前那个"1.59 ms 其它"的黑箱没了。

net 节省里 **−0.1747 ms 不是专家路径**：它来自 `y * top_w` 与 `sum(dim=1)` 那两个
elementwise kernel 的消失（routing 权重直接乘进 down kernel），
以及 96 个 kernel/token 的 launch 减少。

**verdict 变化**：p0-04 是 `H-A+H-B+H-C`（dominant = gather），
p2-02 是 `H-B+H-C`（**dominant = attention, 1.2110 ms = busy 的 30%**）。

## ★ 为什么现在 attention 是明确的下一步

`ctx=4096` 那一跑（Stage A）已经给出判据：

```text
slot_ratio 4.000   total_delta +3.2143 ms   moe_growth 0.985x   attention_share 101.5%
attention      1.2167 -> 4.4794 ms  (3.68x)
expert_gather  1.6733 -> 1.6318     (0.975x)   ← MoE 路径与 ctx 无关，符合设计
expert_gemm    1.1793 -> 1.1786     (0.999x)
elementwise    0.9661 -> 0.9603     (0.994x)
```

StaticCache 每步 attend **全部** `max_cache_len` 个槽位（ctx=1024 时 1056 个），
而 decode 第 33 步实际只填了 33 个。**扫描量 100% 由槽位数决定，与已填充长度无关。**

`fmha_cutlassF_f16_aligned_64x128_rf_sm80` 每层花 **75.69 µs**（ctx=1024）/ 280 µs（ctx=4096），
而 FLOP 下界只有几微秒 —— 它是**延迟受限**的：grid 只有 `batch × heads = 16` 个 program
落在 170 个 SM 上，每个 program 沿 KV 走 65（或 258）个 block。

下一步（D-2b）：写一个 attention kernel，把循环上界改成**从设备内存读出的已填充长度**。
launch 形状保持静态（图安全），但不再扫未填充槽位。预期收益量级：ctx=1024 下最多省
~1.2 ms，ctx=4096 下最多省 ~4.4 ms —— **比去 gather 更大**。

## Model / Hardware / Runtime

```text
model:    allenai/OLMoE-1B-7B-0924 (base)
dtype:    float16        layers 16 / hidden 2048 / experts 64 / top-k 8 / ffn 1024
GPU:      NVIDIA GeForce RTX 5090, sm_120, 170 SMs, 31.36 GiB
driver:   595.71.05
测量带宽: 1561.9 GB/s (只读，3 次中位数)
Python 3.12.3 / torch 2.8.0+cu128 / transformers 4.57.6 / triton 3.4.0
```

## Configuration

```text
bit-width: fp16        execution: cudagraph        moe: grouped
ctx: 1024              batch: 1                    max_new_tokens: 32
warmup/timed: 5 / 20   kv cache: StaticCache (max_cache_len=1056)
kernel 配置（实测赢家，7 轮交错 6/7）:
  gate_up_act  BLOCK_N=32  BLOCK_K=128  SPLIT_K=1  num_warps=4
  down_sum     BLOCK_N=64  BLOCK_K=128  SPLIT_K=4  num_warps=8
```

## Verdict

```text
verdict:    PASS
outcome:    supported（方向），量级被高估（2.07 -> 1.19 ms）
notes:
  假设成立：expert id 只进指针算术后可以去掉 gather 拷贝，形状仍静态，
  同路径 KL 恰好为 0、32 token 序列与 p0-04 逐 token 相同。
  1.293x 已由模型级实测确认。

  但**收益预测错了 42%**，原因是字节账把"额外搬运的字节"当成"额外 DRAM 流量"：
  gather 臂实测跑在它自己 DRAM 下界的 120.8%，即副本大部分留在 L2。
  这条要写进 docs/METRICS.md 的口径里，否则会被下一轮重犯。
```

## Follow-ups

**按收益排序**（依据同一份 per-kernel 表 + ctx 差分）：

1. **D-2b：带设备端长度上界的 attention kernel。**
   当前最大项（ctx=1024 占 30%，ctx=4096 占 53%），且**几乎全部是未填充槽位**。
   两条实现路线：(a) 从设备内存读已填充长度作为 kernel 内的循环上界
   （launch 形状静态即可，**图安全** —— 这正是 vLLM 类 kernel 的做法）；
   (b) 多图分桶（按已填充长度捕获 K 张图，host 侧计数器分发，无同步，但要 K 份图池显存）。
   先做 (a)。
2. **D-2a：融合剩下的小 kernel。** `latency_floor` 仍是 1.0635 ms、977 个 kernel/token。
   router 的 `gemv + softmax + topk + bitonic sort` 可以合成一个 kernel；
   `gemvx` 65 次调用（attention 的 q/k/v/o 投影 + lm_head）是 [W] 里 537 MB 的 attention 权重，
   这部分是**必要**流量，但 9.21 µs/次的 mean 说明它在 latency-bound 一侧。
3. **D-1：dequant 融进同一个 grouped kernel**（`EXP-p2-01`，E2，本项目的立论）。
   现在才有意义 —— 结构已经就位，而且 W4 会把专家权重从 1.61 GB 降到 0.40 GB。
   注意 dequant 版的门有真容差，而 fp16 版的门是紧的（KL=0），所以顺序是先 fp16 后 dequant。
4. **`ctx=4096` 的 p2-02**：attention 占比 53%，是 attention kernel 收益最大的工作点。
5. `latency_floor` 的 4 µs 阈值目前是"空 kernel 量级"的估计，不是本卡实测值。
   要报这条结论前应当实测一次最小 kernel 时间。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、ctx=1024、
              OLMoE-1B-7B-0924、transformers 4.57.6 + StaticCache、
              triton 3.4.0、上面那组 kernel 配置
不适用于：batch > 1（M 变大后 grouped GEMM 才可能有意义，本次 T·top_k = 8 行）；
          其它 ctx（attention 占比会变）；其它架构；量化配置（见 EXP-p2-01）
原因：
  - "grouped 一定比 gather+bmm 快"只在**调好参数**后成立：默认配置下它慢 2.17 倍，
    同一个 kernel 的 spread 是 3.17x。跨实现引用必须带上 kernel 配置。
  - 1.293x 是**延迟**改善。它同时是"去掉了 1.61 ms 的拷贝时间"与
    "新增了 1.14 ms 的自研 kernel"，两者都是这台机器上的实测值。
  - T·top_k = 8 时每行有各自的权重矩阵，**不存在共享的 A 矩阵**，所以这是
    batched GEMV 而不是 GEMM：`tl.dot` 的 M 维没有可乘之机。
    batch > 1 时 M 变大，结论需要重测。
```
