# EXP-p2-04-eager-attention

> 模型级实验。与 `EXP-p2-03`（隔离测量）是**不同层次**的证据，所以不同 EXP id：
> p2-03 不加载模型，p2-04 走完整 cudagraph 臂 + 正确性门。

- **ID**: `EXP-p2-04-eager-attention`（`ctx=1024` 与 `ctx=4096`）
- **Date**: 2026-10-06
- **Operator**: `benchmarks/runner.py`，AutoDL RTX 5090（同实例、同 driver 595.71.05）
- **Status**: `done`
- **原始数据**: `results/EXP-p2-04-eager-attention__ctx={1024,4096}/result.json`
- **对照**: `EXP-p2-02-moe-grouped-gemm`（ctx=1024，同批重测）与
  `EXP-p0-04-cudagraph-static-moe`（ctx=4096，同机同 driver）

## Hypothesis（跑之前写）

```text
假设：
  OLMoE 加载后默认 `_attn_implementation == "sdpa"`，其 FMHA 内核 tile 是 64x128；
  decode 的 q_len=1 浪费 98.4% 的 tile。换成 "eager" 应当更快，且收益随 ctx 放大。

预期结果（依据 EXP-p2-03 的隔离测量，已提前写死）：
  attention 块 1.04x @ctx=1024 / 5.80x @ctx=4096
  -> ctx=4096 省 ~3.44 ms/token（8.4873 -> ~5.05 ms）
  -> ctx=1024 只省 ~0.05 ms/token（1.2%）

反证信号：
  - 模型级无收益 → 隔离测量不能外推；
  - 同路径 KL / top-1 门失败 → 换实现改变了数值。
```

## ★ 结果：方向对，**量级错了 5.7 倍**（而且是往好的方向）

```text
                     p2-02 (grouped+sdpa)   p2-04 (grouped+eager)
per-token p50 (ms)        4.0829                 3.1051          = 1.315x
量化达成率                 37.0%                  48.6%
                        p0-04 (gather+sdpa)    p2-04 (grouped+eager)
per-token p50 (ms)        8.4873                 3.4788          = 2.440x
量化达成率                 17.8%                  43.4%
带宽达成率                 21.8%                  53.3%
```

**累计相对 eager 基线**（82.64 ms/token，同机同 driver 实测）：

```text
ctx=1024 :  3.1051 ms  ->  26.6x   达成率 48.6%
ctx=4096 :  3.4788 ms  ->  23.8x   达成率 43.4%
```

### 正确性（两个 ctx 都通过）

```text
gate                 : PASS      (两组都是)
kl (同路径 eager)     : 0.0
top-1 agreement      : 1.0
token sequence match : True
seed token match     : True
config.attn          : eager     （已写进结果文件的 config）
```

### 预测错在哪：隔离测量系统性**低估**了多算子实现

`EXP-p2-03` 预测 ctx=1024 只省 0.049 ms（1.04×），**实测省 0.978 ms**。

逐桶对账（`benchmarks/attribution.py`，`unmatched = 0`，ctx=1024）：

| bucket (ms/token) | p2-02 | p2-04 | Δ |
|---|---|---|---|
| **attention（FMHA）** | 1.2110 | **0.0000** | **−1.2110** |
| custom kernel | 1.1426 | 1.1373 | −0.0053 |
| expert gather | 0.0600 | 0.0600 | −0.0001 |
| expert gemm | 0.6361 | 0.7982 | +0.1622 |
| routing | 0.1123 | 0.1516 | +0.0393 |
| other | 0.9606 | 1.0375 | +0.0770 |
| latency floor | 1.0635 | 1.1491 | +0.0856 |
| **busy 合计** | **4.0309** | **3.0604** | **−0.9705** |
| kernels/token | 977 | 1,053 | +76 |

**`fmha = 0`**（两个 ctx 都查过，不只是"掉出 top 列表"）。
FMHA 的 1.2110 ms 被 eager 的四个算子替代，它们的**全部**代价只有 ~0.212 ms：

```text
gemvNSP_kernel       0.0928 ms  occ 16  mean 5.80 us   <- q·k^T
gemvx::kernel        0.0373 ms  occ 16  mean 2.33 us   <- w·v
cunn_SoftMaxForward  0.0393 ms  occ 16  mean 2.45 us
softmax_warp_forward 0.0145 ms  occ 16
elementwise(add)     0.0282 ms  occ 16
                     ────────
                     ~0.212 ms   vs   FMHA 1.2110 ms   = 5.71x
```

对账：−1.2110 + 0.212（eager 内核）+ 0.1622 + 0.0393 + 0.0770 ≈ −0.72，加上
`latency_floor` 与 `other` 的分类重叠，与实测 −0.9705 一致（两个桶跨类别统计，不是独立相加）。

**为什么隔离测量算不出来**：`tools/bench_decode_attention.py` 把每个臂都**eager 地**
跑在紧循环里，而它们最终是被 **capture 进 CUDA Graph** 的。进图之后
每算子一次的分配与启动开销消失 —— 多算子实现拿到一笔"图补贴"，
单算子（FMHA）拿不到。所以隔离测量对**多算子臂**不具外推性。

> ★ 这条要作为工具的限制写下来：`EXP-p2-03` 里 arm B（单算子）与模型级数字吻合到
> 6-7%，让我误以为整个 harness 都能外推；**一个臂对上了不代表另一个臂也对上。**
> 单算子臂可以外推，多算子臂必须在模型级（或至少在 capture 后的图里）测。

## Verdict

```text
verdict:    PASS
outcome:    supported
notes:
  假设成立且被放大：换 attention 实现在 ctx=4096 值 2.440x（预估 1.68x），
  ctx=1024 值 1.315x（预估 1.02x）。两个 ctx 的门都通过（同路径 KL 恰好 0、
  32 token 序列与对照逐 token 相同），且 fmha 内核确认消失。

  一行配置（attn_implementation="eager"）拿到 26.6x / 23.8x 累计加速，
  达成率 48.6% / 43.4%。这是本项目至今**单位成本最低**的一次改动。

  代价要说清楚：token 序列与 p2-02 一致，但**换了实现就换了数值**
  （informational vs HF+动态 cache：p2-04 是 7.18e-05，p2-02 是 1.42e-06）。
  也就是说 p2-02 那个"更准"的额外好处在这一次被换掉了 —— 两者都远低于
  1e-5 的同路径门所关心的量级，但口径不同，引用时必须带 `config.attn`。
```

## Follow-ups

1. **`ctx=4096` 的 `p2-02`（grouped + sdpa）没跑过**，所以 3.4788 是对着
   `p0-04`（gather + sdpa）比的。要拆开 "grouped 的贡献" 与 "eager 的贡献"，
   需要一条 `grouped + sdpa @ ctx=4096`。本条的 2.440x 是**两项合并**的效果，
   不能当成其中任一项的单独收益。（ctx=1024 的 1.315x 是干净的：
   同批、同配置、只差 `attn`。）
2. **下一步是 attention kernel**（`EXP-p2-03` 已给出靶子）：split-KV
   flash-decoding，长度从设备 scalar 读。靶子 `0.02493 ms/层`（ctx=4096）——
   相对现在 eager 的 ~0.0133 ms/层 × 16 = 0.212 ms/token，注意**eager 已经比
   arm D 的隔离数字更快**（0.120 vs 0.212 ms/token）；所以 kernel 的靶子要在
   **capture 后**重新测，不能再用隔离数字。
3. `latency floor` 现在是 1.1491 ms / 1,053 kernels，已超过 attention 成为最大项。
   router 的 `gemv + softmax + topk + bitonic sort` 与 65 次 `gemvx`
   （attention 的 q/k/v/o 投影 + lm_head）是下一批融合目标。
4. `TTFT` 在 ctx=4096 是 186.451 ms（p0-04 是 140.920）—— **变慢了 32%**，
   而 per-token 快 2.44x。prefill 走的是逐专家循环，与本次改动无关，
   所以这更像是 prefill 的测量波动或显存状态差异，但**没有解释**。
   要报端到端数字前必须查清。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、q_len=1、
              OLMoE-1B-7B-0924（16 heads × 128 head_dim，MHA）、
              transformers 4.57.6、StaticCache、cudagraph 臂、
              moe=grouped、attn=eager
不适用于：q_len > 1（FMHA 的 tile 浪费消失，结论会反转 —— 这也是生产系统
          默认用 SDPA 的原因）；有 GQA 的模型（heads 更少，grid 更小）；
          分页 KV；其它 arch；非 cudagraph 臂
原因：
  - 收益的直接来源是"FMHA 的 64 行 tile 对 q_len=1 浪费 98.4%"，
    这是 q_len=1 特有的；q_len 变大后 FMHA 会反过来占优。
  - 全部数字与 `config.attn` 绑定：换实现会改数值，不能与 sdpa 的结果混引。
```
