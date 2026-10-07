# EXP-p4-01-moe-routing-overhead

> E4：**MoE 路由/打包开销分解** —— 它在总量里占多少、随量化怎么变、以及它是不是收益上限。
> 计划书 §7 把 E4 与 E7 并列为"两个差异化核心"：E7 回答"为什么省了字节没有等比例加速"，
> E4 回答"量化在 MoE 上的天花板在哪"。

- **ID**: `EXP-p4-01-moe-routing-overhead`
- **Date**: 2026-10-07
- **Operator**: `benchmarks.runner`（GPU；本实验无轴 → 只有 1 个 cell id，另跑 2 格同轮参照）
- **Status**: `done`
- **原始数据**：
  - 申报格：`results/EXP-p4-01-moe-routing-overhead/result.json`
    （首尾两次运行各存一份拷贝：`results/EXP-p4-01-run/p401_declared_{A,B}_result.json`）
  - 同轮参照格（属于别的实验，**只作参照**，见 §5.3）：`results/EXP-p4-01-run/p401_grouped_{fp16,w8a16}_result.json`
- **对照**: `EXP-p0-02-baseline-fp16`（同一个 hf 路径的 fp16 基线）、`EXP-p2-01-dequant-fusion`
  （量化 × hf 路径的 6 格，本记录 §5.2 的跨轮参照）、`EXP-p2-02-moe-grouped-gemm`（把 gather
  换成分组 GEMM 的那一步）、`EXP-p4c-01-in-graph-attention`（换掉 attention 之后的现役天花板）

## 1. Hypothesis（跑之前写 —— 随矩阵早已声明，本文件没有事后修改它）

```text
假设：路由/打包开销不随量化下降，构成收益上限。

预期结果：`routing_share` 在 fp16 与 W8/W4 之间基本不变（绝对毫秒数不变），
          所以权重字节即便降到 1/4，端到端也只能拿到"扣掉这一块之后"的那部分收益。

反证信号：路由开销 <5% → 上限假设不成立。
```

## 2. Model / Hardware / Runtime

```text
model:              allenai/OLMoE-1B-7B-0924（-0924，本地 snapshot）
arch:               olmoe（16 层 / hidden 2048 / 64 experts / top-8 / ffn 1024）
dtype:              float16（KV cache fp16）
GPU:                RTX 5090 32 GB (sm_120, 170 SM)，单卡；measured bandwidth 1561.9 GB/s
                    driver 595.71.05；Python 3.12.3 / torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6
```

## 3. Configuration

```text
申报格（= 矩阵里的 EXP-p4-01，它没有声明任何轴，所以只有这一格）：
    bit_width=fp16  execution=eager  moe=hf（HF 原路径：gather/scatter）  ctx=1024  batch=1
同轮参照格：
    bit_width=fp16  execution=eager  moe=grouped  ctx=1024
    bit_width=w8a16 execution=eager  moe=grouped  ctx=1024   dequant=fused  group=128
```

★ 为什么申报格是 `moe=hf`：矩阵里这条实验没有 `config.moe` → runner 的默认是"不替换 MoE"。
这正是计划书写 E4 时的那条路径（`EXP-p2-02` 之后才有了 grouped）。

## 4. 协议：本轮**有效**

| 检查 | 结果 |
|---|---|
| 首尾同格金丝雀（申报格跑两次） | p50 82.8637 → 82.0065 ms（**−1.03%**）；`busy` 10.147 → 10.146 ms（**+0.01%**）；`routing_share` 44.2% → 44.1% |
| 占用快照（每格 pre/post） | **每格都是 `2 MiB, 0–27 %`**，无任何外人进程（记录不杀） |
| exit code / schema / 门 | 4/4 `exit=0`、4/4 `schema problems: 0`、4/4 `gate=PASS` |

## 5. Results

### 5.1 申报格：路由/打包是**最大的一个桶**

`EXP-p4-01-moe-routing-overhead`（fp16 / eager / hf / ctx=1024），p50 **82.86 ms**、host **87.7%**、
`busy` **10.147 ms**，按 GPU self time 分桶：

| 桶 | ms/步 | busy 占比 |
|---|---|---|
| **`routing_pack`**（cub 的 select/reduce/compact + `indexFuncSmallIndex` + `index_elementwise`） | **4.478** | **44.2%** |
| `gemm`（`gemvx`） | 2.768 | 27.3% |
| `other` | 2.534 | 25.0% |
| `memcpy`（DtoH） | 0.349 | 3.4% |

⇒ 反证信号（<5%）**未被触发**：路由/打包是 44.2%，比 GEMM 还大。

### 5.2 它**不随量化下降**（本记录的核心测量）

同一个 hf 路径、同一个分类器，四个位宽的 `routing_pack` **绝对毫秒数**：

| 位宽 | `routing_pack` ms | `gemm` ms | `other` ms | `busy` ms | `routing_share` |
|---|---|---|---|---|---|
| fp16（本实验，本轮） | **4.478** | 2.768 | 2.534 | 10.147 | 44.2% |
| w8a16（`EXP-p2-01`，跨轮参照） | **4.512** | 3.445 | 48.747 | 57.251 | 7.9% |
| w8a8（同上） | **4.526** | 3.460 | 54.225 | 62.815 | 7.2% |
| w4a16（同上） | **4.555** | 3.392 | 139.745 | 148.759 | 3.1% |

**路由/打包在权重位宽从 fp16 降到 4-bit 的全程里只变了 1.7%（4.478 → 4.555 ms）**，
而 `[W]` 减半/再减半。原因很直接：这一块是**激活上的索引/压缩机器**（cub 的
select/reduce/compact 找 top-8 的 token 并打包专家激活），与专家权重的位宽无关。

★ 注意**份额反而下降**（44.2% → 7.9% / 7.2% / 3.1%）—— 那是**分母爆炸**造成的：
`other` 桶（hf 路径上逐专家的量化/反量化开销）从 2.534 涨到 139.7 ms。
"份额变小"在这里**不是**好消息，它正是 hf 路径吃不下量化的原因（`EXP-p2-01` 已记录：那条路径
host/launch 受限）。

★ 另外：OLMoE 的**路由权重**本身只有 4.19 MB（16 层 × 2048 × 64 × 2 B，见 payload 的
`bytes.breakdown.router`），按实测带宽折 0.0027 ms —— 所以 E4 说的"路由开销"**不是**那个
小矩阵乘法，而是**专家激活的索引/打包机器**。

### 5.3 这个上限**已经被本项目自己拆掉了一半**（同轮参照）

同轮、同 `ctx` 的 grouped 路径：

| 格 | p50 | `busy` | `routing_share`（分类器口径） |
|---|---|---|---|
| fp16 × grouped | 9.8841 ms | 3.165 ms | **0.0%** |
| w8a16 × grouped | 12.4795 ms | 2.880 ms | **0.0%** |

⭐ **这个 0.0% 是分类器的盲区，不是"路由不要钱了"**：分类器只认
`cub::` / `indexfunc` / `index_elementwise` / `write_indices` 这四个模式，
而 grouped 路径把专家选择放在**我们自己的 Triton kernel 里**（`ids_ptr` 进指针算术），
路由的 `topk` 则是 torch 算子、不在模式表里。所以：

- 可以确定的：**cub 那套 gather/scatter 机器在 grouped 路径上确实消失了**
  （同一分类器从 4.478 ms 掉到 0，busy 10.147 → 3.165 ms 里就有它的贡献）；
- **不能**确定的：grouped 路径上"路由+打包"的确切毫秒数 —— 要给它一个数，必须先把分类器
  扩到覆盖 torch 的 `topk`/`sort`/`index_select`/`gather`/`scatter` 等，并给新规则加测试。
  那是**计量口径的改动**（会改 `routing_share_pct` 这个一等字段的口径），按 `AGENTS.md` §6
  要先问用户 —— 本记录**不给**这个数。

## 6. Verdict

```text
verdict:    PASS
outcome:    supported
notes:      在申报臂（hf 路径 / fp16 / eager / ctx=1024）上，路由+打包是 GPU busy 里**最大**的
            一个桶：4.478 ms = **44.2%**（> 反证门 5% 近 9 倍）。更关键的是它**不随量化下降**：
            fp16 → w8a16 → w8a8 → w4a16 的绝对毫秒数 4.478 / 4.512 / 4.526 / 4.555（±0.9%），
            而 [W] 降了 4 倍 ⇒ 假设成立：**这一块构成量化收益的上限**，且上限与位宽无关。
            ★ 两条限定：(1) 44.2% 是 **GPU busy** 的份额，而这一格的 wall time 有 87.7% 是 host —
            所以它在**这个臂**上不是端到端瓶颈，但在 host 被消掉的 graph 臂上就是硬上限；
            (2) 计划书把"路由/打包"当成本项目的差异化核心之一，本轮实测把它**量化清楚了**，
            也同时显示**本项目自己已经把它拆掉**（grouped 路径，§5.3）——现在现役的天花板
            是 elementwise/copy 长尾与权重流（`EXP-p4c-01` §5.4）。
```

## 7. Follow-ups

- **要 grouped 路径的路由毫秒数**：先扩分类器（把 torch 的 `topk`/`sort`/`index_select`/`gather`/
  `scatter`/`index_copy` 纳入，`_act_combine`/`_out_reduce` 也算打包），并给新规则加测试 ——
  这会改 `routing_share_pct` 的口径，**要先问用户**（`AGENTS.md` §6）。
- **现役天花板**：`EXP-p4c-01` §5.4 的 elementwise/copy 长尾（~35%）与 64 次
  `_dequant_gemm_kernel` 投影调用（~5.4 µs/次的 launch 开销）。这两条都**没有矩阵条目**。
- **不要重跑**这条实验来"再确认一次"：它无轴，重跑会覆盖同一个 `result.json`；
  真要重跑请先归档并标 v1/v2（`AGENTS.md` §2 规则 9）。
- **与其它实验的差异**：`EXP-p2-02` 是"换掉 gather"的**实现**实验（1 格，fp16）；
  本实验是**归因**实验（路由占比 + 位宽不变性）。两者数字不可混比：§5.2 的
  w8a16/w8a8/w4a16 三行来自 `EXP-p2-01` 的轮次（跨轮参照，金丝雀带 ±1% 是它的误差界），
  只有 fp16 那一行是**本轮**测量。

## 8. Applicability

```text
本结论适用于：RTX 5090 32GB (sm_120)、fp16、单卡、batch=1、ctx=1024、
              OLMoE-1B-7B-0924、**HF 的 gather/scatter MoE 路径**（moe=hf）、eager 执行臂、
              measured bw 1561.9 GB/s
不适用于：grouped 路径的**路由毫秒数**（分类器盲区，§5.3 已说明不给数）；
          其它模型（路由形状 = num_experts × top_k × hidden 决定这一块的大小）；
          其它 ctx（未测）；graph 臂（本轮是 eager；graph 下 host 被消掉，这一块的**份额**
          会大得多，但绝对值仍应与位宽无关 —— 需要单独测才能引用）；sm_70/V100
原因：这一块的代价来自"每个 token 为每个专家选中/打包激活"的索引机器，与权重位宽正交；
      它的绝对大小随 experts/top_k/hidden 变化，所以倍率不可迁移，机制可迁移。
```
