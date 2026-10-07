# EXP-p4c-01-in-graph-attention

> E7 的后续（用户 2026-10-07 批准的新 EXP id）：把 attention core 换成自研 split-KV kernel，
> 看能不能把 graph 步里"拿不掉的那 55%"压下去 —— 从而提高**量化的**端到端收益。
> ★ 用新 id 而不是给 `EXP-p4b-01` 加 `attn` 轴：p4b 的 16 格已收口，不动已落盘工件。

- **ID**: `EXP-p4c-01-in-graph-attention`
- **Date**: 2026-10-07
- **Operator**: `benchmarks.runner`（GPU，8 格 + 首尾同格金丝雀）
- **Status**: `done`
- **原始数据**（`results/EXP-p4c-01-in-graph-attention__bit_width=*_attn=*_ctx=*/result.json`）：
  每格另存一份带标签的拷贝在 `results/EXP-p4c-01-run/v4_<label>_result.json`
  （首尾金丝雀是**同一个 cell id**，第二次运行会覆盖第一次，所以必须另存）
- **对照**: `EXP-p4b-01-truncation`（E7：量化 × execution 轴）、`EXP-p2-05-attn-kernel`
  （点：fp16 + kernel 的图内对照；本实验是它的**量化 × kernel 的 2×2 交互**）

## 1. Hypothesis（跑之前写 —— 已随矩阵在 `25f7868` 提交，本文件没有事后修改它）

```text
假设：§2.8.7 的实测归因显示量化 graph 步里 fmha 占 33–65%、elementwise/copy 占 12–23%，
      而权重流只有 23–42%；换掉 attention core 应当压低前者，从而提高 w8a16 的 speedup
      （1.11 → 更高），且 fp16 臂也同向变快但幅度更小（它的权重流占比更高）。

预期结果：w8a16 × cudagraph 的 speedup 从 1.11 上升；kernel 臂的 fmha 行消失。

反证信号：kernel 臂与 sdpa 臂的 speedup 差落在金丝雀带内（说明那个头寸不在 attention core），
          或 kernel 臂的 KL / 图重放门失败。
```

## 2. Model / Hardware / Runtime

```text
model:              allenai/OLMoE-1B-7B-0924（-0924，本地 snapshot）
arch:               olmoe（16 层 / hidden 2048 / 64 experts / top-8 / ffn 1024）
dtype:              float16（KV cache 也是 fp16）
total params:       6,919,028,736      activated: 1,178,861,568
GPU:                RTX 5090 32 GB (sm_120, 170 SM)，单卡
measured bandwidth: 1561.9 GB/s（tools/probe_machine.py，Gate-0）
driver:             595.71.05
Python 3.12.3 / torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6
```

## 3. Configuration（每格一份 `result.json` 的 `config` 段）

```text
bit-width:          fp16 / w8a16        group size: 128（w8a16）
dequant:            fused（量化格）；fp16 格恒为 none（不存在要融合的反量化）
execution:          cudagraph（**本实验的执行臂固定**：E7 已判定 fused+cudagraph 才是兑现路径）
moe:                grouped（16 个 block 全部替换）
attn:               sdpa（HF 默认实现的显式写法）/ kernel（本仓库 split-KV，16 个 core 全部替换）
context length:     1024 / 4096         batch: 1        max new tokens: 32
warmup / timed:     5 / 20（cudagraph 格每格 repeats=3，取 p50）
```

## 4. 协议：本轮**有效**

| 检查 | 结果 |
|---|---|
| 首尾同格金丝雀（`fp16 × sdpa × 1024`，cudagraph） | 4.0680 ms → 4.0762 ms = **+0.20%** ✓ |
| 占用快照（每格 pre/post，`nvidia-smi --query-compute-apps`） | **每格 pre 都是空**（无外人进程；不杀任何进程） |
| 每格 `host_share` 自检（>5% 即污染） | 0.7%–1.9%，全部通过 |
| exit code / schema | 8/8 `exit=0`、8/8 `schema problems: 0` |
| 正确性门（硬规则 2） | 8/8 `gate=PASS`：w8a16 格引用 `EXP-p1-04`（kl 0.001273@1024 / 0.000714@4096），fp16 格由**图重放**门把守（`graph_replay.kl = 0.0`） |

★ 为什么金丝雀必须是 **cudagraph 同格**：p4b 第 1 轮的教训是 eager 金丝雀对共租户不敏感
（只漂 1.16%，而那一轮三个 graph 格全废）。本实验全部是 graph 格，所以金丝雀就是其中一格。

## 5. Results

### 5.1 端到端 per-token p50（ms，batch=1，ctx 两档）

| bit-width | ctx | `attn=sdpa` | `attn=kernel` | 换 kernel 的增益 |
|---|---|---|---|---|
| fp16 | 1024 | 4.0721（金丝雀两次均值） | **2.9883** | **1.36×** |
| w8a16 | 1024 | 3.6716 | **2.5914** | **1.42×** |
| fp16 | 4096 | 7.3275 | **3.2243** | **2.27×** |
| w8a16 | 4096 | 6.9607 | **2.8514** | **2.44×** |

**量化的 speedup（`p50(fp16)/p50(w8a16)`，>1 = 量化更快）**

| ctx | `attn=sdpa` | `attn=kernel` | 变化 |
|---|---|---|---|
| 1024 | 1.1091 | **1.1532** | **+4.0%** |
| 4096 | 1.0527 | **1.1308** | **+7.4%** |

**两个杠杆合起来 vs 原始 fp16 基线（`fp16 × sdpa × cudagraph`）**

| ctx | 最好的一格 | 相对基线 |
|---|---|---|
| 1024 | `w8a16 × kernel` = 2.5914 ms | **1.571×** |
| 4096 | `w8a16 × kernel` = 2.8514 ms | **2.570×** |

（对照：只有量化时是 1.11×/1.05×；只有 attention kernel 时是 1.36×/2.27×。）

### 5.2 机制：attention core 那一段到底是多少

| 格 | `fmha`（sdpa） | `_attn_split_kernel`（kernel） | 比值 |
|---|---|---|---|
| fp16 × 1024 | 1.210 ms | 0.1233 ms | 9.8× |
| w8a16 × 1024 | 1.216 ms | 0.1210 ms | 10.0× |
| fp16 × 4096 | 4.478 ms | 0.3651 ms | 12.3× |
| w8a16 × 4096 | 4.477 ms | 0.3651 ms | 12.3× |

***两个 bit-width 的 fmha 完全一样***（1.210 vs 1.216；4.478 vs 4.477）—— 因为这条路径的
KV/attention 在两条臂里都是 fp16。所以被换掉的这一段是**与位宽无关的加性常数**：
@4096 省下 **4.10 ms**（fp16）与 **4.11 ms**（w8a16），几乎相等。
这解释了下一条推论（也是本记录唯一**被部分否证**的子命题）：

> 假设里写"fp16 臂也同向变快但**幅度更小**"。实测：fp16 确实变快（1.36×/2.27×），
> 但**按比例算量化臂反而赚得更多**（1.42×/2.44×），而按绝对时间两者几乎一样（4.10 vs 4.11 ms）。
> 原因就是上面那句：被减掉的是一个与位宽无关的常数 —— 总量小的那条臂（量化）
> 相对收益自然更大。**子命题方向写反了，记录在此，不改假设。**

### 5.3 字节账与达成率（口径见 `docs/METRICS.md` §3.2）

```text
[W] @w8a16/g128 = 1.3008 GB      [T] @ctx=1024 = 1.4350 GB      [T] @ctx=4096 = 1.8376 GB
[W] @fp16       = 2.3577 GB      [T] @fp16/ctx=1024 = 2.0514 GB  [T] @fp16/ctx=4096 = 2.8946 GB
measured bandwidth = 1561.9 GB/s（Gate-0 实测；不用 spec）

量化达成率 = floor_W / p50：
    w8a16 × kernel × 1024 = 0.8328 / 2.5914 = 32.1%     （只有 sdpa 时是 0.8328/3.6716 = 22.7%）
    w8a16 × kernel × 4096 = 0.8328 / 2.8514 = 29.2%
带宽达成率 = floor_T / p50：
    w8a16 × kernel × 1024 = 0.9187 / 2.5914 = 35.5%
    w8a16 × kernel × 4096 = 1.1765 / 2.8514 = 41.3%
```

### 5.4 换完之后**下一步**的瓶颈（profiler per-kernel self time，覆盖 busy 的 99.0%）

`w8a16 × kernel × 1024`（2.5914 ms，host 1.9%，994 kernel/步）逐类归并：

| 桶 | ms/步 | 占比 | 说明 |
|---|---|---|---|
| attention 投影（`_dequant_gemm_kernel` ×64） | 0.5245 | 20.6% | 64 次调用 ~8.2 µs/次，其中字节成本只有 ~2.8 µs ⇒ **launch 主导** |
| MoE GEMV（`_gate_up_act_kernel_q` 0.4446 + `_down_sum_kernel_q` 0.2589） | 0.7035 | 27.7% | 权重流，地板 0.8328 里的大头 |
| attention core（`_attn_split_kernel` ×16） | 0.1210 | 4.8% | 已经从 30.1% 掉到这里 |
| elementwise / copy / reduce 长尾 | ~0.88 | **~35%** | 66×`direct_copy` + 65×`reduce` + 64×`elementwise` + 65×`Binary` … |
| lm_head（`gemvx` ×1） | 0.1271 | 5.0% | 词表 50304 × 2048 |
| routing/index + 其它 | ~0.17 | ~7% | |

⇒ **新的头号瓶颈是那 35% 的 elementwise/copy 长尾**（几百个几微秒级的 kernel），
不是权重流、也不是 attention core。

### 5.5 VRAM（allocator 峰值，分列）

```text
w8a16 × kernel × 1024: weights 12.89 GiB / kv(net) -6.02 / activations 10.79 / workspace 0.0
                       peak total 17.67 GiB（32 GB 卡余量充足，本实验不受显存约束）
```

## 6. Verdict

```text
verdict:    PASS
outcome:    supported（主命题）；子命题（fp16 幅度更小）**部分否证**，见 §5.2
notes:      换掉 attention core 把 w8a16 的 speedup 从 1.109 抬到 1.153（ctx=1024，+4.0%）
            与 1.053 → 1.131（ctx=4096，+7.4%），都远大于 0.20% 的金丝雀带；
            两个杠杆合起来拿到 **1.57×（1024）/ 2.57×（4096）**（vs fp16 × sdpa 基线）。
            机制：被换掉的 attention 段是**与位宽无关的加性常数**（@4096 两边都省 ~4.1 ms），
            所以它压低了所有臂的总时间，而**比例上**对总量更小的量化臂更有利。
            ⇒ E7 的"截断"不只来自 host/launch：**attention 实现**是另一个大块的非权重时间。
```

## 7. Follow-ups

- **下一步最值钱**：那 35% 的 elementwise/copy 长尾（§5.4）。要先用 profiler 把它们的来源
  分类（残差相加 / RMSNorm 的 cast / 路由 index 写 / `direct_copy` 的 fp16<->fp16），
  再决定是融合进已有 kernel 还是减少中间张量。这是一条**没有矩阵条目**的新方向，
  按 `AGENTS.md` §6 必须先问用户（加 `EXP-` id 或加轴）。
- **次值钱**：attention 投影的 64 次 `_dequant_gemm_kernel` 调用（0.5245 ms，~8.2 µs/次）
  里 ~5.4 µs 是 launch 开销 —— 与 `EXP-p2-01` 的"19 参数 → 9 参数"同一类问题；
  q/k/v 三次调用合并成一次是明确可做的下一步。
- **不要做**：不需要重跑 p4b 的 16 格（sdpa 臂的数已由 `EXP-p4b-01` 覆盖，本实验的 sdpa 格
  只是同轮对照）。`w4a16` 若要用 kernel 臂，引用门仍 FAIL（只能作 evidence）。
- **与其它实验的差异**：`EXP-p2-05` 是 **fp16** 下的 attn 实现对照；本实验是
  **fp16 × w8a16 × {sdpa,kernel}** 的 2×2，回答的是"换 attention 能否改变**量化收益**"。
  两者不可混比（不同批次、不同轮）。

## 8. Applicability

```text
本结论适用于：RTX 5090 32GB (sm_120)、fp16、单卡、batch=1、ctx ∈ {1024, 4096}、
              OLMoE-1B-7B-0924（16 层/64 专家/top-8）、w8a16 group=128 + --dequant fused、
              execution=cudagraph、attn ∈ {sdpa, kernel}、measured bw 1561.9 GB/s
不适用于：其它模型族与层数（attention 占比随层数/头数变化）；其它 ctx（>4096 超出模型上限）；
          w8a8 / w4a16 / mxfp4（引用门 FAIL，只能作 evidence）；eager 执行臂（E7 已证那条路径
          是 host 受限，本结论的 1.57×/2.57× 只在 cudagraph 下成立）；V100 等 sm_70（无 cp.async，
          本仓库 split-KV kernel 的路子不可直接搬）
原因：attention 段的大小取决于 ctx 与模型结构（它是加性常数），而"换掉它提高量化收益"
      这个结论只依赖"它非负且与位宽无关"这一条，所以可迁移的部分是**机制**；
      具体倍数必须重测。
```
