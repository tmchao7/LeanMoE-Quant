# EXP-p2-05-attn-kernel

> 自研 decode attention（split-KV flash-decoding，长度从设备 scalar 读）。
> 与 `EXP-p2-03`（隔离测量）、`EXP-p2-04`（模型级、换成 HF eager）是**三个层次**的证据，
> 所以第三个 EXP id。本条的靶子**来自图内**，不是隔离数字。

- **ID**: `EXP-p2-05-attn-kernel`（`ctx=1024` 与 `ctx=4096`）
- **Date**: 2026-10-06
- **Operator**: `benchmarks/runner.py`（模型级）+ `tools/bench_attn_in_graph.py`（图内臂对照），
  AutoDL RTX 5090 32GB（sm_120，170 SM，driver 595.71.05，实测带宽 1561.9 GB/s）
- **Status**: `done`
- **原始数据**:
  - `results/EXP-p2-05-attn-kernel__ctx={1024,4096}/result.json` — runner（模型级）
  - `results/EXP-p2-05-attn-kernel__ctx={1024,4096}/bench_attn_in_graph.json` — 图内交错臂对照 + K1/K2/K3/K4
  - `results/EXP-p2-05-attn-kernel__ctx=1024/bench_attn_in_graph_repeat.json` — ctx=1024 的重复运行（复现性）
  - `results/attn-stepA-confirm/EXP-p2-04-eager-attention__ctx=1024/result.json` — 上机前重测 p2-04 的图内靶子
- **对照**: `EXP-p2-04-eager-attention`（`attn=eager`，其余同）、
  `EXP-p2-02-moe-grouped-gemm`（ctx=1024，sdpa prefill）、
  `EXP-p0-04-cudagraph-static-moe`（ctx=4096，sdpa prefill）

★ **2026-10-24 F4 刷新**：KV 直写默认开后同轮重测，采用 p50 **2.1946 / 2.4326 ms**
（F4 前 2.2406/2.4779；`results/EXP-f1-refresh/` F4 轮）。
★ **2026-10-24 采纳刷新**：route+rope+mask 默认开后同轮重测，采用 p50 2.2406 / 2.4779 ms（P2j 值 2.5134/2.7483 留上文；`results/EXP-f1-refresh/`，kernel/步 377）。

## Hypothesis（跑之前写）

```text
假设：
  把 decode attention 的 q·kᵀ / softmax / w·v / mask-add 这一串小 kernel 换成一次
  split-KV flash-decoding 调用；长度从**设备内存**读（不是 shape、不是 host 变量），
  所以换长度重放仍然给出新长度的答案。grid = (num_splits, heads) 覆盖 170 个 SM。

预期结果（依据 EXP-p2-03 的隔离测量，提前写死，不许事后改）：
  attention 块 0.2121 ms/token（ctx=1024）应降到接近 arm D 的隔离值
  -> per-token 低于 3.1051 ms（ctx=1024）、3.4788 ms（ctx=4096）

反证信号：
  - 图内 per-token 不低于 p2-04；
  - 换长度重放后给出旧长度的答案（err >= 1e-2）；
  - 同路径 KL 不为 0。
```

**门在跑之前冻结**（`src/lemoq/kernels/attn_decode.py`，有测试钉住不许悄悄放宽）：

| 门 | 内容 | 阈值 |
|---|---|---|
| K1 | kernel vs **fp64 oracle** | err ≤ 基线 err × 1.5 且 `allclose(1e-3, 1e-3)` |
| K2 | graph+kernel vs 同路径 eager+kernel | `KL ≤ 1e-5`（期望恰好 0） |
| K3 | vs HF + 动态 cache | 只记录，不当门槛 |
| K4 | 长度 N 捕获 → M 重放 | `correct_err < 1e-2` **且**判别力 `stale_err ≥ 1.0` |
| D | 图内臂对照 | kernel 更快、逐轮 wins 过半、attention 那几行消失 |

## 结果

### 图内交错对照（`tools/bench_attn_in_graph.py`，两条臂各自捕获全模型 decode 图）

容量固定 `ctx + 32`（= p2-04 的 runner 口径），同一份 token 脚本驱动两条臂，
7 轮逐轮交错，`warmup=5 / timed=20`：

| | eager（HF 数学实现） | kernel（本仓库） | 比值 |
|---|---|---|---|
| ctx=1024 per-token p50 | **3.1128** ms（MAD 0.0013） | **3.0014** ms（MAD 0.0059） | **1.0371×** |
| ctx=1024 wins | 0/7 | **7/7** | |
| ctx=4096 per-token p50 | **3.4905** ms（MAD 0.0025） | **3.2317** ms（MAD 0.0010） | **1.0801×** |
| ctx=4096 wins | 0/7 | **7/7** | |
| ctx=1024 gpu busy | 3.0730 ms | 2.9548 ms | −0.118 |
| ctx=1024 kernels/call | 1054 | 990 | −64 |
| ctx=4096 gpu busy | 3.4396 ms | 3.1847 ms | −0.255 |

★ ctx=1024 独立跑了三次（三次都是 7 轮交错）：eager 3.1108 / 3.1145 / 3.1128，
kernel 2.9959 / 2.9974 / 3.0014 —— 极差 0.06% / 0.18%，远小于臂间差。

### 模型级（runner，`attn=kernel`、`moe=grouped`、cudagraph 臂）

```text
                     ctx=1024            ctx=4096
TTFT p50 (ms)         55.4534            138.1539     <- sdpa prefill（见下）
per-token p50 (ms)     3.0077             3.2356
per-token p99 (ms)     3.0083             3.2480
tok/s                 332.47             309.06
gpu busy (ms)          2.9663             3.1929
kernels/call            994                994          (p2-04: 1053)
quant attainment       50.2%              46.7%         (p2-04: 48.6% / 43.4%)
bandwidth attainment   53.0%              57.3%         (p2-04: 51.4% / 53.3%)
累计 vs 82.64 ms       27.5x              25.5x
```

对照 `EXP-p2-04` 的记录值（3.1051 / 3.4788）：`3.1051 → 3.0077`（−3.1%）、
`3.4788 → 3.2356`（−7.0%）。**§7 的验收判据满足**，但量级远小于隔离测量给的希望 —— 见下。

### 正确性（两个 ctx 全部通过）

```text
K1 kernel vs fp64 oracle : PASS   kernel err 1.507e-05 / baseline err 1.507e-05（比值 1.00 ≤ 1.5）
                                  allclose(atol=1e-3, rtol=1e-3) = True
K2 graph vs 同路径 eager  : KL = 0.0     top-1 = 1.0   seed/序列都相同（两个 ctx 都是）
K3 informational vs HF+动态 cache（runner 里的值）: KL = 1.36e-06 (ctx=1024) / 1.21e-06 (ctx=4096)，top-1 = 1.0
K4 换长度（图内）        : PASS   correct_err = 0.0e+00，stale_err = 1.5146 (ctx=1024) / 1.6914 (ctx=4096)
                          对照组（eager 臂走同一套判据）也 PASS → 判据本身是好的
未填充槽位不得被读到      : PASS   把 [filled, seq) 的 k/v 全填 NaN，输出无 NaN 且与干净输入逐位相同
```

K4 的做法（`tools/bench_attn_in_graph.py::_length_change`）：在 ctx 位置捕获，
把设备端 `filled_len` 改成 33 后重放，与**同路径 eager** 在同一位置的答案比。
两个曾经的 bug 都显式避开了：改的是**设备内存**（不是 host 变量），
并且除了 `correct_err` 还要求 `stale_err`（旧长度答案与新长度正确答案的距离 = 判别力）
≥ 1.0 —— 达不到就判 `INCONCLUSIVE`，不是 PASS。

### attention 那几行确实消失了（按 (name, occurrences) 集合差，不手挑名字）

ctx=1024（每层出现一次的行，单位 ms/token）：

```text
消失（eager 有、kernel 没有）                        新增（kernel 有、eager 没有）
  gemvNSP_kernel              0.09289  <- q·k^T       _attn_split_kernel   0.12326
  cunn_SoftMaxForwardSmem     0.03943  <- softmax      _attn_combine_kernel 0.01335
  elementwise(add, occ 16)    0.02831  <- mask add
  BUnaryFunctor(mul, occ 16)  0.01364
                             ───────                    ───────
                             0.17427                    0.13661        净省 0.03766
```

ctx=4096：消失 `gemvx::kernel` 0.27743 + `cutlass::Kernel2<wmma f16>` 0.18948
（N=ctx 时 cuBLAS 在 GEMV 与 wmma GEMM 之间换内核）+ softmax 0.04291 + mask add 0.02942
+ BUnaryFunctor 0.01313 = **0.55237**；新增 split 0.36474 + combine 0.01370 = **0.37844**；
净省 0.17393。

**留在两边的共享行**（不能算进 attention 的账）：`_gate_up_act_kernel`、
`_down_sum_kernel`、`sbtopk::gatherTopK`、`bitonicSortKVInPlace`（routing）、
`gemvx::kernel` occ 16 0.03762（**o_proj**）、`softmax_warp_forward` occ 16 0.01459
（**routing 的 softmax**）、`_out_reduce_kernel`。

> ⚠️ 这一条纠正了手稿 §2 的账：那张 `≈0.212 ms/token` 的表把 **o_proj 的 GEMV（0.0373）**
> 与 **routing softmax（0.0145）** 都算进了 attention 核心。图内真正的 attention core
> 只有 **0.1743 ms/token**（ctx=1024）。手挑名字会把这个错误固化成结论，
> 所以工具改为集合差。

### num_splits 参数敏感性（**报全部**，单轮扫描）

```text
num_splits :    1        2        4        8       11(*)     16       32      spread
ctx=1024   : 3.44917  3.22828  3.13384  3.11228  3.07846  3.07919  3.11924   1.12x
ctx=4096   : 4.74677  3.88065  3.49999  3.36150  3.34348  3.36659  3.42106   1.42x
                                                    (*) = 默认值 ceil(SM/heads) = ceil(170/16) = 11
```

默认值在两个 ctx 上都**恰好落在网格最快点** —— 这是测出来的，不是设计出来的
（`num_splits=None` 只按 SM 数推导）。扫描是单臂、顺序、单轮，只用于定位量级与
默认值的位置；结论用上面那组交错对照。分片太少时（1-2）明显变慢，说明
"grid 只覆盖 16 个 SM" 确实是延迟的成因之一，方向与 §3.1 一致。

### 带宽账（kernel 自身离墙多远）

KV 读量 = `2 × 16 层 × 16 heads × ctx × 128 × 2 B`：

```text
ctx=1024 : 134.2 MB  带宽下限 85.9 us   实测 split kernel 123.3 us   -> 1089 GB/s = 69.7% 实测带宽
ctx=4096 : 536.9 MB  带宽下限 343.7 us  实测 split kernel 364.7 us   -> 1472 GB/s = 94.2% 实测带宽
```

ctx=4096 时这个 kernel 基本已经贴住实测带宽（94%），**这就是收益不可能更大的原因**：
eager 臂读的是同样的字节，只是多付了中间张量（0.1056 MB 的 softmax 权重矩阵往返）
与 4 条而不是 1 条 kernel 的启动/收尾。

## ★ 隔离测量又错了一次（这次是**高估**，而且是同一类错误）

`EXP-p2-03` 的隔离数字暗示 attention 可以从 0.21 降到 0.03 ms/token（约 7×）。
图内实测：attention core **0.1743 → 0.1366**（ctx=1024，−0.038），
整 token **3.1128 → 3.0014**（−0.111）。差 3 倍，原因有两个，都不是"测量噪声"：

1. **靶子本身就是错的**（手稿 §2 的 0.212 表包含了 o_proj 与 routing softmax，
   真实 attention core 是 0.1743）。这与 p2-04 那次的方向相反：那次隔离测量
   **低估**了多算子臂（图补贴），这次是**高估**了可攻击面（口径把邻居算进来了）。
2. **本 kernel 自己的代价不能忽略**：0.1366 ms/token 里有 0.1233 是 split kernel，
   而它在 ctx=4096 已达实测带宽的 94%。"降到接近 arm D 的隔离值"这个预期
   在图内不可能成立 —— arm D 的那个 0.03265 是 fp16 权重矩阵 + 单 SM 的
   理想化算例，不是这个卡上读 134 MB KV 的物理下限。

一句话：**图内靶子必须在图内建（§7 A），而且"核心"的边界要用集合差定，不能手挑。**

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
| `ctx=1024` | 3.0077 | 2.5134 | -16.4% | **2.5134** | PASS → **PASS** | 采用刷新值（图内臂，host 2.0%） |
| `ctx=4096` | 3.2356 | 2.7483 | -15.1% | **2.7483** | PASS → **PASS** | 采用刷新值（图内臂，host 1.9%） |

**结论**：图内臂的绝对延迟已按新实现更新（上表"采用"列）；host 主导格子的 pre-P2i 值保留，
其**结论不受影响** —— 融合 norm 省下的是每步 ~0.34–0.48 ms 的常数，在 8–272 ms 的 host 受限
步长里低于噪声（这一点 `EXP-p2d` 已经量过：同一个 kernel 改进在 eager 臂里完全看不见）。
把 host 主导的 18 格在**安静的机器**上重跑，列为 **P2k**。

## Verdict

```text
verdict:    PASS
outcome:    supported
notes:
  假设成立：图内 per-token 3.0077 ms (ctx=1024) / 3.2356 ms (ctx=4096)，
  低于 p2-04 的 3.1051 / 3.4788（§7 的验收判据），且 attention 的五条行消失、
  换成 _attn_split_kernel + _attn_combine_kernel；K1/K2/K4 与"未填充槽位不被读"
  四条门全部通过，K2 的 KL 恰好为 0。

  **但收益只有 1.03x / 1.08x（图内交错），不是隔离测量暗示的 7x。**
  这是本条最重要的结果：attention core 在图内只占 0.1743 ms/token，
  而自研 kernel 已经是带宽受限的（ctx=4096 达实测带宽 94%）。
  所以"消掉 attention"这件事本身的天花板就是 −0.17 ms/token（−5.6%）@ctx=1024。

  副产品：informational vs HF+动态 cache 的 KL 从 p2-04 的 7.18e-05 降到 1.36e-06
  （本 kernel 在 fp32 里累加整段 log-sum-exp），top-1 仍是 1.0。
```

## Follow-ups

1. **`BLOCK_N` 没有扫**（固定 128）。ctx=1024 时每个 split 的 chunk 是 94，
   BLOCK_N=128 意味着 26% 的槽位是掩码空转；BLOCK_N=64 值得试。
   另外可试"chunk 取整到 BLOCK_N 的倍数"来消掉这部分空转。
2. **去掉每步的 4D mask 更新**（§3.7）。本条的 mask 仍然每步 `fill_`/`zero_`，
   这是**刻意的**：eager 臂需要它，而两条臂的 host 侧开销必须相同，
   否则交错对照量到的是 host 差异。kernel 本身不读 mask（NaN 尾部测试已证）。
   去掉那两次 fill 约省 2 次 launch/token —— **未测**，留给下一条实验。
3. **TTFT 的口径**：本条的 TTFT（55.45 / 138.15 ms）是 **sdpa prefill**
   （`--attn kernel` 按模型默认实现加载，只替换 decode core），
   而 p2-04 的 58.09 / 186.45 是 **eager prefill**。两者不可直接比。
   这同时**解释了手稿 §10 的第 2 条**（ctx=4096 TTFT 涨 32% 未解释）：
   同 ctx 下 sdpa prefill 的 `p0-04` 是 140.92 ms、`p2-05` 是 138.15 ms，
   而 eager prefill 的 `p2-04` 是 186.45 ms —— 证据指向 prefill 的 attention 实现。
   但这是跨记录比较，**不是单变量对照**：要坐实需要一条 `grouped + sdpa @ ctx=4096`
   （`EXP-p2-04` 的 follow-up #1 也正缺这一条）。
4. **prefill 也是 attention 的大头**：ctx=4096 的 TTFT 是 138.15 ms，其中 sdpa
   在 q_len=4096 上不再浪费 tile，所以本 kernel 对 prefill 没有帮助。
   如果端到端（TTFT + 32 token）是目标，下一刀应该切 prefill，而不是 decode attention。
5. **隔离 harness 不打算加这条臂**：`tools/bench_decode_attention.py` 保持原样。
   理由是它给出的数字已经被证明会误导（p2-04 低估、p2-05 高估），
   加一条臂只会多一个容易被误引的数字。要图内数字就用 `tools/bench_attn_in_graph.py`。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、q_len=1、
              OLMoE-1B-7B-0924（16 heads × 128 head_dim，MHA，无 GQA）、
              transformers 4.57.6、StaticCache、cudagraph 臂、moe=grouped、
              attn=kernel、ctx ∈ {1024, 4096}、triton 3.4.0、
              实测带宽 1561.9 GB/s
不适用于：q_len > 1（kernel 明确拒绝，prefill 仍走 HF）；
          有 GQA 的模型（`num_key_value_heads != num_attention_heads` 时直接报错，
          不静默算错）；分页 / 非连续 KV；batch > 1（grid 第 0 维只有 heads）；
          sm_70（无实测；需要 `cp.async` 这条臂在矩阵里被声明）；
          非 fp16（kernel 按 fp16 存储、fp32 累加写的）
原因：
  - 收益上限由"attention core 在图内占多少"决定（0.1743 ms/token @ctx=1024），
    这与 heads 数、head_dim、KV 读量绑定，换模型要重新测；
  - 全部性能数字与 `config.attn=kernel` + `num_splits=11` + `BLOCK_N=128` 绑定；
  - K4（换长度正确）依赖 `GraphBuffers.filled_len` 这个设备 buffer 的存在，
    它由 `StaticCacheDecodeSession` 在图外每步 `fill_` —— 换执行栈就要重新建立这条链路。
```
