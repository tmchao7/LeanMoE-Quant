# EXP-p4c-01-in-graph-attention

> E7 的后续（用户 2026-10-07 批准的新 EXP id）：把 attention core 换成自研 split-KV kernel，
> 看能不能把 graph 步里"拿不掉的那 55%"压下去 —— 从而提高**量化的**端到端收益。
> ★ 用新 id 而不是给 `EXP-p4b-01` 加 `attn` 轴：p4b 的 16 格已收口，不动已落盘工件。

- **ID**: `EXP-p4c-01-in-graph-attention`
- **Date**: 2026-10-07
- **Operator**: `benchmarks.runner`（GPU，8 格 + 首尾同格金丝雀）
- **Status**: `done`（**规范数字 = v3**，2026-10-08：decode 改用自研融合 RMSNorm，见 §5.7；
  v2 = 投影 kernel 修复后（§5.6），v1 = 首轮）
- **原始数据**（`results/EXP-p4c-01-in-graph-attention__bit_width=*_attn=*_ctx=*/result.json`）：
  每格另存一份带标签的拷贝：v4 = v1、v5 = v2、**v3_\<label\>_result.json = 当前规范（v3）**
  （首尾金丝雀是**同一个 cell id**，后一次会覆盖前一次，所以必须另存）；
  v2 的归档：`results/EXP-p4c-01-run/pre_p2i_hf_norm/`
- **对照**: `EXP-p4b-01-truncation`（E7：量化 × execution 轴）、`EXP-p2-05-attn-kernel`
  （点：fp16 + kernel 的图内对照；本实验是它的**量化 × kernel 的 2×2 交互**）

★ **2026-10-24 采纳刷新 = v4（canonical）**：F1+F2 默认开后的同轮重测（`results/EXP-f1-refresh/`）。量化收益：sdpa 臂 **1.1948**@1024 / **1.0912**@4096；kernel 臂 **1.4392**@1024 / **1.3711**@4096；最好格 `w8a16×kernel` = **1.5586/1.8140 ms**，vs `fp16×sdpa`（3.5344/6.7734）= **2.27× / 3.73×**（v3 值 1.1955/1.3034、1.87×/3.11× 留 §5.7，跨轮不混比）。开关作用域与 ADOPTION_SYNC §3 预告一致：route 全 MoE 格、rope/mask 限 kernel 格（kernel/步 377）、packed 仅量化 kernel 2 格（kernels/步 345）。

★★ **2026-10-24 F4 采纳刷新 = v5（canonical）**：KV 直写（`LEMOQ_KV_DIRECT_WRITE`，五 True）。
F4 证据链：GPU K 门 9/9（kernel vs HF update() 逐位，`tests/test_kv_direct_write_gpu.py`）；
独立进程 A/B **VALID**（off 1.5576 → on 1.5118 ms，**−2.94%**，5/5 对，逐对差 41–49 µs ≈ 账本
预测 50.3 µs，canary 0.154%，host ≤3.6%；`results/EXP-f4-ab-20261025/`）；刷新 23/23 OK、
canary 0.336%（`results/EXP-f1-refresh/`，F4 轮）。kernel 格 −1.8~−3.0%：fp16×kernel
**2.1960/2.4326**、w8a16×kernel **1.5116/1.7626 ms**（kernels/步 345→**329**：32 aten → 16 triton）；
kernel 杠杆 **1.4528/1.3801**、最好格 **2.36×/3.85×**。A/B 显著性：n=5 全胜 sign p=0.0625，
采纳依据=方向一致+效应量与预注册账本吻合（METRICS §2.2 口径）。

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

### 5.3 字节账与达成率（口径见 `docs/reference/METRICS.md` §3.2）

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

### 5.6 ★ v2：投影 kernel 改成"按组广播 scale"后重测（2026-10-07，kernel `7d1f92b`）

§5.4 把 64 次 `_dequant_gemm_kernel`（0.5245 ms，8.2 µs/次）列为"次值钱"的靶子。
按"先测再改"的顺序做了两件事：

**(a) 先在图里测 config（新工具 `tools/bench_dequant_gemm_graph.py`，micro-graph = 16 层 × 4 投影
= 64 次调用，逐轮交错重放）** —— 结果是"调参已经用尽"：

```text
B4K256w2s3（生产默认）  8.140 µs/次   523 GB/s（字节地板 2.72 µs = 1561.9 GB/s）
另外 7 个候选            8.62 / 10.54 / 11.49 / 13.05 / 14.04 / 18.43 / 22.30 µs/次
```
⇒ 8.14 µs 里只有 2.72 µs 是字节，**5.4 µs 是结构性的**。
（★ 这条也**否证**了 `EXP-p2-01` §6.4 那个隔离扫描的排名 —— 那次量的是 Triton launcher
（3 参数 16.0 µs / 20 参数 31.8 µs），不是 kernel。图里没有 launcher，所以排名必须重测。）

**(b) 改 kernel**：老写法 `g = ks // GROUP_SIZE` **逐元素** gather scale，
而一个 k 块里每 128 个元素共用同一个值 ⇒ BLOCK_N=4/2 warps 下每线程每个 k 块发 ~128 条 gather。
改成按组装成 `(BLOCK_N, BLOCK_K // GROUP_SIZE)` 的小片再广播。分支条件是 constexpr（图安全）、
**签名不变**（launcher 成本不变）、host 侧新增"BLOCK_K 与 group_size 必须互相整除"的校验。

**改后重测（同一工具、同一进程、交错）**：

| config | 改前 µs/次 | 改后 µs/次 | 达到带宽（改前→改后） |
|---|---|---|---|
| `B4K256w2s3`（默认） | 8.140 | **5.157** | 523 → **826 GB/s** |
| `B8K256w4s3` | 8.619 | **5.147** | 494 → 828 GB/s |

⇒ 同一个 config **快 1.58×（−36.6%）**；`B8K256w4s3` 只比默认快 0.2%（**噪声内，不换默认**）。
K1：全部候选与默认在 `PARITY_GATE`（atol/rtol 1e-3）内（`max|Δ|` ≤ 4.883e-04）；
★ 工具第一版判据写成"逐位相同"是错的 —— 换 BLOCK_K 就是换 fp32 累加顺序，重排舍入合法。

**(c) 全模型 v2 轮**（v1 已归档到 `results/EXP-p4c-01-run/v1_archive/`；canary `fp16×sdpa×1024`
4.0584 → 4.0713 ms = **+0.32%**；每格占用快照全空；9/9 exit=0、`schema problems: 0`、gate PASS）：

| 格 | v1 p50 | **v2 p50** | Δ |
|---|---|---|---|
| fp16 × sdpa × 1024 | 4.0721 | 4.0649 | −0.2% |
| fp16 × kernel × 1024 | 2.9883 | 2.9879 | −0.0% |
| w8a16 × sdpa × 1024 | 3.6716 | **3.4787** | −5.3% |
| w8a16 × kernel × 1024 | 2.5914 | **2.3913** | **−7.7%** |
| fp16 × sdpa × 4096 | 7.3275 | 7.3437 | +0.2% |
| fp16 × kernel × 4096 | 3.2243 | 3.2215 | −0.1% |
| w8a16 × sdpa × 4096 | 6.9607 | **6.7719** | −2.7% |
| w8a16 × kernel × 4096 | 2.8514 | **2.6529** | **−7.0%** |

投影桶（profiler，64 次）：**0.5245 → 0.3254 ms** —— 与 micro-graph 预测的
64 × 5.15 µs = 0.330 ms **几乎完全一致**（这也验证了工具的代表性）。
fp16 格一字未动（它们不过这个 kernel）✓。

**v2 的结论数字**：

| 指标 | v1 | **v2** |
|---|---|---|
| 量化 speedup（sdpa 臂）@1024 / @4096 | 1.109 / 1.053 | **1.169 / 1.084** |
| 量化 speedup（kernel 臂）@1024 / @4096 | 1.153 / 1.131 | **1.250 / 1.214** |
| 最好一格 vs fp16 × sdpa 基线 | 1.571× / 2.570× | **1.700× / 2.768×** |
| 量化达成率（w8a16 × kernel，[W]/p50） | 32.1% / 29.2% | **34.8% / 44.3%** |

### 5.7 ★ v3：decode 的 RMSNorm 换成自研融合 kernel 后重测（2026-10-08，P2i）

> `EXP-p4c-02` §5.6 量出：HF 的 `OlmoeRMSNorm` 每步 65 个 norm × 7 个 kernel ≈ 0.43 ms，
> 换成自研融合 kernel 后图内全步快 **1.2358×**（15/15 轮）。P2i 把它接进正式路径
> （`--rmsnorm {fused,hf}`，**默认 fused**，实现名进 payload 的 `config.rmsnorm`），
> 并按硬规则 2 在**新实现下**重测了引用门（见 `EXP-p1-04` 的 v2 小节）。
> **本节的 8 格是换实现后重测的同一批 cell id**；改前（v2，HF norm）的工件归档在
> `results/EXP-p4c-01-run/pre_p2i_hf_norm/`，本文件 §5.1/§5.6 的数字属于 v2、**不要混用**。

**协议（与 §4 同口径，另加两件事）**：首尾同格金丝雀 `fp16×sdpa×1024`；
每格前后记占用快照；**新增 payload 自检 `config.rmsnorm`**（cell id 里没有这个轴，
实现名只能靠记录）。

```text
金丝雀：canaryA 3.5909 ms  →  canaryB 3.5885 ms   漂移 −0.067%  ⇒ 本轮**有效**
host_share：8 格全部 ≤3.2%（图内臂的正常值），`rmsnorm=fused` 在 9 份 payload 里全部为真
占用：全程有**外来进程**（另一项目的 `pareto_offload.py`，3440 MiB）——按协议记录、不杀；
      金丝雀与 host_share 都证明它没有影响本轮（唯一一次污染见下）
```

| 格（bit_width × attn × ctx） | v2（HF norm） | **v3（融合 norm）** | Δ | `speedup_cudagraph`（vs 同格 fp16） | v2 的同一比值 |
|---|---|---|---|---|---|
| `w8a16 × kernel × 1024` | 2.3913 | **1.9241** | **−19.5%** | **1.3034** | 1.2495 |
| `w8a16 × sdpa × 1024` | 3.4787 | 3.0036 | −13.7% | **1.1955** | 1.1699 |
| `fp16 × kernel × 1024` | 2.9879 | 2.5079 † | −16.1% | — | — |
| `fp16 × sdpa × 1024` | 4.0713 | 3.5909 / 3.5885（金丝雀对） | −11.8% | — | — |
| `w8a16 × kernel × 4096` | 2.6529 | **2.1951** | −17.3% | **1.2555** | 1.2147 |
| `w8a16 × sdpa × 4096` | 6.7719 | 6.2751 | −7.3% | **1.0875** | 1.0843 |
| `fp16 × kernel × 4096` | 3.2215 | 2.7559 | −14.5% | — | — |
| `fp16 × sdpa × 4096` | 7.3437 | 6.8243 | −7.1% | — | — |

> † 主轮里 `fp16 × kernel × 1024` 的 payload 自检报 **`host_share=51.1%`**（p50 5.04 ms，
> 而 GPU busy 只有 2.47 ms）——按 `AGENTS.md` §6 的判据这是**被污染**的格子，
> 数字作废。同一格单独重跑两次：**2.5120 / 2.5079 ms**（host 2.2% / 2.0%，两次漂移 −0.16%），
> 表里取规范 `result.json` 的 **2.5079**（重跑日志 `v3_fp16_kernel_1024_rerun{1,2}.log`）。
> ★ 自检的价值就在这里：它把"看起来只是慢一点的 fp16 格"揪了出来，否则量化收益会被**低估 0.5 ms**。

**组合（最好的格 vs fp16 × 最差的 core 选择）**：

| ctx | 最好格 | vs `fp16 × sdpa` | v2 的同一比值 |
|---|---|---|---|
| 1024 | `w8a16 × kernel` **1.9241 ms** | **1.866×** | 1.703× |
| 4096 | `w8a16 × kernel` **2.1951 ms** | **3.109×** | 2.768× |

**达成率**（口径 `docs/reference/METRICS.md` §3.2，分母 = 实测带宽，`attainment` 字段）：
`w8a16×kernel×1024` **43.3%**（v2 44.3%）、`w8a16×kernel×4096` 37.9%、
`fp16×kernel×1024` 60.2%、`fp16×kernel×4096` 54.8%、`w8a16×sdpa×4096` 13.3%。

**机制（为什么量化侧的提升更大）**：融合 norm 省下的是**每步一个常数**（两种位宽都省
~0.34–0.48 ms，见逐格 Δ）。常数从**较小的分母**上扣，相对收益自然更大 ——
与 §5.1 那条"attention 实现是位宽无关的常数"是同一个算术。

**这一节之后，本实验的规范数字是 v3**（v1/v2 仅作对账）。

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
            ★ v2（§5.6，kernel 7d1f92b）之后：量化 speedup 升到 **1.169（sdpa）/ 1.250（kernel）**
            @1024 与 1.084 / 1.214 @4096，最好一格 **1.700× / 2.768×**；v1 的数字与全部工件
            保留在 §5.1/§5.6 与 v1_archive 里可对账。
```

## 7. Follow-ups

- **下一步最值钱**：那 35% 的 elementwise/copy 长尾（§5.4）。要先用 profiler 把它们的来源
  分类（残差相加 / RMSNorm 的 cast / 路由 index 写 / `direct_copy` 的 fp16<->fp16），
  再决定是融合进已有 kernel 还是减少中间张量。这是一条**没有矩阵条目**的新方向，
  按 `AGENTS.md` §6 必须先问用户（加 `EXP-` id 或加轴）。
- ✅ **已做（§5.6）**：attention 投影的 64 次 `_dequant_gemm_kernel`（0.5245 ms）→
  按组广播 scale 后 **0.3254 ms**（5.15 µs/次、826 GB/s）。剩下的 ~2.4 µs/次里还有 q/k/v
  合并成一次调用的空间（3 次 → 1 次），但**先要有人重跑受影响的既有实验**：
  `EXP-p2-01`（隔离表的 fused 行是 launcher 口径）与 `EXP-p4b-01` 的 w8a16 cudagraph 格
  （它们的绝对毫秒数是**改前**的）。两者都不改判定，只改量级 —— 按规则 9，重跑前先归档。
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
