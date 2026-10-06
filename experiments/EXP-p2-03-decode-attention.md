# EXP-p2-03-decode-attention

> 隔离测量，不是模型级实验。目的是在写 attention kernel **之前**先确定
> 它值不值得写、以及有没有更便宜的路。结论是：有一条**一行配置**的路先赢，
> 然后再谈 kernel。

- **ID**: `EXP-p2-03-decode-attention`
- **Date**: 2026-10-06
- **Operator**: `tools/bench_decode_attention.py`，AutoDL RTX 5090（与 p2-02 同一实例、同一 driver）
- **Status**: `done`（Stage 0：隔离测量。Stage 1：模型级验证，见 Follow-ups）
- **原始数据**:
  - `results/EXP-p2-03-decode-attention/attn_ctx1024.json`（`kv_len=1056, filled=33`）
  - `results/EXP-p2-03-decode-attention/attn_ctx4096.json`（`kv_len=4128, filled=1025`）
  - `kernel_attn_ctx1024.json` / `kernel_attn_ctx4096.json`（**第一版工具跑的，时序受争用污染，只用其定性部分**）

## Hypothesis（跑之前写）

```text
假设：
  EXP-p2-02 的归因表说单项最大的是 fmha_cutlassF_f16_aligned_64x128_rf_sm80
  （1.2110 ms/token，30%）。但 transformers 4.57.6 的 OlmoeAttention.forward 读起来
  是 eager 数学实现（matmul -> 加 mask -> softmax -> matmul），源码里没有 SDPA。
  两者矛盾，必须先解决；否则可能去写一个不必要的 kernel。

预期结果：
  矛盾会在"实际加载的是另一个类"或"归因错了"之间二选一。
  同时测出所有合理实现的差距，用来决定该不该写 kernel。

反证信号：
  - 现有 SDPA 路径已经接近"只读已填充槽位"的下界 → 不该写 kernel；
  - 或者裁剪路径既快又能进图 → 应该改配置而不是写 kernel。
```

## 先解决矛盾：我读错了类

```text
config._attn_implementation  : sdpa
attn 类                       : transformers.models.olmoe.modeling_olmoe.OlmoeSdpaAttention
forward qualname              : OlmoeSdpaAttention.forward
source has sdpa               : True
```

`OlmoeAttention`（eager）与 `OlmoeSdpaAttention`（SDPA）是**同一个文件里的两个类**，
模型加载的是后者。我第一版 `inspect.getsource` 读的是前者，所以一度以为
"fmha 内核不该存在"。**归因是对的，是我的读法错了。**

隔离测量再次确认：arm B（SDPA + 4D float mask）正好只有一个内核，
名字与归因表逐字相同：

```text
fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEffAttention::AttentionKernel<
    cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128, true, true>::Params)
```

**tile 是 `64x128` —— 64 个 query 行，而 decode 的 q_len 是 1。** 98.4% 的 tile 是空的。
这正是"M=1 让 tile kernel 变浪费"的同一原理，与 grouped 路径为什么是 GEMV 而不是 GEMM 同源。

## Results：交错对照（**必须交错，见下**）

`MAD` 是可分辨率的直接量度；`wins` 是 12 轮里最快的次数。

### ctx=1024，filled=33（`scan_waste = 32.0x`，MAD 地板 0.00199 ms）

| 臂 | ms/层 | 相对现路径 | wins | 换长度后还算对？ |
|---|---|---|---|---|
| D sdpa_narrow | 0.03265 | **2.17×** | 12/12 | ❌ err=10.2 |
| A **eager_math** | 0.06776 | **1.04×** | 0/12 | ✅ err=0.0045 |
| B **sdpa_float4d（现在走的）** | 0.07080 | 1.00× | 0/12 | ✅ err=0.0044 |
| F eager_narrow | 0.07377 | 0.96× | 0/12 | ❌ err=10.2 |
| C sdpa_bool4d | 0.07444 | 0.95× | 0/12 | ✅ err=0.0044 |

### ctx=4096，filled=1025（`scan_waste = 4.0x`，MAD 地板 0.00049 ms）

| 臂 | ms/层 | 相对现路径 | wins | 换长度后还算对？ |
|---|---|---|---|---|
| D sdpa_narrow | 0.02493 | **10.43×** | 12/12 | ❌ err=10.0 |
| A **eager_math** | 0.04487 | **5.80×** | 0/12 | ✅ err=0.0040 |
| F eager_narrow | 0.04826 | 5.39× | 0/12 | ❌ err=10.0 |
| B **sdpa_float4d（现在走的）** | 0.26008 | 1.00× | 0/12 | ✅ err=0.0040 |
| C sdpa_bool4d | 0.26405 | 0.98× | 0/12 | ✅ err=0.0040 |

### 交叉验证：隔离测量能复现模型级数字

```text
arm B × 16 层  =  1.133 ms（ctx=1024）  模型级 attention 桶 1.2110 ms   差 6.4%
arm B × 16 层  =  4.161 ms（ctx=4096）  模型级 attention 桶 4.4794 ms   差 7.1%
```

两个 ctx 都在 7% 以内 —— 也就是说这个隔离测量**替代得了**加载 13 GB 模型的
那一跑，这对后续调参的价值很大。

## ★ 三条结论

### 1. 一行配置就值 5.80×（ctx=4096）

`attn_implementation="eager"`（arm A）在 ctx=4096 下把 attention 块从
0.26008 降到 0.04487 ms/层：

```text
省 0.2152 ms/层 × 16 层 = 3.44 ms/token
8.4873 -> ~5.05 ms/token（1.68×）
```

ctx=1024 下只有 1.04×（省 0.048 ms/token，1.2%）—— **同一个改动在两个 ctx 上
价值差 70 倍**。所以这条要按 ctx 分别报，不能只报一个倍数。

### 2. 裁剪路径（arm D）快 10.43×，但**不能直接用**

D 把 k/v 裁到已填充长度、完全不要 mask。它把长度烘进了 **shape**，
所以捕获时是 33 个槽位就永远按 33 个算 —— 序列到 64 时它给出 33 的答案，
**不报错**。这正是本实验的第二阶段检查抓到的：`err = 10.0`。

**"能捕获"和"用得了"是两件事。** 第一版工具只测前者，给 D 报了 `capturable: true`
—— 那是最危险的一个答案。判据已改成"换长度重放后是否还算对"，并且为
D/F 的失败构造了判别力（把 filled 之后的 value 置成 20.0，忽略新长度会差 O(10)）。

### 3. 于是目标被量化了

要拿到 D 的 10.43×/2.17×，同时保住 `correct_after_length_change = True`，
只有一条路：**把长度从设备内存读进 kernel，shape 保持静态**。
`0.02493 ms/层`（ctx=4096）就是这个 kernel 的**靶子**，不是猜测。

## ★ 方法学：为什么必须交错，以及第一版错在哪

```text
第一版（顺序测，且 GPU 被另一个项目占用）：
     A_eager_math 0.05084   B_sdpa_float4d 0.07663   -> A = 1.51x
交错重测（12 轮，MAD 0.00199）：
     A_eager_math 0.06776   B_sdpa_float4d 0.07080   -> A = 1.04x
```

**1.51× 是争用噪声。** 这台机器是共享的：`pareto_offload.py`（不属于本仓库）
当天两次抢 GPU，其中一次正好横穿第一版测量。顺序测每个臂时，后一个臂继承了
前一个的时钟/L2/争用状态 —— 上一轮 grouped GEMM 也正是因此得到过同一个 config
相差 29% 的两个结果。

所以：**共享机器上的臂对照必须交错，并且报 MAD。** 比 MAD 小的差值不是结论。

第二版自己的两个 bug 也记在这里（它们都会让判据变成"永不失败"）：

1. 换长度时只改了 host 变量、没改**设备端** mask。对 mask 类实现来说那等于
   "用过期 mask 重放"，本来就该错 —— 判据在测自己的失误。
2. softmax 在 ~1000 个随机 key 上近似均匀，1025 与 2051 两种长度的答案**趋同**，
   于是判据对"忽略新长度"的实现失灵（ctx=4096 上误报 True）。
   现在把 filled 之后的 value 置成 20.0，判别力是构造出来的。

**一个不可能失败的测试比没有测试更糟，因为它会被当作证据引用。**

## Configuration

```text
shapes:  batch=1, heads=16, head_dim=128, dtype=fp16
         kv_len=1056 filled=33     /    kv_len=4128 filled=1025
         （= StaticCache 的 max_cache_len 与 decode 第 33 步的真实已填长度）
交错：   12 轮 round-robin，iters=50，warmup=10
正确性： 每个臂先与 fp64 SDPA（只在前缀上）比 max_abs_err，再谈时间

⚠️ 测量期间这台机器**是共享的**（`pareto_offload.py` 不属本仓库，占用 14.7 GiB）。
   交错 + MAD 是缓解手段，不是消除。要引用绝对值前应在独占机器上复测。
```

## Verdict

```text
verdict:    PASS（隔离测量达成目的：判定该不该写 kernel、靶子是多少）
outcome:    supported
notes:
  矛盾已解决（我读错了类，归因是对的）。三条结论：
  1) attn_implementation="eager" 在 ctx=4096 值 5.80x，在 ctx=1024 只值 1.04x；
  2) 裁剪路径快 10.43x 但把长度烘进 shape，实测换长度后 err=10.0，不可直接用；
  3) 自研 kernel 的靶子是 0.02493 ms/层（ctx=4096）。
```

## Follow-ups

1. **先做零风险的那条：模型级验证 `attn_implementation="eager"`。**
   一行配置，预期 ctx=4096 从 8.4873 → ~5.05 ms/token。
   门与 p2-02 相同（同路径 KL=0 + 32 token 序列 + informational vs HF）。
   要分别报 ctx=1024 与 ctx=4096 —— 它在两个 ctx 上价值差 70 倍。
   **必须记 `EXP-p2-04` 而不是复用 p2-03 的 ID**：那是模型级结果，本文件是隔离测量。
2. **再写 kernel（真正的工作）**：split-KV flash-decoding，长度从设备 scalar 读。
   靶子 `0.02493 ms/层`（ctx=4096）/ `0.03265`（ctx=1024），
   而且必须过第二阶段的 `correct_after_length_change` 检查。
   前人的实现可参照 vLLM 的 `triton_decode_attention.py`（split-KV 两段式：
   部分 (m, l, acc) + log-sum-exp 合并）与 Triton 官方 FA2 教程的在线 softmax。
3. `--profile` 已经能列出每个臂的内核组成；写 kernel 前先跑一次确认
   arm D 的内核形态（它是我们要复刻的形态）。
4. 一个未解释的观测：arm A 在 ctx=4096 是 10.9 ns/槽位，在 ctx=1024 是 64 ns/槽位
   —— **不是线性的**。写 kernel 前值得弄清它的成本结构（固定开销 vs 扫描），
   否则会照着一个错的模型去优化。

## Applicability

```text
本结论适用于：RTX 5090 (sm_120)、fp16、单卡、batch=1、q_len=1、
              OLMoE-1B-7B-0924（16 heads × 128 head_dim，MHA 无 GQA）、
              transformers 4.57.6、StaticCache（连续布局）
不适用于：q_len > 1（tile 浪费的论证不成立，FMHA 会反过来占优）；
          有 GQA 的模型（heads 数变小，grid 更小）；分页 KV；其它 arch
原因：
  - "eager 比 SDPA 快"的直接原因是 FMHA 的 64 行 tile 对 q_len=1 浪费 98.4%。
    q_len 变大后这个结论会反转 —— 这也解释了为什么生产系统默认用 SDPA。
  - 全部数字绑定在这台机器的争用状态下测得（交错缓解，未消除）。
```
