# HANDOFF — decode attention kernel（split-KV flash-decoding）

> 写给**下一轮专注做这个 kernel 的 agent**。读完这一份就能直接开工。
>
> - **写于**: 2026-10-06 晚
> - **仓库 HEAD**: `main = 6a6b1c1`（已推 GitHub，服务器已同步）
> - **任务**: 写一个 decode attention kernel，**长度从设备 scalar 读**，
>   替换当前的 `attn_implementation="eager"`
> - **前置状态**: 全部离线测试绿（258 passed）；本任务**还没开始写一行 kernel**
>
> 通用环境与纪律见 [HANDOFF.md](HANDOFF.md)（§1–§4 服务器接入/环境/命令，
> §7–§9 硬规则/陷阱/协作纪律仍然有效；它的 §5/§10 已过期）。

---

## 1. 为什么是 attention，以及为什么必须读设备端长度

`EXP-p2-04` 之后 ctx=1024 的 per-token 是 **3.1051 ms**。首个可攻击项已经不是
gather（已消掉）而是**我们自己那些小 kernel 的堆积**与 attention 本体。

`EXP-p2-03` 的隔离测量给出三条硬事实：

| 臂 | ctx=1024 | ctx=4096 | 换长度后还算对？ |
|---|---|---|---|
| SDPA + 4D float mask（**HF 默认**） | 0.07080 | 0.26008 | ✅ |
| HF eager 数学实现（**现在用的**） | 0.06776 | 0.04487 | ✅ |
| SDPA 裁到已填充、**不要 mask** | **0.03265** | **0.02493** | ❌ err=10.0 |

第三条是最快的，**而且不可用**：它把长度烘进了 **shape**，捕获时是 33 个槽位
就永远按 33 个算 —— 序列走到 64 时给出 33 的答案，**不报错**。
判据抓到它时 `err = 10.0`（正确答案是 O(1)），而 mask 类实现是 `0.004`。

**所以任务的定义很精确：拿到第三条的收益，同时保住 `correct_after_length_change`。**
唯一的路是让 kernel 从**设备内存**读已填充长度作为循环上界 —— launch 形状保持静态，
就能进图。

---

## 2. ★ 靶子必须在**图内**重建（不要用隔离数字）

`EXP-p2-04` 教会的一件事：**隔离测量对多算子实现不具外推性。**
`tools/bench_decode_attention.py` 把每个臂 eager 地跑在紧循环里，而它们最终要被
capture 进图 —— 进图后每算子的分配与启动开销消失，**多算子实现拿到一笔"图补贴"**。
结果：隔离预测 ctx=1024 省 0.049 ms，实测省 **0.978 ms**（差 5.7 倍）。

当时 `arm B`（单算子 FMHA）与模型级数字吻合到 6–7%，所以我误以为整个 harness 可外推。
**一个臂对上不代表另一个臂也对上。**

### 图内的当前 attention 代价（可直接引用）

从 `results/EXP-p2-04-eager-attention__ctx={1024,4096}/result.json` 的
`bottleneck.top_kernels` 里取**出现 16 次**（每层一次）的那些 rows：

```text
ctx=1024（per-token 3.1051 ms，busy 3.0604）      [来自 EXP-p2-05 的集合差，见下]
  gemvNSP_kernel                            0.09289  occ 16   <- q·kᵀ 或 w·v
  cunn_SoftMaxForwardSmem<8, …>             0.03943  occ 16   <- attention softmax
  elementwise_kernel<128,4,…CUDAFunctor_add> 0.02831  occ 16   <- + mask
  vectorized_elementwise_kernel<4,BUnary…>  0.01364  occ 16
                                            ───────
  attention 核心                             0.17427 ms/token

ctx=4096（per-token 3.4788 ms，busy 3.4390）
  gemvx::kernel<int,int,…>                  0.27743  occ 16   <- q·kᵀ 或 w·v
  cutlass_80_wmma_tensorop_f16_s161616gemm  0.18948  occ 16   <- 另一个矩阵乘
  cunn_SoftMaxForwardSmem<8, …>             0.04291  occ 16
  elementwise_kernel<128,4,…CUDAFunctor_add> 0.02942  occ 16
  vectorized_elementwise_kernel<4,BUnary…>  0.01313  occ 16
                                            ───────
  attention 核心                             0.55237 ms/token

自研 kernel 的代价（同一份 artifact）：`_attn_split_kernel` + `_attn_combine_kernel`

    ctx=1024 : 0.12326 + 0.01335 = 0.13661 ms/token
    ctx=4096 : 0.36474 + 0.01370 = 0.37844 ms/token
    ctx=4096 的 split kernel 单独是 0.36474，KV 字节下限 0.3437 -> **94.2% 实测带宽**

> ★ **上面两张表必须用「集合差」得到，不能按名字手挑。** 本文件的第一版就是手挑的，
> 得到 0.212 / ~0.40 —— **两个都错**（ctx=1024 多算了 0.038：把 o_proj 的 GEMV
> 与 routing 的 softmax 算进了 attention，同时漏了一个 elementwise）。
> 正确做法：两条臂都 profile，取**只在 eager 臂出现、出现次数 = 层数**的行
> （`tools/bench_attn_in_graph.py` 的 `attention_core.vanished_rows` 就是这个集合）。
> 这个错误是 `EXP-p2-05` 在 A 步发现的 —— 靶子被高估，收益就会被低估。

**验收判据**：新 kernel 让 per-token 低于 `3.1051 ms`（ctx=1024）/ `3.4788 ms`（ctx=4096），
并且 attention 核心那几行消失、换成自己的 kernel 名且总时间更低。

---

## 3. 设计（前人经验 + 本仓库已验证的部分）

### 3.1 为什么必须切 KV（split-KV）

decode 时 `batch=1, q_len=1, heads=16` → grid 只有 `batch × heads = 16` 个 program，
而卡上有 **170 个 SM**。**91% 的 SM 是空的。** 这是 "delay-bound" 的直接原因，
也是 [Flash-Decoding](https://pytorch.org/blog/flash-decoding/) 要解决的问题：
把 KV 维切开，每片算部分结果，再合并。

### 3.2 两段式结构

```text
kernel 1   grid = (num_splits, heads)
           每个 (split, head) 在自己的 KV 区间上做在线 softmax，
           写出部分 (m_i, l_i, acc[128])  -> 形状 (heads, num_splits, 128+2)
kernel 2   grid = (heads,)
           按 log-sum-exp 合并各 split：
              M = max_j m_j ;  out = Σ_j acc_j · exp2(m_j − M) / Σ_j l_j · exp2(m_j − M)
```

`num_splits` 的起点建议 `ceil(sm_count / (heads))` ≈ 170/16 ≈ **10**，
但**必须实测扫描**（见 §5 的"参数敏感性本身就是结果"）。

### 3.3 数值骨架（照 [Triton 官方 FA2 教程](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)）

```text
m_i   = -inf (fp32)        运行最大值
l_i   = 1.0  (fp32)        运行和      <- 教程里初值是 1.0，注意与 0.0 的差别
acc   = 0    (fp32)        累加器
qk_scale = scale · 1.44269504              # 1/ln2，为走 exp2
for 每个 KV block:
    qk = dot(q, kᵀ) · qk_scale
    m_ij = max(m_i, max(qk, 1))
    p    = exp2(qk − m_ij[:, None])
    alpha = exp2(m_i − m_ij)
    l_i  = l_i·alpha + sum(p, 1)
    acc  = acc·alpha + dot(p, v)
    m_i  = m_ij
epilogue: acc = acc / l_i
```

**exp2 而不是 exp**：快，且 `1/ln2` 的缩放是精确可复现的写法（教程同款）。

### 3.4 长度从设备 scalar 读（这是图安全的关键，也是本任务的立身之本）

`benchmarks/cudagraph.py::StaticCacheDecodeSession` 的结构**已经为这件事准备好了**，
先读一遍再动手：

```python
# replay() —— 注意 _set_step_inputs 在图**外**、在专用 stream 上执行
with torch.cuda.stream(self.stream), torch.no_grad():
    self._set_step_inputs()      # 每步在这里 fill_ 设备张量
    self.graph.replay()          # 图里**只有** model forward

# _set_step_inputs()
self.buffers.token.fill_(self.current_token_id)
self.buffers.cache_position.fill_(self.position)
self.buffers.attention_mask.fill_(min_dtype)
self.buffers.attention_mask[..., : self.position + 1].zero_()
```

**每步在图外更新设备张量是既有机制**，所以：

1. 在 `GraphBuffers.allocate` 里加一个 `filled_len`（`torch.int32`, shape `(1,)`）buffer；
2. `_set_step_inputs` 里 `self.buffers.filled_len.fill_(self.position + 1)`；
3. 把它挂到 cache 对象上（例如 `session.cache._lemoq_filled_len = self.buffers.filled_len`），
   这样被替换的 `OlmoeAttention.forward` 可以通过 `past_key_values` 拿到它
   —— **不要用模块级全局变量**；
4. kernel 里 `n = tl.load(filled_ptr)`，循环上界用 `n`。

**为什么这是图安全的**：CUDA Graph 记录的是 **launch**，不是执行。
kernel 在运行时从设备内存读循环上界，没有任何 host 同步，launch 形状（grid）不变。
这正是 vLLM / FlashInfer 这类系统的做法。

### 3.5 图安全的硬约束（违反会 capture 失败或静默算错）

| 约束 | 原因 |
|---|---|
| **禁止 `@triton.autotune`** | 它会真的跑多个 config 来计时 → capture 期间往图里塞计划外的 kernel。配置在**图外**扫好，显式传参 |
| **避开 `tl.make_tensor_descriptor` / TMA** | descriptor 路径需要 global scratch；不设 `triton.set_allocator` 时默认会分配 → capture 隐患。用普通指针算术 |
| 首次调用前必须 `warm()` | Triton 首次调用 JIT 编译；在 capture 期间做这件事**报错信息会指向别处**（`moe_grouped.py::warm` 是现成范例） |
| 循环上界不得来自 host | 只有设备端 load 出来的标量可以做上界 |
| 不用 `atomic_add` 合并 | 非确定性 → 同路径 KL=0 的门不可复现。用确定性归约 |

### 3.6 最小侵入的集成方式

**只替换 attention core，别重写整个 attention 模块。**

`transformers 4.57.6` 的 `modeling_olmoe.py` 里**有两个类**（这是本会话踩过的坑）：

```text
OlmoeAttention          eager 数学实现
OlmoeSdpaAttention      SDPA 实现      <- 默认加载的是这个（_attn_implementation="sdpa"）
```

要替换的是当前生效的那个。**保留 HF 的全部前后处理**（`q_norm`/`k_norm` 投影、
RoPE `apply_rotary_pos_emb`、`past_key_values.update()` 的 cache 写入、
最后 `transpose(1,2).contiguous().reshape` + `o_proj`），只把中间的
`matmul → +mask → softmax → matmul` 换成一次 kernel 调用。

这样做的理由：投影/RoPE/cache 的数值偏离风险最小，而且换了它们对性能没有帮助
（那些是 `[W]` 里必要的 attention 权重读量）。

**输出契约**：kernel 要给出 `(1, 16, 1, 128)` 的 fp16，与 HF 的 `attn_output` 一致。

### 3.7 不需要 mask 了

如果 kernel 只读 `[0, filled_len)`，那么 **4D mask 完全不需要**：decode 时
`q_len=1`，所有已填充的 key 都可见，没有因果结构要表达。

→ 可以不再传给 attention（也省掉 `_set_step_inputs` 里那两次 `fill_`/`zero_`）。
但要**显式记录**这个决定：mask 是 HF 用来屏蔽未填充槽位的唯一机制，
不再需要它是因为 kernel 自己就不读那些槽位了。

---

## 4. 门（**上机之前**冻结，任何一条不得为上机后的结果放宽）

| 门 | 内容 | 阈值 |
|---|---|---|
| **K1** kernel 级 | 与 **fp64 oracle** 比（不是与 eager 路径比 —— 累加顺序不同） | 误差不差于基线 × 1.5，且 `allclose(atol=1e-3, rtol=1e-3)`（`moe_grouped.PARITY_GATE` 是现成范例） |
| **K2** 端到端同路径 | graph+kernel vs eager+kernel | `KL ≤ 1e-5`（期望恰好 0） |
| **K3** informational | vs HF + 动态 cache | 只记录，**不当门槛**（跨实现差异是预期的） |
| **K4 换长度** | 长度 N 捕获 → 改成 M 重放 → 是否给出 M 的答案 | `err < 1e-2`。**这是本任务的核心判据** |
| **K5 回归栅栏** | 32-token 序列与 `EXP-p2-04` 对照 | 不一致先查是否 tie 翻转，再判 FAIL |

K4 的现成实现在 `tools/bench_decode_attention.py::replay_after_length_change`，
**但先读它的注释**：它第一版有两个 bug（只改 host 变量没改设备 mask；
softmax 在 ~1000 个随机 key 上近似均匀导致两种长度的答案趋同），
两个都让它"永不失败"。**一个不可能失败的测试比没有测试更糟，因为它会被引用。**

---

## 5. ⚠️ 坑清单（全部是本会话实测踩过的）

1. **"能捕获" ≠ "用得了"。** 形状里烘进长度会**静默算错**（实测 err=10 vs 0.004）。
   判据必须是"换长度重放后还算对"，不是"`torch.cuda.graph` 没报错"。
2. **隔离测量对多算子实现不外推**（§2）。单算子臂可外推，多算子臂必须在图内测。
3. **共享机器上必须交错测臂**：`pareto_offload.py`（不属于本仓库）会抢 GPU。
   顺序测会把争用当成差异 —— 实测同一个 config 两次差 **29%**，
   eager 的收益假报成 **1.51×**（交错重测是 1.04×）。
   **报 `wins`（逐轮最快的次数）与 MAD，不只是中位数。**
4. **参数敏感性本身就是结果**：grouped GEMV 的 36 点网格 spread 是 **3.17×**，
   而第一版默认配置落在**最慢的一角**，让闸门判了 NULL、差点放弃整条路线。
   扫描要**报全部**（含默认值在网格中的位置），不只报最好的那个。
5. **Triton 首次调用必须 warm**，否则 capture 期的报错指向别处。
6. **`_set_step_inputs` 在图外执行**（这是长度来源的机制，见 §3.4）。
7. **prefill 永远走逐专家循环**；grouped 与 attention kernel 都只作用于 T=1。
   `static_moe_forward` 按 token 数分派，别破坏它。
8. **每次测量先与 fp64 oracle 对齐**：快的错实现不是结果。（第一版 grouped 闸门
   就是只测时间没测正确性，被 `test_grouped…` 抓到。）
9. **别把系统测成 5.7 倍误差**：`OlmoeAttention` 与 `OlmoeSdpaAttention` 是两个类，
   读源码前先 `print(model.config._attn_implementation)` 和
   `type(model.model.layers[0].self_attn).__name__`。
10. **`nsys` 没装**（只有 `/usr/local/cuda/bin/ncu`，不在 PATH 上 → 用
    `tools/probe_machine.py::_find_tool` 的 fallback 模式）。launch 归因靠 `torch.profiler`。

---

## 6. 环境与命令

```bash
# 服务器（免密）
ssh 5090b          # = connect.weste.seetacloud.com:45735，容器 hostname autodl-container-uwvncb97au-763233b0

# 必设（否则 13 GB 模型会落到 30 GB 系统盘上）
export HF_HOME=/root/autodl-tmp/hf
export PATH=/usr/local/cuda/bin:$PATH
PY=/root/miniconda3/bin/python

# 模型（字节数已校验 = 13,838,721,960）
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

# 先查 GPU 是否被占（**每次 timing 前都要查**）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 离线门（Mac 或服务器都可）
uv run --extra dev pytest -m "not gpu" -q
uv run --extra dev ruff check . && uv run --extra dev black --check .
uv run --extra dev python -m tools.check_docs

# 已有的隔离 attention bench（可加 --profile 看每臂 kernel 组成）
$PY -m tools.bench_decode_attention --kv-len 1056 --filled 33 --repeats 6

# 模型级（要新建 EXP id，见 §7）
$PY -m benchmarks.runner --exp <新 ID> --model "$M" --ctx 1024 \
    --batch 1 --max-new-tokens 32 --warmup 5 --timed 20 --dtype float16

# 结果回收（results/ 是 gitignored，不会自动保留）
rsync -az 5090b:/root/LeanMoE-Quant/results/ ./results/
```

**环境版本**：Python 3.12.3 / torch 2.8.0+cu128 / transformers 4.57.6 /
triton 3.4.0 / driver 595.71.05 / 实测带宽 1561.9 GB/s。
磁盘：`/` 25 GB free、`/root/autodl-tmp` 38 GB free。

---

## 7. 建议的执行顺序

| 步 | 做什么 | 为什么这个顺序 |
|---|---|---|
| **A** | 在图内建立 attention 靶子：把 `EXP-p2-04` 的 result.json 读出来（§2 有现成数值），再跑一次 `--profile` 式确认 | 隔离数字不可用，靶子必须来自图内。**不要跳过** |
| **B** | 写 oracle（纯 PyTorch fp32 的 split-KV + LSE 合并）+ 离线测试 | 上一轮的教训：公式/索引错误不需要 GPU 就能全抓出来，而且它们会给出"看起来合理"的错结果 |
| **C** | 写 kernel（`src/lemoq/kernels/attn_decode.py` + `_attn_triton.py`，分开是为了 Mac 上没有 triton 也能跑离线测试） | 照 `moe_grouped.py` / `_grouped_triton.py` 的既有结构 |
| **D** | 图内臂对照（eager vs 新 kernel），**交错 + MAD** | 这是"值不值得接进模型"的闸门 |
| **E** | 接进模型（最小侵入地换 attention core，§3.6） | 只有 D 通过才做 |
| **F** | 新 EXP id（建议 `EXP-p2-05-attn-kernel`）+ 记录 + commit | 见下 |

**新 EXP id 的必要性**：`EXP-p2-03` 是隔离测量、`EXP-p2-04` 是模型级、这一条是
**新 kernel**。三条是不同层次的证据，混用 id 会让结果文件的口径错乱
（本会话已经因为"`EXP-p2-02` 的 id 里没有 `cudagraph`"导致 runner 静默跑成 eager 臂
去连 huggingface —— 臂从**实验矩阵读**，别从 id 猜）。

---

## 8. 纪律（会踩错的）

- **merge 不 rebase**，不 force-push。
- 每个实验必须留：`results/<EXP-ID>/result.json` + `experiments/<EXP-ID>.md` + 一个 commit。
- **Mac 上产生的数字绝不进 `results/` 或任何结论**（Mac 没有 NVIDIA GPU）。
- 不提交 `results/`、权重、`.env`、密钥。
- **agent 不得放宽正确性门。** KL/top-1 失败就写 `FAIL`。
- **不碰**别人的进程与目录：`/root/MegaKernel-vLLM`、`/root/models/Qwen3-0.6B`、
  `/root/autodl-tmp/mf-work/`（`pareto_offload.py`）。
- 负结果是结果。本项目的价值有一半在这些负结果上。

---

## 9. 现在就能引用的数字（省得重新推导）

```text
# 累计（per-token p50，RTX 5090 / fp16 / batch=1 / 同机同 driver 595.71.05）
eager(SDPA,无图) 82.64 ms  ->  p0-04 5.2637  ->  p2-02 4.0715  ->  p2-04 3.1051
                                      15.73x          20.3x           26.6x
ctx=4096: p0-04 8.4873 -> p2-04 3.4788 (23.8x)   达成率 17.8% -> 43.4%

# 当前 ctx=1024 的归因（p2-04，busy 3.0604 ms / 1053 kernels，unmatched=0）
custom kernel(_gate_up_act+_down_sum)  1.1373
expert gemm(投影+lm_head)              0.7982
latency floor(<4us 的 ~1050 个 kernel)  1.1491   <- 下一个目标
routing(gemv+softmax+topk+bitonic)     0.1516
expert gather                          0.0600
attention 核心                         0.17427  <- ★ 集合差，不是手挑（见 §2 的警告）
other                                  1.0375

# p2-05 之后（attention kernel 已接进模型）
ctx=1024 : 3.0077 ms  达成率 50.2%   累计 27.5x
ctx=4096 : 3.2356 ms  达成率 46.7%   累计 25.5x
           attention 核心 -> _attn_split_kernel 0.12326 + _attn_combine_kernel 0.01335

# 字节账（fp16, 每 token）        [W] 2,357,723,136   [T]@1024 2,491,940,864
# 实测带宽 1561.9 GB/s             floor_W 1.5095 ms   floor_T@1024 1.5955 ms
# 即当前 3.1051 ms 已是 floor_T 的 2.0 倍

# kernel 间的已知量（可当 sanity 检查）
gate_up (E,2F,H)=(64,2048,2048) fp16 = 67,108,864 B/层
down    (E,H,F)=(64,2048,1024) fp16 = 33,554,432 B/层
```

## 10. 两个未解项（别当成已解决）

1. **`ctx=4096` 的 `grouped+sdpa` 从未跑过**，所以 p2-04 的 ctx=4096 `2.440×` 是
   **grouped 与 eager 两项合并**的效果，不能归给任一项。ctx=1024 的 `1.315×` 是干净的。
2. **`ctx=4096` 的 TTFT 涨了 32%**（140.9 → 186.5 ms）而 per-token 降了 2.44×。
   prefill 走逐专家循环、与 p2-04 的改动无关，所以更像测量波动或显存状态差异，
   **但没有解释**。要报端到端数字前必须查清。
