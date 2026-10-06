# MoE 与 CUDA Graph：为什么 HF 的实现进不了图，以及怎么修

> 这份文档记录一个**离线预判**及其依据。预判本身不是结果 —— 它必须由
> `EXP-p0-03`（HF 原版路径）和 `EXP-p0-04`（固定形状路径）在真机上证实或否证。
> 提前写下来的价值在于：到服务器上时不会把"CUDA Graph 能不能用"和
> "我的实现对不对"两个问题搅在一起。

---

## 1. 预判

**`EXP-p0-03`（HF 原版 `OlmoeSparseMoeBlock`）极可能返回 `blocked` 或正确性门失败。**

`EXP-p0-04`（固定形状 MoE）应当通过 capture 与正确性门。

---

## 2. 证据链

### 2.1 profile 里的直接证据

`EXP-p0-02` 在 RTX 5090 上对 OLMoE fp16 / batch=1 / ctx=1024 做了单步 profile：

```
单步 kernel 总数                    8,277
Memcpy DtoH (Device -> Pinned)      1,024 次
cub::DeviceSelectSweepKernel        1,024 次
cub::DeviceReduceSingleTileKernel   1,024 次
at::native::indexFuncSmallIndex     1,024 次
```

而 `64 experts x 16 layers = 1,024` —— **数字完全吻合**。

也就是说：每一层、每一个专家，都恰好触发一次 device→host 拷贝。

### 2.2 机制

`OlmoeSparseMoeBlock.forward`（transformers 4.57.6）对**全部 64 个专家**做 Python 循环：

```python
for expert_idx in range(self.num_experts):
    expert_layer = self.experts[expert_idx]
    idx, top_x = torch.where(expert_mask[expert_idx])      # <-- 问题在这
    current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
    final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
```

`torch.where` → `nonzero`：**输出的长度取决于数据**（该专家分到几个 token）。
要在 host 上分配输出张量，就必须先知道"有几个" —— 所以先把 count 拷回 host。
那 1,024 次 `Memcpy DtoH` 就是这个 count 回读。

### 2.3 HF 自己的说法

同文件 `modeling_olmoe.py` 里：

```python
_can_compile_fullgraph = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)
```

`torch.compile(fullgraph=True)` 与 CUDA Graph 在这一点上受**同一条约束**：
图一旦固定，就不能再有依赖数据形状的分支。

### 2.4 两种失败模式（都要考虑）

| 模式 | 表现 | 危险程度 |
|---|---|---|
| **硬失败** | capture 期间 host 同步被禁止 → `operation not permitted when stream is capturing` | 好：会当场报错 |
| **静默错** | 假设侥幸捕获成功：输出缓冲大小在 capture 时冻结，replay 时路由分布不同 → 越界写或漏算 | **危险**：不抛异常，只能靠 KL/top-1 门发现 |

第二种是这条实验**必须**有正确性门的原因。只看 latency 会拿到一个假的好数字。

---

## 2.5 已排除的嫌疑犯（离线核实）

预判的价值取决于"是不是只有这一处有问题"。把静态缓存/attention 路径逐个查过之后：

| 位置 | 结论 | 依据 |
|---|---|---|
| `_prepare_4d_causal_attention_mask_with_cache_position` | ✅ 安全 | 传 4D mask 时**原样返回**（`causal_mask = attention_mask`），每步不新建 mask |
| `StaticLayer.update` | ✅ 安全 | 用 `index_copy_` **原地**写，不分配；返回整段 cache，地址固定 |
| `_update_causal_mask` 里的 `0.0 in attention_mask` | ✅ 不走这条路 | 只在 `_attn_implementation == "flash_attention_2"` 时触发，默认是 `sdpa` |
| `StaticCache.lazy_initialization` | ✅ 安全 | 只在首次 update 时分配；session 的 `prefill()` 在 capture **之前**就把它初始化完了 |
| **`OlmoeSparseMoeBlock` 的 `torch.where`** | ❌ **唯一的阻塞点** | 单步 1,024 次 DtoH = 64 experts × 16 layers |

**结论：如果只改 MoE 的 routing 就能 capture 成功，那 `EXP-p0-04` 应当通过。**
若 p0-04 也 blocked，说明上述某个"安全"判断错了 —— 那时 `failure_record` 的
`stage` 字段会指向具体环节。

⚠️ 上表里两条（4D mask 恒等、原地写 + 整段返回）已经写成
`tests/test_static_cache_decode.py` 里的**冻结前提测试**。
升级 transformers 时它们会先断，而不是等到 GPU 上才发现。

```
静态缓存的另一个后果：update 返回的是**整段** max_cache_len 的 cache，
所以每步 attention 实际attend 1056 个位置（ctx=1024 + 32），而不是已填充的 ~33 个。
mask 是唯一阻止模型看到未填充区域的机制 —— 这就是 session 必须逐位置更新
4D mask 的原因，不是可选的优化。
```

---

## 3. 修法：把数据依赖的形状整体消掉

### 3.1 关键观察

decode 时 `top_k` 是**固定**的（OLMoE = 8），batch = 1，q_len = 1。
所以每 token 恰好有 `top_k` 个 (token, expert) 对 —— **形状完全静态**：

```
ids = topk(...)                                  # (T*K,)          形状固定
xr  = x.repeat_interleave(K, dim=0)              # (T*K, hidden)   形状固定
gu  = bmm(gate_up[ids], xr)                      # (T*K, 2*ffn)    形状固定
```

`gate_up[ids]` 是**用张量索引**（不是 Python int），因此可以进图。
没有任何 `torch.where` / `nonzero` / `.item()` / `.cpu()`，
也没有 `for expert in range(num_experts)` 这种随数据变化的循环。

实现在 `src/lemoq/kernels/moe_static.py`。

### 3.2 为什么不是"直接循环 top-k 个专家"

因为 `self.experts` 是 `nn.ModuleList`，用专家 id 取模块需要 **Python int**，
而 id 在 GPU 上 → `.item()` → host 同步 → 又是 capture 杀手。

所以必须先把 64 个专家的权重**堆成张量**（图外做一次），之后用张量索引。

### 3.3 代价（必须说清楚）

`gate_up[ids]` 是一次**真实的 gather 拷贝**。OLMoE fp16 下每 token：

```
理想（只读 8 个激活专家）        ~1.5 GiB
gather 读 + 写                   ~3.0 GiB
bmm 再读一遍                     ~1.5 GiB
```

即**多花约 3 GiB/token 的显存流量**换掉 8,277 次 kernel launch。
按实测 1559 GB/s，这约等于 **+2 ms**；而 launch 开销是 **89.6 ms**。
这笔交易在可行性阶段是划算的。

**但它不是终局。** 真正的解法是 Phase 2 的 grouped GEMM（Triton）：
在 kernel 内直接由专家 id 算指针，**没有 gather 拷贝**，把上面 3 GiB 降回 1.5 GiB。
`moe_static` 是**正确性参照 + 可行性门**，不是最终性能形态。

---

## 4. 服务器上的执行顺序

```bash
cd /root/LeanMoE-Quant
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

# 0) 离线回归必须先绿
/root/miniconda3/bin/python -m pytest -m "not gpu" -q

# 1) GPU smoke：小图能不能 capture（失败就停，不加载 13 GB 模型）
/root/miniconda3/bin/python -m pytest tests/test_cudagraph.py -m gpu -q

# 2) p0-03：HF 原版路径（预期 blocked）
/root/miniconda3/bin/python -m benchmarks.runner \
  --exp EXP-p0-03-baseline-cudagraph --ctx 1024 --dtype float16 --force

# 3) p0-04：固定形状 MoE 路径
/root/miniconda3/bin/python -m benchmarks.runner \
  --exp EXP-p0-04-cudagraph-static-moe --ctx 1024 --dtype float16 --static-moe --force
```

**读结果时的判据：**

| 情况 | 含义 | 下一步 |
|---|---|---|
| p0-03 blocked，p0-04 通过 | 预判成立：瓶颈确实是 routing 的动态形状 | 拿 p0-04 的 latency 做 launch 消除的真实收益 |
| p0-03 **也**通过 | 预判错了 —— 那更好，HF 原版就够了 | 检查 p0-03 的 KL/top-1 是否真通过（防静默错） |
| p0-04 也 blocked | 阻塞点不止 routing（如 attention mask 或 StaticCache） | 看 `failure_record` 的 stage 字段定位 |
| 两者都通过但 KL 失败 | **静默错**：图捕获成功但 replay 结果不对 | 这是最有价值的结果之一，说明必须有正确性门 |

**任何情况下都不要**：p0-03 失败后把 `execution` 改成 eager 写进同一个结果文件。
失败本身是结果。

---

## 5. 离线已完成的验证

在 Mac（CPU）上，`tests/test_moe_static.py` 用**真实的** `OlmoeSparseMoeBlock`
与 `OlmoeForCausalLM` 做 parity 对照：

- 固定形状前向与 HF 原版**数值一致**（`norm_topk_prob` 开/关、单/多 token）；
- 12 组不同随机路由 + 一组极端路由（所有 token 挤到同一个专家）下都一致；
- 输出形状与中间形状**不随数据变化**（这是 CUDA Graph 真正依赖的性质）；
- 静态源码检查：模块里没有 `torch.where` / `nonzero` / `.item()` / `.cpu()`，
  也没有 `for ... in range(num_experts)`；
- 整模型替换后输出一致，且原始 `experts` 被释放（否则 16 层会多占 12 GiB）。

这些在 CPU 上跑，**不需要 GPU，也不产生任何可引用到 `results/` 的数字**。
