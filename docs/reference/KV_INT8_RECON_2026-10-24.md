# E6 KV INT8 侦察（2026-10-24，离线轮）

> **定位**：立项材料，不是状态页。E6（矩阵已声明 `EXP-p5-01`）是 ctx=4096 上最大的
> **已声明**杠杆；本文把工业界/学术界的做法、本仓库的工程形状、预注册阶梯与验收判据
> 一次写清。**★ 2026-10-24 夜用户批准立项**——`experiments/EXP-p5-01-kv-int8.md`（预注册
> 记录）+ `src/lemoq/kernels/kv_int8.py`（参考数学/守卫，K 门对照物）+ `tests/test_kv_int8{,_gpu}.py`
> （CPU 6 条 + K 门 GPU 骨架，kernel 落地同名 API 自动激活）已落盘；**kernel 在 10-25 窗实现**
> （照记录 §K门API 的名字/语义）。

## 0. 一句话结论

KV int8 对称量化（v1：K/V 都 per-token scale）把每步 KV 读字节减半：**@4096 省 ≈172 µs
（步 1.76 → ≈1.59 ms，−10%）**，且因为 scale 可**折叠进 attention 分数**，内层循环**零解包
ALU**；主要风险在 K 的粒度（KIVI/KVQuant 证据：K 的离群通道需要 per-channel/pre-RoPE）——
用预注册阶梯逐级升级，门 FAIL 不硬采纳。

## 1. 字节账（为什么它是最大的杠杆）

| 量 | 值 | 出处/口径 |
| --- | --- | --- |
| KV 每步读字节（fp16，ctx=L） | `2(K+V)×16层×16头×128维×L×2B = 128 KB/token` | 架构事实（MODELS.md §3.1）；decode 每步注意力读全量 cache |
| @4096 | **512 MiB/步 → 地板 344 µs**（1560.3 GB/s） | `results/machine_probe.json` 实测带宽 |
| @1024 | 128 MiB/步 → 86 µs | 同上 |
| **ctx 差分实证** | v5 p50 差 1.7626−1.5116=**251 µs** ≈ 384 MiB fp16 的地板 246 µs | `EXP-p4c-01` v5 payload；KV 读基本贴带宽地板（ncu 计数器被 host 挡，口径见 NCU_EVIDENCE） |
| **int8 后** | @4096 省 **≈172 µs（−10% 步时）**；@1024 省 ≈43 µs（−2.8%） | 本文推导 |

对比：F5（lm_head）≈60 µs 且 ctx 无关；E6 的绝对收益随 ctx 线性涨——**这就是"端侧长上下文"
叙事的主杠杆**（同时 VRAM 里 KV 占用减半，直接扩大可部署 ctx）。

## 2. 工业界/学术界已有做法（检索 2026-10-24）

| 方案 | 粒度 | 对本仓库的启示 |
| --- | --- | --- |
| [KIVI](https://arxiv.org/abs/2402.02750)（ICML 2024，[code](https://github.com/jy-yuan/KIVI)；[HF 博客](https://huggingface.co/blog/kv-cache-quantization)已收录） | **K per-channel（沿通道分组）/ V per-token**，非对称，2-bit 也可行 | 非对称粒度是经验共识；**K 的离群在通道维**——per-token 量 K 是 v1 的头号风险 |
| [KVQuant](https://arxiv.org/abs/2404.18013)（NeurIPS 2024） | **pre-RoPE per-channel K** + 离群隔离 + dense-and-sparse | 量化挂在 **RoPE 之前**更准（RoPE 混通道会把离群摊开）；本仓库 RoPE 融合在 decode 前向里，K 进 cache 时已过 RoPE ⇒ v1 只能 post-RoPE，粒度补偿要靠阶梯 |
| [vLLM FP8 KV](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache)（[状态博客](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html)） | per-tensor 或 per-head `k_scale`/`v_scale`；kernel 内 descale（FA3/FlashInfer）或先反量化 | 尺度交给 kernel 折叠是成熟路线；**未标定 scale=1.0 会导致复读**（[实例](https://ai-muninn.com/en/blog/dgx-spark-fp8-kvcache-repetition)）⇒ 我们的 per-token/per-channel 动态 amax 天然免标定 |
| Attention sink（[KVSink](https://arxiv.org/abs/2508.04257)、StreamingLLM 实践） | 前 4 个左右 sink token 保留高精度 | **v1 不保留、让门说话**；v3 升级项（静态 cache 前 4 槽留 fp16 很便宜） |
| [RotateKV](https://www.ijcai.org/proceedings/2025/0690.pdf)（IJCAI 2025） | 旋转后 2-bit 可行 | 记录在案；超出 int8 范围，不做 |

避坑汇总：int8 通常安全（各源一致），**2-bit 以下才必须上旋转/离群隔离**；本方案封顶 int8，
不越界。

## 3. 本仓库的工程形状（两处挂点都在手上）

### 3.1 写路径 = F4 的 `kv_direct_write` 就是天然挂点

`kv_direct.py`（F4，已默认开）每个新 token 的 K/V 行以 `grid=(2H,)` 直写静态 cache。
v1 扩展：同一 kernel 内
1. 对该行求 `amax`（行内归约，单 block 已满足）→ `scale = amax/127`；
2. 写 `int8` 行 + 写 fp16 `scale`（K、V 各一份；额外字节 ≈1.5%，16 层合计 ~4 MiB @4096）；
3. 图捕获兼容：int8 cache 与 scale buffer 都是**静态地址**（照 k/v buffer 的建法），捕获前分配。

ⓘ K per-token 的 scale 逐 token 独立 ⇒ 不需要 KIVI 的 residual/重量化机器（那是 per-channel
才需要的，留给 v2）。

### 3.2 读路径 = scale 折叠进分数，内层零解包（关键代数）

split-KV kernel 读 K/V tile 后目前直接做 `q·k` / 加权求和。两种粒度都可**把 scale 移出内层**：

- **K per-token**（v1）：`s_t·(q·k_t)` —— score 算完乘 `s_t`（每 token 一次标量乘，在
  softmax 之前的 logits 上）；
- **K per-channel**（v2）：`q·(k_t⊙s) = (q⊙s)·k_t` —— `q` 一次性按通道缩放（每头 128 维
  向量、每步一次），cache 完全不动；
- **V per-token**（v1）：`Σ(a_t·s_t)·v_t` —— attention 权重乘 `s_t` 后照常加权。

⇒ **int8 存储不引入任何逐元素反量化乘法**，性能上限就是纯带宽减半（代价是分数路径两次
标量乘，~µs 级）。K1 parity 注意：折叠后与"先反量化再点积"**代数等价但浮点结合序不同**
（照 mxfp4 的教训：不写逐位断言，PARITY_GATE 1e-3 + fp64 oracle）。

### 3.3 门与记账接线

- **正确性**：KV 量化直接改 logits ⇒ **必须新 GATE 行**（照 F5 的独立表先例：`KV_INT8_GATE`
  按 ctx 取行），probe 加 arm kind 或 runner 加开关 `LEMOQ_KV_INT8`（默认关 + fail-closed
  拒绝，与 F5 同款）；
- **字节账**：`benchmarks/bytes.py` 的 KV 记账加 int8 分支（权重账不动）；
- **VRAM**：int8 cache + scale buffer 替换 fp16 cache（减半），payload 的 VRAM 拆分自动跟随。

## 4. 预注册阶梯（一次一变量）

| 版 | 变更 | 通过判据 |
| --- | --- | --- |
| v1 | K/V 都 **per-token** 对称 int8（写路径 F4 挂点 + 读路径 score 折叠） | 新门行 PASS（两 ctx）+ K1 1e-3 + A/B Δ>0 且 >MAD |
| v2（v1 门 FAIL 才做） | K 改 **per-channel**（q 折叠；写路径改行间归约 + 小 residual） | 同上 |
| v3（门 PASS 但贴线才做） | 前 4 个 sink token 保留 fp16（dense-and-sparse） | 贴线判据解除（对照 v2 门值） |

## 5. 验收红线（与全项目一致）

- 不 relax 门；无门行不发布延迟（schema 强制）；覆盖 payload 先归档；
- 金丝雀 ≤5%；一轮一变量（v1 一次只动 KV 位宽，不动注意力实现/位宽其他轴）；
- 图格的金丝雀必须用 graph canary（eager canary 对共租不敏感——P2j 教训）；
- Mac 不产数字；工仵 rsync 回 Mac 后才算轮终。

## 6. 预算与风险

- **工程量**：写 kernel 扩展（amax+量化）+ 读 kernel 折叠（K/V 两处）+ 门接线 + probe/runner
  开关 + bytes 记账 + 测试 ≈ **1–1.5 个 GPU 窗**（比 F4 大，比 E7 小）；
- **风险**：①v1 门 FAIL（K per-token 离群）——阶梯已备，v2 的写路径复杂度是主要代价；
  ②与 F4 的耦合（量化挂在直写路径上，F4 关闭时 KV 量化退回 HF cache 写——需要
  `applicable` 守卫拒绝而非静默跑错臂）；③长尾：ctx=4096 的 teacher-forcing 门跑得慢。
- **不做**：FP8 KV（与 int8 同字节，先例是 kernel descale 路线，我们的向量路径 int8 更简单）、
  2-bit/旋转（超出本项目 int8 范围）。

**待用户批准后**：登记 EXP-p5-01 的执行计划（照本文档），从 v1 开始。
