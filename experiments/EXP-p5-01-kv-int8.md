# EXP-p5-01-kv-int8

> ★ E6 立项（2026-10-24 夜用户批准，材料 [`KV_INT8_RECON_2026-10-24`](../docs/reference/KV_INT8_RECON_2026-10-24.md)）。
> 预注册：假设与判据**先于** kernel 写定（本骨架与参考数学/测试同 commit 落盘，kernel 在
> 10-25 窗实现并按本文件钉死的 API/判据受 K 门检验）。

- **ID**: `EXP-p5-01-kv-int8`（矩阵条目已存在：`configs/experiments.yaml`，ctx [1024, 2048, 4096]）
- **Date**: 2026-10-24（立项/骨架）；kernel 与测量 2026-10-25 窗
- **Operator**: agent（GPU：租用 RTX 5090 32 GB，sm_120，实测读带宽 1560.3 GB/s）
- **Status**: `done`（★ **2026-10-25 全链闭环并采纳**：K 门 8/8 → 门两 ctx PASS
  （v3 forced 轨迹 KL 0.001276/0.000622、top-1 0.9818/0.9879，落表 `552d2ac`）→
  A/B 两 ctx VALID 5/5（@4096 **−112.4 µs/−6.54%**、@1024 −11.5 µs/−0.78%）→
  翻默认 `d4d02fb`（第七开关）→ KV8=1 刷新 23/23（canary −0.427%）→ v7 canonical
  **1.4602/1.6075 ms**、最好格 **2.44×/4.25×**。方法学三课全档：首轮假 0
  （DynamicCache 回退）、v2 自由贪心复利口径、A/B run1 capture 期初始化
  （身份语义修复 `8aab5c1`））
- **上游规格**: `docs/reference/KV_INT8_RECON_2026-10-24.md`（字节账/工业界做法/工程形状/阶梯）

## Hypothesis（矩阵原文 + 预注册细化）

```text
矩阵假设（原文不动）：
  KV 量化省的是带宽不是显存；@4K 约省 9% 总流量
矩阵反证信号（原文不动）：
  收益 <5% 或精度崩塌 → 该杠杆对本模型无意义

细化预期（v1：K/V 都 per-token 对称 int8）：
  - 字节账：KV 读 = 2×16层×16头×128维×L×2B(fp16) → int8 减半。@4096：512 MiB/步
    （地板 344 µs @1560.3 GB/s）→ 省 ≈172 µs/步（−10%）；@1024：−≈43 µs（−2.8%）。
    ctx 差分实证（v5 payload 1.7626−1.5116=251 µs ≈ 384 MiB fp16 地板 246 µs）说明
    KV 读贴带宽地板 ⇒ 省的字节应当兑现成时间（scale 折叠 ⇒ 内层零解包，见 RECON §3.2）。
  - 精度：门 = 独立 KV_INT8_GATE_CITATIONS（照 F5 先例：空表 fail-closed，新测量抄入）。
    v1 预期 per-token int8 过门（int8 通常安全，RECON §2 各源一致）；若 v1 FAIL ⇒
    按预注册阶梯升级 v2（K per-channel，q 折叠），v2 再 FAIL ⇒ 杠杆关闭。

反证信号（出现即视为假设被否证，按硬规则 8 记 evidence 不发布延迟）：
  - A/B 每步 Δ ≤ 0 或 < MAD（"省的字节没兑现成时间"——scale 折叠设计失效）；
  - v1+v2 门都 FAIL（精度崩塌，"对本模型无意义"）；
  - 账本口径：KV share 的下降与 A/B Δ 方向相反（归因断裂，另立调查）。
```

## Method（v1 阶段；一轮一变量——本实验只动 KV 位宽）

```text
写路径:  kv_direct_write（F4，已默认开）的挂点上扩 `kv_int8_write` Triton kernel：
         每 token 行（(1,H,1,D)）内 amax 归约 → scale=amax/127（fp32 算、fp16 存）
         → int8 行 + fp16 scale 写入**静态地址** buffer（图捕获兼容）。
         int8 cache/scale buffer 在 capture 前分配（照 k/v buffer 的建法）。
读路径:  split-KV kernel 的 score 折叠（RECON §3.2 代数）：
         K per-token：logits 乘 s_t；V per-token：attention 权重乘 s_t —— 内层零解包。
守卫:    kv_int8_applicable（本文件参考模块）不满足即显式回退 fp16 cache（绝不静默算错）。
K 门:    ① write 侧 kernel vs 参考数学**逐位**（codes/scale 同一 fp32 舍入链：
         amax → /127 → round(half-to-even) → clamp ±127；kernel 必须复现参考舍入，
         对不齐是实现问题不是判据问题——F4 先例）；② roundtrip 逐元素
         |err| ≤ 0.6·scale；③ 读侧 fold vs 先反量化再点积：fp64 oracle、PARITY_GATE
         1e-3（结合序不同，不写逐位断言——mxfp4 教训）。
精度门:  probe/runner 加开关 `LEMOQ_KV_INT8`（默认关 + fail-closed，照 F5 同款），
         5 窗 wikitext + ones 对照，两 ctx（±2048 按矩阵补测）。
A/B:     run_ab_fresh_process.sh，ENV_ON/OFF，canonical 格（w8a16×kernel @1024 与 @4096
         各一轮——KV 收益是 ctx 的函数，两 ctx 都要直接证据）。
采纳:    门行落表 → A/B VALID → 翻默认 → 刷新（run_f1_refresh.sh 扩 KV8=1 叠加）→
         照 ADOPTION_SYNC 体例回填（`[W]` 不变、KV 字节减半、VRAM 拆分更新）。
```

## Model

```text
model:           allenai/OLMoE-1B-7B-0924（本地快照，fp16）
KV 形状:         16 层 × 16 头 × head_dim 128；StaticCache (1,H,max_len,D) per layer
kv 字节/步:      fp16 = 128 KB/token（全层）→ int8 = 64 KB/token + scale ≈ +1.5%
```

## K 门 API（kernel 实现必须按这些名字/语义；测试已预置）

```text
src/lemoq/kernels/kv_int8.py:
  quantize_per_token_int8(x)      # 参考：沿最后一维；返回 (int8 codes, fp16 scale)
  quantize_per_channel_int8(x)    # 参考（v2 阶梯）：沿倒数第二维（token 维）
  dequant_int8(codes, scale)      # 参考：codes.float() * scale.float()
  kv_int8_rows_applicable(k, v)   # 守卫：cuda/fp16/(1,H,1,D)/D 2 幂/连续
  kv_int8_write(...)              # ★ 10-25 窗落地的 Triton kernel（K 门骨架已按名预置）
tests/test_kv_int8.py             # CPU：参考数学性质 + 守卫 + 折叠代数
tests/test_kv_int8_gpu.py         # GPU：write 逐位 + roundtrip（kernel 落地自动激活）
```

## 结果

### K 门（kernel 级，2026-10-25）——8/8 全过

写路径 `kv_int8_write` 4/4（codes/scale 逐位、零行、roundtrip ≤0.6·scale、守卫拒绝）+
读路径 int8 split-KV 4/4（fp64 oracle parity 1e-3：含部分填充、多 split、cache 写散布、
bulk 量化）。过程修掉三个真 bug：读路径 head-stride（reshape 后 3D 的 `stride(1)`=D 不是
S·D，`0f1a025`）、测试参考的 batched 广播串头（×3，全部冤枉了 kernel）、fp16-scale 除法
1-ulp（kernel 约定：fp32 除、fp16 存）。

### 门（KL/top-1，2026-10-25）——**两 ctx PASS**（v3 forced 轨迹口径）

**方法学两课（工件全保留）**：

- **首轮（DynamicCache）假 0 作废**：`kv_int8_runs=0`、KL 恰 0.0——换核 decode 前向在
  `filled is None` 时回退 HF 原路，int8 从未执行。修复 = probe 轨迹改
  `StaticCacheDecodeSession` 会话（`_lemoq_filled_len` 设备张量挂 cache 对象；
  `97b71c0`）。同 commit 修掉一个真污染点：`_lemoq_kv_int8_bufs` 挂在模块上只增不减，
  prefill 现在会显式作废它（换窗口/换 prompt 的静默复用旧 codes）。
- **v2（自由贪心）是复利口径，不是门**：臂自己 argmax，borderline token 一翻后续位置
  全是苹果比橘子（@1024 w1 KL 6.92/top1 0.27；@4096 w3/w4 同型）。**未翻窗的 KL 才是
  纯扰动**：@1024 ≈1e-4、@4096 ≈2-4e-4——比门限低两个量级。预注册"重跑**同一条**贪心
  轨迹"的正确实现是 **forced**（喂 fp16 参考轨迹的 token 流，照
  `check_fusion_candidate.collect` 的机制；`963bee7`）。v2 两 ctx 工件保留作
  "生成确实会变（贪心 borderline 翻 token）"的行为证据——与 w8a8 门复测时
  "fp16 参考的贪心 token 也在动"是同型现象，不是精度崩塌。

**v3 门实测（5 窗 wikitext + ones 对照、fused norm、StaticCache 会话、forced）**，
工件 `results/EXP-p5-01-kv-int8/gate_e6_kvint8{,_ctx4096}_v3.json`，已落表
`KV_INT8_GATE_CITATIONS`（`552d2ac`，钉死测试含假 0 复发自检：每窗 runs=512>0）：

| ctx | KL mean（5 窗） | KL max 窗 | top-1（集合级） | 判定 |
| --- | --- | --- | --- | --- |
| 1024 | **0.001276** | 0.00423 | **0.981818**（3/165 翻） | **PASS** |
| 4096 | **0.000622** | 0.00120 | **0.987879**（2/165 翻） | **PASS** |

ones 对照：8.2e-04（@1024）/ 7.4e-04（@4096），top1 均 1.0。noop 三臂 Δ 恰 0（恢复干净）。
与权重门对照：@1024 KL 0.001276 vs w8a16 权重门 0.001361——同量级；@4096 反而更稳
（0.000622）。

### A/B（独立进程·同轮相邻）

- **run1（2026-10-25）整轮作废——保留作证据**（`results/EXP-e6-ab-4096-run1/`）：on 臂
  5/6 全部 `blocked`，`failure.message = "kv_int8 的 buffer 初始化不能发生在 capture 里"`。
  根因是当时的失效语义（prefill 作废）与 `capture()` 的内部顺序冲突：capture() 在 warmup
  **之后**还有第二次 `prefill()` 复位（`cudagraph.py:703`），把 warmup 阶段建好的 int8
  buffer 作废 ⇒ capture 期懒初始化撞守卫。off 臂正常（1.7156，canonical 一致），canonical
  payload 由脚本恢复。**修复（`8aab5c1`）**：失效语义改为 **cache 对象身份**（weakref 锚
  `_lemoq_kv_int8_src_cache_ref`）——同对象重复 prefill（同 prompt 重灌，数值等价）复用
  （capture 安全）；换对象（probe 换窗/换会话）重建（免跨会话污染）；CPU 测试钉身份矩阵。
- **修复后单臂图冒烟（@4096）**：`status=done`、gate PASS、`kv_int8_effective=True`、
  p50 **1.6031** ms（canonical off 1.7165 ⇒ −113 µs / −6.6%，单跑非判定值）、host 3.8%。
- **正式轮（两 ctx 均 VALID，采纳判据齐）**，工件 `results/EXP-e6-ab-{4096-run2,1024-run1}/`：

| ctx | off mean（ms） | on mean（ms） | Δ | 对胜 | 金丝雀 | ratio MAD | host | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1.71798 | **1.60558** | **−112.4 µs（−6.54%）** | **5/5** | −0.064% | 0.059% | ≤4.2% | **VALID** |
| 1024 | 1.4689 | **1.4574** | **−11.5 µs（−0.78%）** | **5/5** | −0.401% | 0.138% | ≤5%（逐格过） | **VALID** |

- on 臂 10/10 全部 `kv_int8_effective=True`、gate PASS；kernels/步 329 → **345**
  （+16 = int8 镜像写 kernel 每层一次）——写路径多付 16 次 launch 仍净赚。
- **收益对预测的兑现率（诚实账）**：@4096 理想 −172 µs（字节地板），实测 −112.4 µs
  = **65%**；@1024 理想 −43 µs，实测 −11.5 µs = **27%**。缺口 = int8 读 kernel 的
  scale 折叠开销 + 每步 16 次镜像写；杠杆随 ctx 放大的定性预测成立（矩阵反证线
  "收益<5%" 在 @4096 被 comfortably 超过：−6.54%）。
- **翻默认**（`d4d02fb`，第七开关全 True）+ 无 env 金丝雀：p50 **1.6053**、
  `kv_int8_effective=True`、gate PASS、host 4.0%——与 A/B on 臂 1.6056 双向核对一致。

### 刷新与采纳（2026-10-25，全部完成）

- **翻默认**：`FUSION_DEFAULTS["kv_int8"] = True`（`d4d02fb`，第七开关全 True）。
  oracle-backend 的两条 CPU 测试显式 env=0（N6 纪律）。
- **无 env 金丝雀**：p50 **1.6053**、`kv_int8_effective=True`、gate PASS、host 4.0%——
  与 A/B on 臂 1.6056 双向核对一致。
- **KV8=1 刷新 23/23 格**（`run_f1_refresh.sh` F2=1 KV=1 LMHEAD=1 KV8=1）：金丝雀
  **−0.427%**、round_canary exit 0、failed.txt 空、逐格自检 `kv_int8_effective`
  （kernel 格 True / 其余 False）全对。v7 canonical：

| 格（图内） | v6 | **v7** | Δ |
| --- | --- | --- | --- |
| w8a16×kernel @1024 | 1.4706 | **1.4602** | −10.4 µs |
| w8a16×kernel @4096 | 1.7165 | **1.6075** | **−109.0 µs** |
| fp16×kernel @1024 | 2.1946（F4 轮） | 2.2028 | +0.4%（≈金丝雀噪声级；@1024 KV 份额小，±写镜像相抵） |
| fp16×kernel @4096 | 2.4326（F4 轮） | **2.3484** | −84.2 µs |
| sdpa 格（全部） | — | 不动 | E6 不作用（kv_int8_effective=False ✓） |

- **派生量（v7）**：E7（p4b）**1.2253/1.1025**；attention 杠杆 **1.5086/1.4609**；
  最好格 **2.4421/4.2506**（@4096 破 4×）；权重流份额 **40.9%/23.5%** ⇒ 上限
  **~1.69×/~1.31×**。账本：attn_split@4096 家族 0.3650 → **0.2402 ms**（gap 转负
  −103.9 µs——int8 读已快过 **fp16 口径**的字节地板，这是 E6 的签名；下一任 leader
  仍是 gate_up/down_sum config 类，F7 已预注册）。
- **两条如实记录的口径工件**（不改测量工具，留给后续）：
  1. payload 的 `kv_read_bytes` 仍按 fp16 512 MiB 记账（E6 未进字节模型）⇒ 账本
     attn_split 的"地板"按 int8 字节 268.4 MB 重算是 172 µs（gap +68 µs）；
  2. `vram.kv_gib = −5.544`：分配器检查点法在 E6 下出负值（方法工件）。且当前实现
     **保留 fp16 cache 作写侧正确性网**（int8 只是读源）⇒ VRAM 实际 +int8 镜像
     ≈ +0.27 GiB，"KV 显存减半"是去掉 fp16 网的未来变体，不是本轮事实。

## Verdict

**supported（采纳，2026-10-25）**。预注册假设"KV 量化省的是带宽不是显存；@4K 约省 9%
总流量"兑现为 @4096 **−109 µs（−6.4%）**（>5% 反证线）；@1024 −11.5 µs（0.78%）——
杠杆随 ctx 放大的定性预测成立，小 ctx 兑现率低（27% vs 65%）是 scale 折叠开销 +
每步 16 次镜像写的诚实账。精度：K 门 8/8、门两 ctx PASS（forced 轨迹 KL 0.0013/0.0006、
top-1 0.982/0.988）。三条 A/B 判据（5/5 对、Δ>MAD、金丝雀）两 ctx 全过。

## Applicability

- **硬件/实现绑定**：sm_120（RTX 5090）、triton 3.4.0、torch 2.8.0+cu128、CUDA Graph。
  V100（sm_70）没有这条路（无图捕获的等价前提与带宽量级都不同）——换卡必须重测。
- **模型形状绑定**：GQA 拒绝（`num_key_value_groups != 1` 显式 NotImplementedError）；
  head_dim=128、fp16 q/k/v 是守卫 `kv_int8_rows_applicable` 的边界。
- **口径**：收益是 **batch=1 单流 decode** 的；门是 forced 轨迹口径（165 位置），
  自由生成会因 borderline 翻 token 分岔（v2 工件）——这是行为证据不是精度崩塌，
  与权重门 w8a8 复测的"参考轨迹自己也在动"同型。
- **边界**：量化粒度 per-token 对称（v2 阶梯 K per-channel 未启用——v1 已过门不需要）；
  fp16 cache 仍维护（VRAM 不降反升 ~0.27 GiB）；`kv_read_bytes`/`kv_gib` 两条口径
  工件见结果节。
