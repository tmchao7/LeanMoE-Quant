# EXP-p5-01-kv-int8

> ★ E6 立项（2026-10-24 夜用户批准，材料 [`KV_INT8_RECON_2026-10-24`](../docs/reference/KV_INT8_RECON_2026-10-24.md)）。
> 预注册：假设与判据**先于** kernel 写定（本骨架与参考数学/测试同 commit 落盘，kernel 在
> 10-25 窗实现并按本文件钉死的 API/判据受 K 门检验）。

- **ID**: `EXP-p5-01-kv-int8`（矩阵条目已存在：`configs/experiments.yaml`，ctx [1024, 2048, 4096]）
- **Date**: 2026-10-24（立项/骨架）；kernel 与测量 2026-10-25 窗
- **Operator**: agent（GPU：租用 RTX 5090 32 GB，sm_120，实测读带宽 1560.3 GB/s）
- **Status**: `running`（★ kernel 工程 + **门已过** 2026-10-25：K 门 8/8（见结果节）；
  门两 ctx PASS——v3 forced 轨迹口径 KL 0.001276/0.000622、top-1 0.9818/0.9879，
  已落表 `KV_INT8_GATE_CITATIONS`（`552d2ac`）。方法学两课全记录在案：首轮假 0
  （DynamicCache 回退）与 v2 自由贪心复利口径（不是门）——工件分代保留。
  **剩**：A/B（两 ctx）→ Δ>MAD ⇒ 翻默认 → KV8=1 刷新 → 回填）
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
- **正式轮**：（跑中：run2 @4096 PAIRS=5；@1024 待跑。预期 @4096 −150~170 µs、@1024 −43 µs。）

### 刷新与采纳

（待跑：A/B VALID 且 Δ>MAD ⇒ 翻默认 → KV8=1 刷新 23 格 → 回填。）

## Verdict

（待 A/B；门链已齐：K 门 8/8 + 门两 ctx PASS。）

## Applicability

（待收尾时写。）
