# EXP-p5-01-kv-int8

> ★ E6 立项（2026-10-24 夜用户批准，材料 [`KV_INT8_RECON_2026-10-24`](../docs/reference/KV_INT8_RECON_2026-10-24.md)）。
> 预注册：假设与判据**先于** kernel 写定（本骨架与参考数学/测试同 commit 落盘，kernel 在
> 10-25 窗实现并按本文件钉死的 API/判据受 K 门检验）。

- **ID**: `EXP-p5-01-kv-int8`（矩阵条目已存在：`configs/experiments.yaml`，ctx [1024, 2048, 4096]）
- **Date**: 2026-10-24（立项/骨架）；kernel 与测量 2026-10-25 窗
- **Operator**: agent（GPU：租用 RTX 5090 32 GB，sm_120，实测读带宽 1560.3 GB/s）
- **Status**: `running`（★ kernel 工程完成 2026-10-25：写路径 `kv_int8_write` K 门 4/4 逐位
  （`2269c11`）+ 读路径 int8 split-KV parity 4/4 vs fp64 oracle 1e-3（`0f1a025`，含部分填充/
  多 split）+ cache 写散布/bulk 量化逐位（`8a844d6`）= **8/8**。★ 读路径首跑抓到并修掉
  head-stride bug（reshape 后 3D 的 stride(1)=D 被当 head 步长，head 0 恰好对、其余头全错）。
  ★★ 门测量首轮**无效（假 0）**：probe 的轨迹用 HF DynamicCache，换核 decode 前向在
  `filled is None` 时回退原路 ⇒ int8 从未执行（`kv_int8_runs=0`、KL 恰 0.0）——工件保留作
  证据（`results/EXP-p5-01-kv-int8/gate_e6_kvint8*.json`）。**剩**：门口径改走 StaticCache
  轨迹（探针需挂 `_lemoq_filled_len` 会话机制）→ A/B → 翻默认）
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

（待 kernel 窗；只填实测。）

## Verdict

（待门 + A/B 齐。）

## Applicability

（待收尾时写。）
