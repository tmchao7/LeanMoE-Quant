---
feature: fused-rope
status: delivered
updated: 2026-10-11
branch: opt/fused-rope
commits: 7b37d21..e5e0631
---

# Fused RoPE（decode 路径 rotate_half 链融合为单 Triton kernel）

## Report

**What was built** — 新模块 `src/lemoq/kernels/rope.py`：HF `apply_rotary_pos_emb`
的融合候选，**同一 Triton kernel 每层 launch 两次**（q、k 各一，160 → 32
kernel/step），fp32 中间计算、按原 dtype 舍入；数学与 `rotate_half` 半切逐项一致。
call site 在 `attn_decode._decode_forward_impl`（**唯一 kernel 路径入口**），按
`LEMOQ_FUSED_ROPE`（默认关）分支——prefill、HF 回退、已录路径零影响；非 decode
形状 / 无 CUDA / 无 triton / bf16（1 ulp≈8e-3 过不了 K1）一律**逐位退回 HF**。
`tools/profile_graph.py` 新增 `--ab-fused-rope`（三路 A/B 互斥泛化）。

**Verification** — Mac 四门：ruff PASS、black PASS、`pytest -m "not gpu"` **501 passed**、
check_docs OK。服务器 `pytest -m gpu` **23 passed, 1 skipped**（fused vs HF parity
走 `attn_decode.PARITY_GATE` 同口径 atol/rtol 1e-3 × slack 1.5，fp16/fp32 过；
bf16 与非 decode 形状回退逐位过）。GPU A/B 两轮：
第 1 轮金丝雀 **5.923% >5% ⇒ VOID**（日志与工件保留为 `fused_rope_ab.void1.*`，
漂移全在 host：off 臂 busy 1.8725→1.8742 仅 +0.09%）；第 2 轮
（`results/fused_rope_ab.json`，ctx=1024 w8a16 grouped×kernel×fused，4 run）：
**status=measured**；金丝雀 **+3.661% ≤5%**；**tokens_equal=True**（4 run 32-token
逐位相同）；off 均值 **1.9655 ms** vs on **1.8007 ms**，**treatment_faster=8.39%**。
机制证据：busy off 1.8718/1.8746 vs on 1.6791/1.6787（臂内散布 <0.2%，
**−0.194 ms ≈ rope 块预算全数回收**，与 [S1] 的 0.216 ms 估算吻合）。
**结论：supported——本轮最大收益（+8.39% wall / −10.4% busy）**，且证据双口径
（wall + busy）一致。注意事项：第 2 轮 3/4 run 的 host_share>5%（共享卡 host 噪声），
但判定依据的 busy 臂内极稳、wall 与 busy 方向一致。**默认仍关**，采纳为默认
需另行批准（届时须重测 grouped 已录格，P2d 同款）。

**Journey log**
1. 金丝雀只在 wall 上被 host 噪声打爆（第 1 轮 5.923% VOID）而 busy 分毫不动——
   wall 口径的 VOID 要照章重跑，但 busy 是更稳的机制证据；后续可把 busy 稳定性
   自检纳入 `_run_env_ab`（评审建议）。
2. bf16 的 1 ulp≈8e-3，物理上过不了 1e-3 的 K1——dtype 白名单收窄到 fp16/fp32
   比放宽门更诚实（门不能为数据改）。
3. spec 初稿写"单 kernel grid=Hq+Hk"，实现选择了更简单的"同 kernel 两次 launch"
   （避免 triton 里对指针做 where 选择）——交付时已修正 spec 口径。
4. rope 的 mul/add 散在 `BinaryFunctor`/`CUDAFunctor_add` family 里，单看名字
   归因不出来——kernel 分解表要按**功能块**（neg+cat+mul+add 的 rope 链）汇总
   才能看出 0.216 ms 的真靶子；这正是本轮收益 8.39% 的来源。
5. 候选队列余项：KV index_copy（0.0555 ms，必要写，只能融合）、copies 族
   （0.17 ms/149 调用，来自 HF RMSNorm 的 dtype cast——已被 P2i fused norm
   覆盖一部分，需重新分解）、lm_head gemvx 0.127 ms（带宽地板近）。

## [S1] Problem

`results/graph_profile.json` 的 kernel 表里，HF `apply_rotary_pos_emb` 的链
**散在四个 family 里、合计约 0.216 ms ≈ 11.5% busy**，是当前最大的可融合块：

| family | 次数/step | ms/step | 来源 |
|---|---|---|---|
| `neg_kernel` | 32 | 0.0454 | `rotate_half` 的 `-x2`（q,k ×16 层） |
| `CatArrayBatchedCopy` | 32 | 0.0545 | `rotate_half` 的 `cat((-x2, x1))` |
| `BinaryFunctor` mul | 64 | 0.0917 | `q*cos`、`rot*sin`（q,k ×2 ×16 层） |
| add（CUDAFunctor_add 的 rope 份额） | ~32 | ~0.024 | `+` 合并两路 |

每层 10 个 kernel、全步 160 个，全是小张量（q 16×1×128 fp16、k 4×1×128），
纯 launch/延迟 bound。vLLM 官方 fusion 文档没有 rope 单列，但 `fused_add_rms_norm`
类 pointwise 链融合是公认做法；本仓 P2i 的 fused RMSNorm 是同型先例。

## [S2] Design

1. **新模块 `src/lemoq/kernels/rope.py`**：
   - `apply_rotary_pos_emb_fused(q, k, cos, sin, position_ids=None, unsqueeze_dim=1)`，
     签名与 HF 一致；**同一个 Triton kernel 每层 launch 两次**（q、k 各一，
     program = 一个 head，一个 program 算完 128 维），输出新张量（`empty_like`，
     不 in-place，与 HF 的 out-of-place 语义相同）。
   - 数学与 HF 逐项一致：`rotate_half` 半切拼接、`out = q*cos + rot(q)*sin`；
     kernel 内 fp32 计算、store 舍入 fp16（K1 ≤ `PARITY_GATE` 1e-3 由 GPU 测试钉住）。
   - **形状守卫**：非 `(1, H, 1, D)`（batch≠1 / q_len≠1 / D 非偶数）→ 退回调用
     HF 原函数（kernel 路径本就只服务 decode；防御性降级，不静默算错）。
   - 无 CUDA/triton（Mac）→ 退回 HF（与 `dequant_mode` fallback 同风格）。
   - `warm()`：capture 前 JIT 预编译（对齐 `moe_grouped.warm` 惯例）。
2. **开关**：`LEMOQ_FUSED_ROPE`（默认 `"0"`）。分支点在
   `attn_decode._decode_forward_impl` 的本地 import 处（**唯一的 kernel 路径
   call site**；prefill / eager-HF 回退路径继续用 HF 原函数，不受影响）。
   Python 相位读取 ⇒ capture 烘焙所选实现，采纳前不改默认值。
3. **A/B**：`tools/profile_graph.py --ab-fused-rope`（复用 `_run_env_ab`，
   env=`LEMOQ_FUSED_ROPE`，臂 off/on）；验收三件套与前两轮相同：
   ① off 臂首尾金丝雀 ≤5%；② 4 run 32-token 逐位相同（fused rope 不得改变
   贪心轨迹——不等 ⇒ 否决）；③ off/on wall 均值差如实记录。
4. **正确性依据**：`run_cudagraph` 的门用**同路径 eager**（同一 session、同一
   env）做参照 ⇒ 两臂同实现，KL 1e-5 / top1 1.0 门不受实现切换影响；
   fused vs HF 的绝对差异走 K1（`PARITY_GATE` 1e-3，GPU 测试）。

## [S3] Out of Scope

- 不动 prefill / HF 回退路径的 rope；不改 `OlmoeRotaryEmbedding` 的 cos/sin 生成
  （每步仅一次，非本轮目标）；
- 不做 rope 融进 q/k proj 的更大融合；不做 index_copy（KV 写是必要工作）；
- 不新增矩阵格、不碰已录 result.json、不改默认值。

## Tasks

- [x] T1: `rope.py` kernel + 形状/无triton回退 + `warm` — acceptance: `pytest -m "not gpu"` 下回退与签名契约绿；代码过 ruff/black（covers: S2.1）
- [x] T2: `attn_decode` 分支点 + 单元测试（回退矩阵、源码契约、默认关） — acceptance: `tests/test_fused_rope.py` 绿；env 关时 import 走 HF 原函数（covers: S2.1, S2.2）
- [x] T3: GPU K1 parity（fused vs HF，≤1e-3）+ `pytest -m gpu` 全绿 — acceptance: 服务器 23 passed 1 skipped（covers: S2.4; depends: T1, T2）
- [x] T4: `--ab-fused-rope` + 服务器同轮 A/B + 回填 Report — acceptance: `results/fused_rope_ab.json` 三验收齐全，Report 写结论（covers: S2.3; depends: T3）
