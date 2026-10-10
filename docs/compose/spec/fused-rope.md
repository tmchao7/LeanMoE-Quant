---
feature: fused-rope
status: designed
updated: 2026-10-11
branch: opt/fused-rope
commits: 
---

# Fused RoPE（decode 路径 rotate_half 链融合为单 Triton kernel）

## Report

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
     签名与 HF 一致；单 Triton kernel，`grid = Hq + Hk`（每 head 一个 program，
     一个 program 同时算完 128 维），输出新张量（`empty_like`，不 in-place，
     与 HF 的 out-of-place 语义相同）。
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

- [ ] T1: `rope.py` kernel + 形状/无triton回退 + `warm` — acceptance: `pytest -m "not gpu"` 下回退与签名契约绿；代码过 ruff/black（covers: S2.1）
- [ ] T2: `attn_decode` 分支点 + 单元测试（回退矩阵、源码契约、默认关） — acceptance: `tests/test_fused_rope.py` 绿；env 关时 import 走 HF 原函数（covers: S2.1, S2.2）
- [ ] T3: GPU K1 parity（fused vs HF，≤1e-3）+ `pytest -m gpu` 全绿 — acceptance: 服务器 10+ 项 GPU 测试过（covers: S2.4; depends: T1, T2）
- [ ] T4: `--ab-fused-rope` + 服务器同轮 A/B + 回填 Report — acceptance: `results/fused_rope_ab.json` 三验收齐全，Report 写结论（covers: S2.3; depends: T3）
