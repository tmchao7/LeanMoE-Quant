---
feature: skip-mask-fill
status: designed
updated: 2026-10-11
branch: opt/skip-mask-fill
commits: 
---

# Skip Mask Fill（graph 臂跳过每步两次 mask 更新）

## Report

## [S1] Problem

`StaticCacheDecodeSession._set_step_inputs()`（`benchmarks/cudagraph.py`）在**图外**
每步执行两次 attention_mask 更新：`fill_(min_dtype)` 全量 + `zero_()` 前缀。
split-KV decode kernel **不读 mask**（只读设备端 `filled_len`，见
`attn_decode.py` §"不再需要 4D mask"），mask 只有 eager 臂（HF sdpa）需要——
代码注释（`cudagraph.py:614-617`）明确把"去掉这两次 fill_"标为后续单独改动。
当前 graph 臂每步白做 2 次 kernel launch + host dispatch（估计 0.5–1.8% 的
1.94 ms 步），是否真实可测、收益几何，需要一次同进程 A/B 给出诚实答案。

## [S2] Design

1. **`_set_step_inputs(*, skip_mask: bool = False)`**（`benchmarks/cudagraph.py`）：
   `skip_mask=True` 时跳过 `attention_mask.fill_(min_dtype)` 与
   `attention_mask[..., :position+1].zero_()` 两行；其余四个 fill
   （token / cache_position / position_ids / filled_len）**照做**（graph 输入，
   必须每步更新）。`skip_mask=False`（默认）行为与现状逐字节相同。
2. **开关**：环境变量 `LEMOQ_SKIP_MASK_FILL`（默认 `"0"`）。只在
   `replay()` / `replay_at_position()` 里读取并传参；`step()`（eager）与
   `capture()`（含 warmup）**永远 False**——eager 臂依赖新鲜 mask，capture
   期保持与现状完全一致。env 在图外每步读取，不进 capture 烘焙。
3. **A/B 工具**（`tools/profile_graph.py`）：把 `run_routing_ab` 泛化为按
   任意 env 切臂的同进程交替 A/B，新增 `--ab-mask-fill`（臂标签
   `off/on`，env=`LEMOQ_SKIP_MASK_FILL`）；`--ab-fused-route` 行为不变。
   验收三件套与 routing-fusion 相同：
   ① 基线臂（off）首尾金丝雀漂移 ≤5%，超 ⇒ 整轮 VOID；
   ② 全部 run 的 32-token 序列逐位相同（跳过 fill 不得改变贪心轨迹——
   若不同，说明图内有代码读 mask，候选立即否决）；
   ③ off/on 两臂 wall 的同进程均值差，方向与量级如实记录
   （**低于噪声的收益也如实写**：refuted/null 同样是交付）。
4. **不改默认值**：env 缺省关闭 ⇒ 已录格与现有路径零影响；采纳为默认
   需另行批准（重测成本另议）。

## [S3] Out of Scope

- 不动 eager 臂的 mask 语义、不动 capture 逻辑、不动 mask buffer 形状；
- 不做 KV 写入融合、attention QKVO 反量化、其余 4 个 fill 的去留；
- 不新增/改动任何矩阵格子或已录 result.json；
- 不在本轮把开关默认打开。

## Tasks

- [ ] T1: `_set_step_inputs` 加 `skip_mask` 参数，replay 路径按 env 传入 — acceptance: env 关（默认）时行为与现状逐字节相同；env 开时 replay 跳过两次 mask 更新、其余 fill 照做；eager/capture 永不跳（covers: S2.1, S2.2）
- [ ] T2: 单元测试钉住开关矩阵 — acceptance: `tests/test_skip_mask_fill.py` 覆盖 默认关=全 fill / 开=replay 跳 mask 且 filled_len 仍填 / eager step 不跳；`pytest -m "not gpu"` 绿（covers: S2.1, S2.2）
- [ ] T3: `profile_graph --ab-mask-fill` 泛化 env A/B — acceptance: 交替 off/on 臂，产出金丝雀漂移、tokens_equal、两臂 wall 均值与差值 JSON；`--ab-fused-route` 原行为不回归（covers: S2.3）
- [ ] T4: 服务器同轮 GPU A/B 并回填 Report — acceptance: `results/skip_mask_fill_ab.json` status 字段与三验收数字齐全，Report 写入结论（support/refuted 如实）（covers: S2.3; depends: T1, T2, T3）
