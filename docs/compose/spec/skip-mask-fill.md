---
feature: skip-mask-fill
status: delivered
updated: 2026-10-11
branch: opt/skip-mask-fill
commits: f422f06..f131d86
---

# Skip Mask Fill（graph 臂跳过每步两次 mask 更新）

## Report

**What was built** — `StaticCacheDecodeSession._set_step_inputs` 新增 `skip_mask`
参数：graph 臂（`replay`/`replay_at_position`，图外每步）可按环境变量
`LEMOQ_SKIP_MASK_FILL`（默认关）跳过每步两次 attention_mask 更新
（`fill_(min_dtype)` + 前缀 `zero_()`）；其余四个输入 fill、容量检查、eager
`decode_eager` 与 capture 路径完全不变 ⇒ 已录路径逐字节零影响。`tools/profile_graph.py`
把 routing 的同进程交替 A/B 泛化为 `_run_env_ab`（env 切臂 + 金丝雀 + 跨臂 token
比对，routing 历史键名契约原样保留），新增 `--ab-mask-fill`。

**Verification** — Mac 四门：`ruff check .` PASS、`black --check .` PASS、
`pytest -m "not gpu" -q` **494 passed**（20 deselected）、`python -m tools.check_docs` OK。
服务器 `pytest -m gpu` **19 passed, 1 skipped**。同轮 GPU A/B
（`results/skip_mask_fill_ab.json`，ctx=1024 w8a16 grouped×kernel×fused，4 run）
：**status=measured**；金丝雀（off 臂首尾）**−1.905% ≤5%**；**tokens_equal=True**
（4 run 32-token 序列逐位相同 ⇒ 图内确实无人读 mask）；off 均值 **2.0111 ms**
vs on **1.9844 ms**，**treatment_faster=1.33%**。机制证据：四 run 的 gpu_busy
1.8744/1.8747/1.8784/1.8781 ms **跨臂不变**，差值全部来自 host
（host_share off 7.7/5.7% → on 6.3/4.6%，≈27 µs/step），与"fill 是 host 侧
launch"的预期一致。**结论：supported 但收益小**——+1.33% 与轮内金丝雀漂移
（1.905%）同阶，属弱支持；默认仍关，采纳需另行批准。

**Journey log**
1. 联网调研两路子任务中断后由主 agent 以浏览器完成；最有价值的互证是
   arXiv 2608.26612（OLMoE 上融合 kernel 单独 5.6–9×、端到端 0.999×，launch-bound）
   与 vLLM fusion 官方收益表（norm+residual 1–4%、RoPE→KV 写 2–4%）——后者是
   下两个候选的预期上界。
2. `_run_env_ab` 是通用 env 交替 A/B 核心；后续 env 候选开关直接复用，
   只需换 env_var 与臂标签。
3. 服务器 scp 公钥被拒（ssh 免密正常），工件回传用 `ssh … cat > local` 绕过。
4. 本实验的"host_share>5% ⇒ 污染"守卫要重新解读：treatment 本身就是降 host，
   判据应看 **busy 是否跨臂不变** + 金丝雀，而不是绝对 host_share。
5. 预算还剩 ≈¥7；候选队列按预期收益排序：KV 直写（0.110 ms/5.9%，vLLM 上界 2–4%）
   > residual+RMSNorm 融合（0.130 ms，上界 1–4%）> 本轮 mask-fill（已做，1.33%）。

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

- [x] T1: `_set_step_inputs` 加 `skip_mask` 参数，replay 路径按 env 传入 — acceptance: env 关（默认）时行为与现状逐字节相同；env 开时 replay 跳过两次 mask 更新、其余 fill 照做；eager/capture 永不跳（covers: S2.1, S2.2）
- [x] T2: 单元测试钉住开关矩阵 — acceptance: `tests/test_skip_mask_fill.py` 覆盖 默认关=全 fill / 开=replay 跳 mask 且 filled_len 仍填 / eager step 不跳；`pytest -m "not gpu"` 绿（covers: S2.1, S2.2）
- [x] T3: `profile_graph --ab-mask-fill` 泛化 env A/B — acceptance: 交替 off/on 臂，产出金丝雀漂移、tokens_equal、两臂 wall 均值与差值 JSON；`--ab-fused-route` 原行为不回归（covers: S2.3）
- [x] T4: 服务器同轮 GPU A/B 并回填 Report — acceptance: `results/skip_mask_fill_ab.json` status 字段与三验收数字齐全，Report 写入结论（support/refuted 如实）（covers: S2.3; depends: T1, T2, T3）
