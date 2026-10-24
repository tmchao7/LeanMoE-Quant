# FUSION_OPPORTUNITIES — 融合优化的剩余空间（任务规格）

> **性质**：给**下一个 agent** 的可执行任务规格。不是状态页（看 [RESULTS.md](../RESULTS.md)），
> 不是展望（看 [ROADMAP.md](ROADMAP.md)）。本文的每个"现状"数字都从既有工件复算得出，
> 每条命令都可重跑；**每条"估计"都明确标了 est 并给出推导**，不得当作已测结果引用。
>
> 生成 2026-10-10（基于 `w8a16 × kernel × cudagraph @ctx=1024` = **1.9241 ms** 这一格）；
> 若该格的 p50 已变（例如某项已采纳），**先重算 §1 的账本再动手**。

---

## 0. 一页纸：先做什么

| 顺序 | 事项 | 省下 | 依据强度 | 成本 | 依赖 |
| --- | --- | --- | --- | --- | --- |
| **F1** | 采纳三项**已测**融合（RoPE / routing / mask-fill），重测 graph 格 | **≈0.24 ms** | **measured**（三份同轮 A/B 工件） | 1 天卡时 | 无 |
| **F2** | `q/k/v` 三合一 GEMM（P2e） | 0.07–0.10 ms | est（字节账 + 并行度算得出） | 半天代码 + 半天测 | 无 |
| **F3** | 残差 add 融进 GEMM epilogue | 0.04–0.08 ms | est（classic epilogue） | 半天 | 无 |
| **F4** | KV 写入融进 RoPE / attention（去掉 `cat` + `index_copy`） | 0.07–0.09 ms | est（每次 1.7 µs 写 8 KB） | 1 天，风险最高 | 需改 `StaticCache` 接口 |
| **F5** | `lm_head` 也量化成 w8a16 | 0.06 ms | est（已在墙上，字节减半） | 门是主要成本 | 需重跑精度三链 |
| **F6** | 专家 GEMV 上 w4a16 | **0.30 ms** | est，**但被精度门阻塞** | 要先修量化算法 | ❌ 门 FAIL，不可做 |

**F1 是第一优先，理由不是收益最大，而是它是唯一"数字已测、代码已写、只差默认开着"的一项。**
现在的账面上有一笔 **0.24 ms（12.5%）的已测收益没有兑现**，这是全项目最大的账面缺口。

> ★ **2026-10-24 采纳轮已执行**（F1+F2 默认开，23/23 OK、canary 0.30%）；执行手册见
> [handoff/HANDOFF_2026-10-25.md](handoff/HANDOFF_2026-10-25.md) §2（S2/F4）。**本文 §1 的账本
> 是刷新前基线**（1.9241 ms 那一格），刷新后已在 canonical 上重算（§F4 附正式账本）：
> S2 = **F4 KV 直写**（可捕获 50.3 µs 排名第一，F3 出榜）。
> ★ **2026-10-25：整表重算已机械化** —— `python -m tools.operator_ledger <格目录>`，
> 内置 §1.2 的固定字节表与预注册捕获率，直接输出排名（对刷新前 payload 复算与
> HANDOFF §5 预演逐条一致；见 ISSUES"2026-10-25 复盘" N6/N7）。

**F6 不要做**：`w4a16` 的引用门是 FAIL（`GATE_CITATIONS`：`kl=0.093560 / top1=0.866667` @1024，
`0.059331 / 0.866667` @4096）。硬规则 2 要求先过门再谈性能；修法是换量化算法（GPTQ/AWQ），
不是改 kernel。见 [ROADMAP.md](ROADMAP.md) Tier B6 / C2。

---

## 1. 账本（可复算，动手前先核一遍）

### 1.1 钱花在哪

基于 `results/EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024/result.json`
（`status=done`，p50 **1.9241 ms**，539 kernel/步，`host_share` 2.8%）：

| 类别 | ms/token | 占步 | 明细 |
| --- | --- | --- | --- |
| **自研 kernel** | **1.2536** | 65.2% | gate_up 0.4431 / dequant_gemm 0.3233 / down 0.2579 / attn 0.1210 / rmsnorm 0.0815 / out_reduce 0.0140 / attn_combine 0.0127 |
| **ATen 家族** | **0.6029** | 31.3% | elementwise 0.2029 / gemvx(lm_head) 0.1641 / routing 0.0955 / index 0.0566 / cat 0.0544 / reduce 0.0151 / softmax 0.0143 |
| 覆盖 | 96.5% | — | `bottleneck.top_kernels` 覆盖 |

### 1.2 自研 kernel 离墙多远（必要字节 ÷ 1561.9 GB/s；该轮旧实例值，当前只读 1560.3，差 0.1%，本表不重算）

| kernel | 调用/步 | 实测 ms | 字节 MB | 地板 ms | 达成率 |
| --- | --- | --- | --- | --- | --- |
| `_gate_up_act_kernel_q` | 16 | 0.4431 | 545.3 | 0.3491 | **78.8%** |
| `_down_sum_kernel_q` | 16 | 0.2579 | 272.6 | 0.1745 | 67.7% |
| `_dequant_gemm_kernel` | 64 | 0.3233 | 272.6 | 0.1745 | **54.0%** ← 最低 |
| `gemvx`（lm_head，fp16） | 1 | 0.1270 | 206.0 | 0.1319 | 103.9%（已在墙上） |
| `_attn_split_kernel` @1024 | 16 | 0.1210 | 134.2 | 0.0859 | 71.0% |
| `_attn_split_kernel` @4096 | 16 | 0.3644 | 536.8 | 0.3437 | **94.3%** |

### 1.3 纯延迟尾巴（**最被低估的一块**）

下面六族合计 **0.4388 ms = 整步的 22.8%**，而它们的必要字节加起来 **< 30 MB（地板 ≈0.02 ms）**
⇒ **≈95% 是 launch / 延迟，不是带宽**：

| 家族 | ms | 典型每次成本 |
| --- | --- | --- |
| elementwise（6 个不同 kernel） | 0.2029 | — |
| routing（`gatherTopK` + `bitonicSortKV`） | 0.0955 | 2 个 kernel × 16 |
| index（`index_copy`） | 0.0566 | 32 次 × 1.7 µs |
| cat（`CatArrayBatchedCopy`） | 0.0544 | 32 次 × 1.7 µs |
| reduce（4 个） | 0.0151 | — |
| softmax（`softmax_warp_forward`） | 0.0143 | 16 次 |

★ **决策口径**：这一块要按"每次 launch 多少 µs"看，不能按"占比"看。0.0544 ms 看起来只有 2.8%，
但它是 32 次只为写 8 KB —— 这种才是融合的甜点。

### 1.4 当前达成率

`attainment.bw_attainment_pct = 47.7%`，字节地板 `floor_T_ms = 0.9187 ms`（分母是
`results/machine_probe.json` 的实测带宽，**不是规格值** —— 硬规则 4）。

---

## 2. 逐项任务规格

### F1 — 采纳三项已测融合（**measured，第一优先**）

#### 现状：✅ 已采纳（2026-10-24 刷新轮验收，`FUSION_DEFAULTS` 四 True）

| 项 | 环境变量 | 收益 | 工件 | 验收 |
| --- | --- | --- | --- | --- |
| 融合 RoPE（7 算子 → 1 kernel） | `LEMOQ_FUSED_ROPE=1` | **8.39%**（1.9655 → 1.8007 ms） | `results/fused_rope_ab.json` | `tokens_equal=true`、金丝雀 3.661% |
| 融合 routing（softmax+topK+sort → 1） | `LEMOQ_FUSED_ROUTE=1` | **3.0%**（1.9521 → 1.8935 ms） | `results/routing_ab.json` | `tokens_equal=true`、金丝雀 2.215% |
| 跳过 mask-fill | `LEMOQ_SKIP_MASK_FILL=1` | **1.33%**（2.0111 → 1.9844 ms） | `results/skip_mask_fill_ab.json` | `tokens_equal=true`、金丝雀 −1.905% |

复跑 A/B：

```bash
export PATH=/root/miniconda3/bin:$PATH
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
python -m tools.profile_graph --model $M --ctx 1024 --ab-fused-rope   # 一次只变一个变量
python -m tools.profile_graph --model $M --ctx 1024 --ab-fused-route
python -m tools.profile_graph --model $M --ctx 1024 --ab-mask-fill
```

#### ★ 2026-10-23 账本复核 + 采纳前正确性证据

- **账本基准格仍有效**：`EXP-p4c-01__bit_width=w8a16_attn=kernel_ctx=1024` 的 payload 仍是
  **p50 1.9241 ms / 539 kernel/步 / `host_share` 2.8% / `rmsnorm=fused`**，与本文生成时一致。
- **需要刷新的 graph 格实算 23 个**（`execution=cudagraph` 且 `status=done`，按
  `(exp_id, bit_width, attn, ctx)` 去重）。
- **三项候选都过了自然文本正确性门**：`tools/check_fusion_candidate.py --ctx 1024 --steps 32
  --windows 5 --execute`（独立 HF 参考 + 两臂 graph/eager 自检 + 跨臂自由 greedy 轨迹）：

| 候选 | `correctness_gate` | 跨臂轨迹 | 两臂 vs HF（165 位置汇总） | 工件 |
| --- | --- | --- | --- | --- |
| `rope` | **PASS** | 5/5 一致 | baseline KL 0.001302 / top-1 0.9758；candidate KL 0.001345 / top-1 0.9758，均 PASS | `fusion-check/rope-natural-20261023.json` |
| `routing` | **PASS** | 5/5 一致 | KL 0.001302 → 0.001298，top-1 均 0.9758，均 PASS | `fusion-check/routing-natural-20261023.json` |
| `mask` | **PASS** | 5/5 一致 | KL 0.001302 → 0.001302，top-1 均 0.9758，均 PASS | `fusion-check/mask-natural-20261023.json` |

★ 逐窗口看，**window 1 在 baseline 那一臂也同样是 FAIL**（33 位置、top-1 0.9394）——那是
w8a16 量化门在单个窗口上的边界，不是候选的问题；工具按 165 位置汇总判定，两臂汇总门均 PASS。
工件在 GPU 机 `/root/autodl-tmp/lemoq-w4-data/fusion-check/`，Mac 镜像在
`lemoq-artifacts/EXP-p1-05-2026-10-23/fusion-check/`（仓库外、不入 git）。
**这不是**性能轮次、不改 `GATE_CITATIONS`、不翻默认；三项 A/B 的收益仍**不得相加**。

#### ★ 纪律红线：三份 A/B **不得相加或相乘**

三份工件的基线不同（1.9655 / 1.9521 / 2.0111 ms），是**三轮独立测量**。按硬规则 3，
跨轮数字不得相除或相乘。所以：

- ❌ 不许写"三项合计 12.7%，所以 1.9241 → 1.680 ms"；
- ✅ 只允许写"三项**分别**是 8.39% / 3.0% / 1.33%，**合并收益须单轮实测**"；
- 若给估计值，必须标 `est`：`1.9241 × 0.9161 × 0.970 × 0.9867 ≈ 1.69 ms（est，待单轮实测）`。

#### 采纳流程（照抄 P2j，见 `scripts/rounds/run_p2j_refresh.sh`）

1. 打开开关**作为默认**（不是只改 env 默认值 —— 要把默认写进代码/配置，并在
   `config` 里记录实现，像 `rmsnorm: fused` 那样可查）；
2. 重测 **23 个 `execution=cudagraph` 且 `status=done` 的格**（去重后计数，
   清单见 §5 的复算命令）。★ **2026-10-23 实测：三项的作用域不同**——
   `route` 走 `moe_static`/`moe_grouped`，**所有格**都会变；`rope`/`skip_mask_fill`
   只在 `attn=kernel` 的 forward 上（**6 格**），其中 `packed_qkv` 还要求格是**量化**的
   ⇒ 只有 **2 格**（p4c-01 的 w8a16×kernel 1024/4096）会同时受三项影响，其余格只受
   `route` 影响。刷新脚本 `scripts/rounds/run_f1_refresh.sh` 已按 `kernel` + `quantized`
   逐格设开关（先 `DRY=1` 看清单）；不要在非 kernel 格上开 `mask-fill`——它会显式拒绝；
3. 每轮必须：**首尾同格金丝雀**（`python -m tools.round_canary head.json tail.json --graph`，
   退出码非零 = VOID）+ 逐格占用快照；
4. 覆盖前归档到 `results/<EXP>-run/vN_archive/`（硬规则 9）；
5. 逐记录回填"★ F1 刷新"小节；`RESULTS.md` / `PORTFOLIO.md` / `README.md` 的数字同步；
6. 四门绿后提交（`ruff && black --check && pytest -m "not gpu" && check_docs`）。

#### 风险

- RoPE 融合臂的 **bf16 路径被回退过**（commit `e5e0631`：1 ulp ≈ 8e-3 过不了 K1 1e-3）。
  采纳前确认 fp16 臂仍是逐位等价；**不要**为了让它"能跑"而放宽 `PARITY_GATE`。
- 三项同时开 = 三个变量同时变，违反硬规则 3 的"一次一个变量"。**验收轮要分开测**，
  只有确认各自成立后，才允许一轮测"三项全开"的组合，并标 `combined`。

---

### F2 — `q/k/v` 三合一 GEMM（est 0.07–0.10 ms）

#### 现象

`_dequant_gemm_kernel` 是全项目达成率最低的自研 kernel（**54.0%**），每步 64 次
（16 层 × q/k/v/o 四次），每次只读 4.26 MB。

#### 根因（算出来的，不是猜的）

```
q_proj: N=2048, BLOCK_N=4  →  grid = 512 个 program
num_warps=2                →  1024 个 warp
1024 / 170 SM              =  6 warp/SM
SM 能装 ~48–64 warp        ⇒  occupancy ≈ 10–12%
```

**瓶颈是"在飞的 warp 太少，存不下访存延迟"**，不是带宽。单次地板 2.73 µs 对一个需要
ramp-up 的 kernel 几乎是全部时间（实测 5.05 µs/次，gap 2.3 µs/次 × 64 次 = 0.149 ms）。

★ 附带教训：`BLOCK_N=4` 是**隔离微基准**里的大 N 最优解（`_dequant_triton.py:83` 注释），
而图内每次只有 2048 行。**同一配置在不同尺度下最优解不同** —— 融合前必须按图内的真实形状重扫。

#### ★ 2026-10-23 定位复核 + 候选状态

- 用图内 `w8a16 × attn=kernel × ctx=1024` 那一格（p50 1.9241 ms）复算：
  `_dequant_gemm_kernel` **0.3233 ms/步、占步 16.8%、达成率 54.0%、离地板 149 µs** ——
  三个自研 kernel 里**最差**，也是整步最大的可回收缺口（`_gate_up_act` 绝对最大但已
  78.8%；`_down_sum` 67.7%）。所以"量化上线后问题最大的算子"就是它。
- 已实现、**默认开（2026-10-24 采纳）**：`src/lemoq/kernels/packed_qkv.py` + 环境变量
  `LEMOQ_PACKED_QKV=1`（只对 w8 / `dequant=fused` / 对称 / 无 bias / `attn=kernel` 生效，
  其余**显式拒绝**，不静默退回三次调用）；是否生效记进 payload 的 `config.packed_qkv`。
- 验收状态：CPU 契约测试 8 条（含 reference 路径的打包等价）；GPU **K1 逐位对照 PASS**
  （packed 一次 launch 与三次独立调用**逐位相同**，最大误差 0）。
- **★ 合规 A/B 已拿到（2026-10-23，同轮相邻、每臂独立进程）**：在
  `EXP-p4c-01 w8a16 × attn=kernel × ctx=1024` 这一格上交替跑 off/on/off/on/on/off
  （每臂新进程，首尾同为 off 作金丝雀）：
  - off **1.9541 / 1.9529 / 1.9538 ms**（host 3.5–3.7%），on **1.8636 / 1.8575 / 1.8581 ms**
    （host 3.8–4.2%）；
  - **1.9536 → 1.8597 ms = −93.9 µs（−4.81%），3/3 对获胜，比值 MAD ≈ 0.12%**；
  - 首尾同格金丝雀漂移 **−0.015%**；两臂 gate 均 **PASS**；`kernels_per_step` **539 → 507**
    （正好 −32：q/k/v 48 次 → 16 次）——证明实现确实生效。
  这一轮满足项目的 `host_share ≤5%` + 同轮相邻 + 每对胜负判据，**可作为 F2 的采纳证据**。
- **同进程交错 A/B 跑了三轮（2026-10-23），三轮都按 `host_share` 规则判 VOID**（方法学对照）：
  `tools/profile_graph --ab-packed-qkv`，3 对交替 + 尾金丝雀，三轮 `tokens_equal=true`、
  canary 2.40% / 2.65% / 3.14%，但每轮 `host_share` 5.4–7.1% > 5% ⇒ 三轮都 `invalid`。
  可复现的 host 无关信号（每轮 3/3 对获胜）：
  **1.8839 → 1.7852（−98.7 µs，−5.24%）**、**1.8829 → 1.7846（−98.3 µs，−5.22%）**、
  **1.8813 → 1.7852（−96.1 µs，−5.11%）** —— 三轮极差 <3 µs，约为该 kernel 149 µs
  地板缺口的 2/3；wall 差 +4.57% / +3.98% / +4.32% 同向但**不作数**。
  ★ 结构性原因：两臂的绝对 host 开销几乎相同（≈0.115 ms/step），而打包臂的 wall 更小
  ⇒ 同一个 host 开销在它身上占的比例更高，`host_share` 更容易越过 5%。所以**合规轮必须在
  安静的机器上做**（runner 的图格路径在该状态下曾测到 2.8%）；这台共享机今天做不到。
  下一步：安静窗口用 `scripts/rounds/run_f1_refresh.sh` 的同类协议（归档 + graph 金丝雀）
  做合规 A/B，再与 F1 一起 combined 采纳 + 23 格刷新。
- **打包形状的 config 重扫（2026-10-23）**：`tools/bench_dequant_gemm_graph --out-dim 6144`
  （16 层 × 1 次打包调用/步，N=6144、K=2048，7 个 config 交错 15 轮）：赢家 `B8K256w4s3`
  只比默认 `B4K256w2s3` 快 **+0.21%**（9.378 vs 9.398 µs/call，11/15 轮）⇒ **config 不动**
  （远低于 5% 门槛）。打包调用实测 **1362.8 GB/s = 实测读带宽地板的 87.3%**
  （12.78 MB ÷ 9.378 µs，分母 1560.3 GB/s），而单投影调用只有 54% ——
  这就是"把在飞 warp 补上"之后的形状。**剩下的 ~13% 与 o_proj 的 16 次独立调用**
  才是更深一层融合（q/k/v/o 进 attention core）才动的东西，属新假设。

#### 改法

- 把 `q_proj/k_proj/v_proj` 三份权重在 host 侧**打包成一份连续 buffer**（N = 2048×3 = 6144），
  一次 launch 出三份输出；grid 变 1536（18 warp/SM），且 `hidden_states` 只读一遍；
- 保持 `o_proj` 独立（它的 epilogue 要接 F3 的残差）；
- 约束不变：`BLOCK_K` 必须整除 `group_size`、grid 是 host 常量、不能 `@triton.autotune`
  （CUDA Graph capture 安全，见 `_dequant_triton.py` 模块 docstring）。

#### 验收

- `PARITY_GATE` 1e-3 不得放宽；新增 K1 用例覆盖"三合一 vs 三次独立调用"逐位一致；
- 同轮 A/B（交错），`tokens_equal=true`，金丝雀 ≤5%；
- 记录 `kernels_per_call` 应从 539 降到约 539−32=507（q/k/v 从 48 次变 16 次）；
- 落盘 `result.json` + 记录小节 + 提交。

#### 风险

- 打包 buffer 会让 **VRAM 峰值**上升（三份权重同时在显存）；按硬规则 6 报告分项 VRAM；
- 若 `hidden_states` 的消费者不止这三处，不要把它烧进 kernel（见 §4 注意事项 ⑤）。

---

### F3 — 残差 add 融进 GEMM epilogue（est 0.04–0.08 ms）

#### 现象

`_out_reduce_kernel`（`_grouped_triton.py:198`）把 `(H,)` fp32 写回 fp16 输出，然后
PyTorch 再做 `hidden = residual + mlp_out`。每步 **32 次**这样的 add（16 层 × 2 个子层），
每次读写 2048×2 B。它们落在 elementwise 家族（0.2029 ms / 6 个 kernel）里。

#### 改法

给 `_out_reduce_kernel` 和 `_dequant_gemm_kernel` 各加：

```
residual_ptr,  # (H,) 或 (N,) —— 上一层输出
STORE_RESIDUAL: tl.constexpr,   # True 时 out = residual + acc
```

kernel 尾多一条 load + add，**零额外字节**（省掉一次读 + 一次写 + 一次 launch）。

#### 验收

- 逐位等价：残差加法的顺序必须与 PyTorch 一致（`acc + residual` vs `residual + acc`
  在 fp32 下相同，但要确认累加精度：现在 `_out_reduce` 用 fp32 累加后转 fp16，
  融进 epilogue 后**保持同一顺序**）；
- 同上 A/B + `tokens_equal`；
- `kernels_per_call` 应减少约 32。

#### 风险

★ **只能融"单消费者"的张量。** residual 只有一个消费者（下一层），所以可以融；
`hidden_states` 同时喂 attention 和 MoE，**不能**融（会导致重算而不是重读）。

---

### F4 — KV 写入融进 RoPE / attention（✅ **已采纳 2026-10-25**：est 0.07–0.09，实测 kernel 格 −1.8~−3.0%）

#### 现象

`index_elementwise_kernel ×32`（0.0546）+ `CatArrayBatchedCopy ×32`（0.0544）= **0.1090 ms**，
只为了把 32 个 8 KB 的 KV 写进 `StaticCache`。平均 **1.7 µs/次**写 8 KB —— 纯 launch。
★ **2026-10-24 刷新后正式账本**（`tools/operator_ledger`，canonical payload p50 1.5586 ms）：
`kv_write_idx` 32 次 × 实测 **0.0560 ms**、gap **55.9 µs**、可捕获 **50.3 µs —— 排名第 1**；
F3 残差 gap 24.1 µs **低于 30 µs 入榜线**（不再是被选退路）；lm_head gap −5.3 µs 照旧被过滤。
**S2 = F4 确认**（预注册规则机械执行）。

#### 改法（两条路，选一条）

1. **RoPE 融合时直写**：fused RoPE 已经在寄存器里算出 k/v，顺手写进 cache 的 `[pos]` 位置，
   省掉 `torch.cat`；
2. **in-place append**：让 `_attn_split_kernel` 读写同一块 buffer（`_lemoq_filled_len`
   已经是设备端取值的机制，见 `attn_decode.py:679`），彻底去掉 cat。

#### 验收

- ★ **这一项最容易改错数值**：cache 的写入位置错了不会崩，只会让 attention 读到旧值。
  必须用"逐 token 序列完全一致"作为硬门（`tokens_equal`），而不是只看 KL；
- 建议先写一个**离线单元测试**：给定 `filled_len` 与 `pos`，断言 cache 的
  `[0:filled_len+1]` 内容与 HF `StaticCache.update()` 逐位相同；
  ✅ **2026-10-24 已就绪**：`src/lemoq/kernels/kv_write.py`（设备无关参照
  `write_kv_slot`，含 lazy-init/同 buffer 语义）+ `tests/test_kv_direct_write.py`
  （逐步流逐位对照、越界哨兵、判别力自检 ×2；**F4 kernel 落地后在 GPU 上原样重跑
  即为它的 K 门**）；
- A/B + 金丝雀 + 归档 + 记录。

#### 风险

- `StaticCache` 是 HF 的内部接口，改动会绑死 transformers 版本（当前 4.57.6）——
  必须在 `ENVIRONMENT.md` 记版本约束；
- 与 `past_key_values.update()` 的语义耦合，是**最容易被上游破坏**的一处。

---

### F5 — `lm_head` 也量化成 w8a16（est 0.06 ms）

#### 现象

`gemvx ×1` = 0.1270 ms，读 206 MB fp16，达成率 **103.9%（已在墙上）**。w8a16 后字节减半
⇒ 地板 0.066 ms。

#### 为什么现在是 fp16

`keep_lm_head_fp16=True` 是**刻意的精度选择**（`src/lemoq/quant/linear.py:148`），
因为 lm_head 直接决定 logits，对 KL/top-1 最敏感。`EXP-p1-02` 的档 5 结论：
"只量化 experts、attention/lm_head 留 fp16 单独不够"。

★ **N7 口径（2026-10-25）**：F5 **不是** S2 意义上的"捕获缺口"——lm_head 实测 0.1270 ms
低于字节地板 0.1319 ms（达成率 103.9%，`gap ≈ −5 µs ≤ 0`），按 HANDOFF §5 的预注册规则
会被自动过滤（此前交接预演里的 "127 × 0.5 = 63" 用错了分子）。F5 的收益来自**把地板本身
减半**（0.1319 → 0.066 ms），那需要新 `GATE_CITATIONS` 行，属 S3、先问用户。

#### 改法

`--keep-lm-head-fp16` 关掉 + 重跑精度三链（KL/top-1、PPL、MMLU）。**门是主要成本，不是代码。**

#### 验收

- `GATE_CITATIONS` **不得**手改 —— 必须来自新测量 + 新工件（`benchmarks/runner.py` 注释明确）；
- 三链都要过：KL ≤ 门、PPL 漂移 <1%、MMLU 掉点 <2 点；
- 若任一门 FAIL，如实记 evidence，**不发布延迟**（硬规则 8）。

---

### F6 — 专家 GEMV 上 w4a16（est 0.30 ms，**阻塞**）

#### 为什么收益最大

专家权重 w8 = 805.3 MB，占一步字节的 **62.9%**（`bytes.breakdown` 实测）。再减半就是
整步最大的一块，理论上省 ~0.30 ms。

#### 为什么现在不能做

`GATE_CITATIONS` 里 `w4a16` 两行都是 **FAIL**：

```
("w4a16", 128, 1024): kl=0.093560, top1=0.866667   # top-1 远低于 0.95
("w4a16", 128, 4096): kl=0.059331, top1=0.866667
```

硬规则 2：门没过就没有性能结论。**这不是 kernel 问题，是量化算法问题**
（per-group 对称 RTN 在 4-bit 下精度不够）。

#### 正确的前置任务

按 [ROADMAP.md](ROADMAP.md) C2 单独立项：GPTQ/AWQ（或非对称 + 校准集）vs RTN 的对照，
同一模型同一门下比 PPL/KL。**只有门 PASS 之后，才轮到 F6 的 kernel 工作。**

---

## 3. 学术 / 工业界的融合方法（与本文机会的对应）

| 方法 | 核心思想 | 对应 |
| --- | --- | --- |
| [FlashAttention](https://arxiv.org/abs/2205.14135)（IO-aware） | 融合收益来自**减少 HBM↔SRAM 往返**；分块 + 在线 softmax 把中间量留在片上 | `_attn_split_kernel` 已是此路（71%/94% 达成率） |
| [Blockbuster](https://arxiv.org/abs/2505.07829) | 用 block program 显式建模内存层级，规则式融合；论文里**自动重新发现 FlashAttention**，并演示把 **RMSNorm + FFN-SwiGLU（三次 matmul + Hadamard + reduce + elementwise）融成单个 mega-kernel** | ★ F2 + F3：`gate_up → silu·mul → down → 残差` |
| [ClusterFusion](https://arxiv.org/abs/2508.18850) | QKV 投影 + Attention + O 投影**合成一个 kernel**，用 cluster 级 `ClusterReduce`/`ClusterGather` 把中间量留在片上；H100 上 1.61× | ★ F2 的 q/k/v 三合一 + attention |
| [Neptune](https://arxiv.org/abs/2510.08726) | 打破循环携带依赖 + 代数修正项，把"归约 → GEMM"链自动融成一个循环；平均 1.35× | `down_sum` 的 SPLIT_K 归约 + `out_reduce` |
| [DNNFusion](https://arxiv.org/abs/2108.13342) | 先按算子**数学性质改写图**再融合，融合机会多 8.8× | §4 注意事项 ⑤（RMSNorm 的 scale 可延后到 GEMM 之后） |
| [XLA 融合分析](https://arxiv.org/abs/2301.13062) | 归纳 XLA 的融合判定与实际收益；明确指出**融合不一定更快** | 我们的 36 点 config 网格差 3.2× 就是这个道理 |
| [GPUOS](https://arxiv.org/abs/2604.17861) | 单长驻 kernel + 原子任务队列，彻底消除 launch；小算子负载最高 15.3× | F1 的终极形态，但**与 CUDA Graph 互斥** |

---

## 4. 注意事项（红线，前三条是本项目踩过的坑）

**① 融合的收益 = 省下的 DRAM 往返，不是省下的 kernel 数。**
若两个算子的中间结果本来就落在 L2（96 MB），融合几乎无收益。判断标准只有一个：
把要融的那段手算字节账，看它离地板多远。RoPE 链能拿 8.39%，是因为 160 个 kernel 各自
只有 2–3 µs 的固定成本；`_gate_up_act` 已 78.8%，再融收益很小。

**② 融合会反噬 occupancy（live range 变长 → 寄存器压力上升）。**
本项目的证据就是那张 36 点网格：`BLOCK_N=128/SK=1` 让 `down_sum` 只有 16 个 program
落在 170 SM 上，**比最优配置慢 3.2×**（`moe_grouped.py:83` 注释）。融合越多，
每 program 的寄存器/共享内存越大，可能从"两个快的"变成"一个慢的"。
⇒ 每次融合后**重扫 config**，不要沿用旧值。

**③ 数值语义必须逐位对齐；冲突时**放弃融合**。**
最真实的一例：fused RoPE 的 bf16 路径 **1 ulp ≈ 8e-3，过不了 1e-3 的 K1 门** ⇒ 直接回退到
HF 逐位实现（commit `e5e0631`）。融合不是"顺手优化"，它必须先证明逐位等价。

**④ CUDA Graph 是硬约束，不是建议。**
- 不能用 `@triton.autotune`（capture 期间会真跑多 config，往图里塞计划外 kernel）；
- grid 必须是 host 侧常量；
- 不能有 device-side sync、不能动态分配；
- 新 kernel 必须在 capture **之前** warm 过（JIT 编译）。
⇒ 这也是"持久化 kernel / GPUOS 类方案"与 CUDA Graph **互斥**的原因。

**⑤ 不要跨"物化点"融合。**
中间量若被多处使用，烧进 kernel 会导致**重算**而不是重读。判断方法：数这个张量的消费者。
residual 只有 1 个（下一层）⇒ 可融；`hidden_states` 有 2 个（attention + MoE）⇒ 不可融。

**⑥ 先 profile 再融合，而且归因方法本身要先通过验证。**
`EXP-p4c-02` 的 **module-level 归因是 VOID 的**（它自己的预注册 K1 没过：eager/graph
top-8 重合 5/8，记录里如实写 void 而不是引用）。用错的归因去指导融合会融错地方。
正确做法是 `tools/profile_graph.py` 的**图内逐 kernel 分解** + 同轮 A/B。

**⑦ 小 kernel 用"每次 launch 多少 µs"记账，不用"占比"。**
`0.0544 ms / 32 次 = 1.7 µs/次` 才有决策价值；"2.8%"会让人以为不值得做。

**⑧ 融合要能回退、要能被测。**
所有候选做成 env 开关默认关 + `tokens_equal` 门，这样"候选"与"已采纳"的边界是机械可查的。

---

## 5. 复算命令（动手前先跑一遍）

```bash
cd /Users/tmchao7/repos/02_personal/LeanMoE-Quant

# 1) 账本：分类 + 达成率（本文 §1）
.venv-mac/bin/python - <<'PY'
import json, collections
p = "results/EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024/result.json"
d = json.load(open(p))
tk, p50 = d["bottleneck"]["top_kernels"], d["latency"]["per_token_p50_ms"]
print("p50 = %.4f ms, kernels/step = %s" % (p50, d["bottleneck"]["kernels_per_step"]))
for k in sorted(tk, key=lambda x: -(x["self_ms_per_token"] or 0))[:12]:
    print("  %-58s %8.4f %5.1f%%" % (k["name"][:58], k["self_ms_per_token"],
                                     100 * k["self_ms_per_token"] / p50))
print("attainment:", d["attainment"]["bw_attainment_pct"], "floor_T:", d["attainment"]["floor_t_ms"])
PY

# 2) 需要重测的 graph 格（F1 采纳时用）
.venv-mac/bin/python - <<'PY'
import json, glob
seen = set()
for p in glob.glob("results/**/result.json", recursive=True):
    try: d = json.load(open(p))
    except Exception: continue
    if d.get("status") != "done": continue
    c = d.get("config") or {}
    if c.get("execution") != "cudagraph": continue
    seen.add((d.get("exp_id"), c.get("bit_width"), c.get("attn"), c.get("ctx")))
print("需要重测的 graph 格：", len(seen))
for k in sorted(seen, key=str): print("  ", k)
PY

# 3) 三份融合 A/B 的现状
for f in fused_rope_ab routing_ab skip_mask_fill_ab; do
  .venv-mac/bin/python -c "
import json; d=json.load(open('results/$f.json'))
print('%-20s %-9s tokens_equal=%-5s faster=%s%%' % ('$f', d.get('status'),
      d.get('tokens_equal'), d.get('treatment_faster_pct', d.get('fused_faster_pct'))))"
done
```

---

## 6. 明确不做

| 不做 | 原因 |
| --- | --- |
| 在 `w4a16` 门 FAIL 的情况下报 4-bit 性能 | 硬规则 2/8；先修量化算法（ROADMAP C2） |
| 放宽 `PARITY_GATE` / `docs/reference/METRICS.md` §1 / `GATE_CITATIONS` | 永不 |
| 把三份 A/B 的收益相加相乘 | 硬规则 3（跨轮不可比） |
| 引入 `@triton.autotune` | 破坏 CUDA Graph capture 安全 |
| 持久化 kernel（GPUOS 类）改造 | 与 CUDA Graph 互斥，且本项目价值在 batch=1 的纯字节预算 |
| 为了"融合更多"而改矩阵轴 / cell id | 需先问用户（硬规则 9） |
| 在共享卡上有同租户时强行测 | 记录、等待、重跑；金丝雀 >5% 整轮 VOID |

---

## 7. 与其他文档的关系

| 文档 | 关系 |
| --- | --- |
| [RESULTS.md](../RESULTS.md) | 已采纳结果的唯一权威；本文的任何采纳动作都要回填到这里 |
| [ROADMAP.md](ROADMAP.md) | 展望与岗位映射；本文是其中"性能优化"这一支的可执行细化 |
| [reference/NCU_EVIDENCE.md](reference/NCU_EVIDENCE.md) | kernel 级达成率的出处与口径 |
| [reference/FUSION_RESEARCH.md](reference/FUSION_RESEARCH.md) | 联网调研：方法 + 十条避坑点（§3/§4 的外部依据扩展版；公开报告值非实测） |
| [handoff/HANDOFF_2026-10-10.md](handoff/HANDOFF_2026-10-10.md) | 交接摘要（状态，不是任务） |
| `scripts/rounds/run_p2j_refresh.sh` | F1 采纳流程的模板（31 格刷新的实际脚本） |
| [AGENTS.md](../AGENTS.md) | 硬规则与四门；本文所有"验收"都以它为准 |