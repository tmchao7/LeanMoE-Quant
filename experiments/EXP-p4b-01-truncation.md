# EXP-p4b-01-truncation

> **状态：`done` —— P1/P2/P3 与 `EXP-p2-01` 的 dense fused 臂都已落地，16/16 格收口（§2.9），
> E7 已判定（§2.8）。**
> ★ **量级已按新 kernel 刷新（2026-10-07 晚，P2d 轮）**：投影 kernel 在 `7d1f92b` 改成
> "按组广播 scale"（图内 8.14 → 5.15 µs/次），受影响的 **fused** 格已重跑；v1 工件归档在
> `results/EXP-p4b-01-run/v1_archive/`。刷新后的 **E7 判定数字**：fused 臂的
> `speedup_cudagraph` **1.1071 → 1.1703**（ctx=1024）/ **1.0512 → 1.0839**（ctx=4096）；
> separate 臂（不过那个 kernel）一字未动：0.7674 / 0.8530 —— 这正是"实现 > 位宽"的又一条对照。
> eager 列在 v2 里不可分辨（A/B 比 1.0093 → 1.0091，host 受限）。判定不变。
> ★ **16/16 格已收口（2026-10-07，§2.9）**：4-bit 的 nibble 解包落地后，p4b 声明的每个格子
> 都有落盘工件（8 格 gate PASS + 8 格 gate FAIL 的 evidence）。证据显示 **实现 > 位宽**：
> 同一个 `w4a16 × cudagraph`，separate 臂 _(9.12)_ ms / 1810 kernel vs fused 臂 **_(3.38)_ ms /
> 978 kernel**（2.7×）；而 `w8a16 → w4a16`（fused）的边际字节拿回了带宽地板的 **85%** ——
> 权重只占这一步 ~23%，所以"减半"换不到可观的绝对收益。★ W4A16 的门仍是 FAIL（0.0933 > 0.05），
> 那些数字**只能作 evidence**。
> ★ **E7 判定（2026-10-07，§2.8.5）**：截断效应**成立，但只在 fused 臂上兑现** ——
> 同一个 w8a16 实现只换 execution 轴：`speedup_eager` 0.76 → `speedup_cudagraph` **1.17**
> （ctx=1024；ctx=4096 是 0.79 → 1.05），`host_share` 77.5% → 1.2%。量级是 **1.05–1.11×**，
> **不是**矩阵猜的 2×（2× 那一行是 W4A16，引用门 FAIL 且 4-bit grouped 未实现，跑不了）。
> ★ 预注册的 `separate` 臂对照（v1 的语义）**没有**这个效应（0.767 → 0.767 @1024）：
> 它的成本是 GPU 侧的反量化物化，而 cudagraph 只拿得掉 host/launch ⇒ **P2-01 的融合是
> cudagraph 兑现量化收益的前提**。第 1 轮因共享卡上的外人进程整轮 **VOID**（§2.8.2）。
> ★ 本记录最重要的产出不只是门，而是 §3 的**依赖链**（P0–P3）。P1 已从"不存在"变成
> "已落地（8-bit、K1 门 PASS）"，见 §2.7。
> ★ §2.7 同时给出第一次量化臂实测：**负结果**（0.82×/0.80×），并定位了它的**混淆来源**
> （experts 走融合 kernel、attention 走 separate 臂）—— 这条直接决定做 `EXP-p2-01`，
> 而 `EXP-p2-01` 的结论（eager 融合 −5%、隔离测 +21%）又是 §2.8 判定 E7 的前提。

- **ID**: `EXP-p4b-01-truncation`
- **Date**: 2026-10-06（假设与门冻结）→ 2026-10-07（P2 补测 + fp16 基线 + P1 落地 + 量化臂首测
  + `EXP-p2-01` + **P3 接线与 E7 判定**）
- **Operator**: `benchmarks.runner`（`--bit-width {fp16,w8a16,w8a8} --execution {eager,cudagraph}
  --moe grouped --ctx {1024,4096} [--dequant {separate,fused}]`）+ `tools/quant_probe.py`（P2）
  + `tools/bench_grouped_gemm_quant.py`（§2.7 机制）
- **Status**: `done`（**16/16 格已测**：8 格 gate PASS、8 格 gate FAIL 的 evidence；
  E7 已判定 §2.8.5；4-bit 路径已落地 §2.9）
- **原始数据**: `results/EXP-p4b-01-truncation__bit_width=*_execution=*_ctx=*/result.json`；
  每轮的金丝雀/占用/日志在 `results/EXP-p4b-01-run/`（第 1 轮作废件在 `round1-void/`，
  P3 第 1 轮污染件在 `p3-*.log` 与 `v1_archive/`）；隔离 kernel 微基准
  `results/bench_quant_grouped.json`
- **对照**: `EXP-p1-04-prompt-set-gate`（门，含 ctx=4096 复测）、`EXP-p0-04`（静态 MoE + cudagraph）、
  `EXP-p2-02`（grouped + cudagraph，per-token 4.083 ms）、`EXP-p2-01`（E2：反量化融合 vs 独立 ——
  它的 fused 臂就是 §2.8 里让量化第一次 >1× 的那个实现）

★ **2026-10-24 R3 de-publish**：`w8a8@4096` 的引用门按 10 窗复测**重判为 FAIL**（KL 均值 0.0262 > 0.01，重尾 w7 单窗 0.2247；top-1 均值 0.963636 与 5 窗同值），用户批准撤销其已发布延迟 —— 下文 F4 刷新的 **8.8232** 与 F1+F2 刷新的 **8.8189** 自此**只作 evidence**（`gate_failed_evidence`），不再是性能结论；门行现引 `results/EXP-p1-04-r3-10win-ctx4096.json`。详见 `EXP-p1-04` v3 小节。

★ **2026-10-24 F4 刷新**：F4 作用域=kernel 格，本实验（sdpa）格仅重盖章（config 补
`kv_direct_write=False`）；p50 噪声内：fp16 3.5446/6.7805、w8a16 2.9589/6.2219、
w8a8@4096 8.8232（**R3 de-publish 后已撤销发布**）；E7 = **1.1979/1.0898**（2026-10-24 值 1.1974/1.0912 留档）。

★ **2026-10-25 F5 采纳刷新 = v6（canonical）**：lm_head 也量化（六 True）。三链 PASS + A/B VALID
（−63.3 µs，−4.13%，5/5 对，canary 0.131%；前两轮宿主机共租尖峰 fail-closed 作废，归档
`EXP-f5-ab-20261025-invalid-run{1,2}`）+ 刷新 22/22 干净（canary −0.058%）。新采用 p50（cudagraph）：
fp16 **3.5628/6.8445**、w8a16 **2.9149/6.226**；E7 `speedup_cudagraph` = **1.2223（1024）/ 1.0994（4096）**
（F4 值 1.1979/1.0898 留档）。量化步/fp16 步份额 **41.2%/25.1%** ⇒ 上限 ~1.70×/~1.34×。
w4a16 两格 gate FAIL 如旧（evidence）；w8a8@4096 格 gate FAIL 如旧（R3 裁决，照跑只作 evidence）。
逐格 `lm_head_w8_effective` 量化格 True / fp16 格 False（刷新谓词机械执行）。

★ **2026-10-24 采纳刷新（F1+F2 默认开）**：`FUSION_DEFAULTS` 四 True 后 23 格同轮重测（`results/EXP-f1-refresh/`，canary 0.30%、逐格 host_share ≤3.4%）。新采用 p50（cudagraph）：fp16 **3.5421/6.7841**、w8a16 **2.9582/6.2172**、w8a8@4096 **8.8189** ms（**R3 de-publish 后已撤销发布**）；E7 `speedup_cudagraph` = **1.1974（1024）/ 1.0912（4096）**（P2j 值 1.1911/1.0888 留 §P2j，跨轮不混比）。w4a16 两格 gate FAIL 如旧（8.5879/11.8202 仅 evidence）。无 env 金丝雀：kernel 格 1.5590（Δ+0.03%）、sdpa 格 2.9542（Δ−0.14%，`packed_qkv_effective=False` 正确 no-op）。

## Hypothesis（先写，跑之前冻结；来自矩阵，原文不动）

```text
statement:        量化对 token 延迟的收益被 host/launch 开销截断：
                  W4A16+eager < 1.5x，W4A16+cudagraph > 2x
refutation_signal: 两者接近 → 截断效应不存在，量化可独立兑现
combined:         true（同时改 bit_width 与 execution —— 矩阵已显式标注）
```

### ★ 假设与正确性门的冲突（跑之前必须写清楚，否则会自欺）

假设的量化臂是 **W4A16**，但 `EXP-p1-04` 实测 W4A16 的门是 **FAIL**
（5 窗口真实 prompt 集 mean KL 0.0933 > 0.05）。按 `AGENTS.md` 硬规则 2：

> A performance number may only be published for a configuration whose KL / top-1
> has passed the thresholds. A faster-but-different model is a failed experiment.

**后果（本记录冻结）：**

1. 假设里 W4A16 的两个定量预测（`< 1.5x` / `> 2x`）**在现有门下永远不会作为性能结论
   被发表**。要把它们变成可发表的，需要的是先让 W4A16 过门（阶梯第 2/3 档），
   不是放宽门 —— 门不动。
2. **E7 的结构性主张可检验**，用**唯一过门**的量化臂 `w8a16`（mean KL 0.001273 < 0.01）：

   ```
   speedup_eager     = p50(fp16, eager)     / p50(w8a16, eager)      # 同 MoE、同 ctx
   speedup_cudagraph = p50(fp16, cudagraph) / p50(w8a16, cudagraph)
   截断成立           : speedup_cudagraph 与 speedup_eager 的差 > 本会话金丝雀测出的噪声带
   否证（矩阵原文）   : 两者接近（在噪声带内）→ 截断效应不存在
   ```
   不发明新的数值阈值：矩阵给的 1.5/2 是 W4A16 的，对 w8a16 只用定性判据 + 实测噪声带。
3. W4A16/W8A8 的格子照跑（它们是"门失败"这一事实的 evidence），`verdict=FAIL`，
   **其延迟不得引用为性能结论**。

### `combined: true` 的合法性（硬规则 3）

规则 3 允许 combined 的前提是"个体效应已知"：

- **execution 轴**：`EXP-p0-04` 已单独测过 fp16 的 eager vs cudagraph（含 1024/4096 两个 ctx）；
- **bit_width 轴**：`EXP-p1-01` → `EXP-p1-04` 已单独测过量化对精度的影响（并修好了仪器）。

个体效应都在手，本条测的才是**交互**（截断），不是三个变量的混合。

## 1. 为本条做的矩阵改动（两条，均已实现并有测试）

```yaml
factors:
  bit_width: ["fp16", "w8a16", "w8a8", "w4a16"]   # + w8a16（4 格），共 16 格
config:
  moe: grouped                                     # 新增：两臂必须同一 MoE 路径
```

1. **`+w8a16`**：与 `EXP-p2-01` 同一个理由 —— 不加它，E7 的量化臂全是 `verdict=FAIL` 的行，
   头号结论没有任何可发表的数字。测试 `test_e7_truncation_matrix_is_complete` 已同步到 16 格，
   并断言 `bit_width` 集合与**两臂同一种 moe**。
2. **`config.moe: grouped`**（硬规则 1，两臂同一路径）：
   - `hf` **不能 capture** —— `EXP-p0-03` 就是这么 blocked 的，cudagraph 臂必须 static/grouped；
   - 若 eager 用 hf、cudagraph 用 grouped，execution 轴就和 MoE 实现混在一起，测出来的
     "截断"分不清是 host 开销还是 MoE 实现差异；
   - 选 `grouped` 而不是 `gather`（旧名 static）：`EXP-p2-02` 已在 grouped+cudagraph 下过门
     （per-token 4.083 ms、TTFT 55.28 ms），是当前最优 fp16 路径；`gather` 的 gather 税
     （3.2212 GB/token，占单步 39%）是**设备侧**代价，会盖住 E7 要测的 **host** 侧开销。

## 2. 今天离线已交付的 runner 能力（全部有离线测试）

| # | 内容 | 验证 |
|---|---|---|
| 1 | `run_baseline` 接受 `moe=` 并做 MoE 替换（eager 臂从此能用 grouped/gather） | `inspect.signature` + docstring 断言 |
| 2 | `_weight_bytes_actual` 统一补三类 `parameters()` 看不见的权重：`QuantLinear` buffer、`_static_experts`、tie | 两条 CPU 测试（替换前后字节守恒、量化后必须比不传 qreport 多算） |
| 3 | 第四条拒绝：**量化 × 非 hf MoE** 在加载前失败 | CLI 测试（fp16 臂不受影响） |
| 4 | 修正量化 × cudagraph 拒绝的**理由**（原来写"接线未完成"，实际是 §3 的 P1 缺失） | 同上族测试 |
| 5 | `dry-run` 打印 `moe=`；payload `config.moe/moe_blocks_replaced` 记录 | dry-run 测试断言 `moe=grouped` |
| 6 | **顺带修好一个先于本条存在的 bug**：`static` 是旧名，实现叫 `gather`（`VALID_MOE_IMPLS=("gather","grouped")`，随 grouped kernel 一起改的）。今天 `--moe static`、`--static-moe` 别名、`scripts/run_cudagraph_gate.sh`、以及矩阵里 p0-04 的声明**都会撞 `unknown moe 'static'`**（p0-04 的格子是 done，所以一直没人触发）。已在 main 单点归一化 `static→gather`，CLI choices 补 `gather`，docstring 改名 | 归一化后的 dry-run 通过；测试用 `impl="gather"` |

## 3. ★ 依赖链（E7 要跑起来，按顺序满足）

```
P0  eager 臂接 MoE 替换 + static→gather 归一      ✅ 已交付（§2）
P1  量化 × 可 capture 的 MoE 路径                  ✅ **已落地（8-bit）**（§2.7）
P2  ctx=4096 的量化门实测                          ✅ 已完成（2026-10-07，见 §2.5）
P3  量化 + cudagraph 的 quant_cfg 接线             ❌ 仍未做 —— 现在是 E7 判定的唯一阻塞
```

### P1 为什么**曾经**不存在（保留作分析；2026-10-07 已落地，见 §2.7）

E7 的两臂要各自内部 like-for-like：`fp16` vs `量化` 必须在**同一个** MoE 路径上比。
把四条路都摆出来：

| | eager | cudagraph |
|---|---|---|
| **hf** | ✅ fp16、✅ 量化（p1/p2-01 的路径） | ❌ **不能 capture**（`EXP-p0-03` blocked：逐专家 Python 循环 + 数据依赖形状） |
| **grouped** | ✅ 量化 × grouped **不存在** | ✅ fp16（`EXP-p2-02` done）；❌ 量化 × grouped **不存在** |

`量化 × grouped` 不存在的原因是结构性的，两个方向都堵：

- **先量化再替换**：`replace_moe_blocks → StaticShapedMoE → fuse_experts` 读的是
  `experts[i].gate_proj.weight`；`QuantLinear` 没有 `.weight`（权重在 packed buffer 里）→ 崩；
- **先替换再量化**：替换会 `del block.experts` 并把权重 fuse 成 `FusedExperts(gate_up, down)`
  **普通 dataclass 张量**（不在 `parameters()`、也不是 `nn.Linear`）→ `quantize_model`
  找不到任何 expert 模块，**静默只量化 attention**（配置声称量化了 experts 但没有）。
- 换 `gather`（静态 gather+bmm）同样堵：它也吃 fuse 出来的 fp16 张量。

所以 E7 的量化臂需要**新的东西**，两条候选（都不在本记录里实现）：

1. **量化感知的 expert kernel**：grouped GEMV 直接读 packed 权重 + per-group scale
   —— 就是 `EXP-p2-01` 的 fused dense kernel 做到的事，推广到 expert 的 grouped 形状
   （`moe_grouped._gate_up_act_kernel` 的量化版）。真正的"融合"，也最符合本项目的技术判断；
2. **先 fuse、再量化 fused 张量 + 每步反量化**（`separate` 语义搬进 `static_moe_forward`）：
   capture 安全（静态形状），但**每步物化 gate_up** ≈ 536 MB/step —— 这个代价本身就是
   E2/E7 要测的对象，可以先用它做上界，但结论必须写明"这是 separate 臂"。

> 判决：**P1 是 P2/P3 的前置**，也是整个 Phase 4b 的前置。
> 在 P1 落地前，p4b 能诚实跑的只有 fp16 的 4 格（把 execution 轴的 fp16 基线打出来）。
> **量化 × hf × eager** 可以用 `--moe hf` 覆盖跑出来当 evidence（`cfg.moe=hf` 会被如实记录），
> 但它与 grouped 臂不同路径，**不进 E7 的任何比值**。

### P2：ctx=4096 的门 —— ✅ 已完成（2026-10-07）

`GATE_CITATIONS` 原来只覆盖 `(bit_width, g128, ctx=1024)`。ctx=4096 的 prompt 不同 → 轨迹不同 →
KL 是**新测量**。已上卡补测（同仪器：`tools/quant_probe.py --ctx 4096 --prompt-windows 5
--include-ones-control`，约 15 分钟），原始工件
`results/EXP-p1-04-prompt-set-gate/probe_prompt_set_ctx4096.json`，完整结果写在
`experiments/EXP-p1-04-prompt-set-gate.md` 的「ctx=4096 复测」节。要点：

| 臂 @4096 | KL mean | top-1 mean | 合取判定 |
|---|---|---|---|
| `w8a16` | 0.000714 ✅ | 0.9818 ✅ | **PASS** |
| `w8a8` | 0.004606 ✅ | 0.9455 ❌ | **FAIL**（top-1 拦下） |
| `w4a16` | 0.053876 ❌ | 0.8545 ❌ | **FAIL** |

★ 这条补测暴露了一个**必须写进代码**的陷阱：`w8a8@4096` 的 KL 已进 0.01 门，
**只看 KL 就会判 PASS 并允许发表延迟** —— 而它的 top-1 不过门。按 `METRICS.md` §1 的
合取规则，`runner.GATE_CITATIONS` 现在存实测的 `kl` / `top1`，`gate` 由
`gate_from_metrics()` **推导**（不再手写），并有测试钉住这条合取。这条规则**不改动任何
已有判定**（w8a16@1024 两判据都过；w8a8/w4a16@1024 的 KL 本就超门）。

**补测之后**：量化 × ctx=4096 不再因"没有门实测"被拒绝；但在 grouped MoE 上仍被 P1 拒绝。

## 2.5 第 1 轮 fp16 基线：**VOID**（占用导致，2026-10-07）

按 §7 第 2 步跑了 fp16 的 4 格 + 首尾金丝雀。**本轮按预登记协议作废**，数字不得引用：

| 检查 | 实测 | 后果 |
|---|---|---|
| 金丝雀 `fp16×eager×ctx=1024` 首 vs 尾 | p50 15.5417 → 9.7763 ms，**相对差 37.10%** | 阈值是 **5%** → **本轮作废**（§5 K2.1） |
| `cudagraph×ctx=1024` | **`blocked`**：capture 阶段 `AcceleratorError: CUDA error: out of memory` | 无延迟数字 |
| 占用（`nvidia-smi --query-compute-apps`） | 外人进程 `pareto_offload.py`（另一条工作线 `EXP-mf-p1-09`），峰值 **17.8 GB** | **记录、不杀**（用户明确要求；§5 K2.2） |

机制自洽：金丝雀 A 的 per-token 样本是**双峰**的（~15 ms 与 ~55 ms 交替），
正是共享卡上另一个作业做 burst 的特征；B 段的样本回到单峰 ~9.8 ms。
`cudagraph` 的 capture 需要 ~13.8 GB 权重 + 图缓冲，被外人进程的 17.8 GB 挤爆 —— 这是
**OOM（资源竞争）**，不是"cudagraph 在这张卡上不可行"（`EXP-p2-02` 已在干净卡上成功）。

原始证据保留在 `results/EXP-p4b-01-run/`（`canaryA/B_result.json`、`contention.log`
逐格占用快照），作废的 `result.json` 归档在 `round1-void/`。**等卡空闲后重跑**（重跑见 §2.6）。

## 2.6 第 2 轮 fp16 基线：**有效**（2026-10-07）

等外人进程退出（`nvidia-smi` 回到 0% / 2 MiB）后按同一协议整轮重跑，全部 `--force`。

**金丝雀（本会话噪声带）**：`fp16×eager×ctx=1024` 首 vs 尾
p50 = **9.7324 → 9.7420 ms**，相对差 **0.10%** ≪ 5% → **本轮有效**。
两组样本都是单峰（9.62–10.37 / 9.66–9.85），无第 1 轮那种 ~55 ms 的 burst 尾巴。
每格计时前后的 `nvidia-smi --query-compute-apps` 都是空（无外人进程）。

**execution 轴的 fp16 基线**（MoE=grouped，batch=1，dtype=fp16，warmup 5 / timed 20）：

| 格 | status | per-token p50 | p99 | TTFT p50 | `host_share` | `gpu_p50` | kernels/call |
|---|---|---|---|---|---|---|---|
| `fp16 × eager × 1024` | done | **9.732 ms** | 10.28 | 56.01 | **67.3%** | 9.564 | 981 |
| `fp16 × cudagraph × 1024` | done | **4.070 ms** | 4.07 | 57.50 | **1.4%** | 4.070 | 978 |
| `fp16 × eager × 4096` | done | **9.671 ms** | 12.26 | 111.87 | **60.6%** | 9.507 | 981 |
| `fp16 × cudagraph × 4096` | done | **7.311 ms** | 7.32 | 139.84 | **0.7%** | 7.311 | 978 |

### 1. 执行方式本身就是 1.3–2.4×（`host_share` 从 ~65% 掉到 <2%）

```
ctx=1024:  eager 9.732  -> cudagraph 4.070  = 2.39x   (host_share 67.3% -> 1.4%)
ctx=4096:  eager 9.671  -> cudagraph 7.311  = 1.32x   (host_share 60.6% -> 0.7%)
```

这两条正是 E7 要的**机制量**：`host_share_pct` 是"host/launch 开销占总步时的比例"，
cudagraph 把它压到 <2% —— 即**在 fp16 上就已经把 host 侧开销几乎消掉了**。
这就是为什么"截断"判定必须用 `speedup_cudagraph` 与 `speedup_eager` 的**差**（§6）。

### 2. 交叉核对与既有记录一致（不是孤立数字）

- `cudagraph×1024 = 4.070 ms` vs `EXP-p2-02` 的 **4.083 ms** —— 差 0.3%，同一条路径的重现；
- `cudagraph×4096 = 7.311 ms` 比 1024 慢 1.80×（KV 变长、attention 占比上升），
  与 `EXP-p0-04`/`p2-03` 的方向一致。

### 3. ★ 这条记录**仍不能判定 E7**

E7 的判据（§6）要比较**量化臂**的两个 speedup：

```
speedup_eager     = p50(fp16,eager)     / p50(w8a16,eager)
speedup_cudagraph = p50(fp16,cudagraph) / p50(w8a16,cudagraph)
```

现在只有分子（fp16 四格），**分母（w8a16）不存在** —— 量化 × grouped MoE 的路径还没实现
（P1）。所以本记录的 verdict 仍是"**不可判定**"，不是"支持"或"否证"。
按纪律：**不给出任何 E7 结论，也不把 fp16 的 1.32×/2.39× 说成"量化收益"** ——
它们是 execution 轴的 fp16 基线，与量化无关。

### 4. 正确性门

四格的 `correctness.gate`：`eager` 走自洽性门，`cudagraph` 走图回放忠实性门
（与**同路径 eager** 比：两格都是 `kl=0.0`、`top1_agreement=1.0`）→ 四格 **PASS**。
fp16 无 KL 门可引（本配置即参考），与 §5 K1 一致。

## 2.7 P1 落地与第一次量化臂测量（2026-10-07）

### P1 是什么（记录 §3 的候选 1，已实现）

新增 `lemoq.kernels.moe_grouped_quant`：**在融合后的 (E,2F,H)/(E,H,F) 张量上做
per-group RTN**，packed int8 + fp16 scale 留在 GPU 上，decode 由新 kernel
（`_group_up_act_kernel_q` / `_down_sum_kernel_q`）**在寄存器里解包** ——
不物化 fp16 权重。形状全静态、expert id 只进指针算术 → 与 fp16 grouped 一样可 capture
（P3 的前提）。

- **K1 门（kernel 级）已在 5090 上 PASS**：量化 kernel vs 同权重 fp32 oracle，
  容差 `PARITY_GATE`（atol/rtol 1e-3），无放宽。
  `tests/test_kernel_parity.py::test_triton_quantized_grouped_matches_oracle`。
- **语义锚点**（有测试钉住）："先融合再量化" 与 "逐专家量化再融合" **逐位相同**
  （RTN 逐行逐组，融合不跨行）—— 否则 P1 会悄悄换掉 p1-04 门的量化语义。
- 边界（显式报错，不静默算错）：4-bit、量化 × gather、`BLOCK_K` 不整除 group_size、
  hidden/ffn 不被 group_size 整除。
- 顺序说明：记录 §3 建议"fused dense kernel 先行"，本会话先做 **grouped experts**
  （experts 占 `[W]` 的 96%，是 E7 的关键路径）；`EXP-p2-01` 的 dense fused 臂仍待实现，
  它现在是**下一步**（理由见下面 §3 的机制发现）。

### 单元格测量（golden canary 0.70% → 本轮有效）

`fp16` vs `w8a16`，两臂同走 grouped、同 eager，逐格相邻执行（`--force` 重测）：

| 格 | per-token p50 | p99 | TTFT | `host_share` | `gpu_p50` | `[W]`/token | attainment |
|---|---|---|---|---|---|---|---|
| `fp16 × eager × 1024` | **9.9021 ms** | 12.52 | 58.09 | 68.0% | 9.95 | 2.36 GB | 15.4% |
| `w8a16 × eager × 1024` | **12.0154 ms** | 14.04 | 64.05 | 63.1% | 11.97 | **1.30 GB** | 6.9% |
| `fp16 × eager × 4096` | **9.9086 ms** | 10.32 | 112.70 | 61.9% | 9.81 | 2.36 GB | 15.2% |
| `w8a16 × eager × 4096` | **12.3566 ms** | 15.20 | 125.40 | 59.3% | 12.14 | 1.30 GB | 6.7% |

```
speedup_eager = p50(fp16)/p50(w8a16)
  ctx=1024:  9.9021 / 12.0154 = 0.824x
  ctx=4096:  9.9086 / 12.3566 = 0.802x
```

`[W]` 从 2.36 GB 降到 1.30 GB（**55.2%**，与 p1-04 的"49.4%+scale"预算一致），
**但延迟变慢 20–24%**。这是本项目的第一个量化臂实测，且它是**负结果**。

### ★ 机制：慢的不是 expert kernel（隔离微基准）

`tools/bench_grouped_gemm_quant.py`（同进程、交错、OLMoE 形状 E=64/2F=2048/H=2048/K=8）：

| 配置 | fp16 p50 | int8 p50 | int8/fp16 | fp16 有效带宽 | int8 有效带宽 |
|---|---|---|---|---|---|
| 默认 | 0.1090 ms | **0.0905 ms** | **0.830** | 923 GB/s | 565 GB/s |
| 扫描中最好的 int8（num_warps=8） | 0.1095 ms | 0.0969 ms | 0.885 | 919 GB/s | 527 GB/s |

**结论（两条，都重要）**：

1. **量化 expert kernel 本身是更快的**（0.830×，即省 17%），方向与字节数一致；
   但字节比是 0.508，所以它**远没有**达到字节比例 —— 即便隔离测，int8 也只用到
   实测带宽的 **36%**（fp16 是 59%）。这条路径**不是带宽受限的**，
   所以"权重少一半"不可能换到"时间少一半"。扫描没能找到比默认更快的配置。
2. 因此单元格上那 +2.02 ms（gpu_p50 9.95 → 11.97）**不能**归给 expert kernel ——
   按隔离测，expert 侧应该**省 0.30 ms/token**。差在别处：**这一格同时量化了
   attention**（`quant_spec_for_cfg` 的 targets=attention+experts，与门逐字一致），
   而 attention 走的是 `QuantLinear` 的 **separate 臂**（每次 forward 把 packed 反量化成
   fp16 再 GEMM）。那正是 `EXP-p2-01`（E2）要测的慢路径，也解释了 `host_share`
   虽然从 68% 降到 63%、但 `gpu_p50` 反而涨了。

> **诚实结论**：这一次测量把两个不同的反量化语义混在了一格里 —— experts = **融合**
> （新 kernel），attention = **separate**（旧 QuantLinear）。所以它**不能**回答
> "量化能不能兑现收益"，只能回答"当前这条混合路径是净负的"。要拆开，必须让 attention
> 也走融合 kernel —— 那就是 **P2-01 的 dense fused 臂**，它从"建议先做"变成了
> **下一步的硬前提**。

### ★ 归因已被**直接测量**证实（不是推断）

`tools/bench_quant_linear.py`：隔离测 `QuantLinear`（separate 臂）在两 token 的
attention 形状上的反量化税（同进程交错）：

| 投影 | fp16 p50 | separate p50 | 倍数 |
|---|---|---|---|
| q_proj 2048×2048 | 0.02077 ms | 0.05434 ms | ×2.62 |
| k_proj 512×2048 | 0.02067 ms | 0.05475 ms | ×2.65 |
| v_proj 512×2048 | 0.02074 ms | 0.05490 ms | ×2.65 |
| o_proj 2048×2048 | 0.02064 ms | 0.05458 ms | ×2.64 |

```
每 token（16 层）：fp16 1.3251 ms  →  separate 3.4970 ms   delta = +2.1719 ms
```

**与单元格实测对得上**：单元格 `gpu_p50` 的差是 **+2.02 ms**（9.95 → 11.97），
而 expert kernel 按隔离测应**省 0.30 ms/token** → 预计净差 ≈ 2.17 − 0.30 = **1.87 ms**，
与实测 +2.02 ms 同量级（差 0.15 ms，在两条独立微基准的误差内）。

⇒ **量化臂这次的负结果，几乎全部来自 attention 的 separate 臂**（每次 forward 物化
fp16 权重），而不是"量化权重没收益"。expert 侧的新 kernel 方向是对的（隔离测快 17%）。
这同时说明：**在 expert kernel 只有 36% 带宽达成率、attention 还在物化权重的情况下，
"量化能不能兑现收益"这个问题现在无法回答** —— 而它正是 E2/E7 的核心问题。

### ★ 上一条归因的**修正**（2026-10-07，`EXP-p2-01` 的 dense fused 臂落地后）

上面 §2.7 把 +2.02 ms 归给"物化 fp16 权重的写+读流量"。`EXP-p2-01` 把物化**整条去掉**
（fused 臂一趟只读 packed、在寄存器里解压）后，**单元格没有变快，反而慢了 5%**：

| `EXP-p2-01`（`moe=hf`，同进程交错 60 对） | sep p50 | fus p50 | ratio fus/sep | Δ 中位 | MAD |
|---|---|---|---|---|---|
| `w8a16` | 192.06 ms | 201.79 ms | 1.0507 | −10.08 ms | 2.51 |
| `w4a16` | 398.05 ms | 384.36 ms | 0.9656 | +13.43 ms | 1.81 |

⇒ 那 +2.02 ms **不是（或不只是）DRAM 流量**，而是"每次 forward 多几个 torch 算子 +
一次多余的 GEMM 调用"的 **host/launch 成本**：`EXP-p2-01` 的格子里 `host_share`
60.7–80.4%、每步 1.9–4.3 万个 kernel、带宽达成率 0.4–0.5%，isolated 测里 fused 每调用
33.5 µs 中有 ~16 µs 是 triton launcher 地板。**预测**（非测量）：把 attention 换成 fused
后，p4b 的 grouped 格大约只能拿回 0.4–0.5 ms（≈ 2.02 ms 的 1/4）；要真测，得先给
cell id 加 `moe` 轴，否则 grouped 重跑会覆盖已落盘的 hf 格。详见
`experiments/EXP-p2-01-dequant-fusion.md` §6.4–§6.5。

**对 P3/E7 的意义（收紧，不是放松）**：既然 eager 臂的税主要是 host/launch，那么
`cudagraph`（p4b §2.6：fp16 上 host_share 67% → 1.4%）才是让 fused kernel 的
GPU 优势（2.6 µs vs 28 µs 工作量）兑现的前提 —— 这也让 P3 的优先级更高。

> ★ **2026-10-07 已完成并验证**：P3 落地后，fused 臂的 `speedup_cudagraph` = **1.11**（ctx=1024）
> vs `speedup_eager` 0.76；而 separate 臂仍是 0.767 → **§2.8.5 的 E7 判定**。

### P3 的真实工作量（上卡前先看清，2026-10-07 排查）

P3 **不只是**"把 `quant_cfg` 传给 `run_cudagraph`"。排查发现两处必须先改，否则会写出
**错的记录**：

1. `run_cudagraph` 完全没有 `quant_cfg` 参数 —— 量化模型进不了 capture（当前是拒绝）。
2. `build_payload_cudagraph` 里 `config.bit_width` **硬编码 `"fp16"`**，且
   `correctness.gate` 只用了 `graph_correctness_passed`（图 vs 同路径 eager）。
   直接接线的后果：量化格会写成 `bit_width: fp16` 且**跳过引用门**
   （`EXP-p1-04` 的 KL/top-1）—— 那正好违反硬规则 2。P3 必须同时把
   `correctness_for_cell(cfg, consistency_passed=…)` 接进来（引用门 ∧ 图回放门）。

**在哪之前不要跑**：`w8a8` 的 cudagraph 格现在会被这条漏掉的门放行。

### E7 判定：2026-10-07 已判定（P3 落地）→ 见 §2.8

当时只有 eager 臂（cudagraph 臂要 P3 接线），所以那 0.82×/0.80× 只是 **eager 臂的实测比值**，
且混了两种反量化语义。**P3 已在同一天落地并跑完，判定写在 §2.8。**

## 2.8 P3 落地与 **E7 判定**（2026-10-07）

### 2.8.1 P3 接线：三处必须先修的坑

| # | 坑 | 修法 | 实测验证 |
|---|---|---|---|
| 1 | `run_cudagraph` 没有 `quant_cfg`（量化进不了 capture） | 量化在 capture **之前**：`replace_moe_blocks(impl, quant=qspec)` → `quantize_for_cfg`；量化 grouped kernel 的首调编译落在 `capture()` 自己的 warmup 相位（专用 stream、capture 之前）→ **fused kernel 因此可 capture** | w8a16 × cudagraph 的 6 个格子全部 capture/replay 成功 |
| 2 | `build_payload_cudagraph` 把 `config.bit_width` 硬编码 `"fp16"`，且 `gate` 只用图回放门 | 三个字段来自 cfg；量化格的门 = `correctness_for_cell(cfg, consistency_passed=图回放门)` = **引用门 ∧ 图回放门** | w8a8 × cudagraph 的 gate = **FAIL**（引用 KL 0.0328/0.0046）→ 延迟只进 `_measurement.gate_failed_evidence`。**这就是"跳过引用门"会放行的那一格** |
| 3 | `cudagraph.blocked_result` 也硬编码 fp16（本轮实测撞上） | 三个字段来自 config | 第 1 轮一次 capture OOM 写出的 payload 里 `bit_width=fp16` —— 已改，见 §2.8.2 |

第 3 条是"写错记录"的实例，不是理论：被阻塞的量化格**必须以量化位宽落盘**，
否则"这一格为什么没有数字"就是一条错的记录。

### 2.8.2 第 1 轮：**VOID**（共享卡上的外人进程；记录，不杀）

19:17:03–19:22:10 期间 pid **31929** 占着卡（峰值 **14.0 GB**，逐格快照见
`results/EXP-p4b-01-run/p3-contention.log`）：

| 症状 | 实测 | 干净时的值 |
|---|---|---|
| `w8a16 × cudagraph × 1024`（fused）capture | **OOM**：`Tried to allocate 1024 MiB … this process has 17.49 GiB … Process 31929 has 13.38 GiB` | 该格在 v2 成功（3.6737 ms） |
| `fp16 × cudagraph × 4096` | **22.0633 ms**、host_share **67.1%** | v1 7.311 ms / 0.7%；v2 7.3211 ms / 0.8% |
| `w8a16 × cudagraph × 4096`（fused） | 14.9888 ms、host_share 54% | v2 6.9645 ms / 0.9% |

★ **教训（写进协议）**：`EXP-p2-01` 用的 eager 金丝雀对这种污染**不敏感** —— 那一轮 eager 金丝雀只漂
1.16%，而上面三格全废。所以 v2 加了两件事：（a）**cudagraph 臂自己的同格首尾金丝雀**；
（b）每格后的 payload 自检：graph 格的 `host_share > 5%` 就是"有共租户"的硬信号。
协议仍是**记录不杀**，等它退出后整轮重跑；v1 的 `result.json` 未用于任何结论。

### 2.8.3 第 2 轮：**有效**（两张金丝雀都 <1%，14 格 exit=0、`schema problems: 0`）

| 金丝雀（同格首尾） | 首 | 尾 | 相对差 | 门槛 |
|---|---|---|---|---|
| `fp16 × eager × 1024`（K2.1 指定） | 9.7751 ms | 9.7223 ms | **−0.54%** | 5% |
| `w8a16 × cudagraph × 1024`（本轮新增，cudagraph 噪声带） | 5.2998 ms | 5.3028 ms | **+0.06%** | 5% |

每格计时前后 `nvidia-smi --query-compute-apps` **均为空**（`p3-v2-contention.log`），
且脚本在每格后打印 payload 自检（无 `!!`）。

### 2.8.4 数据（同轮、相邻执行；p50，单位 ms）

| 格 | eager | cudagraph（separate 臂） | cudagraph（fused 臂） | `host_share` eager→cg | `[W]` |
|---|---|---|---|---|---|
| `fp16 × 1024` | 9.7751 / 9.7223（金丝雀 A/B） | **4.0670** | — | 67.0% → 1.1% | 2.358 GB |
| `fp16 × 4096` | 9.8716 | **7.3211** | — | 61.2% → 0.8% | 2.358 GB |
| `w8a16 × 1024` | 12.3909（A/B sep）/ 12.5036（A/B fus，v2） | **5.2998 / 5.3028**（金丝雀 A/B） | **3.4751**（v1: 3.6737） | 72.2% → 1.6% | 1.301 GB |
| `w8a16 × 4096` | 12.5658 / 12.7140（v2） | **8.5825** | **6.7545**（v1: 6.9645） | 73.6% → 0.9% | 1.301 GB |
| `w8a8 × 1024`（evidence） | — | _(6.0640)_ | — | — → 0.4% | 1.301 GB |
| `w8a8 × 4096`（evidence） | — | _(9.3360)_ | — | — → 0.2% | 1.301 GB |

- eager 的 `w8a16` 格是**同进程交错 A/B**（每臂 60 样本）：`separate` 与 `fused` 差 0.9%/0.5%，
  `separate` 胜 54/60 与 41/60 —— 与 `EXP-p2-01` 的 5% 负号一致（host-bound，融合拿不到）。
- 图的量化格口径已经正确：`[W]` 用**量化后**的分母（1.301 GB）、门是引用门 ∧ 图回放门
  （`correctness.kl = 0.001273`（引用）、`correctness.graph_replay.kl = 0.0`、
  `_measurement.dequant.activity` 显示 `arm=fused` 时 `fused_calls = 4480`）。
- graph 格每格只有 **3 个 repeat**（既有协议：`measure_graph_gap(repeats=3)`，每 repeat 是 20 次重放的均值），
  所以 p99 基本等于 p50（例：4.0670/4.0678）。这是既有口径，本轮没有改它。

### 2.8.5 ★ E7 判定：**截断效应成立，但只在 fused 臂上兑现**

`speedup = p50(fp16) / p50(w8a16)`（**>1 = 量化更快**；fp16 取同轮金丝雀/同轮格）：

| ctx | 量化臂 | `speedup_eager` | `speedup_cudagraph` | 差（cg − eager） | `host_share` |
|---|---|---|---|---|---|
| 1024 | `separate`（v1 的语义，预注册对照） | 0.7855 | 0.7674 | −0.0181 | 77.5% → 0.6% |
| 1024 | **`fused`（P2-01 的实现）** | 0.7775 | **1.1703** | **+0.3928（51%）** | 77.5% → 1.6% |
| 4096 | `separate` | 0.7933 | 0.8530 | +0.0597（7.5%） | 73.6% → 0.5% |
| 4096 | **`fused`** | 0.7890 | **1.0512** | **+0.2622（33%）** | 73.6% → 0.9% |

判定（严格按 §6）：

1. **fused 臂：支持"截断存在"。** 同一个实现只换 execution 轴，比值从 **0.76 → 1.11**（ctx=1024）
   与 **0.79 → 1.05**（ctx=4096）；差值 0.26–0.35 **远大于**噪声带（0.06%–0.54%），
   且 `host_share` 同时从 ~75% 掉到 <1.5%（§6 要求的机制自洽）。
   ⇒ `w8a16 × cudagraph × 1024 = 3.4751 ms`（v2）是**本项目第一个 >1× 的量化格**（fp16: 4.0670 ms）。
   ★ eager 那一列在 v2 里不可分辨：A/B 比（fus/sep）1.0093 → **1.0091**，绝对 speedup 的
   2% 漂移是 eager 宿主噪声 —— 与 `EXP-p2-01` v2 的 1.0442/1.0523（改前 1.0518/1.0532）同一件事：
   **这个 kernel 的改进在 host 受限的 eager 臂里看不见，只在图里兑现。**
2. **separate 臂（预注册的 v1 语义）：不支撑任何"量化兑现"的读法。** 0.767 → 0.767（1024，
   在噪声带内）/ 0.793 → 0.853（4096，仍 <1）。原因不是 host —— 它的成本是**GPU 侧的反量化
   物化**，而 cudagraph 只拿得掉 host/launch。⇒ **P2-01 的融合是 cudagraph 兑现量化收益的前提**；
   只做 P3 不做 P2-01，会得出"量化在 cudagraph 下也不行"的错结论。
3. **量级：1.05–1.11×，不是矩阵猜的 2×。**（**为什么**：见 §2.8.7 的 kernel 归因 —— 权重流只占这一步的 23–42%，所以天花板本来就只有 ~1.3×（ctx=4096）到 ~1.74×（ctx=1024）。） 2× 那一行是 `W4A16`（引用门 FAIL 0.0933 > 0.05，
   且 grouped 的 4-bit expert 路径未实现 → 那 4 格**跑不了**）。即便在 graph 里，这一步也不是
   权重带宽受限的：`w8a16` 的 `[W]` 地板 0.8328 ms vs 实测 3.6737 ms（达成率 **22.7%**），
   fp16 是 1.5095 vs 4.0670（37.1%）。"权重减半"换不到"时间减半"。
4. **evidence 格不得引用**：`w8a8 × cudagraph` = _(6.0640)_ / _(9.3360)_ ms（gate FAIL，
   `latency` 全 null）→ 对 fp16 的比值 0.67 / 0.78，比 w8a16 差（fake-quant 激活 + 2002 kernel/步）。

### 2.8.6 现状：哪些格现在能跑（P3 落地后）

| 格 | 状态 |
|---|---|
| `fp16 × {eager,cudagraph} × {1024,4096}` | ✅ 已测（本轮重跑，与 §2.6 一致：cg 4.0670/7.3211 vs 4.070/7.311） |
| `w8a16 × eager × {1024,4096}` | ✅ 已测（本轮同轮 + 交错 A/B；v1 的 12.0154/12.3566 归档在 `results/EXP-p4b-01-run/v1_archive/`） |
| `w8a16 × cudagraph × {1024,4096}` | ✅ **已测（P3）**：5.2998/5.3028 与 8.5825（separate）、3.6737 与 6.9645（fused，evidence） |
| `w8a8 × cudagraph × {1024,4096}` | ✅ 已测（evidence，gate FAIL） |
| `w8a8 × eager × {1024,4096}` | ⬜ 未跑（v1 也没有；要跑就得跟其余格同轮，且只作 evidence） |
| `w4a16 × *` | ⛔ 拒绝（引用门 FAIL + grouped 的 4-bit expert 未实现） |

### 2.8.7 为什么是 1.11× 而不是 2×：图内的 kernel 归因（回答 §2.8.5 第 3 条）

同一批格子的 `bottleneck.top_kernels`（profiler 的 per-kernel self time，覆盖 busy 的 99.4–99.8%）
按类归并（ms/步；括号是 share）：

| 桶 | fp16 ×1024 | w8a16 sep ×1024 | **w8a16 fused ×1024** | fp16 ×4096 | w8a16 fused ×4096 |
|---|---|---|---|---|---|
| attention（`fmha`/sdpa） | 1.216 (30.2%) | 1.217 (23.1%) | 1.217 (33.5%) | 4.461 (61.5%) | 4.473 (64.8%) |
| MoE GEMV（本仓库 triton） | 1.122 (27.9%) | 0.725 (13.8%) | **0.705 (19.4%)** | 1.131 (15.6%) | 0.707 (10.2%) |
| elementwise/copy | 0.851 (21.2%) | **2.591 (49.2%)** | 0.830 (22.9%) | 0.847 (11.7%) | 0.840 (12.2%) |
| gemm/gemv（含 attention 投影） | 0.641 (15.9%) | 0.542 (10.3%) | 0.689 (19.0%) | 0.629 (8.7%) | 0.689 (10.0%) |
| routing/index | 0.060 | 0.058 | 0.055 | 0.059 | 0.058 |
| 其它 | 0.111 | 0.113 | 0.111 | 0.111 | 0.111 |
| **kernel 数/步** | 978 | 1234 | **978** | 978 | 978 |

几条读得出来的东西：

1. **量化能碰的只有"权重流"那两块**：MoE GEMV（1.122 → 0.705，`_gate_up_act_kernel_q` 0.444 +
   `_down_sum_kernel_q` 0.260）与 attention 投影（cuBLAS `gemvx` 0.603 → `_dequant_gemm_kernel`
   0.525，64 次 × 8.2 µs）。合计 1.725 → 1.230 ms。
2. **拿不掉的占大头**：`fmha` 1.216 + elementwise/copy 0.851 + routing/other 0.17 ≈ **2.24 ms（55%）**。
   这是 E7 的**天花板**：即便把权重流压到 0，ctx=1024 的这一步最多也只能到
   `(4.067 − 1.725) / 4.067 = 0.58×`（即上限 1.74×）；ctx=4096 时 fmha 占 61.5%，
   权重流 1.72/7.32 = 23% ⇒ **上限只有 ~1.30×**。实测 1.11× / 1.05× 就落在这个上限之下。
3. **separate 臂的钱花在哪**：elementwise/copy 从 0.851 涨到 **2.591 ms**（`unrolled_elementwise`
   ×194/步 + `elementwise_kernel` ×129/步 —— 就是物化 fp16 权重的算子链），kernel 数 978 → 1234。
   这解释了 §2.8.5 第 2 条：cudagraph 把 host 拿掉了，但**拿不掉这些 GPU 算子**。
4. **ctx 越长，量化的相对收益越小**：注意力线性增长（1.22 → 4.46 ms），权重流几乎不动，
   所以 1.11×（1024）→ 1.05×（4096）。

## 2.9 4-bit 与 evidence 格：第 3 轮（2026-10-07）

### 2.9.1 先补的 4-bit 路径（P1 的 nibble 解包）

`w4a16` 的 4 格此前**跑不了**（量化 grouped 只做 8-bit）。本轮先补上：

- `_grouped_triton` 的两个量化 kernel 加 `BITS: tl.constexpr`，nibble 展开走共享的
  `_expand_int4`（`tl.join` + `tl.reshape`，`k = 2j + which`，低半字节在前）。
  ★ 符号扩展用纯算术 `((x & 0xF) ^ 8) - 8`，**不是** `tl.where` —— `_grouped_triton.py`
  有一条 AST 守卫禁 where（它是对的，不该为了新代码放宽）。
- `moe_grouped_quant`：packed 变 uint8（最后一维 H//2 / F//2）。**所有把最后一维当
  "未打包宽度"的地方都修了**（`hidden`、`dequant_*`、逐专家反量化、`quant_launch_plan`
  的形状 shim）—— 不改就会按半宽算 grid，静默算错。
- **K1 门在 5090 上 PASS**（`test_triton_quantized_grouped_matches_oracle` 现在跑 8-bit
  与 4-bit 两条路径，容差仍是 `PARITY_GATE` 1e-3）；离线新增 4-bit 的**语义锚点**：
  "先融合再量化" vs "逐专家量化" 逐位相同（含 nibble 打包）。
- 上机第一轮撞到并修掉一个真错：4-bit 分支最初用 `k // 2` 当字节索引（每个 uint8 被加载两次），
  `tl.reshape` 的元素数直接对不上 → Triton 编译期报错。字节轴长度必须是 `BLOCK_K // 2`。

### 2.9.2 第 3 轮：**有效**（两张金丝雀都 <0.5%，11 格 exit=0、`schema problems: 0`）

| 金丝雀（同格首尾） | 首 | 尾 | 相对差 |
|---|---|---|---|
| `fp16 × eager × 1024` | 9.8354 ms | 9.7893 ms | **−0.47%** |
| `w8a16 × cudagraph × 1024` | 5.3069 ms | 5.2996 ms | **−0.14%** |

占用：`w4a16 × cudagraph × 1024（fused）` 的 pre 快照上有外人进程 **pid 39721（1636 MiB）**
（`p3-v3-contention.log`，**记录不杀**）；该格之后立刻消失。这一格是 evidence（gate FAIL），
且它的 `host_share = 1.3%` 与同轮其它 graph 格（0.6–1.3%）同量级 —— 但按纪律，
**它不构成任何可引用的数字**。

### 2.9.3 evidence 表（4 格 w4a16 + 2 格 w8a8；**gate FAIL → 延迟不得引用**）

斜体 = 实测原始值，只在 `_measurement.gate_failed_evidence`，`latency` 段为 null（硬规则 2）。

| 格 | eager | cudagraph（separate 臂） | cudagraph（fused 臂） | `[W]` | kernel/步 |
|---|---|---|---|---|---|
| `w4a16 × 1024` | _(17.3946)_ | _(9.1203)_ | **_(3.3812)_** ⚠️改前 | 0.764 GB | 981 / 1810 / **978** |
| `w4a16 × 4096` | _(18.1360)_ | _(12.3093)_ | — | 0.764 GB | 981 / 1810 |
| `w8a8 × 1024` | _(17.7624)_ | _(6.0640)_ | — | 1.301 GB | 1749 / 2002 |
| `w8a8 × 4096` | _(17.9230)_ | _(9.3360)_ | — | 1.301 GB | 1749 / 2002 |

（`w8a8` 的 cudagraph 两格来自 §2.8 第 2 轮，同轮金丝雀与本节一致。）

### 2.9.4 这三个数字说明了什么（都只到 evidence 层）

1. **实现 > 位宽**：同一个 `w4a16 × cudagraph × 1024`，separate 臂 _(9.1203)_ ms / 1810 kernel
   vs fused 臂 **_(3.3812)_ ms / 978 kernel** —— **2.7×**。与 §2.8 的结论同源：
   在 graph 里决定成败的是"反量化是不是被融合进 GEMM"，不是位宽。
2. **边际字节几乎贴着带宽地板**：`w8a16 × cudagraph × 1024`（fused，§2.8）= 3.6737 ms →
   `w4a16 × cudagraph × 1024`（fused）= **3.3812 ms**，Δ = 0.2925 ms，而 Δ`[W]` = 0.537 GB
   ÷ 1561.9 GB/s = **0.3438 ms** ⇒ 边际上拿回了 **85%**。
   所以"权重减半换不到时间减半"不是因为字节不值钱，而是因为**权重只占这一步的 ~23%**
   （§2.8.7 的 kernel 归因）。（跨轮比较：两轮金丝雀差 0.1–0.6%，见 §2.9.2。）
3. **矩阵里的 W4A16 行仍然不可发表**：引用门 FAIL（KL 0.0933 > 0.05），
   而 *已部署的* separate 实现下 4-bit 在**两个 execution 轴上都比 8-bit 差**
   （eager 17.39 vs 12.68；cudagraph 9.12 vs 5.30 —— nibble 解包把 kernel 数从 1234 抬到 1810）。
   按硬规则 2，这些只能作 evidence。要在 W4A16 上报数字，必须先修精度（`REMEDIATION_LADDER`），
   那是独立的精度实验，不在本记录里。

> ⚠️ §2.9.3 里 **`w4a16 × cudagraph × 1024（fused）` 的 _(3.3812)_ 是改前口径**（P2d 轮只重跑了
> 可发表的 fused 格 + `EXP-p2-01` 的 6 格）。它是 gate FAIL 的 evidence、不构成结论；
> 若要把这个数写进任何地方，先按 P2d 的流程重跑它（归档 + 标 v2）。

### 2.9.5 矩阵完备性

p4b 声明的 **16/16 格现在都有 `result.json`**：8 格 gate PASS（fp16 ×4、w8a16 ×4），
8 格 gate FAIL 的 evidence（w8a8 ×4、w4a16 ×4）。**门槛一个都没有动。**

## 4. 现状表：哪些格子现在能跑（**P3 落地后更新 → 细节见 §2.8.6**）

| 格 | 状态 | 说明 |
|---|---|---|
| `fp16 × eager × grouped × {1024, 4096}` | ✅ 已测（§2.6；本轮 §2.8 重跑一致） | P0 已交付 |
| `fp16 × cudagraph × grouped × {1024, 4096}` | ✅ 已测（§2.6；本轮 §2.8 重跑一致） | p0-04/p2-02 的成熟路径 |
| `w8a16 × eager × grouped × {1024, 4096}` | ✅ **已测**（§2.7；本轮同轮 + 交错 A/B） | P1：experts 量化走融合 kernel |
| `w8a16 × cudagraph × grouped × {1024, 4096}` | ✅ **已测（P3，§2.8）** | separate 5.2998/8.5825 ms；fused 3.6737/6.9645 ms（evidence） |
| `w8a8 × cudagraph × grouped × {1024, 4096}` | ✅ 已测（evidence，§2.8.5 第 4 条） | 引用门 FAIL → `latency` 全 null，延迟不得引用 |
| `w8a8 × eager × grouped` | ⬜ 未跑（v1 也没有） | 要跑就得与其余格同轮；只作 evidence |
| `w4a16 × eager/cudagraph × {1024,4096}` | ✅ 已测（evidence，§2.9） | 4-bit nibble 解包已落地；引用门 FAIL → 延迟不得引用 |
| `w8a8 × eager × {1024,4096}` | ✅ 已测（evidence，§2.9） | 同轮补齐 |
| 量化 × `gather` | ⛔ 拒绝 | gather 直接索引 fp16 融合张量，无量化版 |
| 任意量化 × 其它 ctx/g | ⛔ 拒绝 | 没有门实测 |

**16/16 格已测（2026-10-07 收口）**：8 格 gate PASS（fp16 ×4、w8a16 ×4）+ 8 格 gate FAIL 的
evidence（w8a8 ×4、w4a16 ×4，含 4-bit 的 grouped nibble 解包）。细节见 §2.9。

## 5. 上机前冻结的门

### K1 — 正确性门（规则 2）

与 `EXP-p2-01` 同一张引用表（`benchmarks/runner.GATE_CITATIONS`，测试钉住与工件一致）。
门是**合取**（`METRICS.md` §1：KL **或** top-1 不达标即不得报性能），
聚合口径 = 5 窗口**均值**；`gate` 由 `gate_from_metrics()` 从实测值推导：

| 配置 | 5 窗口 mean KL | top-1 mean | 门 | 能否报性能 |
|---|---|---|---|---|
| fp16 | 参照 | 参照 | consistency | ✅（自洽性门） |
| `w8a16 / g128 / ctx=1024` | 0.001273 | 0.9818 | 0.01 / 0.95 | ✅ |
| `w8a8 / g128 / ctx=1024` | 0.032753 | 0.9758 | 0.01 | ❌ verdict=FAIL，只作 evidence |
| `w4a16 / g128 / ctx=1024` | 0.093323 | 0.8606 | 0.05 | ❌ 同上 |
| `w8a16 / g128 / ctx=4096` | 0.000714 | 0.9818 | 0.01 / 0.95 | ✅ |
| `w8a8 / g128 / ctx=4096` | 0.004606 | **0.9455 (<0.95)** | 0.01 / 0.95 | ❌ **FAIL（top-1 拦下）** |
| `w4a16 / g128 / ctx=4096` | 0.053876 | 0.8545 | 0.05 | ❌ 同上 |
| 其它量化 × 别的 ctx/g | 未测 | 未测 | — | ⛔ runner 拒绝 |

### K2 — E7 测量协议

1. **格是独立进程**（每格一次 runner 调用 → 一次模型加载）→ 跨格比较有机器状态漂移风险。
   协议：
   - 每个 `(bit_width × ctx)` 的 eager/cudagraph **相邻执行**；
   - **金丝雀**：`fp16 × eager × ctx=1024` 在本轮**开头和结尾各跑一次**，
     两者 per-token p50 的相对差 = 本会话的噪声带；超过 5% → 本轮作废（记录占用，不杀进程）；
   - E7 的"截断 vs 接近"判定用这个实测噪声带做参照（§Hypothesis 的判据）。
2. 计时前 `nvidia-smi --query-compute-apps` 查占用；发现外人进程**记录**，不动它
   （实测过：外人进程把 ctx=4096 从 3.2356 ms 顶到 10.38 ms、host-share 67.5%）。
3. 报矩阵声明的 metrics：`host_share / gpu_p50 / host_p50 / per_token_p50 / bw_attainment`
   （`host_share_pct` 是本条的一等指标，它就是"截断"的机制量）。
4. 带宽地板只用**实测 1561.9 GB/s**（`results/machine_probe.json`）。
5. 报绝对值与比值（Hoefler 规则 #1）。

## 6. 判据（跑完怎么判）

```text
支持（截断存在）: speedup_cudagraph 与 speedup_eager 的差 > 金丝雀噪声带，
                  且 host_share 同时下降（机制自洽：host 侧开销下降 → 量化收益回来）
否证（矩阵原文） : 两者在噪声带内接近 → 截断不存在，量化可独立兑现
分不出         : 差 < 噪声带的 2 倍且 host_share 变化也 < 噪声 → 记 inconclusive
W4A16 行       : 无论跑出什么，verdict=FAIL（引用门 0.0933 > 0.05），延迟不进结论
```

## 7. 执行顺序（给上卡的会话）

1. ✅ **P2** 补测 ctx=4096 的门 → 已并进 `GATE_CITATIONS`（合取推导）+ 测试（§2.5）；
2. ✅ 跑 §4 表里 fp16 的 4 格（开头/结尾金丝雀）→ **execution 轴的 fp16 基线**（§2.6）；
3. ✅ **P1** 量化感知 expert 路径（grouped，8-bit）→ K1 门在 5090 上 PASS，量化臂跑通（§2.7）；
4. ✅ **P3** 把 `quant_cfg` 接进 `run_cudagraph`（两处必修的坑 + 第三处 blocked 路径）→
   量化 × cudagraph 12 格跑通、**E7 判定见 §2.8.5**；
5. ✅ **`EXP-p2-01` 的 dense fused 臂**（attention）：已落地并跑完 6 格（记录 §6）。
   ★ 它不只是"拆开混淆"：§2.8 显示**没有它，cudagraph 也兑现不了量化收益**（separate 臂
   `speedup_cudagraph` 0.767，fused 臂 1.107）；
6. ✅ 4-bit 的 nibble 解包（P1 现在做 4 与 8-bit，K1 门两条路径都 PASS）；`w8a8 × eager`
   的 2 格 evidence 也补齐了 → **16/16 格收口**（§2.9）。
   ★ 结论：4-bit 的**边际**字节几乎贴着带宽地板（85%），但权重只占这一步 ~23%，
   所以"减半"换不到可观的绝对收益；而 *已部署的* separate 实现下 4-bit 反而更慢。
   要报 W4A16 的数字，必须先做**精度**侧（引用门 0.0933 > 0.05），那是独立实验；
7. 每格 `results/<cell>/result.json` + 本记录补"结果"节 + commit。

## Applicability（按已测的 12 格 + 4 个 evidence 格填全）

```text
适用于：RTX 5090（sm_120、170 SM、驱动 595.71.05）、**实测**读带宽 1561.9 GB/s、
        dtype=fp16、batch=1、ctx ∈ {1024,4096}、MoE=grouped（两臂同一路径）、
        attention=sdpa（模型默认）、q_len=1 decode、warmup 5 / timed 20、
        torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6、
        量化 = per-group 对称 RTN g=128、targets=attention+experts（与 p1-04 门逐字一致）；
        §2.6/§2.7 的数字在各自金丝雀噪声带（0.10% / 0.70%）内有效；
        §2.8 的数字绑定本轮两张金丝雀：fp16×eager×1024 −0.54%、w8a16×cudagraph×1024 +0.06%；
        **E7 的判定只在 "w8a16 + P2-01 的 fused attention" 这个实现上成立**
        （1.05–1.11×），且 cudagraph 格只有 3 个 repeat
不适用于：W4A16/W8A8 的**性能结论**（引用门 FAIL；它们的延迟只作 evidence —— §2.9 的表里
          那些斜体值**不得**被引用为性能结论，包括 4-bit fused 的 _(3.3812)_ ms）；
          §2.9 的 4-bit evidence（它的 pre 快照上有过 1636 MiB 的外人进程，记录不杀）；
          4-bit fused expert（未实现）；量化 × gather；量化 × 其它 ctx/g；batch>1；
          **把 E7 的 1.05–1.11× 外推到 W4A16 或其它 ctx**（2× 那一行没有测过：W4A16 门 FAIL
          且 grouped 4-bit 未实现）；把 1.05–1.11× 当成"接近带宽墙"的证据
          （w8a16 的 [W] 地板 0.8328 ms vs 实测 3.6737 ms，达成率只有 22.7%）；
          V100（无 triton/无 cp.async，只能做 fp16 对照）
原因：    截断的量级绑定在这台卡的 host/launch 开销上（eager 的 host_share ~75% → cudagraph <1.5%
          是本条机制量）；而 cudagraph 之后剩下的差距是 GPU 侧（反量化物化 / kernel 数），
          两者都换机器、换 triton 版本就得重测。
```

## ★ P2j 刷新（2026-10-08）：在融合 RMSNorm 实现下重测本条的格子

> **为什么**：`EXP-p4c-02` §5.6 把 decode 的 `OlmoeRMSNorm`（每个 norm 7 个 kernel）换成自研
> 融合 kernel（图内全步 **1.2358×**），P2i 又把它接成 runner 的**默认**实现（`--rmsnorm fused`）。
> 按硬规则 2/7：门要在新实现下重测（已做，见 `EXP-p1-04` 的 v2 小节），**绝对延迟也要重测**。
>
> **协议**：改前 payload 逐格归档到 `results/EXP-p2j-refresh/archive/`；每格前后记占用快照；
> 首尾同格金丝雀（`w8a16×cudagraph×ctx=1024`：**3.0084 → 3.0124 ms，漂移 +0.13%** ⇒ 轮次有效）；
> payload 自检 `config.rmsnorm == "fused"`、`rms_norms_replaced == 65`（**13 格全部通过**）。
>
> ★★ **本轮把结果分成了两类，因为环境不允许全盘替换**：共享卡上全程有**外来作业**
> （另一项目的 `bench_load_30b.py` / `pareto_offload.py`，CPU 满载）。**图内臂**（host ≤5%）
> 不受影响，刷新值被采用；**host 主导的格子**（host 60–96%）的 wall-clock 被压爆 ——
> 判别实验：用**旧的** `--rmsnorm hf` 在同一环境重跑 `w8a8·eager·separate`，
> `kernels_per_call` 精确回到归档值 **24405**（`gpu_busy` 也一致 61.4 vs 61.7），
> 但 `host_p50` 从归档的 **1.52 ms** 变成 **35.2 ms**（融合版那次 206.1 ms）。
> ⇒ **host 才是变量，不是实现**。那 18 格的刷新值存到 `results/EXP-p2j-refresh/contended/`，
> 规范 payload **恢复为改前版本**（记录里采用它）。
>
> **每条实现改变的签名**：所有格子的 `kernels_per_call` 都**恰好 −455**（= 7 kernel/norm × 65 norm/步），
> `gpu_busy_ms` 一律略降 —— 这是"融合 norm 确实生效"的硬证据，与 wall-clock 无关。


| 格 | 改前 p50 (ms) | 刷新实测 (ms) | Δ | **记录采用** | gate 改前 → 采用 | 说明 |
|---|---|---|---|---|---|---|
| `fp16 · cudagraph · ctx=1024` | 4.0670 | 3.5881 | -11.8% | **3.5881** | PASS → **PASS** | 采用刷新值（图内臂，host 1.5%） |
| `fp16 · cudagraph · ctx=4096` | 7.3211 | 6.8269 | -6.8% | **6.8269** | PASS → **PASS** | 采用刷新值（图内臂，host 0.8%） |
| `fp16 · eager · ctx=1024` | 9.8841 | 8.2915 | -16.1% | **9.8841** | PASS → **PASS** | **作废**：刷新时 host 主导（67%，共租 CPU 压住 host） |
| `fp16 · eager · ctx=4096` | 9.8716 | 8.1796 | -17.1% | **9.8716** | PASS → **PASS** | **作废**：刷新时 host 主导（60%，共租 CPU 压住 host） |
| `w4a16 · cudagraph · ctx=1024` | 9.1203 | 8.6194 | -5.5% | **8.6194** | FAIL → **FAIL** | 采用刷新值（图内臂，host 0.8%） |
| `w4a16 · cudagraph · ctx=4096` | 12.3093 | 11.7834 | -4.3% | **11.7834** | FAIL → **FAIL** | 采用刷新值（图内臂，host 0.7%） |
| `w4a16 · eager · ctx=1024` | 17.3946 | 16.2229 | -6.7% | **17.3946** | FAIL → **FAIL** | **作废**：刷新时 host 主导（83%，共租 CPU 压住 host） |
| `w4a16 · eager · ctx=4096` | 18.1360 | 16.1702 | -10.8% | **18.1360** | FAIL → **FAIL** | **作废**：刷新时 host 主导（78%，共租 CPU 压住 host） |
| `w8a16 · cudagraph · ctx=1024` | 3.4749 | 3.0124 | -13.3% | **3.0124** | PASS → **PASS** | 采用刷新值（图内臂，host 1.9%） |
| `w8a16 · cudagraph · ctx=4096` | 6.7545 | 6.2702 | -7.2% | **6.2702** | PASS → **PASS** | 采用刷新值（图内臂，host 1.0%） |
| `w8a16 · eager · ctx=1024` | 12.3909 | 11.3447 | -8.4% | **12.3909** | PASS → **PASS** | **作废**：刷新时 host 主导（79%，共租 CPU 压住 host） |
| `w8a16 · eager · ctx=4096` | 12.5658 | 11.5566 | -8.0% | **12.5658** | PASS → **PASS** | **作废**：刷新时 host 主导（75%，共租 CPU 压住 host） |
| `w8a8 · cudagraph · ctx=1024` | 6.0640 | 5.5868 | -7.9% | **5.5868** | FAIL → **FAIL** | 采用刷新值（图内臂，host 0.7%） |
| `w8a8 · cudagraph · ctx=4096` | 9.3360 | 8.8509 | -5.2% | ~~8.8509~~ | FAIL → PASS ⚠️重判 → **FAIL（R3）** | **de-publish（2026-10-24）**：10 窗 KL 0.0262>0.01 ⇒ 已发布值只作 evidence |
| `w8a8 · eager · ctx=1024` | 17.7624 | 16.6528 | -6.2% | **17.7624** | FAIL → **FAIL** | **作废**：刷新时 host 主导（83%，共租 CPU 压住 host） |
| `w8a8 · eager · ctx=4096` | 17.9230 | 17.1949 | -4.1% | **17.9230** | FAIL → **FAIL** | **作废**：刷新时 host 主导（79%，共租 CPU 压住 host） |

**结论**：图内臂的绝对延迟已按新实现更新（上表"采用"列）；host 主导格子的 pre-P2i 值保留，
其**结论不受影响** —— 融合 norm 省下的是每步 ~0.34–0.48 ms 的常数，在 8–272 ms 的 host 受限
步长里低于噪声（这一点 `EXP-p2d` 已经量过：同一个 kernel 改进在 eager 臂里完全看不见）。
把 host 主导的 18 格在**安静的机器**上重跑，列为 **P2k**。


**⚠️ 判定变化（这一节最要紧的一条）**：``w8a8 · cudagraph · ctx=4096`` 的引用门从 **FAIL** 变成 **PASS** ——
`EXP-p1-04` 在融合 norm 下重测后，`w8a8@4096` 的 top-1 从 0.9455 变成 0.9636（阈值 0.95）。
按硬规则 2 该格现在**发布延迟**（上表"采用"列）；但按 `EXP-p1-04` v2 §1 的三条证据，
这个门是**骑线**的（差 3/165 个位置，且新 norm 改变了 fp16 参考的贪心轨迹），
所以 **不得**把这一格当成"w8a8 在 4096 上可用"的证据 —— 正确表述是"在这个测法下不可分辨"。
本条其余 `w8a8@1024` / `w4a16` 的格子仍然是 gate FAIL（延迟只进 `gate_failed_evidence`）。

**★★ R3 终裁（2026-10-24，本节上一段的历史到此为止）**：上面"该格现在发布延迟"的结论**已被撤销**。
预注册的 10 窗复测（`results/EXP-p1-04-r3-10win-ctx4096.json`）显示 `w8a8@4096` 的 KL 均值
**0.0262 > 0.01**（重尾：w7 单窗 0.2247、sd 0.066、spread 71.7×）—— 5 窗的 PASS **不稳健**，
判定随窗口数翻转，"骑线"从疑点变成了实测事实。用户批准 **de-publish**：门行改引 10 窗工件
（推导 FAIL），本格照跑照测，但 latency 只进 `_measurement.gate_failed_evidence`（与 w4a16
同一语义）。这个格子的完整历史因此是：v1 FAIL（top-1 骑线）→ v2 PASS（同 3/165 翻转）
→ **v3 FAIL（10 窗 KL 均值超门）** —— 三版工件都在，没有一格被悄悄删掉。
