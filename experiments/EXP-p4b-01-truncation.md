# EXP-p4b-01-truncation

> **状态：`running` —— 假设与门已冻结；execution 轴的 fp16 基线已测（§2.6），
> 量化臂仍被 P1 挡住（§3）。因此本记录目前**不能给出 E7 的判定**。**
> ★ 这条记录最重要的产出不是门，而是 §3 的**依赖链**：把 E7（本项目的头号结论）
> 拆成了 P0–P3 四个前置条件，其中 **P1 是真正的新功能**（量化感知的 expert 路径）。
> 在 P1 落地之前，E7 的量化臂**无路可走** —— runner 会直接拒绝，而不是写错记录。
> P2（ctx=4096 的门）已于 2026-10-07 补测完成（§2.5）。

- **ID**: `EXP-p4b-01-truncation`
- **Date**: 2026-10-06（假设与门冻结）→ 2026-10-07（P2 补测 + fp16 基线）
- **Operator**: `benchmarks.runner`（`--bit-width fp16 --execution {eager,cudagraph}
  --moe grouped --ctx {1024,4096}`）+ `tools/quant_probe.py`（P2）
- **Status**: `running`（4/16 格已测且有效；12 格被 P1/P3 挡住）
- **原始数据**: `results/EXP-p4b-01-truncation__bit_width=fp16_execution=*_ctx=*/result.json` ×4
  （第 1 轮作废件归档在 `results/EXP-p4b-01-run/round1-void/`）；
  占用/金丝雀证据在 `results/EXP-p4b-01-run/`
- **对照**: `EXP-p1-04-prompt-set-gate`（正确性门实测，含 ctx=4096 复测）、`EXP-p0-04`（静态 MoE + cudagraph）、
  `EXP-p2-02`（grouped + cudagraph，per-token 4.083 ms）、`EXP-p2-01`（fused dense kernel，同一批前置）

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
P1  量化 × 可 capture 的 MoE 路径                  ❌ 不存在 —— E7 量化臂的真正阻塞点
P2  ctx=4096 的量化门实测                          ✅ 已完成（2026-10-07，见 §2.5）
P3  量化 + cudagraph 的 quant_cfg 接线             ❌ 在 P1 之后才有意义
```

### P1 为什么不存在（这是本记录最要紧的分析）

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

## 4. 今天就能跑的格子（开卡即用）

| 格 | 状态 | 说明 |
|---|---|---|
| `fp16 × eager × grouped × {1024, 4096}` | ✅ 可跑 | P0 已交付；eager+grouped 是新路径（cudagraph 侧 p2-02 已过门） |
| `fp16 × cudagraph × grouped × {1024, 4096}` | ✅ 可跑 | p0-04/p2-02 的成熟路径；4096+grouped 是新格 |
| `w8a16/w8a8/w4a16 × eager × grouped` | ⛔ 拒绝（P1） | 用 `--moe hf` 可跑成 evidence 行，不进 E7 比值 |
| `w8a16/w8a8/w4a16 × cudagraph × grouped` | ⛔ 拒绝（P1+P3） | P1 之后才有意义 |
| 任意量化 × ctx=4096 | 门已补测（P2 ✅），但 grouped 上仍 ⛔ 拒绝（P1） | 用 `--moe hf` 可跑成 evidence 行 |

**4 格可跑 / 12 格被 P1–P3 挡住。** 这就是这条记录的诚实现状。

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
3. ⬜ **P1** 的实现：量化感知 expert 路径 —— 按 `EXP-p2-01` 记录 §4 的
   fused dense kernel 先行，再推广到 grouped；**不许**用"先 fuse 再量化"当成 fused 臂；
4. ⬜ P1 落地 → P3 接线 → 跑 16 格全矩阵 → 才能给出 E7 判定；
5. 每格 `results/<cell>/result.json` + 本记录补"结果"节 + commit。

## Applicability（按已测的 fp16 四格填全）

```text
适用于：RTX 5090（sm_120）、实测带宽 1561.9 GB/s（results/machine_probe.json）、
        dtype=fp16、batch=1、ctx ∈ {1024,4096}、MoE=grouped（两臂同一路径）、
        attention=sdpa（模型默认）、q_len=1 decode、warmup 5 / timed 20、
        torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6；
        §2.6 的 fp16 数字在该轮金丝雀噪声带 0.10% 内有效
不适用于：E7 的任何判定（量化臂不存在 —— P1 未落地）；
          W4A16/W8A8 的性能结论（引用门 FAIL）；
          eager 与 cudagraph 用**不同 MoE 路径**的任何组合（会与 execution 轴混在一起）；
          batch>1；其它 ctx；V100（无 triton/无 cp.async，只能做 fp16 对照）
原因：    截断的量级绑定在这台卡的 host/launch 开销与实测带宽上；
          P1 的缺失使量化臂目前只能以 hf 路径存在，不可与 grouped 臂同表比较。
```
