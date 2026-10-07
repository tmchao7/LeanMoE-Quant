# EXP-p4b-01-truncation

> **状态：`running` —— P1/P2/P3 与 `EXP-p2-01` 的 dense fused 臂都已落地，E7 已判定（§2.8）。**
> ★ **E7 判定（2026-10-07，§2.8.5）**：截断效应**成立，但只在 fused 臂上兑现** ——
> 同一个 w8a16 实现只换 execution 轴：`speedup_eager` 0.76 → `speedup_cudagraph` **1.11**
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
- **Status**: `running`（**12/16 格已测且有效**；w8a8×eager 的 2 格未跑；w4a16 的 4 格被
  引用门 + 4-bit grouped 挡住。E7 已判定 —— §2.8.5）
- **原始数据**: `results/EXP-p4b-01-truncation__bit_width=*_execution=*_ctx=*/result.json`；
  每轮的金丝雀/占用/日志在 `results/EXP-p4b-01-run/`（第 1 轮作废件在 `round1-void/`，
  P3 第 1 轮污染件在 `p3-*.log` 与 `v1_archive/`）；隔离 kernel 微基准
  `results/bench_quant_grouped.json`
- **对照**: `EXP-p1-04-prompt-set-gate`（门，含 ctx=4096 复测）、`EXP-p0-04`（静态 MoE + cudagraph）、
  `EXP-p2-02`（grouped + cudagraph，per-token 4.083 ms）、`EXP-p2-01`（E2：反量化融合 vs 独立 ——
  它的 fused 臂就是 §2.8 里让量化第一次 >1× 的那个实现）

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
| `w8a16 × 1024` | 12.6838（A/B sep）/ 12.8018（A/B fus） | **5.2998 / 5.3028**（金丝雀 A/B） | **3.6737** | 77.5% → 0.6–1.2% | 1.301 GB |
| `w8a16 × 4096` | 12.4429 / 12.5111 | **8.5825** | **6.9645** | 73.6% → 0.5–0.9% | 1.301 GB |
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
| 1024 | `separate`（v1 的语义，预注册对照） | 0.7665 | 0.7674 | **+0.0009（0.1%）** | 77.5% → 0.6% |
| 1024 | **`fused`（P2-01 的实现）** | 0.7594 | **1.1071** | **+0.3477（46%）** | 77.5% → 1.2% |
| 4096 | `separate` | 0.7933 | 0.8530 | +0.0597（7.5%） | 73.6% → 0.5% |
| 4096 | **`fused`** | 0.7890 | **1.0512** | **+0.2622（33%）** | 73.6% → 0.9% |

判定（严格按 §6）：

1. **fused 臂：支持"截断存在"。** 同一个实现只换 execution 轴，比值从 **0.76 → 1.11**（ctx=1024）
   与 **0.79 → 1.05**（ctx=4096）；差值 0.26–0.35 **远大于**噪声带（0.06%–0.54%），
   且 `host_share` 同时从 ~75% 掉到 <1.5%（§6 要求的机制自洽）。
   ⇒ `w8a16 × cudagraph × 1024 = 3.6737 ms` 是**本项目第一个 >1× 的量化格**（fp16: 4.0670 ms）。
2. **separate 臂（预注册的 v1 语义）：不支撑任何"量化兑现"的读法。** 0.767 → 0.767（1024，
   在噪声带内）/ 0.793 → 0.853（4096，仍 <1）。原因不是 host —— 它的成本是**GPU 侧的反量化
   物化**，而 cudagraph 只拿得掉 host/launch。⇒ **P2-01 的融合是 cudagraph 兑现量化收益的前提**；
   只做 P3 不做 P2-01，会得出"量化在 cudagraph 下也不行"的错结论。
3. **量级：1.05–1.11×，不是矩阵猜的 2×。** 2× 那一行是 `W4A16`（引用门 FAIL 0.0933 > 0.05，
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

## 4. 现状表：哪些格子现在能跑（**P3 落地后更新 → 细节见 §2.8.6**）

| 格 | 状态 | 说明 |
|---|---|---|
| `fp16 × eager × grouped × {1024, 4096}` | ✅ 已测（§2.6；本轮 §2.8 重跑一致） | P0 已交付 |
| `fp16 × cudagraph × grouped × {1024, 4096}` | ✅ 已测（§2.6；本轮 §2.8 重跑一致） | p0-04/p2-02 的成熟路径 |
| `w8a16 × eager × grouped × {1024, 4096}` | ✅ **已测**（§2.7；本轮同轮 + 交错 A/B） | P1：experts 量化走融合 kernel |
| `w8a16 × cudagraph × grouped × {1024, 4096}` | ✅ **已测（P3，§2.8）** | separate 5.2998/8.5825 ms；fused 3.6737/6.9645 ms（evidence） |
| `w8a8 × cudagraph × grouped × {1024, 4096}` | ✅ 已测（evidence，§2.8.5 第 4 条） | 引用门 FAIL → `latency` 全 null，延迟不得引用 |
| `w8a8 × eager × grouped` | ⬜ 未跑（v1 也没有） | 要跑就得与其余格同轮；只作 evidence |
| `w4a16 × *` | ⛔ 拒绝 | 引用门 FAIL（0.0933 > 0.05）+ grouped 的 4-bit expert 未实现 |
| 量化 × `gather` | ⛔ 拒绝 | gather 直接索引 fp16 融合张量，无量化版 |
| 任意量化 × 其它 ctx/g | ⛔ 拒绝 | 没有门实测 |

**12 格已测且有效 / 2 格可跑未测（w8a8 × eager）/ 4 格被引用门 + 4-bit 挡住。**

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
6. ⬜ 4-bit 的 nibble 解包（P1 目前只做 8-bit）；`w8a8 × eager` 的 2 格 evidence。
   ★ 4-bit 是**下一步最值钱的一格**：`w4a16` 的 `[W]` 是 fp16 的 25%，而 §2.8.5 第 3 条说明
   上限不在字节——要先把门修过（p1-04）或明确标 FAIL；
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
不适用于：W4A16/W8A8 的**性能结论**（引用门 FAIL；它们的 cudagraph 延迟只作 evidence）；
          4-bit fused expert（未实现）；量化 × gather；量化 × 其它 ctx/g；batch>1；
          **把 E7 的 1.05–1.11× 外推到 W4A16 或其它 ctx**（2× 那一行没有测过：W4A16 门 FAIL
          且 grouped 4-bit 未实现）；把 1.05–1.11× 当成"接近带宽墙"的证据
          （w8a16 的 [W] 地板 0.8328 ms vs 实测 3.6737 ms，达成率只有 22.7%）；
          V100（无 triton/无 cp.async，只能做 fp16 对照）
原因：    截断的量级绑定在这台卡的 host/launch 开销上（eager 的 host_share ~75% → cudagraph <1.5%
          是本条机制量）；而 cudagraph 之后剩下的差距是 GPU 侧（反量化物化 / kernel 数），
          两者都换机器、换 triton 版本就得重测。
```
