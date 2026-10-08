# EXP-p2-01-dequant-fusion

> **状态：`done`（2026-10-07 上卡执行；verdict = `refuted`，见 §6）。**
> ★ 2026-10-07：`EXP-p4b-01` 的 P1 落地后第一次量化臂实测是**负结果**（0.82×/0.80×），
> 而机制分析（`EXP-p4b-01-truncation` 记录 §2.7 + `tools/bench_grouped_gemm_quant.py`）
> 把原因定位到**反量化语义混用**：experts 走融合 kernel（隔离测**快 17%**），
> attention 仍走 `QuantLinear` 的 **separate 臂**（每次 forward 反量化）。
> 于是本条从"建议先做"变成 **拆开那个混淆的硬前提** —— 本会话把 dense fused 臂落地了。
> ★ **结论（一句话）**：K1 数值门 PASS、kernel 本身在隔离测里快 21%，但在**预注册的
> eager 格**里 fused 比 separate **慢 5%**（w8a16/w8a8），因为那条路径是 host/launch 受限的
> （host_share 71.7%、每步 19,029 个 kernel、gpu_busy 只有 57 ms / 199 ms）：
> **融合省的字节，被每次调用的 Python/launch 开销吃掉了。** E2 的 1.3-2× 在本配置**否证**。

- **ID**: `EXP-p2-01-dequant-fusion`
- **Date**: 2026-10-06（假设与门冻结）→ **2026-10-07（执行）**
- **Operator**: 上卡会话（RTX 5090 32 GB）
- **Status**: `done`（6 格全部有 `result.json`；2 格 gate PASS、4 格 gate FAIL 作 evidence）
- **原始数据**: `results/EXP-p2-01-dequant-fusion__bit_width=<x>_dequant=<y>/result.json` × 6，
  另存 `results/EXP-p2-01-run/`（每格日志、`p201_canaryA/B_result.json`、`contention.log` 占用快照）
- **对照**: `EXP-p1-04-prompt-set-gate`（正确性门的实测来源）、`EXP-p2-02`/`p2-04`/`p2-05`
  （同一台机器的 fp16 延迟口径与交错计时协议）

## Hypothesis（先写，跑之前冻结）

来自 `configs/experiments.yaml`（矩阵是假设的唯一真值来源）：

```text
statement:        融合带来 1.3-2x（E2）
refutation_signal: 差值 <10% → 反量化不是瓶颈，融合判断被否证
```

物理含义（`src/lemoq/kernels/reference.py` 的口径）：

    separate : y = x @ dequant(W).T   # 每次 forward 物化 [out,in] 的 fp16 权重，写+读一遍
    fused    : y = matmul(x, W, scale) # 读 packed 权重时当场解压，不落中间张量
    两条路径**数学结果相同**，差别只在是否多出一次权重的写+读流量。

本实验的执行臂是 eager（矩阵未声明 execution 轴；`_execution_for_exp` 对本实验返回 eager）。

## 1. ★ 轴的改动（为什么矩阵里有 `w8a16`）

矩阵原先的 `bit_width: [w8a8, w4a16]` 是在 `EXP-p1-01`（其结论现已作废）时期写的。
`EXP-p1-04` 之后的门实测：

| bit_width（g128、targets=attention+experts、ctx=1024） | 5 窗口 mean KL | 门（W8=0.01/W4=0.05） |
|---|---|---|
| `w8a16` | **0.001273** | ✅ PASS |
| `w8a8` | 0.032753 | ❌ FAIL |
| `w4a16` | 0.093323 | ❌ FAIL |

**不加 `w8a16`，本实验没有任何一格能按硬规则 2 发表性能数字。** 所以给 `bit_width`
轴加 `w8a16`（+2 格，共 6 格）——这不是扩变量（`dequant` 才是本实验的变量），
是让矩阵与 `EXP-p1-04` 的门结论对齐。这条改动的理由写在矩阵注释与本记录里。

## 2. 上机之前冻结的三条门

### K1 — kernel 级数值门（fused 必须等于参考）

与 `moe_grouped` / `attn_decode` **完全一致**的 PARITY_GATE（不得放宽）：

```python
tol_atol = 1e-3 ; tol_rtol = 1e-3 ; tol_slack = 1.5
# err_* 都是相对 fp64 oracle 的最大绝对误差；
# 判据：allclose(atol, rtol) 且 err_kernel <= err_baseline * tol_slack
```

- 参考实现：`lemoq.kernels.reference.dequant_gemm_fused_reference`（纯 PyTorch，CPU 可测）；
  落点：`tests/test_kernel_parity.py::test_triton_fused_gemm_matches_reference`
  （**骨架已存在**，上机后把 `_triton_fused_gemm` 的桩换成真 kernel 即可）。
- 两条参考路径的数学等价性与"等于 fp32 反量化 matmul"**离线已测**（该文件现有 2 条测试）。

### K2 — 端到端正确性门（硬规则 2）

**引用 `EXP-p1-04` 的实测，不重测、不放宽**：

- `w8a16 / g128 / ctx=1024 / targets=(attention,experts)` → KL mean **0.001273 < 0.01**
  → **唯一可发表性能的配置**。w8a8、w4a16 的格子照跑（它们是"融合没有改变正确性判定"
  的 evidence），但 `verdict=FAIL`，**其延迟不得作为性能结论引用**。
- runner 已强制执行：`GATE_CITATIONS` 表 + 三条拒绝（见 §3）。引用值与 `EXP-p1-04`
  的落盘工件由 `tests/test_runner_cells.py::test_gate_citations_match_p104_artifact`
  逐字钉住。
- 换任何一项（g=64、ctx=4096、targets 含 router）= 没有门实测 → runner 直接拒绝。
  这不是保守，是把"补测精度"作为硬前提。

### K3 — 性能测量协议

1. **同进程、交错**（round-robin + MAD）：`separate` 与 `fused` 必须在同一次运行里交替测，
   口径与 `EXP-p2-05` 一致；单臂顺序跑的数字不采信。
2. **计时前查占用**：`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`。
   实测过外人进程把 ctx=4096 的 per-token 从 3.2356 ms 顶到 10.38 ms（host-share 67.5%）。
   发现占用先记录，不得去杀别人的进程。
3. 报 p50/p99（`docs/METRICS.md` §2），并报 `dequant_share`（物化权重占单步时间的比例，
   由隔离微基准或 ncu 得到；`NCU_METRICS` 见 `lemoq.kernels.autotune`）。
4. 带宽地板只用**实测** 1561.9 GB/s（`results/machine_probe.json`）。
5. 报绝对值与比值（Hoefler 规则 #1）。

## 3. 离线已交付（本记录的一部分，全部有测试）

> ★ 2026-10-07 更新：本表是**离线**交付；§4 的 5 条现在**全部完成**（kernel、`dequant_share`、
> 6 格、记录、兜底未触发）。执行结果与机制写在 §6。

| # | 内容 | 验证方式 | 现状 |
|---|---|---|---|
| 1 | cell id 由**矩阵声明的轴**拼出（`build_exp_id`），顺序 = yaml 里 factors 键的顺序，然后 ctx | `results/` 里**全部**带轴格子的 id 可被复原（测试逐个断言） | ✅ |
| 2 | 加载前的拒绝：本机没有 triton/CUDA 时 fused 被拒 / 无门实测 / 量化+cudagraph 未接线 | 3 条 CLI 测试（无 GPU 机器上跑） | ✅（fused 的拒绝理由已从"kernel 未实现"改为"本机不可用"） |
| 3 | `GATE_CITATIONS` 与 `EXP-p1-04` 工件逐字一致 | 工件对照测试（工件缺失时 skip） | ✅ |
| 4 | `quantize_for_cfg` + `weight_bytes_actual` 的 buffer 陷阱 + `byte_budget(quant=…)` 接线 | CPU tiny OLMoE：替换计数、前向可跑、packed/scale 在 buffer 里 | ✅ |
| 5 | CLI：`--bit-width / --dequant / --group-size` + dry-run 打印格子 id 与门状态 | dry-run 测试 ×3 | ✅ |
| 6 | 矩阵 `w8a16` 轴 + 计数 48→50 | `test_matrix.py` | ✅ |

★ 陷阱记录：`QuantLinear` 把 packed/scale 存成 **buffer**，而 `weight_bytes_actual`
只数 `model.parameters()` —— 不补上 buffer 会让量化格的驻留字节**少算 96%**
（experts 占大头）。测试 `test_quantize_for_cfg_replaces_and_model_still_forwards`
直接断言这一点，防止回退。

## 4. 上卡前还差（顺序即执行顺序）—— **全部完成（2026-10-07）**

1. **写 fused kernel** ✅ —— `src/lemoq/kernels/_dequant_triton.py`（4/8-bit，无 autotune、
   grid 为 host 常量 → P3 可 capture）。K1 数值门 **PASS**（§6.1）。
   `QuantLinear` 的 fused 分支只对 **decode 单 token** 生效，prefill / 多 token 退回 separate
   并**计数**（`_measurement.dequant.activity`）。
2. **`dequant_share` 测量接入** ✅ —— 不再是 `None`：来自**同进程交错 A/B** 的
   `(p50_separate − p50_fused) / p50_separate`，写进 `attribution.dequant_share_pct`
   （gate PASS 的格子才有；FAIL 格按硬规则 2 置 None）。**不是 ncu 的 dram 计数**，payload 里
   写了 `dequant_share_source` 说明口径。
3. 跑 6 格 ✅ —— 命令与预注册一致（`--exp EXP-p2-01-dequant-fusion --bit-width {w8a16,w8a8,w4a16}
   --dequant {separate,fused}`，ctx=1024 默认）。★ 注意：矩阵**没有为 p2-01 声明 `config.moe`**，
   所以 6 格走的是 **hf** MoE 路径（experts 也被 `QuantLinear` 替换）—— 这决定了格子的绝对
   时间（192-398 ms/token）与"哪些结论能外推"（§6.5）。
4. 每格：`result.json` + 本记录补"结果"节 + commit ✅（§6；raw 另存 `results/EXP-p2-01-run/`）。
5. 兜底（vLLM/torchao 当"已融合"臂）—— **未触发**，自研 kernel 的 K1 门一次通过。

## 5. 判据（跑完之后怎么判）

```text
支持假设：fused/separate 的 per-token 比值 ∈ [1.3, 2.0]，且 dequant_share 显示
          separate 的物化流量占单步时间的可观比例；
否证信号：差值 <10%（矩阵原文）→ 反量化不是瓶颈，融合判断被否证 ——
          按计划书，贡献转落到"字节预算分析 + E7 截断解释"；
分不出：  两臂差值小于交错测量的 MAD → 记 inconclusive，不许往任一方向解读。
```

**门槛不因结果改动**：K1 的 1e-3/1e-3/1.5、K2 的 0.01/0.05/0.95 全部来自
`docs/METRICS.md` §1 与既有 kernel 的 PARITY_GATE。

## 6. 结果（2026-10-07，RTX 5090 32 GB，commit `e4a3803`）

### 6.1 K1 数值门：**PASS**（kernel 与参考实现一致，容差未放宽）

```bash
# 服务器
export PATH=/root/miniconda3/bin:$PATH
python -m pytest tests/test_kernel_parity.py -q      # 6 passed, 1 skipped
#   test_triton_fused_gemm_matches_reference  ← 本条的 K1 门
```

覆盖：8-bit / 4-bit（4-bit 走 nibble 寄存器展开，独立代码路径）、m=1（decode）与 m=4、
fp16 scale（**部署口径**，`QuantLinear` 按字节账存 fp16）与 fp32 scale、
`k=500` 不被 group_size 整除（尾部补零由 padded_in 表达）、group_size=64。
判据 = `PARITY_GATE`（`atol/rtol 1e-3`），**一次通过，未放宽**。

另做了一次**模块级**的端到端交叉核对（8-bit、四个 attention 形状、真实 `QuantLinear`）：
`fused` 与 `separate` 的输出 `max_abs_err ≈ 9.8e-4`，相对 `max|y|` 为 **5.1e-4 … 6.0e-4**
（差异只来自 fp32 累加顺序与 fp16 舍入）。

### 6.2 测量协议（K3）与污染检查

| 检查 | 实测 | 判定 |
|---|---|---|
| 计时前 `nvidia-smi --query-compute-apps` | 6 格前后**均为空**（`results/EXP-p2-01-run/contention.log`） | ✅ 无外人进程 |
| 首尾金丝雀（`w8a16 × separate`，同格 `--force` 重跑） | p50 **190.8908 → 192.0597 ms**，相对差 **0.61%** ≪ 5% | ✅ **本轮有效** |
| 金丝雀 p99 / 样本 | A: p99 200.77，min/max 189.19/204.42；B: p99 252.52，min/max **188.11/298.75**（1 个 ~299 ms 的 host 抖动） | 记录，不改判定（p50 与配对中位数稳健） |
| 两臂计时方式 | **同进程、逐步 round-robin**（两臂交替走同一条 KV 流，按步号配对）；每臂 3 轮 × 20 步 = 60 样本 | ✅ K3.1 |
| 首调 JIT 编译 | 单独记、**不计入稳态**：`separate` 格首调 fused=503.3 ms vs separate=202.8 ms（~300 ms 是 Triton 编译）；`fused` 格首调 fused=196.2 ms（编译已在前面的 warmup 相位完成） | ✅ `docs/METRICS.md` §2 |
| 臂是否真的跑到 | `_measurement.dequant.activity`：`fused_calls=26,880`、`fallbacks={multi_token: 161,280}`（prefill 必然回退） | ✅ 不是"空对照" |

### 6.3 六个格子（`moe=hf` —— 矩阵未为 p2-01 声明 `config.moe`，见 §6.5）

`p50/p99` 单位 ms；`sep`/`fus` 是**同一次运行里**两臂的 p50（`interleaved_ab.arms`）；
`Δ` 是逐对配对的中位数差（`sep − fus`，负 = fused 慢）；`wins` 是逐对胜负。

| 格 | gate | p50 | p99 | sep p50 | fus p50 | ratio fus/sep | Δ 中位 | MAD | wins（60 对） | host% | kernel/步 | `[W]` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `w8a16 × separate` | PASS | **192.06** | 252.52 | 192.06 | 201.79 | **1.0507** | −10.08 | 2.51 | sep 56 / fus 4 | 71.7 | 19,029 | 1.301 GB |
| `w8a16 × fused` | PASS | **198.73** | 218.00 | 188.94 | 198.73 | **1.0518** | −9.11 | 1.52 | sep 60 / fus 0 | 71.7 | 19,029 | 1.301 GB |
| `w8a8 × separate` | FAIL | _(305.37)_ | _(311.11)_ | 305.37 | 321.53 | 1.0529 | −16.30 | 1.98 | sep 60 / fus 0 | 80.4 | 24,405 | 1.301 GB |
| `w8a8 × fused` | FAIL | _(318.55)_ | _(323.90)_ | 302.45 | 318.55 | 1.0532 | −16.27 | 1.43 | sep 60 / fus 0 | 80.3 | 24,405 | 1.301 GB |
| `w4a16 × separate` | FAIL | _(398.05)_ | _(449.38)_ | 398.05 | 384.36 | **0.9656** | +13.43 | 1.81 | **fus 59** / sep 1 | 60.7 | 43,219 | 0.764 GB |
| `w4a16 × fused` | FAIL | _(383.95)_ | _(417.26)_ | 397.66 | 383.95 | **0.9655** | +14.05 | 2.41 | **fus 58** / sep 2 | 61.0 | 43,219 | 0.764 GB |

- 斜体 `_(…)_` = **gate FAIL 的格子**：按硬规则 2，`latency` 段为 `null`，实测值只在
  `_measurement.gate_failed_evidence` 里（**不得**被引用为性能结论）。这是本次给 runner
  加的机械执行点，`validate_result` 有对偶测试。
- 其它共同量：TTFT p50 199.1/200.7（w8a16）、312.4/316.6（w8a8）、407.8/416.0（w4a16）；
  `host_share` 60.7–80.4%；`gpu_busy` 57.2 ms（w8a16）、62.8（w8a8）、148.9（w4a16）；
  `[T]` = 1.435 GB；达成率 **0.4–0.5%**（带宽地板 1561.9 GB/s）；VRAM 峰值 12.95 GiB（含
  加载相位的 fp16 模型）；`bytes.budget_vs_actual_delta_pct = 456%`——这是**预期的**：
  驻留的是全部 64 个专家，而 `[W]` 只算每 token 的 top-8（payload 里已有该注）。

### 6.4 隔离测量：kernel 本身是快的，慢的是"每次调用的 host 开销"

**(a) 模块级、同进程交错**（7 轮 round-robin、每轮 200 次调用、8-bit、attention 四投影之和
→ 这里的数字是**每 token 每层**，×16 层就是每 token）：

| 臂 | p50（ms/token/层） | MAD | ×16 层 |
|---|---|---|---|
| `fp16`（`F.linear`，未量化） | 0.0503 | 0.0007 | **0.805 ms** |
| `fused`（本 kernel，默认配置） | 0.1338 | 0.0019 | **2.141 ms** |
| `separate`（物化 fp16 权重 + `F.linear`） | 0.1624 | 0.0028 | **2.599 ms** |

★ **v2 重测（2026-10-07 晚，P2d 轮）**：投影 kernel 在 `7d1f92b` 改进后重跑本实验 6 格，
同进程交错 A/B 的 `fused/separate` 比：

| bit_width | v1 比 | **v2 比** | v1 → v2 |
|---|---|---|---|
| w8a16 | 1.0518 | **1.0442** | 判定不变 |
| w8a8 | 1.0532 | **1.0523** | 判定不变 |
| w4a16 | 0.9655 | **0.9681** | 判定不变 |

⇒ **E2 的判定一个字都不用改**，而且这本身是一条证据：新 kernel 在**图内**让投影快 1.58×
（`EXP-p4c-01` §5.6），但在这个 **host 受限的 eager 臂**里完全看不见（host 61–81%）。
同一件事在 `EXP-p4b-01` 的图内臂里是可见的（fused `speedup_cudagraph` 1.1071 → 1.1703）。
v1 的 6 个 `result.json` 归档在 `results/EXP-p2-01-run/v1_archive/`。

★ **口径提醒（2026-10-07 补）**：这张表的 fused/separate 两行**都被 Triton launcher 主导**
（同节 (b) 自己测出 launcher 地板 16–32 µs）。图里没有 launcher，`EXP-p4c-01` §5.4 的 profiler
给的是 fused kernel **8.2 µs/次**；`EXP-p4c-01` §5.6 又把 kernel 本身从 8.14 改到 **5.15 µs/次**
（按组广播 scale）。所以本表的**绝对值是改前、且是 launcher 口径**，不要与图内数字混比；
E2 的**判定**（eager 1.05×、host 受限）不受这两件事影响。

⇒ 隔离测里 **fused 比 separate 快 1.21×**（每调用 33.5 µs vs 40.6 µs）——但两者都远慢于
fp16 的 12.6 µs。默认配置是声明空间的一个**子集**扫描（B4K512w4 并列最快 0.1338，
其余 7 个配置 0.1344–0.1377，差 ≤3%）。**不是全网格 autotune**，别当 autotune 引用。

**(b) Triton launcher 的地板**（干净卡、300 次 launch）：3 参数 **16.0 µs**、20 参数 **31.8 µs**
（≈1 µs/参数）、`torch.add` 5.6 µs。所以 fused 的每次调用（33.5 µs）**基本就是 launcher**，
kernel 本体（读 4 MB int8 + 写 8 KB ≈ 2.6 µs）被完全掩盖；而 separate 的 40.6 µs 里
也主要是 Python/torch 算子链。为此本会话把 kernel 签名从 19 参数砍到 9 参数
（连续契约，stride 由形状推出），fused 每 token/层从 0.1575 → 0.1338 ms。

### 6.5 ★ 判定：E2 的 "1.3–2×" 在本配置 **否证**（不是 inconclusive）

按 §5 的预注册判据：

```text
支持假设：ratio ∈ [1.3, 2.0]          → 实测 w8a16 = 1.0507 / 1.0518（**fused 更慢**）   ✗
否证信号：差值 < 10%                   → 实测差 5.1%，且 |Δ|=9.1–10.1 ms > MAD 1.5–2.5 ms
                                        （可分辨，所以这对差值是真信号，不是噪声）        ✓ 触发
分不出  ：|差值| < MAD                 → 不适用（差值远大于 MAD）
```

**outcome = `refuted`**：在预注册的 eager 配置里，融合既没有 1.3–2×，也没有可测收益。
`w4a16` 的 evidence 反而显示 fused **快 3.6%**（58–59/60 对），但它的门 FAIL、不得发表。

机制（三条都有实测支撑，不靠推断）：

1. **这 6 格全是 host/launch 受限**：`host_share` 60.7–80.4%、`gpu_busy` 只有 57–149 ms 而
   单步 192–398 ms、每步 **19,029–43,219** 个 kernel、带宽达成率 0.4–0.5%。
   ⇒ 在这种状态下，决定每步时间的是**每次 `QuantLinear` 调用的 Python/torch 开销**，
   不是权重字节。
2. **融合省的字节正好被这次调用开销吃掉**：fused 每调用 33.5 µs，separate 40.6 µs（隔离测），
   但 fused 的 33.5 µs 里 ~16 µs 是 launcher 地板；单元格里两臂的 host 成本因此只差 5%
   （方向还与隔离测相反 —— 隔离测时设备未被饱和，单元格里 host 已饱和）。
3. **4-bit 例外解释了方向**：4-bit 的 separate 臂要走 nibble 解包算子链（`unpack_4bit`），
   单元格里 kernel 数从 19,029 涨到 **43,219**、单步 398 ms；fused 把解包挪进 kernel，
   于是**快 3.6%**。即"融合的收益取决于 separate 臂有多少 host 算子"，而不是取决于字节比。

**对 `EXP-p4b-01` §2.7 那句 "+2.02 ms = attention 的 separate 臂" 的修正**：那 +2.02 ms
**不是（或不只是）物化权重的 DRAM 流量**。本会话把物化**整条去掉**（fused 臂一趟只读 packed），
单元格不但没有变快，还慢了 5%。所以 §2.7 的税主要是"每次 forward 多几个 torch 算子 + 一次
多余 GEMM 调用"的**延迟/算子数**成本。
→ 对 p4b 的 grouped 格（ctx=1024、w8a16、attention 64 次调用/步）**外推**：隔离测里
fused 只拿回 separate 超出 fp16 的 27 µs/call 中的 7 µs，所以预计只能拿回 ~0.4–0.5 ms
（≈ 那 2.02 ms 的 1/4）。**这是预测，不是测量** —— 要在 p4b 的 grouped 格上直接测，
得先给 cell id 加 `moe` 轴（否则 grouped 的重跑会**覆盖**已落盘的 hf 格）。

### 6.6 本条**没有**回答的问题（不许被读成结论）

- **量化 + cudagraph 的收益**：本条只跑 eager 臂。cudagraph 会把 host/launch 开销压掉
  （p4b §2.6：fp16 上 67% → 1.4%），届时 fused 的 kernel 优势（2.6 µs vs 28 µs 的 GPU 工作）
  才可能兑现 —— 那是 **P3 / E7**，本轮**没有**测，故**不给 E7 判定**。
- **grouped MoE 配置下的 attention 融合**：未测（§6.5 末尾）。
- **带宽达成率**：本条 0.4–0.5% 只说明 hf+量化的这条路径离带宽墙极远，**不能**用来评价
  kernel 的带宽效率（那个数在 §6.4 的隔离测里）。

## Applicability（按实测填全）

```text
适用于：  RTX 5090 32 GB（sm_120、170 SM、驱动 595.71.05）、**实测**读带宽 1561.9 GB/s、
          python 3.12.3 / torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6、
          OLMoE-1B-7B-0924（bf16 ckpt → fp16）、单卡、batch=1、dtype=fp16、ctx=1024、
          group_size=128、targets=(attention, experts)、MoE 实现 = hf（矩阵未给 p2-01 声明
          config.moe）、量化格 4/8-bit；
          K1 的 kernel 数值结论（4/8-bit、m=1、fp16 scale、g=64/128）对同一 (torch, triton, sm_120)
          组合成立；换 triton 版本必须重跑 K1。
不适用于：其它 group_size / ctx / targets（门没有实测，runner 会拒绝）；
          batch>1；量化 × cudagraph（**未接线**，runner 仍拒绝 —— 那是 P3）；
          量化 × gather（无量化版）；
          4-bit 的 grouped expert 路径（nibble 解包未实现）；
          w8a8 / w4a16 的**性能结论**（两格门 FAIL，延迟只作 evidence）；
          **grouped MoE 配置下的 attention 融合**（未测，见 §6.5）；
          V100 (sm_70) / 其它架构 —— 没有任何数字可以搬过去。
原因：    两臂的差值绑定在这台卡的 **launcher/host 开销**上（triton 3 参数 launch 16 µs、
          20 参数 32 µs，见 §6.4b），而 hf+量化路径每步要发 1.9–4.3 万个 kernel：
          host 侧一换机器/一换版本，符号都可能翻转。K2 的门只在 p1-04 的配置上实测过。
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
| `w4a16 · dequant=fused` | 382.9683 | 381.7138 | -0.3% | **382.9683** | FAIL → **FAIL** | **作废**：刷新时 host 主导（62%，共租 CPU 压住 host） |
| `w4a16 · dequant=separate` | 423.9902 | 396.4504 | -6.5% | **423.9902** | FAIL → **FAIL** | **作废**：刷新时 host 主导（62%，共租 CPU 压住 host） |
| `w8a16 · dequant=fused` | 200.6536 | 198.1238 | -1.3% | **200.6536** | PASS → **PASS** | **作废**：刷新时 host 主导（72%，共租 CPU 压住 host） |
| `w8a16 · dequant=separate` | 189.0652 | 192.8617 | +2.0% | **189.0652** | PASS → **PASS** | **作废**：刷新时 host 主导（72%，共租 CPU 压住 host） |
| `w8a8 · dequant=fused` | 324.8301 | 320.5665 | -1.3% | **324.8301** | FAIL → **FAIL** | **作废**：刷新时 host 主导（81%，共租 CPU 压住 host） |
| `w8a8 · dequant=separate` | 304.8367 | 615.8398 | +102.0% | **304.8367** | FAIL → **FAIL** | **作废**：刷新时 host 主导（93%，共租 CPU 压住 host） |

**结论**：图内臂的绝对延迟已按新实现更新（上表"采用"列）；host 主导格子的 pre-P2i 值保留，
其**结论不受影响** —— 融合 norm 省下的是每步 ~0.34–0.48 ms 的常数，在 8–272 ms 的 host 受限
步长里低于噪声（这一点 `EXP-p2d` 已经量过：同一个 kernel 改进在 eager 臂里完全看不见）。
把 host 主导的 18 格在**安静的机器**上重跑，列为 **P2k**。
