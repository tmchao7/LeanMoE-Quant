# EXP-p2-01-dequant-fusion

> **状态：`planned` —— 本记录不含任何性能数字。** 它是上卡**之前**写的：假设来自
> 矩阵，三条门在本记录冻结，跑完之前不得改动（改门必须单独成 commit 并写明理由）。
> 离线已交付的部分与还差的部分分列在 §3/§4，谁来跑一目了然。

- **ID**: `EXP-p2-01-dequant-fusion`
- **Date**: 2026-10-06（假设与门冻结；**无结果**）
- **Operator**: 尚未执行（需要 RTX 5090）
- **Status**: `planned`
- **原始数据**: 无（跑完后应有 `results/EXP-p2-01-dequant-fusion__bit_width=<x>_dequant=<y>/result.json` × 6）
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

| # | 内容 | 验证方式 |
|---|---|---|
| 1 | cell id 由**矩阵声明的轴**拼出（`build_exp_id`），顺序 = yaml 里 factors 键的顺序，然后 ctx | `results/` 里**全部**带轴格子的 id 可被复原（测试逐个断言） |
| 2 | 三条拒绝在**任何模型加载之前**失败：fused kernel 未实现 / 无门实测 / 量化+cudagraph 未接线 | 3 条 CLI 测试（无 GPU 机器上跑） |
| 3 | `GATE_CITATIONS` 与 `EXP-p1-04` 工件逐字一致 | 工件对照测试（工件缺失时 skip） |
| 4 | `quantize_for_cfg` + `weight_bytes_actual` 的 buffer 陷阱 + `byte_budget(quant=…)` 接线 | CPU tiny OLMoE：替换计数、前向可跑、packed/scale 在 buffer 里 |
| 5 | CLI：`--bit-width / --dequant / --group-size` + dry-run 打印格子 id 与门状态 | dry-run 测试 ×3 |
| 6 | 矩阵 `w8a16` 轴 + 计数 48→50 | `test_matrix.py` |

★ 陷阱记录：`QuantLinear` 把 packed/scale 存成 **buffer**，而 `weight_bytes_actual`
只数 `model.parameters()` —— 不补上 buffer 会让量化格的驻留字节**少算 96%**
（experts 占大头）。测试 `test_quantize_for_cfg_replaces_and_model_still_forwards`
直接断言这一点，防止回退。

## 4. 上卡前还差（顺序即执行顺序）

1. **写 fused kernel**。按 `src/lemoq/kernels/__init__.py` 记录的决定，未编译的
   Triton kernel **不在离线阶段写**（离线无法发现编译/语义错误，只会在计费机器上
   制造调试时间）。落点：`src/lemoq/kernels/_dequant_triton.py`（仅 GPU 路径 import）
   + `QuantLinear` 的 fused 分支（`dequant="fused"`）。用 `separate` 与 `fused` 的
   数值相同性做 K1。
2. **`dequant_share` 测量接入**（隔离微基准或 ncu），当前 payload 里是 `None`。
3. 跑 6 格（`separate` 三格**不依赖 kernel**，可先跑）：

   ```bash
   python -m benchmarks.runner --exp EXP-p2-01-dequant-fusion \
       --bit-width {w8a16,w8a8,w4a16} --dequant {separate,fused}
   ```

   `--dry-run` 可先核对格子 id 与门状态（已在 Mac 上验证 6 格 id 正确）。
4. 每格：`results/<cell>/result.json` + 本记录补"结果"节 + commit（status=done）。
5. 若到 25 GPU 小时 fused 仍未达实测带宽的 70%：走矩阵声明的 `fallback`
   （用现成融合 kernel（vLLM/torchao）当"已融合"臂），并在本记录写明走了兜底。

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

## Applicability（上机后必须按实测填全）

```text
适用于：  （待填：GPU 型号、实测带宽、驱动、dtype、batch=1、ctx=1024、
          group_size=128、targets=attention+experts、torch/triton 版本）
不适用于：其它 group_size / ctx / targets（门没有实测，runner 会拒绝）；
          batch>1；W8A8 激活路径的性能结论（该配置门 FAIL）；
          量化 + cudagraph（接线未完成，runner 会拒绝）
原因：    收益比绑定在这台卡的带宽与 launch 开销上；K2 的门只在 p1-04 的配置上实测过。
```
