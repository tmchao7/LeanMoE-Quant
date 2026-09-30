# SPECS — 机器规格与硬约束

> **状态说明**：本文区分 **已确认** 与 **待实测** 两类。任何"待实测"项在 Phase 0 落盘前，
> 不得用于计算或写进结论。

---

## 1. 主开发机：V100 32 GB

| 项 | 值 | 状态 |
|---|---|---|
| 架构 | Volta, **sm_70** | 已确认（型号即决定架构） |
| 显存 | **32 GB HBM2** | ✅ 已确认 |
| 物理卡数 | 15 | ✅ 已确认 |
| 本项目用卡 | **1** | 项目约束 |
| 型号（SXM2 / PCIe） | — | ⏳ 待实测 |
| 可达带宽 | 规格约 ~900 GB/s | ⏳ **必须实测**，勿用规格值 |
| FP16 张量核吞吐 | 规格约 ~125 TFLOPS | ⏳ 待实测 |
| Shared memory / SM | 预期 ~96 KB | ⏳ 待编译探针确认 |
| 拓扑（NVLink/NVSwitch） | — | ⏳ 待实测（本项目不用多卡，但要知道） |

### 1.1 sm_70 特性矩阵（对本项目的影响）

| 特性 | sm_70 是否有 | 对本项目的影响 |
|---|---|---|
| BF16 张量核 | ❌ | **模型必须 fp16**；bf16 checkpoint 需转换并验证精度 |
| `cp.async` | ❌（Ampere 起） | 反量化与 GEMM 的重叠只能靠双缓冲 + 手工调度 |
| TMA | ❌（Hopper 起） | 无法使用张量内存加速器 |
| `wgmma` | ❌（Hopper 起） | 只能用 `mma.sync` / `nvcuda::wmma` |
| Thread-block cluster / DSMEM | ❌ | 无法用分布式共享内存 |
| Cooperative groups `grid.sync()` | ✅ | 可用（MegaKernel-vLLM 用过） |
| FP16 / INT8 张量核 | ✅ | 本项目的计算基础 |
| INT4 张量核 | ⚠️ 有但生态弱、路径不成熟 | W4A16 实际是"反量化到 fp16 → fp16 张量核" |

**关键推论**：W4A16 在 V100 上**每个权重都要过一次反量化**，其相对开销比在 Ampere+ 上更高。
这不是缺陷，而是本项目更尖锐的实验条件——也必须在结论中说明。

### 1.2 报告必须附带的适用区间声明

任何结论都要写明：

> 本结论在 **V100 32GB（sm_70，实测可达带宽 ____ GB/s）** 上、**fp16** 精度、
> **单卡**、**batch ≤ ___** 条件下成立；由于 V100 无 BF16 与 `cp.async`，不保证迁移到
> Ampere/Hopper/Blackwell，也不保证在 bf16 下成立。

---

## 2. 对照机与选卡（可租）

### 2.1 跨代特性矩阵 —— **已核实，不是规格传闻**

下表不是从 datasheet 抄的，而是从**本地 LLVM NVPTX 后端的指令谓词**逐条读出来的
（`llvm-project/llvm/lib/Target/NVPTX/NVPTXInstrInfo.td`、`NVPTXIntrinsics.td`，
2026-09 checkout）。这可核对、可复现：

| 特性 | V100 sm_70 | 4090 sm_89 | 5090 sm_120 | H100 sm_90a |
|---|---|---|---|---|
| `cp.async` | ❌ `Requires<[SM80]>` | ✅ | ✅ | ✅ |
| TMA | ❌ | ❌ | ✅（经 SM90 基础架构） | ✅ |
| Blackwell 扩展 TMA | ❌ | ❌ | ❌ `[SM100f,SM110f]` | ❌ |
| **`wgmma`** | ❌ | ❌ | ❌ **`Predicates=[SM90a]`** | ✅ |
| `tcgen05` | ❌ | ❌ | ❌ `[SM100a,103a,110a]` | ❌ |
| FP8 转换 | ❌ | ✅ `SM89 & PTX8.1` | ✅ | ✅ |
| **MX block-scale MMA**（MXFP4/NVFP8） | ❌ | ❌ | ✅ **`[SM120f]`** | ❌ |
| bf16 张量核 | ❌ | ✅ | ✅ | ✅ |
| `grid.sync()` | ✅ | ✅ | ✅ | ✅ |

**两条容易踩错的推论**：

1. **40/50 系消费卡都拿不到 `wgmma`**——`wgmma` 门在 `SM90a`，而 `a` 变体不被任何后代架构
   隐含。所以"租 40/50 卡以获得 Hopper 式 producer/consumer 流水写法"这个动机**部分落空**：
   你拿到的是 `cp.async`（4090/5090）、TMA（仅 5090）、FP8（两者），但**不是 wgmma**。
   要那个信号只有 sm_90（H100）——昂贵，且代码无法从别的卡迁移。
2. **5090 有一个被忽略的优势**：`hasMMABlockScale → [SM120f]`，即消费级 Blackwell
   支持 **block-scale（MXFP4 / NVFP8）MMA**。这是 2025–2026 量化 kernel 的主流格式方向，
   而 **V100 和 4090 都做不到**。这给项目留了一条天然扩展线：`W4A16 → MXFP4`。

### 2.2 选卡结论（**待用户最终确认**）

| 角色 | 卡 | 理由 | 成本 |
|---|---|---|---|
| **主力（开发）** | **校园 V100 32GB** | 免费、15 卡、长期可用 → **无限迭代**。方法论、带宽标定、`E1/E4/E5` 敏感度矩阵这些需要几十次重复的实验，只可能在这里做完 | ≈ 0 |
| **对照（租）** | **RTX 5090 32GB** | **与 V100 同显存**（同一模型同样放得下）→ 只变带宽这一个变量的干净对照；带宽近 2×；独有 MX block-scale；且能试 30B INT4 单点 | 短时租用 |
| **跳过** | ~~RTX 4090 24GB~~ | **被 5090 支配**：24 GB 装不下 30B INT4 前后的 KV/workspace，带宽只比 V100 规格高约 12%（对照太弱），**且同样没有 wgmma** | — |
| **可选收尾** | L40S / A6000 48GB | **唯一能跑 Qwen3-30B-A3B INT4 单点的"便宜"档位**（V100 32GB 装不下）；带宽 864/768 GB/s 反而不如 5090 | 需确认可租到 |
| **可选收尾** | H100 48/80GB | 仅当"想要 Hopper 级 kernel 实现（wgmma+TMA）"这个信号时才值；代码不可从 V100 迁移 | 贵，短窗口 |

**为什么 5090 优于 4090，且与价格无关**：单次实验的**信息量**取决于"变了几个变量"。
5090 与 V100 同为 32 GB，因此"同一模型、同一 batch、只换带宽"是成立的；
4090 的 24 GB 会强迫你同时改模型或 batch，对照立刻脏掉。租卡总时长只有 8–12 小时量级
（见 `PLAN.md` §8），价差在总成本里可以忽略，**信息量才是稀缺资源**。

**跨卡纪律（修订）**：跨卡默认只能对照**结论**。但**若实现用 Triton 写成同一份源码**，
则同一 kernel 可以在 sm_70/sm_89/sm_120 上编译运行——此时**可以对照实现**，
这比原计划（只对照结论）更强。这正是 `PLAN.md` §3 把 Triton 排在前面的一个额外理由。
前提：**Triton 对 sm_70 的支持是本项目的最弱环节，必须在阶段 0 用编译探针验证**
（已加入 §4 的探测字段）。

### 2.3 租卡纪律（`PLAN.md` §1.3 的补充）

- 一个开机周期只跑一个完整实验矩阵，**禁止"开卡改代码"**；
- 上卡前必须本地 `--dry-run` 通过；
- 结果落 JSON 支持断点续跑；
- **优先 5090 而不是 4090**（理由见 §2.2）。

---

## 3. 显存预算（**推导值，非实测**）

> 下表是算术推演，用于阶段 0 选型，**不得当作测量结果引用**。

以 fp16（2 B/参数）计算权重占用：

| 模型 | 总参 | fp16 权重 | 4bit 权重 | 32GB V100 余量 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | ~1.0 GB | ~0.3 GB | 充裕 |
| Qwen2.5-1.5B | 1.5B | ~3.0 GB | ~0.8 GB | 充裕 |
| OLMoE-1B-7B | ~6.9B | ~13.8 GB | ~3.5 GB | ✅ 充足 |
| Qwen3-4B | 4.0B | ~8.0 GB | ~2.2 GB | ✅ |
| Qwen3-8B | 8.2B | ~16.4 GB | ~4.5 GB | ✅ 紧但可行 |
| Qwen3-30B-A3B | 30.5B | ~61.0 GB | ~15.3 GB | ❌ 装不下（KV+激活+workspace 挤爆）|

`OLMoE-1B-7B` 与 `Qwen3-30B-A3B` 的架构数字引自
[MegaKernel-vLLM docs/16](../../MegaKernel-vLLM/docs/16_MoE_A100可行性调研.md)；
其余为按参数量与 2 B/参数的算术推演。

---

## 4. Phase 0 探测脚本要产出的字段

`tools/probe_machine.py --json results/machine_probe.json` 至少包含：

```text
gpu_name, compute_capability, sm_count, arch
vram_total_bytes
physical_gpu_count
driver_version, cuda_runtime_version, torch_version
measured_bandwidth_gbps          # median of >= 3 runs, with dispersion
measured_bandwidth_read_only_gbps  # 只读模式（reduction）—— decode 是读主导，copy 会低估上限
measured_fp16_tflops
cp_async_supported               # compile probe
shared_mem_per_block_max_bytes
bf16_tensor_core_available       # expected: false
# --- 阶段 0 新增（补齐前置风险）---
triton_available_and_working_on_sm70   # ⚠️ 本项目的最大前置风险：Triton 对 Volta 支持最弱
ncu_available                    # Nsight Compute 是否可用（推理优化岗硬要求；非 cooperative kernel 可用）
cuda_toolkit_version             # 共享机上是否已有可编译的 nvcc / ptxas
torch_cuda_build                 # 需与上一条匹配，否则 cpp_extension 编译会失败
nvcc_compiles_for_sm70           # 用 torch.utils.cpp_extension 真编译一次小程序
disk_free_bytes                  # OLMoE + Qwen3-8B 权重落盘需要空间
```

每一项都要有**测量方法**与**不确定性**（重复次数 / 离散度），否则视为未测。

**为什么新增这四项**：

- `triton_*`：`PLAN.md` §3 把 Triton 排在 CUDA 之前（因为它让跨卡对照实现成为可能），
  但 Triton 对 sm_70 的支持历来是最弱的一档。**这个探针失败会让阶段 2 全盘返工**，
  必须在阶段 0 就确认。
- `ncu_available`：按 `MegaKernel-vLLM/docs/15` §4.2，profiling 是**推理优化岗的硬要求**，
  而 kernel 级证据（`dram__throughput`、occupancy、stall reasons）是本项目目前缺的那一块。
- `cuda_toolkit_version` / `nvcc_compiles_for_sm70`：`AGENTS.md` 明令**不得动共享机的
  CUDA/驱动**。若机上没有可用的 nvcc，或与 torch 的 CUDA 构建不匹配，**自写 CUDA kernel
  这条路在主力机上就不成立**——这必须在阶段 0 查清，而不是阶段 2 才发现。
- `disk_free_bytes`：`ENVIRONMENT.md` §3 提了但没给数字。
