# SPECS — 机器规格与硬约束

> **状态说明**：本文区分 **已确认** 与 **待实测** 两类。任何"待实测"项在 Phase 0 落盘前，
> 不得用于计算或写进结论。

---

## 1. 主开发机：RTX 5090 32 GB（sm_120）

| 项 | 值 | 状态 |
|---|---|---|
| 架构 | Blackwell 消费级，**sm_120** | 已确认（型号即决定架构） |
| 显存 | **32 GB GDDR7** | ✅ |
| 物理卡数 | 1（租用） | 项目约束 |
| 可达带宽 | 规格约 ~1792 GB/s | ⏳ **必须实测**，勿用规格值 |
| FP16 张量核吞吐 | — | ⏳ 待实测 |
| Shared memory / SM | 待编译探针确认 | ⏳ |
| 计费 | 按量（**非抢占式**） | 用户确认 |
| 磁盘 | 系统盘 100 GB，**需确认关机保留** | ⏳ |

### 1.1 sm_120 特性（对本项目的影响）

完整跨代矩阵见 §2.1（从本地 LLVM 谓词核实）。sm_120 与本项目相关的点：

| 特性 | 有 | 对本项目的影响 |
|---|---|---|
| `cp.async` | ✅ | **可以做标准的 producer/consumer 双缓冲流水**（V100 做不到） |
| TMA | ✅ | 可用的张量内存加速器 |
| FP8 | ✅ | **新增 W8A8-FP8 位宽档**（NVIDIA JD 明写 FP8） |
| **MX block-scale（MXFP4/NVFP8）** | ✅ | 消费级独有；项目的高级臂，也是需要解释的靶子 |
| bf16 张量核 | ✅ | 参考实现可用 bf16（注意其 greedy 非确定性，见 §1.3） |
| `wgmma` | ❌ | **门在 `SM90a`，40/50 系都没有**——拿不到 Hopper 式 warp-specialized 写法 |

### 1.2 第二数据点（免费）：学校 V100 32 GB（sm_70）

只用作答"量化收益如何随带宽变化"的第二个点（900 vs 1792 GB/s）。V100 的硬约束：

1. **无 BF16**：bf16 退化为 fp32 模拟。**一律 fp16**，bf16 checkpoint 必须转换并验证落差。
2. **无 `cp.async` / TMA / `wgmma`**：反量化与 GEMM 的重叠只能靠双缓冲 + 手工调度。
3. **带宽 ~900 GB/s**（规格）：本项目中低于 5090。
4. **共享机器**：不要动 CUDA/驱动/系统 Python；不要长期占满显存；跑完释放。

### 1.3 报告必须附带的适用区间声明

任何结论都要写明实际测量的那台机器，模板：

> 本结论在 **RTX 5090 32GB（sm_120，实测可达带宽 ____ GB/s）** 上、**fp16** 精度、
> **单卡**、**batch = 1**、**context = ___** 条件下成立。
> 由于 40/50 系消费卡均无 `wgmma`，且 V100 另缺 `cp.async`/FP8/FP4/bf16，
> **本结论不保证迁移到其它架构**，需要重新测量。

**两个必须显式声明的非确定性来源**：

- **bf16 greedy decode 在 logit 平局处是非确定性的** → 正确性门槛必须用 KL + top-1
  一致率，**不得用"逐 token 严格相等"**；
- **host/launch 开销依赖主机 CPU** → host CPU 型号必须随结果一起记录。

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

### 2.2 选卡结论（**已定：RTX 5090**）

| 角色 | 卡 | 理由 |
|---|---|---|
| **主力（已定）** | **RTX 5090 32GB（租，≈¥3/h）** | 唯一同时具备 `cp.async` + TMA + FP8 + **MX block-scale** 的档位；带宽 ~1792 GB/s 约为 V100 的 2× |
| **第二数据点** | 学校 V100 32GB（免费） | 跨带宽对照（900 → 1792）；无需额外成本 |
| **降级分支**（蹲不到 5090 时） | **RTX 4090 24GB** | **可用**：保留 `cp.async` + FP8，仅丢 TMA 与 MXFP4。**且 40 系没有 sm_120 那个 `no kernel image` 陷阱**（sm_89 从 torch 2.0 起就支持） |
| 避免 | L40S 48GB | 带宽 **864 GB/s，低于 V100 规格 900** —— 花钱换更低带宽，对本项目是反向优化 |
| 避免 | RTX A6000 48GB | **Ampere sm_86，没有 FP8** |
| 可选收尾 | H100 48/80GB | 仅当要 Hopper 级 kernel 实现信号（`wgmma`+TMA）时才值 |

**选卡铁律：带宽 > 显存。** OLMoE fp16 只有 13.84 GB，**24 GB 已足够**；而本项目的主指标是
带宽达成率，所以显存买到的边际价值远低于带宽与架构特性。**48G 不等于更快。**

**降级分支会自动生效**：`configs/experiments.yaml` 按 arch 挂载臂，
Gate-0 探测到 `sm_89` 时自动去掉 TMA/MXFP4 臂；探测到 `sm_86` 时再去掉 FP8 臂。

**跨卡纪律（修订）**：跨卡默认只能对照**结论**。但**若实现用 Triton 写成同一份源码**，
则同一 kernel 可以在 sm_70/sm_89/sm_120 上编译运行——此时**可以对照实现**。
这正是 `PLAN.md` §3 把 Triton 排在前面的额外理由。
前提：Triton 对老架构的支持较弱，**必须在 Gate-0 用编译探针验证**（见 §4）。

### 2.3 租卡纪律

- **只选按量计费，绝不选抢占式**（抢占会中断长实验）；
- 一个开机周期只跑一个完整实验矩阵，**禁止"开卡改代码"**；
- 上卡前必须本地 `--dry-run` 通过；
- 结果落 JSON 支持断点续跑；
- **镜像选系统镜像（`Ubuntu-nvidia`）**，不要选预装 PyTorch/vLLM 的平台镜像——
  那些是容器类型（不能再装 Docker）且 pin 了不透明的 torch 版本；
- 下单前**必须确认关机保留系统盘**，否则每次开机重付 4–6 h 配置税。

---

## 3. 显存预算（**推导值，非实测**）

> 下表是算术推演，用于阶段 0 选型，**不得当作测量结果引用**。

以 fp16（2 B/参数）计算权重占用：

| 模型 | 总参 | fp16 权重 | 4bit 权重 | 32GB 卡余量 |
|---|---|---|---|---|
| Qwen2.5-0.5B | 0.5B | ~1.0 GB | ~0.3 GB | 充裕 |
| Qwen2.5-1.5B | 1.5B | ~3.0 GB | ~0.8 GB | 充裕 |
| **OLMoE-1B-7B-0924** | **6.92B** | **13.84 GB**（实测字节数一致） | ~3.5 GB | ✅ 充足 |
| Qwen3-4B | 4.0B | ~8.0 GB | ~2.2 GB | ✅ |
| Qwen3-8B | 8.2B | ~16.4 GB | ~4.5 GB | ✅ 紧但可行 |
| Qwen3-30B-A3B | 30.5B | ~61.0 GB | ~15.3 GB | ⚠️ 短 ctx 也许可行，**需复核** |

> **重要**：本项目**不靠显存取胜**——OLMoE fp16 只有 13.84 GB，24 GB 卡已足够。
> 选卡的判据是**带宽与架构特性**，不是显存（见 §2.2）。

`OLMoE-1B-7B-0924` 的架构数字已由 `config.json` 逐项核实（见
[docs/MODELS.md](MODELS.md) §3.1）；`Qwen3-30B-A3B` 引自
[MegaKernel-vLLM docs/16](../../MegaKernel-vLLM/docs/16_MoE_A100可行性调研.md)；
其余为按参数量与 2 B/参数的算术推演。

⚠️ `Qwen3-30B-A3B` 的 32GB 可行性此前记为"❌ 装不下"。该结论取决于假定的 KV 池与 ctx，
**在 32GB 上做短 ctx 单点未必不可行**，上机时应实测复核，不要沿用旧结论。

---

## 4. Phase 0 探测脚本要产出的字段

`tools/probe_machine.py --json results/machine_probe.json` 至少包含：

```text
gpu_name, compute_capability, sm_count, arch
vram_total_bytes
physical_gpu_count, gpu_count_used
driver_version, cuda_runtime_version, torch_version
host_cpu_model                     # host/launch 开销依赖它
clocks_sm_mhz, clocks_mem_mhz      # 查降频
measured_bandwidth_gbps            # median of >= 3 runs, with dispersion
measured_bandwidth_read_only_gbps  # 只读模式（reduction）—— decode 是读主导，copy 会低估上限
measured_fp16_tflops
# --- 架构特性（按探测到的 arch 挂载，见 §2.1 矩阵）---
cp_async_supported
tma_supported
fp8_conversion_supported
mx_block_scale_supported          # sm_120 独有
bf16_tensor_core_available
wgmma_supported                   # 预期 false：门在 SM90a
shared_mem_per_block_max_bytes
# --- 前置风险（补齐）---
triton_available_and_working      # 跑通一个最小 tl.dot；失败会让 Phase 2 全盘返工
ncu_available                     # Nsight Compute；kernel 级证据
nsys_available
cuda_toolkit_version              # nvcc / ptxas 版本
torch_cuda_build                  # 需与上一条匹配，否则 cpp_extension 编译会失败
cpp_extension_compiles            # 用 torch.utils.cpp_extension 真编译一次小程序
disk_free_bytes                   # 模型落盘所需空间
```

每一项都要有**测量方法**与**不确定性**（重复次数 / 离散度），否则视为未测。
`tools/probe_machine.py` 在无 GPU 时以 `--dry-run` 产出**完整结构**（所有字段带 `null`
或 `not_measured`），以便消费端代码先在 Mac 上写完。

**为什么这些是前置风险**：

- `triton_*`：`PLAN.md` §3 把 Triton 排在 CUDA 之前（它让跨卡对照实现成为可能），
  但 Triton 对老架构支持较弱。**这个探针失败会让 Phase 2 全盘返工。**
- `ncu_available`：按 `MegaKernel-vLLM/docs/15` §4.2，profiling 是**推理优化岗的硬要求**，
  而 kernel 级证据（`dram__throughput`、occupancy、stall reasons）是本项目目前缺的那一块。
- `cpp_extension_compiles`：核心交付是自写 kernel，而 kernel 需要能编译。
  若编不了，路线要改成 Triton 优先——**必须在 Gate-0 得出，不能等到 Phase 2。**
- `host_cpu_model` / `clocks_*`：host/launch 开销已升为一等指标，它依赖主机 CPU 与是否降频。
- `disk_free_bytes`：`ENVIRONMENT.md` §3 提了但没给数字。
