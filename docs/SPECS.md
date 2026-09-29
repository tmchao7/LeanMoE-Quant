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

## 2. 对照机（可租）

| 卡 | 显存 | 规格带宽 | 用途 | 状态 |
|---|---|---|---|---|
| RTX 5090 | 32 GB | ~1792 GB/s | **首选对照**：同显存、近 2× 带宽 | 用户确认可租 |
| RTX 4090 | 24 GB | ~1008 GB/s | 次选 | 用户确认可租 |
| L40S | 48 GB | ~864 GB/s | 30B INT4 单点 | 需确认可租到 |
| A6000 | 48 GB | ~768 GB/s | 同上 | 需确认 |

**跨卡纪律**：跨卡只能对照**结论**，不能对照**实现**（kernel 依赖架构特性）。

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
[MegaKernel-vLLM docs/16](../MegaKernel-vLLM/docs/16_MoE_A100可行性调研.md)；
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
measured_fp16_tflops
cp_async_supported               # compile probe
shared_mem_per_block_max_bytes
bf16_tensor_core_available       # expected: false
```

每一项都要有**测量方法**与**不确定性**（重复次数 / 离散度），否则视为未测。
