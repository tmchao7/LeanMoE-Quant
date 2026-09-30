# ENVIRONMENT — 机器、环境、网络与模型获取

> 本文是**环境层**的权威说明。计划见 [PLAN.md](PLAN.md)，记录格式见
> [EXPERIMENTS.md](EXPERIMENTS.md)，服务器上机步骤见 [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)。

**规则按机器分列。** 不要把一台机器的约束套到另一台上——租的机器我们全权掌控，
学校机器是共享的，Mac 只是文档与 CPU 逻辑验证。

---

## 1. 机器

### 1.1 主开发机：租用 RTX 5090 32 GB（sm_120）

| 项 | 值 |
|---|---|
| GPU | RTX 5090，32 GB，**sm_120（Blackwell 消费级）** |
| 计费 | 按量（**不用抢占式**——长实验会被中断） |
| 磁盘 | 系统盘 100 GB（**必须确认关机能保留**，否则每次开机重付配置税） |
| 成本 | ≈ ¥3/h |

**为什么是它**：本项目瓶颈是**带宽 + 架构特性**，不是显存（OLMoE fp16 只有 13.84 GB）。
5090 是本项目能拿到的、唯一同时具备 `cp.async` + TMA + FP8 + **MX block-scale（MXFP4/NVFP8）**
的档位，且带宽 1792 GB/s 约为 V100 的 2 倍。

**特性矩阵（从本地 LLVM NVPTX 谓词核实，见 [docs/SPECS.md](docs/SPECS.md) §2.1）**：
`cp.async` ✅ / TMA ✅ / FP8 ✅ / **MXFP4 block-scale ✅** / bf16 ✅ / **`wgmma` ❌**。
`wgmma` 门在 `SM90a`，**40 系和 50 系消费卡都没有**。

### 1.2 第二数据点（免费）：学校 V100 32 GB（sm_70）

| 项 | 值 |
|---|---|
| GPU | Tesla V100，32 GB，sm_70 |
| 物理卡数 | 15（**本项目只用 1 卡**） |
| 成本 | 边际成本 ≈ 0 |

**只用作第二个带宽数据点**（900 vs 1792 GB/s），回答"量化的收益如何随带宽变化"。
它缺 `cp.async`/TMA/FP8`/FP4`/bf16 张量核，**装不下现代 kernel 臂**。

**硬约束（写进每份报告）**：无 BF16（bf16 退化为 fp32 模拟）；无 `cp.async`/TMA/`wgmma`；
**共享机器**——不要动 CUDA/驱动/系统 Python，不要长期占显存，跑完释放，实验前先看一眼
有没有别人在跑。

### 1.3 Mac（文档 + CPU 逻辑验证）

| 项 | 值 |
|---|---|
| 硬件 | Apple Silicon arm64，**32 GiB RAM**，10 核 |
| OS | macOS 27.0.1 |
| 磁盘 | **725 GiB 空闲** |
| 默认 `python3` | **3.9.6（Xcode 自带）** —— 违反 `requires-python>=3.10`，**永远不要直接用** |

**允许**：跑 CPU 功能测试、单测、`--dry-run`、量化数学对照。
**禁止**：**任何 Mac 产生的数字都不得进入 `results/` 或任何结论**（无 NVIDIA GPU）。

---

## 2. 环境搭建

### 2.1 目标版本（两端对齐）

| 包 | Mac | 服务器 | 说明 |
|---|---|---|---|
| Python | **3.12** | **3.12** | 由 `.python-version` 固定 |
| torch | **2.14.0（CPU）** | **2.14.0 + CUDA** | 两端同大版本，避免 API 漂移 |
| CUDA 索引 | — | **按实际驱动定**（见下） | `cu126/cu128/cu129/cu130/cu132` 均有 2.14.0 轮子 |
| transformers | **4.57.6** | **4.57.6** | 最后一个 4.x；OLMoE 需 ≥4.45；避开 5.x 破坏性变更 |
| triton | 不装 | **交给 torch 依赖解析** | 自带 ptxas，**无需系统 CUDA toolkit** |

**选 CUDA 索引的铁律**：看 `nvidia-smi` 的驱动版本，**不要看镜像的宣传名**。
驱动向后兼容（新驱动能跑旧 CUDA 运行时），反之不行。

> ⚠️ **sm_120 的经典陷阱**：torch < 2.7 的轮子里**没有 sm_120 kernel**，会报
> `CUDA error: no kernel image is available for execution on the device`。
> 2.14.0 不受影响。装完**必须立刻做死亡测试**：
> ```bash
> uv run python -c 'import torch; x=torch.ones(2, device="cuda"); print(x + x)'
> ```

`configs/constraints.txt` 在**首次验证通过后**冻结精确版本。

### 2.2 ⚠️ 本机环境现状（2026-09 实测，含一处此前误记的更正）

| | Mac | 服务器 |
|---|---|---|
| Python | 默认 `python3` = **3.9.6**（Xcode）；`uv` 已备 **3.10.20 / 3.12.13**；miniforge3 有 **3.13.13** | 待测 |
| `uv` | ✅ **已装，0.11.25**，在 `~/.local/bin/uv` 且在 PATH | 待测 |
| `conda` | ✅ 有（`~/miniforge3`） | 待测 |
| torch | ✅ **已装 CPU 2.14.0**（`.venv-mac/`）+ transformers 4.57.6 | 待测 |

> **更正**：本文档此前写着"Mac 只有 Python 3.9.6，没有 uv、没有 conda"——**那是错的**，
> 第一次检查跑在了 PATH 不同的 shell 里且没有复核。真实情况如上表。
> 唯一的小坑是 `python3` 默认解析到 Xcode 3.9.6，用 `.python-version` + `uv run` 即可绕开，
> **不需要安装任何东西**。

### 2.3 网络、镜像与代理

**核心策略：能用国内镜像就不用代理。** 代理只留给 GitHub 与 npm。

| 用途 | 走什么 | 是否需要代理 |
|---|---|---|
| 模型权重 | `HF_ENDPOINT=https://hf-mirror.com`（已验证 200；备用 modelscope） | ❌ |
| Python 包 | 阿里云 / 清华 / 华为云 PyPI 镜像（均验证 200） | ❌ |
| **torch** | `--index-url https://download.pytorch.org/whl/<cuXXX>` | 视平台而定 |
| GitHub / npm（装 opencode） | 可能需要代理 | ✅ |
| `huggingface.co` | **不可达**（DNS/网络） | — |

**注意**：`uv pip install --index-url X` 会**替换** PyPI。所以**先装 torch，再装其余包**，
或对其余包显式加 `-i <国内镜像>`。

```bash
scripts/preflight.sh     # 直连 vs 代理分别探测，输出路由建议 → results/preflight.json
scripts/proxy.sh on|off|status|test
```

**凭据只来自 `.env`（gitignored）**，模板见 `configs/env.env.example`。
**绝不提交** API key、代理密码、token。

### 2.4 ⚠️ 编译能力（上机后必须查，可能改变技术方案）

本项目核心交付是**自写 kernel**，而 kernel 需要能编译。

- 租的机器我们全权掌控，**可以装 toolkit**——这条约束只对学校的共享 V100 成立；
- 但即便如此，仍要查清：`nvcc`/`ptxas` 版本、与 torch CUDA 构建是否匹配、
  `torch.utils.cpp_extension` 能否真编译通过一个小程序；
- **Triton 自带 ptxas，不需要系统 toolkit**——这是把 Triton 排在 CUDA 前面的又一个理由。

---

## 3. 模型获取

`configs/models.yaml` 是唯一来源。

**走国内镜像，不访问 `huggingface.co`**：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download allenai/OLMoE-1B-7B-0924      # 3 shard，13,838,721,960 B
hf download Qwen/Qwen2.5-0.5B             # dev / --dry-run / Mac CPU 冒烟
```

| 角色 | 模型 | 精度 | 权重（参考） | 用途 |
|---|---|---|---|---|
| dev | `Qwen/Qwen2.5-0.5B`、`1.5B` | fp16 | 1.0 / 3.0 GB | kernel 与量化调试；**Mac CPU 冒烟** |
| **primary (MoE)** | **`allenai/OLMoE-1B-7B-0924`** | fp16 | **13.84 GB** | **MoE 主线实验** |
| dense 对照 | `Qwen/Qwen3-4B`、`8B` | fp16 | 8.0 / 16.4 GB | 同规模 dense vs MoE |
| stretch（需 48GB 卡） | `Qwen/Qwen3-30B-A3B` | INT4 | 15.3 GB | scale 单点 |

> ⚠️ **`-0924` 后缀必需**：`allenai/OLMoE-1B-7B` **不存在**（hf-mirror 对不存在的仓库返回
> 401，与"无权限"同码，极易误判）。已核实的正确仓库是 `allenai/OLMoE-1B-7B-0924`。
> 用 **base**，不用 `-Instruct`（主门槛是 KL / wikitext-2 PPL）。

**下载纪律**

- 下载前预检磁盘；缓存路径由 `HF_HOME` 指定，**不要硬编码进代码**；
- **下载后必须校验字节数**：OLMoE 应为 `13,838,721,960`（`scripts/fetch_models.sh` 自动做）；
- 已存在的路径写进 §5 本机附录，而不是写死。

---

## 4. 上机清单

完整命令序列见 [docs/SERVER_RUNBOOK.md](docs/SERVER_RUNBOOK.md)。**Gate-0** 是最先做的一步。

**Gate-0 A. 设备事实（写 `results/machine_probe.json`）**

- [ ] GPU 确切型号（**从 `nvidia-smi` 取，不写系列名**）/ compute capability / 显存
- [ ] **实测可达带宽**（≥3 次重复，报中位数与离散度）——所有理论下限都引用它
- [ ] **只读模式带宽**（reduction）——decode 是读主导，纯 copy 会低估上限
- [ ] 架构特性：`cp.async` / TMA / FP8 / MX block-scale / bf16 / 共享内存上限
- [ ] **Triton 能否跑通一个 `tl.dot`**（⚠️ 最大前置风险）
- [ ] **`ncu` 是否可用**（推理优化岗硬要求；kernel 级证据是本项目目前缺的一块）
- [ ] **编译能力**：`nvcc`/`ptxas` 版本与 torch CUDA 构建匹配性 + 真编译一次
- [ ] **host CPU 型号**（host/launch 开销依赖它）+ 运行时钟（查降频）
- [ ] 磁盘剩余空间

**Gate-0 B. 模型与参考基线**

- [ ] OLMoE-1B-7B-0924 与 Qwen2.5-0.5B 能在目标精度下加载并前向
- [ ] 与 HF 参考的**分布一致性**（KL / top-1 一致率 —— 见下方更正）
- [ ] checkpoint 的 bf16 → fp16 转换精度落差记录（PPL / KL）

> ⚠️ **更正**：此前这里写的是"与 HF 参考**逐 token 对齐**（greedy，同输入同输出）"。
> **在 bf16 下这个门槛不成立**——bf16 greedy decode 在 logit 平局处是**非确定性**的，
> 会随机失败并让你追一个不存在的 bug。正确门槛是 **KL 散度 + top-1 一致率**
> （见 [docs/METRICS.md](docs/METRICS.md) §1），不要用严格逐 token 相等。

**Gate-0 C. 计时与显存口径**

- [ ] 计时工具：warmup ≥ 5、计次 ≥ 20、报 p50/p99 + 离散度
- [ ] 显存分列：权重 / KV / 激活 / workspace，用 allocator 峰值
- [ ] **host/kernel 时间拆分**（`benchmarks/hostgap.py`，用 nsys 交叉验证）
- [ ] `--dry-run` 在 Mac 上跑通逻辑

**Gate-0 D. 门槛**

- [ ] A/B/C 全部落盘后，才允许进入 Phase 1（量化功能链路）

---

## 5. 本机附录（随使用更新）

```text
# 模型缓存路径
# 服务器主机名 / SSH 别名 / SSH 端口
# 已下载模型清单与磁盘占用

# Mac 验证环境（2026-09 建立）
#   .venv-mac/  →  python 3.12.13, torch 2.14.0+cpu, transformers 4.57.6
#   用途：CPU 功能测试与量化数学对照；不产出任何可报告的数字
```

（待上机后填写。）
