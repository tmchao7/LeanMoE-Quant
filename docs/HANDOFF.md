# HANDOFF — LeanMoE-Quant 交接文档

> 写给**下一个 agent**（Codex）或接手的人。读完这一份就能直接干活，不需要重跑探索。
>
> - **写于**: 2026-09-30
> - **仓库 HEAD**: `4b6fe0a`（`main`，已推 GitHub，本地/服务器均干净）
> - **阶段**: Phase 0 已完成。**基线与瓶颈归因已出结果，量化尚未开始。**
> - **计费**: 服务器按量 ¥2.88/h，**现在仍在计费中**。

---

## 0. 三分钟版本

**做了什么**：在 AutoDL 租的 RTX 5090 上完成了 Gate-0（环境验证）与 fp16 基线。
发现了一个**会决定整个项目走向**的事实。

**最重要的结论**：

```
roofline 理论下限   1.598 ms/token
实测 wall          100.10 ms/token   (慢 62.6 倍)
但 GPU 真正在算只有  10.49 ms        (10.5%)
host / launch      89.62 ms          (89.5%)
每 token kernel 数   8,277 个
```

瓶颈**不是带宽，是 launch 开销**。HF 的 `OlmoeSparseMoeBlock` 对全部 64 个专家做
Python 循环，而每 token 只激活 8 个。

**后果**：W4A16 有 3.09 倍字节缩减，但单独做端到端只有 **1.011x**（+1.1%）。
**必须先消 launch 开销，量化才测得出来。**

---

## 1. 服务器接入

### SSH

```bash
ssh 5090
```

别名在 **Mac 侧**的 `~/.ssh/config`：

```
Host 5090
  HostName connect.westd.seetacloud.com
  Port 23960
  User root
  IdentityFile ~/.ssh/id_ed25519
```

**免密已生效。** 这个别名是在 Mac 上配的，服务器上**没有**它
（服务器上配的是 `github.com` 的 deploy key，见第 4 节）。

### 容器与硬件（实测，非规格）

| 项 | 值 |
|---|---|
| hostname | `autodl-container-8me3mxapw7-a53d4911` |
| OS | Ubuntu, Linux 5.15.0-78-generic, x86_64 |
| GPU | **NVIDIA GeForce RTX 5090** |
| compute capability | **sm_120**（170 SMs） |
| 显存 | 32607 MiB = **31.36 GiB 可用** |
| driver | **580.95.05** |
| **实测带宽（只读，roofline 用这个）** | **1559.2 GB/s** |
| 实测带宽（copy） | 1526.7 GB/s |
| 主机 CPU | Intel Xeon Platinum 8470Q |
| **容器 cgroup 配额** | **25 vCPU / 92 GiB RAM**（宿主 208 核是共享的，别信 `nproc`） |

> ⚠️ **绝不能用规格值 1792 GB/s 算 roofline**（项目硬规则 4）。
> 实测 1559.2 GB/s 才是分母。两者的比值 0.87 是合理的可达比例。

### 磁盘（AutoDL 的坑）

| 挂载点 | 总量 | 可用 | 用途 |
|---|---|---|---|
| `/` | 30 GB | **25 GB** | 系统盘，**很小** |
| `/root/autodl-tmp` | 50 GB | **38 GB** | **数据盘，模型放这** |
| `/root/autodl-fs` | 200 GB | 200 GB | 跨实例共享（软链到 `/autodl-fs/data`） |

**必须设**：`export HF_HOME=/root/autodl-tmp/hf`。否则 OLMoE 的 12.89 GiB
会把系统盘塞满。

---

## 2. 环境（已配好，不用重装）

```bash
# 解释器 —— 复用了镜像自带的 conda python，不是新建 venv
/root/miniconda3/bin/python3

python       3.12.3
torch        2.8.0+cu128        # 镜像自带，已由平台验证能在 sm_120 上跑
cuda build   12.8
transformers 4.57.6             # 最后一个 4.x
triton       3.4.0              # 自带 ptxas，不需要系统 toolkit
numpy        2.3.2
modelscope   1.40.1
ninja        1.13.2             # cpp_extension 的硬依赖
```

### 已实测的工具可用性

| 工具 | 状态 | 备注 |
|---|---|---|
| `nvcc` | ✅ 12.8.93 | 在 `/usr/local/cuda/bin`，**不在 PATH 上** |
| `ptxas` | ✅ | 同上 |
| `cpp_extension` 编译 | ✅ | 需要 ninja；探针会自动设 `CUDA_HOME` |
| Triton | ✅ | 最小 kernel 跑通 |
| `ncu` | ✅ | Nsight Compute 2025.1.1，在 `/usr/local/cuda/bin/ncu` |
| `nsys` | ❌ **未安装** | 所以 launch 归因改用 `torch.profiler` |

> ⚠️ **`nvcc` 和 `ncu` 不在 PATH 上**。裸 `shutil.which("nvcc")` 会误报"没有"。
> 探针已经修好了（会查 `/usr/local/cuda/bin`），但你自己写脚本时要记得。

### 环境已经装好了，**不要重装**

```bash
# 如果非要重建，用这个（已验证）
cd /root/LeanMoE-Quant
export HF_HOME=/root/autodl-tmp/hf
bash scripts/bootstrap.sh --no-torch --python=/root/miniconda3/bin/python
```

`--no-torch` 很关键：它会把 `torch` 从依赖列表里剔掉，避免覆盖镜像自带的
2.8.0+cu128。**重装 torch 是负收益**（镜像的版本已由平台验证过与 nvcc 匹配）。

---

## 3. 代码与模型位置

```bash
# 仓库（git 干净，HEAD = 4b6fe0a）
/root/LeanMoE-Quant

# 模型（ModelScope 下载，字节数已校验 = 13,838,721,960）
/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

# 机器探针结果（JSON，gitignored）
/root/LeanMoE-Quant/results/machine_probe.json

# 已完成的实验结果（gitignored）
/root/LeanMoE-Quant/results/EXP-p0-02-baseline-fp16__ctx=1024/
/root/LeanMoE-Quant/results/EXP-p0-02-baseline-fp16__ctx=4096/
```

`HF_HOME` 设为 `/root/autodl-tmp/hf`（目前是空的，之前 hf-mirror 的半成品已清理）。

### ⚠️ 服务器上还有另一个项目，**不要碰**

```bash
/root/MegaKernel-vLLM          # 另一个项目（并行跑的）
/root/models/Qwen3-0.6B        # 它的模型，1.5 GB
```

**我的工作全部在 `/root/autodl-tmp/` 与 `/root/LeanMoE-Quant/` 里。**
不要删除、不要移动 `/root/models` 和 `/root/MegaKernel-vLLM`。

---

## 4. 命令行速查

**每一个命令前都要设这两个环境变量**（建议写进 `~/.bashrc` 或每次 export）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
```

### 同步代码

```bash
# 服务器自己拉（deploy key 已配好，可读可写）
cd /root/LeanMoE-Quant && git fetch origin && git reset --hard origin/main

# 从 Mac 推过去（如果改了没提交）
rsync -azc --exclude '.venv-mac/' --exclude '__pycache__/' --exclude 'results/' \
  ./ 5090:/root/LeanMoE-Quant/
```

服务器的 deploy key 在 `~/.ssh/id_ed25519_github`，已加到 GitHub 的
`tmchao7/LeanMoE-Quant` 仓库（**只对这一个仓库生效**，不是账号级 key）。
已验证：`ssh -T git@github.com` → `Hi tmchao7/LeanMoE-Quant!`，
且 `git ls-remote origin` 可用。服务器上不方便再创建其它仓库的 key。

服务器上 git 需要（已配过，重装系统后要重配）：

```bash
git config --global --add safe.directory /root/LeanMoE-Quant
```

### 跑探针（Gate-0）

```bash
cd /root/LeanMoE-Quant
/root/miniconda3/bin/python -m tools.probe_machine --json results/machine_probe.json
```

### 跑基线

```bash
cd /root/LeanMoE-Quant
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/root/autodl-tmp/hf
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

python3 -m benchmarks.runner --exp EXP-p0-02-baseline-fp16 --ctx 1024 \
  --max-new-tokens 32 --warmup 5 --timed 20 --model "$M"
```

- `--dry-run` 只看要跑什么，不执行
- `--force` 覆盖已有结果（不加则跳过已完成的）
- 单次约 **1~2 分钟**（加载 5 s + warmup + 20 步 × 100 ms + profiler + hostgap）

### 跑测试（离线，不需要 GPU）

```bash
cd /root/LeanMoE-Quant
/root/miniconda3/bin/python -m pytest -m "not gpu" -q      # 140 passed
```

### 下模型（换源后）

```bash
bash scripts/fetch_models.sh                    # auto：优先 ModelScope
bash scripts/fetch_models.sh --source ms        # 强制 ModelScope（快，~15 MiB/s）
bash scripts/fetch_models.sh --source hf        # 强制 hf-mirror（慢，~2.2 MiB/s）
bash scripts/fetch_models.sh --check            # 只校验，不下载
```

---

## 5. 关键实测结果（可直接引用）

### EXP-p0-02 fp16 基线

| 指标 | ctx=1024 | ctx=4096 |
|---|---|---|
| TTFT p50 / p99 (ms) | 111.101 / 159.959 | 171.397 / 183.601 |
| **per-token p50 / p99 (ms)** | **100.102 / 105.657** | **108.771 / 114.935** |
| tok/s @ batch=1 | 9.99 | 9.19 |
| `[W]` bytes | 2,357,723,136 | 2,357,723,136 |
| `[T]` bytes | 2,491,940,864 | 2,894,594,048 |
| floor_W / floor_T (ms) | 1.5121 / 1.5982 | 1.5121 / 1.8565 |
| **量化达成率** | **1.5%** | **1.4%** |
| **带宽达成率** | **1.6%** | **1.7%** |
| VRAM peak (GiB) | 13.3417 | 14.6811 |
| **GPU busy (ms)** | **10.49** | **10.94** |
| **host share** | **89.9%** | **89.7%** |
| **kernels / token** | **8,277** | **8,277** |
| slack（wall / floor_T） | 62.6x | 58.6x |
| bucket: routing_pack | 44.8% | 42.6% |
| bucket: gemm | 26.7% | 27.8% |
| bucket: other | 25.1% | 26.3% |
| bucket: memcpy | 3.4% | 3.2% |

### `[W]` 分项（fp16）

```
attention          536,870,912   22.8%
router               4,194,304    0.2%
experts (top-k)   1,610,612,736  68.3%   <- 量化的主要目标
lm_head             206,045,184    8.7%
embedding_row             4,096    0.0%
─────────────────────────────────
总计              2,357,723,136
```

### 瓶颈归因（GPU 时间去向）

```
routing_pack    44.8%     4.6 ms/step    <- torch.where + 逐专家循环
gemm            27.0%     2.8 ms/step
other           25.2%     2.6 ms/step
memcpy           3.4%     0.4 ms/step

kernel 明细（单步）: gemvx 449 次 | cub::DeviceSelectSweep 1024 次
                    cub::DeviceReduce 1024 次 | elementwise 1024 次
                    Memcpy DtoH 1024 次 | flash attention 仅 16 次
每 kernel 平均: GPU 1.26 µs  vs  wall 16.95 µs
```

### 单个 MoE 块单测

```
per call            8.996 ms
experts available   64
experts activated   8
bytes if all        805,306,368
bytes if top-k      100,663,296      (比值 8.0x)
```

### ★ 截断计算（本项目的核心论证）

```python
T_bytes(fp16)  = 1.512 ms
T_bytes(W4A16) = 0.489 ms      # 3.09x 更少字节
host           = 89.62 ms      # 量化**不会**减少这一项

只做量化       : (1.512+89.62)/(0.489+89.62) = 1.011x   # 只快 1.1%
只做 CUDA Graph: (1.512+10.49)/(0.489+10.49) = 1.093x   # 快 9.3%
两者都做       : 3.09x                                   # 真实上限
```

---

## 6. 代码结构

```
benchmarks/
  bytes.py       # [W]/[T] 字节预算（纯 Python，无 torch）
  timing.py      # percentile / summarize / measure
  hostgap.py     # GPU busy vs host 拆分 + 截断预测
  vram.py        # allocator 峰值分项（weights/KV/act/workspace）
  schema.py      # result.json 的 schema 与校验器
  harness.py     # 矩阵展开 / arch 裁剪 / 断点续跑（纯编排）
  runner.py      # ★ 真正执行实验并填 result.json
  report.py      # 汇总报告
  dryrun.py      # 离线端到端（小随机模型，reportable=false）

src/lemoq/
  quant/         # numpy_oracle / rtn / pack / api
  kernels/       # reference / autotune（真 kernel 还没写）

tools/
  probe_machine.py   # 机器探针（带宽/架构/工具链）
  check_docs.py      # 文档死链与表格检查

scripts/
  bootstrap.sh       # 环境搭建（幂等）
  fetch_models.sh    # 换源下载 + 字节校验
  preflight.sh       # 网络预检
  proxy.sh           # 代理开关
  run_server_gate0.sh # Gate-0 一键

configs/
  models.yaml        # 模型单一真相源
  experiments.yaml   # 声明式实验矩阵（含 hypothesis/refutation_signal）
  constraints.txt    # 依赖 pin（尚未在 GPU 上冻结）
  env.env.example    # 代理/镜像模板（占位符）

experiments/
  TEMPLATE.md                    # 实验记录模板（必须用）
  EXP-p0-02-baseline-fp16.md     # ★ 已完成的基线记录，读它

tests/                # 140 个测试，-m "not gpu" 全绿（Mac 与服务器都能跑）
```

---

## 7. 项目硬规则（会踩错的）

1. **不许编造结果**。未测的写 `planned` / `null`，不写"达成"。`null` = 未测 ≠ `false` = 测了但没有。
2. **正确性门槛先于性能**。KL/top-1 没过 → **只记录失败，不报性能数字**。schema 校验器会拦。
3. **一次只改一个变量**。位宽/group size/fusion 不能同一次run 改。
4. **roofline 必须用实测带宽**（1559.2 GB/s），绝不用规格 1792。
5. **结论必须写上适用卡片范围**（GPU 型号 + 实测带宽 + dtype + batch）。
6. **VRAM 必须分项报**（weights/KV/act/workspace），用 allocator 峰值，不用 `nvidia-smi` 总数。
7. **每个测量写成 `experiments/<EXP-ID>.md`**（从 TEMPLATE 复制），原始 JSON 在 `results/`。

### 还有两条血泪教训

- **`[W]` vs `[T]` 绝不能混用**。`[W]` = 每 token 权重字节（与 ctx 无关）；
  `[T]` = `[W]` + KV 读(ctx)。混用是本项目"达成率虚高"的主要来源。
- **bf16/fp16 的 greedy decode 在 logit 平局处非确定性**。所以正确性门槛用
  **KL + top-1**，不能用"逐 token 完全相等"。

---

## 8. ⚠️ 陷阱清单（都是真踩过的）

### 8.1 带宽探针曾报错 20 倍

第一版报 **76 GB/s**（真值 1559）。成因是漏了"除以迭代数"。
**它看起来完全可信**：数值稳定、CoV 0.1%、三次重复一致。
现在有合理性告警（低于规格 25% 就报警）与回归测试锁住。
**如果你自己写带宽测量，先检查时间是否已换算到单次迭代。**

### 8.2 `host_share` 的残差法会撒谎

`wall - cuda_event` 给出 **0.0%**，真值 **89.5%**。
原因：GPU 在等 host 时，event 窗口会跟着拉长，两者测同一段时间，相减得 0。
已改用 `torch.profiler` 的 `self_device_time` 之和。
**残差法保留在 `hostgap.py` 里做对照，但结论要用 `gpu_busy_ms`。**

### 8.3 工具不在 PATH 上

`nvcc` / `ncu` 在 `/usr/local/cuda/bin`，PATH 里没有。
`shutil.which()` 会误报"没有"。

### 8.4 `--no-torch` 的坑（已修）

`bootstrap.sh --no-torch` 原本仍会从 `constraints.txt` 装 `torch==2.11.0`，
**覆盖**镜像里验证过的 2.8.0。现在会过滤掉。有回归测试。

### 8.5 跨仓库链接

`PLAN.md` 引用 `../MegaKernel-vLLM/...` 是**有意的**。
独立 clone（服务器/CI）里那些目录不存在，所以检查器把它们归为"仓库外"，
不算死链。别把它改回"报错"。

### 8.6 模型仓库名

**必须是 `allenai/OLMoE-1B-7B-0924`**。
`allenai/OLMoE-1B-7B` **不存在** —— hf-mirror 对不存在的仓库返回 401，
与"无权限"同码，极易误判成权限问题。

### 8.7 下载源的选择

| 源 | 实测速度 | 说明 |
|---|---|---|
| **ModelScope** (8 路并行) | **~15 MiB/s** | 默认 |
| hf-mirror（单流） | ~2.2 MiB/s | 退路 |
| huggingface.co + AutoDL 学术加速 | **0.45 MiB/s** | **比 hf-mirror 还慢 5 倍** |

`/etc/network_turbo` 存在，但**别用**。`huggingface.co` 直连不可达（curl 000）。

### 8.8 nproc 是假的

容器 cgroup 限制 **25 vCPU / 92 GiB**。`nproc` 报 208（宿主值）。
写多进程代码时按 25 算。

---

## 9. 协作纪律

- **三个写入者**：这台 5090、学校 V100、Mac。**merge，绝不 rebase，绝不 force-push。**
- 服务器上如果跑 agent，**在特性分支工作**，Mac 只 `git pull --no-rebase`。
- **不能提交**：`results/`、模型权重、`.env`、任何 API key、`__pycache__`。
- 提交信息用 `chore: / docs: / env: / baseline: / quant: / kernel: / bench: / analysis:` 前缀，
  并在正文引用实验 id（如 `EXP-p1-01`）。
- **Mac 上跑的任何数字都不能进 `results/` 或结论**（Mac 没有 NVIDIA GPU）。
  Mac 可以跑 CPU torch 做功能测试，仅此而已。

### 仓库根部的 `opencode.json`

只有 `$schema` + `permission`，**没有 key**。用户的全局
`~/.config/opencode/opencode.json` 里有 4 个真 key，**不要把 key 复制进仓库**。

---

## 10. 下一步该做什么（按优先级）

### 🔴 第一步（必做，且是后面一切的前提）

**给 runner 加 CUDA Graph 执行臂。**

理由：不做这一步，量化实验的端到端数字**无法解释** —— 89.5% 的时间在 host，
量化的影响会被完全淹没（上面算过：只做量化 = 1.011x）。等于烧钱测噪音。

预期：100 ms → 10 ms 量级（消除 8,277 个 kernel 的逐个 launch）。

`configs/experiments.yaml` 里已经有 `execution: eager / cudagraph` 这个因子，
runner 的 `config` 段也已经写入 `execution` 字段，但没有实现。

配套要做的：
- 按 `EXP-p0-02` 的同一配置再跑一遍 cudagraph 臂，做对照；
- 注意 CUDA Graph 要求静态 shape → 需要固定 KV 长度（或用静态 buffer + 拷贝）。

### 🟠 第二步

`EXP-p4-01`：MoE routing/packing 开销。
上面 `moe_block` 的量（8.996 ms，64 个专家全循环 vs 8 个激活）就是它的上界证据。

### 🟡 第三步

`EXP-p1-01`：W8A8 / W8A16 / W4A16 × group{32,64,128} 的精度门槛。
**只谈精度，不谈性能。** fp16 基线（本条）是参考基准。

注意：`docs/METRICS.md` 的 KL 阈值（W8 <0.01、W4 <0.05）是**建议值**，需按实测调整。
门是 KL + top-1，**不能用逐 token 相等**（平局非确定性）。

### 待办

- 可选的 ctx=4096 的 cudagraph 臂。
- `configs/constraints.txt` 里的 pin 还没在 GPU 上冻结 —— 首次稳定运行后可以 `pip freeze`。
- `src/lemoq/kernels/` 下的真 Triton dequant-GEMM kernel **还没写**。
  `tests/test_kernel_parity.py` 里有 gpu-marked 的骨架等着接。

---

## 11. 交接自检清单

新 agent 上手后先跑这几条确认环境没坏：

```bash
ssh 5090

# 1. GPU 还在、还是 sm_120
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

# 2. 代码是最新的
cd /root/LeanMoE-Quant && git log --oneline -1     # 应为 4b6fe0a 或更新
git status --short                                  # 应为空

# 3. 依赖齐全
/root/miniconda3/bin/python -c "import torch,transformers,triton; \
  print(torch.__version__, transformers.__version__, triton.__version__, torch.cuda.is_available())"

# 4. 模型在
ls /root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master/*.safetensors

# 5. 测试绿
/root/miniconda3/bin/python -m pytest -m "not gpu" -q

# 6. 实测带宽（应约 1559 GB/s）
/root/miniconda3/bin/python -m tools.probe_machine --json /tmp/probe.json >/dev/null && \
  /root/miniconda3/bin/python -c "import json;print(json.load(open('/tmp/probe.json'))['bandwidth']['read_only_gbps'],'GB/s')"
```

六条都过 → 直接开始第 10 节的第一步。

---

## 12. 成本

- ¥2.88/h 按量计费，**现在仍在计费**。
- 从接入到基线完成约 **2 小时**，约 **¥6**。
- **关机前记得**：确认 `results/` 已回收（它们是 gitignored，**关机不会自动保留**），
  重要结果要同步回 Mac 或写进 `experiments/*.md`（那个是入 git 的）。

```bash
# 关机前把结果拉回 Mac
rsync -az 5090:/root/LeanMoE-Quant/results/ ./results/
```

**我已经拉过一份到 Mac 的 `results/`**（148 KB），包含两条真基线 + 探针 JSON。
但注意 `results/` 里**大部分是 `planned` 骨架，不是真结果**：

```
EXP-p0-01-machine-probe/                    真（探针）
EXP-p0-02-baseline-fp16__ctx={1024,4096}/   真（本文件第 5 节的数据来源）
EXP-p1-01 / EXP-p2-01 / EXP-p4-01 / EXP-p4b-01 / _dryrun/   status=planned（空骨架）
```

判据是 `result.json` 里的 `status` 字段：`planned` = 没跑过。
**引用数字前先看 `status` 和 `correctness.kl` 是否为 `null`。**
