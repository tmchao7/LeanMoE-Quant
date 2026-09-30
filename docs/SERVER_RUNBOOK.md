# SERVER_RUNBOOK — 服务器到手后的操作序列

> 目标：**从裸机到第一批实测数据，尽可能一条命令接一条命令地走完。**
> 环境原则见 [ENVIRONMENT.md](../ENVIRONMENT.md)，硬件规格见 [SPECS.md](SPECS.md)，
> 指标口径见 [METRICS.md](METRICS.md)。

**在开机前**：本仓库在 Mac 上应当已经 `pytest -m "not gpu"` 全绿。
**不要在未验证过的状态下按小时烧钱。**

---

## 0. 下单前的检查清单（在网页上做，不花钱）

**通用**

- [ ] **GPU 确切型号**：`5090` 优先；若只有 40 系，`4090` 24GB 也可用（见 [SPECS.md](SPECS.md) §2.2）
- [ ] 避开 **L40S**（带宽 864 GB/s，低于 V100 规格）与 **RTX A6000**（Ampere，无 FP8）
- [ ] **计费方式选「按量」**，不要「抢占式」/包周（抢占会中断长实验）
- [ ] **余额够**：一次性配置就要 4–6 h。按 ¥3/h 算，**先充 ¥50 再开机**。
      余额耗尽通常会被强制关机，可能连装到一半的环境都保不住
- [ ] **确认有空闲卡**（列表里的「空闲GPU」列不能是 `0 / N`）
- [ ] GPU 数量选 **1**（单卡项目）
- [ ] 确认实例**关机后磁盘保留**（否则每次开机重付配置税）

**平台差异：基础镜像怎么选**

| 平台 | 选什么 | 理由 |
|---|---|---|
| **AutoDL** | **基础镜像 → PyTorch**（如 `2.8.0 / 3.12(ubuntu22.04) / 12.8`） | ① AutoDL 已把 驱动+CUDA+torch+sm_120 这组测过，直接消掉本项目 1 号风险；② **自带 CUDA toolkit → 有 `nvcc`**，这是 `cpp_extension` 探针与自写 kernel 的前提；③ torch 2.8.0 已 ≥ 2.7，满足 sm_120 硬下限。<br>**不要选 Miniconda**——它不带 torch 也不带 toolkit，等于把上面三条全部自己重做一遍 |
| 优云智算等有系统镜像的平台 | **系统镜像 → `Ubuntu-nvidia` 22.04** | 平台预装的 PyTorch/vLLM 镜像常是容器类型（不能再装 Docker）且 pin 不透明；系统镜像给一个正常 Ubuntu + 驱动，其余交给 `uv` |

> **Miniconda 与 Miniforge 的区别**：只是默认 channel 不同（Miniconda 用 Anaconda `defaults`，
> Miniforge 用 `conda-forge`）。**对本项目没有影响**，不要在这个选择上花时间。

**AutoDL 特有的两个坑**

1. **系统盘只有 30 GB，数据盘在 `/root/autodl-tmp`。**
   OLMoE 12.89 GiB + torch/CUDA ≈ 28 GB，**几乎撑爆系统盘**。开机后第一件事：

   ```bash
   export HF_HOME=/root/autodl-tmp/hf          # 模型放数据盘
   export VENV_DIR=/root/autodl-tmp/.venv      # venv 也放数据盘
   ```

   免费 50 GB 数据盘够用（OLMoE 13.84 + Qwen 0.5B 1 + venv ≈ 30 GB），**先不用扩容**。

2. **镜像的 CUDA 版本决定 torch 索引，不是驱动、也不是镜像宣传名。**
   例如镜像标 `CUDA 12.8` → 用 `cu128`。原因：Gate-0 的 `cpp_extension` 探针与后续自写
   kernel 都用**镜像里的 `nvcc`** 编译，torch 的 CUDA 构建必须与之一致。
   各索引可用版本不同（实测）：`cu126→2.14.0`、**`cu128→2.11.0`**、`cu129→2.13.0`、
   `cu130→2.14.0`、`cu132→2.14.0`。

   ```bash
   # 想跳过 torch 安装、直接用镜像自带的 torch（最快到第一批数据）
   bash scripts/bootstrap.sh --no-torch
   # 注意：torch 版本会与 Mac 上的不同 —— 记进 results/env-<date>.md
   ```


---

## 1. 一次性配置（约 4–6 h，只付一次）

```bash
# 1.1 拿到仓库
git clone git@github.com:tmchao7/LeanMoE-Quant.git
cd LeanMoE-Quant

# 1.2 代理与镜像（cp 后编辑；.env 已 gitignored，绝不提交）
cp configs/env.env.example .env
$EDITOR .env
bash scripts/preflight.sh        # 看直连/代理各自哪些站点通，按建议配 .env

# 1.3 环境（幂等；会在最后做 sm_120 死亡测试，失败即停）
bash scripts/bootstrap.sh
```

**如果 `.env` 里的代理需要生效到子进程**，用 `source` 而不是直接执行：

```bash
source scripts/proxy.sh on
```

**bootstrap 失败时的常见原因**：

| 报错 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available` | torch 对 sm_120 无 kernel（torch < 2.7） | 确认 pin 的是 `torch==2.14.0`；改 `TORCH_CUDA_INDEX` |
| `torch.cuda.is_available() == False` | 驱动/容器未透传 GPU | `nvidia-smi` 确认；检查镜像类型 |
| 装包超时 | 直连 PyPI 被限速 | 用 `.env` 里的阿里云/华为云镜像 |

---

## 2. Gate-0（约 1 h，¥1–3）—— **必须先做**

```bash
bash scripts/preflight.sh
.venv/bin/python -m tools.probe_machine --json results/machine_probe.json
bash scripts/run_server_gate0.sh          # 上面两步 + 裁决，一次跑完
```

Gate-0 产出并回答：

1. **arch 是 sm_120 / sm_89 / sm_86 / sm_70** → 决定挂载哪些实验臂；
2. **实测可达带宽 + 只读带宽** → 之后所有理论下限都引用它，**禁用规格值**；
3. **Triton 能否跑通 `tl.dot`** → 失败会让 Phase 2 全盘返工；
4. **`ncu` 是否可用** → kernel 级证据是本项目最缺的加分项；
5. **`cpp_extension` 能否编译** → 决定自写 CUDA kernel 这条路走不走得通。

**回填（必须做，否则数字没有出处）**：把稳定事实写进 [SPECS.md](SPECS.md) §1，
并把 `results/machine_probe.json` 中**稳定**的部分摘录进提交（`results/` 本身是 gitignored）。

---

## 3. 模型获取 + 基线

```bash
bash scripts/fetch_models.sh              # 自动校验 OLMoE 字节数 = 13,838,721,960
.venv/bin/python -m pytest -m "not gpu"   # 在服务器上复跑 Mac 上已通过的测试
```

然后跑第一个真实实验（`EXP-p0-02`）：fp16 基线 + 与 HF 参考的
**KL 散度 / top-1 一致率**。

> ⚠️ **不要用「逐 token 严格相等」当门槛**：bf16 greedy decode 在 logit 平局处是
> **非确定性的**，会随机失败并让你追一个不存在的 bug。用 KL + top-1 一致率。

---

## 3.1 EXP-p0-03：CUDA Graph feasibility gate（必须先做，再碰量化）

`codex/p0-03-cudagraph` 已在 Mac 完成本地测试，但 **CUDA Graph capture/replay 只能在
RTX 5090 上验证**。这一步不是量化实验，也不允许用 Mac 产生任何结果。

先确认服务器 checkout 已包含提交 `124e0c4` 或更新版本，然后只运行 `ctx=1024`：

```bash
cd /root/LeanMoE-Quant
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf
export PATH=/root/miniconda3/bin:/usr/local/cuda/bin:$PATH

# 1) 离线回归；必须先绿
/root/miniconda3/bin/python -m pytest -m "not gpu" -q

# 2) GPU toy graph smoke；失败就停止，不加载 13 GB 模型
/root/miniconda3/bin/python -m pytest tests/test_cudagraph.py -m gpu -q

# 3) 真实 OLMoE StaticCache + 单 token graph feasibility gate
/root/miniconda3/bin/python -m benchmarks.runner \
  --exp EXP-p0-03-baseline-cudagraph \
  --model /root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master \
  --ctx 1024 \
  --batch 1 \
  --max-new-tokens 32 \
  --warmup 5 \
  --timed 20 \
  --dtype float16 \
  --force
```

合法结果只有两类：

- `status=done`：capture/replay 成功且 KL/top-1 gate 通过，才可读取 latency；
- `status=blocked` 或 `status=failed`：性能字段必须为 `null`，记录具体阶段/错误，
  然后进入固定 metadata 的 MoE routing 设计。

**不要**在 p0-03 失败后把 execution 改成 eager 继续写入同一个结果，也不要立即跑
W8/W4。先保留失败证据；失败本身就是判断 HF 动态 routing 是否阻塞 CUDA Graph 的实验结果。

---

## 4. 在服务器上跑 agent（opencode）

**★ 两步走，严格区分凭据与配置：**

1. **仓库级 `opencode.json`（可提交）** —— 只含 `$schema`、`permission` 白名单、模型名。
   **绝不含 API key。** 形态参照 `MegaKernel-vLLM/opencode.json`。
2. **key 走环境变量或服务器本地** `~/.local/share/opencode/auth.json`，**不进仓库**。

```bash
# 安装（需要 npm / GitHub 可达 → 这一步通常要走代理）
curl -fsSL https://opencode.ai/install | bash     # 或者 npm i -g opencode-ai
export PATH="$HOME/.opencode/bin:$PATH"
```

**双写纪律**（见 [AGENTS.md](../AGENTS.md) "Remotes and sync"）：

- agent 只在**特性分支**上工作并推送；Mac 只 `git pull --no-rebase`；
- **绝不 rebase、绝不 force-push**；
- 每次迭代必须留下三样：`results/<EXP-ID>/result.json`、
  `experiments/<EXP-ID>.md`（按 [TEMPLATE](../experiments/TEMPLATE.md) 填）、一个提交；
- **agent 不得自行放宽正确性门槛**：KL/top-1 不过就是 `FAIL`。

---

## 5. 关机前

- [ ] `results/` 里的原始数据已备份（它不在版本库里）
- [ ] 稳定结论已摘录进提交（数字 + 出处 + 适用区间）
- [ ] `nvidia-smi` 确认没有遗留进程占显存
- [ ] 若平台按小时计费：**确认实例已停机**（保留磁盘）

---

## 6. 故障速查

| 症状 | 先查什么 |
|---|---|
| 数字比预期慢很多 | **host/launch 占比**（`benchmarks/hostgap.py`）；用 nsys 交叉验证。batch=1 的收益可能被 host 开销截断 |
| 达成率 > 100% | 字节账算错了，或权重命中 L2 —— **必须解释，不能当成绩报** |
| Triton kernel 编不过 | 该 arch 的 Triton 支持范围；退回纯 PyTorch/`torch.compile` 路径并在记录里标注 |
| 结果不可复现 | clocks 是否降频、warmup 是否足够、是否有其它进程占 GPU |
| OOM | 用显存**分列**（weights/KV/acts/workspace）定位是哪一列爆的 |
