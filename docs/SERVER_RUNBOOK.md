# SERVER_RUNBOOK — 服务器到手后的操作序列

> 目标：**从裸机到第一批实测数据，尽可能一条命令接一条命令地走完。**
> 环境原则见 [ENVIRONMENT.md](../ENVIRONMENT.md)，硬件规格见 [SPECS.md](SPECS.md)，
> 指标口径见 [METRICS.md](METRICS.md)。

**在开机前**：本仓库在 Mac 上应当已经 `pytest -m "not gpu"` 全绿。
**不要在未验证过的状态下按小时烧钱。**

---

## 0. 下单前的检查清单（在网页上做，不花钱）

- [ ] **GPU 确切型号**：`5090` 优先；若只有 40 系，`4090` 24GB 也可用（见 [SPECS.md](SPECS.md) §2.2）
- [ ] 避开 **L40S**（带宽 864 GB/s，低于 V100 规格）与 **RTX A6000**（Ampere，无 FP8）
- [ ] **镜像选「系统镜像 → `Ubuntu-nvidia` 22.04」**，不要选预装 PyTorch/vLLM 的平台镜像
      （平台镜像是容器类型、不能再装 Docker，且 pin 了不透明的 torch 版本）
- [ ] **计费方式选「按量」**，不要「抢占式」（抢占会中断长实验）
- [ ] **确认关机保留系统盘**（否则每次开机重付 4–6 h 配置税）
- [ ] 系统盘 ≥ 100 GB（OLMoE 12.89 GiB + torch/CUDA ~15 GB + 余量）

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
