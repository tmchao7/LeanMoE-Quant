# NCU_EVIDENCE — kernel 级证据（Phase 2 的最后一个交付物）

> **状态（2026-10-08）：部分交付 —— 计数器路径在本机被宿主策略挡住，改用"解析字节 / 隔离计时"
> 口径交出 kernel 级带宽达成率。** 阻断原因、原始报错、以及替代口径的两处必须做的修正都在 §2。
> 这份文件不是实验矩阵里的一格（没有 `EXP-ID`）：它是 `PLAN.md` §5 列的 Phase 2 交付物，
> 工件在 `results/ncu_evidence/`（机器本地，gitignored）。

## 0. 结论（先看这三行）

1. **fp16 的 grouped 专家 GEMV 已经在带宽墙上**：整层一次调用 **71.0%**（cold，见 §2）
   的实测可达带宽；dense 融合反量化 GEMM **94.9%**。这两个 kernel 没有剩下多少可榨的。
2. **量化的字节 → 时间的换算率不是 1:1，而且系统性偏低**：专家权重字节 **÷1.97 / ÷3.88**，
   但时间只 **÷1.52 / ÷2.24**（96.3 → 63.5 / 43.0 µs），达成率从 71.0% 掉到 56.0% / 44.2%。
   这是"字节预算"立论在 **kernel 级**的边界：packed 权重省下的是 DRAM 流量，
   代价是 kernel 里解包的 ALU/issued 指令，后者按字节摊薄不了。
3. **kernel 级的数字能对上步级的结果**：16 层 ×（96.3 − 63.5 µs）= **0.524 ms/token**，
   占 `EXP-p4b-01` 的 E7 实测步级差值（3.5881 − 3.0084 = 0.580 ms）的 **90%**。
   也就是说，那一步的量化收益，九成可以由**专家 GEMV 一个 kernel** 解释完。
   这条比"整体快了 1.19×"有用得多：它把收益落到了具体 kernel 上。

## 1. 计数器路径：本机不可用（原始证据）

`ncu` 装好了、版本也对，指标清单是本仓库自己的契约（`lemoq.kernels.NCU_METRICS`），
但**任何** profiling 都被宿主拒绝：

```text
$ /usr/local/cuda/bin/ncu --version
Version 2025.1.1.0 (build 35528883) (public-release)

$ ncu --kernel-name regex:_rms_norm_kernel --launch-count 1 \
      --metrics gpu__time_duration.sum --csv <python> -m tools.ncu_evidence --subject rmsnorm_h2048
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

两个宿主侧事实（实测，均在容器内读到）：

| 事实 | 值 | 含义 |
| --- | --- | --- |
| 驱动的模块参数 | `RmProfilingAdminOnly: 1`（`/proc/driver/nvidia/params`） | 只有具备 `CAP_SYS_ADMIN` 的用户可以 profiling |
| 本容器的能力集 | `CapEff: 00000000a80425fb`，**第 21 位（`CAP_SYS_ADMIN`）为 0** | 我们不满足那个条件 |

★ **注意上一条命令用的是 `gpu__time_duration.sum`** —— 一个连性能计数器都不需要的指标，
它也被拒。所以这不是"某个指标在 sm_120 上不支持"，而是**整条 profiling 通道关闭**。
两者都在宿主（驱动 + 容器启动参数）上，容器内无法改：既不能给自己加 capability，
也不能在共享宿主上 `modprobe` 那个参数。

按 [NVIDIA 官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)，
解法在**启动容器/实例**的那一侧（`--cap-add=SYS_ADMIN`，或宿主把
`NVreg_RestrictProfilingToAdminUsers` 设为 0）。本项目的机器是**按小时租的共享容器**，
拿不到那一侧的控制权 ⇒ 这一格记为**阻断**，而不是"没做"。

工具本身是可复核的、也是能在允许 profiling 的主机上直接跑的：
`python -m tools.ncu_evidence --print-commands` 打印全部 ncu 命令行；
`--cache-control all` / `--clock-control none` 已按 §2 的理由写死。

## 2. 替代口径：kernel 级带宽达成率（它是什么、必须修正什么）

```text
attainment = 解析字节 ÷ (cold 计时 − 事件地板) ÷ 实测可达带宽
```

- **解析字节**是手算的必要流量（top_k 个专家 × 形状 × 位宽；见工件的 `meta.bytes_note`），
  不是从测量反推的；
- **实测可达带宽 1560.3 GB/s（只读；copy 1529.5）** 来自 `results/machine_probe.json`
  （硬规则 4：不用规格值）。★ 本文各表的**达成率是按 10-06/08 旧实例的 1561.9 GB/s 算的**
  （只读值差 0.1%，不重算）；换机器/换轮次必须重测，不能跨卡套用；
- 口径与记录里步级的"带宽达成率"**同一个**，只是从"一步"缩到"一个 kernel / 一次调用"。

计数器能多给的只有**为什么**（stall 归因、achieved occupancy、sector/request），
给不了**是多少**。所以 §3 的表回答了"到没到墙"，但没有回答"为什么没到墙"。

### 已经踩过、并修掉的两个坑（不是理论风险，是第一版真的错了）

| 坑 | 第一版的错误结果 | 修正 |
| --- | --- | --- |
| **L2 不是 DRAM** | `dequant` 报 237%、`grouped_fp16` 报 94% 的"带宽达成率"（工作集 104.7 / 100.7 MB，而 L2 只有 96 MB） | 每次计时前用 256 MB 的 memset 冲掉 L2（在事件对**之外**），只认 `cold` 列 |
| **事件本身有开销** | 事件对的地板就有 **5.472 µs**，而融合 RMSNorm 整个调用只有 4.1 µs ⇒ 会把地板当成 kernel 时间 | 显式测地板并减去；减完为负的一律记 `null`（**测不出来就说测不出来**，不写 0%） |
| 字节归属 | per-kernel 的 subject 拿到了**整层**的字节数，于是 gate_up"达成率 97%"是假的 | 字节按 subject 的 scope 算（`gate_up` / `down` / `layer`）；隔离臂只用整层 scope，因为一次调用会发 2–4 个 kernel |

### 本轮有效性

- **首尾同格金丝雀**：`grouped_fp16_layer` cold min 0.0963 → 0.0983 ms，漂移 **+2.09%** ≪ 5% ⇒ 本轮有效；
- **同租户**：本轮 `nvidia-smi --query-compute-apps` **为空**（空闲独占）；
  此前一轮实测到 MoE-Flow 的 `pareto_offload.py`（Qwen3-30B，928% CPU）在卡上，
  协议是**记录、绝不 kill**（AGENTS.md §6）；
- **设备**：NVIDIA GeForce RTX 5090（sm_120），不锁时钟。

## 3. 结果

硬件：RTX 5090 32 GB（sm_120），实测可达带宽 **1560.3 GB/s（只读；旧实例 1561.9）**，L2 **96 MB**，空闲独占，
金丝雀漂移 +2.09%。`cold min` 是主判据；`warm` 一列是背靠背调用的 p50（含宿主路径）。

| subject（一次调用） | 解析字节 | 工作集 | cold min | cold p50 | warm p50 | **达成率 cold** | 达成率 warm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grouped_fp16_layer` | 100.7 MB | 100.7 MB | 0.0963 ms | 0.0983 ms | 0.0687 ms | **71.0%** | 93.8% |
| `grouped_w8a16_layer` | 51.1 MB | 51.1 MB | 0.0635 ms | 0.0655 ms | 0.0769 ms | **56.0%** | 42.6% |
| `grouped_w4a16_layer` | 26.0 MB | 26.0 MB | 0.0430 ms | 0.0430 ms | 0.0766 ms | **44.2%** | 21.7% |
| `dequant_lm_head_w8a16` | 104.6 MB | 104.7 MB | 0.0758 ms | 0.0778 ms | 0.0283 ms | **94.9%** | 237.2% |
| `attn_split_kv_1024` | 8.4 MB | 8.4 MB | 0.0143 ms | 0.0143 ms | 0.0658 ms | **58.3%** | 8.2% |
| `rmsnorm_h2048` | 8 KB | 8 KB | 0.0041 ms | 0.0061 ms | 0.0177 ms | **null** | 0.0% |

工件：`results/ncu_evidence/isolated.json`（每格含 15 次 cold 与 5 轮 warm 的原始样本、事件地板、
同租户快照、金丝雀）。专家形状取 OLMoE 单层：`gate_up (64,2048,2048)` + `down (64,1024,2048)`，
固定取前 8 个专家（真实路由只换行号、不换流量 —— 这条已写进 `meta`，避免它变成第三个变量）。

## 4. 从表里读出来的五件事

### 4.1 两个 kernel 已经在墙上，没有剩下的空间

- grouped fp16 专家 GEMV：**cold 71.0% / warm 93.8%**；
- dense 融合反量化 GEMM（lm_head 形状）：**cold 94.9%**。

`EXP-p4c-01` 记过那一次投影 kernel 的修复（每元素 gather → 每 tile 装载 + 广播，
8.14 → 5.15 µs/call，523 → 826 GB/s）。本表是那条修复的**终点证据**：94.9% 意味着
这条路径基本没有可再榨的带宽了。剩下的空间在别处（元素级尾巴、launch 数）。

### 4.2 量化的"字节 → 时间"换算率系统性偏低（kernel 级立论边界）

| 位宽 | 字节比（vs fp16） | cold 时间比 | 达成率 | 16 层/token |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 1.00×（100.7 MB） | 1.00×（96.3 µs） | 71.0% | 1.540 ms |
| w8a16 | **1.97× 更少**（51.1 MB） | 1.52× 更快（63.5 µs） | 56.0% | 1.016 ms |
| w4a16 | **3.88× 更少**（26.0 MB） | 2.24× 更快（43.0 µs） | 44.2% | 0.688 ms |

读法：**省下的字节是 DRAM 流量，代价是 kernel 内解包的 ALU/发指令成本**；
后者不随位宽线性下降，于是达成率单调下滑。这解释了为什么"权重 ÷4"拿不到"延迟 ÷4"——
不是 host 的错（这是隔离测量，没有 host），是 kernel 自己不再带宽受限了。
它与 `EXP-p2-01`（E2 `refuted`）方向一致，但**更强的说法**在这里：
不是"融合没兑现"，而是"融合之后，瓶颈从 DRAM 挪到了解包"。

> ⚠️ 边界：`w4a16` 的引用门 FAIL（`EXP-p1-04` v2），所以它的数字**只作 evidence**，
> 不得作为性能结论引用（硬规则 2/8）。w8a16 是唯一过门的量化配置。

### 4.3 kernel 级的数字解释了步级结果（90%）

```text
16 层 × (96.3 − 63.5 µs)  = 0.524 ms/token      ← 隔离测量给出的预测节省
EXP-p4b-01（E7）实测步级差 = 3.5881 − 3.0084    = 0.580 ms/token
                           0.524 / 0.580        = 90.4%
```

同一个模型、同一张卡、同一个位宽，**只由专家 GEMV 一个 kernel** 就解释了 E7 步级收益的九成，
剩下的 0.056 ms 归给 attention/投影那侧的量化差。这是本项目第一次把步级结论
**闭合到 kernel 级**，也是"为什么只有 1.19×"这条叙事的硬支撑：
收益有上限、上限可定位、且已经兑现了九成。

### 4.4 attention 的一次调用在隔离下是 **host 主导**的

`attn_split_kv_1024`：cold **0.0143 ms**（58.3% of wall，纯 GPU），warm **0.0658 ms**（8.2%）。
差 4.6× —— 背靠背调用时每次要花 ~66 µs 的**宿主路径**（Python + 2 次 launch），
而 GPU 只干 14 µs 的活。这与 `EXP-p4c-01`（"只有进图才赢"）是同一件事的两个侧面：
自研 attention kernel 的价值要等 CUDA Graph 把宿主路径消掉之后才兑现。

（口径说明：cold 那一列里，计时前的 256 MB memset 让 GPU 保持忙碌，
宿主有足够时间把 kernel 排进队列 ⇒ cold 测到的是纯 GPU 时间；
这正是"cold 比 warm 快"的原因，也是这个方法的**机制**而不是巧合。）

### 4.5 融合 RMSNorm 用这个口径**测不出来** —— 这本身就是结论

`rmsnorm_h2048` 一次调用只有 8 KB 流量，cold min **4.1 µs < 事件地板 5.472 µs** ⇒ 记 `null`。
它的价值不在字节：`EXP-p4c-02` §5.6 量到的是 **994 → 539 kernel/步**，全步 **1.2358×**。
一个 8 KB 的 kernel 不可能靠带宽赢 —— 本表把这个直觉变成了"我们测了，测不出来，所以不编数"。

## 5. 适用区间与已知偏差

| 项 | 值 / 说明 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，sm_120，32 GB（结论只在这一档上成立） |
| 带宽口径 | 实测可达 **1560.3 GB/s（只读；copy 1529.5）**（`results/machine_probe.json`），非规格值；本文达成率按旧实例 1561.9 算（差 0.1%，不重算） |
| batch / 形状 | batch=1，seq=1 的 decode；专家取 top-8；attention ctx=1024 / MHA 16×128 |
| 时钟 | **不锁**；同租户为空（记录在工件里） |
| `cold` 的偏差方向 | 含**单次 launch** 的间隔 ⇒ <10 µs 的 kernel 达成率是**下界**（正确的读法是"延迟受限"） |
| `warm` 的偏差方向 | 工作集 < L2 时会被 L2 抬成 >100%（见 dequant 的 237.2%）⇒ 不能当 DRAM 数字 |
| 专家选择 | 固定取前 8 个专家：字节/形状与真实路由一致，只有行号不同（可复现性优先） |
| 计数器缺口 | **没有** stall 归因 / achieved occupancy / sector-request —— 需要 profiling 许可的主机 |
| 不与步级数字混比 | 本表是**隔离**测量；步级数字见对应记录，两者只在 §4.3 显式对照 |

## 6. 复现

```bash
# 服务器（租的 5090）
export PATH=/root/miniconda3/bin:$PATH
cd /root/LeanMoE-Quant

# 1) 先确认 profiling 是否被放开（被拒就记阻断，不要绕）
ncu --metrics gpu__time_duration.sum --launch-count 1 /root/miniconda3/bin/true || true

# 2) 计数器路径（本机被拒：ERR_NVGPUCTRPERM）
python -m tools.ncu_evidence --print-commands                    # 审查将要执行的命令行
python -m tools.ncu_evidence --only grouped_w4a16_gate_up        # 单格

# 3) 替代口径（本文件的数据就是这条产出的）
python -m tools.ncu_evidence --isolated --json results/ncu_evidence/isolated.json
```

## 7. 还缺什么

- **计数器**：`smsp__warp_issue_stalled_*`、`sm__warps_active`、`l1tex__average_t_sectors_per_request`
  —— 4-bit 的 `sectors/request` 尤其想看（nibble 展开最容易不合并）。
  门槛是宿主侧放开 profiling；工具与指标清单已就绪，换一台允许 profiling 的机器即可。
- `EXP-p3-01` / `EXP-p3-02`（FP8 / MXFP4）：本表只覆盖 fp16 / w8a16 / w4a16 三条；
  sm_120 的 block-scale 路径需要它们自己的证据。
