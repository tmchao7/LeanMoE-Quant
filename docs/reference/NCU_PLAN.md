# NCU_PLAN — ncu 补测清单（**换机才能做**）

> 状态：**blocked（宿主策略）**。本文是"等拿到有 `CAP_SYS_ADMIN` 的机器时照着跑"的清单，
> 不是已完成的测量。**里面没有任何实测数字。**
> 现状与替代口径见 [NCU_EVIDENCE.md](NCU_EVIDENCE.md) §1。

## 0. 为什么被阻断（先记下来，免得下次重复排查）

在租用的 RTX 5090 宿主上：

- `RmProfilingAdminOnly: 1`；
- 容器 `CapEff` 无 `CAP_SYS_ADMIN`；
- 结果：连 **counter-free** 的 `gpu__time_duration.sum` 都被拒
  （原始报错见 `NCU_EVIDENCE.md` §2）。

★ 那条 CapEff 值是**在那台机器上**读到的。换机器后必须**重新确认**，不能把这段
结论直接抄进新产物（这正是 `tools/ncu_evidence.py` 现在不再硬编码它的原因）。

上机第一步：

```bash
grep CapEff /proc/self/status
cat /proc/driver/nvidia/params | grep -i RmProfiling
ncu --version
```

## 1. 要测的 kernel（按优先级）

| # | kernel | 为什么 | 关联结论 |
|---|---|---|---|
| 1 | `_gate_up_act_kernel`（fp16 / 8-bit / 4-bit） | grouped MoE 的主算子，占一步 42.9% 的 GPU 时间 | E7 的 1.19×、NCU_EVIDENCE §4.3 |
| 2 | `_down_sum_kernel`（同上三种位宽） | 同上；`SPLIT_K=4` 是否真的换来了并行度 | 同上 |
| 3 | `_rms_norm_kernel`（fused RMSNorm） | 融合后是否引入新的 occupancy 天花板 | EXP-p4c-02 的 1.2358× |
| 4 | `_dequant_gemm_kernel`（dense fused dequant） | 融合反量化被判 refuted，想知道它到底卡在哪 | EXP-p2-01 |
| 5 | `_attn_split_kernel` / `_attn_combine_kernel` | split-KV 是否真的铺满 SM | EXP-p4c-01 |

每个 kernel 三种位宽各测一次（fp16 / w8a16 / w4a16），因为"字节 ÷3.88、时间只 ÷2.24"
这个现象（`NCU_EVIDENCE.md` §4.2）需要 stall 归因才能解释。

## 2. 要抓的 counter

### 2.1 必需（决定"瓶颈是什么"）

| counter / section | 回答什么 |
|---|---|
| `gpu__time_duration.sum` | kernel 时长（先确认这个能拿到，它是后面所有率的分母） |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | **实际 occupancy**（不是估计） |
| `launch__registers_per_thread` | register 用量 —— 融合是否把它顶到上限 |
| `launch__shared_mem_per_block_static` / `_dynamic` | shared memory 占用 |
| `launch__occupancy_limit_registers` / `_shared_mem` / `_blocks` | occupancy 被哪一项限住 |
| `smsp__average_warp_latency_issue_stalled_*_per_issue_active`（stall reason 全套） | **卡在哪**：long_scoreboard / barrier / mio_throttle / … |
| `dram__bytes.sum` + `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 是否真的贴到带宽墙（本项目所有达成率的分母） |
| `l2__throughput.avg.pct_of_peak_sustained_elapsed` | 是否被 L2 兜住（隔离计时里刻意 flush 的就是它） |

### 2.2 4-bit 专用（解释"字节 ÷3.88 时间只 ÷2.24"）

| counter | 为什么 |
|---|---|
| `smsp__inst_executed.sum`（按指令类型分组） | nibble 展开（`_expand_int4`）到底多了多少指令 |
| `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | 是否卡在 ALU（解包是整数运算，不是访存） |

## 3. 验收判据（跑完怎么算"这一格拿到了"）

| 判据 | 阈值 / 条件 |
|---|---|
| **counter 可达** | `gpu__time_duration.sum` 必须拿到值；拿不到 ⇒ 本轮仍 blocked，写 blocked 不写数字 |
| **occupancy 与估计对得上** | 实测 `sm__warps_active` 与 `tools/occupancy_probe.py` 的 estimate 方向一致；差得远要在记录里解释（估计漏了什么） |
| **dram 字节与手算对得上** | `dram__bytes.sum` 与 `NCU_EVIDENCE.md` 里"必要字节"的手算值相差 ≤ 若干百分点（具体阈值上机前定，并写明理由） |
| **stall 归因非空** | 至少能指出占比最大的一个 stall reason；全是 "selected" 说明采样不足 |
| **覆盖** | 5 个 kernel × 3 种位宽 = 15 次，缺一次就在产物里标 `partial`，不得标 done |

## 4. 产物与落盘

- 落 `results/ncu_evidence/ncu_<kernel>_<bits>.json`（每个 kernel 一份）；
- 汇总进 `results/ncu_evidence/isolated.json` 的同级 `ncu_real.json`，并把
  `isolated.json` 的 `ncu.ncu_attempted` 从 `false` 改成 `true`、
  `ncu_blocked_by` 填**本轮**实测到的原因（不要再抄别的机器的值）；
- 更新 `docs/reference/NCU_EVIDENCE.md`：把"替代口径"与"实测口径"并列，说明两者差多少。

## 5. 不要做的事

- 不要把 estimate 当实测引用（`tools/occupancy_probe.py` 的输出必须带 `kind: "estimate"`）；
- 不要把别的机器上读到的 CapEff / 报错抄进本轮产物（`tools/ncu_evidence.py` 已修）；
- 不要因为 ncu 拿到了就把历史数字重新算一遍达成率 —— 那是**跨轮**，违反硬规则 3；
  新口径只在新一轮里产出，单列表。
