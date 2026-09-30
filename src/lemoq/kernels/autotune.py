"""autotune — Triton/CUDA kernel 的参数空间与 ncu 指标清单。

**为什么参数空间也要提前写下**：

    `PLAN.md` §3 把 dequant-GEMM 作为主贡献。它的性能取决于 tile / group / warps 的
    选择，而"我调了几个参数"不是结论——**参数敏感性本身是一个可报告的结果**
    （对应阿里星 JD 里的 "auto-tuning & codegen"）。

    把搜索空间**在跑之前**写死，是为了满足 `AGENTS.md` 规则 3（一次一个变量）：
    上机时按这个网格扫，不临时加参数，否则结果不可比。

**ncu 指标清单**：按 `MegaKernel-vLLM/docs/15` §4.2，profiling 是推理优化岗的硬要求。
kernel 级证据是本项目目前最缺的一块，所以指标也提前定好。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 参数搜索空间
#
# 命名遵循 Triton 惯例。每个轴都说明**为什么可能影响**，否则调参就成了碰运气。
# ---------------------------------------------------------------------------
DEQUANT_GEMM_AUTOTUNE_SPACE: dict[str, Any] = {
    "axes": {
        "BLOCK_M": {
            "values": [1, 2, 4, 8, 16, 32],
            "why": (
                "batch=1 时 M=1 —— 这是本项目与常规 GEMM 最不同的地方。"
                "大 BLOCK_M 会浪费在 padding 上（MegaKernel-vLLM 在 WMMA 上"
                "实测过 tile padding 到 16 的浪费）"
            ),
        },
        "BLOCK_N": {
            "values": [64, 128, 256],
            "why": "输出维度切分；过大则寄存器压力与 shared 用量上升",
        },
        "BLOCK_K": {
            "values": [32, 64, 128],
            "why": (
                "K 维是反量化的分组维。BLOCK_K 与 group_size 的关系决定" "每组需要几次 scale 重载"
            ),
        },
        "GROUP_SIZE_M": {
            "values": [1, 4, 8],
            "why": "L2 复用；M=1 时基本无效，保留以验证这一点",
        },
        "num_warps": {"values": [2, 4, 8], "why": "并行度与 occupancy 的权衡"},
        "num_stages": {
            "values": [2, 3, 4],
            "why": (
                "软件流水深度。⚠️ V100(sm_70) 无 cp.async，深流水的收益来源不同；"
                "sm_89/sm_120 有 cp.async，行为会不一样"
            ),
        },
    },
    "held_fixed": {
        "dtype": "float16",
        "accum_dtype": "float32",
        "order": ["M", "N", "K"],
        "num_bits": [4, 8],
        "note": "搜索时不得同时改 bit-width（AGENTS.md 规则 3）",
    },
    "reporting": {
        "must_record": [
            "autotune 首次编译耗时（不得计入稳态）",
            "每个 config 的 p50/p99",
            "最优 config 与默认 config 的差值",
        ],
        "budget": "单次 autotune 上限 20 分钟，超时则退回默认 config 并记录",
    },
}

# ---------------------------------------------------------------------------
# ncu 指标清单
#
# 这决定了"我的 kernel 到没到带宽墙"这个问题能否被回答。
# ---------------------------------------------------------------------------
NCU_METRICS: dict[str, Any] = {
    "primary": {
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "★ 最关键：DRAM 吞吐占峰值的比例。它就是'带宽达成率'的 kernel 级版本",
        "gpu__time_duration.sum": "kernel 实际时长，与 CUDA event 交叉验证",
    },
    "occupancy": {
        "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved occupancy",
        "launch__registers_per_thread": "寄存器压力",
        "launch__occupancy_limit_registers": "是否被寄存器限制",
        "launch__shared_mem_per_block_allocated": "shared 用量（与 sm_70 的 96KB 上限对照）",
    },
    "stalls": {
        "smsp__average_warp_latency_per_inst_issued.ratio": "平均 warp 延迟",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct": "★ 等内存 —— 反量化融合是否真的把访存隐藏住了",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct": "等 shared/寄存器 —— 解压路径的开销",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "同步开销",
    },
    "memory_pattern": {
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "全局读的 sector 数",
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio": "★ 每个请求的 sector 数 —— packed 权重的访问是否合并（4-bit 加载极易不合并）",
        "lts__t_sector_hit_rate.pct": "L2 命中率（注意：高命中会让'达成率 >100%'）",
    },
    "how_to_run": [
        "# sm_120/89 上 ncu 可用于普通 kernel（本项目不用 cooperative launch）",
        "ncu --set full --kernel-name regex:dequant -o results/ncu/dequant_gemm python -m benchmarks.runner ...",
        "# 只要关键指标时：",
        "ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,"
        "sm__warps_active.avg.pct_of_peak_sustained_active python -m benchmarks.runner ...",
    ],
    "caveat": (
        "MegaKernel-vLLM 记录过：ncu 的 kernel replay 与 cooperative launch 冲突，会导致 "
        "驱动挂起。本项目不用 cooperative kernel，所以不受影响——但若日后引入，"
        "必须改用 nsys。"
    ),
}
