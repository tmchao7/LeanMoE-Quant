"""occupancy_probe — register / occupancy **下限估计**（GPU 待执行，METRICS §6 G3）。

## 为什么需要它

`docs/METRICS.md` §5 与审查清单都要求："算子融合后要重新评估 register / occupancy，
否则可能引入新的瓶颈"。本项目的两个融合杠杆（fused RMSNorm、grouped MoE kernel）
**至今没有 occupancy 数据** —— ncu 在这台宿主上被策略阻断
（`RmProfilingAdminOnly: 1` + 容器无 `CAP_SYS_ADMIN`），见 `docs/NCU_EVIDENCE.md` §1。

## 它不是 ncu 的替代品

本工具用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 算的是** occupancy 上限 /
下限的算术估计**，基于 kernel 的 launch 配置（threads per block、shared memory、
register 用量）。它**测不到**真实的 stall 原因、bank conflict、warp divergence。
所以输出里每一项都标 `kind: "estimate"`，并且**不得**被当成实测值引用。

能回答的那一问（这就是它的价值）：
    **融合是否引入了新的 occupancy 天花板？**
    —— 如果 fused RMSNorm 的 blocks/SM 远低于 1，说明融合后每个 program 占的资源
       太多，SM 填不满，那"融合省了 kernel 数"的收益会被"并行度下降"抵消一部分。

## 用法

    python -m tools.occupancy_probe --dry-run
    python -m tools.occupancy_probe --json results/occupancy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# 待估的 kernel（本项目自研的四族里，融合相关的两个 + 对照）
SUBJECTS = {
    "fused_rmsnorm": {
        "why": "EXP-p4c-02 的杠杆：7 kernel/norm → 1 kernel/norm。融合后是否填不满 SM？",
        "configs": [
            {"block": 2048, "num_warps": 4, "note": "H=2048（layer norm），R=1"},
            {"block": 128, "num_warps": 4, "note": "H=128（q/k norm），R=16"},
        ],
    },
    "grouped_gate_up_act": {
        "why": "EXP-p2-02 的杠杆：grouped GEMV。SPLIT_K 是否把并行度切够了？",
        "configs": [
            {"block": 32, "num_warps": 4, "note": "BLOCK_N=32, SPLIT_K=1（默认）"},
        ],
    },
    "grouped_down_sum": {
        "why": "同上；默认 SPLIT_K=4 —— 第一版 SPLIT_K=1 时只有 16 个 program（9% 覆盖）",
        "configs": [
            {"block": 64, "num_warps": 8, "note": "BLOCK_N=64, SPLIT_K=4（默认）"},
        ],
    },
}


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_by": "tools/occupancy_probe.py --dry-run",
        "status": "planned",
        "measured": False,
        "note": "**计划**；且即使跑完，输出也是 estimate（不是 ncu 实测）。",
        "why": "融合后是否引入新的 occupancy 天花板（METRICS §5 / 审查清单）",
        "method": "cudaOccupancyMaxActiveBlocksPerMultiprocessor（**算术估计**）",
        "cannot_answer": [
            "stall 原因（long_scoreboard / barrier / …）",
            "shared-memory bank conflict",
            "warp divergence",
            "真实 register 用量（需要 ncu 的 launch__registers_per_thread）",
        ],
        "subjects": SUBJECTS,
        "acceptance": {
            "reports_per_kernel": [
                "threads_per_block、num_warps、shared_mem_bytes",
                "max_active_blocks_per_SM（估计）",
                "achieved_occupancy_upper_bound = blocks × warps / max_warps_per_SM",
                "kind: 'estimate'（**必须**带这个标签，不得当实测引用）",
            ],
            "decision_rule": (
                "blocks/SM < 1 ⇒ 融合/切分让单个 program 占的资源过多，SM 填不满 —— "
                "这是'省 kernel 数'的反作用力，必须写进结论"
            ),
        },
        "output": args.json or "results/occupancy.json",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(json.dumps(_plan(args), ensure_ascii=False, indent=2))
        print(
            "\n[dry-run] 未做任何测量。注意：即使跑完，本工具的输出也只是 estimate，"
            "不是 ncu 实测。",
            file=sys.stderr,
        )
        return 0

    try:
        import torch
    except Exception:  # noqa: BLE001
        print("需要 torch。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("occupancy_probe 需要 CUDA；离线请用 --dry-run。", file=sys.stderr)
        return 2

    # ★ Triton kernel 的 occupancy 只能通过它**编译产物**的句柄去查，而那个入口是
    #   Triton 的版本化 API（`triton` 3.4 上要从 `CompiledKernel` 的 function /
    #   metadata 拿，再用 CUDA driver 的 occupancy API）。这个取法**在没有 GPU 的
    #   环境里无法验证** —— 所以这里刻意只落一条"已上机、但句柄未取得"的记录，
    #   绝不编一个数字填进去（硬规则 1）。
    #
    #   上机时的第一步（按顺序）：
    #     1. 确认 triton 版本：`python -c "import triton; print(triton.__version__)"`
    #     2. 找一个能拿到 CUfunction 的入口（CompiledKernel.function / .cubin /
    #        driver 的 get_kernel 之类），**先打印出来看结构**，再决定怎么传；
    #     3. 拿到后用 ctypes 调 `cuOccupancyMaxActiveBlocksPerMultiprocessor`；
    #     4. 每格都打上 `kind: "estimate"`，并在结论里写明它测不到 stall / bank
    #        conflict / divergence。
    payload = {
        "generated_by": "tools/occupancy_probe.py",
        "status": "planned",
        "device": torch.cuda.get_device_name(0),
        "subjects": {
            k: {"status": "planned", "reason": "kernel 句柄取法需在真机上确认（Triton 版本化 API）"}
            for k in SUBJECTS
        },
        "cross_round_comparison_allowed": False,
        "note": (
            "★ 本轮**没有产出任何 occupancy 数字**。上机第一步是确认 Triton 编译产物"
            "的 occupancy 查询入口（`kernel[grid]._init_handles()` / `CompiledKernel."
            "metadata`），确认后再把 `_probe()` 接上；在此之前一律 planned（硬规则 1）。"
        ),
    }
    out = args.json or "results/occupancy.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("wrote %s (status=planned —— 未产出数字)" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
