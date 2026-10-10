"""occupancy_probe — register / occupancy **下限估计**（GPU 待执行，METRICS §6 G3）。

## 为什么需要它

`docs/reference/METRICS.md` §5 与审查清单都要求："算子融合后要重新评估 register / occupancy，
否则可能引入新的瓶颈"。本项目的两个融合杠杆（fused RMSNorm、grouped MoE kernel）
**至今没有 occupancy 数据** —— ncu 在这台宿主上被策略阻断
（`RmProfilingAdminOnly: 1` + 容器无 `CAP_SYS_ADMIN`），见 `docs/reference/NCU_EVIDENCE.md` §1。

## 它不是 ncu 的替代品

本工具用**解析式**（regs / shared / threads 三个除法取最小，纯算术、可离线单测）
估算 occupancy 上限，输入是编译产物里的 `n_regs` / `shared` 与 launch 配置
（threads per block = num_warps × 32）。它**不调用** CUDA occupancy API，也**测不到**
真实的 stall 原因、bank conflict、warp divergence。
所以输出里每一项都标 `kind: "estimate"`，并且**不得**被当成实测值引用。

能回答的那一问（这就是它的价值）：
    **融合是否引入了新的 occupancy 天花板？**
    —— 如果 fused RMSNorm 的 blocks/SM 远低于 1，说明融合后每个 program 占的资源
       太多，SM 填不满，那"融合省了 kernel 数"的收益会被"并行度下降"抵消一部分。

## 用法

    python -m tools.occupancy_probe --dry-run
    python -m tools.occupancy_probe --json results/occupancy.json

## 上界怎么算（解析式，不依赖 ncu）

从**编译产物**拿三样东西：`n_regs`（每线程寄存器）、`n_spills`、`metadata.shared`（动态
shared memory 字节）。再用设备上限做三个除法，取最小：

    blocks_by_regs    = regs_per_sm    // (n_regs * threads_per_block)
    blocks_by_smem    = smem_per_sm    // max(shared_bytes, 1)
    blocks_by_threads = max_threads_sm // threads_per_block
    blocks_per_sm     ≤ min(三者)

这是 **occupancy 的解析上界**（不是实测 occupancy）：它不考虑 warp 调度、尾效应、
寄存器分配的粒度对齐。所以每一项都标 `kind: "estimate"`、`bound: "upper"`。

★ 这个算术**可以离线单测**（见 `tests/test_occupancy_probe.py`）—— 上机前就能确认
它没有算错，上机时只剩"能不能取到句柄"这一个未知。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# 各 compute capability 的 SM 资源上限。优先用运行时设备属性覆盖（见 `_limits`），
# 这张表是**查不到时的兜底**，值取自 CUDA 编程指南的 per-SM 表。
_SM_LIMITS = {
    (12, 0): {"regs_per_sm": 65536, "smem_per_sm": 228 * 1024, "max_threads_sm": 1536},
    (9, 0): {"regs_per_sm": 65536, "smem_per_sm": 228 * 1024, "max_threads_sm": 2048},
    (8, 9): {"regs_per_sm": 65536, "smem_per_sm": 100 * 1024, "max_threads_sm": 1536},
    (8, 6): {"regs_per_sm": 65536, "smem_per_sm": 100 * 1024, "max_threads_sm": 1536},
    (7, 0): {"regs_per_sm": 65536, "smem_per_sm": 96 * 1024, "max_threads_sm": 2048},
}


def _threads_of(cfg: dict[str, Any]) -> int:
    """Triton 程序的线程数 = num_warps × 32。

    2026-10-10 修：原来把 `cfg["block"]`（元素数）当 threads_per_block 传给上界
    函数 —— H=2048 的 norm 配置因此被算成"2048 线程/块 > 1536 上限 ⇒ occupancy 0%"
    的假结论。block 是每程序处理的元素数，线程数只由 num_warps 决定。
    """
    return int(cfg.get("num_warps", 4)) * 32


def occupancy_upper_bound(
    *, n_regs: int, threads_per_block: int, shared_bytes: int, limits: dict[str, int]
) -> dict[str, Any]:
    """三个除法取最小 ⇒ blocks/SM 上界 + 占用率上界。**纯算术，可离线单测。**"""
    regs = max(int(n_regs), 1)
    tpb = max(int(threads_per_block), 1)
    by_regs = int(limits["regs_per_sm"]) // (regs * tpb)
    by_smem = int(limits["smem_per_sm"]) // max(int(shared_bytes), 1)
    by_threads = int(limits["max_threads_sm"]) // tpb
    blocks = max(0, min(by_regs, by_smem, by_threads))
    return {
        "kind": "estimate",
        "bound": "upper",
        "blocks_per_sm_bound": blocks,
        "occupancy_upper_pct": round(100.0 * blocks * tpb / int(limits["max_threads_sm"]), 1),
        "limited_by": min(
            [("regs", by_regs), ("shared", by_smem), ("threads", by_threads)],
            key=lambda kv: kv[1],
        )[0],
        "inputs": {
            "n_regs": regs,
            "threads_per_block": tpb,
            "shared_bytes": int(shared_bytes),
            **limits,
        },
        "note": (
            "解析上界（不等同实测 occupancy）：不含 warp 调度、尾效应与寄存器分配粒度。"
            "blocks/SM 远小于 1 ⇒ 融合后 SM 填不满，并行度下降会抵消一部分收益。"
        ),
    }


def _limits(torch_mod: Any) -> dict[str, int]:
    """设备上界：优先读运行时属性，读不到回退到 `_SM_LIMITS` 表。"""
    props = None
    try:
        props = torch_mod.cuda.get_device_properties(0)
    except Exception:  # noqa: BLE001
        props = None
    if props is not None:
        cc = (int(getattr(props, "major", 0)), int(getattr(props, "minor", 0)))
        base = dict(_SM_LIMITS.get(cc) or _SM_LIMITS[(12, 0)])
        for key, attr in (
            ("regs_per_sm", "regs_per_multiprocessor"),
            ("smem_per_sm", "shared_memory_per_multiprocessor"),
            ("max_threads_sm", "max_threads_per_multi_processor"),
        ):
            val = getattr(props, attr, None)
            if isinstance(val, int) and val > 0:
                base[key] = val
        return base
    return dict(_SM_LIMITS[(12, 0)])


def _iter_compiled(obj: Any, depth: int = 0):
    """深度优先扫 dict/tuple/list 的任意嵌套，产出带 `n_regs` 的 CompiledKernel。

    2026-10-10 真机（triton 3.4.0）实测的第三种布局：`device_caches[device]` 是
    **4 元组 `(cache_dict, GPUTarget, backend, fn)`**，CompiledKernel 藏在元组第 0 项
    的 dict 里 —— 旧实现只认 dict，于是把跑过的 kernel 当成"没编译产物"（G3 首轮
    planned 的根因）。dict 之外的元素（GPUTarget/backend/函数）不是容器，直接跳过。
    """
    if depth > 5:
        return
    if isinstance(obj, dict):
        nodes: Any = obj.values()
    elif isinstance(obj, (tuple, list)):
        nodes = obj
    else:
        return
    for value in nodes:
        if hasattr(value, "n_regs"):
            yield value
        else:
            yield from _iter_compiled(value, depth + 1)


def _find_compiled_kernel(jit_fn: Any) -> Any | None:
    """从 JITFunction 上找到**已编译**的那份产物。

    Triton 版本差异都收在 `_iter_compiled` 里：
      - 3.x 早期：`.cache` = dict-of-dict；
      - 较新：`.device_caches` = dict-of-dict；
      - 3.4.0：`.device_caches[device]` = **tuple**(cache_dict, …)。
    扫不到返回 None（调用方据此保持 planned，不编数字）。
    """
    for attr in ("device_caches", "cache"):
        cache = getattr(jit_fn, attr, None)
        if cache is None:
            continue
        for ck in _iter_compiled(cache):
            return ck
    return None


def _compiled_kernel_info(jit_fn: Any, *, attempts: list[str]) -> dict[str, Any] | None:
    """取 `n_regs / n_spills / shared`；取不到返回 None 并记录尝试过的路径。

    ★ 前提是 kernel **已经被真跑过一次**（缓存里才有产物）—— 所以调用方必须先
      用真实张量跑一次，见 `_warm_kernels`。这里只负责读。
    """
    try:
        ck = _find_compiled_kernel(jit_fn)
        if ck is None:
            attempts.append("cache 里没有编译产物（kernel 没跑过？）")
            return None
        attempts.append("cache")
        meta = getattr(ck, "metadata", None)
        shared = int(getattr(meta, "shared", 0) or 0)
        n_regs = int(getattr(ck, "n_regs", 0) or 0)
        if n_regs <= 0:
            attempts.append("n_regs=0（该 Triton 版本不暴露寄存器数）")
            return None
        return {
            "n_regs": n_regs,
            "n_spills": int(getattr(ck, "n_spills", 0) or 0),
            "shared_bytes": shared,
            "via": "JITFunction.cache/device_caches",
        }
    except Exception as exc:  # noqa: BLE001
        attempts.append("读取编译产物失败(%s)" % type(exc).__name__)
        return None


def _warm_kernels(torch_mod: Any, attempts: list[str]) -> None:
    """用**真实张量**把三个待估 kernel 各跑一次，好让编译产物进缓存。

    形状取项目里真实使用的那些（见 SUBJECTS 的 note）。任何一步失败都不抛 ——
    失败信息进 `attempts`，最终该 subject 保持 `planned`。
    """
    try:
        from lemoq.kernels import rmsnorm as _rn

        x = torch_mod.randn(1, 2048, dtype=torch_mod.float16, device="cuda")
        w = torch_mod.ones(2048, dtype=torch_mod.float16, device="cuda")
        _rn.rms_norm_fused(x, w, 1e-6)
        attempts.append("ran rms_norm_fused")
    except Exception as exc:  # noqa: BLE001
        attempts.append("rms_norm_fused 失败(%s)" % type(exc).__name__)

    try:
        from lemoq.kernels import moe_grouped as _mg
        from lemoq.kernels.moe_static import FusedExperts

        e, ffn, hidden = 4, 64, 32
        gate_up = torch_mod.randn(e, 2 * ffn, hidden, dtype=torch_mod.float16, device="cuda")
        down = torch_mod.randn(e, hidden, ffn, dtype=torch_mod.float16, device="cuda")
        fused = FusedExperts(gate_up=gate_up.contiguous(), down=down.contiguous())
        ids = torch_mod.zeros(1, 1, dtype=torch_mod.long, device="cuda")
        top_w = torch_mod.ones(1, 1, dtype=torch_mod.float16, device="cuda")
        x = torch_mod.randn(1, 1, hidden, dtype=torch_mod.float16, device="cuda")
        # ★ 参数顺序 = grouped_moe_forward(x, fused, ids, top_w)（2026-10-10 修：
        #   原来写成 (x, ids, top_w, fused, top_k=1) —— 顺序错 + 没有 top_k 形参，
        #   真机上直接 TypeError，grouped 两族 kernel 因此进不了缓存）。
        _mg.grouped_moe_forward(x, fused, ids, top_w)
        attempts.append("ran grouped_moe_forward")
    except Exception as exc:  # noqa: BLE001
        attempts.append("grouped_moe_forward 失败(%s)" % type(exc).__name__)


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

    # ★ 上界算术是解析式的；这里只解决"能不能从编译产物里取到 n_regs / shared"。
    #   两个入口都失败 ⇒ 保持 `planned` 并**记下失败的入口**，不编数字（硬规则 1）。
    limits = _limits(torch)
    subjects: dict[str, Any] = {}
    kernel_map = {
        "fused_rmsnorm": ("lemoq.kernels.rmsnorm", "_rms_norm_kernel"),
        "grouped_gate_up_act": ("lemoq.kernels._grouped_triton", "_gate_up_act_kernel"),
        "grouped_down_sum": ("lemoq.kernels._grouped_triton", "_down_sum_kernel"),
    }
    warm_attempts: list[str] = []
    _warm_kernels(torch, warm_attempts)
    for name, spec in SUBJECTS.items():
        mod_name, attr = kernel_map.get(name, (None, None))
        info = None
        attempts = list(warm_attempts)
        if mod_name:
            try:
                import importlib

                mod = importlib.import_module(mod_name)
                fn = getattr(mod, attr, None)
                if fn is None:
                    attempts.append("kernel attr %s 不存在" % attr)
                else:
                    info = _compiled_kernel_info(fn, attempts=attempts)
            except Exception as exc:  # noqa: BLE001
                attempts.append("import %s 失败(%s)" % (mod_name, type(exc).__name__))
        if info is None:
            subjects[name] = {
                "status": "planned",
                "why": spec["why"],
                "attempts": attempts,
                "reason": (
                    "编译产物信息取不到（入口/属性名随 Triton 版本变）—— 保持 planned，"
                    "不编数字（硬规则 1）"
                ),
            }
            continue
        subjects[name] = {
            "status": "measured_estimate",
            "why": spec["why"],
            "via": info["via"],
            "n_regs": info["n_regs"],
            "n_spills": info["n_spills"],
            "configs": [
                dict(
                    cfg,
                    **occupancy_upper_bound(
                        n_regs=info["n_regs"],
                        threads_per_block=_threads_of(cfg),
                        shared_bytes=info["shared_bytes"] + int(cfg.get("extra_smem_bytes", 0)),
                        limits=limits,
                    ),
                )
                for cfg in spec["configs"]
            ],
        }

    any_measured = any(v["status"] == "measured_estimate" for v in subjects.values())
    payload = {
        "generated_by": "tools/occupancy_probe.py",
        "status": "done" if any_measured else "planned",
        "device": torch.cuda.get_device_name(0),
        "sm_limits": limits,
        "subjects": subjects,
        "cross_round_comparison_allowed": False,
        "note": (
            "★ 所有数值都是 `kind=estimate` / `bound=upper` 的**解析上界**，不是 ncu 实测："
            "测不到 stall / bank conflict / warp divergence（见 docs/reference/NCU_EVIDENCE.md §1）。"
            "`status=planned` 表示编译产物取不到 —— 那种情况下**没有产出任何数字**。"
        ),
    }
    out = args.json or "results/occupancy.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    if payload["status"] == "done":
        n_measured = sum(
            1 for v in payload["subjects"].values() if v["status"] == "measured_estimate"
        )
        print(
            "wrote %s (status=done —— %d/%d 族 kernel 有解析上界)"
            % (out, n_measured, len(payload["subjects"]))
        )
    else:
        print("wrote %s (status=planned —— 未产出数字)" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
