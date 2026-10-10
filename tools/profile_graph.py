#!/usr/bin/env python3
"""profile_graph — 当前 decode 路径的 in-graph kernel 分解（决定下一口咬哪里）。

背景（2026-10-10）：graph 格的 `result.json` 里 `top_kernels` 全是 0.000，而带真实
per-kernel ms 的工件全是 E2 时代 hf 旧路径（147/56 ms gpu_total）——没有任何工件能
回答"3.01 ms 的一步里谁占大头"。本工具直接调 `benchmarks.runner.run_cudagraph()`，
其内部的 `measure_graph_gap` 自带 wall + CUDA event + profiler kernel_table，
把 `res["gap"]` 原样落盘并打印 top 表。

★ **不写任何 result.json**：这是定位工具（同 batch_sweep/occupancy_probe 一档），
  不是实验格子——不触碰硬规则 9，也不产生可引用的"性能结论"（结论要走矩阵+记录）。

用法（5090）：
    python -m tools.profile_graph --model <MODEL> --json results/graph_profile.json
    python -m tools.profile_graph --model <MODEL> --bit-width fp16 --dequant none \
        --attn sdpa --json results/graph_profile.fp16.json          # 对照臂

退出码：0 = measured；1 = 构建/捕获/测量失败（原因进 stderr 与 JSON 的 error 字段）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.runner import run_cudagraph


def build_payload(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """把 run_cudagraph 的结果收成定位用 JSON（gap 原样保留）。"""
    gap = res.get("gap") or {}
    return {
        "generated_by": "tools/profile_graph.py",
        "status": "measured" if gap.get("top_kernels") else "incomplete",
        "config": cfg,
        "wall_p50_ms": gap.get("wall_p50_ms"),
        "gpu_busy_ms": gap.get("gpu_busy_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "kernels_per_call": gap.get("kernels_per_call"),
        "gap": gap,
        "note": (
            "定位用 kernel 分解，不是实验格子：share_pct = self_device_time ÷ gpu_busy "
            "（profiler 口径，per graph step）。要引用数字必须另立格子走矩阵。"
        ),
    }


def print_table(payload: dict[str, Any]) -> None:
    gap = payload.get("gap") or {}
    print(
        "step: wall_p50=%.4f ms  gpu_busy=%.4f ms  host=%.1f%%  kernels/step=%.0f"
        % (
            payload.get("wall_p50_ms") or 0.0,
            payload.get("gpu_busy_ms") or 0.0,
            payload.get("host_share_pct") or 0.0,
            payload.get("kernels_per_call") or 0.0,
        )
    )
    rows = gap.get("top_kernels") or []
    print("%8s %10s %8s  %s" % ("ms/step", "share%", "occ/stp", "kernel"))
    for r in rows:
        print(
            "%8.4f %10.2f %8.1f  %s"
            % (
                r.get("self_ms_per_token") or 0.0,
                r.get("share_pct") or 0.0,
                r.get("occurrences_per_token") or 0.0,
                str(r.get("name"))[:90],
            )
        )
    tail = (gap.get("kernels_per_call") or 0) - sum(
        r.get("occurrences_per_token") or 0.0 for r in rows
    )
    if tail > 0:
        print("  … tail of ~%.0f more kernel occurrences/step not in the top table" % tail)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--moe", default="grouped", choices=["hf", "static", "gather", "grouped"])
    ap.add_argument("--attn", default="kernel", choices=["sdpa", "eager", "kernel"])
    ap.add_argument("--bit-width", default="w8a16", choices=["fp16", "w8a16", "w8a8", "w4a16"])
    ap.add_argument("--dequant", default="fused", choices=["none", "separate", "fused"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--rmsnorm", default="fused", choices=["fused", "hf"])
    ap.add_argument("--json", default="results/graph_profile.json")
    args = ap.parse_args(argv)

    quantized = args.bit_width != "fp16"
    cfg = {
        "model": args.model,
        "ctx": args.ctx,
        "batch": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "dtype": args.dtype,
        "moe": args.moe,
        "attn": args.attn,
        "bit_width": args.bit_width,
        "dequant": args.dequant,
        "group_size": args.group_size,
        "rmsnorm": args.rmsnorm,
    }
    out = Path(args.json)
    try:
        res = run_cudagraph(
            model_id=args.model,
            ctx=args.ctx,
            batch=args.batch,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            timed=args.timed,
            dtype_name=args.dtype,
            moe=args.moe,
            attn=args.attn,
            quant_cfg=(
                {
                    "bit_width": args.bit_width,
                    "group_size": args.group_size,
                    "ctx": args.ctx,
                    "dequant": args.dequant,
                }
                if quantized
                else None
            ),
            rmsnorm=args.rmsnorm,
        )
    except Exception as exc:  # noqa: BLE001 — 定位工具：失败要原因，不要 traceback 刷屏
        payload = {
            "generated_by": "tools/profile_graph.py",
            "status": "failed",
            "config": cfg,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:500]),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[profile_graph] FAILED: %s" % payload["error"], file=sys.stderr)
        return 1

    payload = build_payload(res, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(payload)
    print("[profile_graph] wrote %s (status=%s)" % (out, payload["status"]))
    return 0 if payload["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
