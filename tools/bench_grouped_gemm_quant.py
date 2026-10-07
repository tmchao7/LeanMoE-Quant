#!/usr/bin/env python3
"""bench_grouped_gemm_quant — 隔离微基准：fp16 vs 8-bit **量化** grouped GEMV。

## 为什么需要它（E7 的机制证据，不是"再测一遍性能"）

`EXP-p4b-01` 的量化格（`w8a16 × grouped × eager`）**比 fp16 慢**。但"慢"本身不是机制，
而且单元格上的时间被 host 开销（~65%）稀释。本工具在同一进程里、**交错**测两条路径的
**kernel 时间**，并算出每次调用实际读了多少字节、达成多少带宽：

    fp16 : (E, 2F, H) fp16 + (E, H, F) fp16
    int8 : 同样的形状，但权重是 packed int8 + per-group fp16 scale

判读方式（写进实验记录时必须一起给）：

    - 若 int8 的 kernel 时间随字节数**大致减半** → 这条路径是带宽受限的，
      量化收益应该在单元格上兑现（没兑现的话，瓶颈在别处）；
    - 若 int8 **没有变快甚至更慢** → 该路径不是带宽受限的，
      权重省下来的字节在 decode 上兑现不了 —— 多出来的 int8→fp32 解包 ALU 反而要付钱。

★ 交错（round-robin）而不是两条臂各跑一遍：单臂顺序跑会把机器状态漂移算成方案差异。
这条纪律与 `EXP-p2-01`/`p2-05` 的 K3 一致。

用法：

    python -m tools.bench_grouped_gemm_quant --hidden 2048 --ffn 1024 --experts 64 --top-k 8
    python -m tools.bench_grouped_gemm_quant --sweep --json results/bench_quant_grouped.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any


def _bytes_read(top_k: int, ffn: int, hidden: int, *, bits: int, group_size: int) -> int:
    """每次调用**从 HBM 读**的权重字节（fp16 权重 vs packed int8 + scale）。"""
    params = top_k * (2 * ffn * hidden + hidden * ffn)  # gate_up + down
    if bits == 16:
        return params * 2
    packed = params * bits // 8
    scales = top_k * (2 * ffn * (hidden // group_size) + hidden * (ffn // group_size)) * 2
    return packed + scales


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "p99_ms": ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))],
        "min_ms": ordered[0],
        "mad_ms": statistics.median([abs(x - statistics.median(ordered)) for x in ordered]),
    }


def run(
    *,
    hidden: int = 2048,
    ffn: int = 1024,
    experts: int = 64,
    top_k: int = 8,
    group_size: int = 128,
    warmup: int = 10,
    repeats: int = 50,
    sweep: bool = False,
) -> dict[str, Any]:
    import torch

    from lemoq.kernels.moe_grouped import grouped_moe_forward
    from lemoq.kernels.moe_grouped_quant import (
        grouped_moe_forward_quant,
        quantize_fused_experts,
    )
    from lemoq.kernels.moe_static import FusedExperts

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA（这是 kernel 级微基准）")

    dev = "cuda"
    torch.manual_seed(0)
    fused = FusedExperts(
        gate_up=(torch.randn(experts, 2 * ffn, hidden, device=dev) * 0.05).half(),
        down=(torch.randn(experts, hidden, ffn, device=dev) * 0.05).half(),
    )
    quantized = quantize_fused_experts(fused, bits=8, group_size=group_size)

    x = (torch.randn(hidden, device=dev) * 0.5).half()
    ids = torch.randint(0, experts, (top_k,), device=dev, dtype=torch.int64)
    weights = torch.full((top_k,), 1.0 / top_k, device=dev).half()

    fp16_bytes = _bytes_read(top_k, ffn, hidden, bits=16, group_size=group_size)
    int8_bytes = _bytes_read(top_k, ffn, hidden, bits=8, group_size=group_size)

    result: dict[str, Any] = {
        "shapes": {
            "experts": experts,
            "ffn": ffn,
            "hidden": hidden,
            "top_k": top_k,
            "group_size": group_size,
        },
        "bytes_read": {"fp16": fp16_bytes, "int8": int8_bytes},
        "timing_method": "torch.cuda.Event 包住单次调用，交错（round-robin）测两条臂",
        "warmup": warmup,
        "repeats": repeats,
    }

    configs: list[dict[str, Any] | None] = [None]
    if sweep:
        configs = [
            None,
            {"gate_up_act": {"BLOCK_N": 64}, "down_sum": {"BLOCK_N": 64}},
            {"gate_up_act": {"BLOCK_N": 128}, "down_sum": {"BLOCK_N": 128}},
            {"gate_up_act": {"BLOCK_K": 64}, "down_sum": {"BLOCK_K": 64}},
            {"down_sum": {"SPLIT_K": 1}},
            {"gate_up_act": {"SPLIT_K": 2}, "down_sum": {"SPLIT_K": 2}},
            {"gate_up_act": {"num_warps": 8}, "down_sum": {"num_warps": 4}},
        ]

    arms: list[dict[str, Any]] = []
    for config in configs:
        # ★ 交错：每轮先 fp16 再 int8（同进程、同输入、同机器状态）
        fp16_samples: list[float] = []
        int8_samples: list[float] = []
        for _ in range(warmup):
            grouped_moe_forward(x, fused, ids, weights)
            grouped_moe_forward_quant(x, quantized, ids, weights, config=config)
        torch.cuda.synchronize()
        for _ in range(repeats):
            for fn, bucket in (
                (lambda: grouped_moe_forward(x, fused, ids, weights), fp16_samples),
                (
                    lambda: grouped_moe_forward_quant(x, quantized, ids, weights, config=config),
                    int8_samples,
                ),
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                bucket.append(start.elapsed_time(end))

        fp16 = _summarize(fp16_samples)
        int8 = _summarize(int8_samples)
        fp16_bw = fp16_bytes / (fp16["p50_ms"] / 1000.0) / 1e9
        int8_bw = int8_bytes / (int8["p50_ms"] / 1000.0) / 1e9
        entry = {
            "config": config or "default",
            "fp16": dict(fp16, effective_gbps=fp16_bw),
            "int8": dict(int8, effective_gbps=int8_bw),
            "int8_over_fp16_p50": int8["p50_ms"] / fp16["p50_ms"],
            "bytes_ratio": int8_bytes / fp16_bytes,
        }
        arms.append(entry)
        print(
            "%-58s fp16 %7.4f ms (%6.1f GB/s) | int8 %7.4f ms (%6.1f GB/s) | int8/fp16 %.3f"
            % (
                str(entry["config"])[:58],
                fp16["p50_ms"],
                fp16_bw,
                int8["p50_ms"],
                int8_bw,
                entry["int8_over_fp16_p50"],
            ),
            flush=True,
        )

    result["arms"] = arms
    result["note"] = (
        "字节比 %.3f；int8 若没有接近这个比例的加速，说明该路径不是带宽受限的，"
        "权重侧省下的字节在 decode 上兑现不了。" % (int8_bytes / fp16_bytes)
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--sweep", action="store_true", help="扫若干 kernel 配置")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    payload = run(
        hidden=args.hidden,
        ffn=args.ffn,
        experts=args.experts,
        top_k=args.top_k,
        group_size=args.group_size,
        warmup=args.warmup,
        repeats=args.repeats,
        sweep=args.sweep,
    )
    print("\n" + payload["note"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
