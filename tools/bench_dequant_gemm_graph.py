#!/usr/bin/env python3
"""bench_dequant_gemm_graph — **在图内**扫 fused dequant-GEMM 的 config（`EXP-p4c-01` §5.4 的后续）。

## 为什么必须进图（这一步的立论）

`EXP-p2-01` §6.4 的隔离扫描（7 轮 round-robin × 200 次调用）选出的赢家是 `B4K512w4`
（0.1338 ms/token/层）—— 但那个测量**被 Triton launcher 主导**：同节实测 launcher 地板是
3 参数 16.0 µs、20 参数 31.8 µs，而这个 kernel 本体读 4 MB int8 + 8 KB 输出只需 ~2.6 µs。
也就是说，**隔离扫描量的是 Python launcher，不是 kernel**。

图里没有 Python launcher。`EXP-p4c-01` §5.4 的 profiler 显示这 64 次调用共 **0.5245 ms**，
即 **8.2 µs/次**（占 w8a16 步的 20.6%）—— 于是"哪个 config 最快"必须在图里重测：
隔离扫描的排名对图**没有约束力**。

## 它测什么

把 OLMoE attention 的投影形状搬进一张 micro-graph：16 层 × 4 个投影，每个
(N=2048, K=2048)、8-bit、g=128、M=1、输出 fp16 → 一张图里 **64 次** `dequant_gemm_fused`。

每个 config 各捕获一张图，然后**逐轮交错重放**（同一进程、同一份权重与输入）：
共用一张图重放会把"上一次的 L2 余温"算进来，交错才能把 config 之间的差异与顺序偏差分开。

## 判据（跑之前写）

    K1  每个 config 的输出必须与默认 config **逐位相同**（同一套数学，只是分块不同）
        —— 不同就是 kernel 的 bug，先修 kernel 再谈性能
    D   per-call GPU 时间（p50）：默认 config 应当复现 §5.4 的 ~8.2 µs；
        赢家必须比默认快 >5%（金丝雀级的差异没有意义）

用法：

    python -m tools.bench_dequant_gemm_graph --json results/.../dequant_gemm_graph.json
    python -m tools.bench_dequant_gemm_graph --configs B4K256w2 B8K256w4 B16K128w4 ...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# config 名 <-> dict：'B<n>K<k>w<warps>s<stages>'
# ---------------------------------------------------------------------------
def parse_config(name: str) -> dict[str, int]:
    """`B8K256w4s3` -> {"BLOCK_N": 8, "BLOCK_K": 256, "num_warps": 4, "num_stages": 3}。"""
    import re  # noqa: PLC0415

    m = re.fullmatch(r"B(\d+)K(\d+)w(\d+)(?:s(\d+))?", name)
    if not m:
        raise ValueError("config 名要写成 B<block_n>K<block_k>w<warps>[s<stages>]：%r" % name)
    cfg = {
        "BLOCK_N": int(m.group(1)),
        "BLOCK_K": int(m.group(2)),
        "num_warps": int(m.group(3)),
        "num_stages": int(m.group(4)) if m.group(4) else 3,
    }
    return cfg


def config_name(cfg: dict[str, int]) -> str:
    return "B%dK%dw%d%s" % (
        cfg["BLOCK_N"],
        cfg["BLOCK_K"],
        cfg["num_warps"],
        ("s%d" % cfg["num_stages"]) if "num_stages" in cfg else "",
    )


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "p99_ms": ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--projections", type=int, default=4, help="每层几个投影（q/k/v/o = 4）")
    ap.add_argument("--dim", type=int, default=2048, help="OLMoE: hidden = N = K = 2048")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8])
    ap.add_argument("--rounds", type=int, default=15, help="交错重放的轮数")
    ap.add_argument("--configs", nargs="*", default=None, help="config 名列表（默认一小片空间）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("bench_dequant_gemm_graph 需要 CUDA 设备（本机没有）", file=sys.stderr)
        return 2

    from lemoq.kernels._dequant_triton import (  # noqa: PLC0415
        DEQUANT_GEMM_CONFIGS,
        dequant_gemm_fused,
    )
    from lemoq.quant import rtn as qrtn  # noqa: PLC0415

    dev = "cuda"
    n_groups = args.dim // args.group_size
    names = args.configs or [
        "B4K256w2s3",  # ← 生产默认（EXP-p2-01 隔离扫描的赢家）
        "B4K128w2s3",
        "B8K256w4s3",
        "B8K128w4s3",
        "B16K256w4s3",
        "B16K128w8s3",
        "B32K128w8s3",
        "B32K256w8s2",
    ]
    configs = [(n, parse_config(n)) for n in names]

    # ---- 权重：真实 RTN + 打包口径（与 QuantLinear 的 8-bit 路径一致）----
    print(
        "构造 %d 层 × %d 投影 的权重（N=K=%d, %d-bit, g=%d）"
        % (args.layers, args.projections, args.dim, args.bits, args.group_size)
    )
    torch.manual_seed(0)
    packed, scales = [], []
    for _ in range(args.layers * args.projections):
        w = (torch.randn(args.dim, args.dim, device=dev) * 0.02).float()
        q = qrtn.quantize_rtn(w, bits=args.bits, group_size=args.group_size)
        codes = q["codes_padded"]
        if args.bits == 8:
            qw = codes.to(torch.int8).reshape(args.dim, args.dim).contiguous()
        else:
            from lemoq.quant import pack as qpack  # noqa: PLC0415

            flat = codes.reshape(int(codes.numel()) // args.dim, args.dim)
            qw = qpack.pack_4bit(flat).reshape(args.dim, -1).contiguous()
        packed.append(qw)
        scales.append(q["scale"].half().reshape(args.dim, n_groups).contiguous())
    x = (torch.randn(1, args.dim, device=dev, dtype=torch.float16) * 0.5).contiguous()
    total_bytes = sum(
        p.numel() * p.element_size() + s.numel() * s.element_size() for p, s in zip(packed, scales)
    )

    # ---- 参考输出（默认 config，eager 调用一次）----
    ref = torch.cat(
        [
            dequant_gemm_fused(
                x, p, s, group_size=args.group_size, bits=args.bits, out_dtype=torch.float16
            )
            for p, s in zip(packed, scales)
        ],
        dim=1,
    ).clone()

    results: dict[str, dict[str, Any]] = {}
    graphs: dict[str, Any] = {}
    outs: dict[str, Any] = {}

    for name, cfg in configs:
        try:
            # warm（JIT + 首次 launch）必须在 capture 之前
            dequant_gemm_fused(
                x,
                packed[0],
                scales[0],
                group_size=args.group_size,
                bits=args.bits,
                out_dtype=torch.float16,
                config=cfg,
            )
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            print("  %-14s 编译失败：%s" % (name, exc))
            continue

        # ★ 不在图里做 `out[...] = ...` 拷贝：那会往图里塞 64 个额外的 memcpy 节点，
        #   把测量污染成"kernel + memcpy"。捕获时直接留下每次调用返回的张量
        #   （图内 `torch.empty` 来自图私有池，重放后指针稳定）—— 与生产路径一致：
        #   QuantLinear 也是每次调用各自分配输出。
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for p, s in zip(packed, scales):
                    dequant_gemm_fused(
                        x,
                        p,
                        s,
                        group_size=args.group_size,
                        bits=args.bits,
                        out_dtype=torch.float16,
                        config=cfg,
                    )
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ys = [
                dequant_gemm_fused(
                    x,
                    p,
                    s,
                    group_size=args.group_size,
                    bits=args.bits,
                    out_dtype=torch.float16,
                    config=cfg,
                )
                for p, s in zip(packed, scales)
            ]
        graph.replay()
        torch.cuda.synchronize()
        out = torch.cat(ys, dim=1).clone()
        # K1：与默认 config 比 **PARITY_GATE**（不是逐位 —— 换 BLOCK_K 会重排 fp32 累加）
        from lemoq.kernels.reference import tolerance_report  # noqa: PLC0415

        rep = tolerance_report(ref, out)
        graphs[name] = graph
        outs[name] = out
        results[name] = {
            "bitwise_equal_to_default": bool(torch.equal(out, ref)),
            "parity_gate_pass": rep["allclose_atol_1e-3"],
            "max_abs_err": rep["max_abs_err"],
        }
        print(
            "  %-14s captured | 逐位相同=%-5s | PARITY_GATE(1e-3)=%-5s | max|Δ|=%.3e"
            % (name, bool(torch.equal(out, ref)), rep["allclose_atol_1e-3"], rep["max_abs_err"])
        )

    if not graphs:
        print("没有任何 config 捕获成功", file=sys.stderr)
        return 2

    # ---- 逐轮交错重放 ----
    order = list(graphs)
    samples: dict[str, list[float]] = {n: [] for n in order}
    for r in range(args.rounds):
        for name in order:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graphs[name].replay()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) * 1000.0)
        del r

    calls = args.layers * args.projections
    print(
        "\n%-14s %10s %10s %10s %12s %10s %8s"
        % ("config", "p50(ms)", "min(ms)", "µs/call", "达到带宽", "MB/call", "wins")
    )
    winners = dict.fromkeys(order, 0)
    for r in range(args.rounds):
        best = min(order, key=lambda n: samples[n][r])
        winners[best] += 1
    base = "B4K256w2s3" if "B4K256w2s3" in samples else order[0]
    # ★ 基准必须用 base 的 **p50**，不能用它的第 0 轮原始样本。
    #   这就是 AGENTS.md §0 点名过的那个坑（E7 的 1.1927 vs 1.1911：草稿拿了
    #   金丝雀的**首值** 3.0084 当分母，采用值是末值 3.0124）。在这里它的形态是
    #   "单样本 vs 对方的中位数"，而且第 0 轮紧跟 capture，最容易带上残留的
    #   warm-up / allocator 状态。D 判据（赢家必须比默认快 >5%）读的就是这一列。
    base_p50 = _summarize(samples[base])["p50_ms"]
    rows = []
    for name in order:
        s = _summarize(samples[name])
        us = s["p50_ms"] * 1000.0 / calls
        gbs = total_bytes / (s["p50_ms"] / 1000.0) / 1e9
        mb = total_bytes / calls / 1e6
        results[name].update(
            {
                "p50_ms_for_%d_calls" % calls: s["p50_ms"],
                "min_ms": s["min_ms"],
                "us_per_call": us,
                "achieved_GBps": gbs,
                "MB_per_call": mb,
                "wins": winners[name],
                # 两侧同为 p50：`base_p50 / p50(name) − 1`。正 = 该配置比默认快。
                "vs_default_pct": (
                    round(100.0 * (base_p50 / s["p50_ms"] - 1.0), 3) if name != base else 0.0
                ),
                "vs_default_basis": "p50 vs p50（不是首轮单样本）",
            }
        )
        rows.append((name, s["p50_ms"], s["min_ms"], us, gbs, mb, winners[name]))
    for name, p50, mn, us, gbs, mb, w in sorted(rows, key=lambda r: r[1]):
        print("%-14s %10.4f %10.4f %10.3f %10.1f %10.2f %8d" % (name, p50, mn, us, gbs, mb, w))

    payload = {
        "shape": {
            "layers": args.layers,
            "projections": args.projections,
            "dim": args.dim,
            "group_size": args.group_size,
            "bits": args.bits,
            "calls_per_replay": calls,
        },
        "weights_bytes_per_replay": total_bytes,
        "rounds": args.rounds,
        "default_config": {
            k: DEQUANT_GEMM_CONFIGS["dequant_gemm"][k]
            for k in ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages")
        },
        "results": results,
    }
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n写入 %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
