#!/usr/bin/env python3
"""bench_quant_linear — 隔离测量 `QuantLinear`（**separate 臂**）的反量化税。

## 为什么需要它

`EXP-p4b-01` 记录 §2.7 的单元格实测：量化格（`w8a16 × grouped × eager`）比 fp16 慢
（0.824×/0.802×）。而隔离微基准显示**量化 expert kernel 本身快 17%** —— 所以那 +2 ms
不能归给 experts。剩下两个候选：

    (a) attention 被量化，走的是 `QuantLinear` 的 **separate 臂**：
        每次 forward 先 `dequantized_weight()` 物化 fp16 权重，再做 `F.linear`
        —— 多出"写一遍 + 读一遍"权重，外加一长串 elementwise kernel 的 launch；
    (b) 别的东西（路由、dtype 转换…）。

本工具在 OLMoE 的 attention 形状上，**同进程交错**测：

    fp16      : F.linear(x, w)
    separate  : QuantLinear.forward(x)      # 物化 fp16 权重再 GEMM

并把每次调用的时间和"多读写的权重字节"一起报出来。乘 16 层就是每 token 的量。
判读：若 separate 的成本与单元格上那 +2 ms 同量级 → §2.7 的归因成立。

用法：
    python -m tools.bench_quant_linear --hidden 2048 --kv-heads 512 --layers 16
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "p99_ms": ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))],
        "min_ms": ordered[0],
    }


def run(
    *,
    hidden: int = 2048,
    kv_heads_dim: int = 512,
    layers: int = 16,
    group_size: int = 128,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, Any]:
    import torch

    from lemoq.quant.linear import SchemeSpec
    from lemoq.quant.linear import QuantLinear as QL

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    torch.manual_seed(0)
    spec = SchemeSpec.from_flags("w8a16", group_size, targets=("attention",))
    # OLMoE 的四个 attention 投影（decode 时每次 1 个 token）
    shapes = [
        ("q_proj", hidden, hidden),
        ("k_proj", kv_heads_dim, hidden),
        ("v_proj", kv_heads_dim, hidden),
        ("o_proj", hidden, hidden),
    ]

    per_layer_fp16 = 0.0
    per_layer_sep = 0.0
    detail = []
    for name, out_features, in_features in shapes:
        linear = torch.nn.Linear(in_features, out_features, bias=False, device="cuda").half()
        quant = QL(linear, spec, name=name).cuda().eval()
        x = (torch.randn(1, 1, in_features, device="cuda") * 0.5).half()

        for _ in range(warmup):
            torch.nn.functional.linear(x, linear.weight)
            quant(x)
        torch.cuda.synchronize()

        fp16_samples: list[float] = []
        sep_samples: list[float] = []
        for _ in range(repeats):
            for fn, bucket in (
                (lambda: torch.nn.functional.linear(x, linear.weight), fp16_samples),
                (lambda: quant(x), sep_samples),
            ):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                bucket.append(start.elapsed_time(end))

        fp16 = _summarize(fp16_samples)
        sep = _summarize(sep_samples)
        per_layer_fp16 += fp16["p50_ms"]
        per_layer_sep += sep["p50_ms"]
        detail.append(
            {
                "name": name,
                "shape": [out_features, in_features],
                "fp16": fp16,
                "separate": sep,
                "separate_over_fp16": sep["p50_ms"] / fp16["p50_ms"],
            }
        )
        print(
            "%-8s %-14s fp16 %8.5f ms | separate %8.5f ms | x%.2f"
            % (
                name,
                "%dx%d" % (out_features, in_features),
                fp16["p50_ms"],
                sep["p50_ms"],
                sep["p50_ms"] / fp16["p50_ms"],
            ),
            flush=True,
        )

    payload = {
        "shapes": {"hidden": hidden, "kv_dim": kv_heads_dim, "layers": layers},
        "scheme": spec.name,
        "detail": detail,
        "per_layer": {"fp16_ms": per_layer_fp16, "separate_ms": per_layer_sep},
        "per_token_16_layers": {
            "fp16_ms": per_layer_fp16 * layers,
            "separate_ms": per_layer_sep * layers,
            "delta_ms": (per_layer_sep - per_layer_fp16) * layers,
        },
        "timing_method": "torch.cuda.Event 包住单次调用，同进程交错（round-robin）",
        "note": (
            "attention 的 separate 臂每次 forward 物化 fp16 权重（写+读一遍）并多做一串 "
            "elementwise kernel —— 这就是 §2.7 里不能归给 expert kernel 的那部分。"
        ),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--kv-heads", type=int, default=512, help="OLMoE: num_kv_heads*head_dim")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    payload = run(
        hidden=args.hidden,
        kv_heads_dim=args.kv_heads,
        layers=args.layers,
        group_size=args.group_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    pt = payload["per_token_16_layers"]
    print(
        "\n每 token（%d 层）：fp16 %.4f ms | separate %.4f ms | delta %+.4f ms"
        % (args.layers, pt["fp16_ms"], pt["separate_ms"], pt["delta_ms"])
    )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        print("wrote %s" % args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
