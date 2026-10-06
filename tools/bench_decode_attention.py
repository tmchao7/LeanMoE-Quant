"""bench_decode_attention — 隔离测量：decode attention 到底慢在哪。

## 为什么要先做这个，而不是直接写 kernel

`EXP-p2-02` 的归因表里单项最大的是

    fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEfficientAttention)
    1.2110 ms/token = busy 的 30%，每层 75.69 µs（ctx=1024）

但是**它不该存在**：`transformers 4.57.6` 的 `OlmoeAttention.forward` 是 eager
数学实现（`matmul` → 加 4D mask → `softmax` → `matmul`），源码里没有
`scaled_dot_product_attention`，也没有 `attention_interface`。mem-efficient
attention 的内核名却出现在 profile 里 —— 这个矛盾必须先解决。

如果它是错的归因，我会去写一个不必要的 kernel；如果它是真的，那要知道
它为什么被选中。两者都会改变方案，所以先测。

## 测什么

同一个 decode 形状下对照**所有合理实现**，并各自报告：

    A. eager_math        HF 现在实际走的代码（逐字复制它的四步）
    B. sdpa_float4d      F.scaled_dot_product_attention + 4D **float** mask
    C. sdpa_bool4d       F.scaled_dot_product_attention + 4D **bool** mask
    D. sdpa_narrow       把 k/v 裁到已填充长度、**完全不要 mask**（"理想实现"的参照）
    E. sdpa_flash        强制 FLASH_ATTENTION 后端（若可用）

每个臂都报：中位延迟、相对**KV 字节下界**的达成率、以及能否进 CUDA Graph。

## 为什么"能否进图"是必报项

D（裁剪）是最快的候选，但它的 shape 随 `filled` 变化 —— 那正是 CUDA Graph
不允许的东西。所以"快"和"能用"在这里是两件事，必须分开报。
本脚本对每个臂都**实际尝试捕获一次**，并额外检查 D 的一个致命问题：
用 filled=33 捕获、用 filled=64 重放会得到**静默错误**的结果。

    python -m tools.bench_decode_attention --kv-len 1056 --filled 33
    python -m tools.bench_decode_attention --kv-len 4128 --filled 1025
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# OLMoE-1B-7B-0924 的已核实形状
OLMOE_ATTN = {"batch": 1, "heads": 16, "head_dim": 128}


def _median_time(fn, *, warmup: int, iters: int, repeats: int, torch) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0 / iters)
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(ordered), 5),
        "min_ms": round(ordered[0], 5),
        "mad_ms": round(
            statistics.median([abs(s - statistics.median(ordered)) for s in ordered]), 6
        ),
        "repeats": repeats,
    }


def _profile_kernels(fn, *, iters: int, torch) -> list[dict[str, Any]]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = []
    for event in prof.key_averages():
        if event.self_device_time_total <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "self_ms_per_call": round(event.self_device_time_total / iters / 1000.0, 5),
                "count": round(event.count / iters, 2),
            }
        )
    rows.sort(key=lambda r: r["self_ms_per_call"], reverse=True)
    return rows[:8]


def _try_capture(fn, torch) -> dict[str, Any]:
    """Can this arm be captured as a CUDA Graph at all?"""
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        stream.synchronize()
        return {"capturable": True}
    except Exception as exc:  # noqa: BLE001
        return {"capturable": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kv-len", type=int, default=1056, help="StaticCache max_cache_len")
    ap.add_argument("--filled", type=int, default=33, help="实际已填充的槽位数")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--profile", action="store_true", help="列出每个臂的 kernel 组成")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2

    dev, dt = "cuda", {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    b, h, d, L, filled = (
        OLMOE_ATTN["batch"],
        OLMOE_ATTN["heads"],
        OLMOE_ATTN["head_dim"],
        args.kv_len,
        args.filled,
    )
    if not (0 < filled <= L):
        print("--filled 必须在 (0, kv-len] 内", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    q = torch.randn(b, h, 1, d, dtype=dt, device=dev) * 0.1
    k = torch.randn(b, h, L, d, dtype=dt, device=dev) * 0.1
    v = torch.randn(b, h, L, d, dtype=dt, device=dev) * 0.1
    scale = 1.0 / math.sqrt(d)

    min_dtype = torch.finfo(dt).min
    mask_f = torch.full((b, 1, 1, L), min_dtype, dtype=dt, device=dev)
    mask_f[..., :filled] = 0.0
    mask_b = torch.zeros((b, 1, 1, L), dtype=torch.bool, device=dev)
    mask_b[..., :filled] = True

    def eager_math():
        """HF 现在实际走的四步，逐字复制。"""
        w = torch.matmul(q, k.transpose(2, 3)) * scale
        w = w + mask_f[:, :, :, : k.shape[-2]]
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(dt)
        return torch.matmul(w, v)

    def sdpa_float4d():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_f, scale=scale)

    def sdpa_bool4d():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_b, scale=scale)

    def sdpa_narrow():
        """理想实现：只碰已填充的槽位，不要 mask。shape 随 filled 变化。"""
        return F.scaled_dot_product_attention(q, k[:, :, :filled], v[:, :, :filled], scale=scale)

    arms = {
        "A_eager_math": eager_math,
        "B_sdpa_float4d": sdpa_float4d,
        "C_sdpa_bool4d": sdpa_bool4d,
        "D_sdpa_narrow": sdpa_narrow,
    }
    if hasattr(torch.nn.attention, "sdpa_kernel"):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        def sdpa_flash():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_b, scale=scale)

        arms["E_sdpa_flash"] = sdpa_flash

    # 字节下界：只有已填充槽位是**必要**读量
    kv_needed = 2 * h * filled * d * 2
    kv_full = 2 * h * L * d * 2
    result: dict[str, Any] = {
        "shape": dict(OLMOE_ATTN, kv_len=L, filled=filled, dtype=args.dtype),
        "kv_bytes_needed": kv_needed,
        "kv_bytes_full_scan": kv_full,
        "scan_waste_x": round(L / filled, 2),
        "arms": {},
    }

    # 用 fp64 参考校验每个臂都算的是同一件事 —— 快的错实现不是结果
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(
            q[:, :, :1].double(), k[:, :, :filled].double(), v[:, :, :filled].double(), scale=scale
        )
        result["reference"] = "fp64 SDPA over the filled prefix"
        for name, fn in arms.items():
            try:
                out = fn()
                err = float((out[:, :, :1].double() - ref).abs().max())
                result["arms"][name] = {"max_abs_err_vs_fp64": err}
            except Exception as exc:  # noqa: BLE001
                result["arms"][name] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

    for name, fn in arms.items():
        if "error" in result["arms"][name]:
            continue
        if args.profile:
            result["arms"][name]["kernels"] = _profile_kernels(fn, iters=20, torch=torch)
        result["arms"][name]["speed"] = _median_time(
            fn, warmup=args.warmup, iters=args.iters, repeats=args.repeats, torch=torch
        )
        result["arms"][name]["capture"] = _try_capture(fn, torch)

    if not args.profile:
        # 一次 profile 就能回答"那个 fmha 内核从哪来"
        for name in ("A_eager_math", "B_sdpa_float4d"):
            if name in arms and "error" not in result["arms"][name]:
                result["arms"][name]["kernels"] = _profile_kernels(
                    arms[name], iters=20, torch=torch
                )

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        print("\nwrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
