"""bench_decode_attention — 隔离测量：decode attention 到底慢在哪、哪条路能用。

## 为什么要先做这个，而不是直接写 kernel

`EXP-p2-02` 的归因表里单项最大的是

    fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEffAttention)
    1.2110 ms/token = busy 的 30%，每层 75.69 µs（ctx=1024）

实测确认它来自 **SDPA**：OLMoE 加载后 `config._attn_implementation == "sdpa"`，
模块是 `OlmoeSdpaAttention` —— 同一个文件里**另一个类**，不是 eager 的
`OlmoeAttention`（我第一版读错了类，所以一度以为"fmha 不该存在"）。
归因是对的；但"必须用 SDPA"不是。

## 测什么

同一个 decode 形状下对照所有合理实现，每个都报**延迟 + 能否进图 + 换长度后是否还算对**。

    A  eager_math       HF 的 eager 实现（= `attn_implementation="eager"` 的效果）
    B  sdpa_float4d     当前实际走的路：SDPA + 4D float mask
    C  sdpa_bool4d      SDPA + 4D bool mask
    D  sdpa_narrow      SDPA，k/v 裁到已填充长度，**完全不要 mask**
    F  eager_narrow     matmul→softmax→matmul，裁到已填充长度，不要 mask（理想实现的形态）
    E  sdpa_flash       强制 FLASH_ATTENTION 后端（多数情况下不可用）

## ★ 为什么"能捕获"这个判据单独不够（第一版就是错的）

第一版只做 `with torch.cuda.graph(...): fn()`，于是 arm D 报了 `capturable: true`。
**那个报告是危险的**：D 的 shape 由长度决定，捕获时是什么长度就永远是什么长度；
换一个长度重放会拿到**上一个长度的答案** —— 不报错，静默算错。

所以这里做的是**两段式**检查：

    1. 在长度 N 捕获一张图；
    2. 把真实长度改成 M > N，重放；
    3. 把重放的结果与 M 的正确答案比。

读"当前长度"的实现（B/C 靠原地改 mask；F/未来的自研 kernel 靠读 device scalar）
重放时给出 M 的答案 → `correct_after_length_change: true`；
把长度烘进 shape 的实现（D）给出 N 的答案 → **false**。

这正是自研 kernel 必须保住的性质，所以它是必报项，不是附注。

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
    med = statistics.median(ordered)
    return {
        "median_ms": round(med, 5),
        "min_ms": round(ordered[0], 5),
        "mad_ms": round(statistics.median([abs(s - med) for s in ordered]), 6),
        "repeats": repeats,
    }


def _profile_kernels(fn, *, iters: int, torch, top: int = 6) -> list[dict[str, Any]]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    rows = [
        {
            "name": e.key[:110],
            "self_ms_per_call": round(e.self_device_time_total / iters / 1000.0, 5),
            "count": round(e.count / iters, 2),
        }
        for e in prof.key_averages()
        if e.self_device_time_total > 0
    ]
    rows.sort(key=lambda r: r["self_ms_per_call"], reverse=True)
    return rows[:top]


def replay_after_length_change(
    fn, *, length_state: dict[str, int], capture_at: int, replay_at: int, torch
) -> dict[str, Any]:
    """Capture at one length, replay at another; the caller then checks the answer.

    This is the check that separates a real decode kernel from a shape-baked one.
    `fn` must stash its output tensor (the caller reads it back after replay), so
    that what is compared is what the **graph** wrote, not a fresh eager call.
    """
    try:
        length_state["filled"] = capture_at
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn()
        stream.synchronize()

        # 换长度：B/C 的 mask 是设备张量，原地改就能影响重放；
        # D/F 读的是 host 变量，改了也影响不到已捕获的图 —— 这正是要暴露的差别。
        length_state["filled"] = replay_at
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        return {"capturable": True}
    except Exception as exc:  # noqa: BLE001
        return {
            "capturable": False,
            "correct_after_length_change": False,
            "note": "报错（响亮失败，至少不是静默错）：%s" % str(exc)[:130],
        }


def _interleaved(
    arms: dict[str, Any], *, rounds: int, warmup: int, iters: int, torch
) -> dict[str, Any]:
    """Round-robin the arms so interference cannot be mistaken for an arm difference.

    ★ 为什么这里必须交错：这台机器是共享的，`pareto_offload.py`（另一个项目）
    会随时抢 GPU。顺序测每个臂时，后一个臂继承了前一个的时钟/L2/争用状态 ——
    上一轮 grouped GEMM 就因为这个问题得到过同一个 config 相差 29% 的结果。
    交错执行让每个臂在每一轮经历相同的邻居，另一个人跑不跑对所有臂是同等的。

    报 median / MAD / 逐轮 wins：在所有轮里都赢，比"总中位数最小"强得多。
    """
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {k: [] for k in arms}
    ranks: list[dict[str, int]] = []
    for _ in range(rounds):
        round_ms: dict[str, float] = {}
        for name, fn in arms.items():
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0 / iters
            samples[name].append(ms)
            round_ms[name] = ms
        order = sorted(round_ms, key=lambda n: round_ms[n])
        ranks.append({n: order.index(n) + 1 for n in arms})
    report: dict[str, Any] = {}
    for name, values in samples.items():
        ordered = sorted(values)
        med = statistics.median(ordered)
        report[name] = {
            "median_ms": round(med, 5),
            "min_ms": round(ordered[0], 5),
            "max_ms": round(ordered[-1], 5),
            "mad_ms": round(statistics.median([abs(v - med) for v in values]), 6),
            "wins": sum(1 for r in ranks if r[name] == 1),
            "rounds": len(values),
            "samples_ms": [round(v, 5) for v in values],
        }
    noise = statistics.median([v["mad_ms"] for v in report.values()])
    return {
        "method": "interleaved round-robin",
        "rounds": rounds,
        "iters": iters,
        "per_arm": report,
        "noise_floor_ms": round(noise, 6),
        "caveat": "可分辨的最小差异约等于 MAD；比 MAD 小的差值不要当成结论",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kv-len", type=int, default=1056)
    ap.add_argument("--filled", type=int, default=33)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-length-check", action="store_true", help="跳过换长度检查（更快）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2

    dev, dt = "cuda", {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    b, h, d, L, filled0 = (
        OLMOE_ATTN["batch"],
        OLMOE_ATTN["heads"],
        OLMOE_ATTN["head_dim"],
        args.kv_len,
        args.filled,
    )
    if not (0 < filled0 <= L):
        print("--filled 必须在 (0, kv-len] 内", file=sys.stderr)
        return 2
    filled1 = min(L, filled0 * 2 + 1)

    torch.manual_seed(0)
    q = torch.randn(b, h, 1, d, dtype=dt, device=dev) * 0.1
    k = torch.randn(b, h, L, d, dtype=dt, device=dev) * 0.1
    v = torch.randn(b, h, L, d, dtype=dt, device=dev) * 0.1
    scale = 1.0 / math.sqrt(d)
    min_dtype = torch.finfo(dt).min

    # 原地可改的 mask（与 session 每步 fill_ 的做法一致）+ 可变的 host 长度
    mask_f = torch.full((b, 1, 1, L), min_dtype, dtype=dt, device=dev)
    mask_b = torch.zeros((b, 1, 1, L), dtype=torch.bool, device=dev)
    out_keep: dict[str, Any] = {}

    def set_filled(n: int) -> None:
        mask_f.fill_(min_dtype)
        mask_f[..., :n] = 0.0
        mask_b.zero_()
        mask_b[..., :n] = True

    length_state = {"filled": filled0}
    set_filled(filled0)

    def eager_math():
        w = torch.matmul(q, k.transpose(2, 3)) * scale
        w = w + mask_f
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(dt)
        return torch.matmul(w, v)

    def sdpa_float4d():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_f, scale=scale)

    def sdpa_bool4d():
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask_b, scale=scale)

    def sdpa_narrow():
        n = length_state["filled"]
        return F.scaled_dot_product_attention(q, k[:, :, :n], v[:, :, :n], scale=scale)

    def eager_narrow():
        n = length_state["filled"]
        w = torch.matmul(q, k[:, :, :n].transpose(2, 3)) * scale
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(dt)
        return torch.matmul(w, v[:, :, :n])

    arms = {
        "A_eager_math": eager_math,
        "B_sdpa_float4d": sdpa_float4d,
        "C_sdpa_bool4d": sdpa_bool4d,
        "D_sdpa_narrow": sdpa_narrow,
        "F_eager_narrow": eager_narrow,
    }

    def correct_for(n: int):
        return F.scaled_dot_product_attention(
            q[:, :, :1].double(), k[:, :, :n].double(), v[:, :, :n].double(), scale=scale
        )

    result: dict[str, Any] = {
        "shape": dict(OLMOE_ATTN, kv_len=L, filled=filled0, dtype=args.dtype),
        "length_check": {
            "capture_at": filled0,
            "replay_at": filled1,
            "enabled": not args.no_length_check,
        },
        "kv_bytes_needed": 2 * h * filled0 * d * 2,
        "kv_bytes_full_scan": 2 * h * L * d * 2,
        "scan_waste_x": round(L / filled0, 2),
        "reference": "fp64 SDPA over the filled prefix",
        "arms": {},
    }

    # 先确认每个臂算的是同一件事 —— 快的错实现不是结果
    with torch.no_grad():
        ref0 = correct_for(filled0)
        for name, fn in arms.items():
            try:
                result["arms"][name] = {
                    "max_abs_err_vs_fp64": float((fn().double() - ref0).abs().max())
                }
            except Exception as exc:  # noqa: BLE001
                result["arms"][name] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

    live = {n: f for n, f in arms.items() if "error" not in result["arms"][n]}
    interleaved = _interleaved(
        live, rounds=args.repeats * 2, warmup=args.warmup, iters=args.iters, torch=torch
    )
    result["interleaved"] = {k: v for k, v in interleaved.items() if k != "per_arm"}
    for name in live:
        result["arms"][name]["speed"] = interleaved["per_arm"][name]

    for name, fn in arms.items():
        entry = result["arms"][name]
        if "error" in entry:
            continue
        if not args.no_length_check:
            with torch.no_grad():
                want = correct_for(filled1)

            # 为了让"图写进哪块缓冲区"这件事显式，这里用 fn 自己的返回张量：
            # 捕获时 fn 把结果写进 out_keep[name]，重放后同一个张量里是新结果。
            def captured_fn(_fn=fn, _name=name):
                out = _fn()
                out_keep[_name] = out
                return out

            entry["length_change"] = replay_after_length_change(
                captured_fn,
                length_state=length_state,
                capture_at=filled0,
                replay_at=filled1,
                torch=torch,
            )
            # 重放之后 out_keep[name] 里是**图算出来的**东西
            if name in out_keep:
                entry["length_change"]["err_of_graph_output"] = float(
                    (out_keep[name].double() - want).abs().max()
                )
                entry["length_change"]["correct_after_length_change"] = bool(
                    entry["length_change"]["err_of_graph_output"] < 1e-2
                )
            length_state["filled"] = filled0
            set_filled(filled0)
        if args.profile:
            entry["kernels"] = _profile_kernels(fn, iters=20, torch=torch)

    if not args.profile:
        for name in ("B_sdpa_float4d", "A_eager_math", "F_eager_narrow"):
            if name in arms and "error" not in result["arms"].get(name, {}):
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
