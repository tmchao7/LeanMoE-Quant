#!/usr/bin/env python3
"""profile_graph_scopes — 把 decode 步的 GPU 时间**按模块**归因，并给出**完整** kernel 表。

## 为什么需要它（`EXP-p4c-01` §5.4 → `EXP-p4c-02`）

`EXP-p4c-01` §5.4 的 kernel 桶说"elementwise/copy 长尾 ≈0.88 ms、~330 个 kernel"，
但 runner 的 payload 只存 top-25 行 —— 足以看见最大的几个（`direct_copy`、`reduce_kernel`、
`elementwise_kernel`），**不足以决定怎么修**：修法取决于这些 kernel 是"某个模块里的
一长串 torch 算子"，还是"Hermes 型的隐式 cast"，两者对应的动作完全不同。

## 关键限制（决定了本工具怎么做）

**CUDA Graph 重放期间没有 Python 帧** —— 图是一整块，`torch.profiler.record_function`
的作用域只能记在 eager 执行上。所以：

    eager + 作用域   →  按模块归因（谁花了时间）        ← 本工具的主产出
    graph 重放       →  完整 kernel 表（与单元格同一口径）← 次产出，用来核对两者的 kernel 组成

两条路径用**同一个 `StaticCacheDecodeSession`**（同 StaticCache、同 4D mask、同 buffer 形状），
所以"eager 的算子组成"与"图里的算子组成"是同一批算子 —— 差别只在 host 间隔（`EXP-p0-04` 的结论）。

## 判据（跑之前写）

    K1  eager 与 graph 两条路径的 kernel 名集合必须高度重合（前 8 名按 self time 排序一致），
        否则"用 eager 归因解释图"不成立，本工具的结论作废
    D   报每个作用域的 GPU self time 与占比；**长尾必须落到具体作用域**，
        落不下去（某作用域名下 >10% 的时间不可归属）就写"无法归属"，不猜

用法：

    python -m tools.profile_graph_scopes --model "$M" --bit-width w8a16 --attn kernel \
        --ctx 1024 --steps 20 --json results/EXP-p4c-02-elementwise-tail/scopes_w8a16_kernel.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 归因到 decoder 层的哪些子模块（顺序 = 前向顺序，报告里按它排）
SCOPE_MODULES = (
    "input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "post_attention_layernorm",
    "mlp",
    "self_attn",
)


def _wrap_scopes(model: Any, torch: Any) -> list[str]:
    """给每层的目标子模块的 forward 套一层 `record_function`。**必须在 capture 之前**。"""
    wrapped: list[str] = []
    layers = model.model.layers

    def make(prefix: str, name: str, fn: Any) -> Any:
        label = "%s.%s" % (prefix, name)

        def forward(*args: Any, **kwargs: Any) -> Any:
            with torch.profiler.record_function(label):
                return fn(*args, **kwargs)

        return forward

    for i, layer in enumerate(layers):
        for name in SCOPE_MODULES:
            mod = getattr(layer, name, None)
            if mod is None or not hasattr(mod, "forward"):
                continue
            mod.forward = make("L", name, mod.forward)  # 按**模块名**聚合（16 层合并）
            wrapped.append(name)
    # 最后一层 norm + lm_head 单独记（不在 layer 里）
    if hasattr(model.model, "norm"):
        model.model.norm.forward = make("T", "final_norm", model.model.norm.forward)
        wrapped.append("final_norm")
    if hasattr(model, "lm_head"):
        model.lm_head.forward = make("T", "lm_head", model.lm_head.forward)
        wrapped.append("lm_head")
    return wrapped


def _probe(torch: Any, session: Any, *, steps: int, eager: bool) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

    for _ in range(3):
        (session.decode_eager() if eager else session.replay())
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            (session.decode_eager() if eager else session.replay())
        torch.cuda.synchronize()
    return {"profiler": prof}


def _rows(prof: Any, *, steps: int) -> list[dict[str, Any]]:
    out = []
    for e in prof.key_averages():
        tot = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        if tot <= 0.0:
            continue
        out.append(
            {
                "name": e.key,
                "self_ms_per_step": tot / steps / 1000.0,
                "count_per_step": e.count / float(steps),
            }
        )
    out.sort(key=lambda r: -r["self_ms_per_step"])
    return out


def _bucket(name: str) -> str:
    n = name.lower()
    if "attn_split" in n or "fmha" in n or "flash" in n:
        return "attention_core"
    if "_gate_up_act_kernel" in n or "_down_sum_kernel" in n:
        return "moe_gemv"
    if "_dequant_gemm_kernel" in n:
        return "projection_fused"
    if "gemvx" in n or "gemm" in n or "cutlass" in n:
        return "gemm_other"
    if "copy" in n:
        return "copy"
    if "reduce" in n or "mean" in n:
        return "reduce"
    if "elementwise" in n:
        return "elementwise"
    if "cub::" in n or "index" in n:
        return "routing_index"
    return "other"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bit-width", default="w8a16")
    ap.add_argument("--attn", default="kernel", choices=["sdpa", "eager", "kernel"])
    ap.add_argument("--dequant", default="fused")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("profile_graph_scopes 需要 CUDA", file=sys.stderr)
        return 2

    from benchmarks.cudagraph import StaticCacheDecodeSession  # noqa: PLC0415
    from benchmarks.runner import _load_model, quantize_for_cfg  # noqa: PLC0415
    from lemoq.kernels.moe_static import replace_moe_blocks  # noqa: PLC0415

    quantized = args.bit_width != "fp16"
    quant_cfg = {
        "bit_width": args.bit_width,
        "group_size": 128,
        "dequant": args.dequant if quantized else "none",
        "moe": "grouped",
        "execution": "cudagraph",
        "ctx": args.ctx,
        "attn": args.attn,
    }
    load_attn = None if args.attn == "kernel" else args.attn
    model, _dtype = _load_model(args.model, "float16", "cuda", attn=load_attn)
    moe_replaced = replace_moe_blocks(model, impl="grouped")
    torch.cuda.synchronize()
    qreport = quantize_for_cfg(model, quant_cfg) if quantized else None
    torch.cuda.synchronize()
    attn_replaced = 0
    if args.attn == "kernel":
        from lemoq.kernels.attn_decode import replace_attention_cores  # noqa: PLC0415
        from lemoq.kernels.attn_decode import warm as warm_attn  # noqa: PLC0415

        attn_replaced = replace_attention_cores(model, backend="triton")
        heads = int(model.config.num_attention_heads)
        head_dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // heads))
        warm_attn(
            heads=heads,
            head_dim=head_dim,
            kv_len=args.ctx + args.max_new_tokens,
            filled=args.ctx + args.max_new_tokens,
            dtype=torch.float16,
            device="cuda",
        )
        torch.cuda.synchronize()

    ids = torch.full((1, args.ctx), 1, dtype=torch.long, device="cuda")
    session = StaticCacheDecodeSession(
        model, ids, ctx=args.ctx, max_new_tokens=args.max_new_tokens, device="cuda"
    )
    torch.cuda.synchronize()

    # ---- (1) eager + 作用域：按模块归因 ----
    wrapped = _wrap_scopes(model, torch)
    eager = _probe(torch, session, steps=args.steps, eager=True)
    eager_rows = _rows(eager["profiler"], steps=args.steps)
    # 作用域事件：名字就是我们打的 label（"L.q_proj" / "T.lm_head"）
    scope_rows = [r for r in eager_rows if r["name"].startswith(("L.", "T."))]
    kernel_rows = [r for r in eager_rows if not r["name"].startswith(("L.", "T."))]

    # ---- (2) graph 重放：完整 kernel 表（与单元格同口径）----
    session.reset()
    session.prefill()
    torch.cuda.synchronize()
    cap = session.capture(warmup=5)
    graph = _probe(torch, session, steps=args.steps, eager=False)
    graph_rows = _rows(graph["profiler"], steps=args.steps)

    # K1：两条路径的 kernel 名集合必须高度重合（前 8 名一致）
    top_eager = [r["name"] for r in kernel_rows[:8]]
    top_graph = [r["name"] for r in graph_rows[:8]]
    overlap = len(set(top_eager) & set(top_graph))

    def _sum(rows: list[dict[str, Any]]) -> float:
        return sum(r["self_ms_per_step"] for r in rows)

    eager_total = _sum(kernel_rows)
    graph_total = _sum(graph_rows)
    payload = {
        "tool": "tools/profile_graph_scopes.py",
        "config": quant_cfg,
        "shape": {
            "ctx": args.ctx,
            "max_new_tokens": args.max_new_tokens,
            "steps_profiled": args.steps,
            "batch": 1,
        },
        "setup": {
            "moe_blocks_replaced": moe_replaced,
            "attention_cores_replaced": attn_replaced,
            "quantized_targets": (qreport or {}).get("replaced_per_target"),
            "scopes_wrapped": sorted(set(wrapped)),
        },
        "k1_top8_kernel_overlap": overlap,
        "eager": {
            "gpu_ms_per_step_total": eager_total,
            "kernels_per_step": sum(r["count_per_step"] for r in kernel_rows),
            "by_scope_ms": {r["name"]: r["self_ms_per_step"] for r in scope_rows},
            "kernels": kernel_rows,
        },
        "graph": {
            "gpu_ms_per_step_total": graph_total,
            "kernels_per_step": sum(r["count_per_step"] for r in graph_rows),
            "buckets_ms": {},
            "kernels": graph_rows,
        },
        "capture": cap,
    }
    # 分桶（graph 口径）
    buckets: dict[str, float] = {}
    for r in graph_rows:
        buckets[_bucket(r["name"])] = buckets.get(_bucket(r["name"]), 0.0) + r["self_ms_per_step"]
    payload["graph"]["buckets_ms"] = {
        k: round(v, 4) for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])
    }

    # ---- 打印 ----
    print("=== 配置: %s ===" % json.dumps(quant_cfg, ensure_ascii=False))
    print("作用域已挂: %s" % ", ".join(payload["setup"]["scopes_wrapped"]))
    print(
        "\n[eager + 作用域] GPU %.3f ms/步 | 归因（作用域 %d 个，kernel %d 个）:"
        % (eager_total, len(scope_rows), len(kernel_rows))
    )
    for r in scope_rows[:20]:
        print(
            "    %-24s %8.4f ms  %5.1f%%"
            % (r["name"], r["self_ms_per_step"], 100 * r["self_ms_per_step"] / eager_total)
        )
    attributed = _sum(scope_rows)
    print(
        "    %-24s %8.4f ms  %5.1f%%" % ("<作用域合计>", attributed, 100 * attributed / eager_total)
    )
    print(
        "    %-24s %8.4f ms  %5.1f%%"
        % (
            "<未归属 kernel>",
            eager_total - attributed,
            100 * (eager_total - attributed) / eager_total,
        )
    )
    print(
        "\n[graph 重放] GPU %.3f ms/步 | kernel %d/步 | 桶:"
        % (graph_total, payload["graph"]["kernels_per_step"])
    )
    for k, v in payload["graph"]["buckets_ms"].items():
        print("    %-18s %8.4f ms  %5.1f%%" % (k, v, 100 * v / graph_total))
    print("\nK1 前 8 kernel 重合 %d/8 | eager top: %s" % (overlap, top_eager[:3]))
    print("                          graph top: %s" % (top_graph[:3],))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n写入 %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
