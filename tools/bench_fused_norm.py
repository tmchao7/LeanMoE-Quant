#!/usr/bin/env python3
"""bench_fused_norm — HF `OlmoeRMSNorm` vs `F.rms_norm`（**图内**、同进程交错 A/B）。

## 为什么要做这一刀（`EXP-p4c-02` §5 + `--stacks` 归因）

`EXP-p4c-02` 的完整 kernel 表把"elementwise/copy 长尾"拆成 13 个具名家族；按 CPU 栈归因
（`profile_graph_scopes --stacks`）显示最大的一条是 **`aten::_to_copy` ×150/步（0.185 ms）** ——
全是 dtype cast。顺着 cast 数下去就是 HF 的 `OlmoeRMSNorm.forward`：

    hidden_states.to(torch.float32)        # cast 1（fp16→fp32）
    .pow(2).mean(-1)                       # pow + reduce(MeanOps)
    hidden_states * torch.rsqrt(v + eps)   # rsqrt + mul
    return self.weight * ...to(input_dtype)  # cast 2（fp32→fp16）+ mul

**每个 norm 7 个 kernel**，而 OLMoE 每层有 4 个 norm（q_norm / k_norm / input / post）+
最后 1 个 = **65 个/步** ⇒ 仅这一处就是 ~455 个 kernel/步，正好解释长尾里
`reduce_kernel<MeanOps>` ×65、`rsqrt` ×65、几族 elementwise ×64–65、以及 130 次 cast。

`torch.nn.functional.rms_norm` 是**原生融合 kernel**（内部 fp32 累加、一次写回），
数学上就是同一个归一化。注意**唯一的数值差异**：HF 先 `x*rstd` 舍入到 fp16、再与 fp16 的
`weight` 相乘；融合 kernel 在 fp32 里乘完再舍入一次 ⇒ 每个元素可能差 1 ulp fp16。

## 判据（跑之前写）

    K1  两条路径的 logits KL：**必须 < 1e-3**（这是"同一个归一化"的量化判据），
        且 top-1 一致率 ≥ 0.99；否则**不采纳**（数值不可接受），数字只作记录
    D   图内 per-token p50：融合臂必须快 >3%（低于金丝雀量级就不值得动一行 HF 的语义）

用法：

    python -m tools.bench_fused_norm --model "$M" --bit-width w8a16 --attn kernel \
        --ctx 1024 --rounds 15 --json results/EXP-p4c-02-elementwise-tail/fused_norm_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import types
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _fused_norm_forward(self: Any, hidden_states: Any) -> Any:
    """`OlmoeRMSNorm.forward` 的替换：一次原生融合 kernel（内部 fp32 累加）。"""
    import torch.nn.functional as F  # noqa: PLC0415

    return F.rms_norm(
        hidden_states,
        (int(self.weight.shape[0]),),
        self.weight,
        float(self.variance_epsilon),
    )


def replace_rms_norms(model: Any) -> int:
    """实例级替换每个 `OlmoeRMSNorm.forward`（可逆：原方法存在 `_lemoq_original_norm`）。"""
    n = 0
    for module in model.modules():
        if type(module).__name__ == "OlmoeRMSNorm" and not hasattr(module, "_lemoq_original_norm"):
            module._lemoq_original_norm = module.forward
            module.forward = types.MethodType(_fused_norm_forward, module)
            n += 1
    return n


def restore_rms_norms(model: Any) -> int:
    n = 0
    for module in model.modules():
        if hasattr(module, "_lemoq_original_norm"):
            module.forward = module._lemoq_original_norm
            del module._lemoq_original_norm
            n += 1
    return n


def _summarize(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "p99_ms": ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--bit-width", default="w8a16")
    ap.add_argument("--attn", default="kernel", choices=["sdpa", "eager", "kernel"])
    ap.add_argument("--dequant", default="fused")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--profile",
        action="store_true",
        help="额外给每条臂的图重放做一次 CUDA profile，打印各自前 10 个 kernel（解释为什么快/慢）",
    )
    args = ap.parse_args(argv)

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("bench_fused_norm 需要 CUDA", file=sys.stderr)
        return 2

    from benchmarks.cudagraph import (  # noqa: PLC0415
        StaticCacheDecodeSession,
        compare_logits,
    )
    from benchmarks.runner import (  # noqa: PLC0415
        _load_model,
        quant_spec_for_cfg,
        quantize_for_cfg,
    )
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
    qspec = quant_spec_for_cfg(quant_cfg) if quantized else None
    replace_moe_blocks(model, impl="grouped", quant=qspec)
    torch.cuda.synchronize()
    if quantized:
        quantize_for_cfg(model, quant_cfg)
    if args.attn == "kernel":
        from lemoq.kernels.attn_decode import replace_attention_cores  # noqa: PLC0415
        from lemoq.kernels.attn_decode import warm as warm_attn  # noqa: PLC0415

        replace_attention_cores(model, backend="triton")
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

    def _arm() -> dict[str, Any]:
        """★ 每条臂必须有自己的 session：`StaticCacheDecodeSession` 里存着它自己的那张图，
        共用一个 session 会把 A 臂重放成 B 的图（第一版就是这么错的）。"""
        sess = StaticCacheDecodeSession(
            model, ids, ctx=args.ctx, max_new_tokens=args.max_new_tokens, device="cuda"
        )
        torch.cuda.synchronize()
        sess.prefill()
        torch.cuda.synchronize()
        return {"session": sess, "capture": sess.capture(warmup=5)}

    # ---- 臂 A：HF 原实现（先捕获 —— 图记录的是内核，之后的实例替换不影响它）----
    arm_a = _arm()

    # ---- 臂 B：融合 norm（实例级替换后**新建 session** 再捕获）----
    replaced = replace_rms_norms(model)
    if replaced == 0:
        print("没有找到 OlmoeRMSNorm —— 模型结构变了，拒绝给结论", file=sys.stderr)
        return 3
    arm_b = _arm()

    # ---- 逐轮交错重放（每臂先 reset 到同一位置，收集逐步 logits）----
    sessions = {"hf_rmsnorm": arm_a["session"], "fused_rms_norm": arm_b["session"]}
    logits: dict[str, list[Any]] = {"hf_rmsnorm": [], "fused_rms_norm": []}
    for name, sess in sessions.items():
        sess.reset()
        sess.prefill()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {k: [] for k in sessions}
    for r in range(args.rounds):
        for name, sess in sessions.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sess.replay()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) * 1000.0)
            if r < 8:
                logits[name].append(_flatten_logits(sess.logits_cpu()))
    winners = dict.fromkeys(sessions, 0)
    for r in range(args.rounds):
        winners[min(sessions, key=lambda n: samples[n][r])] += 1

    # ★ 每一步都各自 reset/prefill 过，位置同步推进，所以逐步 logits 可比
    cmp = compare_logits(logits["hf_rmsnorm"], logits["fused_rms_norm"])
    stats = {k: _summarize(v) for k, v in samples.items()}
    speedup = stats["hf_rmsnorm"]["p50_ms"] / stats["fused_rms_norm"]["p50_ms"]
    payload = {
        "tool": "tools/bench_fused_norm.py",
        "config": quant_cfg,
        "shape": {"ctx": args.ctx, "max_new_tokens": args.max_new_tokens, "rounds": args.rounds},
        "norms_replaced": replaced,
        "arms": stats,
        "wins": winners,
        "speedup_fused_over_hf": speedup,
        "correctness_vs_hf": cmp,
        "capture": {"hf": arm_a["capture"], "fused": arm_b["capture"]},
        "k1_kl_below_1e-3": bool(cmp["kl"] < 1e-3),
        "k1_top1_at_least_0.99": bool(cmp["top1_agreement"] >= 0.99),
    }
    if args.profile:
        from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

        profiled: dict[str, Any] = {}
        for name, sess in sessions.items():
            for _ in range(3):
                sess.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(args.rounds):
                    sess.replay()
                torch.cuda.synchronize()
            rows = []
            for e in prof.key_averages():
                tot = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
                if tot <= 0:
                    continue
                rows.append(
                    {
                        "name": e.key,
                        "ms_per_step": tot / args.rounds / 1000.0,
                        "count_per_step": e.count / float(args.rounds),
                    }
                )
            rows.sort(key=lambda r: -r["ms_per_step"])
            profiled[name] = {
                "gpu_ms_per_step": sum(r["ms_per_step"] for r in rows),
                "kernels_per_step": sum(r["count_per_step"] for r in rows),
                "top": rows[:10],
            }
        payload["profile"] = profiled
        for name in sessions:
            pr = profiled[name]
            print(
                "\n[%s] GPU %.3f ms/步 | kernel %.0f/步"
                % (name, pr["gpu_ms_per_step"], pr["kernels_per_step"])
            )
            for r in pr["top"]:
                print(
                    "    %8.4f ms ×%-6.1f %s"
                    % (r["ms_per_step"], r["count_per_step"], r["name"][:74])
                )

    print("\n%-16s %10s %10s %8s" % ("arm", "p50(ms)", "min(ms)", "wins"))
    for name in sessions:
        print(
            "%-16s %10.4f %10.4f %8d"
            % (name, stats[name]["p50_ms"], stats[name]["min_ms"], winners[name])
        )
    print(
        "\n融合臂 vs HF：**%.4fx**（%.4f → %.4f ms）"
        % (speedup, stats["hf_rmsnorm"]["p50_ms"], stats["fused_rms_norm"]["p50_ms"])
    )
    print(
        "数值：KL=%.3e（K1 <1e-3: %s）| top-1 一致=%.4f（K1 ≥0.99: %s）"
        % (
            cmp["kl"],
            payload["k1_kl_below_1e-3"],
            cmp["top1_agreement"],
            payload["k1_top1_at_least_0.99"],
        )
    )
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n写入 %s" % args.json)
    return 0


def _flatten_logits(x: Any) -> Any:
    import torch  # noqa: PLC0415

    return x.detach().reshape(1, -1).to(device="cpu", dtype=torch.float32)


if __name__ == "__main__":
    raise SystemExit(main())
