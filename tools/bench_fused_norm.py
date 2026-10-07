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


def _torch_norm_forward(self: Any, hidden_states: Any) -> Any:
    """`OlmoeRMSNorm.forward` → `F.rms_norm`（torch 原生融合 kernel）。"""
    import torch.nn.functional as F  # noqa: PLC0415

    return F.rms_norm(
        hidden_states,
        (int(self.weight.shape[0]),),
        self.weight,
        float(self.variance_epsilon),
    )


def _triton_norm_forward(self: Any, hidden_states: Any) -> Any:
    """`OlmoeRMSNorm.forward` → 本仓库自研融合 kernel（`lemoq.kernels.rmsnorm`）。

    ★ 与 `F.rms_norm` 的决定性区别：本 kernel **先舍入到 fp16、再在 fp16 里乘 weight**，
      与 HF 的舍入点一致；`F.rms_norm` 在 fp32 里乘完再舍入，于是多出一次整行 fp32 乘
      （实测 0.189 ms / 65 次，正是它慢 5% 的原因 —— 见 `EXP-p4c-02` §5.5）。
    """
    from lemoq.kernels.rmsnorm import rms_norm_fused  # noqa: PLC0415

    return rms_norm_fused(
        hidden_states,
        self.weight,
        float(self.variance_epsilon),
        out_dtype=hidden_states.dtype,
    )


_IMPLS = {"hf": None, "torch": _torch_norm_forward, "triton": _triton_norm_forward}


def set_norm_impl(model: Any, impl: str) -> int:
    """把每个 `OlmoeRMSNorm.forward` 切到 `impl`（`hf` = 用原方法）。返回命中数。

    ★ 可反复切换：原方法只在第一次记录（`_lemoq_original_norm`），所以
      "hf → torch → triton" 这种累进捕获是可推理的 —— 每条臂捕获时用的是当时的实现。
    """
    if impl not in _IMPLS:
        raise KeyError("未知实现 %r；可用 %s" % (impl, sorted(_IMPLS)))
    fn = _IMPLS[impl]
    n = 0
    for module in model.modules():
        if type(module).__name__ != "OlmoeRMSNorm":
            continue
        if not hasattr(module, "_lemoq_original_norm"):
            module._lemoq_original_norm = module.forward
        module.forward = module._lemoq_original_norm if fn is None else types.MethodType(fn, module)
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
    ap.add_argument(
        "--impls",
        nargs="*",
        default=["hf", "torch", "triton"],
        help="要对比的 norm 实现，第一条必须是 hf（参照）",
    )
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
    impls = list(args.impls)
    if not impls or impls[0] != "hf":
        print("--impls 的第一条必须是 hf（它是参照）", file=sys.stderr)
        return 2

    def _arm(name: str) -> dict[str, Any]:
        """★ 每条臂必须有自己的 session：`StaticCacheDecodeSession` 里存着自己那张图，
        共用一个 session 会把 A 臂重放成 B 的图（第一版就是这么错的）。
        捕获前先切到该臂的实现 —— 图记录的是内核，之后的切换不影响已捕获的图。"""
        if name != "hf":
            if set_norm_impl(model, name) == 0:
                print("没有找到 OlmoeRMSNorm —— 模型结构变了，拒绝给结论", file=sys.stderr)
                raise SystemExit(3)
        sess = StaticCacheDecodeSession(
            model, ids, ctx=args.ctx, max_new_tokens=args.max_new_tokens, device="cuda"
        )
        torch.cuda.synchronize()
        sess.prefill()
        torch.cuda.synchronize()
        return {"session": sess, "capture": sess.capture(warmup=5), "impl": name}

    arms: dict[str, Any] = {}
    for impl in impls:
        arms["hf_rmsnorm" if impl == "hf" else impl + "_rms_norm"] = _arm(impl)
    sessions = {k: v["session"] for k, v in arms.items()}
    logits: dict[str, list[Any]] = {k: [] for k in sessions}
    for sess in sessions.values():
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

    stats = {k: _summarize(v) for k, v in samples.items()}
    ref = "hf_rmsnorm"
    # ★ 每条臂都各自 reset/prefill 过，位置同步推进 ⇒ 逐步 logits 可比
    correctness = {k: compare_logits(logits[ref], logits[k]) for k in sessions if k != ref}
    speedup = {k: stats[ref]["p50_ms"] / stats[k]["p50_ms"] for k in sessions if k != ref}
    payload = {
        "tool": "tools/bench_fused_norm.py",
        "config": quant_cfg,
        "shape": {"ctx": args.ctx, "max_new_tokens": args.max_new_tokens, "rounds": args.rounds},
        "arms": stats,
        "wins": winners,
        "speedup_vs_hf": speedup,
        "correctness_vs_hf": correctness,
        "capture": {k: v["capture"] for k, v in arms.items()},
        "k1": {
            k: {
                "kl_below_1e-3": bool(c["kl"] < 1e-3),
                "top1_at_least_0.99": bool(c["top1_agreement"] >= 0.99),
            }
            for k, c in correctness.items()
        },
    }
    if args.profile:
        from torch.profiler import ProfilerActivity, profile  # noqa: PLC0415

        profiled: dict[str, Any] = {}
        # ★ 主循环已经推进了 `rounds` 步；StaticCache 容量只有 ctx+max_new_tokens=1056 ⇒
        #   额外再放 3+rounds 步会直接 "decode position exceeded StaticCache capacity"（实测撞上）。
        #   所以 profile 前先 reset+prefill，并且只profile min(rounds, 10) 步。
        steps_prof = min(args.rounds, 10)
        for name, sess in sessions.items():
            sess.reset()
            sess.prefill()
            torch.cuda.synchronize()
            for _ in range(3):
                sess.replay()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(steps_prof):
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
                        "ms_per_step": tot / steps_prof / 1000.0,
                        "count_per_step": e.count / float(steps_prof),
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

    print("\n%-16s %10s %10s %8s %10s" % ("arm", "p50(ms)", "min(ms)", "wins", "vs hf"))
    for name in sessions:
        sp = speedup.get(name)
        print(
            "%-16s %10.4f %10.4f %8d %10s"
            % (
                name,
                stats[name]["p50_ms"],
                stats[name]["min_ms"],
                winners[name],
                ("%.4fx" % sp) if sp else "—",
            )
        )
    print("")
    for name, c in correctness.items():
        k1 = payload["k1"][name]
        print(
            "%-16s KL=%.3e（<1e-3: %s）| top-1=%.4f（≥0.99: %s）| 速度 %.4fx | D(>3%%): %s"
            % (
                name,
                c["kl"],
                k1["kl_below_1e-3"],
                c["top1_agreement"],
                k1["top1_at_least_0.99"],
                speedup[name],
                speedup[name] > 1.03,
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
