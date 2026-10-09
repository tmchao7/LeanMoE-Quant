"""batch_sweep — batch 扫描（**GPU 待执行**，见 docs/METRICS.md §6 G2）。

## 为什么它单独存在

`docs/METRICS.md` §2 声明了 `throughput @ batch ∈ {1,2,4,8}` 作为次要指标
（"用于展示量化收益随 batch 的衰减"），但已落盘的 98 份 `result.json` 里
**`config.batch` 全是 1** —— 零条数据。本工具补这一格。

## ★ 硬规则 3：不得跨轮比较

本轮产出的数字**禁止**与已有的 batch=1 历史数字放进同一张表、也禁止相除得出
"衰减倍数"。原因是历史数字是**别的轮次**在别的机器状态下测的（共享卡、同租户、
时钟），跨轮比较会把机器漂移算成方案差异。

所以：
- 本工具**自己**在**同一轮里**把 batch=1 也测一遍（`--include-batch-1`，默认开），
  这样"随 batch 的衰减"是**同轮**比值，可以用；
- 输出里 `cross_round_comparison_allowed: false` 是写死的，汇总脚本
  （`tools/summarize_results.py`）会据此把曲线**单列一张表**，并在表头标注轮次。

## 口径

- 与 `benchmarks/runner.py` 同样的 warmup ≥ 5 / 计次 ≥ 20 / 报 p50 + p99 + stddev；
- **冷态守卫（2026-10-09 加）**：任何测量之前先做一次**全局预热**（最小与最大 batch
  各一遍，不计时）；每个 batch 的**前 `--discard-repeats` 轮计时丢弃**（默认 1），
  丢弃值仍落盘到 `discarded_ms`。加它的原因见 `_warmup_pass` 的 docstring ——
  前两轮就是因为 batch=1 冷启（142 ms vs 稳态 92 ms）被判 VOID；
- 每个 batch 都是**同一轮**、同一 shape / dtype / GPU / warmup / 计次；
- 报三件东西：`tok/s`（实测吞吐）、`per-step p50/p99`（decode 步延迟）、
  `per-token = per-step / batch`（便于与 batch=1 的 per-token 口径对照）；
- 首格与末格跑**同格金丝雀**（batch=1），漂移 >5% ⇒ 整轮 VOID；VOID 轮的实测值
  落进 `void_evidence`（标注 `must_not_be_cited`），`rows` 为空 —— 与 runner 对
  gate FAIL 格的处理同款。

## 用法

    python -m tools.batch_sweep --dry-run
    python -m tools.batch_sweep --model <M> --ctx 1024 --json results/batch_sweep.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

DEFAULT_BATCHES = (1, 2, 4, 8)
CANARY_MAX_DRIFT_PCT = 5.0


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generated_by": "tools/batch_sweep.py --dry-run",
        "status": "planned",
        "measured": False,
        "note": "**计划**，不是结果。未测的值一律 planned（硬规则 1）。",
        "batches": [int(b) for b in args.batch],
        "ctx": args.ctx,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "repeats": args.repeats,
        "discard_repeats": args.discard_repeats,
        "cold_start_guard": (
            "测量前对最小/最大 batch 各做一遍全局预热（不计时）；每个 batch 丢弃前 "
            "%d 轮计时（值仍落盘 discarded_ms）" % args.discard_repeats
        ),
        "include_batch_1_in_this_round": bool(args.include_batch_1),
        "per_batch_outputs": [
            "tok_per_s（实测吞吐）",
            "per_step_p50_ms / p99_ms（decode 步延迟）",
            "per_token_p50_ms = per_step / batch（与 batch=1 口径对照用）",
            "stddev + 样本量 n",
        ],
        "canary": {
            "protocol": "首格与末格跑同格（batch=1）；漂移 > %.1f%% ⇒ 整轮 VOID"
            % CANARY_MAX_DRIFT_PCT,
            "on_void": "保留日志，不发布任何数字",
        },
        "cross_round_comparison_allowed": False,
        "cross_round_note": (
            "本轮数字不得与历史 batch=1 数字混表或相除。要算'随 batch 的衰减'，"
            "必须用**本轮**自己测的 batch=1（--include-batch-1 默认开）。"
        ),
        "acceptance": {
            "must_report": [
                "GPU / dtype / ctx / warmup / timed / repeats（完整配置）",
                "每个 batch 的样本量 n 与 stddev",
                "金丝雀漂移百分比",
                "显存峰值（batch 增大时这是主要约束）",
            ],
            "curve_is_a_separate_table": True,
        },
        "output": args.json or "results/batch_sweep.json",
    }


def _warmup_pass(model: Any, ids: Any, *, steps: int) -> None:
    """一次**不计时**的完整预热：prefill + `steps` 个 decode 步。

    ★ 为什么要它（2026-10-09 两轮 VOID 的根因）：HF 路径的**第一个**被计时的
      batch 会撞上三重冷态 —— CUDA 时钟还没拉到稳态、allocator 还在扩张、首次
      decode 走的是没被 CUDA Graph / cudnn 命中过的 kernel。实测 batch=1 首轮
      **142.26 ms**，而同一轮里 batch=2/4/8 只有 92–98 ms ⇒ 金丝雀漂移 56%。
      预热必须发生在**任何一次测量之前**，且要覆盖最小与最大 batch（显存布局不同）。
    """
    import torch

    with torch.no_grad():
        out = model(ids, use_cache=True)
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = out.past_key_values
        for _ in range(max(int(steps), 1)):
            out = model(tok, past_key_values=past, use_cache=True)
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = out.past_key_values
    torch.cuda.synchronize()


def _time_decode(
    model: Any,
    ids: Any,
    *,
    batch: int,
    warmup: int,
    timed: int,
    repeats: int,
    discard: int = 1,
):
    """跑一个 batch 的 decode 计时。

    返回 `(kept_ms, discarded_ms, peak_bytes)`：`discarded_ms` 是前 `discard` 个
    repeat 的耗时，**不进统计**（冷态守卫），但落盘供诊断 —— 丢掉不等于不知道它多少。
    """
    import torch

    # 预热：prefill 相位 + 若干 decode 步，全部不计时
    for _ in range(max(int(warmup), 1)):
        _warmup_pass(model, ids, steps=max(int(warmup), 5))

    kept: list[float] = []
    discarded: list[float] = []
    peak = 0
    n_repeats = max(int(repeats), 1) + max(int(discard), 0)
    for i in range(n_repeats):
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = model(ids, use_cache=True)
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = out.past_key_values
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(int(timed), 20)):
            with torch.no_grad():
                out = model(tok, past_key_values=past, use_cache=True)
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = out.past_key_values
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0 / max(int(timed), 20)
        if i < max(int(discard), 0):
            discarded.append(ms)
        else:
            kept.append(ms)
        peak = max(peak, int(torch.cuda.max_memory_allocated()))
    return kept, discarded, peak


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", default=",".join(str(b) for b in DEFAULT_BATCHES))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=10, help="预热轮数（含 decode 步）；不计时")
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument(
        "--repeats", type=int, default=5, help="保留的计时轮数（另有 --discard-repeats 个丢弃轮）"
    )
    ap.add_argument(
        "--discard-repeats",
        type=int,
        default=1,
        help="每个 batch 前几轮计时**丢弃**（冷态守卫）；丢弃值仍落盘到 discarded_ms 供诊断",
    )
    ap.add_argument(
        "--include-batch-1",
        action="store_true",
        default=True,
        help="本轮自己也测 batch=1 —— 这样'随 batch 衰减'是同轮比值（硬规则 3）",
    )
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    args.batch = [int(b) for b in str(args.batch).split(",") if b.strip()]
    if args.include_batch_1 and 1 not in args.batch:
        args.batch = [1] + args.batch

    if args.dry_run:
        print(json.dumps(_plan(args), ensure_ascii=False, indent=2))
        print(
            "\n[dry-run] 未做任何测量。\n"
            "  ★ 输出曲线必须**单列一张表**，并标注本轮轮次；\n"
            "  ★ 不得与历史 batch=1 数字混表或相除（硬规则 3）。",
            file=sys.stderr,
        )
        return 0

    try:
        import torch
    except Exception:  # noqa: BLE001
        print("需要 torch。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("batch_sweep 需要 CUDA；离线请用 --dry-run。", file=sys.stderr)
        return 2
    if not args.model:
        print("--model 必填（或 --dry-run）。", file=sys.stderr)
        return 2

    from transformers import AutoModelForCausalLM

    from benchmarks.timing import summarize

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    model = model.to("cuda").eval()

    # ★ 全局预热**在任何测量之前**：覆盖最小 batch 与最大 batch（显存布局不同），
    #   把时钟拉到稳态。见 `_warmup_pass` 的 docstring（两轮 VOID 的教训）。
    for b in sorted({min(args.batch), max(args.batch)}):
        _warmup_pass(
            model,
            torch.full((b, args.ctx), 1, dtype=torch.long, device="cuda"),
            steps=max(int(args.warmup), 5),
        )
    print("  预热完成（batch=%s，不计时）" % sorted({min(args.batch), max(args.batch)}), flush=True)

    rows: list[dict[str, Any]] = []
    canary_first = None
    for b in args.batch:
        ids = torch.full((b, args.ctx), 1, dtype=torch.long, device="cuda")
        samples, discarded, peak = _time_decode(
            model,
            ids,
            batch=b,
            warmup=args.warmup,
            timed=args.timed,
            repeats=args.repeats,
            discard=args.discard_repeats,
        )
        st = summarize(samples)
        row = {
            "batch": b,
            "n_samples": st["n"],
            "n_discarded": len(discarded),
            "discarded_ms": [round(x, 4) for x in discarded],
            "per_step_p50_ms": round(st["p50"], 4),
            "per_step_p99_ms": round(st["p99"], 4),
            "per_step_stddev_ms": round(st["stddev"], 4),
            "per_token_p50_ms": round(st["p50"] / b, 4),
            "tok_per_s": round(1000.0 * b / st["p50"], 2) if st["p50"] else None,
            "peak_allocated_bytes": peak,
            "samples_ms": [round(x, 4) for x in samples],
        }
        rows.append(row)
        if b == 1 and canary_first is None:
            canary_first = st["p50"]
        print(
            "  batch=%d  step_p50=%.4f ms  tok/s=%s%s"
            % (
                b,
                st["p50"],
                row["tok_per_s"],
                ("   [丢弃首轮 %s]" % row["discarded_ms"]) if discarded else "",
            ),
            flush=True,
        )

    # 金丝雀：末尾再测一次 batch=1
    canary_last = None
    if canary_first is not None:
        ids = torch.full((1, args.ctx), 1, dtype=torch.long, device="cuda")
        samples, _, _ = _time_decode(
            model,
            ids,
            batch=1,
            warmup=args.warmup,
            timed=args.timed,
            repeats=args.repeats,
            discard=args.discard_repeats,
        )
        canary_last = summarize(samples)["p50"]

    drift = None
    if canary_first and canary_last:
        drift = round(100.0 * (canary_last - canary_first) / canary_first, 3)
    void = drift is not None and abs(drift) > CANARY_MAX_DRIFT_PCT

    payload = {
        "generated_by": "tools/batch_sweep.py",
        "status": "done" if not void else "void",
        "ctx": args.ctx,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "repeats": args.repeats,
        "discard_repeats": args.discard_repeats,
        "global_warmup_batches": sorted({min(args.batch), max(args.batch)}),
        "dtype": "float16",
        "device": torch.cuda.get_device_name(0),
        "canary": {
            "batch_1_first_ms": canary_first,
            "batch_1_last_ms": canary_last,
            "drift_pct": drift,
            "max_drift_pct": CANARY_MAX_DRIFT_PCT,
            "void": bool(void),
        },
        "cross_round_comparison_allowed": False,
        "cross_round_note": "本轮数字不得与历史 batch=1 数字混表或相除（硬规则 3）。",
        # ★ VOID 轮的实测值**不**进 `rows`（那会看起来像可引用的数字），
        #   但也不能凭空丢掉 —— 与 runner 对 gate FAIL 格的处理同款，放进
        #   `void_evidence` 并显式标注"不得引用"。它正是诊断漂移来源的唯一材料。
        "rows": [] if void else rows,
        "void_evidence": (
            {
                "why": "金丝雀漂移 %.3f%% > %.1f%% ⇒ 整轮 VOID" % (drift, CANARY_MAX_DRIFT_PCT),
                "must_not_be_cited": True,
                "rows": rows,
            }
            if void
            else None
        ),
        "note": (
            "金丝雀漂移 %.3f%% 超过 %.1f%% ⇒ 整轮 VOID，不发布任何数字"
            "（原始值只在 `void_evidence` 里，仅用于诊断）。" % (drift, CANARY_MAX_DRIFT_PCT)
            if void
            else "同轮测量；'随 batch 的衰减'只能用本轮自己的 batch=1 做基准。"
        ),
    }
    out = args.json or "results/batch_sweep.json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\nwrote %s (status=%s, canary drift=%s%%)" % (out, payload["status"], drift))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
