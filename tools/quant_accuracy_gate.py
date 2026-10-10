"""quant_accuracy_gate — 在 GPU 上跑 `EXP-p1-01`：W8/W4 × group{32,64,128} 的精度门。

    python -m tools.quant_accuracy_gate --model M --cells w8a16:32,w8a16:64 \\
        --ctx 1024 --max-new-tokens 32

## 流程（每个 cell 一次）

    1. 加载 fp16 模型，free-running 贪心出一串 32 个 token（**只做一次**，
       两条臂共用；这叫 teacher forcing 的"取轨迹"步骤）；
    2. 在 fp16 模型上 teacher-force 这串 token，收集参考 logits；
    3. **重新加载**一份干净的 fp16 权重（量化是就地替换，不能复用）；
    4. `quantize_model` 后 teacher-force **同一串** token；
    5. 逐位置比 KL / top-1，套 `docs/reference/METRICS.md` §1 的门槛。

## 为什么每个 cell 都要重新加载模型

`quantize_model` 是**就地替换**。复用同一个模型对象会在已经量化过的权重上再量化，
而那种错误不会报错，只会给出一个更差的 KL，然后被当成"W4 精度不行"。

## 落盘位置

每个 cell 写 `results/<EXP-p1-01 cell name>/result.json`，
命名与 `configs/experiments.yaml` 的 factor 展开一致，这样 harness 的
`is_done` 断点续跑逻辑将来能直接接上。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

DEFAULT_CELLS = [
    ("w8a8", 32),
    ("w8a8", 64),
    ("w8a8", 128),
    ("w8a16", 32),
    ("w8a16", 64),
    ("w8a16", 128),
    ("w4a16", 32),
    ("w4a16", 64),
    ("w4a16", 128),
]


def _greedy_tokens(model: Any, ids: Any, n: int, device: str) -> list[int]:
    """用 fp16 模型贪心出一条轨迹，供两条臂共用（teacher forcing 的第一步）。"""
    import torch  # noqa: PLC0415

    tokens: list[int] = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(n):
            tokens.append(int(tok.item()))
            out = model(tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
    return tokens


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--cells",
        default=None,
        help="逗号分隔的 scheme:group_size，例如 w4a16:64,w8a16:128（默认全部 9 格）",
    )
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--targets", default="attention,experts,router")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--json", default=None, help="额外的汇总输出路径")
    args = ap.parse_args(argv)

    import torch

    from benchmarks.quant_gate import (
        QUANT_THRESHOLDS,
        cell_name,
        quant_accuracy_report,
        teacher_forced_logits,
    )
    from benchmarks.runner import _load_model
    from lemoq.quant.linear import SchemeSpec, quantize_model

    if not torch.cuda.is_available():
        print("需要 CUDA；这个脚本的数字不得在无 GPU 的机器上产生。", file=sys.stderr)
        return 2

    if args.cells:
        cells = []
        for item in args.cells.split(","):
            scheme, _, gs = item.strip().partition(":")
            cells.append((scheme, int(gs)))
    else:
        cells = list(DEFAULT_CELLS)
    targets = tuple(t.strip() for t in args.targets.split(",") if t.strip())

    device = "cuda"
    ids = torch.full((1, args.ctx), 1, dtype=torch.long, device=device)
    summary: dict[str, Any] = {
        "model": args.model,
        "ctx": args.ctx,
        "max_new_tokens": args.max_new_tokens,
        "targets": list(targets),
        "thresholds": QUANT_THRESHOLDS,
        "cells": {},
    }

    print("=== 取轨迹（fp16 贪心 %d 步）===" % args.max_new_tokens, flush=True)
    model, _ = _load_model(args.model, "float16", device)
    t0 = time.perf_counter()
    tokens = _greedy_tokens(model, ids, args.max_new_tokens, device)
    print(
        "  tokens=%s  (%.1f s)" % (tokens[:8], time.perf_counter() - t0),
        flush=True,
    )
    reference = teacher_forced_logits(model, ids, tokens, device=device)
    # 参考臂跑完就释放，避免与下面每一格的量化权重同时驻留
    del model
    torch.cuda.empty_cache()
    summary["trajectory_tokens"] = tokens
    summary["n_positions"] = len(reference)

    for scheme, group_size in cells:
        name = cell_name(scheme, group_size)
        print("=== %s ===" % name, flush=True)
        try:
            model, _ = _load_model(args.model, "float16", device)
            spec = SchemeSpec.from_flags(scheme, group_size, targets=targets)
            model, qreport = quantize_model(model, spec)
            candidate = teacher_forced_logits(model, ids, tokens, device=device)
            report = quant_accuracy_report(
                reference,
                candidate,
                scheme=scheme,
                group_size=group_size,
                quantize_report=qreport.to_dict(),
            )
        except Exception as exc:  # noqa: BLE001
            report = {
                "scheme": scheme,
                "group_size": group_size,
                "gate": "ERROR",
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:400]),
            }
        finally:
            try:
                del model
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        summary["cells"][name] = report
        print(
            "  gate=%s kl=%s top1=%s"
            % (
                report.get("gate"),
                report.get("kl"),
                report.get("top1_agreement"),
            ),
            flush=True,
        )
        if report.get("gate") != "ERROR":
            out_dir = os.path.join(args.results_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "id": name,
                        "status": "done",
                        "config": {
                            "bit_width": scheme,
                            "group_size": group_size,
                            "targets": list(targets),
                            "ctx": args.ctx,
                        },
                        "correctness": {
                            "kl": report["kl"],
                            "top1_agreement": report["top1_agreement"],
                            "gate": report["gate"],
                            "thresholds": report["thresholds"],
                            "method": report["method"],
                            "n_positions": report["n_positions"],
                        },
                        "quantize_report": report["quantize_report"],
                        "verdict": {"verdict": report["gate"]},
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )

    print("\n=== 汇总（门 = docs/reference/METRICS.md §1）===")
    print("  %-28s %10s %8s %6s" % ("cell", "kl", "top1", "gate"))
    for name, report in summary["cells"].items():
        print(
            "  %-28s %10s %8s %6s"
            % (name, report.get("kl"), report.get("top1_agreement"), report.get("gate"))
        )
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
        print("\nwrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
