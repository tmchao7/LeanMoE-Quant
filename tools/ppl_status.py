"""ppl_status — 生成 / 校验 `results/PPL_STATUS.json`（PPL 的**声明清单**）。

## 为什么需要这份清单

`docs/METRICS.md` §1 把 PPL 列为带门槛（相对退化 <1%）的正确性指标。但历史上
**98 份 `result.json` 的 `ppl_fp16 / ppl_this / ppl_delta` 全是 `null`，且没有任何
说明** —— 那既可能是"未测"，也可能是"漏记"，读者分不出来。项目硬规则 1 要求
未测的值显式写成未测，而"没有声明的 null"不满足这一点。

两条路可以修：
    1. 把 PPL 真的写进每份 result.json（要 GPU，见待办 G5）；
    2. 给每份 result.json 一句声明，说明这三个字段为什么是 null。

本工具走第 2 条，**通过新增一份清单文件而不是重写 98 份历史工件** ——
`AGENTS.md` §6 规定覆盖已落盘的 result.json 必须先问用户并归档。

## 用法

    python -m tools.ppl_status --dry-run          # 只打印将要写什么
    python -m tools.ppl_status                    # 写 results/PPL_STATUS.json
    python -m tools.ppl_status --check            # 校验现有工件是否都有声明（退出码非 0 = 有缺失）

★ 本工具**不产出任何 PPL 数值**。它只记录"哪些格子测了、哪些没测、为什么"。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

PPL_FIELDS = ("ppl_fp16", "ppl_this", "ppl_delta")

# 未测的原因。写成常量是为了让 98 份工件共享**同一句**声明 —— 声明各不相同就等于没声明。
REASON_NOT_MEASURED = (
    "PPL 尚未测量：`tools/quant_ppl.py` 已实现（wikitext-2 test 前 8192 token + 配对 ΔNLL），"
    "但其结果还没有写进任何 result.json —— 数字目前只存在于 EXP-p1-03 记录 §2 的表格里。"
    "见 docs/METRICS.md §6 待办 G5。"
)
BLOCKED_BY = "gpu（quant_ppl 需要 CUDA 加载模型；Mac 上的数字不得进 results/，硬规则）"
PLANNED_METHOD = (
    "wikitext-2-raw-v1 test 前 8192 token，不重叠 ctx 长窗口，每个 token 计一次；"
    "fp16 与量化臂在**同一串** token 上算（teacher forcing）。"
)


def scan(results_dir: str = "results") -> dict[str, Any]:
    """扫所有 result.json，判定每格的 PPL 是"已测"还是"未测"。"""
    cells: dict[str, Any] = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        exp_id = payload.get("exp_id")
        if not isinstance(exp_id, str):
            continue
        corr = payload.get("correctness") or {}
        measured = [k for k in PPL_FIELDS if (corr or {}).get(k) is not None]
        if measured:
            cells[exp_id] = {
                "status": "measured",
                "measured_fields": measured,
                "source": os.path.relpath(path, results_dir),
                "note": (corr.get("ppl_note") or ""),
            }
        else:
            cells[exp_id] = {
                "status": "not_measured",
                "reason": REASON_NOT_MEASURED,
                "blocked_by": BLOCKED_BY,
                "planned_method": PLANNED_METHOD,
                "source": os.path.relpath(path, results_dir),
            }
    return cells


def build(cells: dict[str, Any]) -> dict[str, Any]:
    n_measured = sum(1 for v in cells.values() if v.get("status") == "measured")
    return {
        "generated_by": "tools/ppl_status.py",
        "kind": "PPL declaration index（**不含任何 PPL 数值**）",
        "purpose": (
            "让 `ppl_*` 为 null 的格子有一句统一声明，从而区分'未测'与'漏记'。"
            "`benchmarks.schema.validate_ppl_declaration` 会读这份清单。"
        ),
        "n_cells_total": len(cells),
        "n_measured": n_measured,
        "n_not_measured": len(cells) - n_measured,
        "ppl_fields": list(PPL_FIELDS),
        "note": (
            "★ 本清单**不是**测量结果。PPL 数值要等 GPU 上跑 `tools/quant_ppl.py` 并写入"
            " result.json 之后才有（docs/METRICS.md §6 待办 G5）。"
        ),
        "cells": cells,
    }


def check(cells: dict[str, Any]) -> int:
    """每格是否都有声明（有值 or 有 reason）。返回缺失数。"""
    missing = [
        k
        for k, v in cells.items()
        if v.get("status") != "measured" and not str(v.get("reason") or "").strip()
    ]
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="默认 <results-dir>/PPL_STATUS.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="只校验，不写")
    args = ap.parse_args(argv)

    cells = scan(args.results_dir)
    index = build(cells)
    missing = check(cells)

    if args.check:
        print(
            "PPL 声明检查：%d 格，已测 %d，未测 %d，缺声明 %d"
            % (index["n_cells_total"], index["n_measured"], index["n_not_measured"], missing)
        )
        return 1 if missing else 0

    out = args.out or os.path.join(args.results_dir, "PPL_STATUS.json")
    print(
        "PPL 声明清单：%d 格（已测 %d / 未测 %d）"
        % (index["n_cells_total"], index["n_measured"], index["n_not_measured"])
    )
    print("  声明原文（所有未测格子共用一句）：")
    print("    %s" % REASON_NOT_MEASURED)
    if args.dry_run:
        print("\n  [dry-run] 将写入 %s（未实际写入）" % out)
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
