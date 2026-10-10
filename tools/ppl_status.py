"""ppl_status — 生成 / 校验 / 合并 PPL 的声明与数值（`results/PPL_STATUS.json`）。

## 为什么需要这份清单

`docs/reference/METRICS.md` §1 把 PPL 列为带门槛（相对退化 <1%）的正确性指标。但历史上
**98 份 `result.json` 的 `ppl_fp16 / ppl_this / ppl_delta` 全是 `null`，且没有任何
说明** —— 那既可能是"未测"，也可能是"漏记"，读者分不出来。项目硬规则 1 要求
未测的值显式写成未测，而"没有声明的 null"不满足这一点。

三条路对应三种用法：
    1. 声明（离线）：`--dry-run` / 默认模式 —— 给每份 null 的格子一句统一声明；
    2. 校验：`--check` —— 所有格子要么有值、要么有声明，退出码非 0 = 有缺失；
    3. 合并（上机，G5）：`--merge <quant_ppl.json>` —— 把 `tools/quant_ppl.py`
       的输出写进指定 cell 的 result.json（写前先归档，规则 9），然后重建清单。

★ 模式 1 **不产出任何 PPL 数值**；模式 3 写入的数值必须来自 results/ 下
`quant_ppl.json` 工件（它自带 fp16 参照臂与 noop 自检臂），不得手工填数。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from typing import Any

PPL_FIELDS = ("ppl_fp16", "ppl_this", "ppl_delta")

# 未测的原因。写成常量是为了让 98 份工件共享**同一句**声明 —— 声明各不相同就等于没声明。
REASON_NOT_MEASURED = (
    "PPL 尚未测量：`tools/quant_ppl.py` 已实现（wikitext-2 test 前 8192 token + 配对 ΔNLL），"
    "但其结果还没有写进任何 result.json —— 数字目前只存在于 EXP-p1-03 记录 §2 的表格里。"
    "见 docs/reference/METRICS.md §6 待办 G5。"
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
            " result.json 之后才有（docs/reference/METRICS.md §6 待办 G5）。"
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


# ---------------------------------------------------------------------------
# G5：把 quant_ppl 的输出合并进 cell 工件
# ---------------------------------------------------------------------------


def _load_ppl_run(path: str, arm_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """读 quant_ppl.json，返回 (fp16 参照, 指定臂, 元信息)。"""
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    fp16 = run.get("fp16") or {}
    arms = run.get("arms") or {}
    if arm_id not in arms:
        raise SystemExit("quant_ppl.json 里没有臂 %r（有：%s）" % (arm_id, sorted(arms)))
    arm = arms[arm_id]
    for lbl, entry, key in (("fp16", fp16, "ppl"), ("arm", arm, "ppl")):
        if not isinstance(entry.get(key), (int, float)):
            raise SystemExit("%s 臂缺 %s 字段 —— quant_ppl.json 不完整，拒绝合并" % (lbl, key))
    # noop 自检臂：ΔNLL 必须恰好为 0，否则恢复不干净，整份工件不可信。
    # ★ 键名是 `paired_vs_fp16.delta_nll_per_token`（quant_ppl.py 的实际产出）；
    #   2026-10-10 评审前这里读的是不存在的 `paired.mean`，防呆从未触发过。
    for nid, nentry in arms.items():
        if nentry.get("kind") == "noop":
            paired = nentry.get("paired_vs_fp16")
            if not isinstance(paired, dict) or not isinstance(
                paired.get("delta_nll_per_token"), (int, float)
            ):
                raise SystemExit(
                    "noop 臂 %s 缺 paired_vs_fp16.delta_nll_per_token —— 无法验证快照"
                    "恢复是否干净，拒绝合并（先跑 tools/quant_ppl.py 重新生成）" % nid
                )
            mean = float(paired["delta_nll_per_token"])
            if abs(mean) > 1e-9:
                raise SystemExit(
                    "noop 臂 %s 的 ΔNLL=%.3e ≠ 0 —— 快照恢复不干净，本次 PPL 运行不可信，"
                    "拒绝合并（先修 quant_ppl 的快照/恢复）" % (nid, mean)
                )
    meta = {k: run.get(k) for k in ("model", "dataset", "n_tokens", "ctx_window", "metric")}
    return fp16, arm, meta


def _ppl_note(meta: dict[str, Any], arm_id: str, fields: list[str]) -> str:
    scope = "ppl_fp16/ppl_this/ppl_delta" if "ppl_this" in fields else "仅 ppl_fp16（参照值）"
    return (
        "PPL 实测（%s <- tools/quant_ppl.py）：wikitext-2-raw-v1 test 前 %s token，"
        "ctx=%s 不重叠窗口、每 token 计一次（窗口首 token 不计），fp16 与量化臂在同一串 "
        "token 上 teacher forcing。ppl_delta = ppl_this/ppl_fp16 − 1（相对退化；"
        "METRICS §1 门槛 <1%%）。arm=%s。本 cell 写入范围：%s。"
        % (
            meta.get("dataset") or "wikitext-2-raw-v1",
            meta.get("n_tokens"),
            meta.get("ctx_window"),
            arm_id,
            scope,
        )
    )


def merge(
    ppl_json: str,
    arm_id: str,
    quant_cells: list[str],
    ref_cells: list[str],
    results_dir: str = "results",
    archive_dir: str | None = None,
) -> int:
    """把 quant_ppl.json 的数值写进 cell 的 result.json（写前归档，规则 9）。"""
    fp16, arm, meta = _load_ppl_run(ppl_json, arm_id)
    ts = time.strftime("%Y%m%d-%H%M%S")
    archive_dir = archive_dir or os.path.join(results_dir, "ppl_merge_archive", ts)
    os.makedirs(archive_dir, exist_ok=True)

    written = 0
    for exp_id, is_quant in [(c, True) for c in quant_cells] + [(c, False) for c in ref_cells]:
        path = os.path.join(results_dir, exp_id, "result.json")
        if not os.path.exists(path):
            print("!! 跳过（找不到 %s）" % path)
            continue
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("exp_id") != exp_id:
            print("!! 跳过（%s 的 exp_id=%r 与目录名不符）" % (path, payload.get("exp_id")))
            continue
        # 规则 9：覆盖前归档
        shutil.copy2(path, os.path.join(archive_dir, "%s.result.json" % exp_id))
        corr = payload.setdefault("correctness", {})
        corr["ppl_fp16"] = float(fp16["ppl"])
        if is_quant:
            corr["ppl_this"] = float(arm["ppl"])
            corr["ppl_delta"] = float(arm["ppl"]) / float(fp16["ppl"]) - 1.0
            fields = ["ppl_fp16", "ppl_this", "ppl_delta"]
        else:
            fields = ["ppl_fp16"]
        corr["ppl_note"] = _ppl_note(meta, arm_id, fields)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        written += 1
        print(
            "wrote %s: ppl_fp16=%.4f%s"
            % (
                exp_id,
                float(fp16["ppl"]),
                (
                    (" ppl_this=%.4f ppl_delta=%+.4f%%" % (arm["ppl"], corr["ppl_delta"] * 100))
                    if is_quant
                    else ""
                ),
            )
        )
    print("共写入 %d 份；原值归档在 %s" % (written, archive_dir))
    paired = arm.get("paired_vs_fp16")
    if isinstance(paired, dict) and isinstance(paired.get("delta_nll_per_token"), (int, float)):
        print(
            "  配对 ΔNLL/token = %+.6f (paired SE %s, n=%s) —— 该值**不**写进 cell，"
            "只活在 quant_ppl.json 里"
            % (
                float(paired["delta_nll_per_token"]),
                paired.get("paired_se", "—"),
                paired.get("n_pairs", "—"),
            )
        )
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None, help="默认 <results-dir>/PPL_STATUS.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check", action="store_true", help="只校验，不写")
    # G5 合并模式
    ap.add_argument("--merge", metavar="QUANT_PPL_JSON", default=None)
    ap.add_argument("--arm", default=None, help="quant_ppl.json 里作为 ppl_this 的臂 id")
    ap.add_argument(
        "--quant-cell",
        action="append",
        default=[],
        help="写入 ppl_fp16/ppl_this/ppl_delta 的 cell（exp_id，可重复）",
    )
    ap.add_argument(
        "--ref-cell", action="append", default=[], help="只写 ppl_fp16 的参照 cell（可重复）"
    )
    ap.add_argument("--archive-dir", default=None)
    args = ap.parse_args(argv)

    if args.merge:
        if not args.arm or not (args.quant_cell or args.ref_cell):
            ap.error("--merge 需要 --arm 与至少一个 --quant-cell / --ref-cell")
        n = merge(
            args.merge,
            args.arm,
            args.quant_cell,
            args.ref_cell,
            results_dir=args.results_dir,
            archive_dir=args.archive_dir,
        )
        if n == 0:
            return 1
        # 合并完顺手重建清单
        cells = scan(args.results_dir)
        index = build(cells)
        out = args.out or os.path.join(args.results_dir, "PPL_STATUS.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(
            "\nPPL 声明清单已重建：%d 格（已测 %d / 未测 %d）-> %s"
            % (index["n_cells_total"], index["n_measured"], index["n_not_measured"], out)
        )
        return 0

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
