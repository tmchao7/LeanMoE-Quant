"""report — 把 `results/*/result.json` 汇总成可粘贴进 README 的 Markdown。

规则（来自 `docs/reference/METRICS.md` §5 与 Hoefler 的 12 条）：

    - **只报比值不算数**：加速比旁边必须给绝对值（ms/token）；
    - **报告全部点，包括输的**——不挑好看的；
    - **三态标注**：`已实测` / `待标定` / `待跑`（未测的一律显示 `—`，不显示 0）；
    - 跨卡/跨阶段/跨模型的数字**不放进同一张表**。

用法：
    python -m benchmarks.report                    # 打印所有已完成的实验
    python -m benchmarks.report --stage p4b        # 只看某阶段
    python -m benchmarks.report --check            # 只做 schema 校验，非零退出表示有问题
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmarks.schema import (
    PPL_STATUS_PATH,
    load_ppl_index,
    sanitize_gate_failed_perf,
    validate_result,
)

RESULTS_GLOB = "results/*/result.json"


def _is_archive_path(path: str) -> bool:
    """归档目录（`v1_archive/` 等）不是规范工件，不进汇总表（规则 9：归档 = 旧版本）。"""
    return any(seg == "archive" or seg.endswith("_archive") for seg in Path(path).parts)


def load_results(pattern: str = RESULTS_GLOB) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(glob.glob(pattern)):
        if _is_archive_path(p):
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            # 存量 gate==FAIL 工件里滞留的实测值（2026-10-10 评审实证）在读取时
            # 统一挪进 evidence 并置 None —— 展示与校验都按硬规则 2 的口径走，
            # 磁盘上的历史文件本体不改写。
            sanitize_gate_failed_perf(data)
            data["_path"] = p
            out.append(data)
        except Exception as exc:  # noqa: BLE001
            out.append({"_path": p, "_load_error": "%s: %s" % (type(exc).__name__, exc)})
    return out


def _fmt(value: Any, spec: str = "%.3f", dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        return spec % float(value)
    except (TypeError, ValueError):
        return str(value)


def _tristate(payload: dict[str, Any]) -> str:
    """标注：已实测 / 待跑 / 待标定 / 作废。"""
    status = payload.get("status")
    if status == "void":
        return "作废"  # 金丝雀判整轮 VOID —— 数字在档（诊断用）但不得引用
    if status != "done":
        return "待跑"
    corr = (payload.get("correctness") or {}).get("gate")
    if corr == "FAIL":
        return "待标定"  # 有过测量但没过门槛 → 不得当结论
    return "已实测"


def summary_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "| EXP-ID | 状态 | 位宽 | dequant | exec | ctx | p50 ms/tok | p99 | host占比 | 带宽达成率 | 量化达成率 | KL | top-1 | verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "_load_error" in r:
            lines.append(
                "| %s | 读取失败 | %s | | | | | | | | | | | |"
                % (Path(r["_path"]).parent.name, r["_load_error"][:40])
            )
            continue
        if r.get("_reportable") is False:
            continue  # dryrun 之类不得进正式表
        cfg = r.get("config") or {}
        lat = r.get("latency") or {}
        att = r.get("attribution") or {}
        attn = r.get("attainment") or {}
        corr = r.get("correctness") or {}
        ver = r.get("verdict") or {}
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                r.get("exp_id", "?"),
                _tristate(r),
                cfg.get("bit_width") or "—",
                cfg.get("dequant") or "—",
                cfg.get("execution") or "—",
                cfg.get("ctx") or "—",
                _fmt(lat.get("per_token_p50_ms")),
                _fmt(lat.get("per_token_p99_ms")),
                _fmt(att.get("host_share_pct"), "%.1f%%"),
                _fmt(attn.get("bw_attainment_pct"), "%.1f%%"),
                _fmt(attn.get("quant_attainment_pct"), "%.1f%%"),
                _fmt(corr.get("kl"), "%.4f"),
                _fmt(corr.get("top1_agreement"), "%.3f"),
                ver.get("verdict") or "—",
            )
        )
    return "\n".join(lines)


def byte_budget_section(results: list[dict[str, Any]]) -> str:
    rows = ["| EXP-ID | [W] B | [T] B | ctx | KV 占 [T] |", "|---|---|---|---|---|"]
    seen = False
    for r in results:
        b = r.get("bytes") or {}
        if b.get("W_total_bytes") is None:
            continue
        seen = True
        t = b.get("T_total_bytes")
        kv = b.get("kv_read_bytes")
        share = "—" if not (t and kv) else "%.1f%%" % (100.0 * kv / t)
        rows.append(
            "| %s | %s | %s | %s | %s |"
            % (
                r.get("exp_id", "?"),
                b.get("W_total_bytes"),
                _fmt(t, "%.0f"),
                (r.get("config") or {}).get("ctx", "—"),
                share,
            )
        )
    return "\n".join(rows) if seen else "（尚无 [W]/[T] 数据）"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default=RESULTS_GLOB)
    ap.add_argument("--results-dir", default="results", help="PPL 声明清单的查找目录")
    ap.add_argument("--stage", default=None, help="只显示该阶段（按 exp_id 前缀匹配）")
    ap.add_argument("--check", action="store_true", help="只做 schema 校验")
    args = ap.parse_args(argv)

    results = load_results(args.pattern)
    if args.stage:
        results = [r for r in results if r.get("exp_id", "").startswith("EXP-%s" % args.stage)]

    # ★ PPL 声明清单：历史工件的 ppl_* 是 null 但**没有** ppl_note，靠这份清单合规
    #   （否则 `tools/ppl_status.py` 存在的意义就没了）。
    ppl_index = load_ppl_index(os.path.join(args.results_dir, "PPL_STATUS.json"))
    if not ppl_index:
        print(
            "note: 未找到 %s —— ppl_* 为 null 的格子会被判'无声明'（跑 "
            "`python -m tools.ppl_status` 生成）" % PPL_STATUS_PATH,
            file=sys.stderr,
        )

    problems: list[str] = []
    for r in results:
        if "_load_error" in r:
            problems.append("%s: %s" % (r["_path"], r["_load_error"]))
            continue
        for p in validate_result(r, ppl_index):
            problems.append("%s: %s" % (r.get("exp_id", r["_path"]), p))

    if args.check:
        for p in problems:
            print("  ✗ %s" % p, file=sys.stderr)
        print("checked %d result file(s), %d problem(s)" % (len(results), len(problems)))
        return 1 if problems else 0

    reportable = [
        r for r in results if r.get("_reportable") is not False and "_load_error" not in r
    ]
    done = [r for r in reportable if r.get("status") == "done"]

    print("# LeanMoE-Quant 结果汇总\n")
    print(
        "共 %d 个 result.json，其中可报告 %d 个、已完成 %d 个。\n"
        % (len(results), len(reportable), len(done))
    )
    print("## 指标表\n")
    print(summary_table(results))
    print("\n## 字节预算\n")
    print(byte_budget_section(results))

    print("\n## 口径提醒\n")
    print(
        "- `[W]`（权重，与 ctx 无关）与 `[T]`（含 KV 读）是**两个分母**，不得混用"
        "（docs/reference/METRICS.md §3.2）"
    )
    print("- **host 占比是一等指标**：它决定量化收益的上限（docs/reference/METRICS.md §2.1）")
    print("- 未测的显示 `—`，不显示 0——`0` 是一个结论，`—` 才是未测")
    print("- 判据：`已实测` 需要 status=done **且** correctness.gate != FAIL")

    if problems:
        print("\n## schema 问题\n", file=sys.stderr)
        for p in problems:
            print("  ✗ %s" % p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
