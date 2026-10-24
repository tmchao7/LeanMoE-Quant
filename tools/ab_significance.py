#!/usr/bin/env python3
"""ab_significance — 对**已落盘**的配对 A/B 工件补算显著性（MATURITY_REVIEW R1）。

为什么存在：2026-10-24 之前落盘的 A/B 工件只有 wins/MAD，没有 p 值与区间。本工具
不重跑、不新测，只对盘上已有的配对差重算（硬规则 1：不产生新测量值）。新轮次
（runner 的 `interleaved_ab`、profile_graph v2）已内嵌同款显著性块，本工具用于
复盘旧工件与汇总。

支持三种工件形状：

    1. profile_graph v2 —— 顶层 `paired_wall.delta_ms`（逐对差已持久化）；
    2. profile_graph v1 —— 只有 `runs[]`（交替臂的逐 run p50），按臂分组重建配对；
    3. runner result.json —— `interleaved_ab.paired.delta_ms`（2026-10-24 起持久化；
       更早的 payload 只有摘要统计，无法重算，如实报 skip）。

用法：
    python -m tools.ab_significance results/fused_rope_ab.json [...] [--out results/ab_significance.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from benchmarks.ab_stats import paired_significance


def _finite_pairs(
    base: list[dict[str, Any]], treat: list[dict[str, Any]]
) -> tuple[list[float], int]:
    """按顺序配对两臂的 run 级 p50；非有限值跳过并计数（fail-closed，不静默变 0）。"""
    deltas: list[float] = []
    skipped = 0
    for a, b in zip(base, treat):
        try:
            da, db = float(a["wall_p50_ms"]), float(b["wall_p50_ms"])
        except (TypeError, ValueError, KeyError):
            skipped += 1
            continue
        if not (math.isfinite(da) and math.isfinite(db)):
            skipped += 1
            continue
        deltas.append(da - db)
    return deltas, skipped


def _extract(payload: dict[str, Any]) -> dict[str, Any] | None:
    """从任意形状的 A/B 工件里取出 (arm_a, arm_b, deltas, provenance)。取不出返回 None。"""
    # 1) runner result.json：interleaved_ab 可能在顶层或 _measurement 下
    holder = payload.get("interleaved_ab")
    if not isinstance(holder, dict):
        meas = payload.get("_measurement")
        if isinstance(meas, dict):
            holder = meas.get("interleaved_ab")
    if isinstance(holder, dict):
        paired = holder.get("paired") or {}
        deltas = paired.get("delta_ms")
        if isinstance(deltas, list) and deltas:
            arms = list(holder.get("arms", {})) or ["arm_a", "arm_b"]
            return {
                "arm_a": arms[0],
                "arm_b": arms[1] if len(arms) > 1 else "arm_b",
                "deltas": [float(d) for d in deltas],
                "provenance": "runner_persisted",
            }
        return None

    # 2) profile_graph v2：逐对差已持久化
    paired_wall = payload.get("paired_wall")
    if isinstance(paired_wall, dict) and isinstance(paired_wall.get("delta_ms"), list):
        runs = payload.get("runs", [])
        arm_names = []
        for r in runs:
            if r.get("arm") not in arm_names:
                arm_names.append(r.get("arm"))
        return {
            "arm_a": arm_names[0] if arm_names else "arm_a",
            "arm_b": arm_names[1] if len(arm_names) > 1 else "arm_b",
            "deltas": [float(d) for d in paired_wall["delta_ms"]],
            "provenance": "profile_graph_v2_persisted",
        }

    # 3) profile_graph v1：从交替 runs 重建（每轮 1 对 ⇒ n=rounds，如实标注）
    runs = payload.get("runs")
    if isinstance(runs, list) and runs and "arm" in runs[0]:
        arm_a = runs[0]["arm"]
        arm_b = next((r["arm"] for r in runs if r["arm"] != arm_a), None)
        if arm_b is None:
            return None
        pairs_only = [r for r in runs if r.get("role", "pair") != "tail_canary"]
        base = [r for r in pairs_only if r["arm"] == arm_a]
        treat = [r for r in pairs_only if r["arm"] == arm_b]
        deltas, skipped = _finite_pairs(base, treat)
        return {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "deltas": deltas,
            "provenance": "v1_reconstructed_from_runs",
            "skipped_invalid_runs": skipped,
        }
    return None


def analyze(path: Path) -> dict[str, Any]:
    """单工件 → 显著性块（或 skip 原因）。只读，不写 result.json。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    got = _extract(payload)
    if got is None:
        return {"source": str(path), "status": "skip", "reason": "无可重算的配对差（只有摘要统计）"}
    sig = paired_significance(got["deltas"], label_a=got["arm_a"], label_b=got["arm_b"])
    out: dict[str, Any] = {
        "source": str(path),
        "status": "ok",
        "provenance": got["provenance"],
        "arms": [got["arm_a"], got["arm_b"]],
        "significance": sig,
    }
    if got.get("skipped_invalid_runs"):
        out["skipped_invalid_runs"] = got["skipped_invalid_runs"]
    if got["provenance"] == "v1_reconstructed_from_runs":
        out["note"] = (
            "v1 工件只含 run 级 p50，每轮 1 对 ⇒ n=rounds；小样本下结论依据方向一致性与"
            "效应量（>MAD），不依据 p 值"
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payloads", nargs="+", type=Path, help="A/B 工件 JSON 路径")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="汇总 JSON 输出路径（如 results/ab_significance.json）",
    )
    args = parser.parse_args(argv)

    results = [analyze(p) for p in args.payloads]
    for r in results:
        if r["status"] != "ok":
            print("%-44s SKIP  %s" % (r["source"], r["reason"]))
            continue
        sig = r["significance"]
        verdict = "显著(p<0.05)" if sig["significant_at_0_05"] else "不显著"
        print(
            "%-44s [%s] %s vs %s  n=%d  median_delta=%+.4f ms  sign_p=%s  wilcoxon_p=%s(%s)  CI95=%s  %s"
            % (
                r["source"],
                r["provenance"],
                r["arms"][0],
                r["arms"][1],
                sig["n_pairs"],
                sig["median_delta_ms"] or 0.0,
                sig["sign_test_p"],
                sig["wilcoxon_p"],
                sig["wilcoxon_method"] or "-",
                sig["median_delta_ci95_ms"],
                verdict,
            )
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "generated_by": "tools/ab_significance.py",
                    "note": "对已落盘工件的**重算**（不新测）；v1 工件 n=rounds，小样本结论看方向与效应量，不看 p",
                    "results": results,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print("written: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
