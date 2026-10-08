"""summarize_results — 生成三张汇总表（**只读**，不产出新的测量值）。

    python -m tools.summarize_results                 # 打印三张表
    python -m tools.summarize_results --json out.json # 落盘

三张表：

    1. **micro vs 端到端**：kernel 级加速比 vs 步级加速比，以及中间的 gap。
       gap 是本项目的核心结论（"量化收益被什么截断"），所以必须**同轮**呈现。
    2. **batch 曲线**：吞吐-延迟曲线。★ **单列**，绝不与历史 batch=1 数字混表或
       相除（硬规则 3）。表头必须带轮次标识。
    3. **精度 delta**：正确性指标（KL / top-1 / PPL / 任务级）与 fp16 的差。
       未测的一律显示 `未测`，**不显示 0**。

★ 纪律（对应验收标准"所有数字都有样本量、条件、上下界、待验证标签"）：
    - 每个数字旁边必须能看到 **n（样本量）**；
    - 有条件（ctx / dtype / GPU / 轮次）；
    - 有上下界（stddev 或 p95–p99 区间）；
    - 有 `kind`（measured / estimate / planned）—— estimate 与 planned 不得伪装成 measured。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any

NA = "未测"


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return NA
    if isinstance(value, float):
        return ("%%.%df" % nd) % value
    return str(value)


def _load(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 表 1：micro vs 端到端
# ---------------------------------------------------------------------------
def table_micro_vs_e2e(results_dir: str = "results") -> dict[str, Any]:
    """从已落盘工件里取 micro 与端到端两套数，**不做任何新的除法之外**的推导。"""
    rows: list[dict[str, Any]] = []
    ncu = _load(os.path.join(results_dir, "ncu_evidence", "isolated.json"))
    if ncu:
        for name, rec in (ncu.get("subjects") or {}).items():
            att = rec.get("attainment_pct_cold_best")
            rows.append(
                {
                    "kernel": name,
                    "level": "micro（隔离一次调用）",
                    "value": _fmt(att, 1) if att is not None else NA,
                    "value_kind": "measured" if att is not None else "planned",
                    "unit": "% of measured achievable bandwidth（达成率）",
                    "n": (rec.get("timing") or {}).get("cold_single", {}).get("n"),
                    "condition": "cold（L2 flush 256 MB）− 事件地板；%s" % ncu.get("device", "?"),
                    "spread": None,
                    "source": "results/ncu_evidence/isolated.json",
                }
            )

    for path in sorted(glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True)):
        payload = _load(path)
        if not payload:
            continue
        lat = payload.get("latency") or {}
        cfg = payload.get("config") or {}
        if lat.get("per_token_p50_ms") is None:
            continue
        rows.append(
            {
                "kernel": payload.get("exp_id", "?"),
                "level": "end-to-end（一个 decode 步）",
                "value": _fmt(lat.get("per_token_p50_ms")),
                "value_kind": "measured",
                "unit": "ms / token",
                "n": (lat.get("samples") or {}).get("per_token_ms", []) or None,
                "condition": "%s / ctx=%s / batch=%s"
                % (cfg.get("bit_width") or "?", cfg.get("ctx"), cfg.get("batch")),
                "spread": {
                    "p99_ms": lat.get("per_token_p99_ms"),
                    "stddev_note": "samples 全量落盘（见 latency.samples）",
                },
                "source": os.path.relpath(path, results_dir),
            }
        )
    return {
        "title": "表 1：micro vs 端到端",
        "note": (
            "两张**不同层级**的数，不是同一条链。gap 的存在本身就是结论 "
            "（权重只占一步 42% ⇒ 量化上限 1.74×）。不得把两列相除当成'效率'。"
        ),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 表 2：batch 曲线（单列）
# ---------------------------------------------------------------------------
def table_batch_curve(results_dir: str = "results") -> dict[str, Any]:
    sweep = _load(os.path.join(results_dir, "batch_sweep.json"))
    if not sweep:
        return {
            "title": "表 2：batch 曲线",
            "status": NA,
            "note": "results/batch_sweep.json 不存在 —— 该曲线**从未测量**（METRICS §6 G2）。",
            "rows": [],
        }
    if sweep.get("status") != "done":
        return {
            "title": "表 2：batch 曲线",
            "status": sweep.get("status", NA),
            "note": sweep.get("note", ""),
            "rows": [],
        }
    return {
        "title": "表 2：batch 曲线（**单列**，不与历史数字混表）",
        "round": {
            "device": sweep.get("device"),
            "ctx": sweep.get("ctx"),
            "dtype": sweep.get("dtype"),
            "warmup": sweep.get("warmup"),
            "timed": sweep.get("timed"),
            "repeats": sweep.get("repeats"),
        },
        "canary_drift_pct": (sweep.get("canary") or {}).get("drift_pct"),
        "cross_round_comparison_allowed": False,
        "rows": sweep.get("rows") or [],
    }


# ---------------------------------------------------------------------------
# 表 3：精度 delta
# ---------------------------------------------------------------------------
def table_accuracy_delta(results_dir: str = "results") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "result.json"), recursive=True)):
        payload = _load(path)
        if not payload:
            continue
        corr = payload.get("correctness") or {}
        cfg = payload.get("config") or {}
        if corr.get("kl") is None and corr.get("top1_agreement") is None:
            continue
        ppl = corr.get("ppl_this")
        rows.append(
            {
                "exp_id": payload.get("exp_id", "?"),
                "config": "%s g=%s ctx=%s"
                % (cfg.get("bit_width"), cfg.get("group_size"), cfg.get("ctx")),
                "kl": _fmt(corr.get("kl"), 6),
                "top1": _fmt(corr.get("top1_agreement"), 4),
                "ppl_this": _fmt(ppl, 4) if ppl is not None else NA,
                "ppl_note": corr.get("ppl_note") or NA,
                "gate": corr.get("gate"),
                "kind": "measured",
                "source": os.path.relpath(path, results_dir),
            }
        )

    task = _load(os.path.join(results_dir, "task_accuracy.json"))
    if task and task.get("status") == "done":
        rows.append(
            {
                "exp_id": "task-level",
                "config": "MMLU 200 题子集 + GSM8K 100 题子集",
                "kl": NA,
                "top1": NA,
                "ppl_this": NA,
                "ppl_note": "任务级精度见 results/task_accuracy.json",
                "gate": task.get("gate", NA),
                "kind": "measured",
                "source": "results/task_accuracy.json",
            }
        )
    else:
        rows.append(
            {
                "exp_id": "task-level",
                "config": "MMLU / GSM8K",
                "kl": NA,
                "top1": NA,
                "ppl_this": NA,
                "ppl_note": "**未测**（METRICS §6 G1）—— 全仓库零数据",
                "gate": NA,
                "kind": "planned",
                "source": "—",
            }
        )
    return {
        "title": "表 3：精度 delta（对 fp16 参考）",
        "note": "未测一律显示 %s，**不显示 0** —— 0 是结论，%s 才是未测。" % (NA, NA),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    tables = [
        table_micro_vs_e2e(args.results_dir),
        table_batch_curve(args.results_dir),
        table_accuracy_delta(args.results_dir),
    ]

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"tables": tables}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print("wrote %s" % args.json)
        return 0

    for t in tables:
        print("\n== %s ==" % t["title"])
        if t.get("note"):
            print("   %s" % t["note"])
        if not t.get("rows"):
            print("   (无行：%s)" % t.get("status", "空"))
            continue
        keys = [k for k in t["rows"][0] if k not in ("n", "samples_ms", "spread", "source")]
        print("   " + " | ".join(keys))
        for r in t["rows"]:
            print("   " + " | ".join(_fmt(r.get(k), 4) for k in keys))
    print(
        "\n★ 每个数字都应能看到：样本量 n / 条件（ctx·dtype·GPU·轮次）/ 上下界 / "
        "kind(measured·estimate·planned)。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
