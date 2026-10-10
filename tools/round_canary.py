#!/usr/bin/env python3
"""round_canary — 轮次金丝雀验收：把"漂移 >5% ⇒ 整轮 VOID"变成退出码（不靠自觉）。

纪律出处：AGENTS §6 共享卡协议、`docs/reference/METRICS.md` §2 —— 每轮首尾测同一格，
漂移 >5% 判 VOID：保留日志、不发布数字。2026-10-10 评审（M1）发现：只有
`tools/batch_sweep.py` 在代码里机械判 VOID，其余轮次脚本只打印 drift 后照常放行。
本工具让任何轮次脚本都能 `&&` 它来守住这条纪律：

    python -m tools.round_canary head_result.json tail_result.json
    python -m tools.round_canary A.json B.json --graph      # 图格附加 host_share 自检
    python -m tools.round_canary A.json B.json --mark       # 判 VOID 时把两份置 status=void（先归档）
    python -m tools.round_canary A.json B.json --out v.json # 机器可读 verdict

退出码：0 = 本轮有效（PASS）；1 = VOID（含一切失败情形，fail-closed）；2 = 用法错误。
fail-closed 的含义：金丝雀指标缺失、payload 读不出、head=0 —— 一律按 VOID 处理，
"缺金丝雀就静默放行"正是这个工具要堵的洞。

`--graph`（AGENTS §6）：cudagraph 格带共租污染的自检 —— 任一端
`attribution.host_share_pct > 5` ⇒ 该格被污染 ⇒ 整轮 VOID。

`--mark`：把被比较的 payload `status` 改为 `void`（先归档到同目录 `archive/`，规则 9）。
status=void 之后：`benchmarks.report` 不再把它算进"已完成"、`harness.is_done` 为 False
（下轮会重跑）；`latency` 段保留原值作诊断——可引用性的开关是 status，不是字段清空。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

CANARY_MAX_DRIFT_PCT = 5.0
GRAPH_HOST_SHARE_MAX_PCT = 5.0
DEFAULT_METRIC = ("latency", "per_token_p50_ms")


def _dig(payload: Any, path: tuple[str, ...]) -> Any:
    node = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def evaluate(
    head: Any,
    tail: Any,
    *,
    metric: tuple[str, ...] = DEFAULT_METRIC,
    max_drift_pct: float = CANARY_MAX_DRIFT_PCT,
    graph: bool = False,
) -> dict[str, Any]:
    """比较首尾两份 payload，返回 verdict（不抛异常——坏输入 = VOID）。"""
    reasons: list[str] = []
    head_ms = tail_ms = drift = None
    try:
        head_ms = _dig(head, metric)
        tail_ms = _dig(tail, metric)
    except Exception as exc:  # noqa: BLE001 — 任何异常都按 VOID
        reasons.append("指标读取异常: %s: %s" % (type(exc).__name__, exc))
    if head_ms is None or tail_ms is None:
        reasons.append(
            "金丝雀指标 %s 缺失（head=%r tail=%r）" % (".".join(metric), head_ms, tail_ms)
        )
    elif not isinstance(head_ms, (int, float)) or not isinstance(tail_ms, (int, float)):
        reasons.append("金丝雀指标不是数值（head=%r tail=%r）" % (head_ms, tail_ms))
    elif float(head_ms) == 0.0:
        reasons.append("head=0，无法计算漂移")
    else:
        drift = round(100.0 * (float(tail_ms) - float(head_ms)) / float(head_ms), 4)
        if abs(drift) > max_drift_pct:
            reasons.append("漂移 %.3f%% 超过阈值 %.1f%%" % (drift, max_drift_pct))
    if graph:
        for label, payload in (("head", head), ("tail", tail)):
            host = _dig(payload, ("attribution", "host_share_pct"))
            if host is None:
                reasons.append(
                    "%s 的 attribution.host_share_pct 缺失（--graph 自检无法通过）" % label
                )
            elif isinstance(host, (int, float)) and float(host) > GRAPH_HOST_SHARE_MAX_PCT:
                reasons.append(
                    "%s 的 host_share=%.1f%% > %.0f%%（疑似共租污染）"
                    % (label, float(host), GRAPH_HOST_SHARE_MAX_PCT)
                )
    return {
        "generated_by": "tools/round_canary.py",
        "metric": ".".join(metric),
        "head_value": head_ms,
        "tail_value": tail_ms,
        "drift_pct": drift,
        "max_drift_pct": max_drift_pct,
        "graph_check": graph,
        "verdict": "VOID" if reasons else "PASS",
        "reasons": reasons,
    }


def mark_void(path: Path) -> dict[str, Any]:
    """把 payload 置 status=void（先归档到同目录 archive/，规则 9）。返回归档路径。"""
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    archive_dir = path.parent / "archive"
    archive_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = archive_dir / ("void_%s_%s" % (path.name, stamp))
    n = 1
    while target.exists():
        target = archive_dir / ("void_%s_%s_%d" % (path.name, stamp, n))
        n += 1
    target.write_text(original, encoding="utf-8")
    payload["status"] = "void"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="轮次金丝雀验收：漂移 >5%（或 --graph 自检不过）⇒ 退出码 1 = 整轮 VOID"
    )
    ap.add_argument("head", help="首轮金丝雀 result.json（先测的那份）")
    ap.add_argument("tail", help="末轮金丝雀 result.json（后测的那份）")
    ap.add_argument(
        "--metric", default=".".join(DEFAULT_METRIC), help="点分路径，默认 latency.per_token_p50_ms"
    )
    ap.add_argument(
        "--max-drift",
        type=float,
        default=CANARY_MAX_DRIFT_PCT,
        help="漂移阈值（%%），默认 %.1f" % CANARY_MAX_DRIFT_PCT,
    )
    ap.add_argument(
        "--graph",
        action="store_true",
        help="cudagraph 格自检：任一端 host_share_pct > %.0f ⇒ VOID" % GRAPH_HOST_SHARE_MAX_PCT,
    )
    ap.add_argument(
        "--mark", action="store_true", help="判 VOID 时把两份 payload 置 status=void（先归档）"
    )
    ap.add_argument("--out", default=None, help="把 verdict JSON 另存到该路径")
    args = ap.parse_args(argv)

    metric = tuple(args.metric.split("."))
    loaded: dict[str, Any] = {}
    for label, path in (("head", args.head), ("tail", args.tail)):
        try:
            loaded[label] = _load(path)
        except Exception as exc:  # noqa: BLE001 — 读不出 = VOID（fail-closed）
            loaded[label] = None
            print(
                "[canary] !! %s payload 读不出（%s: %s）—— 按 VOID 处理"
                % (label, type(exc).__name__, exc),
                file=sys.stderr,
            )

    verdict = evaluate(
        loaded["head"],
        loaded["tail"],
        metric=metric,
        max_drift_pct=args.max_drift,
        graph=args.graph,
    )
    verdict["head_path"] = args.head
    verdict["tail_path"] = args.tail

    if verdict["verdict"] == "VOID" and args.mark:
        for path in (args.head, args.tail):
            try:
                archive = mark_void(Path(path))
                verdict.setdefault("marked_void", []).append(
                    {"path": path, "archive": str(archive)}
                )
            except Exception as exc:  # noqa: BLE001
                verdict.setdefault("mark_errors", []).append(
                    "%s: %s: %s" % (path, type(exc).__name__, exc)
                )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if verdict["verdict"] == "PASS":
        print(
            "[canary] PASS  %s: %s → %s（drift %.3f%% ≤ %.1f%%）"
            % (
                verdict["metric"],
                verdict["head_value"],
                verdict["tail_value"],
                verdict["drift_pct"] or 0.0,
                args.max_drift,
            )
        )
        return 0
    print("[canary] VOID —— 整轮作废，任何数字不得发布：", file=sys.stderr)
    for reason in verdict["reasons"]:
        print("  - %s" % reason, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
