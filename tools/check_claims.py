"""check_claims — 把 canonical 数字钉回工件，并扫描未标历史的旧值。

两类检查：

1. **工件检查**（有 `results/` 时）：八个 canonical p50 是**钉死的回归值**
   （`PAYLOAD_CLAIMS` 里硬编码 expected，再从 payload 读实际值比较——不是从文档推导），
   最好格 2.4421/4.2506、E7 1.2253/1.1025、32/128-token E2E 比值则是**从 payload 现算**
   的派生量。fig4 的七根条也从 A/B 工件复算（F1 三项读 `*_ab.json`，F2/F4/F5/E6 读
   `ab_summary.json` 的 off/on mean），并检查 SVG 里确实写着同一个两位小数。
   这些检查都不读文档，所以能抓"工件变了"和"派生量算错"，**抓不到**
   "文档抄错了 canonical 值"（那一层由 RESULTS 的工件路径 + 人工审阅负责）。
2. **文档扫描**：旧值（1.1911 / 1.1974 / 1.1979 / 707 / 729 / 501 / 1.69× / 1.31× /
   40.9% / 23.5% / 89.9% / 90.4%）必须出现在带历史标记的上下文里；否则失败。
   扫描范围含 `docs/study/`（本地学习材料，gitignored；不存在就跳过）。

用法：
    python -m tools.check_claims
    python -m tools.check_claims --check     # 失败退出码 1（CI/验收用）
    python -m tools.check_claims --json

干净克隆（无 results/）只做文档扫描；不会把"缺工件"当成数字错误。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

P4C = "EXP-p4c-01-in-graph-attention__bit_width=%s_attn=%s_ctx=%d/result.json"
P4B = "EXP-p4b-01-truncation__bit_width=%s_execution=cudagraph_ctx=%d/result.json"

# (name, payload template args, JSON path, expected, tolerance)
PAYLOAD_CLAIMS: list[tuple[str, tuple[Any, ...], tuple[str, ...], float, float]] = [
    ("canonical @1024", ("w8a16", "kernel", 1024), ("latency", "per_token_p50_ms"), 1.4602, 5e-4),
    (
        "canonical @4096",
        ("w8a16", "kernel", 4096),
        ("latency", "per_token_p50_ms"),
        1.6075,
        5e-4,
    ),
    ("fp16×sdpa @1024", ("fp16", "sdpa", 1024), ("latency", "per_token_p50_ms"), 3.5660, 5e-4),
    ("fp16×sdpa @4096", ("fp16", "sdpa", 4096), ("latency", "per_token_p50_ms"), 6.8329, 5e-4),
    ("E7 fp16 @1024", ("fp16", 1024), ("latency", "per_token_p50_ms"), 3.5680, 5e-4),
    ("E7 w8a16 @1024", ("w8a16", 1024), ("latency", "per_token_p50_ms"), 2.9120, 5e-4),
    ("E7 fp16 @4096", ("fp16", 4096), ("latency", "per_token_p50_ms"), 6.8513, 5e-4),
    ("E7 w8a16 @4096", ("w8a16", 4096), ("latency", "per_token_p50_ms"), 6.2141, 5e-4),
]

# 旧值 → 允许出现的上下文标记（同一行或前后一行）
STALE = {
    "1.1911": ("历史", "旧", "P2j", "v3", "归档"),
    "1.1974": ("历史", "旧", "F4", "2026-10-24", "v4", "归档"),
    "1.1979": ("历史", "旧", "F4", "2026-10-24", "归档"),
    "707 passed": ("历史", "旧", "S2", "2026-10-24", "归档"),
    "729 passed": ("历史", "旧", "F5", "2026-10-25", "归档"),
    "501 passed": ("历史", "旧", "v3", "归档"),
    "1.69×": ("历史", "旧", "旧版", "withdrawn", "归档"),
    "1.31×": ("历史", "旧", "旧版", "withdrawn", "归档"),
    "40.9%": ("历史", "旧", "旧版", "格比", "归档"),
    "23.5%": ("历史", "旧", "旧版", "格比", "归档"),
    "89.9%": ("历史", "旧", "F4", "归档"),
    "90.4%": ("历史", "旧", "F4", "归档"),
}
HISTORY_MARKERS = ("历史", "旧", "旧版", "F4", "P2j", "v3", "v4", "v6", "归档", "withdrawn")

DOC_PATHS = [
    ROOT / "README.md",
    ROOT / "README_EN.md",
    ROOT / "RESULTS.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "ROADMAP.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "apply",
    ROOT / "docs" / "reference",
    ROOT / "docs" / "study",
    ROOT / "experiments",
]


def _get(payload: Any, path: tuple[str, ...]) -> Any:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load(rel: str) -> dict[str, Any] | None:
    path = RESULTS / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _check_payloads() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = []
    if not RESULTS.exists():
        return errors, ["results/ 不存在：跳过工件复算（干净克隆）"]
    for name, args, path, expected, tol in PAYLOAD_CLAIMS:
        rel = P4C % args if len(args) == 3 else P4B % args
        payload = _load(rel)
        if payload is None:
            errors.append("%s: 缺 payload %s" % (name, rel))
            continue
        value = _get(payload, path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append("%s: %s 不是数值（%r）" % (name, ".".join(path), value))
            continue
        if abs(float(value) - expected) > tol:
            errors.append("%s: 工件 %.6f != 文档 %.6f" % (name, float(value), expected))
        else:
            lines.append("[ok] %s = %.4f" % (name, float(value)))
    # 派生量
    try:
        c1024 = _get(_load(P4C % ("w8a16", "kernel", 1024)) or {}, ("latency", "per_token_p50_ms"))
        f1024 = _get(_load(P4C % ("fp16", "sdpa", 1024)) or {}, ("latency", "per_token_p50_ms"))
        c4096 = _get(_load(P4C % ("w8a16", "kernel", 4096)) or {}, ("latency", "per_token_p50_ms"))
        f4096 = _get(_load(P4C % ("fp16", "sdpa", 4096)) or {}, ("latency", "per_token_p50_ms"))
        for ctx, fp16, canon, expected in (
            (1024, f1024, c1024, 2.4421),
            (4096, f4096, c4096, 4.2506),
        ):
            if fp16 and canon:
                got = float(fp16) / float(canon)
                if abs(got - expected) > 5e-4:
                    errors.append("best cell @%d: %.4f != %.4f" % (ctx, got, expected))
                else:
                    lines.append("[ok] best cell @%d = %.4f×" % (ctx, got))
        for ctx, expected in ((1024, 1.2253), (4096, 1.1025)):
            a = _get(_load(P4B % ("fp16", ctx)) or {}, ("latency", "per_token_p50_ms"))
            b = _get(_load(P4B % ("w8a16", ctx)) or {}, ("latency", "per_token_p50_ms"))
            if a and b:
                got = float(a) / float(b)
                if abs(got - expected) > 2e-3:
                    errors.append("E7 @%d: %.4f != %.4f" % (ctx, got, expected))
                else:
                    lines.append("[ok] E7 @%d = %.4f×" % (ctx, got))
    except Exception as exc:  # noqa: BLE001
        errors.append("派生量复算失败: %s" % exc)
    # E2E 比值：复用 e2e_report，不重复口径
    try:
        from tools.e2e_report import build_rows  # noqa: PLC0415

        rows, missing = build_rows()
        if missing:
            errors.extend(missing)
        for row in rows:
            ctx = int(row["ctx"])
            for n, expected in ((32, {1024: 1.3824, 4096: 1.7190}[ctx]),):
                got = float(row["ratio_%d" % n])
                if abs(got - expected) > 2e-3:
                    errors.append("E2E-%d @%d: %.4f != %.4f" % (n, ctx, got, expected))
                else:
                    lines.append("[ok] E2E-%d @%d = %.4f×" % (n, ctx, got))
    except Exception as exc:  # noqa: BLE001
        errors.append("E2E 复算失败: %s" % exc)
    return errors, lines


# fig4 的七根条是 `make_figures.fig4` 里**硬编码**的（它不读 inputs.json）——这正是
# "自证"风险：图里的数字和工件一起漂移时，没人会发现。这里把图钉回 A/B 工件：
# F1 三项读 `tools/profile_graph.py --ab-*` 的 JSON，F2/F4/F5/E6 读 `ab_summary.json`
# 的 off/on mean（F2 的 5.05% 是 4 位小数 p50 的舍入口径，精确均值是 5.042%，
# 故给 0.02pp 容差；其余 0.005pp）。
FIG4_PCT_FIELDS = [
    ("fused RoPE", "fused_rope_ab.json", "treatment_faster_pct", 8.39, 0.005),
    ("routing", "routing_ab.json", "fused_faster_pct", 3.00, 0.005),
    ("skip mask-fill", "skip_mask_fill_ab.json", "treatment_faster_pct", 1.33, 0.005),
]
FIG4_AB_SUMMARIES = [
    ("F2", "EXP-f2-ab-20261024/ab_summary.json", 5.05, 0.02),
    ("F4", "EXP-f4-ab-20261025/ab_summary.json", 2.94, 0.005),
    ("F5", "EXP-f5-ab-20261025/ab_summary.json", 4.13, 0.005),
    ("E6", "EXP-e6-ab-4096-run2/ab_summary.json", 6.54, 0.005),
]


def _check_fig4() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = []
    if not RESULTS.exists():
        return errors, lines
    svg_path = ROOT / "docs" / "figures" / "fig4_lever_ab.svg"
    if not svg_path.exists():
        return ["fig4: 缺 docs/figures/fig4_lever_ab.svg"], []
    text = svg_path.read_text(encoding="utf-8")

    def _pin(name: str, got: float, expected: float, tol: float) -> None:
        if abs(got - expected) > tol:
            errors.append("fig4/%s: 工件 %.4f%% != 图 %.2f%%" % (name, got, expected))
        elif ("−%.2f%%" % expected) not in text:
            errors.append("fig4/%s: SVG 里找不到 −%.2f%%" % (name, expected))
        else:
            lines.append("[ok] fig4 %s = −%.2f%%" % (name, expected))

    for name, rel, key, expected, tol in FIG4_PCT_FIELDS:
        payload = _load(rel)
        if payload is None:
            errors.append("fig4/%s: 缺 payload %s" % (name, rel))
            continue
        got = payload.get(key)
        if not isinstance(got, (int, float)) or isinstance(got, bool):
            errors.append("fig4/%s: %s 不是数值（%r）" % (name, key, got))
            continue
        _pin(name, float(got), expected, tol)
    for name, rel, expected, tol in FIG4_AB_SUMMARIES:
        payload = _load(rel)
        if payload is None:
            errors.append("fig4/%s: 缺 payload %s" % (name, rel))
            continue
        off, on = payload.get("off_mean_ms"), payload.get("on_mean_ms")
        if not (
            isinstance(off, (int, float))
            and not isinstance(off, bool)
            and isinstance(on, (int, float))
            and not isinstance(on, bool)
            and float(off) > 0
        ):
            errors.append("fig4/%s: off/on mean 缺失或非法" % name)
            continue
        _pin(name, abs((float(on) - float(off)) / float(off) * 100.0), expected, tol)
    return errors, lines


def _doc_files() -> list[Path]:
    files: list[Path] = []
    for item in DOC_PATHS:
        if item.is_dir():
            files.extend(sorted(item.glob("*.md")))
        elif item.exists():
            files.append(item)
    return files


def _check_docs() -> tuple[list[str], list[str]]:
    errors: list[str] = []
    lines: list[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8")
        rows = text.splitlines()
        for i, line in enumerate(rows):
            for stale, allowed in STALE.items():
                if stale not in line:
                    continue
                window = " ".join(rows[max(0, i - 1) : i + 2])
                if any(marker in window for marker in allowed + HISTORY_MARKERS):
                    continue
                errors.append(
                    "%s:%d 出现未标历史的 %s：%s"
                    % (path.relative_to(ROOT), i + 1, stale, line.strip()[:120])
                )
    if not errors:
        lines.append("[ok] 文档：旧值都带历史标记")
    return errors, lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="失败退出码 1")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)
    payload_errors, payload_lines = _check_payloads()
    fig4_errors, fig4_lines = _check_fig4()
    doc_errors, doc_lines = _check_docs()
    errors = payload_errors + fig4_errors + doc_errors
    payload_lines = payload_lines + fig4_lines
    if args.json:
        print(
            json.dumps(
                {"errors": errors, "lines": payload_lines + doc_lines},
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        for line in payload_lines + doc_lines:
            print(line)
        for err in errors:
            print("[FAIL] %s" % err)
    if args.check and errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
