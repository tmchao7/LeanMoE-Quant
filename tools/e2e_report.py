"""e2e_report — 从 canonical payload 反算 TTFT + N-token 端到端延迟。

这不是新测量：只读 `results/` 里已有的 `EXP-p4c-01` 四格，算

    E2E(N) = TTFT_p50 + N × per_token_p50

并给出 canonical（w8a16+lm_head+KV8 × kernel）相对 fp16×sdpa 的比值。
用途：把"2.44×/4.25× 是稳态 decode 步、不是端到端请求"这句话变成一个可复算的表。

用法：
    python -m tools.e2e_report            # markdown 表
    python -m tools.e2e_report --check    # 缺工件/非有限值 ⇒ 退出码 1
    python -m tools.e2e_report --json

口径边界：
    - TTFT 是 payload 的 `latency.ttft_p50_ms`（prefill 未优化；量化格的 prefill
      走 separate dequant，所以 canonical 的 TTFT 反而更高）；
    - per_token 是 32 步 decode 的 p50；E2E 假设后续 N 步同 p50；
    - 只在 results/ 存在时可用；干净克隆会打印 skipped，不编数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

CASES: dict[str, dict[int, str]] = {
    "canonical": {
        1024: "EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=1024/result.json",
        4096: "EXP-p4c-01-in-graph-attention__bit_width=w8a16_attn=kernel_ctx=4096/result.json",
    },
    "fp16_sdpa": {
        1024: "EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=sdpa_ctx=1024/result.json",
        4096: "EXP-p4c-01-in-graph-attention__bit_width=fp16_attn=sdpa_ctx=4096/result.json",
    },
}
TOKENS = (32, 128)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _load(rel: str) -> dict[str, Any] | None:
    path = RESULTS / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _case(ctx: int) -> dict[str, Any] | None:
    rows: dict[str, Any] = {}
    for arm, by_ctx in CASES.items():
        payload = _load(by_ctx[ctx])
        if payload is None:
            return None
        latency = payload.get("latency") or {}
        ttft = latency.get("ttft_p50_ms")
        per_token = latency.get("per_token_p50_ms")
        if not (_finite(ttft) and _finite(per_token)):
            return None
        rows[arm] = {
            "ttft_ms": float(ttft),
            "per_token_ms": float(per_token),
            "payload": by_ctx[ctx],
        }
    return rows


def build_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for ctx in (1024, 4096):
        cases = _case(ctx)
        if cases is None:
            missing.append("ctx=%d: canonical / fp16×sdpa payload 缺失或 latency=null" % ctx)
            continue
        canonical = cases["canonical"]
        baseline = cases["fp16_sdpa"]
        row: dict[str, Any] = {
            "ctx": ctx,
            "canonical_ttft_ms": canonical["ttft_ms"],
            "canonical_per_token_ms": canonical["per_token_ms"],
            "baseline_ttft_ms": baseline["ttft_ms"],
            "baseline_per_token_ms": baseline["per_token_ms"],
            "payloads": {
                "canonical": canonical["payload"],
                "fp16_sdpa": baseline["payload"],
            },
        }
        for n in TOKENS:
            e2e_canonical = canonical["ttft_ms"] + n * canonical["per_token_ms"]
            e2e_baseline = baseline["ttft_ms"] + n * baseline["per_token_ms"]
            row["e2e_%d_ms" % n] = e2e_canonical
            row["baseline_e2e_%d_ms" % n] = e2e_baseline
            row["ratio_%d" % n] = e2e_baseline / e2e_canonical
        rows.append(row)
    return rows, missing


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    print("| ctx | arm | TTFT p50 (ms) | decode p50 (ms/token) | E2E-32 (ms) | E2E-128 (ms) |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| %d | canonical | %.4f | %.4f | %.2f | %.2f |"
            % (
                row["ctx"],
                row["canonical_ttft_ms"],
                row["canonical_per_token_ms"],
                row["e2e_32_ms"],
                row["e2e_128_ms"],
            )
        )
        print(
            "| %d | fp16 × sdpa | %.4f | %.4f | %.2f | %.2f |"
            % (
                row["ctx"],
                row["baseline_ttft_ms"],
                row["baseline_per_token_ms"],
                row["baseline_e2e_32_ms"],
                row["baseline_e2e_128_ms"],
            )
        )
    print()
    print("| ctx | E2E-32 ratio (fp16÷canonical) | E2E-128 ratio |")
    print("| ---: | ---: | ---: |")
    for row in rows:
        print("| %d | %.4f× | %.4f× |" % (row["ctx"], row["ratio_32"], row["ratio_128"]))
    print()
    print("> decode-step ratio 2.44×/4.25× is steady-state only; prefill/TTFT is not optimized.")
    print("> Sources: `results/EXP-p4c-01-in-graph-attention__...` four payloads.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="缺工件/非有限值 ⇒ 退出码 1")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)
    rows, missing = build_rows()
    if args.json:
        print(json.dumps({"rows": rows, "missing": missing}, indent=2, ensure_ascii=False))
    else:
        if rows:
            _print_markdown(rows)
        for item in missing:
            print("[missing] %s" % item)
        if not rows and not missing:
            print("[skipped] results/ 不存在——干净克隆不编端到端数字。")
    if args.check and (missing or len(rows) != 2):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
