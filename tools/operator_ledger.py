"""operator_ledger — S2 算子选择的预注册规则的机械化执行器。

规则（预注册，见 docs/handoff/HANDOFF_2026-10-24.md §5 与
docs/handoff/ADOPTION_SYNC_2026-10-24.md §7）：

    1. 取（刷新后的）`EXP-p4c-01__bit_width=w8a16_attn=kernel_ctx=1024` 的
       `bottleneck.top_kernels`；
    2. `gap = 实测 − 必要字节 / 实测带宽`（必要字节由模型形状固定，
       F1/F2 不改变它 —— 见 docs/FUSION_OPPORTUNITIES.md §1.2）；
    3. 只留 `gap ≥ 30 µs`；`可捕获量 = gap × 捕获率`
       （位精确延迟消除 0.9 / 配置改善 0.3 / 需精度门 0.5）；选最大者。

★ 纪律：**刷新轮之前的 payload 跑出的账本只用于核对与演练，不得用于挑选 S2**
（FUSION_OPPORTUNITIES §0 ★：采纳轮跑完后必须整表重算，不要用旧账本挑 S2）。

用法：

    python -m tools.operator_ledger results/EXP-p4c-01-...__ctx=1024/result.json [--json out.json]
    python -m tools.operator_ledger <cell-dir-or-result.json> ...   # 可多格并排

必要字节表（OLMoE-1B-7B：16 层 / hidden 2048 / 64 专家 top-8 / w8 group 128，
形状定死、不随刷新变；出处 docs/FUSION_OPPORTUNITIES.md §1.2，按 2026-10-10
隔离复算；带宽默认取 results/machine_probe.json 的实测只读值）：
    自研 kernel 按名字匹配；`gemvx` 按 occurrences 区分 lm_head（1 次）与 router
    （16 次）；`_attn_split_kernel` 的 KV 读取随 ctx 变化（1024/4096 实测两档）。
    launch 主导的家族（KV 写回 index_copy+cat、残差 add）必要字节 < 1 MB，
    地板按字节算但可忽略 —— 它们是"每次 launch 多少 µs"的账（§1.3 决策口径）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# 必要字节（MB / step）：形状定死。self-kernel 出自 FUSION_OPPORTUNITIES §1.2；
# launch 主导家族的字节按 32 次 × 8 KB（KV 写回）与 32 次 × 12 KB（残差 add）计。
BYTES_MB_PER_STEP: dict[str, float] = {
    "gate_up_act": 545.3,
    "down_sum": 272.6,
    "dequant_gemm": 272.6,  # F2 前后同总量（64×4.26 → 16×12.78 + 16×4.26）
    "lm_head": 206.0,
    "router_gemv": 4.2,  # 16 × 64×2048×2B
    "attn_split@1024": 134.2,
    "attn_split@4096": 536.8,
    "kv_write_idx": 0.25,  # 32 × 8 KB
    "kv_write_cat": 0.25,  # 32 × 8 KB
    "residual_add": 0.375,  # 32 × 12 KB
}

# 捕获类别 → 预注册捕获率（HANDOFF §5）。
CAPTURE_LATENCY_REMOVAL = 0.9  # 位精确的 launch/延迟消除（F3 残差 epilogue、F4 KV 直写）
CAPTURE_CONFIG = 0.3  # 配置/occupancy 改善（自研 kernel 的达成率缺口）
CAPTURE_NEEDS_GATE = 0.5  # 需先过精度门（F5 lm_head 量化 —— 新 GATE_CITATIONS 行，先问用户）

# top_kernels 的长模板名 → (账本键, 类别, 备注)。"__exclude__" = 只列出不进排名。
NAME_RULES: list[tuple[str, str, str, str]] = [
    ("_gate_up_act_kernel_q", "gate_up_act", "config", "F2/F3 之外的自研最大头（已 78.8%）"),
    ("_down_sum_kernel_q", "down_sum", "config", "F3 的 epilogue 载体"),
    (
        "_dequant_gemm_kernel",
        "dequant_gemm",
        "config",
        "F2 本轮已采纳其 config 改善；剩余缺口属新假设，不进排名",
    ),
    ("_attn_split_kernel", "attn_split", "config", "bytes 随 ctx"),
    ("index_elementwise", "kv_write_idx", "latency", "F4 KV 直写的一半"),
    ("CatArrayBatchedCopy", "kv_write_cat", "latency", "F4 KV 直写的另一半"),
    ("CUDAFunctor_add", "residual_add", "latency", "F3 残差 epilogue"),
    ("gemvx", "__gemvx__", "", "按 occurrences 区分 lm_head / router"),
    ("_rms_norm_kernel", "__exclude__", "", "fused RMSNorm 已采纳"),
    ("sbtopk", "__exclude__", "", "fused route 已采纳（F1）"),
    ("bitonicSortKV", "__exclude__", "", "fused route 已采纳（F1）"),
    ("softmax_warp_forward", "__exclude__", "", "attn 内 softmax"),
]


def _classify(name: str, occ: int | None) -> tuple[str, str, str]:
    for needle, key, cls, note in NAME_RULES:
        if needle in name:
            if key == "__gemvx__":
                if occ == 1:
                    return "lm_head", "gate", "F5：需新 GATE_CITATIONS 行（先问用户）"
                return "router_gemv", "exclude", "router gemv，gap 必然微小"
            return key, cls, note
    return "__exclude__", "", "未入账本表（列出不排名）"


def _floor_ms(bytes_mb: float, bandwidth_gbps: float) -> float:
    return bytes_mb / 1e3 / bandwidth_gbps * 1e3  # MB / (GB/s) → ms


def _load_payload(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    if d.get("status") != "done":
        raise SystemExit("拒绝：%s 的 status=%s（只对 done 格算账本）" % (path, d.get("status")))
    for key in ("latency", "bottleneck", "config"):
        if not d.get(key):
            raise SystemExit("拒绝：%s 缺 %s 段" % (path, key))
    return d


def ledger_for(payload_path: Path, bandwidth_gbps: float, min_gap_us: float) -> dict[str, Any]:
    d = _load_payload(payload_path)
    p50 = d["latency"]["per_token_p50_ms"]
    cfg, bk = d["config"], d["bottleneck"]
    ctx = int(cfg.get("ctx") or 1024)
    rows: list[dict[str, Any]] = []
    for k in bk.get("top_kernels") or []:
        name = k.get("name") or ""
        occ = k.get("occurrences_per_token")
        measured = k.get("self_ms_per_token")
        if measured is None:
            continue
        key, cls, note = _classify(name, occ)
        if key == "dequant_gemm":
            # 预注册预演（HANDOFF §5）里它不进 S2 排名：F2（packed q/k/v）就是它的
            # config 改善、在本轮采纳；剩下的 ~13% + o_proj 独立调用属新假设
            # （FUSION_OPPORTUNITIES §F2），按"一次一个变量"不与 S2 混测。
            in_ranking = False
        else:
            in_ranking = None  # 占位，下面按 gap 重算
        if key == "attn_split":
            key = "attn_split@%d" % ctx if ctx in (1024, 4096) else "__exclude__"
            if key == "__exclude__":
                note = "ctx=%d 无字节表（只有 1024/4096 两档）" % ctx
        bytes_mb = BYTES_MB_PER_STEP.get(key)
        floor = _floor_ms(bytes_mb, bandwidth_gbps) if bytes_mb is not None else None
        gap_us = (measured - floor) * 1e3 if floor is not None else None
        capture = {
            "latency": CAPTURE_LATENCY_REMOVAL,
            "config": CAPTURE_CONFIG,
            "gate": CAPTURE_NEEDS_GATE,
        }.get(cls)
        capturable_us = gap_us * capture if (gap_us is not None and capture is not None) else None
        if in_ranking is None:
            in_ranking = bool(capturable_us is not None and gap_us >= min_gap_us)
        rows.append(
            {
                "family": key,
                "name": name[:72],
                "occ": occ,
                "measured_ms": measured,
                "bytes_mb": bytes_mb,
                "floor_ms": floor,
                "gap_us": gap_us,
                "capture_class": cls or None,
                "capturable_us": capturable_us,
                "note": note,
                "in_ranking": in_ranking,
            }
        )
    # 组合候选（预注册预演的两个"多 kernel 合一"项，单独汇总）
    by_family = {r["family"]: r for r in rows if r["family"] in BYTES_MB_PER_STEP}
    combos: dict[str, dict[str, Any]] = {}
    if {"kv_write_idx", "kv_write_cat"} <= by_family.keys():
        kv = by_family["kv_write_idx"]
        cat = by_family["kv_write_cat"]
        combos["F4_KV_DIRECT_WRITE"] = {
            "families": ["kv_write_idx", "kv_write_cat"],
            "measured_ms": kv["measured_ms"] + cat["measured_ms"],
            "gap_us": kv["gap_us"] + cat["gap_us"],
            "capturable_us": (kv["gap_us"] + cat["gap_us"]) * CAPTURE_LATENCY_REMOVAL,
        }
    if "residual_add" in by_family:
        add = by_family["residual_add"]
        combos["F3_RESIDUAL_EPILOGUE"] = {
            "families": ["residual_add"],
            "measured_ms": add["measured_ms"],
            "gap_us": add["gap_us"],
            "capturable_us": add["gap_us"] * CAPTURE_LATENCY_REMOVAL,
        }
    ranked = sorted((r for r in rows if r["in_ranking"]), key=lambda r: -(r["capturable_us"] or 0))
    combo_ranked = sorted(
        (c for c in combos.values() if c["gap_us"] >= min_gap_us),
        key=lambda c: -(c["capturable_us"] or 0),
    )
    return {
        "payload": str(payload_path),
        "cell": payload_path.parent.name,
        "exp_id": d.get("exp_id"),
        "config": {
            k: cfg.get(k) for k in ("bit_width", "attn", "execution", "moe", "ctx", "rmsnorm")
        },
        "fusion_switches": {
            k: cfg.get(k)
            for k in (
                "fused_rope",
                "fused_route",
                "skip_mask_fill",
                "packed_qkv",
                "packed_qkv_effective",
            )
        },
        "p50_ms": p50,
        "kernels_per_step": bk.get("kernels_per_step"),
        "bandwidth_gbps": bandwidth_gbps,
        "min_gap_us": min_gap_us,
        "rows": rows,
        "combos": combos,
        "ranking_single": ranked,
        "ranking_combo": combo_ranked,
        "caveat": (
            "账本口径：必要字节由模型形状固定（FUSION_OPPORTUNITIES §1.2）；"
            "gap<%.0f µs 与未入表 family 不进排名。若 payload 的融合开关是未记录/False"
            "（刷新轮尚未跑或刷新前测量），此账本只是刷新前基线 —— 不得用于挑 S2。" % min_gap_us
        ),
    }


def print_ledger(led: dict[str, Any]) -> None:
    print(
        "== 格：%s（p50=%.4f ms, %s kernels/step, 带宽 %.1f GB/s）=="
        % (led["cell"], led["p50_ms"], led["kernels_per_step"], led["bandwidth_gbps"])
    )
    print("   switches: %s" % led["fusion_switches"])
    hdr = "%-14s %5s %9s %8s %8s %8s %6s %8s  %s"
    print(hdr % ("family", "occ", "实测ms", "字节MB", "地板ms", "gapµs", "类别", "可捕µs", "备注"))
    for r in sorted(led["rows"], key=lambda r: -(r["measured_ms"] or 0)):
        if r["family"] == "__exclude__":
            continue
        print(
            ("%-14s %5s %9.4f %8s %8.4f %8s %6s %8s  %s")
            % (
                r["family"],
                r["occ"],
                r["measured_ms"],
                "—" if r["bytes_mb"] is None else "%.1f" % r["bytes_mb"],
                r["floor_ms"] if r["floor_ms"] is not None else 0.0,
                "—" if r["gap_us"] is None else "%.1f" % r["gap_us"],
                r["capture_class"] or "—",
                "—" if r["capturable_us"] is None else "%.1f" % r["capturable_us"],
                r["note"] or "",
            )
        )
    print("\n-- 排名（gap ≥ %.0f µs，按可捕获量降序）--" % led["min_gap_us"])
    print("   单 family：")
    for r in led["ranking_single"]:
        print(
            "     %-14s gap %6.1f µs × %s = %6.1f µs"
            % (r["family"], r["gap_us"], r["capture_class"], r["capturable_us"])
        )
    print("   组合候选：")
    for c in led["ranking_combo"]:
        print(
            "     %-22s gap %6.1f µs × 0.9 = %6.1f µs"
            % (c_key(led, c), c["gap_us"], c["capturable_us"])
        )
    if not led["ranking_combo"]:
        print("     （无）")
    print("\n注：%s" % led["caveat"])


def c_key(led: dict[str, Any], c: dict[str, Any]) -> str:
    for name, cand in led["combos"].items():
        if cand is c:
            return name
    return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("payloads", nargs="+", help="result.json 或其所在格目录")
    ap.add_argument(
        "--bandwidth",
        type=float,
        default=1560.3,
        help="实测可达带宽 GB/s（默认 1560.3 = machine_probe 只读值；硬规则 4）",
    )
    ap.add_argument(
        "--min-gap-us", type=float, default=30.0, help="进排名的 gap 下限（预注册 30 µs）"
    )
    ap.add_argument("--json", dest="json_out", help="把账本写到该 JSON 文件")
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for p in args.payloads:
        path = Path(p)
        if path.is_dir():
            path = path / "result.json"
        if not path.is_file():
            raise SystemExit("找不到 payload：%s" % path)
        paths.append(path)

    ledgers = [ledger_for(p, args.bandwidth, args.min_gap_us) for p in paths]
    for led in ledgers:
        print_ledger(led)
        print()
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(ledgers, indent=2, ensure_ascii=False))
        print("wrote", args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
