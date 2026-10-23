#!/usr/bin/env python3
"""profile_graph — 当前 decode 路径的 in-graph kernel 分解（决定下一口咬哪里）。

背景（2026-10-10）：graph 格的 `result.json` 里 `top_kernels` 全是 0.000，而带真实
per-kernel ms 的工件全是 E2 时代 hf 旧路径（147/56 ms gpu_total）——没有任何工件能
回答"3.01 ms 的一步里谁占大头"。本工具直接调 `benchmarks.runner.run_cudagraph()`，
其内部的 `measure_graph_gap` 自带 wall + CUDA event + profiler kernel_table，
把 `res["gap"]` 原样落盘并打印 top 表。

★ **不写任何 result.json**：这是定位工具（同 batch_sweep/occupancy_probe 一档），
  不是实验格子——不触碰硬规则 9，也不产生可引用的"性能结论"（结论要走矩阵+记录）。

用法（5090）：
    python -m tools.profile_graph --model <MODEL> --json results/graph_profile.json
    python -m tools.profile_graph --model <MODEL> --bit-width fp16 --dequant none \
        --attn sdpa --json results/graph_profile.fp16.json          # 对照臂

退出码：0 = measured；1 = 构建/捕获/测量失败（原因进 stderr 与 JSON 的 error 字段）。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.runner import run_cudagraph
from tools.round_canary import drift_pct, finite_number, json_safe

ABLATE_ARM_ENV = "LEMOQ_FUSED_ROUTE"
MASK_FILL_ARM_ENV = "LEMOQ_SKIP_MASK_FILL"
ROPE_ARM_ENV = "LEMOQ_FUSED_ROPE"
PACKED_QKV_ENV = "LEMOQ_PACKED_QKV"


def _invoke(args: argparse.Namespace) -> dict[str, Any]:
    """按 CLI 参数跑一次 run_cudagraph（单发与 A/B 共用）。"""
    quantized = args.bit_width != "fp16"
    return run_cudagraph(
        model_id=args.model,
        ctx=args.ctx,
        batch=args.batch,
        max_new_tokens=args.max_new_tokens,
        warmup=args.warmup,
        timed=args.timed,
        dtype_name=args.dtype,
        moe=args.moe,
        attn=args.attn,
        quant_cfg=(
            {
                "bit_width": args.bit_width,
                "group_size": args.group_size,
                "ctx": args.ctx,
                "dequant": args.dequant,
            }
            if quantized
            else None
        ),
        rmsnorm=args.rmsnorm,
    )


def _extract_ab(res: dict[str, Any]) -> dict[str, Any]:
    gap = res.get("gap") or {}
    correctness = res.get("correctness") or {}
    return {
        "wall_p50_ms": gap.get("wall_p50_ms"),
        "gpu_busy_ms": gap.get("gpu_busy_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "tokens": correctness.get("graph_output_tokens"),
        "seed_token": correctness.get("graph_seed_token"),
    }


def _run_env_ab(
    args: argparse.Namespace,
    *,
    env_var: str,
    arm_a: str,
    arm_b: str,
    generated_by: str,
    note: str,
) -> dict[str, Any]:
    """Interleave adjacent pairs, then a separate same-arm tail canary.

    `measured` means protocol-valid diagnostic evidence, never adoption approval.
    The tail is checked for correctness/contamination but excluded from arm statistics.
    """
    rounds = args.ab_rounds
    if type(rounds) is not int or rounds < 1:
        raise ValueError("ab_rounds must be a positive integer")
    if type(args.max_new_tokens) is not int or args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if env_var == MASK_FILL_ARM_ENV and args.attn != "kernel":
        raise ValueError("skip-mask-fill requires the mask-independent kernel backend")
    if env_var == PACKED_QKV_ENV and args.attn != "kernel":
        raise ValueError("packed q/k/v requires attn=kernel (the F2 first-round scope)")
    seq = [arm_a, arm_b] * rounds + [arm_a]
    switches = (ABLATE_ARM_ENV, MASK_FILL_ARM_ENV, ROPE_ARM_ENV, PACKED_QKV_ENV)
    saved = {key: os.environ.get(key) for key in switches}
    runs: list[dict[str, Any]] = []
    reasons: list[str] = []
    try:
        for key in switches:
            os.environ[key] = "0"
        for i, arm in enumerate(seq):
            os.environ[env_var] = "1" if arm == arm_b else "0"
            one = {
                "i": i,
                "arm": arm,
                "role": "tail_canary" if i == len(seq) - 1 else "pair",
                "fusion_env": {key: os.environ[key] for key in switches},
            }
            try:
                one.update(_extract_ab(_invoke(args)))
            except Exception as exc:  # preserve partial evidence and always restore env
                one.update(
                    wall_p50_ms=None,
                    gpu_busy_ms=None,
                    host_share_pct=None,
                    tokens=None,
                    seed_token=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                reasons.append("run i=%d failed: %s" % (i, one["error"]))
            runs.append(one)
            print(
                "[ab] run %d arm=%s wall_p50=%r busy=%r"
                % (i, arm, one["wall_p50_ms"], one["gpu_busy_ms"])
            )
            if "error" in one:
                break
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    paired = [r for r in runs if r["role"] == "pair"]
    base = [r for r in paired if r["arm"] == arm_a]
    treat = [r for r in paired if r["arm"] == arm_b]
    if len(runs) != len(seq) or runs[-1]["role"] != "tail_canary":
        reasons.append("尾金丝雀缺失 / round incomplete")
    invalid_metrics = []
    for r in runs:
        for key in ("wall_p50_ms", "gpu_busy_ms", "host_share_pct"):
            valid = finite_number(
                r[key],
                positive=key != "host_share_pct",
                maximum=100.0 if key == "host_share_pct" else None,
            )
            if not valid:
                invalid_metrics.append({"i": r["i"], "field": key, "value": r[key]})
    if invalid_metrics:
        reasons.append("指标缺失或无效（wall/busy须有限正数，host须为[0,100]，见 invalid_metrics）")
    host_share_warnings = [
        {"i": r["i"], "arm": r["arm"], "host_share_pct": r["host_share_pct"]}
        for r in runs
        if finite_number(r["host_share_pct"]) and r["host_share_pct"] > 5.0
    ]
    if host_share_warnings:
        reasons.append("图格 host_share > 5%：轮次无效，不能采纳")

    def drift_of(key):
        if len(runs) != len(seq) or not all(
            finite_number(r[key], positive=True) for r in (runs[0], runs[-1])
        ):
            return None
        value = drift_pct(runs[0][key], runs[-1][key])
        return value if finite_number(abs(value)) else None

    drift = drift_of("wall_p50_ms")
    busy_drift = drift_of("gpu_busy_ms")
    if drift is None or abs(drift) > 5.0:
        reasons.append("金丝雀缺失或 wall 漂移超过 5%")
    valid_tokens = all(
        isinstance(r["tokens"], list)
        and len(r["tokens"]) == args.max_new_tokens
        and all(type(t) is int and t >= 0 for t in r["tokens"])
        and type(r["seed_token"]) is int
        and r["seed_token"] >= 0
        for r in runs
    )
    tokens_equal = valid_tokens and all(r["tokens"] == runs[0]["tokens"] for r in runs)
    seeds_equal = valid_tokens and all(r["seed_token"] == runs[0]["seed_token"] for r in runs)
    if not tokens_equal or not seeds_equal:
        reasons.append("token/seed 证据缺失、不合法或跨臂不一致")
    degenerate = bool(tokens_equal and len(set(runs[0]["tokens"])) <= 1)

    def stats(arm):
        vals = [r["wall_p50_ms"] for r in arm]
        if not vals or not all(finite_number(v, positive=True) for v in vals):
            return None, None, None
        median = statistics.median(vals)
        return statistics.mean(vals), median, statistics.median(abs(v - median) for v in vals)

    mean_a, median_a, mad_a = stats(base)
    mean_b, median_b, mad_b = stats(treat)
    deltas = [
        a["wall_p50_ms"] - b["wall_p50_ms"]
        for a, b in zip(base, treat)
        if finite_number(a["wall_p50_ms"], positive=True)
        and finite_number(b["wall_p50_ms"], positive=True)
    ]
    delta_median = statistics.median(deltas) if deltas else None
    return json_safe(
        {
            "generated_by": generated_by,
            "protocol_version": 2,
            "env_var": env_var,
            "status": "measured" if not reasons else "invalid",
            "reasons": reasons,
            "adoption_verdict": "not_evaluated",
            "adoption_requirements": "Independent oracle + natural-text gate + GPU regression; diagnostic A/B alone is insufficient.",
            "canary_drift_pct": round(drift, 3) if drift is not None else None,
            "tokens_equal": tokens_equal,
            "seeds_equal": seeds_equal,
            "mean_wall_baseline_ms": round(mean_a, 4) if mean_a is not None else None,
            "mean_wall_treatment_ms": round(mean_b, 4) if mean_b is not None else None,
            "treatment_faster_pct": (
                round(100 * (mean_a - mean_b) / mean_a, 2) if mean_a and mean_b else None
            ),
            "stat": "mean_of_run_p50s",
            "median_wall_baseline_ms": median_a,
            "median_wall_treatment_ms": median_b,
            "mad_wall_baseline_ms": mad_a,
            "mad_wall_treatment_ms": mad_b,
            "paired_wall": {
                "n": len(deltas),
                "delta_ms": deltas,
                "wins": sum(d > 0 for d in deltas),
                "ties": sum(d == 0 for d in deltas),
                "win_rate": sum(d > 0 for d in deltas) / len(deltas) if deltas else None,
                "median_delta_ms": delta_median,
                "mad_delta_ms": (
                    statistics.median(abs(d - delta_median) for d in deltas) if deltas else None
                ),
            },
            "busy_drift_pct": round(busy_drift, 3) if busy_drift is not None else None,
            "busy_stable": busy_drift is not None and abs(busy_drift) <= 5.0,
            "host_share_warnings": host_share_warnings,
            "invalid_metrics": invalid_metrics,
            "degenerate_trajectory": degenerate,
            "inherited_fusion_env": saved,
            "runs": runs,
            "note": note,
        }
    )


def run_routing_ab(args: argparse.Namespace) -> dict[str, Any]:
    """aten→fused 交替 A/B（routing 融合候选专用；键名保持历史契约）。"""
    payload = _run_env_ab(
        args,
        env_var=ABLATE_ARM_ENV,
        arm_a="aten",
        arm_b="fused",
        generated_by="tools/profile_graph.py --ab-fused-route",
        note=(
            "同进程交替 A/B；fused_faster_pct>0 = 融合臂更快。"
            "候选结论：采纳需重测 grouped 路径已录格（P2d 同款），本工具不写 result.json。"
        ),
    )
    payload["canary_aten_drift_pct"] = payload["canary_drift_pct"]
    payload["mean_wall_aten_ms"] = payload["mean_wall_baseline_ms"]
    payload["mean_wall_fused_ms"] = payload["mean_wall_treatment_ms"]
    payload["fused_faster_pct"] = payload["treatment_faster_pct"]
    return payload


def run_mask_fill_ab(args: argparse.Namespace) -> dict[str, Any]:
    """off→on 交替 A/B（skip-mask-fill 候选专用）。"""
    return _run_env_ab(
        args,
        env_var=MASK_FILL_ARM_ENV,
        arm_a="off",
        arm_b="on",
        generated_by="tools/profile_graph.py --ab-mask-fill",
        note=(
            "同进程交替 A/B；treatment_faster_pct>0 = 跳过 mask fill 更快。"
            "tokens_equal 必须为 True —— 若为 False 说明图内有代码读 mask，候选否决。"
            "低于噪声的收益也如实记录；本工具不写 result.json。"
        ),
    )


def run_fused_rope_ab(args: argparse.Namespace) -> dict[str, Any]:
    """off→on 交替 A/B（fused-rope 候选专用）。"""
    return _run_env_ab(
        args,
        env_var=ROPE_ARM_ENV,
        arm_a="off",
        arm_b="on",
        generated_by="tools/profile_graph.py --ab-fused-rope",
        note=(
            "同进程交替 A/B；treatment_faster_pct>0 = fused rope 更快。"
            "tokens_equal 必须为 True —— 若为 False 说明 fused rope 改变了贪心轨迹，候选否决。"
            "低于噪声的收益也如实记录；本工具不写 result.json。"
        ),
    )


def run_packed_qkv_ab(args: argparse.Namespace) -> dict[str, Any]:
    """off→on 交替 A/B（F2 packed q/k/v 候选专用）。"""
    return _run_env_ab(
        args,
        env_var=PACKED_QKV_ENV,
        arm_a="off",
        arm_b="on",
        generated_by="tools/profile_graph.py --ab-packed-qkv",
        note=(
            "同进程交替 A/B；treatment_faster_pct>0 = q/k/v 打包成一次 launch 更快。"
            "tokens_equal 必须为 True（行局部布局 ⇒ 期望逐位一致）；"
            "本工具不写 result.json，采纳仍需归档 + 23 格刷新。"
        ),
    )


def build_payload(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """把 run_cudagraph 的结果收成定位用 JSON（gap 原样保留）。"""
    gap = res.get("gap") or {}
    return {
        "generated_by": "tools/profile_graph.py",
        "status": "measured" if gap.get("top_kernels") else "incomplete",
        "config": cfg,
        "wall_p50_ms": gap.get("wall_p50_ms"),
        "gpu_busy_ms": gap.get("gpu_busy_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "kernels_per_call": gap.get("kernels_per_call"),
        "gap": gap,
        "note": (
            "定位用 kernel 分解，不是实验格子：share_pct = self_device_time ÷ gpu_busy "
            "（profiler 口径，per graph step）。要引用数字必须另立格子走矩阵。"
        ),
    }


def print_table(payload: dict[str, Any]) -> None:
    gap = payload.get("gap") or {}
    print(
        "step: wall_p50=%.4f ms  gpu_busy=%.4f ms  host=%.1f%%  kernels/step=%.0f"
        % (
            payload.get("wall_p50_ms") or 0.0,
            payload.get("gpu_busy_ms") or 0.0,
            payload.get("host_share_pct") or 0.0,
            payload.get("kernels_per_call") or 0.0,
        )
    )
    rows = gap.get("top_kernels") or []
    print("%8s %10s %8s  %s" % ("ms/step", "share%", "occ/stp", "kernel"))
    for r in rows:
        print(
            "%8.4f %10.2f %8.1f  %s"
            % (
                r.get("self_ms_per_token") or 0.0,
                r.get("share_pct") or 0.0,
                r.get("occurrences_per_token") or 0.0,
                str(r.get("name"))[:90],
            )
        )
    tail = (gap.get("kernels_per_call") or 0) - sum(
        r.get("occurrences_per_token") or 0.0 for r in rows
    )
    if tail > 0:
        print("  … tail of ~%.0f more kernel occurrences/step not in the top table" % tail)


def _source_identity() -> dict[str, Any]:
    """Record checkout identity and tracked working diff without exposing patch content."""
    root = Path(__file__).resolve().parents[1]
    try:

        def git(*args):
            return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)

        diff = git("diff", "HEAD", "--binary")
        code_hash = hashlib.sha256()
        for directory in ("src", "benchmarks", "tools", "tests"):
            for path in sorted((root / directory).rglob("*.py")):
                code_hash.update(str(path.relative_to(root)).encode() + b"\0")
                code_hash.update(path.read_bytes() + b"\0")
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(git("status", "--porcelain")),
            "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "code_sha256_including_untracked": code_hash.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "tracked_diff_sha256": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--moe", default="grouped", choices=["hf", "static", "gather", "grouped"])
    ap.add_argument("--attn", default="kernel", choices=["sdpa", "eager", "kernel"])
    ap.add_argument("--bit-width", default="w8a16", choices=["fp16", "w8a16", "w8a8", "w4a16"])
    ap.add_argument("--dequant", default="fused", choices=["none", "separate", "fused"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--rmsnorm", default="fused", choices=["fused", "hf"])
    ap.add_argument("--json", default="results/graph_profile.json")
    ap.add_argument(
        "--ab-fused-route",
        action="store_true",
        help="routing 融合候选 A/B：aten→fused 交替 N 轮（同进程），金丝雀+跨臂 token 比对",
    )
    ap.add_argument(
        "--ab-mask-fill",
        action="store_true",
        help="skip-mask-fill 候选 A/B：off→on 交替 N 轮（同进程），金丝雀+跨臂 token 比对",
    )
    ap.add_argument(
        "--ab-fused-rope",
        action="store_true",
        help="fused-rope 候选 A/B：off→on 交替 N 轮（同进程），金丝雀+跨臂 token 比对",
    )
    ap.add_argument(
        "--ab-packed-qkv",
        action="store_true",
        help="F2 packed q/k/v 候选 A/B：off→on 交替 N 轮（同进程），金丝雀+跨臂 token 比对",
    )
    ap.add_argument(
        "--ab-rounds",
        type=int,
        default=2,
        help="相邻 A/B 对数；2 ⇒ 4 次配对 run + 1 次尾 canary（探索性）",
    )
    args = ap.parse_args(argv)
    ab_flags = {
        "fused_route": args.ab_fused_route,
        "mask_fill": args.ab_mask_fill,
        "fused_rope": args.ab_fused_rope,
        "packed_qkv": args.ab_packed_qkv,
    }
    if sum(bool(v) for v in ab_flags.values()) > 1:
        ap.error(
            "--ab-fused-route / --ab-mask-fill / --ab-fused-rope / --ab-packed-qkv 互斥（一次一个变量）"
        )
    ab_mode = next((name for name, on in ab_flags.items() if on), None)
    cfg = {
        "model": args.model,
        "ctx": args.ctx,
        "batch": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "dtype": args.dtype,
        "moe": args.moe,
        "attn": args.attn,
        "bit_width": args.bit_width,
        "dequant": args.dequant,
        "group_size": args.group_size,
        "rmsnorm": args.rmsnorm,
        "ab_fused_route": bool(args.ab_fused_route),
        "ab_mask_fill": bool(args.ab_mask_fill),
        "ab_fused_rope": bool(args.ab_fused_rope),
        "ab_packed_qkv": bool(args.ab_packed_qkv),
        "ab_rounds": args.ab_rounds if ab_mode else None,
    }
    cfg["fusion_env"] = {
        key: os.environ.get(key, "0")
        for key in (ABLATE_ARM_ENV, MASK_FILL_ARM_ENV, ROPE_ARM_ENV, PACKED_QKV_ENV)
    }
    cfg["source"] = _source_identity()
    cfg["versions"] = {}
    for package in ("torch", "triton", "transformers"):
        try:
            cfg["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            cfg["versions"][package] = None
    out = Path(args.json)
    if out.exists():
        ap.error("输出已存在；请使用新的 --json 路径，保留旧轮次证据")
    try:
        if ab_mode == "fused_route":
            payload = run_routing_ab(args)
            payload["config"] = cfg
        elif ab_mode == "mask_fill":
            payload = run_mask_fill_ab(args)
            payload["config"] = cfg
        elif ab_mode == "fused_rope":
            payload = run_fused_rope_ab(args)
            payload["config"] = cfg
        elif ab_mode == "packed_qkv":
            payload = run_packed_qkv_ab(args)
            payload["config"] = cfg
        else:
            res = _invoke(args)
            payload = build_payload(res, cfg)
    except Exception as exc:  # noqa: BLE001 — 定位工具：失败要原因，不要 traceback 刷屏
        payload = {
            "generated_by": "tools/profile_graph.py",
            "status": "failed",
            "config": cfg,
            "error": "%s: %s" % (type(exc).__name__, str(exc)[:500]),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("[profile_graph] FAILED: %s" % payload["error"], file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if ab_mode:
        print(
            "[ab] canary_drift=%s%%  tokens_equal=%s"
            % (payload.get("canary_drift_pct"), payload.get("tokens_equal"))
        )
        print(
            "[ab] mean wall baseline=%s ms  treatment=%s ms  treatment_faster=%s%%"
            % (
                payload.get("mean_wall_baseline_ms"),
                payload.get("mean_wall_treatment_ms"),
                payload.get("treatment_faster_pct"),
            )
        )
        for r in payload.get("reasons") or []:
            print("[ab] INVALID: %s" % r, file=sys.stderr)
    else:
        print_table(payload)
    print("[profile_graph] wrote %s (status=%s)" % (out, payload["status"]))
    return 0 if payload["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
