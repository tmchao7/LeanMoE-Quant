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
import json
import os
import sys
from pathlib import Path
from typing import Any

from benchmarks.runner import run_cudagraph

ABLATE_ARM_ENV = "LEMOQ_FUSED_ROUTE"
MASK_FILL_ARM_ENV = "LEMOQ_SKIP_MASK_FILL"
ROPE_ARM_ENV = "LEMOQ_FUSED_ROPE"


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
    """env=0/env=1 交替 N 轮的同进程 A/B（通用核心）。

    验收三件套（spec skip-mask-fill [S2.3]，routing-fusion 同款）：
      ① 金丝雀 = 基线臂（arm_a）首尾 wall 漂移 ≤5%（超了整轮无效）；
      ② 全部 run 的 token 序列逐位相同（候选不得改变贪心轨迹）；
      ③ arm_b 臂 wall 对 arm_a 臂的同进程均值差（方向如实记录）。
    """
    rounds = max(int(args.ab_rounds), 1)
    seq = [arm_a, arm_b] * rounds
    runs: list[dict[str, Any]] = []
    saved = os.environ.get(env_var)
    try:
        for i, arm in enumerate(seq):
            os.environ[env_var] = "1" if arm == arm_b else "0"
            one = _extract_ab(_invoke(args))
            one.update({"i": i, "arm": arm})
            runs.append(one)
            print(
                "[ab] run %d arm=%s wall_p50=%.4f ms busy=%.4f ms tokens=%s"
                % (i, arm, one["wall_p50_ms"] or -1, one["gpu_busy_ms"] or -1, one["tokens"])
            )
    finally:
        if saved is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = saved

    base = [r for r in runs if r["arm"] == arm_a]
    treat = [r for r in runs if r["arm"] == arm_b]
    reasons: list[str] = []
    # ★ fail-closed（review P2）：任一 run 的 wall/busy/host_share 缺失 ⇒ 判 invalid，
    #   不崩在 `sum(None)` 上（原实现遇到 gap 不完整的 run 会 TypeError 丢整轮数据）。
    missing = [
        r
        for r in runs
        if r["wall_p50_ms"] is None or r["gpu_busy_ms"] is None or r["host_share_pct"] is None
    ]
    if missing:
        reasons.append(
            "指标缺失（wall/busy/host_share 为 None，run i=%s）——按 fail-closed 判无效"
            % [r["i"] for r in missing]
        )
    drift = None
    if len(base) >= 2 and base[0]["wall_p50_ms"] and base[-1]["wall_p50_ms"]:
        drift = round(
            100.0 * (base[-1]["wall_p50_ms"] - base[0]["wall_p50_ms"]) / base[0]["wall_p50_ms"],
            3,
        )
        if abs(drift) > 5.0:
            reasons.append("%s 金丝雀漂移 %.3f%% > 5%%" % (arm_a, drift))
    else:
        reasons.append("%s 臂不足两次，金丝雀缺失" % arm_a)
    # ★ busy 双口径（review P1；fused-rope Journey log ① 的建议）：wall 金丝雀会被 host
    #   噪声打爆而 busy 不动 —— 把基线臂首尾的 gpu_busy 漂移单列，供判读时对照。
    busy_drift = None
    if len(base) >= 2 and base[0]["gpu_busy_ms"] and base[-1]["gpu_busy_ms"]:
        busy_drift = round(
            100.0 * (base[-1]["gpu_busy_ms"] - base[0]["gpu_busy_ms"]) / base[0]["gpu_busy_ms"],
            3,
        )
    busy_stable = busy_drift is not None and abs(busy_drift) <= 5.0
    # ★ 图格污染观察（review P1）：AGENTS §6 的 host_share>5% ⇒ 污染。**只警告不改判定** ——
    #   "treatment 本身降 host" 时该判据要重释（skip-mask-fill Journey log ④），
    #   判定语义的变更需用户批准；这里先把观察暴露出来。
    host_share_warnings = [
        {"i": r["i"], "arm": r["arm"], "host_share_pct": r["host_share_pct"]}
        for r in runs
        if r["host_share_pct"] is not None and r["host_share_pct"] > 5.0
    ]
    token_sets = {json.dumps(r["tokens"]) for r in runs}
    tokens_equal = len(token_sets) == 1
    if not tokens_equal:
        reasons.append("跨臂 token 序列不一致（%s→%s 改变了贪心轨迹）" % (arm_a, arm_b))
    # ★ 轨迹退化警告（review P3）：合成 prompt 下 token 全同 ⇒ tokens_equal 近乎空判据
    #   （routing-fusion Report 已自认）。只标记，不否决 —— 相等性结论仍成立，只是证据弱。
    degenerate = False
    if tokens_equal and runs:
        degenerate = len(set(runs[0]["tokens"] or [])) <= 1

    def _mean_median(vals: list[Any]) -> tuple[float | None, float | None]:
        if any(v is None for v in vals) or not vals:
            return None, None
        ordered = sorted(vals)
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
        return sum(ordered) / n, float(median)

    mean_a, median_a = _mean_median([r["wall_p50_ms"] for r in base])
    mean_b, median_b = _mean_median([r["wall_p50_ms"] for r in treat])
    delta_pct = round(100.0 * (mean_a - mean_b) / mean_a, 2) if mean_a and mean_b else None
    return {
        "generated_by": generated_by,
        "env_var": env_var,
        "status": "measured" if not reasons else "invalid",
        "reasons": reasons,
        "canary_drift_pct": drift,
        "tokens_equal": tokens_equal,
        "mean_wall_baseline_ms": round(mean_a, 4) if mean_a else None,
        "mean_wall_treatment_ms": round(mean_b, 4) if mean_b else None,
        "treatment_faster_pct": delta_pct,
        # ---- 观察字段（review P1/P2/P3）：不参与 status 判定，供判读 ----
        "stat": "mean_of_run_p50s",
        "median_wall_baseline_ms": round(median_a, 4) if median_a is not None else None,
        "median_wall_treatment_ms": round(median_b, 4) if median_b is not None else None,
        "busy_drift_pct": busy_drift,
        "busy_stable": busy_stable,
        "host_share_warnings": host_share_warnings,
        "degenerate_trajectory": degenerate,
        "runs": runs,
        "note": note,
    }


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
    ap.add_argument("--ab-rounds", type=int, default=2, help="A/B 的 (off,on) 对数；2 ⇒ 4 次 run")
    args = ap.parse_args(argv)
    ab_flags = {
        "fused_route": args.ab_fused_route,
        "mask_fill": args.ab_mask_fill,
        "fused_rope": args.ab_fused_rope,
    }
    if sum(bool(v) for v in ab_flags.values()) > 1:
        ap.error("--ab-fused-route / --ab-mask-fill / --ab-fused-rope 互斥（一次一个变量）")
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
        "ab_rounds": args.ab_rounds if ab_mode else None,
    }
    out = Path(args.json)
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
