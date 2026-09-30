"""runner — 真正执行一个实验并把结果填进 `result.json`。

`harness.py` 只做**编排**（展开矩阵、arch 裁剪、断点续跑、写骨架）；真正的测量在这里。
这个分工是刻意的：harness 在 Mac 上就能跑通，runner 必须在有 GPU 的机器上跑。

用法：

    # 第一条基线：fp16 加载 + 前向 + 首个性级延迟/显存/归因数字
    python -m benchmarks.runner --exp EXP-p0-02-baseline-fp16 --ctx 1024

    # 看一眼要跑什么，不实际执行
    python -m benchmarks.runner --exp EXP-p0-02-baseline-fp16 --dry-run

设计约束：
    - **只填补得到的字段**，其余保持 `None`（= 未测）。绝不为了"填满"而编数字。
    - 正确性门槛没过时，**不报性能数字**（项目硬规则 2，schema 校验器会拦）。
    - 每个数字都带口径：ctx、batch、warmup/timed、是否含 launch 开销。
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.bytes import ModelSpec, byte_budget, weight_budget
from benchmarks.harness import is_done, write_result
from benchmarks.schema import empty_result

PROBE_PATH = "results/machine_probe.json"


# ---------------------------------------------------------------------------
# 环境事实
# ---------------------------------------------------------------------------
def _load_probe(path: str = PROBE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def _hardware_from_probe(probe: dict[str, Any]) -> dict[str, Any]:
    devs = (probe.get("torch") or {}).get("devices") or []
    dev = devs[0] if devs else {}
    bw = probe.get("bandwidth") or {}
    smi = (probe.get("driver") or {}).get("nvidia_smi") or {}
    driver = ""
    if smi.get("stdout"):
        parts = [p.strip() for p in smi["stdout"].split(",")]
        if len(parts) >= 2:
            driver = parts[1]
    return {
        "gpu": dev.get("name"),
        "compute_capability": dev.get("compute_capability"),
        "vram_total_gb": dev.get("vram_total_gb"),
        "gpu_count_used": 1,
        "host_cpu": (probe.get("host") or {}).get("host_cpu"),
        # ★ 只读带宽才是 decode 的相关上限；两者都记下来
        "measured_bw_gbps": bw.get("read_only_gbps"),
        "measured_read_bw_gbps": bw.get("read_only_gbps"),
        "measured_copy_bw_gbps": bw.get("measured_gbps"),
        "driver": driver,
        "clocks": None,
    }


def _runtime() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": None,
        "cuda_build": None,
        "transformers": None,
        "triton": None,
        "commit": _git_commit(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_build"] = getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton

        out["triton"] = getattr(triton, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _model_spec(model_cfg: Any, name: str) -> ModelSpec:
    cfg = dict(model_cfg.to_dict()) if hasattr(model_cfg, "to_dict") else dict(model_cfg)
    return ModelSpec.from_hf_config(cfg, name=name)


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------
def _decode_step(model, tok, past, device, dtype):
    """One autoregressive decode step. Returns (logits_last, new_past)."""
    import torch

    with torch.no_grad():
        out = model(tok, past_key_values=past, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt, out.past_key_values


def run_baseline(
    *,
    model_id: str,
    ctx: int,
    batch: int,
    max_new_tokens: int,
    warmup: int,
    timed: int,
    dtype_name: str,
    device: str = "cuda",
) -> dict[str, Any]:
    """Load the model, measure prefill / per-token latency / VRAM / host-gap."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from benchmarks.hostgap import measure_host_gap
    from benchmarks.timing import summarize
    from benchmarks.vram import VramTracker, split_report

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]

    tr = VramTracker()
    tr.mark("after_torch_import")

    # tokenizer 先加载：确认它是可取的（本基线用合成的 token id，不需要它编码）
    AutoTokenizer.from_pretrained(model_id)

    # ★ `dtype=` 而不是 `torch_dtype=`：transformers 4.57 里 `torch_dtype` 会打
    #   deprecation 警告。**不要**传 `device_map=`：那会拉进 accelerate 依赖，
    #   而且会让权重分散到不同设备从而污染显存账。这里显式 `.to(device)`。
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device).eval()
    tr.mark("after_weights")

    spec = _model_spec(model.config, model_id)

    # 权重实际占用（区别于 checkpoint 里的字节数）。这一步很关键：
    # OLMoE 的 checkpoint 自述是 bf16，我们按 fp16 加载，两者都是 2 字节所以应当接近，
    # 但如果差得远，说明加载路径没按预期走 —— 那样后面所有字节账都是错的。
    weight_bytes_actual = sum(p.numel() * p.element_size() for p in model.parameters())
    tie = bool(getattr(model.config, "tie_word_embeddings", False))
    if tie:
        weight_bytes_actual -= (
            model.get_input_embeddings().weight.numel()
            * model.get_input_embeddings().weight.element_size()
        )

    # 用重复的 token 构造一个长度为 ctx 的 prompt（内容对延迟无影响）
    prompt_ids = [1] * ctx
    ids = torch.tensor([prompt_ids], device=device)

    # ---- prefill (TTFT) ----
    for _ in range(max(warmup, 2)):
        _decode_step(model, ids, None, device, dtype)  # 预热
    torch.cuda.synchronize()
    tr.mark("after_kv")

    ttft_ms: list[float] = []
    for _ in range(max(warmup, 5)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_ms.append((time.perf_counter() - t0) * 1000.0)
    ttft = summarize(ttft_ms)

    # ---- decode: per-token, 从 ctx 继续 ----
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with tr.phase("forward"):
        for _ in range(3):  # 打满 allocator 峰值
            nxt, past = _decode_step(model, nxt, past, device, dtype)

    step_ms: list[float] = []
    for _ in range(max(timed, 20)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nxt, past = _decode_step(model, nxt, past, device, dtype)
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000.0)
    steps = summarize(step_ms)

    # ---- host / kernel 拆分（固定状态下重复同一步） ----
    gap = measure_host_gap(
        lambda: _decode_step(model, nxt, past, device, dtype),
        warmup=5,
        timed=20,
        repeats=3,
    )

    vram = split_report(tr)

    # ---- 字节预算 ----
    budget = byte_budget(spec, ctx=ctx, batch=batch)
    w = weight_budget(spec)

    # ---- 达成率（用实测带宽，绝不用规格值） ----
    probe = _load_probe()
    bw = (probe.get("bandwidth") or {}).get("read_only_gbps")
    attainment: dict[str, Any] = {
        "floor_w_ms": None,
        "floor_t_ms": None,
        "quant_attainment_pct": None,
        "bw_attainment_pct": None,
        "denominator_note": "measured bandwidth from results/machine_probe.json",
    }
    if bw:
        floor_w = (w["total_bytes"] / (bw * 1e9)) * 1000.0
        floor_t = (budget["T"]["total_bytes"] / (bw * 1e9)) * 1000.0
        attainment.update(
            floor_w_ms=round(floor_w, 4),
            floor_t_ms=round(floor_t, 4),
            quant_attainment_pct=round(100.0 * floor_w / steps["p50"], 1),
            bw_attainment_pct=round(100.0 * floor_t / steps["p50"], 1),
        )

    return {
        "spec": spec,
        "weight_bytes_actual": weight_bytes_actual,
        "ttft": ttft,
        "steps": steps,
        "gap": gap,
        "vram": vram,
        "budget": budget,
        "weight": w,
        "attainment": attainment,
        "measured_bw_gbps": bw,
    }


def _consistency_gate(res: dict[str, Any]) -> dict[str, Any]:
    """The falsifiable check available at the fp16 baseline stage.

    没有量化时，"与参考比"是同义反复。此时唯一能证伪的是**自洽性**：

        floor_T = T_bytes / measured_bandwidth   <=   measured_time

    若 floor > 实测时间，说明要么字节账少算了、要么带宽高估了。两者都会污染之后
    每一个达成率，所以这一步必须**在**做量化之前就通过。
    """
    a = res.get("attainment") or {}
    p50 = (res.get("steps") or {}).get("p50")
    floor_t = a.get("floor_t_ms")
    checks = [
        {
            "name": "floor_T <= measured_p50",
            "floor_t_ms": floor_t,
            "measured_p50_ms": p50,
            "passed": bool(floor_t and p50 and floor_t <= p50),
            "why": "理论下限若高于实测时间，则字节账或带宽有误",
        },
        {
            "name": "attainment_T <= 100%",
            "bw_attainment_pct": a.get("bw_attainment_pct"),
            "passed": bool(
                a.get("bw_attainment_pct") is not None and a["bw_attainment_pct"] <= 100.0
            ),
            "why": ">100% 意味着比'只能读这么多字节'还快 —— 物理上不可能",
        },
        {
            "name": "bandwidth was measured, not assumed",
            "measured_bw_gbps": res.get("measured_bw_gbps"),
            "passed": bool(res.get("measured_bw_gbps")),
            "why": "硬规则 4：roofline 必须用实测带宽",
        },
    ]
    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "note": (
            "这是 fp16 阶段能做的唯一可证伪检查。量化阶段（EXP-p1-01 起）"
            "还会额外要求 KL / top-1 对 fp16 参考达标。"
        ),
    }


# ---------------------------------------------------------------------------
# 组装 result.json
# ---------------------------------------------------------------------------
def build_payload(
    exp_id: str, res: dict[str, Any], cfg: dict[str, Any], hypothesis: dict[str, str]
) -> dict[str, Any]:
    spec: ModelSpec = res["spec"]
    probe = _load_probe()
    steps = res["steps"]
    ttft = res["ttft"]
    gap = res["gap"]
    vram = res["vram"]

    payload = empty_result(exp_id, status="done")
    payload["hypothesis"] = hypothesis

    payload["model"] = {
        "repo": spec.name,
        "arch": "olmoe" if spec.is_moe else "dense",
        "dtype": cfg.get("dtype"),
        "layers": spec.layers,
        "hidden": spec.hidden,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "expert_ffn_dim": spec.expert_ffn_dim,
        "total_params": spec.total_params(),
        "activated_params": spec.activated_params_excl_embedding(),
    }
    payload["hardware"] = _hardware_from_probe(probe)
    payload["runtime"] = _runtime()
    payload["config"] = {
        "bit_width": cfg.get("bit_width") or "fp16",
        "group_size": cfg.get("group_size"),
        "dequant": cfg.get("dequant") or "none",
        "fusion": cfg.get("fusion") or "none",
        "execution": cfg.get("execution") or "eager",
        "kv_cache_dtype": "fp16",
        "ctx": cfg.get("ctx"),
        "batch": cfg.get("batch"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "warmup": cfg.get("warmup"),
        "timed": cfg.get("timed"),
    }

    # ★ fp16 基线没有"与参考比"的意义 —— 这个配置**就是**参考实现。
    #   所以这里不写 KL=0 冒充"通过了门槛"（那是同义反复，属于自欺）。
    #   真正可证伪的门槛是**自洽性**：由实测带宽和字节账推出的理论下限，
    #   必须 <= 实测时间。若 >100%，错的一定是字节账或带宽（本项目全部结论的地基）。
    consistency = _consistency_gate(res)
    payload["correctness"] = {
        "kl": None,
        "top1_agreement": None,
        "ppl_fp16": None,
        "ppl_this": None,
        "ppl_delta": None,
        "gate": "PASS" if consistency["passed"] else "FAIL",
        "thresholds": {
            "note": (
                "fp16 基线：KL / top-1 对参考实现无意义（本配置即参考），故为 null。"
                "可证伪的门槛是自洽性 —— 见 consistency。"
            ),
            "kl_max_w8": 0.01,
            "kl_max_w4": 0.05,
            "top1_min": 0.95,
        },
        "consistency": consistency,
    }

    budget = res["budget"]
    payload["bytes"] = {
        "W_total_bytes": budget["W"]["total_bytes"],
        "T_total_bytes": budget["T"]["total_bytes"],
        "kv_read_bytes": budget["KV"]["read_bytes"],
        "breakdown": budget["W"]["items"],
        # ★ 预算 vs 实测：两者必须接近。差得远说明"读多少字节"这个前提错了，
        #   而那正是本项目全部结论的地基（docs/METRICS.md §3）。
        "weight_bytes_actual": res.get("weight_bytes_actual"),
        "budget_vs_actual_delta_pct": (
            round(
                100.0
                * (res["weight_bytes_actual"] - budget["W"]["total_bytes"])
                / budget["W"]["total_bytes"],
                3,
            )
            if res.get("weight_bytes_actual") and budget["W"]["total_bytes"]
            else None
        ),
        "budget_vs_actual_note": (
            "W 是**每 token 实际读到的权重字节**（top-k 专家），故略小于 checkpoint 里"
            "所有专家之和；weight_bytes_actual 是加载后驻留的全部参数。两者不等是预期的，"
            "但差距应能由 expert 未激活 + lm_head 是否计入来解释。"
        ),
    }

    payload["latency"] = {
        "ttft_p50_ms": round(ttft["p50"], 4),
        "ttft_p99_ms": round(ttft["p99"], 4),
        "per_token_p50_ms": round(steps["p50"], 4),
        "per_token_p99_ms": round(steps["p99"], 4),
        "tok_per_s": round(1000.0 / steps["p50"], 2) if steps["p50"] else None,
        "samples": {"ttft_ms": ttft["samples"], "per_token_ms": steps["samples"]},
    }

    payload["attribution"] = {
        "gpu_p50_ms": gap.get("gpu_p50_ms"),
        "host_p50_ms": gap.get("host_p50_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "dequant_share_pct": None,  # 本阶段没有反量化
        "routing_share_pct": None,  # 见 EXP-p4-01
    }

    payload["attainment"] = res["attainment"]

    payload["vram"] = {
        "weights_gib": vram.get("weights_gib"),
        "kv_gib": vram.get("kv_gib"),
        "activations_gib": vram.get("activations_gib"),
        "workspace_gib": vram.get("workspace_gib"),
        "peak_total_gib": vram.get("peak_total_gib"),
        "method": vram.get("method"),
    }

    payload["verdict"] = {
        "verdict": "PASS" if consistency["passed"] else "FAIL",
        "outcome": "supported" if consistency["passed"] else "refuted",
        "notes": (
            "第一条系统级基线，不构成任何量化结论；它建立的是测量管线与绝对数字。"
            "正确性栏位里的 KL/top-1 为 null（对本配置无意义），"
            "实际把守的是 consistency 自洽性检查。"
        ),
    }
    payload["applicability"] = {
        "applies_to": "fp16、单卡、batch=1、ctx=%s、%s"
        % (cfg.get("ctx"), payload["hardware"].get("compute_capability")),
        "does_not_apply_to": "量化配置（见 EXP-p1-*）、batch>1、其它架构",
        "why": "所有数字都绑定在实测带宽 %s GB/s 与该卡的 launch 开销上" % res["measured_bw_gbps"],
    }
    payload["_measurement"] = {
        "timing_method": "perf_counter wall, synchronized per step (includes launch overhead)",
        "ctx_window": "prefill at ctx=%s, then %s decode steps (KV grows by that many tokens)"
        % (cfg.get("ctx"), cfg.get("max_new_tokens")),
        "host_gap_caveats": gap.get("caveats"),
        "vram_split_note": vram.get("note"),
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="EXP-p0-02-baseline-fp16")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    exp_id = "%s__ctx=%d" % (args.exp, args.ctx)
    if is_done(exp_id, args.results_dir) and not args.force:
        print("skip (done): %s" % exp_id)
        return 0

    hypothesis = {
        "statement": "fp16 能在目标卡上加载并前向；测量管线产出的延迟/显存/归因自洽",
        "refutation_signal": "加载失败、OOM、或达成率 >100%（说明字节账或带宽错了）",
    }
    cfg = {
        "bit_width": "fp16",
        "dtype": args.dtype,
        "ctx": args.ctx,
        "batch": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "execution": "eager",
        "dequant": "none",
        "fusion": "none",
        "group_size": None,
    }

    if args.dry_run:
        print("would run: %s" % exp_id)
        print(
            "  model=%s ctx=%d batch=%d dtype=%s" % (args.model, args.ctx, args.batch, args.dtype)
        )
        return 0

    try:
        res = run_baseline(
            model_id=args.model,
            ctx=args.ctx,
            batch=args.batch,
            max_new_tokens=args.max_new_tokens,
            warmup=args.warmup,
            timed=args.timed,
            dtype_name=args.dtype,
        )
    except Exception as exc:  # noqa: BLE001
        import traceback

        print("runner 失败: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return 1

    payload = build_payload(exp_id, res, cfg, hypothesis)
    path, problems = write_result(payload, root=args.results_dir)

    s = res["steps"]
    print("=== %s ===" % exp_id)
    print("  measured bw      : %s GB/s (read-only)" % res["measured_bw_gbps"])
    print("  TTFT p50/p99     : %.3f / %.3f ms" % (res["ttft"]["p50"], res["ttft"]["p99"]))
    print("  per-token p50/p99: %.4f / %.4f ms" % (s["p50"], s["p99"]))
    print("  host share       : %s%%" % res["gap"].get("host_share_pct"))
    print("  VRAM peak        : %s GiB" % res["vram"].get("peak_total_gib"))
    print(
        "  [W] / [T]        : %d / %d bytes"
        % (res["budget"]["W"]["total_bytes"], res["budget"]["T"]["total_bytes"])
    )
    a = res["attainment"]
    print("  floor_W / floor_T: %s / %s ms" % (a["floor_w_ms"], a["floor_t_ms"]))
    print(
        "  attainment       : quant %s%% | bandwidth %s%%"
        % (a["quant_attainment_pct"], a["bw_attainment_pct"])
    )
    if a["quant_attainment_pct"] and a["quant_attainment_pct"] > 100:
        print(
            "  ⚠️ 达成率 >100% —— 字节账或带宽有问题，必须解释（docs/METRICS.md §3）",
            file=sys.stderr,
        )
    print("\n  wrote %s" % path)
    print("  schema problems: %d" % len(problems))
    for p in problems:
        print("    - %s" % p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
