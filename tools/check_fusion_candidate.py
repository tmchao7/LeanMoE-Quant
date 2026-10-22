"""Natural-text correctness preparation for one fusion candidate; no performance timing.

Default --dry-run prints a plan. --execute is a NEW diagnostic run, never a matrix
cell or replacement GATE_CITATION. Scope: OLMoE, batch=1, w8a16/g128/fused.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SWITCHES = {
    "routing": "LEMOQ_FUSED_ROUTE",
    "rope": "LEMOQ_FUSED_ROPE",
    "mask": "LEMOQ_SKIP_MASK_FILL",
}


@contextmanager
def candidate_env(candidate: str | None):
    saved = {key: os.environ.get(key) for key in SWITCHES.values()}
    try:
        for name, key in SWITCHES.items():
            os.environ[key] = "1" if name == candidate else "0"
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def prompt_windows(ids: Any, *, ctx: int, windows: int, offset: int = 0) -> list[Any]:
    """Reuse existing cached natural tokens; never download or accept short windows."""
    import torch

    if ids.dtype != torch.int64 or ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
        raise ValueError("prompt cache must be int64 [N] or [1,N]")
    ids = ids.reshape(-1)
    if min(ctx, windows) < 1 or offset < 0 or ids.numel() < offset + ctx * windows:
        raise ValueError("invalid dimensions or insufficient cached natural tokens")
    if bool((ids < 0).any()):
        raise ValueError("negative token ID")
    out = [ids[offset + i * ctx : offset + (i + 1) * ctx].reshape(1, ctx) for i in range(windows)]
    if any(torch.unique(w).numel() < 2 for w in out):
        raise ValueError("degenerate prompt window; supply natural text")
    return out


def collect(
    session: Any, *, steps: int, graph: bool, forced: list[int] | None = None
) -> dict[str, Any]:
    """Use fixed HF input tokens for KL; free greedy trajectories are collected separately."""
    output = session.prefill()
    logits = [output.logits[:, -1:, :].detach().cpu().float().clone()]
    seed = session.current_token_id
    inputs, tokens = [], []
    if forced is not None and len(forced) != steps:
        raise ValueError("forced sequence has the wrong horizon")
    for i in range(steps):
        if forced is not None:
            session.current_token_id = forced[i]
        inputs.append(session.current_token_id)
        if graph:
            session.replay(synchronize=True)
            last = session.logits_cpu()
        else:
            last = session.decode_eager()[:, -1:, :].detach().cpu().float().clone()
        logits.append(last)
        tokens.append(int(last.argmax(-1).item()))
    return {"logits": logits, "inputs": inputs, "tokens": tokens, "seed": seed}


def _same_path(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.cudagraph import compare_logits, graph_correctness_passed

    report = compare_logits(a["logits"], b["logits"])
    report.update(
        seed_token_match=a["seed"] == b["seed"], token_sequence_match=a["tokens"] == b["tokens"]
    )
    report["passed"] = graph_correctness_passed(report)
    return report


def _execute(args: argparse.Namespace, payload: dict[str, Any], write: Any) -> bool:
    import torch
    from transformers import AutoModelForCausalLM

    from benchmarks.cudagraph import StaticCacheDecodeSession
    from benchmarks.quant_gate import quant_accuracy_report
    from benchmarks.runner import quant_spec_for_cfg, quantize_for_cfg
    from lemoq.kernels.attn_decode import replace_attention_cores
    from lemoq.kernels.moe_static import replace_moe_blocks
    from lemoq.kernels.rmsnorm import replace_rms_norms
    from tools.profile_graph import _source_identity

    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True
    )
    if processes.strip():
        raise RuntimeError("GPU occupied; do not kill other processes: " + processes.strip())
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("this preparation targets one sm_120 CUDA GPU")
    cache = torch.load(args.prompt_cache, map_location="cpu", weights_only=True)
    prompts = prompt_windows(cache, ctx=args.ctx, windows=args.windows, offset=args.offset)
    payload.update(
        source=_source_identity(),
        prompt_sha256=hashlib.sha256(cache.numpy().tobytes()).hexdigest(),
        hardware={"gpu": torch.cuda.get_device_name(0), "compute_capability": [12, 0]},
        runtime={"torch": torch.__version__, "cuda": torch.version.cuda},
    )
    import importlib.metadata

    payload["runtime"].update(
        {p: importlib.metadata.version(p) for p in ("triton", "transformers")}
    )
    with candidate_env(None):
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
            )
            .to("cuda")
            .eval()
        )
        if model.config.model_type != "olmoe":
            raise ValueError("only the verified OLMoE architecture is supported")
        if args.ctx > model.config.max_position_embeddings:
            raise ValueError("ctx exceeds model limit")
        replace_rms_norms(model, "fused")
        references = []
        for prompt in prompts:
            session = StaticCacheDecodeSession(
                model, prompt.to("cuda"), ctx=args.ctx, max_new_tokens=args.steps
            )
            references.append(collect(session, steps=args.steps, graph=False))
            del session
        # Transform the same weights once, releasing original experts, so two models never reside on GPU.
        cfg = {"bit_width": "w8a16", "group_size": 128, "dequant": "fused", "ctx": args.ctx}
        replace_moe_blocks(model, impl="grouped", quant=quant_spec_for_cfg(cfg))
        quantize_for_cfg(model, cfg)
        replace_attention_cores(model, backend="triton")
    payload["windows"] = []
    all_hf, all_base, all_candidate = [], [], []
    for i, (prompt, reference) in enumerate(zip(prompts, references)):
        trajectories = {}
        for label, candidate in (("baseline", None), ("candidate", args.candidate)):
            with candidate_env(candidate):
                session = StaticCacheDecodeSession(
                    model, prompt.to("cuda"), ctx=args.ctx, max_new_tokens=args.steps
                )
                session.capture(warmup=5)  # compile/warm BEFORE capture, no timing
                eager = collect(session, steps=args.steps, graph=False, forced=reference["inputs"])
                graph = collect(session, steps=args.steps, graph=True, forced=reference["inputs"])
                free = collect(session, steps=args.steps, graph=True)
                trajectories[label] = {
                    "graph": graph,
                    "free": free,
                    "self_check": _same_path(eager, graph),
                }
                del session
                gc.collect()
                torch.cuda.empty_cache()
        base, cand = trajectories["baseline"], trajectories["candidate"]
        all_hf.extend(reference["logits"])
        all_base.extend(base["graph"]["logits"])
        all_candidate.extend(cand["graph"]["logits"])
        row = {
            "window": i,
            "offset": args.offset + i * args.ctx,
            "hf_inputs": reference["inputs"],
            "hf_seed": reference["seed"],
            "baseline_self_check": base["self_check"],
            "candidate_self_check": cand["self_check"],
            "trajectory_equal": base["free"]["tokens"] == cand["free"]["tokens"]
            and base["free"]["seed"] == cand["free"]["seed"],
            "baseline_tokens": base["free"]["tokens"],
            "candidate_tokens": cand["free"]["tokens"],
            "degenerate_candidate": len(set(cand["free"]["tokens"])) <= 1,
            "baseline_vs_hf": quant_accuracy_report(
                reference["logits"], base["graph"]["logits"], scheme="w8a16", group_size=128
            ),
            "candidate_vs_hf": quant_accuracy_report(
                reference["logits"], cand["graph"]["logits"], scheme="w8a16", group_size=128
            ),
        }
        payload["windows"].append(row)
        write()
    payload["baseline_vs_hf"] = quant_accuracy_report(
        all_hf, all_base, scheme="w8a16", group_size=128
    )
    payload["candidate_vs_hf"] = quant_accuracy_report(
        all_hf, all_candidate, scheme="w8a16", group_size=128
    )
    passed = (
        payload["baseline_vs_hf"]["gate"] == "PASS"
        and payload["candidate_vs_hf"]["gate"] == "PASS"
        and all(
            w["baseline_self_check"]["passed"]
            and w["candidate_self_check"]["passed"]
            and w["trajectory_equal"]
            for w in payload["windows"]
        )
    )
    payload["correctness_gate"] = "PASS" if passed else "FAIL"
    return passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="existing LOCAL snapshot directory")
    ap.add_argument("--prompt-cache", required=True, help="existing wikitext token .pt file")
    ap.add_argument("--candidate", choices=tuple(SWITCHES), required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--json", required=True, help="NEW diagnostic path, never result.json")
    ap.add_argument("--execute", action="store_true", help="actually run GPU correctness checks")
    ap.add_argument("--dry-run", action="store_true", help="print plan only (default)")
    args = ap.parse_args(argv)
    out = Path(args.json)
    if args.execute and args.dry_run:
        ap.error("--execute and --dry-run are mutually exclusive")
    if min(args.ctx, args.steps) < 1 or args.windows < 5 or args.offset < 0 or args.ctx > 4096:
        ap.error("positive ctx/steps, ctx<=4096, windows>=5 and offset>=0 required")
    if out.name == "result.json" or out.exists():
        ap.error("use a NEW diagnostic path, never an existing file or result.json")
    payload = {
        "generated_by": "tools/check_fusion_candidate.py",
        "status": "planned",
        "config": vars(args),
        "scope": "OLMoE/w8a16/g128/fused/grouped/kernel/fused RMSNorm/batch=1",
        "adoption_verdict": "not_evaluated",
        "performance_measured": False,
        "steps": [
            "GPU micro-oracle tests first",
            "independent fp16 HF+StaticCache trajectory",
            "same HF teacher-forced inputs across both arms",
            "per-arm graph/eager checks",
            "independent HF KL/top1 gate + separate free greedy comparison",
        ],
        "note": "Diagnostic only; does not update GATE_CITATIONS or experiment status. GPU kernel tests and formal performance round remain separate.",
    }
    if not args.execute:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if not Path(args.model).is_dir() or not Path(args.prompt_cache).is_file():
        ap.error("local model/cache must already exist; no download/install")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write():
        from tools.round_canary import json_safe

        out.write_text(
            json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    try:
        passed = _execute(args, payload, write)
        payload["status"] = "correctness_checked"
    except Exception as exc:  # diagnostic failures retain completed windows
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        passed = False
    write()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
