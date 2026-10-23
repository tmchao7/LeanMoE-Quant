"""EXP-p1-05 preparation: plan / calibrate / validate; dry-run unless --execute.

Diagnostic only. Does not register matrix cells, publish latency, update gate
citations, download models/data, or install dependencies. CUDA requires an
explicit budget. Split provenance and complete packed checkpoints are mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lemoq.quant.expert_calibration import (
    calibrate_model,
    fallback_layers,
    file_sha256,
    load_bundle,
    resource_plan,
    save_packed,
)
from tools.check_fusion_candidate import candidate_env, collect
from tools.profile_graph import _source_identity
from tools.round_canary import drift_pct, finite_number, json_safe

ARMS = {"w8_rtn": ("rtn", 8), "experts_w4_rtn": ("rtn", 4), "experts_w4_gptq": ("gptq", 4)}


class Budget:
    def __init__(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("positive finite GPU budget required")
        self.deadline = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("GPU budget exhausted; checkpoint remains incomplete")


def tokenizer_identity(model_dir: Path) -> str:
    paths = sorted(
        p
        for p in model_dir.iterdir()
        if p.is_file()
        and (
            p.name.startswith("tokenizer")
            or p.name in ("vocab.json", "merges.txt", "special_tokens_map.json")
        )
    )
    if not paths:
        raise ValueError("local tokenizer files missing")
    return hashlib.sha256(
        "\n".join(p.name + ":" + file_sha256(p) for p in paths).encode()
    ).hexdigest()


def model_identity(model_dir: Path) -> str:
    paths = sorted(
        [
            *model_dir.glob("*.safetensors"),
            *model_dir.glob("*.bin"),
            *model_dir.glob("*.index.json"),
            model_dir / "config.json",
        ]
    )
    if len(paths) < 2:
        raise ValueError("local config/weights missing")
    return hashlib.sha256(
        "\n".join(p.name + ":" + file_sha256(p) for p in paths).encode()
    ).hexdigest()


def cached_windows(
    path: Path,
    provenance: dict,
    *,
    split: str,
    tokenizer: str,
    count: int,
    length: int,
    seed: int = 42,
    randomize: bool = True,
):
    if (
        provenance.get("status") != "complete"
        or provenance.get("dataset") != "wikitext-2-raw-v1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(provenance.get("revision", "")))
    ):
        raise ValueError("pinned WikiText-2 dataset revision required")
    if provenance.get("tokenizer_sha256") != tokenizer:
        raise ValueError("tokenizer identity mismatch")
    splits = provenance.get("splits", {})
    hashes = [splits.get(s, {}).get("file_sha256") for s in ("train", "validation", "test")]
    if any(not isinstance(h, str) or len(h) != 64 for h in hashes) or len(set(hashes)) != 3:
        raise ValueError("three distinct train/validation/test hashes required")
    if file_sha256(path) != splits[split]["file_sha256"]:
        raise ValueError("split cache hash mismatch")
    ids = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not torch.is_tensor(ids)
        or ids.dtype != torch.int64
        or ids.ndim not in (1, 2)
        or (ids.ndim == 2 and ids.shape[0] != 1)
    ):
        raise ValueError("token cache must be int64 [N] or [1,N]")
    ids = ids.reshape(-1)
    token_hash = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    if token_hash != splits[split].get("token_sha256") or bool((ids < 0).any()):
        raise ValueError("token content hash or domain mismatch")
    content_hashes = [splits[s].get("token_sha256") for s in ("train", "validation", "test")]
    if len(set(content_hashes)) != 3 or any(not h for h in content_hashes):
        raise ValueError("cross-split token content reuse is forbidden")
    if count < 1 or length < 1 or ids.numel() < count * length:
        raise ValueError("insufficient complete split windows")
    starts = list(range(0, int(ids.numel()) - length + 1, length))
    offsets = random.Random(seed).sample(starts, count) if randomize else starts[:count]
    windows = [ids[o : o + length].reshape(1, -1) for o in offsets]
    if any(torch.unique(w).numel() < 2 for w in windows):
        raise ValueError("degenerate prompt cache")
    return windows, {
        "split": split,
        "offsets": offsets,
        "length": length,
        "seed": seed,
        "file_sha256": file_sha256(path),
        "token_sha256": token_hash,
    }


def accuracy_gate(metrics: dict, ppl_delta: float, *, expert_bits: int):
    from benchmarks.quant_gate import QUANT_THRESHOLDS

    if expert_bits not in (4, 8):
        raise ValueError("expert bit-width must be explicit")
    limit = QUANT_THRESHOLDS["kl_w4_max" if expert_bits == 4 else "kl_w8_max"]
    values = (metrics.get("kl"), metrics.get("top1_agreement"), ppl_delta)
    finite = all(type(v) in (int, float) and math.isfinite(v) for v in values)
    passed = (
        finite and 0 <= values[0] < limit and 0.95 <= values[1] <= 1.0 and -1 < ppl_delta < 0.01
    )
    return {
        "passed": bool(passed),
        "thresholds": {"kl_max": limit, "top1_min": 0.95, "ppl_relative_delta_max": 0.01},
        "precision_scope": "experts%d_attention8_activation16" % expert_bits,
    }


def _write(path: Path, data: dict):
    path.write_text(
        json.dumps(json_safe(data), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def _load_model(args):
    from transformers import AutoModelForCausalLM

    from lemoq.kernels.rmsnorm import replace_rms_norms

    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, attn_implementation="sdpa", local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    if model.config.model_type != "olmoe":
        raise ValueError("only the pinned OLMoE architecture is supported")
    replace_rms_norms(model, "fused")
    return model


def _attention_w8(model):
    from lemoq.quant.linear import SchemeSpec, quantize_model

    quantize_model(
        model, SchemeSpec.from_flags("w8a16", 128, targets=("attention",), dequant="fused")
    )


def _check_freeze(evidence: Path, bundle: Path, manifest: dict):
    report = json.loads(evidence.read_text())
    if (
        report.get("status") != "validated"
        or report.get("eval_split") != "validation"
        or not report.get("gate", {}).get("passed")
        or report.get("bundle_sha256") != file_sha256(bundle / "manifest.json")
        or report.get("model_id") != manifest["model_id"]
        or manifest.get("expert_bits") not in (4, 8)
        or report.get("expert_bits") != manifest.get("expert_bits")
        or report.get("precision_plan") != manifest.get("precision_plan")
    ):
        raise ValueError(
            "test evaluation requires successful frozen validation evidence for this exact bundle"
        )


def check_ctx1024(evidence: Path, bundle: Path, split: str):
    report = json.loads(evidence.read_text())
    if (
        report.get("status") != "validated"
        or not report.get("gate", {}).get("passed")
        or report.get("eval_split") != split
        or report.get("config", {}).get("ctx") != 1024
        or report.get("bundle_sha256") != file_sha256(bundle / "manifest.json")
    ):
        raise ValueError("ctx=4096 requires a passed ctx=1024 result for this exact bundle/split")


def profile_opportunity(head: dict, tail: dict) -> dict:
    """One bounded follow-up spec; never present its ceiling as an achieved saving."""
    valid = all(
        finite_number(p.get("wall_p50_ms"), positive=True)
        and finite_number(p.get("host_share_pct"), maximum=5.0)
        for p in (head, tail)
    )
    drift = drift_pct(head["wall_p50_ms"], tail["wall_p50_ms"]) if valid else None
    if not valid or abs(drift) > 5:
        return {
            "status": "invalid",
            "candidate": None,
            "reason": "graph canary/host-share failed",
            "failed_evidence": {"head": head, "tail": tail},
        }
    rows = head.get("top_kernels") or []
    if any(not finite_number(r.get("self_ms_per_token")) for r in rows):
        return {"status": "invalid", "candidate": None, "reason": "invalid kernel bucket evidence"}
    projection = sum(
        r.get("self_ms_per_token", 0) for r in rows if "dequant_gemm" in str(r.get("name", ""))
    )
    return {
        "status": "diagnostic",
        "canary_drift_pct": drift,
        "head": head,
        "tail": tail,
        "candidate": (
            {
                "name": "qkv_projection_fusion",
                "status": "to_be_evaluated",
                "removed_bucket_ceiling_ms": projection * 0.75,
                "note": "At most q/k/v's 3/4 of the four-projection bucket; weight reads remain, so actual savings must be smaller and require a separate A/B.",
            }
            if projection > 0
            else None
        ),
    }


def _calibrate(args, run: Path, payload: dict, model_id: str, tokenizer: str, budget: Budget):
    provenance = json.loads(Path(args.data_manifest).read_text())
    train, train_meta = cached_windows(
        Path(args.train_cache),
        provenance,
        split="train",
        tokenizer=tokenizer,
        count=128,
        length=512,
    )
    validation, val_meta = cached_windows(
        Path(args.validation_cache),
        provenance,
        split="validation",
        tokenizer=tokenizer,
        count=16,
        length=512,
    )
    algorithm, bits = ARMS[args.arm]
    payload.update(
        data={"train": train_meta, "validation": val_meta},
        dataset_revision=provenance["revision"],
        tokenizer_sha256=tokenizer,
        algorithm=algorithm,
        data_provenance=provenance,
        expert_bits=bits,
        precision_plan="experts%d_attention8_activation16" % bits,
        group_size=128,
        algorithm_parameters={
            "block_size": 128,
            "damp": 0.01,
            "act_order": False,
            "min_samples": 256,
            "sample_cap": 2048,
        },
    )
    w8_layers = []
    if args.fallback_from:
        failed = json.loads(Path(args.fallback_from).read_text())
        parent = json.loads((Path(args.parent_bundle) / "manifest.json").read_text())
        if (
            args.arm != "experts_w4_gptq"
            or failed.get("status") != "validated"
            or failed.get("eval_split") != "validation"
            or failed.get("same_path_passed") is not True
            or failed.get("gate", {}).get("passed") is not False
            or failed.get("bundle_sha256")
            != file_sha256(Path(args.parent_bundle) / "manifest.json")
            or parent.get("model_id") != model_id
            or parent.get("algorithm") != "gptq"
            or parent.get("w8_layers")
            or parent.get("data", {}).get("validation") != val_meta
        ):
            raise ValueError(
                "fallback requires the failed full-W4 validation and its exact bundle/data"
            )
        w8_layers = fallback_layers(parent["layers"])
    payload["w8_layers"] = w8_layers
    payload["layers"] = []
    payload["coverage_events"] = []
    w8_reference = None
    if bits == 4:
        if not args.w8_bundle or not args.w8_evidence:
            raise ValueError(
                "W4 requires the complete W8 control bundle and passed validation evidence"
            )
        w8_reference = json.loads((Path(args.w8_bundle) / "manifest.json").read_text())
        if (
            w8_reference.get("status") != "complete"
            or w8_reference.get("expert_bits") != 8
            or w8_reference.get("model_id") != model_id
            or w8_reference.get("data") != payload["data"]
            or w8_reference.get("data_provenance") != provenance
        ):
            raise ValueError(
                "W8 control must use the same checkpoint and calibration/validation data"
            )
        _check_freeze(Path(args.w8_evidence), Path(args.w8_bundle), w8_reference)
        payload["w8_control"] = {
            "bundle_sha256": file_sha256(Path(args.w8_bundle) / "manifest.json"),
            "validation_sha256": file_sha256(Path(args.w8_evidence)),
        }
    model = _load_model(args)
    budget.check()
    _attention_w8(model)
    actual_layers = len(model.model.layers)

    def progress(row):
        if "projection" in row:
            payload["coverage_events"].append(row)
        _write(run / "manifest.json", payload)

    def save(index, quant, summary):
        filename = "layer_%03d.pt" % index
        digest = save_packed(run / filename, quant)
        payload["layers"].append({**summary, "file": filename, "sha256": digest})
        _write(run / "manifest.json", payload)

    calibrate_model(
        model,
        train,
        validation,
        algorithm=algorithm,
        bits=bits,
        check=budget.check,
        progress=progress,
        save_layer=save,
        w8_layers=w8_layers,
    )
    if len(payload["layers"]) != actual_layers:
        raise ValueError("incomplete layer export")
    payload["packed_expert_bytes"] = sum(r["packed_bytes"] for r in payload["layers"])
    if w8_reference:
        payload["expert_byte_reduction_vs_w8"] = (
            1 - payload["packed_expert_bytes"] / w8_reference["packed_expert_bytes"]
        )
        if payload["expert_byte_reduction_vs_w8"] < 0.30:
            raise ValueError("expert byte reduction below the 30% target")
    payload["status"] = "complete"


def _ppl(model, windows, budget):
    import torch.nn.functional as F

    nll, n = 0.0, 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for window in windows:
            budget.check()
            ids = window.to(device)
            logits = model(ids, use_cache=False).logits
            nll += float(
                F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
            )
            n += ids.shape[1] - 1
    return math.exp(nll / n)


def _validate(args, run: Path, payload: dict, model_id: str, tokenizer: str, budget: Budget):
    from benchmarks.cudagraph import StaticCacheDecodeSession, compare_logits
    from lemoq.kernels.attn_decode import replace_attention_cores
    from lemoq.kernels.moe_static import replace_moe_blocks
    from tools.check_fusion_candidate import _same_path

    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("model_id") != model_id:
        raise ValueError("complete matching bundle required")
    if args.ctx == 4096:
        if not args.ctx1024_evidence:
            raise ValueError("ctx=4096 requires --ctx1024-evidence")
        check_ctx1024(Path(args.ctx1024_evidence), bundle, args.eval_split)
    if args.eval_split == "test":
        if not args.freeze_evidence:
            raise ValueError("test is held out; --freeze-evidence is required")
        _check_freeze(Path(args.freeze_evidence), bundle, manifest)
    provenance = json.loads(Path(args.data_manifest).read_text())
    if (
        provenance.get("revision") != manifest["dataset_revision"]
        or tokenizer != manifest["tokenizer_sha256"]
        or provenance != manifest.get("data_provenance")
    ):
        raise ValueError("validation provenance differs from calibrated checkpoint")
    cache = args.test_cache if args.eval_split == "test" else args.validation_cache
    windows, meta = cached_windows(
        Path(cache),
        provenance,
        split=args.eval_split,
        tokenizer=tokenizer,
        count=5,
        length=args.ctx,
        randomize=False,
    )
    ppl_windows, _ = cached_windows(
        Path(cache),
        provenance,
        split=args.eval_split,
        tokenizer=tokenizer,
        count=8,
        length=1024,
        randomize=False,
    )
    payload.update(
        bundle_sha256=file_sha256(bundle / "manifest.json"),
        eval_split=args.eval_split,
        evaluation_data=meta,
        development_only=args.eval_split != "test",
        precision_plan=manifest["precision_plan"],
        expert_bits=manifest["expert_bits"],
        w8_layers=manifest.get("w8_layers", []),
        windows=[],
    )
    model = _load_model(args)
    budget.check()
    if args.ctx > model.config.max_position_embeddings:
        raise ValueError("ctx exceeds model limit")
    references = []
    for window in windows:
        budget.check()
        session = StaticCacheDecodeSession(model, window.cuda(), ctx=args.ctx, max_new_tokens=32)
        references.append(collect(session, steps=32, graph=False))
        del session
    ref_ppl = _ppl(model, ppl_windows, budget)
    quantized = load_bundle(bundle, model, expected_model_id=model_id)
    replace_moe_blocks(model, impl="grouped", prequantized=quantized)
    del quantized
    _attention_w8(model)
    replace_attention_cores(model, backend="triton")
    reference_logits, candidate_logits = [], []
    self_checks = []
    for i, (window, ref) in enumerate(zip(windows, references)):
        budget.check()
        session = StaticCacheDecodeSession(model, window.cuda(), ctx=args.ctx, max_new_tokens=32)
        session.capture(warmup=5)
        eager = collect(session, steps=32, graph=False, forced=ref["inputs"])
        graph = collect(session, steps=32, graph=True, forced=ref["inputs"])
        free = collect(session, steps=32, graph=True)
        check = _same_path(eager, graph)
        self_checks.append(check["passed"])
        metrics = compare_logits(ref["logits"], graph["logits"])
        payload["windows"].append(
            {
                "window": i,
                **metrics,
                "same_path": check,
                "hf_inputs": ref["inputs"],
                "candidate_free_tokens": free["tokens"],
                "hf_free_tokens": ref["tokens"],
            }
        )
        reference_logits.extend(ref["logits"])
        candidate_logits.extend(graph["logits"])
        _write(run / "manifest.json", payload)
        del session
    quant_ppl = _ppl(model, ppl_windows, budget)
    metrics = compare_logits(reference_logits, candidate_logits)
    delta = quant_ppl / ref_ppl - 1
    gate = accuracy_gate(metrics, delta, expert_bits=manifest["expert_bits"])
    gate["passed"] = gate["passed"] and all(self_checks)
    payload.update(
        status="validated",
        aggregate=metrics,
        gate=gate,
        same_path_passed=all(self_checks),
        ppl={"reference": ref_ppl, "candidate": quant_ppl, "relative_delta": delta},
        kl_window_sd=float(torch.tensor([r["kl"] for r in payload["windows"]]).std()),
    )
    if args.profile and gate["passed"]:
        from benchmarks.cudagraph import measure_graph_gap

        budget.check()
        session = StaticCacheDecodeSession(
            model, windows[0].cuda(), ctx=args.ctx, max_new_tokens=32
        )
        session.capture(warmup=5)
        head = measure_graph_gap(session, warmup=5, timed=20, repeats=3)
        budget.check()
        tail = measure_graph_gap(session, warmup=5, timed=20, repeats=3)
        payload["profile"] = profile_opportunity(head, tail)
        payload["performance_measured"] = True  # diagnostic only; no speedup or cell publication


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("plan", "calibrate", "validate"), default="plan")
    ap.add_argument("--arm", choices=tuple(ARMS), default="experts_w4_gptq")
    ap.add_argument("--model")
    ap.add_argument("--data-manifest")
    ap.add_argument("--train-cache")
    ap.add_argument("--validation-cache")
    ap.add_argument("--test-cache")
    ap.add_argument("--bundle")
    ap.add_argument("--w8-bundle")
    ap.add_argument("--w8-evidence")
    ap.add_argument("--parent-bundle")
    ap.add_argument("--fallback-from")
    ap.add_argument("--freeze-evidence")
    ap.add_argument("--ctx1024-evidence")
    ap.add_argument(
        "--profile",
        action="store_true",
        help="optional gated diagnostic profile, never a speedup claim",
    )
    ap.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    ap.add_argument("--ctx", type=int, choices=(1024, 4096), default=1024)
    ap.add_argument("--out", help="NEW run directory, outside results/ on Mac")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpu-budget-seconds", type=float)
    args = ap.parse_args(argv)
    payload = {
        "generated_by": "tools/w4_calibrate.py",
        "reserved_experiment": "EXP-p1-05-w4-expert-calibration",
        "status": "planned",
        "config": vars(args),
        "resource_plan": resource_plan(),
        "adoption_verdict": "not_evaluated",
        "performance_measured": False,
        "required_data": {
            "status": "complete",
            "dataset": "wikitext-2-raw-v1",
            "revision": "pinned dataset commit",
            "tokenizer_sha256": "local tokenizer digest",
            "splits": {
                s: {
                    "file_sha256": "sha256 of .pt",
                    "token_sha256": "sha256 of flattened int64 token bytes",
                }
                for s in ("train", "validation", "test")
            },
        },
    }
    if args.execute and args.dry_run:
        ap.error("--execute and --dry-run are mutually exclusive")
    if not args.execute or args.phase == "plan":
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    required = [args.model, args.data_manifest, args.validation_cache, args.out]
    required += [args.train_cache] if args.phase == "calibrate" else [args.bundle]
    if (
        not all(required)
        or args.gpu_budget_seconds is None
        or not math.isfinite(args.gpu_budget_seconds)
        or args.gpu_budget_seconds <= 0
    ):
        ap.error(
            "local inputs, new --out and a confirmed positive --gpu-budget-seconds are required"
        )
    if args.fallback_from and not args.parent_bundle:
        ap.error("--fallback-from requires --parent-bundle")
    if args.eval_split == "test" and args.phase == "validate" and not args.test_cache:
        ap.error("test requires --test-cache")
    run = Path(args.out)
    if run.exists():
        ap.error("output directory already exists; preserve prior evidence")
    run.mkdir(parents=True)
    _write(run / "manifest.json", payload)
    try:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            text=True,
        )
        if active.strip():
            raise RuntimeError("GPU occupied; wait without killing processes: " + active.strip())
        tokenizer = tokenizer_identity(Path(args.model))
        model_id = model_identity(Path(args.model))
        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
            raise RuntimeError("execution requires a single sm_120 CUDA device")
        budget = Budget(args.gpu_budget_seconds)
        torch.set_float32_matmul_precision("highest")
        payload.update(
            source=_source_identity(),
            model_id=model_id,
            runtime={p: importlib.metadata.version(p) for p in ("torch", "triton", "transformers")},
        )
        with candidate_env(None), torch.no_grad():
            if args.phase == "calibrate":
                _calibrate(args, run, payload, model_id, tokenizer, budget)
            else:
                _validate(args, run, payload, model_id, tokenizer, budget)
            budget.check()  # include the final forward/library call before accepting completion
    except Exception as exc:  # incomplete artifacts retain coverage/finished layers, never loadable
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    _write(run / "manifest.json", payload)
    return (
        0
        if payload["status"] == "complete"
        or (payload["status"] == "validated" and payload["gate"]["passed"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
