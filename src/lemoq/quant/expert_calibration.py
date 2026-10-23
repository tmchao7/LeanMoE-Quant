"""Layer-wise MoE calibration, packed checkpoints and pre-quantized inference loading.

Calibration runs eager forwards only, one layer at a time. It never modifies the
experiment matrix, gate citations, inference kernel bodies or default quantizer.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch

from lemoq.kernels.moe_grouped_quant import QuantFusedExperts
from lemoq.quant.api import dequantize_reference, quantize_linear
from lemoq.quant.gptq import CalibrationError, HessianStats
from lemoq.quant.pack import pack_4bit


def validate_packed(q: QuantFusedExperts, *, experts: int, hidden: int, ffn: int) -> None:
    if not isinstance(q, QuantFusedExperts) or q.bits not in (4, 8):
        raise ValueError("invalid pre-quantized expert container")
    if type(q.group_size) is not int or q.group_size < 2 or q.group_size % 2:
        raise ValueError("invalid grouped scale size")
    if hidden % q.group_size or ffn % q.group_size:
        raise ValueError("group size must divide hidden and ffn")
    divisor = 8 // q.bits
    shapes = {
        "packed_gate_up": (experts, 2 * ffn, hidden // divisor),
        "packed_down": (experts, hidden, ffn // divisor),
        "scale_gate_up": (experts, 2 * ffn, hidden // q.group_size),
        "scale_down": (experts, hidden, ffn // q.group_size),
    }
    for name, shape in shapes.items():
        value = getattr(q, name)
        dtype = (
            torch.float16
            if name.startswith("scale")
            else (torch.uint8 if q.bits == 4 else torch.int8)
        )
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or not value.is_contiguous()
        ):
            raise ValueError("packed shape/dtype/layout mismatch: " + name)
        # RTN computes scales in fp32, then stores fp16. Tiny, valid groups can
        # underflow to zero; the existing inference kernel dequantizes them to zero.
        if name.startswith("scale") and (
            not bool(torch.isfinite(value).all()) or not bool((value >= 0).all())
        ):
            raise ValueError("scales must be finite and nonnegative")
    if len({getattr(q, name).device for name in shapes}) != 1:
        raise ValueError("packed buffers must share a device")


def _pack(q: Any):
    codes = q.codes.detach()
    packed = pack_4bit(codes) if q.bits == 4 else codes.to(torch.int8)
    return packed.cpu().contiguous(), q.scale.detach().half().cpu().contiguous()


def pack_expert_results(gate_up: list[Any], down: list[Any]) -> QuantFusedExperts:
    if not gate_up or len(gate_up) != len(down):
        raise ValueError("gate/up and down must cover the same experts")
    bits, group = gate_up[0].bits, gate_up[0].group_size
    if any(q.bits != bits or q.group_size != group for q in gate_up + down):
        raise ValueError("v1 has one expert bit-width per layer")
    gu, ds = [_pack(q) for q in gate_up], [_pack(q) for q in down]
    result = QuantFusedExperts(
        torch.stack([p for p, _ in gu]),
        torch.stack([s for _, s in gu]),
        torch.stack([p for p, _ in ds]),
        torch.stack([s for _, s in ds]),
        bits,
        group,
    )
    validate_packed(result, experts=len(gu), hidden=gate_up[0].shape[1], ffn=down[0].shape[1])
    return result


@torch.no_grad()
def layer_forward(model: Any, index: int, hidden: Any):
    """The pinned OLMoE eager layer contract, without cache or model-wide replays."""
    seq = hidden.shape[1]
    pos = torch.arange(seq, device=hidden.device, dtype=torch.long)
    mask = torch.full(
        (1, 1, seq, seq), torch.finfo(hidden.dtype).min, dtype=hidden.dtype, device=hidden.device
    ).triu(1)
    embeddings = model.model.rotary_emb(hidden, pos.unsqueeze(0))
    return model.model.layers[index](
        hidden,
        attention_mask=mask,
        position_ids=pos.unsqueeze(0),
        cache_position=pos,
        position_embeddings=embeddings,
        use_cache=False,
        output_attentions=False,
    )[0]


def _stats_pass(
    model: Any,
    index: int,
    inputs: list[Any],
    projection: str,
    check: Callable[[], None],
    *,
    max_samples: int,
    min_samples: int,
    progress: Callable[[dict], None],
):
    block = model.model.layers[index].mlp
    device = next(model.parameters()).device
    stats = {}
    handles = []
    try:
        for e, expert in enumerate(block.experts):
            module = getattr(expert, projection)
            stat = HessianStats.empty(module.in_features, device=device, max_samples=max_samples)
            stats[e] = stat

            def hook(_module, args, _stat=stat):
                _stat.add(args[0])

            handles.append(module.register_forward_pre_hook(hook))
        for hidden in inputs:
            check()
            layer_forward(model, index, hidden.to(device))
    finally:
        for handle in handles:
            handle.remove()
    coverage = {str(e): s.summary() for e, s in stats.items()}
    progress({"layer": index, "projection": projection, "coverage": coverage})
    if any(s.samples < min_samples for s in stats.values()):
        raise CalibrationError(
            "insufficient expert coverage at layer %d / %s" % (index, projection)
        )
    return stats


@torch.no_grad()
def calibrate_model(
    model: Any,
    train: list[Any],
    validation: list[Any],
    *,
    algorithm: str,
    bits: int = 4,
    group_size: int = 128,
    min_samples: int = 256,
    max_samples: int = 2048,
    check: Callable[[], None] = lambda: None,
    progress: Callable[[dict], None] = lambda row: None,
    save_layer: Callable[[int, QuantFusedExperts, dict], None] = lambda *a: None,
    w8_layers: tuple[int, ...] | list[int] = (),
):
    """Quantize experts, propagate their actual reconstructed outputs to the next layer.

    Caller fixes attention at W8 before entering. Router, norm, embedding and
    head are never modified here. RTN arms need no calibration train forwards.
    """
    if (
        algorithm not in ("rtn", "gptq")
        or bits not in (4, 8)
        or (algorithm == "gptq" and bits != 4)
    ):
        raise ValueError("unsupported calibration arm")
    if not validation or (algorithm == "gptq" and not train) or min_samples > max_samples:
        raise CalibrationError("train/validation or sample budget missing")
    if len(set(w8_layers)) != len(w8_layers) or any(
        type(i) is not int or i < 0 or i >= len(model.model.layers) for i in w8_layers
    ):
        raise ValueError("invalid fallback layer IDs")
    params_by_layer = [
        sum(p.numel() for expert in layer.mlp.experts for p in expert.parameters())
        for layer in model.model.layers
    ]
    if (w8_layers and (algorithm != "gptq" or bits != 4)) or sum(
        params_by_layer[i] for i in w8_layers
    ) > sum(params_by_layer) * 0.25:
        raise ValueError("fallback must preserve at least 75% of expert parameters at W4")
    device = next(model.parameters()).device
    train_hidden = (
        [model.get_input_embeddings()(ids.to(device)).detach().cpu() for ids in train]
        if algorithm == "gptq"
        else []
    )
    val_hidden = [model.get_input_embeddings()(ids.to(device)).detach().cpu() for ids in validation]
    summaries = []
    for index, layer in enumerate(model.model.layers):
        check()
        block = layer.mlp
        layer_algorithm = "rtn" if index in w8_layers else algorithm
        layer_bits = 8 if index in w8_layers else bits
        references = []
        for h in val_hidden:
            check()
            references.append(layer_forward(model, index, h.to(device)).cpu())
        gu_stats = (
            _stats_pass(
                model,
                index,
                train_hidden,
                "gate_proj",
                check,
                max_samples=max_samples,
                min_samples=min_samples,
                progress=progress,
            )
            if layer_algorithm == "gptq"
            else {}
        )
        gu_results, down_results = [], []
        coverage = {"gate_up": {str(e): s.summary() for e, s in gu_stats.items()}}
        for e, expert in enumerate(block.experts):
            check()
            w = torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0)
            q = quantize_linear(
                w,
                bits=layer_bits,
                group_size=group_size,
                scheme=layer_algorithm,
                calibration=gu_stats.get(e),
                min_samples=min_samples,
            )
            q.scale = q.scale.half()
            restored = dequantize_reference(q).to(expert.gate_proj.weight.dtype)
            ffn = expert.gate_proj.out_features
            expert.gate_proj.weight.copy_(restored[:ffn])
            expert.up_proj.weight.copy_(restored[ffn:])
            # Keep only CPU packed/export data across experts, not CUDA Hessians/results.
            q.codes, q.scale, q.zero = q.codes.cpu(), q.scale.cpu(), q.zero.cpu()
            gu_results.append(q)
        del gu_stats
        down_stats = (
            _stats_pass(
                model,
                index,
                train_hidden,
                "down_proj",
                check,
                max_samples=max_samples,
                min_samples=min_samples,
                progress=progress,
            )
            if layer_algorithm == "gptq"
            else {}
        )
        coverage["down"] = {str(e): s.summary() for e, s in down_stats.items()}
        for e, expert in enumerate(block.experts):
            check()
            q = quantize_linear(
                expert.down_proj.weight,
                bits=layer_bits,
                group_size=group_size,
                scheme=layer_algorithm,
                calibration=down_stats.get(e),
                min_samples=min_samples,
            )
            q.scale = q.scale.half()
            expert.down_proj.weight.copy_(dequantize_reference(q).to(expert.down_proj.weight.dtype))
            q.codes, q.scale, q.zero = q.codes.cpu(), q.scale.cpu(), q.zero.cpu()
            down_results.append(q)
        del down_stats
        packed = pack_expert_results(gu_results, down_results)
        val_next = []
        numerator = denominator = 0.0
        for hidden, ref in zip(val_hidden, references):
            check()
            got = layer_forward(model, index, hidden.to(device)).cpu()
            if not bool(torch.isfinite(got).all()) or not bool(torch.isfinite(ref).all()):
                raise CalibrationError("non-finite validation layer outputs")
            numerator += float((got.double() - ref.double()).square().sum())
            denominator += float(ref.double().square().sum())
            val_next.append(got)
        params = sum(p.numel() for expert in block.experts for p in expert.parameters())
        summary = {
            "layer": index,
            "algorithm": layer_algorithm,
            "bits": layer_bits,
            "group_size": group_size,
            "expert_params": params,
            "coverage": coverage,
            "validation_relative_mse": numerator / max(denominator, 1e-30),
            "packed_bytes": packed.nbytes(),
        }
        save_layer(index, packed, summary)
        progress(summary)
        summaries.append(summary)
        train_next = []
        for hidden in train_hidden:
            check()
            train_next.append(layer_forward(model, index, hidden.to(device)).cpu())
        train_hidden, val_hidden = train_next, val_next
    return summaries


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_packed(path: Path, quant: QuantFusedExperts) -> str:
    fields = {
        name: getattr(quant, name).cpu()
        for name in ("packed_gate_up", "scale_gate_up", "packed_down", "scale_down")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as fh:
        torch.save({**fields, "bits": quant.bits, "group_size": quant.group_size}, fh)
    return file_sha256(path)


def load_bundle(directory: Path, model: Any, *, expected_model_id: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("model_id") != expected_model_id:
        raise ValueError("incomplete bundle or checkpoint identity mismatch")
    layers = manifest.get("layers", [])
    if len(layers) != len(model.model.layers) or [r.get("layer") for r in layers] != list(
        range(len(layers))
    ):
        raise ValueError("bundle does not cover every model layer exactly once")
    out = {}
    for row, layer in zip(layers, model.model.layers):
        name = row["file"]
        if Path(name).name != name:
            raise ValueError("bundle paths must be local filenames")
        path = directory / name
        if file_sha256(path) != row["sha256"]:
            raise ValueError("packed checkpoint hash mismatch")
        state = torch.load(path, map_location="cpu", weights_only=True)
        q = QuantFusedExperts(**state)
        experts = layer.mlp.experts
        validate_packed(
            q,
            experts=len(experts),
            hidden=experts[0].gate_proj.in_features,
            ffn=experts[0].down_proj.in_features,
        )
        if q.bits != row["bits"] or q.group_size != row["group_size"]:
            raise ValueError("precision manifest does not match packed payload")
        out[row["layer"]] = q
    return out


def fallback_layers(summaries: list[dict], *, max_fraction: float = 0.25) -> list[int]:
    """Whole-layer W8 fallback only, ranked by validation relative reconstruction error."""
    if not 0 <= max_fraction <= 0.25 or not summaries:
        raise ValueError("fallback is bounded to 25% of expert parameters")
    if len({r["layer"] for r in summaries}) != len(summaries) or any(
        type(r["expert_params"]) is not int
        or r["expert_params"] <= 0
        or not math.isfinite(r["validation_relative_mse"])
        or r["validation_relative_mse"] < 0
        for r in summaries
    ):
        raise ValueError("invalid fallback sensitivity evidence")
    total = sum(r["expert_params"] for r in summaries)
    budget, used, selected = total * max_fraction, 0, []
    for row in sorted(summaries, key=lambda r: (-r["validation_relative_mse"], r["layer"])):
        if used + row["expert_params"] <= budget:
            selected.append(row["layer"])
            used += row["expert_params"]
    return selected


def resource_plan(
    *,
    layers: int = 16,
    experts: int = 64,
    hidden: int = 2048,
    ffn: int = 1024,
    train_windows: int = 128,
    seq: int = 512,
    validation_windows: int = 16,
):
    params = layers * experts * 3 * hidden * ffn
    w8 = params + layers * experts * (2 * ffn * hidden // 128 + hidden * ffn // 128) * 2
    w4 = params // 2 + (w8 - params)
    return {
        "kind": "analytic_estimate_not_measurement",
        "train_tokens": train_windows * seq,
        "validation_tokens": validation_windows * seq,
        "expert_matrices": layers * experts * 3,
        "gptq_cholesky_factorizations": layers * experts * 4,
        "gptq_train_layer_forwards": layers * train_windows * 3,
        "validation_layer_forwards_per_arm": layers * validation_windows * 2,
        "expert_fp16_bytes": params * 2,
        "expert_w8_bytes": w8,
        "expert_w4_bytes": w4,
        "max_gram_bytes_one_layer": experts * max(hidden * hidden, ffn * ffn) * 4,
        "factorization_workspace_floor_bytes": 4 * max(hidden, ffn) ** 2 * 4,
        "hidden_cpu_cache_bytes_per_copy": (train_windows + validation_windows) * seq * hidden * 2,
        "export_int32_code_scratch_bytes_one_layer": experts * 3 * hidden * ffn * 4,
        "other_memory": "Full-model weights, current layer activations, CUDA allocator and library workspace must be added; this is not a peak-VRAM guarantee.",
        "gpu_seconds": None,
        "rental_cost": None,
        "time_note": "No runtime estimate is measured offline; budget/pilot approval is required before CUDA execution.",
    }
