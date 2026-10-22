"""Real tiny-MoE integration, export integrity and no-requantization guarantees."""

import copy
import json

import pytest
import torch

from lemoq.kernels.moe_static import replace_moe_blocks
from lemoq.quant import rtn
from lemoq.quant.expert_calibration import (
    calibrate_model,
    fallback_layers,
    layer_forward,
    load_bundle,
    resource_plan,
    save_packed,
)
from lemoq.quant.gptq import CalibrationError


def _model(layers=2):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(45)
    config = OlmoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    return OlmoeForCausalLM(config).float().eval()


@pytest.fixture
def calibrated():
    model = _model()
    original = copy.deepcopy(model)
    train = [torch.arange(1, 17).reshape(1, -1), torch.arange(17, 33).reshape(1, -1)]
    val = [torch.arange(33, 41).reshape(1, -1), torch.arange(41, 49).reshape(1, -1)]
    layers, events = {}, []
    summaries = calibrate_model(
        model,
        train,
        val,
        algorithm="gptq",
        group_size=4,
        min_samples=16,
        max_samples=32,
        progress=events.append,
        save_layer=lambda i, q, row: layers.update({i: q}),
    )
    return original, model, layers, summaries, events


def _save(directory, layers, summaries):
    directory.mkdir()
    rows = []
    for i, q in layers.items():
        name = f"layer_{i}.pt"
        sha = save_packed(directory / name, q)
        rows.append({**summaries[i], "file": name, "sha256": sha})
    manifest = {"status": "complete", "model_id": "tiny", "layers": rows}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_layer_replay_matches_real_model_contract():
    model = _model()
    ids = torch.arange(1, 9).reshape(1, -1)
    with torch.no_grad():
        hidden = model.get_input_embeddings()(ids)
        for i in range(2):
            hidden = layer_forward(model, i, hidden)
        expected = model.model(ids, use_cache=False).last_hidden_state
        torch.testing.assert_close(model.model.norm(hidden), expected, atol=1e-6, rtol=1e-6)


def test_sequential_calibration_refreshes_down_statistics_and_preserves_nonexperts(calibrated):
    original, model, packed, summaries, events = calibrated
    assert [r["projection"] for r in events if "projection" in r] == ["gate_proj", "down_proj"] * 2
    assert list(packed) == [0, 1]
    for row in summaries:
        assert row["algorithm"] == "gptq" and row["bits"] == 4
        assert row["packed_bytes"] == packed[row["layer"]].nbytes()
        assert all(
            s["samples"] == 32 for bucket in row["coverage"].values() for s in bucket.values()
        )
        assert row["validation_relative_mse"] >= 0
    for name, p in original.named_parameters():
        other = dict(model.named_parameters())[name]
        if ".experts." not in name:
            assert torch.equal(p, other), name
    for index, layer in enumerate(model.model.layers):
        q = packed[index]
        gu = q.dequant_gate_up(dtype=torch.float32)
        down = q.dequant_down(dtype=torch.float32)
        for e, expert in enumerate(layer.mlp.experts):
            assert torch.equal(expert.gate_proj.weight, gu[e, :8])
            assert torch.equal(expert.up_proj.weight, gu[e, 8:])
            assert torch.equal(expert.down_proj.weight, down[e])


def test_export_reload_and_inference_never_requantizes(calibrated, tmp_path, monkeypatch):
    original, calibrated_model, packed, summaries, _ = calibrated
    bundle = tmp_path / "bundle"
    _save(bundle, packed, summaries)
    reloaded = load_bundle(bundle, original, expected_model_id="tiny")
    for i in packed:
        assert torch.equal(reloaded[i].packed_gate_up, packed[i].packed_gate_up)
        assert torch.equal(reloaded[i].scale_down, packed[i].scale_down)

    def forbidden(*a, **kw):
        raise AssertionError("calibrated checkpoints must not be quantized/fused again")

    monkeypatch.setattr(rtn, "quantize_rtn", forbidden)
    import lemoq.kernels.moe_static as ms

    monkeypatch.setattr(ms, "fuse_experts", forbidden)
    with torch.no_grad():
        ids = torch.arange(1, 9).view(1, -1)
        expected = calibrated_model(ids, use_cache=False).logits
        assert replace_moe_blocks(original, impl="grouped", prequantized=reloaded) == 2
        got = original(ids, use_cache=False).logits  # multi-token CPU prefill, not a CUDA timing
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "case", ["incomplete", "checkpoint", "missing_layer", "precision", "traversal", "corrupt"]
)
def test_bad_bundles_fail_before_model_mutation(calibrated, tmp_path, case):
    original, _, packed, summaries, _ = calibrated
    bundle = tmp_path / "bundle"
    manifest = _save(bundle, packed, summaries)
    if case == "incomplete":
        manifest["status"] = "failed"
    elif case == "checkpoint":
        manifest["model_id"] = "other"
    elif case == "missing_layer":
        manifest["layers"].pop()
    elif case == "precision":
        manifest["layers"][0]["bits"] = 8
    elif case == "traversal":
        manifest["layers"][0]["file"] = "../other.pt"
    elif case == "corrupt":
        (bundle / manifest["layers"][0]["file"]).write_bytes(b"broken")
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_bundle(bundle, original, expected_model_id="tiny")
    assert all(hasattr(layer.mlp, "experts") for layer in original.model.layers)


def test_missing_expert_coverage_does_not_export_partial_layer():
    model = _model()
    model.model.layers[0].mlp.top_k = 1
    with torch.no_grad():
        model.model.layers[0].mlp.gate.weight.zero_()  # ties choose just one of two experts
    saved, events = [], []
    with pytest.raises(CalibrationError, match="coverage"):
        calibrate_model(
            model,
            [torch.arange(1, 17).view(1, -1)],
            [torch.arange(2, 10).view(1, -1)],
            algorithm="gptq",
            group_size=4,
            min_samples=4,
            progress=events.append,
            save_layer=lambda *a: saved.append(a),
        )
    assert not saved and events[0]["coverage"]
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_invalid_packed_payload_rejected_before_any_replacement(calibrated):
    original, _, packed, _, _ = calibrated
    packed[1].scale_down[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="scales"):
        replace_moe_blocks(original, impl="grouped", prequantized=packed)
    assert all(hasattr(layer.mlp, "experts") for layer in original.model.layers)


def test_rtn_arm_needs_no_calibration_and_budget_stops_export():
    model = _model()
    rows = calibrate_model(
        model, [], [torch.arange(1, 9).view(1, -1)], algorithm="rtn", bits=8, group_size=4
    )
    assert len(rows) == 2 and all(r["bits"] == 8 for r in rows)

    def exhausted():
        raise TimeoutError("budget")

    with pytest.raises(TimeoutError):
        calibrate_model(
            _model(),
            [],
            [torch.arange(1, 9).view(1, -1)],
            algorithm="rtn",
            bits=8,
            group_size=4,
            check=exhausted,
        )


def test_fallback_is_layerwise_and_bounded_by_parameter_bytes():
    rows = [
        {"layer": i, "expert_params": 100, "validation_relative_mse": float(i)} for i in range(16)
    ]
    assert fallback_layers(rows) == [15, 14, 13, 12]
    with pytest.raises(ValueError):
        fallback_layers(rows, max_fraction=0.5)
    with pytest.raises(ValueError, match="75%"):
        calibrate_model(
            _model(),
            [torch.ones(1, 16, dtype=torch.long)],
            [torch.ones(1, 8, dtype=torch.long)],
            algorithm="gptq",
            group_size=4,
            min_samples=4,
            w8_layers=[0],
        )


def test_resource_estimates_are_arithmetic_and_not_runtime_claims():
    p = resource_plan()
    assert p["train_tokens"] == 65536 and p["validation_tokens"] == 8192
    assert p["max_gram_bytes_one_layer"] == 64 * 2048 * 2048 * 4
    assert p["expert_w4_bytes"] < p["expert_w8_bytes"]
    assert p["gpu_seconds"] is p["rental_cost"] is None


def test_checkpoint_overwrite_refused(calibrated, tmp_path):
    path = tmp_path / "layer.pt"
    save_packed(path, calibrated[2][0])
    with pytest.raises(FileExistsError):
        save_packed(path, calibrated[2][0])


def test_bounded_w8_fallback_preserves_remaining_layers_at_w4():
    model = _model(layers=4)
    saved = {}
    rows = calibrate_model(
        model,
        [torch.arange(1, 17).view(1, -1)],
        [torch.arange(17, 25).view(1, -1)],
        algorithm="gptq",
        group_size=4,
        min_samples=4,
        w8_layers=[2],
        save_layer=lambda i, q, row: saved.update({i: q}),
    )
    assert [r["bits"] for r in rows] == [4, 4, 8, 4]
    assert [saved[i].bits for i in range(4)] == [4, 4, 8, 4]


def test_down_hessian_uses_new_quantized_gate_up_activations(monkeypatch):
    from lemoq.quant.gptq import HessianStats

    model = _model()
    observed, hessians = [], []
    handle = (
        model.model.layers[0]
        .mlp.experts[0]
        .down_proj.register_forward_pre_hook(
            lambda _m, args: observed.append(args[0].detach().clone())
        )
    )
    actual = HessianStats.hessian

    def record(self, **kwargs):
        h = actual(self, **kwargs)
        if h.shape == (8, 8):
            hessians.append(h.clone())
        return h

    monkeypatch.setattr(HessianStats, "hessian", record)
    try:
        calibrate_model(
            model,
            [torch.arange(1, 17).view(1, -1), torch.arange(17, 33).view(1, -1)],
            [torch.arange(33, 41).view(1, -1), torch.arange(41, 49).view(1, -1)],
            algorithm="gptq",
            group_size=4,
            min_samples=16,
            max_samples=32,
        )
    finally:
        handle.remove()
    # 2 validation references, 2 gate-stat passes, then 2 freshly quantized down-stat passes.
    assert not torch.equal(observed[2], observed[4])
    refreshed = torch.cat(observed[4:6])
    torch.testing.assert_close(
        hessians[0], 2 * refreshed.T @ refreshed / len(refreshed), atol=1e-6, rtol=1e-6
    )


@pytest.mark.gpu
def test_gptq_packed_weights_match_independent_dense_oracle_on_gpu():
    import torch.nn.functional as F

    from lemoq.kernels.moe_grouped_quant import PARITY_GATE, grouped_moe_forward_quant
    from lemoq.quant import quantize_linear
    from lemoq.quant.expert_calibration import pack_expert_results
    from lemoq.quant.gptq import HessianStats

    torch.manual_seed(19)
    gu, down = [], []
    for _ in range(2):
        hs = HessianStats.empty(256)
        hs.add(torch.randn(256, 256))
        ds = HessianStats.empty(128)
        ds.add(torch.randn(256, 128))
        gu.append(quantize_linear(torch.randn(256, 256) * 0.02, scheme="gptq", calibration=hs))
        down.append(quantize_linear(torch.randn(256, 128) * 0.02, scheme="gptq", calibration=ds))
    q = pack_expert_results(gu, down).to(device="cuda")
    x = torch.randn(1, 256, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor([[0, 1]], device="cuda", dtype=torch.int64)
    weights = torch.tensor([[0.6, 0.4]], device="cuda", dtype=torch.float16)
    got = grouped_moe_forward_quant(x, q, ids, weights)
    baseline = grouped_moe_forward_quant(x, q, ids, weights, backend="oracle")
    gw, dw = q.dequant_gate_up(dtype=torch.float64), q.dequant_down(dtype=torch.float64)
    truth = torch.zeros(256, device="cuda", dtype=torch.float64)
    for i in range(2):
        projected = gw[i] @ x.double().reshape(-1)
        act = F.silu(projected[:128]) * projected[128:]
        truth += (dw[i] @ act) * weights[0, i].double()
    err = (got.double().reshape(-1) - truth).abs().max()
    ref_err = (baseline.double().reshape(-1) - truth).abs().max()
    assert torch.allclose(got, baseline, atol=PARITY_GATE["tol_atol"], rtol=PARITY_GATE["tol_rtol"])
    assert err <= ref_err * PARITY_GATE["tol_slack"] + 1e-12
