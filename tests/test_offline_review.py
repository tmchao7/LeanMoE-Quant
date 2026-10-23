"""Adversarial CPU reproductions for the fusion review; no timing or model download."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from benchmarks.schema import empty_result
from lemoq.kernels import attn_decode as ad
from lemoq.kernels import rope
from tools import occupancy_probe as op
from tools import profile_graph as pg
from tools.round_canary import evaluate


def _canary(value=2.0, host=2.0):
    p = empty_result("EXP-offline-only")
    p["latency"]["per_token_p50_ms"] = value
    p["attribution"]["host_share_pct"] = host
    return p


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0, True, "2"])
@pytest.mark.parametrize("side", ["head", "tail"])
def test_canary_rejects_invalid_latency(bad, side):
    head, tail = _canary(), _canary()
    {"head": head, "tail": tail}[side]["latency"]["per_token_p50_ms"] = bad
    assert evaluate(head, tail)["verdict"] == "VOID"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, "2", 101])
def test_canary_rejects_invalid_host_share(bad):
    assert evaluate(_canary(host=bad), _canary(), graph=True)["verdict"] == "VOID"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, 6.0])
def test_canary_cannot_disable_frozen_drift_limit(bad):
    assert evaluate(_canary(), _canary(), max_drift_pct=bad)["verdict"] == "VOID"


@pytest.mark.parametrize("field", ["exp_id", "config", "runtime", "model", "hardware"])
def test_canary_requires_matching_identity(field):
    head, tail = _canary(), _canary()
    if field == "exp_id":
        tail[field] = "OTHER"
    else:
        tail[field][
            {"config": "ctx", "runtime": "commit", "model": "repo", "hardware": "gpu"}[field]
        ] = "different"
    assert evaluate(head, tail)["verdict"] == "VOID"


def _ab(monkeypatch, mutate=None, rounds=2):
    seen = []

    def invoke(args):
        i = len(seen)
        seen.append(
            {
                key: pg.os.environ.get(key)
                for key in (
                    pg.ROPE_ARM_ENV,
                    pg.ABLATE_ARM_ENV,
                    pg.MASK_FILL_ARM_ENV,
                    pg.PACKED_QKV_ENV,
                )
            }
        )
        res = {
            "gap": {
                "wall_p50_ms": 2.0 if i % 2 == 0 else 1.9,
                "gpu_busy_ms": 1.8,
                "host_share_pct": 2.0,
            },
            "correctness": {"graph_output_tokens": [3, 4, 5], "graph_seed_token": 2},
        }
        if mutate:
            mutate(i, res)
        return res

    monkeypatch.setattr(pg, "_invoke", invoke)
    args = Namespace(ab_rounds=rounds, max_new_tokens=3, attn="kernel")
    return pg.run_fused_rope_ab(args), seen


@pytest.mark.parametrize(
    "key,bad",
    [
        ("wall_p50_ms", float("nan")),
        ("wall_p50_ms", -1),
        ("gpu_busy_ms", "bad"),
        ("host_share_pct", float("inf")),
        ("host_share_pct", 5.1),
    ],
)
def test_ab_rejects_bad_metrics_without_losing_runs(monkeypatch, key, bad):
    p, _ = _ab(monkeypatch, lambda i, r: r["gap"].update({key: bad}) if i == 1 else None)
    assert p["status"] == "invalid"
    assert len(p["runs"]) == 5


@pytest.mark.parametrize("tokens", [None, [], [3], [True, 4, 5]])
def test_ab_missing_or_malformed_tokens_are_not_equal(monkeypatch, tokens):
    p, _ = _ab(monkeypatch, lambda i, r: r["correctness"].update(graph_output_tokens=tokens))
    assert p["status"] == "invalid" and not p["tokens_equal"]


def test_ab_compares_seed(monkeypatch):
    p, _ = _ab(
        monkeypatch, lambda i, r: r["correctness"].update(graph_seed_token=99) if i == 1 else None
    )
    assert p["status"] == "invalid"


def test_ab_encloses_round_and_keeps_tail_out_of_statistics(monkeypatch):
    p, seen = _ab(monkeypatch, lambda i, r: r["gap"].update(wall_p50_ms=2.02) if i == 4 else None)
    assert len(seen) == 5
    assert p["runs"][-1]["role"] == "tail_canary"
    assert p["canary_drift_pct"] == pytest.approx(1.0)
    assert p["mean_wall_baseline_ms"] == 2.0
    assert p["paired_wall"]["wins"] == 2
    assert p["paired_wall"]["median_delta_ms"] == pytest.approx(0.1)
    assert p["mad_wall_baseline_ms"] == 0.0
    assert p["adoption_verdict"] == "not_evaluated"


def test_ab_isolates_other_switches_and_restores_on_success(monkeypatch):
    for key in (pg.ROPE_ARM_ENV, pg.ABLATE_ARM_ENV, pg.MASK_FILL_ARM_ENV, pg.PACKED_QKV_ENV):
        monkeypatch.setenv(key, "1")
    p, seen = _ab(monkeypatch)
    assert all(r[pg.ABLATE_ARM_ENV] == "0" and r[pg.MASK_FILL_ARM_ENV] == "0" for r in seen)
    assert p["runs"][0]["fusion_env"] == seen[0]
    assert all(pg.os.environ[key] == "1" for key in seen[0])


def test_rope_guard_rejects_mismatched_layout_and_dtype(monkeypatch):
    monkeypatch.setattr(rope, "_HAS_TRITON", True)

    def meta(shape=(1, 2, 1, 8), dtype=torch.float16, stride=1, device="cuda:0"):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            is_cuda=True,
            device=device,
            stride=lambda dim: stride,
        )

    q, k, c, s = meta(), meta(), meta((1, 1, 1, 8)), meta((1, 1, 1, 8))
    assert rope._decode_shape_guard(q, k, c, s)
    assert not rope._decode_shape_guard(q, meta(dtype=torch.float32), c, s)
    assert not rope._decode_shape_guard(q, k, c, meta((1, 1, 1, 8), stride=2))
    assert not rope._decode_shape_guard(q, k, c, meta((1, 1, 1, 8), device="cuda:1"))
    assert not rope._decode_shape_guard(meta(()), k, c, s)


@pytest.mark.parametrize("case", ["sequence_stride", "head_stride", "batch", "dim"])
def test_attention_rejects_unsupported_cache_contract(case):
    q = torch.zeros(1, 2, 1, 16, dtype=torch.float16)
    k = torch.zeros(1, 2, 8, 16, dtype=torch.float16)
    v = k.clone()
    if case == "sequence_stride":
        k, v = k[:, :, ::2], v[:, :, ::2]
    elif case == "head_stride":
        v = torch.zeros(1, 4, 8, 16, dtype=k.dtype)[:, ::2]
    elif case == "batch":
        k, v = k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1)
    else:
        k, v = k[..., :8], v[..., :8]
    with pytest.raises((ValueError, NotImplementedError)):
        ad.attn_decode_forward(q, k, v, 2, backend="oracle", config={"num_splits": 2})


@pytest.mark.parametrize(
    "bad",
    [
        torch.tensor([1, 2], dtype=torch.int32),
        torch.tensor([2.0]),
        torch.tensor([-1], dtype=torch.int32),
    ],
)
def test_attention_oracle_validates_length_contract(bad):
    q = torch.zeros(1, 2, 1, 16, dtype=torch.float16)
    k = torch.zeros(1, 2, 8, 16, dtype=q.dtype)
    with pytest.raises(ValueError):
        ad.attn_decode_forward(q, k, k, bad, backend="oracle", config={"num_splits": 2})


def test_empty_routing_never_initializes_or_launches_cuda(monkeypatch):
    from lemoq.kernels import _routing_triton as rt

    def bomb():
        raise AssertionError("empty input must not initialize CUDA or launch")

    monkeypatch.setattr(rt, "_require_cuda_triton", bomb)
    w, i = rt.routing_softmax_topk(torch.empty(0, 64), top_k=8)
    assert w.shape == i.shape == (0, 8)
    assert w.dtype == torch.float32 and i.dtype == torch.int64


def test_sm120_fallback_matches_architecture():
    class Cuda:
        get_device_properties = staticmethod(lambda _: SimpleNamespace(major=12, minor=0))

    assert op._limits(SimpleNamespace(cuda=Cuda))["smem_per_sm"] == 128 * 1024


def test_unknown_architecture_does_not_guess_sm120():
    class Cuda:
        get_device_properties = staticmethod(lambda _: SimpleNamespace(major=99, minor=0))

    with pytest.raises(ValueError):
        op._limits(SimpleNamespace(cuda=Cuda))


def test_ab_tail_canary_and_env_restore_even_after_failure(monkeypatch):
    monkeypatch.setenv(pg.ABLATE_ARM_ENV, "1")
    calls = []

    def boom(args):
        calls.append(pg.os.environ.get(pg.ABLATE_ARM_ENV))
        raise RuntimeError("deliberate capture failure")

    monkeypatch.setattr(pg, "_invoke", boom)
    p = pg.run_routing_ab(Namespace(ab_rounds=2, max_new_tokens=3, attn="kernel"))
    assert p["status"] == "invalid" and len(p["runs"]) == 1
    assert "deliberate capture failure" in p["runs"][0]["error"]
    assert pg.os.environ[pg.ABLATE_ARM_ENV] == "1"


def test_ab_detects_drift_after_final_treatment(monkeypatch):
    p, _ = _ab(monkeypatch, lambda i, r: r["gap"].update(wall_p50_ms=2.2) if i == 4 else None)
    assert p["status"] == "invalid" and p["canary_drift_pct"] == 10.0


def test_ab_mask_cli_rejects_sdpa_before_loading(monkeypatch, tmp_path):
    def bomb(args):
        raise AssertionError("must refuse before loading")

    monkeypatch.setattr(pg, "_invoke", bomb)
    import json

    out = tmp_path / "mask.json"
    assert pg.main(["--model", "m", "--attn", "sdpa", "--ab-mask-fill", "--json", str(out)]) == 1
    assert "mask-independent" in json.loads(out.read_text())["error"]


def test_ab_never_overwrites_existing_evidence(tmp_path):
    out = tmp_path / "existing.json"
    out.write_text("old evidence")
    with pytest.raises(SystemExit):
        pg.main(["--model", "m", "--ab-fused-rope", "--json", str(out)])
    assert out.read_text() == "old evidence"


def test_canary_does_not_round_an_excessive_drift_into_pass():
    assert evaluate(_canary(100), _canary(105.000001))["verdict"] == "VOID"


@pytest.mark.parametrize("head,tail", [(3, 3.15), (2, 2.1), (2, 1.9), (100, 105)])
def test_canary_exactly_five_percent_still_passes(head, tail):
    assert evaluate(_canary(head), _canary(tail))["verdict"] == "PASS"


@pytest.mark.parametrize("bad", [True, "8", 8.5, 0, 65])
def test_routing_topk_domain_is_checked_without_cuda(bad):
    from lemoq.kernels._routing_triton import routing_softmax_topk

    with pytest.raises(ValueError):
        routing_softmax_topk(torch.empty(0, 64), top_k=bad)


def test_occupancy_partial_coverage_is_not_done(monkeypatch, tmp_path):
    import importlib
    import json

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "mock-device")
    monkeypatch.setattr(op, "_limits", lambda _: dict(op._SM_LIMITS[(12, 0)]))
    monkeypatch.setattr(op, "_warm_kernels", lambda *a: None)
    first, rest = object(), object()
    fake = SimpleNamespace(_rms_norm_kernel=first, _gate_up_act_kernel=rest, _down_sum_kernel=rest)
    actual_import = importlib.import_module
    names = {"lemoq.kernels.rmsnorm", "lemoq.kernels._grouped_triton"}
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a: fake if name in names else actual_import(name, *a),
    )
    monkeypatch.setattr(
        op,
        "_compiled_kernel_info",
        lambda fn, **kw: (
            {"via": "mock", "n_regs": 32, "n_spills": 0, "shared_bytes": 0} if fn is first else None
        ),
    )
    out = tmp_path / "occupancy.json"
    assert op.main(["--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == "planned"
    assert payload["coverage"] == {"measured": 1, "expected": 3}
    assert payload["subjects"]["fused_rmsnorm"]["status"] == "measured_estimate"
