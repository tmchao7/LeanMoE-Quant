"""Budget, provenance, freeze/evidence and CLI safety checks, entirely offline."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tools import w4_calibrate as wc
from tools.prepare_w4_data import prepare


def _caches(root):
    meta = {
        "status": "complete",
        "dataset": "wikitext-2-raw-v1",
        "revision": "a" * 40,
        "tokenizer_sha256": "b" * 64,
        "splits": {},
    }
    for i, split in enumerate(("train", "validation", "test")):
        ids = torch.arange(128, dtype=torch.long) + i * 1000
        path = root / (split + ".pt")
        torch.save(ids, path)
        meta["splits"][split] = {
            "file_sha256": wc.file_sha256(path),
            "token_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        }
    return meta


def test_plan_and_default_dry_runs_never_access_gpu_or_write(monkeypatch, tmp_path, capsys):
    def bomb(*a, **kw):
        raise AssertionError("dry run attempted execution")

    monkeypatch.setattr(wc, "_calibrate", bomb)
    monkeypatch.setattr(wc.subprocess, "check_output", bomb)
    out = tmp_path / "unused"
    assert wc.main(["--phase", "calibrate", "--out", str(out)]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["status"] == "planned" and not out.exists()
    assert p["resource_plan"]["gpu_seconds"] is None


@pytest.mark.parametrize("budget", [None, "0", "-1", "nan", "inf"])
def test_gpu_execution_requires_explicit_finite_budget(tmp_path, budget):
    args = [
        "--phase",
        "calibrate",
        "--execute",
        "--model",
        "m",
        "--data-manifest",
        "d",
        "--train-cache",
        "t",
        "--validation-cache",
        "v",
        "--out",
        str(tmp_path / "out"),
    ]
    if budget is not None:
        args += ["--gpu-budget-seconds", budget]
    with pytest.raises(SystemExit):
        wc.main(args)
    assert not (tmp_path / "out").exists()


def test_occupied_gpu_stops_without_loading_models(monkeypatch, tmp_path):
    monkeypatch.setattr(wc.subprocess, "check_output", lambda *a, **k: "1234, 4096 MiB")
    out = tmp_path / "run"
    assert (
        wc.main(
            [
                "--phase",
                "calibrate",
                "--execute",
                "--model",
                "missing",
                "--data-manifest",
                "d",
                "--train-cache",
                "t",
                "--validation-cache",
                "v",
                "--out",
                str(out),
                "--gpu-budget-seconds",
                "30",
            ]
        )
        == 1
    )
    p = json.loads((out / "manifest.json").read_text())
    assert p["status"] == "failed" and "occupied" in p["error"]
    with pytest.raises(SystemExit):
        wc.main(
            [
                "--phase",
                "calibrate",
                "--execute",
                "--model",
                "m",
                "--data-manifest",
                "d",
                "--train-cache",
                "t",
                "--validation-cache",
                "v",
                "--out",
                str(out),
                "--gpu-budget-seconds",
                "30",
            ]
        )


def test_windows_are_deterministic_nonoverlapping_and_split_scoped(tmp_path):
    meta = _caches(tmp_path)
    kwargs = {"split": "train", "tokenizer": "b" * 64, "count": 4, "length": 8}
    a, am = wc.cached_windows(tmp_path / "train.pt", meta, **kwargs)
    b, bm = wc.cached_windows(tmp_path / "train.pt", meta, **kwargs)
    assert am == bm and len(set(am["offsets"])) == 4
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="hash"):
        wc.cached_windows(tmp_path / "test.pt", meta, **kwargs)


@pytest.mark.parametrize(
    "case", ["revision", "tokenizer", "reuse", "token_reuse", "token_hash", "short", "incomplete"]
)
def test_provenance_errors_fail_closed(tmp_path, case):
    meta = _caches(tmp_path)
    count = 4
    if case == "revision":
        meta["revision"] = "main"
    elif case == "tokenizer":
        meta["tokenizer_sha256"] = "other"
    elif case == "reuse":
        meta["splits"]["validation"]["file_sha256"] = meta["splits"]["train"]["file_sha256"]
    elif case == "token_reuse":
        meta["splits"]["test"]["token_sha256"] = meta["splits"]["train"]["token_sha256"]
    elif case == "token_hash":
        meta["splits"]["train"]["token_sha256"] = "c" * 64
    elif case == "short":
        count = 100
    elif case == "incomplete":
        meta["status"] = "incomplete"
    with pytest.raises(ValueError):
        wc.cached_windows(
            tmp_path / "train.pt", meta, split="train", tokenizer="b" * 64, count=count, length=8
        )


@pytest.mark.parametrize(
    "kl,top,ppl,passed",
    [
        (0.049, 0.95, 0.009, True),
        (0.05, 1, 0, False),
        (0.001, 0.94, 0, False),
        (0.001, 1, 0.01, False),
        (float("nan"), 1, 0, False),
        (0.001, 1, -1, False),
    ],
)
def test_mixed_expert_gate_uses_frozen_thresholds(kl, top, ppl, passed):
    report = wc.accuracy_gate({"kl": kl, "top1_agreement": top}, ppl, expert_bits=4)
    assert report["passed"] is passed
    assert report["precision_scope"] == "experts4_attention8_activation16"


def test_test_split_requires_successful_exact_validation_freeze(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    manifest = {
        "status": "complete",
        "model_id": "m",
        "expert_bits": 4,
        "precision_plan": "experts4_attention8_activation16",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    report = {
        "status": "validated",
        "eval_split": "validation",
        "gate": {"passed": True},
        "model_id": "m",
        "expert_bits": 4,
        "precision_plan": "experts4_attention8_activation16",
        "bundle_sha256": wc.file_sha256(bundle / "manifest.json"),
    }
    evidence = tmp_path / "v.json"
    evidence.write_text(json.dumps(report))
    wc._check_freeze(evidence, bundle, manifest)
    (bundle / "manifest.json").write_text(json.dumps({**manifest, "modified": True}))
    with pytest.raises(ValueError, match="exact bundle"):
        wc._check_freeze(evidence, bundle, manifest)


def test_budget_expiration_is_a_hard_stop(monkeypatch):
    monkeypatch.setattr(wc.time, "monotonic", lambda: 100.0)
    budget = wc.Budget(10)
    budget.check()
    monkeypatch.setattr(wc.time, "monotonic", lambda: 110.0)
    with pytest.raises(TimeoutError):
        budget.check()


def test_cpu_preparation_writes_verifiable_cache_and_refuses_overwrite(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, split in enumerate(("train", "validation", "test")):
        (raw / (split + ".txt")).write_text(f"{i}\n\nA different natural text split {i}.")

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return SimpleNamespace(
                input_ids=torch.tensor([[ord(c) for c in text]], dtype=torch.long)
            )

    out = tmp_path / "cache"
    meta = prepare(raw, out, tokenizer=Tokenizer(), tokenizer_hash="b" * 64, revision="a" * 40)
    assert meta["status"] == "complete"
    windows, _ = wc.cached_windows(
        out / "train.pt", meta, split="train", tokenizer="b" * 64, count=2, length=8
    )
    assert len(windows) == 2
    with pytest.raises(FileExistsError):
        prepare(raw, out, tokenizer=Tokenizer(), tokenizer_hash="b" * 64, revision="a" * 40)


def test_4096_requires_1024_evidence_for_same_bundle_and_split(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    report = {
        "status": "validated",
        "gate": {"passed": True},
        "eval_split": "validation",
        "config": {"ctx": 1024},
        "bundle_sha256": wc.file_sha256(bundle / "manifest.json"),
    }
    evidence = tmp_path / "gate.json"
    evidence.write_text(json.dumps(report))
    wc.check_ctx1024(evidence, bundle, "validation")
    with pytest.raises(ValueError):
        wc.check_ctx1024(evidence, bundle, "test")


def test_profile_only_offers_one_unmeasured_followup_after_clean_canary():
    p = {
        "wall_p50_ms": 2.0,
        "host_share_pct": 2.0,
        "top_kernels": [{"name": "_dequant_gemm_kernel", "self_ms_per_token": 0.4}],
    }
    result = wc.profile_opportunity(p, dict(p, wall_p50_ms=2.02))
    assert result["candidate"]["name"] == "qkv_projection_fusion"
    assert result["candidate"]["status"] == "to_be_evaluated"
    assert result["candidate"]["removed_bucket_ceiling_ms"] == pytest.approx(0.3)
    assert wc.profile_opportunity(p, dict(p, host_share_pct=5.1))["status"] == "invalid"
    assert wc.profile_opportunity(p, dict(p, wall_p50_ms=2.2))["status"] == "invalid"
