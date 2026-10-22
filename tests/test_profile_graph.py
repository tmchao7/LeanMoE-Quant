"""test_profile_graph — 定位工具的收数/落盘/失败路径（纯 CPU，mock run_cudagraph）。"""

from __future__ import annotations

import json
import os

import pytest

from tools import profile_graph as pg


def _fake_res():
    return {
        "gap": {
            "wall_p50_ms": 3.01,
            "gpu_busy_ms": 2.95,
            "host_share_pct": 2.0,
            "kernels_per_call": 540.0,
            "top_kernels": [
                {
                    "name": "_down_sum_kernel_q",
                    "self_ms_per_token": 0.80,
                    "occurrences_per_token": 16.0,
                    "mean_us_per_occurrence": 50.0,
                    "share_pct": 27.12,
                },
                {
                    "name": "_rms_norm_kernel",
                    "self_ms_per_token": 0.21,
                    "occurrences_per_token": 65.0,
                    "mean_us_per_occurrence": 3.2,
                    "share_pct": 7.12,
                },
            ],
        }
    }


def test_measured_path_writes_json_and_prints(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(pg, "run_cudagraph", lambda **kw: _fake_res())
    out = tmp_path / "profile.json"
    rc = pg.main(["--model", "dummy/model", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["config"]["bit_width"] == "w8a16"
    assert payload["gap"]["top_kernels"][0]["name"] == "_down_sum_kernel_q"
    assert "定位用 kernel 分解" in payload["note"]
    text = capsys.readouterr().out
    assert "_down_sum_kernel_q" in text and "27.12" in text
    # tail 行：540 - (16+65) = 459 个未列出的 occurrence
    assert "tail of ~459" in text


def test_failure_path_writes_error_and_exits_1(monkeypatch, tmp_path):
    def boom(**kw):
        raise RuntimeError("capture exploded")

    monkeypatch.setattr(pg, "run_cudagraph", boom)
    out = tmp_path / "profile.json"
    rc = pg.main(["--model", "dummy/model", "--json", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "capture exploded" in payload["error"]


def test_build_payload_marks_missing_table_incomplete():
    payload = pg.build_payload({"gap": {}}, {"ctx": 1024})
    assert payload["status"] == "incomplete"


def test_ab_routing_verdict(monkeypatch, capsys, tmp_path):
    """A/B：金丝雀 ≤5% + token 逐位相同 ⇒ measured；任何一条破 ⇒ invalid。"""
    calls = {"n": 0}

    def fake_invoke(args):
        calls["n"] += 1
        i = calls["n"] - 1
        arm_fused = i % 2 == 1
        wall = 1.90 if arm_fused else 2.00  # fused 快 5%
        if i == 2:  # 第二次 aten（金丝雀尾）
            wall = 2.02  # +1% 漂移，≤5%
        return {
            "gap": {"wall_p50_ms": wall, "gpu_busy_ms": wall - 0.05, "host_share_pct": 3.0},
            "correctness": {"graph_output_tokens": [11, 12, 13], "graph_seed_token": 11},
        }

    monkeypatch.setattr(pg, "_invoke", fake_invoke)
    out = tmp_path / "ab.json"
    rc = pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-fused-route", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["tokens_equal"] is True
    assert abs(payload["canary_aten_drift_pct"]) <= 5.0
    assert payload["fused_faster_pct"] > 0
    assert len(payload["runs"]) == 5

    # token 漂移 ⇒ invalid + rc1
    calls["n"] = 0

    def fake_invoke_bad(args):
        res = fake_invoke(args)
        if calls["n"] == 4:
            res = dict(res, correctness=dict(res["correctness"], graph_output_tokens=[99]))
        return res

    monkeypatch.setattr(pg, "_invoke", fake_invoke_bad)
    out = tmp_path / "ab_bad.json"
    rc = pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-fused-route", "--json", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "invalid"
    assert any("token" in r for r in payload["reasons"])


def test_ab_mask_fill_verdict_and_env_restored(monkeypatch, tmp_path):
    """--ab-mask-fill：off/on 交替、env 用后还原、键名与验收齐全。"""
    calls = {"n": 0}
    seen_env: list[str | None] = []

    def fake_invoke(args):
        calls["n"] += 1
        i = calls["n"] - 1
        seen_env.append(os.environ.get("LEMOQ_SKIP_MASK_FILL"))
        arm_on = i % 2 == 1
        wall = 1.94 if arm_on else 1.98  # on 臂快 ~2%
        return {
            "gap": {"wall_p50_ms": wall, "gpu_busy_ms": wall - 0.04, "host_share_pct": 2.0},
            "correctness": {"graph_output_tokens": [21, 22, 23], "graph_seed_token": 21},
        }

    monkeypatch.setattr(pg, "_invoke", fake_invoke)
    monkeypatch.delenv("LEMOQ_SKIP_MASK_FILL", raising=False)
    out = tmp_path / "ab_mask.json"
    rc = pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-mask-fill", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["generated_by"].endswith("--ab-mask-fill")
    assert payload["env_var"] == "LEMOQ_SKIP_MASK_FILL"
    assert payload["tokens_equal"] is True
    assert abs(payload["canary_drift_pct"]) <= 5.0
    assert payload["treatment_faster_pct"] > 0
    assert len(payload["runs"]) == 5
    assert [r["arm"] for r in payload["runs"]] == ["off", "on", "off", "on", "off"]
    # 臂间 env 正确交替，且结束后还原（原环境没有这个变量 ⇒ 被删除）
    assert seen_env == ["0", "1", "0", "1", "0"]
    assert "LEMOQ_SKIP_MASK_FILL" not in os.environ


def test_ab_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        pg.main(
            [
                "--model",
                "m",
                "--ab-fused-route",
                "--ab-mask-fill",
                "--json",
                str(tmp_path / "x.json"),
            ]
        )
    assert excinfo.value.code == 2  # argparse.error


# ---------------------------------------------------------------------------
# review 2026-10-22（P1/P2/P3/P9）：A/B 判定机器的观察字段与 fail-closed
# ---------------------------------------------------------------------------


def _ab_run(wall, busy, tokens, host=2.0):
    return {
        "gap": {"wall_p50_ms": wall, "gpu_busy_ms": busy, "host_share_pct": host},
        "correctness": {"graph_output_tokens": tokens, "graph_seed_token": tokens[0]},
    }


def _ab_invoke_seq(seq):
    it = iter([*seq, seq[0]])  # final same-arm canary, excluded from arm statistics

    def fake(args):
        return next(it)

    return fake


def test_ab_warns_on_host_share_contamination_and_degenerate_tokens(monkeypatch, tmp_path):
    """P1: host_share>5% 的 run 要进 host_share_warnings；全同 token 要标 degenerate_trajectory。"""
    tok = [7, 7, 7, 7]
    monkeypatch.setattr(
        pg,
        "_invoke",
        _ab_invoke_seq(
            [
                _ab_run(2.00, 1.90, tok, host=2.0),
                _ab_run(1.90, 1.80, tok, host=6.3),  # 污染 run
                _ab_run(2.01, 1.91, tok, host=2.1),
                _ab_run(1.91, 1.81, tok, host=2.2),
            ]
        ),
    )
    out = tmp_path / "ab.json"
    rc = pg.main(["--model", "m", "--max-new-tokens", "4", "--ab-fused-route", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["status"] == "invalid"  # enforce the existing graph protocol
    warn_runs = payload["host_share_warnings"]
    assert warn_runs and warn_runs[0]["i"] == 1 and warn_runs[0]["host_share_pct"] == 6.3
    assert payload["degenerate_trajectory"] is True


def test_ab_reports_busy_drift_for_graph_arms(monkeypatch, tmp_path):
    """P1: 基线臂首尾 gpu_busy 漂移要有 busy_drift_pct / busy_stable 字段。"""
    tok = [5, 6, 7, 8]
    monkeypatch.setattr(
        pg,
        "_invoke",
        _ab_invoke_seq(
            [
                _ab_run(2.00, 1.80, tok),
                _ab_run(1.90, 1.75, tok),
                _ab_run(2.01, 2.00, tok),  # paired baseline
                _ab_run(1.91, 1.76, tok),
            ]
        ),
    )
    out = tmp_path / "ab.json"
    pg.main(["--model", "m", "--max-new-tokens", "4", "--ab-fused-route", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["busy_drift_pct"] == 0.0
    assert payload["busy_stable"] is True


def test_ab_missing_wall_fails_closed_not_crash(monkeypatch, tmp_path):
    """P2: 任一 run 的 wall/busy 为 None ⇒ invalid + 原因，而不是 sum() 崩溃。"""
    tok = [1, 2, 3]
    monkeypatch.setattr(
        pg,
        "_invoke",
        _ab_invoke_seq(
            [
                _ab_run(2.00, 1.90, tok),
                _ab_run(None, 1.80, tok),  # gap 不完整
                _ab_run(2.01, 1.91, tok),
                _ab_run(1.91, 1.81, tok),
            ]
        ),
    )
    out = tmp_path / "ab.json"
    rc = pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-fused-route", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["status"] == "invalid"
    assert any("缺失" in r for r in payload["reasons"])
    assert payload["mean_wall_baseline_ms"] is None or payload["mean_wall_treatment_ms"] is None


def test_ab_reports_median_and_stat_label(monkeypatch, tmp_path):
    """P3: 口径声明 stat + median_* 字段（mean 保留）。"""
    tok = [1, 2, 3]
    monkeypatch.setattr(
        pg,
        "_invoke",
        _ab_invoke_seq(
            [
                _ab_run(2.00, 1.90, tok),
                _ab_run(1.90, 1.80, tok),
                _ab_run(2.02, 1.92, tok),
                _ab_run(1.92, 1.82, tok),
            ]
        ),
    )
    out = tmp_path / "ab.json"
    pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-fused-route", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["stat"] == "mean_of_run_p50s"
    assert payload["median_wall_baseline_ms"] == pytest.approx(2.01, abs=1e-6)
    assert payload["median_wall_treatment_ms"] == pytest.approx(1.91, abs=1e-6)


def test_ab_routing_carries_both_key_families(monkeypatch, tmp_path):
    """P9: routing 轮 payload 同时带历史键（fused_*）与通用键（treatment_*）。"""
    tok = [1, 2, 3]
    monkeypatch.setattr(
        pg,
        "_invoke",
        _ab_invoke_seq(
            [
                _ab_run(2.00, 1.90, tok),
                _ab_run(1.90, 1.80, tok),
                _ab_run(2.01, 1.91, tok),
                _ab_run(1.91, 1.81, tok),
            ]
        ),
    )
    out = tmp_path / "ab.json"
    pg.main(["--model", "m", "--max-new-tokens", "3", "--ab-fused-route", "--json", str(out)])
    payload = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "fused_faster_pct",
        "treatment_faster_pct",
        "canary_aten_drift_pct",
        "canary_drift_pct",
        "mean_wall_aten_ms",
        "mean_wall_baseline_ms",
    ):
        assert key in payload, key
