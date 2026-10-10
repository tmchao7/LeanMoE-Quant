"""test_profile_graph — 定位工具的收数/落盘/失败路径（纯 CPU，mock run_cudagraph）。"""

from __future__ import annotations

import json

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
    rc = pg.main(["--model", "m", "--ab-fused-route", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "measured"
    assert payload["tokens_equal"] is True
    assert abs(payload["canary_aten_drift_pct"]) <= 5.0
    assert payload["fused_faster_pct"] > 0
    assert len(payload["runs"]) == 4

    # token 漂移 ⇒ invalid + rc1
    calls["n"] = 0

    def fake_invoke_bad(args):
        res = fake_invoke(args)
        if calls["n"] == 4:
            res = dict(res, correctness=dict(res["correctness"], graph_output_tokens=[99]))
        return res

    monkeypatch.setattr(pg, "_invoke", fake_invoke_bad)
    rc = pg.main(["--model", "m", "--ab-fused-route", "--json", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "invalid"
    assert any("token" in r for r in payload["reasons"])
