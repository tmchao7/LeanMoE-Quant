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
