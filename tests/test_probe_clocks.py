"""test_probe_clocks — R2 时钟/温度快照的解析口径（tools/probe_machine.py）。

钉住的口径（fail-closed）：
    - 正常 CSV 行解析出四个字段；
    - 列数不足 / 空输出 ⇒ 字段全 None + note，**绝不编造数值**；
    - throttle 的全零掩码（0x0000000000000000）归一成 "None"（可读）；
    - query 失败（probe_clocks 层）⇒ error + 字段全 None。
"""

from __future__ import annotations

from tools.probe_machine import parse_clocks_csv, probe_clocks


def test_parse_normal_row():
    out = parse_clocks_csv("2550 MHz, 2380 MHz, 61, 0x0000000000000000")
    assert out == {"sm_mhz": 2550, "mem_mhz": 2380, "temp_gpu_c": 61, "throttle": "None"}


def test_parse_active_throttle_reason_is_kept_verbatim():
    out = parse_clocks_csv("2100 MHz, 2380 MHz, 74, 0x0000000000000004")
    assert out["sm_mhz"] == 2100
    assert out["temp_gpu_c"] == 74
    assert out["throttle"] == "0x0000000000000004"


def test_parse_missing_columns_returns_all_none_with_note():
    out = parse_clocks_csv("2550 MHz, 2380 MHz")
    assert out["sm_mhz"] is None and out["mem_mhz"] is None
    assert out["temp_gpu_c"] is None and out["throttle"] is None
    assert "note" in out


def test_parse_empty_output_returns_all_none():
    out = parse_clocks_csv("")
    assert out["sm_mhz"] is None
    assert "note" in out


def test_parse_nonnumeric_fields_do_not_crash():
    out = parse_clocks_csv("N/A, N/A, N/A, N/A")
    assert out["sm_mhz"] is None and out["temp_gpu_c"] is None
    # N/A 与全零掩码一致地归一成可读的 "None"（缺失不编造）
    assert out["throttle"] == "None"


def test_probe_clocks_query_failure_is_fail_closed(monkeypatch):
    import tools.probe_machine as pm

    monkeypatch.setattr(
        pm,
        "_safe",
        lambda cmd: {"ok": False, "stdout": "", "stderr": "boom"},
    )
    out = probe_clocks()
    assert out["query_ok"] is False
    assert out["error"] == "boom"
    assert out["sm_mhz"] is None and out["throttle"] is None


def test_probe_clocks_success_parses(monkeypatch):
    import tools.probe_machine as pm

    monkeypatch.setattr(
        pm,
        "_safe",
        lambda cmd: {
            "ok": True,
            "stdout": "2400 MHz, 2380 MHz, 55, 0x0000000000000000",
            "stderr": "",
        },
    )
    out = probe_clocks()
    assert out["query_ok"] is True
    assert out["sm_mhz"] == 2400
    assert out["throttle"] == "None"
