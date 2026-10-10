"""test_report_load — 汇总加载的两条行为。

1. 归档目录（`v1_archive/` 等）不进汇总表（规则 9：归档 = 旧版本，不是规范工件）；
2. 存量 gate == FAIL 工件在读取时 sanitize：实测值挪进 evidence、原位置置 None，
   磁盘文件本体不改写（2026-10-10 评审补的展示口径）。
"""

from __future__ import annotations

import json

from benchmarks.report import load_results
from benchmarks.schema import empty_result, validate_result


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_load_results_skips_archive_dirs(tmp_path):
    live = empty_result("EXP-live", status="planned")
    _write(tmp_path / "EXP-live" / "result.json", live)
    archived = empty_result("EXP-archived", status="planned")
    _write(tmp_path / "v1_archive" / "result.json", archived)

    loaded = load_results(str(tmp_path / "*" / "result.json"))
    paths = [r["_path"] for r in loaded]
    assert len(paths) == 1
    assert "EXP-live" in paths[0]
    assert not any("_archive" in p for p in paths)


def test_gate_failed_payload_is_sanitized_on_load(tmp_path):
    payload = empty_result("EXP-fail", status="done")
    payload["correctness"]["gate"] = "FAIL"
    payload["correctness"]["ppl_note"] = "测试：PPL 未测"
    payload["attribution"]["host_share_pct"] = 61.5
    payload["bottleneck"]["measured_p50_ms"] = 382.97
    disk = tmp_path / "EXP-fail" / "result.json"
    _write(disk, payload)

    loaded = load_results(str(tmp_path / "*" / "result.json"))
    assert len(loaded) == 1
    got = loaded[0]
    assert got["attribution"]["host_share_pct"] is None
    assert got["bottleneck"]["measured_p50_ms"] is None
    assert got["_measurement"]["gate_failed_evidence"]["host_share_pct"] == 61.5
    assert validate_result(got) == []

    # 磁盘文件本体不被改写（历史工件只读）
    on_disk = json.loads(disk.read_text(encoding="utf-8"))
    assert on_disk["attribution"]["host_share_pct"] == 61.5
