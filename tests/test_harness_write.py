"""test_harness_write — `write_result` 的两条纪律性行为。

1. 写盘前 sanitize：gate == FAIL 的实测性能值必须被挪进
   `_measurement.gate_failed_evidence`（硬规则 2 的写盘执行点）；
2. 覆盖已存在的 result.json 前必须归档（规则 9 的机械执行点，
   2026-10-10 评审前这只靠各轮脚本自觉）。
"""

from __future__ import annotations

import json

from benchmarks.harness import write_result
from benchmarks.schema import empty_result


def _fail_payload(exp_id: str = "EXP-t-write", notes: str | None = None):
    payload = empty_result(exp_id, status="done")
    payload["correctness"]["gate"] = "FAIL"
    payload["correctness"]["ppl_note"] = "测试 payload：PPL 未测，原因见注"
    payload["latency"]["per_token_p50_ms"] = 9.9
    payload["attribution"]["host_share_pct"] = 55.0
    payload["bottleneck"]["measured_p50_ms"] = 12.3
    payload["attainment"]["quant_attainment_pct"] = 0.4
    payload["verdict"]["notes"] = notes
    return payload


def test_write_result_sanitizes_gate_failed_perf(tmp_path):
    path, problems = write_result(_fail_payload(), root=str(tmp_path))
    assert problems == []
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["latency"]["per_token_p50_ms"] is None
    assert saved["attribution"]["host_share_pct"] is None
    assert saved["bottleneck"]["measured_p50_ms"] is None
    assert saved["attainment"]["quant_attainment_pct"] is None
    ev = saved["_measurement"]["gate_failed_evidence"]
    assert ev["per_token_p50_ms"] == 9.9
    assert ev["host_share_pct"] == 55.0
    assert ev["measured_p50_ms"] == 12.3


def test_write_result_archives_before_overwrite(tmp_path):
    write_result(_fail_payload(notes="v1"), root=str(tmp_path))
    archive_dir = tmp_path / "EXP-t-write" / "archive"
    assert not archive_dir.exists()  # 首次写盘：没有旧文件可归档

    write_result(_fail_payload(notes="v2"), root=str(tmp_path))
    archives = sorted(archive_dir.glob("result_*.json"))
    assert len(archives) == 1
    old = json.loads(archives[0].read_text(encoding="utf-8"))
    assert old["verdict"]["notes"] == "v1"  # 归档的是被覆盖前的旧内容

    current = json.loads((tmp_path / "EXP-t-write" / "result.json").read_text(encoding="utf-8"))
    assert current["verdict"]["notes"] == "v2"

    # 内容完全相同再写一次：不产生新归档
    write_result(_fail_payload(notes="v2"), root=str(tmp_path))
    assert len(list(archive_dir.glob("result_*.json"))) == 1
