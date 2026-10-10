"""test_round_canary — 金丝雀 VOID 验收的机械行为（2026-10-10 评审 M1 的修复）。

要点：
    - 漂移 >5% ⇒ VOID、≤5% ⇒ PASS；
    - **fail-closed**：指标缺失 / head=0 / payload 读不出 一律 VOID（不许静默放行）；
    - --graph：任一端 host_share_pct > 5 ⇒ VOID（共享卡协议的图格自检）；
    - --mark：判 VOID 才标记，先归档再置 status=void；status=void 被 schema 接受、
      不再被 report 计入"已完成"。
"""

from __future__ import annotations

import json

from benchmarks.report import _tristate
from benchmarks.schema import empty_result, validate_result
from tools.round_canary import evaluate, main, mark_void


def _payload(p50=3.0, host_share=1.2, status="done"):
    p = empty_result("EXP-canary", status=status)
    p["latency"]["per_token_p50_ms"] = p50
    p["attribution"]["host_share_pct"] = host_share
    return p


def test_small_drift_passes():
    v = evaluate(_payload(3.0), _payload(3.01))  # +0.333%
    assert v["verdict"] == "PASS"
    assert abs(v["drift_pct"] - 0.3333) < 1e-3
    assert v["reasons"] == []


def test_large_drift_is_void():
    v = evaluate(_payload(3.0), _payload(3.2))  # +6.67%
    assert v["verdict"] == "VOID"
    assert any("漂移" in r for r in v["reasons"])


def test_negative_drift_beyond_threshold_is_void():
    v = evaluate(_payload(3.0), _payload(2.7))  # −10%
    assert v["verdict"] == "VOID"


def test_missing_metric_fails_closed():
    tail = _payload()
    tail["latency"]["per_token_p50_ms"] = None
    v = evaluate(_payload(), tail)
    assert v["verdict"] == "VOID"
    assert any("缺失" in r for r in v["reasons"])


def test_head_zero_fails_closed():
    v = evaluate(_payload(p50=0.0), _payload(p50=0.0))
    assert v["verdict"] == "VOID"
    assert any("head=0" in r for r in v["reasons"])


def test_garbage_payload_fails_closed():
    v = evaluate(None, {"nope": 1})
    assert v["verdict"] == "VOID"


def test_graph_check_flags_contaminated_host_share():
    # drift 本身是 0，但 tail 的 host_share 超限
    v = evaluate(_payload(host_share=1.0), _payload(host_share=51.0), graph=True)
    assert v["verdict"] == "VOID"
    assert any("共租" in r for r in v["reasons"])
    # 不带 --graph 时不查 host_share
    assert evaluate(_payload(host_share=1.0), _payload(host_share=51.0))["verdict"] == "PASS"


def test_graph_check_requires_host_share_present():
    head, tail = _payload(), _payload()
    head["attribution"]["host_share_pct"] = None
    v = evaluate(head, tail, graph=True)
    assert v["verdict"] == "VOID"
    assert any("缺失" in r for r in v["reasons"])


def test_mark_void_archives_then_rewrites(tmp_path):
    path = tmp_path / "canary_result.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    archive = mark_void(path)
    assert archive.exists()
    assert json.loads(archive.read_text(encoding="utf-8"))["status"] == "done"  # 归档是旧内容
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "void"


def test_void_status_is_schema_valid_and_not_reportable(tmp_path):
    payload = _payload(status="void")
    assert validate_result(payload) == []
    assert _tristate(payload) == "作废"


def test_cli_exit_codes_and_out_file(tmp_path):
    head = tmp_path / "head.json"
    tail_good = tmp_path / "tail_good.json"
    tail_bad = tmp_path / "tail_bad.json"
    head.write_text(json.dumps(_payload(3.0)), encoding="utf-8")
    tail_good.write_text(json.dumps(_payload(3.01)), encoding="utf-8")
    tail_bad.write_text(json.dumps(_payload(3.5)), encoding="utf-8")
    out = tmp_path / "verdict.json"

    assert main([str(head), str(tail_good), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "PASS"

    assert main([str(head), str(tail_bad), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "VOID"


def test_cli_mark_only_on_void(tmp_path):
    head = tmp_path / "head.json"
    tail = tmp_path / "tail.json"
    head.write_text(json.dumps(_payload(3.0)), encoding="utf-8")

    # PASS + --mark：不动文件
    tail.write_text(json.dumps(_payload(3.01)), encoding="utf-8")
    assert main([str(head), str(tail), "--mark"]) == 0
    assert json.loads(head.read_text(encoding="utf-8"))["status"] == "done"

    # VOID + --mark：两份都置 void 且有归档
    tail.write_text(json.dumps(_payload(3.5)), encoding="utf-8")
    assert main([str(head), str(tail), "--mark"]) == 1
    assert json.loads(head.read_text(encoding="utf-8"))["status"] == "void"
    assert json.loads(tail.read_text(encoding="utf-8"))["status"] == "void"
    assert list(tmp_path.glob("archive/void_*"))


def test_cli_read_error_exits_void(tmp_path):
    missing = tmp_path / "nope.json"
    tail = tmp_path / "tail.json"
    tail.write_text(json.dumps(_payload()), encoding="utf-8")
    assert main([str(missing), str(tail)]) == 1
