"""test_schema — `result.json` 的形状校验。

这里最重要的是两条**纪律性**断言：

    1. **`None` = 未测**，必须被接受；`0` / `False` = 测到的是 0/否，是结论。
       校验器若把未测当成缺字段，就会逼着人编数字。
    2. **正确性门槛不过，不得报性能数字**（项目硬规则 2）。
"""

from __future__ import annotations

import pytest

from benchmarks.schema import (
    PPL_NOT_MEASURED_NOTE,
    REQUIRED_TOP,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    empty_result,
    load_ppl_index,
    validate_ppl_declaration,
    validate_result,
)


def test_empty_result_is_valid():
    payload = empty_result("EXP-test-01")
    assert validate_result(payload) == []


def test_all_template_sections_present():
    """字段与 experiments/TEMPLATE.md 一一对应；缺 section = 这个结果不存在。"""
    payload = empty_result("EXP-test-01")
    for key in REQUIRED_TOP:
        assert key in payload
    assert set(RESULT_SCHEMA).issubset(set(payload))


def test_none_is_accepted_as_not_measured():
    payload = empty_result("EXP-test-01")
    payload["latency"]["per_token_p50_ms"] = None
    assert validate_result(payload) == []


def test_zero_is_valid_and_distinct_from_none():
    payload = empty_result("EXP-test-01")
    payload["latency"]["per_token_p50_ms"] = 0.0
    assert validate_result(payload) == []
    assert payload["latency"]["per_token_p50_ms"] is not None


def test_missing_top_level_field_is_reported():
    payload = empty_result("EXP-test-01")
    del payload["attribution"]
    problems = validate_result(payload)
    assert any("attribution" in p for p in problems)


def test_missing_nested_field_is_reported():
    payload = empty_result("EXP-test-01")
    del payload["latency"]["per_token_p99_ms"]
    problems = validate_result(payload)
    assert any("per_token_p99_ms" in p for p in problems)


def test_wrong_type_is_reported():
    payload = empty_result("EXP-test-01")
    payload["config"]["ctx"] = "4096"  # 应为 int
    problems = validate_result(payload)
    assert any("config.ctx" in p for p in problems)


def test_bad_status_is_reported():
    payload = empty_result("EXP-test-01")
    payload["status"] = "finished"
    problems = validate_result(payload)
    assert any("status" in p for p in problems)


def test_bad_verdict_is_reported():
    payload = empty_result("EXP-test-01")
    payload["verdict"]["verdict"] = "GOOD"
    problems = validate_result(payload)
    assert any("verdict" in p for p in problems)


def test_failed_gate_must_not_report_performance():
    """项目硬规则 2：KL/top-1 不过门槛，就不许出现性能数字。"""
    payload = empty_result("EXP-test-01")
    payload["correctness"]["gate"] = "FAIL"
    payload["latency"]["per_token_p50_ms"] = 1.23
    problems = validate_result(payload)
    assert any("硬规则 2" in p for p in problems)


def test_failed_gate_with_no_performance_is_fine():
    payload = empty_result("EXP-test-01")
    payload["correctness"]["gate"] = "FAIL"
    assert validate_result(payload) == []


def test_failed_gate_rejects_perf_in_every_section():
    """硬规则 2 的全字段清单（2026-10-10 评审补洞）：attribution / bottleneck /
    attainment / ttft / samples 里的实测值在 gate FAIL 时同样不许存在。"""
    from benchmarks.schema import GATE_FAIL_MUST_BE_NULL

    cases = [
        ("attribution", "host_share_pct", 61.5),
        ("attribution", "kernels_per_call", 43221.0),
        ("bottleneck", "measured_p50_ms", 382.97),
        ("bottleneck", "gpu_total_ms_per_step", 147.0),
        ("attainment", "quant_attainment_pct", 0.1),
        ("latency", "ttft_p50_ms", 10.0),
        ("latency", "samples", {"step": [1.0]}),
    ]
    for section, field, value in cases:
        assert (section, field) in [
            (s, f) for s, fs in GATE_FAIL_MUST_BE_NULL.items() for f in fs
        ], (section, field)
        payload = empty_result("EXP-test-01")
        payload["correctness"]["gate"] = "FAIL"
        payload[section][field] = value
        problems = validate_result(payload)
        assert any("硬规则 2" in p and field in p for p in problems), (section, field, problems)


def test_sanitize_moves_gate_failed_perf_into_evidence():
    from benchmarks.schema import sanitize_gate_failed_perf

    payload = empty_result("EXP-test-01")
    payload["correctness"]["gate"] = "FAIL"
    payload["latency"]["per_token_p50_ms"] = 9.9
    payload["attribution"]["host_share_pct"] = 55.0
    payload["bottleneck"]["measured_p50_ms"] = 12.3
    payload["attainment"]["bw_attainment_pct"] = 0.2

    moved = sanitize_gate_failed_perf(payload)
    assert moved == {
        "per_token_p50_ms": 9.9,
        "host_share_pct": 55.0,
        "measured_p50_ms": 12.3,
        "bw_attainment_pct": 0.2,
    }
    assert payload["latency"]["per_token_p50_ms"] is None
    assert payload["attribution"]["host_share_pct"] is None
    ev = payload["_measurement"]["gate_failed_evidence"]
    assert ev["per_token_p50_ms"] == 9.9
    assert ev["host_share_pct"] == 55.0
    assert validate_result(payload) == []


def test_sanitize_is_noop_for_passing_gate_and_keeps_existing_evidence():
    from benchmarks.schema import sanitize_gate_failed_perf

    # PASS 格：一动不动
    ok = empty_result("EXP-test-02")
    ok["correctness"]["gate"] = "PASS"
    ok["attribution"]["host_share_pct"] = 12.0
    assert sanitize_gate_failed_perf(ok) == {}
    assert ok["attribution"]["host_share_pct"] == 12.0

    # runner 已经暂存过的 latency 字段：不覆盖 evidence 里的原值
    payload = empty_result("EXP-test-03")
    payload["correctness"]["gate"] = "FAIL"
    payload["latency"]["per_token_p50_ms"] = 9.9
    payload["_measurement"] = {
        "gate_failed_evidence": {"per_token_p50_ms": 1.0, "note": "runner 已存"}
    }
    sanitize_gate_failed_perf(payload)
    ev = payload["_measurement"]["gate_failed_evidence"]
    assert ev["per_token_p50_ms"] == 1.0  # setdefault：不覆盖
    assert ev["note"] == "runner 已存"


def test_schema_version_is_enforced():
    payload = empty_result("EXP-test-01")
    payload["schema_version"] = SCHEMA_VERSION + 1
    problems = validate_result(payload)
    assert any("schema_version" in p for p in problems)


def test_non_dict_payload_rejected():
    assert validate_result(["not", "a", "dict"])


def test_empty_exp_id_rejected():
    payload = empty_result("")
    problems = validate_result(payload)
    assert any("exp_id" in p for p in problems)


@pytest.mark.parametrize("section", sorted(RESULT_SCHEMA))
def test_each_section_can_be_flagged_when_none(section):
    payload = empty_result("EXP-test-01")
    payload[section] = None
    problems = validate_result(payload)
    assert any(section in p for p in problems)


# ---------------------------------------------------------------------------
# PPL 声明：`ppl_*` 全 null 时必须有一句"为什么是 null"
# ---------------------------------------------------------------------------
def test_empty_result_carries_a_ppl_note_slot():
    """新 payload 必须**有** ppl_note 这个键（值可以是 None）—— 但它是条件性字段，
    刻意不在 RESULT_SCHEMA 的必需表里（否则 69 份历史工件会全判"缺字段"）。"""
    payload = empty_result("EXP-test-01")
    assert "ppl_note" in payload["correctness"]
    assert "ppl_note" not in RESULT_SCHEMA["correctness"]


def test_planned_skeleton_needs_no_ppl_declaration():
    """`planned` 骨架整份都是 null —— 它的声明就是它自己的 status。

    否则 `harness --dry-run` 产出的骨架会全部不合法（那时"没跑"还要另外写一句
    "为什么没跑"，纯噪音）。
    """
    payload = empty_result("EXP-ppl-planned-01", status="planned")
    assert validate_result(payload) == []
    # 一旦它自称 done，就必须有声明
    payload["status"] = "done"
    payload["verdict"] = {"verdict": "PASS", "outcome": "supported", "notes": "x"}
    assert any("ppl" in p for p in validate_result(payload))


def test_ppl_null_without_a_declaration_is_reported():
    """三个 ppl_* 全 null 且既没有 ppl_note 也没登记在清单里 ⇒ 必须报错。"""
    payload = empty_result("EXP-ppl-none-01")
    payload["status"] = "done"
    payload["verdict"] = {"verdict": "PASS", "outcome": "supported", "notes": "x"}
    assert all(payload["correctness"][k] is None for k in ("ppl_fp16", "ppl_this", "ppl_delta"))
    problems = validate_result(payload)
    assert any("ppl" in p for p in problems), problems


def test_ppl_null_with_a_note_is_accepted():
    payload = empty_result("EXP-ppl-note-01", status="done")
    payload["correctness"]["ppl_note"] = PPL_NOT_MEASURED_NOTE
    assert validate_result(payload) == []


def test_ppl_null_declared_in_the_index_is_accepted():
    """历史工件没有 ppl_note，靠 `results/PPL_STATUS.json` 合规（不必重写）。"""
    index = {
        "EXP-ppl-index-01": {
            "status": "not_measured",
            "reason": "PPL 未测量（待办 G5）",
        }
    }
    payload = empty_result("EXP-ppl-index-01", status="done")
    assert validate_result(payload, index) == []


def test_ppl_index_entry_without_a_reason_does_not_count():
    """清单里的记录必须**同时**有 status 与 reason —— 空壳不算声明。"""
    payload = empty_result("EXP-ppl-index-02", status="done")
    assert validate_ppl_declaration(payload, {"EXP-ppl-index-02": {"status": "not_measured"}})
    assert (
        validate_ppl_declaration(
            payload, {"EXP-ppl-index-02": {"status": "not_measured", "reason": "待 G5"}}
        )
        == []
    )


def test_ppl_with_a_value_needs_no_declaration():
    payload = empty_result("EXP-ppl-ok-01", status="done")
    payload["correctness"]["ppl_fp16"] = 9.1994
    payload["correctness"]["ppl_this"] = 9.2033
    payload["correctness"]["ppl_delta"] = 0.0004
    assert validate_result(payload) == []


def test_ppl_status_index_covers_the_recorded_cells():
    """真工件：`tools/ppl_status.py` 生成的清单必须覆盖所有带 exp_id 的格子。

    这条把"清单存在"钉住 —— 清单没了，69 份历史工件会立刻回到"无声明的 null"。
    """
    index = load_ppl_index()
    if not index:
        pytest.skip("results/PPL_STATUS.json 不在本机（results/ 是 gitignored）")
    assert index, "清单为空"
    bad = [
        k
        for k, v in index.items()
        if v.get("status") != "measured" and not str(v.get("reason") or "").strip()
    ]
    assert not bad, "这些格子未测却没有原因：%s" % bad[:5]
