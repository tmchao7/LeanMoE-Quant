"""test_schema — `result.json` 的形状校验。

这里最重要的是两条**纪律性**断言：

    1. **`None` = 未测**，必须被接受；`0` / `False` = 测到的是 0/否，是结论。
       校验器若把未测当成缺字段，就会逼着人编数字。
    2. **正确性门槛不过，不得报性能数字**（项目硬规则 2）。
"""

from __future__ import annotations

import pytest

from benchmarks.schema import (
    REQUIRED_TOP,
    RESULT_SCHEMA,
    SCHEMA_VERSION,
    empty_result,
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
