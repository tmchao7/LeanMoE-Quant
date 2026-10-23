"""The four decode-path candidate switches: defaults, env override, payload recording.

The point of this module is that a cell can never silently change implementation:
the default lives in code, the env is an explicit override, and the payload records
the effective state.
"""

from __future__ import annotations

import pytest
from conftest import requires_torch

pytestmark = requires_torch

from lemoq.kernels import fusion_switches as fs  # noqa: E402


def test_empty_env_returns_the_code_defaults():
    # 显式空 env 不能改变默认（采纳时只改 FUSION_DEFAULTS 一处）
    assert fs.fusion_states(env={}) == fs.FUSION_DEFAULTS
    assert set(fs.fusion_states(env={})) == set(fs.ENV_VARS)


def test_env_override_is_per_switch(monkeypatch):
    monkeypatch.delenv("LEMOQ_FUSED_ROPE", raising=False)
    monkeypatch.delenv("LEMOQ_FUSED_ROUTE", raising=False)
    monkeypatch.delenv("LEMOQ_SKIP_MASK_FILL", raising=False)
    monkeypatch.delenv("LEMOQ_PACKED_QKV", raising=False)
    assert fs.fusion_enabled("packed_qkv") is False
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "1")
    assert fs.fusion_enabled("packed_qkv") is True
    assert fs.fusion_enabled("fused_rope") is False
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "0")
    assert fs.fusion_enabled("packed_qkv") is False
    # 非 "1" 一律按关处理（不静默接受 "true"/"yes"）
    monkeypatch.setenv("LEMOQ_FUSED_ROPE", "true")
    assert fs.fusion_enabled("fused_rope") is False


def test_payload_fields_match_effective_states():
    env = {"LEMOQ_FUSED_ROUTE": "1"}
    assert fs.fusion_payload_fields(env) == fs.fusion_states(env)
    assert fs.fusion_payload_fields(env)["fused_route"] is True
    assert fs.fusion_payload_fields(env)["fused_rope"] is False


def test_unknown_switch_is_an_error():
    with pytest.raises(KeyError, match="未知融合开关"):
        fs.fusion_enabled("nope")


def test_every_default_has_an_env_var():
    assert set(fs.ENV_VARS) == set(fs.FUSION_DEFAULTS)
    assert all(isinstance(v, bool) for v in fs.FUSION_DEFAULTS.values())
