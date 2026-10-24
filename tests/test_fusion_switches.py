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
    # ★ 默认值在采纳那天会翻成 True（2026-10-24 复核）。这里断言"没有 env 时等于默认"，
    #   不写死 False —— 否则翻默认会红，而翻默认本身不该需要改测试。
    assert fs.fusion_enabled("packed_qkv") is fs.FUSION_DEFAULTS["packed_qkv"]
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "1")
    assert fs.fusion_enabled("packed_qkv") is True
    assert fs.fusion_enabled("fused_rope") is fs.FUSION_DEFAULTS["fused_rope"]
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "0")
    assert fs.fusion_enabled("packed_qkv") is False
    # 非 "1" 一律按关处理（不静默接受 "true"/"yes"）—— 显式关优先于代码默认
    monkeypatch.setenv("LEMOQ_FUSED_ROPE", "true")
    assert fs.fusion_enabled("fused_rope") is False


def test_payload_fields_match_effective_states():
    env = {"LEMOQ_FUSED_ROUTE": "1", "LEMOQ_FUSED_ROPE": "0"}
    assert fs.fusion_payload_fields(env) == fs.fusion_states(env)
    assert fs.fusion_payload_fields(env)["fused_route"] is True
    assert fs.fusion_payload_fields(env)["fused_rope"] is False


def test_packed_qkv_has_two_preconditions():
    """F2 的"请求 ≠ 生效"：不适用时是不装（no-op），不是拒绝。

    ★ 2026-10-24：采纳后 `packed_qkv` 的默认是 True，而 23 个图格里只有 2 个格
    同时满足 `attn=kernel` **且** 量化。若"不适用"仍按第一轮那样 raise，刷新轮的
    21 个格与全部 eager 基线格会直接失败。
    """
    from lemoq.kernels.packed_qkv import packed_qkv_applicable

    assert packed_qkv_applicable("kernel", True) is True
    assert packed_qkv_applicable("kernel", False) is False  # fp16：q/k/v 是 nn.Linear
    assert packed_qkv_applicable("sdpa", True) is False  # 没有 kernel 的 decode 分支
    assert packed_qkv_applicable(None, True) is False


def test_unknown_switch_is_an_error():
    with pytest.raises(KeyError, match="未知融合开关"):
        fs.fusion_enabled("nope")


def test_every_default_has_an_env_var():
    assert set(fs.ENV_VARS) == set(fs.FUSION_DEFAULTS)
    assert all(isinstance(v, bool) for v in fs.FUSION_DEFAULTS.values())
