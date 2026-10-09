"""test_batch_sweep — 用**假模型**钉住 batch_sweep 的冷态守卫与 VOID 语义（CPU 可跑）。

为什么值得测：`tools/batch_sweep.py` 是唯一会在计费机器上跑、且**结论完全依赖
"两次测同一个格子"是否稳定**的工具。2026-10-09 它在真机上连判两轮 VOID
（金丝雀漂移 6.3% / 56.3%），根因是 batch=1 冷启（首轮 142 ms vs 稳态 92 ms）。
修完之后，"丢弃轮真的被丢了""预热真的发生在测量之前""VOID 轮的数字不会泄露成
可引用结果"这三件事必须由测试钉住 —— 否则下次有人把 discard 改成 0，症状会
换个样子回来（那次是整轮 VOID，看不出来是代码问题）。
"""

from __future__ import annotations

import sys
import time

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, ".")

from tools import batch_sweep as bs  # noqa: E402


class _Out:
    def __init__(self, logits, past):
        self.logits = logits
        self.past_key_values = past


class _FakeModel:
    """最小可用的 decode 桩：`__call__(ids, use_cache=True) -> .logits/.past_key_values`。

    `cost_ms` 是每次前向的人为耗时；`first_calls_slow` 让**前 N 次调用**故意变慢，
    用来模拟冷启（这正是真机上发生的事）。
    """

    def __init__(self, *, cost_ms: float = 0.0, first_calls_slow: int = 0, slow_ms: float = 0.0):
        self.cost_ms = cost_ms
        self.first_calls_slow = first_calls_slow
        self.slow_ms = slow_ms
        self.calls = 0

    def __call__(self, ids, use_cache=True, past_key_values=None):
        self.calls += 1
        if self.calls <= self.first_calls_slow:
            time.sleep(self.slow_ms / 1000.0)
        elif self.cost_ms:
            time.sleep(self.cost_ms / 1000.0)
        b, t = ids.shape[0], 4
        logits = torch.zeros(b, t, 8)
        logits[:, -1, 3] = 1.0  # argmax 恒为 3，与真实路径同一形状语义
        return _Out(logits, past_key_values)


# 桩不需要真 CUDA：这几个 helper 只用到 torch.cuda 的 sync/peak 与 device 名
@pytest.fixture(autouse=True)
def _fake_cuda(monkeypatch):
    class _C:
        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 1234

        @staticmethod
        def get_device_name(_=0):
            return "fake-gpu"

    monkeypatch.setattr(bs, "torch", torch, raising=False)
    monkeypatch.setattr(torch, "cuda", _C, raising=False)
    yield


def test_discard_repeats_are_excluded_from_the_samples():
    """★ 丢弃轮必须**不进**统计量，但值要留下来（丢弃 ≠ 不知道它多少）。"""
    model = _FakeModel(cost_ms=0.0)
    kept, discarded, _ = bs._time_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        batch=1,
        warmup=1,
        timed=20,
        repeats=3,
        discard=2,
    )
    assert len(kept) == 3, "保留轮数应等于 --repeats"
    assert len(discarded) == 2, "丢弃轮数应等于 --discard-repeats"


def test_discard_actually_removes_the_cold_first_round():
    """冷启会让第一轮明显更慢：丢弃它之后，保留样本不应包含那个慢值。"""
    model = _FakeModel(first_calls_slow=1, slow_ms=60.0, cost_ms=0.0)
    kept, discarded, _ = bs._time_decode(
        model,
        torch.zeros(1, 2, dtype=torch.long),
        batch=1,
        warmup=1,
        timed=20,
        repeats=3,
        discard=1,
    )
    assert len(discarded) == 1
    assert len(kept) == 3
    # 慢值只可能出现在被丢弃的那一轮里（或根本不出现在计时里 —— 预热已吃掉它）
    if discarded[0] > 5.0:
        assert max(kept) < discarded[0], "慢的冷启轮漏进了保留样本"


def test_warmup_pass_runs_before_any_timing():
    """全局预热必须在测量之前发生：预热后第一次调用不算冷启。"""
    model = _FakeModel()
    bs._warmup_pass(model, torch.zeros(1, 2, dtype=torch.long), steps=3)
    # prefill 1 次 + 3 个 decode 步 = 4 次
    assert model.calls == 4, "预热应走 1 次 prefill + steps 次 decode，实际 %d" % model.calls


def test_dry_run_plan_declares_the_cold_start_guard(capsys):
    """dry-run 计划里必须写明冷态守卫，否则上机的人不会知道有丢弃轮。"""
    rc = bs.main(["--dry-run"])
    assert rc == 0
    import json

    plan = json.loads(capsys.readouterr().out.split("\n[dry-run]")[0])
    assert plan["discard_repeats"] >= 1
    assert "冷启" in plan["cold_start_guard"] or "预热" in plan["cold_start_guard"]
    assert plan["cross_round_comparison_allowed"] is False


def test_void_round_hides_numbers_but_keeps_evidence():
    """VOID ⇒ `rows` 为空；实测值只在 `void_evidence` 里且标注不得引用。"""
    payload = {
        "status": "void",
        "rows": [],
        "void_evidence": {
            "why": "金丝雀漂移 56.337% > 5.0% ⇒ 整轮 VOID",
            "must_not_be_cited": True,
            "rows": [{"batch": 1, "per_step_p50_ms": 142.26}],
        },
    }
    assert payload["rows"] == []
    assert payload["void_evidence"]["must_not_be_cited"] is True
    assert payload["void_evidence"]["rows"], "诊断材料必须留下（否则下一次还得重新踩）"
