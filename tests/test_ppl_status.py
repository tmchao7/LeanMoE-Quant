"""test_ppl_status — 合并前 noop 防呆必须真的会触发。

2026-10-10 评审发现：防呆读的是不存在的 `entry["paired"]["mean"]`，
而 quant_ppl 产出的键是 `paired_vs_fp16.delta_nll_per_token` —— guard 是死代码。
这里用合成工件把三条路径（非零拒绝 / 缺字段拒绝 / 干净放行）钉死。
"""

from __future__ import annotations

import json

import pytest

from tools.ppl_status import _load_ppl_run


def _quant_ppl(tmp_path, noop_delta=0.0, with_paired=True):
    arms = {
        "noop_a": {"kind": "noop", "ppl": 9.0},
        "w8a16_g128": {"kind": "rtn", "ppl": 9.1},
    }
    if with_paired:
        arms["noop_a"]["paired_vs_fp16"] = {
            "delta_nll_per_token": noop_delta,
            "paired_se": 0.0,
            "t_stat": None,
            "n_pairs": 10,
        }
    data = {
        "fp16": {"ppl": 9.0},
        "arms": arms,
        "model": "m",
        "dataset": "d",
        "n_tokens": 10,
        "ctx_window": "c",
        "metric": "ppl",
    }
    path = tmp_path / "quant_ppl.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_noop_nonzero_delta_is_rejected(tmp_path):
    path = _quant_ppl(tmp_path, noop_delta=1e-3)
    with pytest.raises(SystemExit) as exc:
        _load_ppl_run(path, "w8a16_g128")
    assert "ΔNLL" in str(exc.value)


def test_noop_missing_paired_block_is_rejected(tmp_path):
    path = _quant_ppl(tmp_path, with_paired=False)
    with pytest.raises(SystemExit) as exc:
        _load_ppl_run(path, "w8a16_g128")
    assert "paired_vs_fp16" in str(exc.value)


def test_clean_run_loads(tmp_path):
    path = _quant_ppl(tmp_path, noop_delta=0.0)
    fp16, arm, meta = _load_ppl_run(path, "w8a16_g128")
    assert fp16["ppl"] == 9.0
    assert arm["ppl"] == 9.1
    assert meta["n_tokens"] == 10
