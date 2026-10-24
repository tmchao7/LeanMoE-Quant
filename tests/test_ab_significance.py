"""test_ab_significance — 对盘上 A/B 工件的重算口径（tools/ab_significance.py；R1）。

钉住的行为：
    - v1 工件（只有交替 runs）按臂分组重建配对，tail_canary 不算对；
    - v2 工件直接用持久化的 paired_wall.delta_ms；
    - runner result.json 里 _measurement.interleaved_ab.paired.delta_ms 能取到；
    - 只有摘要统计的旧 payload 如实 skip，不伪造数据。
"""

from __future__ import annotations

from pathlib import Path

from tools.ab_significance import analyze


def _v1_payload() -> dict:
    """2026-10-23 实测工件的形状（results/fused_rope_ab.json 同构，数字换小）。"""
    runs = []
    for i, (arm, val) in enumerate([("off", 2.0), ("on", 1.8), ("off", 2.1), ("on", 1.9)]):
        runs.append({"i": i, "arm": arm, "wall_p50_ms": val, "gpu_busy_ms": val * 0.95})
    return {"generated_by": "tools/profile_graph.py --ab-fused-rope", "runs": runs}


def test_v1_reconstructs_pairs_by_arm(tmp_path: Path):
    p = tmp_path / "v1.json"
    p.write_text(__import__("json").dumps(_v1_payload()), encoding="utf-8")
    out = analyze(p)
    assert out["status"] == "ok"
    assert out["provenance"] == "v1_reconstructed_from_runs"
    assert out["arms"] == ["off", "on"]
    sig = out["significance"]
    assert sig["n_pairs"] == 2
    # deltas = [2.0-1.8, 2.1-1.9] 全正 ⇒ on 两对全胜,但 n=2 的双侧 p=0.5
    assert sig["wins"]["on_faster"] == 2
    assert sig["sign_test_p"] == 0.5
    assert sig["significant_at_0_05"] is False
    assert "n=rounds" in out["note"]


def test_v1_tail_canary_not_counted_as_pair(tmp_path: Path):
    import json

    payload = _v1_payload()
    payload["runs"].append({"i": 4, "arm": "off", "role": "tail_canary", "wall_p50_ms": 2.05})
    p = tmp_path / "v1_tail.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = analyze(p)
    assert out["status"] == "ok"
    assert out["significance"]["n_pairs"] == 2, "尾金丝雀是额外的 a 臂 run,不得算进配对"


def test_v2_uses_persisted_deltas(tmp_path: Path):
    import json

    payload = {
        "protocol_version": 2,
        "env_var": "LEMOQ_PACKED_QKV",
        "paired_wall": {"n": 3, "delta_ms": [0.09, 0.11, 0.10]},
        "runs": [
            {"i": 0, "arm": "off", "role": "pair", "wall_p50_ms": 1.95},
            {"i": 1, "arm": "on", "role": "pair", "wall_p50_ms": 1.86},
        ],
    }
    p = tmp_path / "v2.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = analyze(p)
    assert out["provenance"] == "profile_graph_v2_persisted"
    assert out["arms"] == ["off", "on"]
    assert out["significance"]["n_pairs"] == 3


def test_runner_payload_nested_under_measurement(tmp_path: Path):
    import json

    payload = {
        "_measurement": {
            "interleaved_ab": {
                "arms": {"separate": {"n": 60}, "fused": {"n": 60}},
                "paired": {"n_pairs": 60, "delta_ms": [0.001] * 40 + [-0.001] * 20},
            }
        }
    }
    p = tmp_path / "cell.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = analyze(p)
    assert out["status"] == "ok"
    assert out["provenance"] == "runner_persisted"
    assert out["arms"] == ["separate", "fused"]
    sig = out["significance"]
    assert sig["n_pairs"] == 60
    assert sig["significant_at_0_05"] is True  # 40/60 胜,符号检验 p≈0.0016
    assert sig["wilcoxon_method"] == "normal_approx"


def test_summary_only_payload_skips_honestly(tmp_path: Path):
    import json

    payload = {
        "interleaved_ab": {
            "arms": {"separate": {"n": 60}, "fused": {"n": 60}},
            "paired": {"n_pairs": 60, "median_delta_ms": 0.9, "wins": {"fused_faster": 55}},
        }
    }
    p = tmp_path / "old.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = analyze(p)
    assert out["status"] == "skip"
    assert "无可重算" in out["reason"]
