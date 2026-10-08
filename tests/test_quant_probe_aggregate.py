"""test_quant_probe_aggregate — `--aggregate-only` 的两条纪律。

## 为什么需要这个模式（P2i，2026-10-08）

一次 50 分钟的门重测被服务器关机打断时，**已经量完的窗口不能白扔**。但聚合值必须由
**工具自己的代码**算出来 —— 手算的聚合值不能当引用（硬规则 1/7：数字必须有工件）。
所以 `aggregate_windows()` 被抽出来，长跑和 `--aggregate-only` 走**同一段代码**。

本文件钉住两件事：

1. **口径**：`ones*`（退化对照）不进 mean，只记在 `ones_control_kl`；
2. **自曝**：`--aggregate-only` 必须在工件里写明"这次聚合是谁算的、哪些臂的窗口其实不完整"
   （`_aggregate_provenance`），免得后人把不完整臂的 aggregate 当成完整实测读。
"""

from __future__ import annotations

import json

from tools.quant_probe import main


def _artifact() -> dict:
    def entry(kl: float, top1: float, bits: int = 8) -> dict:
        return {"kl": kl, "top1_agreement": top1, "spec": {"weight_bits": bits}}

    return {
        "tool": "tools.quant_probe.py",
        "windows": {"w0": {}, "w1": {}},
        "arms": {
            # 完整臂：5 个真实窗口 + 1 个 ones 对照
            "w8a16_attn_experts_g128": {
                "kind": "rtn",
                "group": "prompt_set",
                "windows": {
                    "w0": entry(0.0010, 1.0),
                    "w1": entry(0.0020, 0.9),
                    "w2": entry(0.0030, 1.0),
                    "w3": entry(0.0040, 1.0),
                    "w4": entry(0.0000, 1.0),
                    "ones_control": entry(0.5, 1.0),  # 退化对照：**不得**进 mean
                },
            },
            # 不完整臂：只有 2 个窗口（长跑被打断时常见）
            "w4a16_attn_experts_g128": {
                "kind": "rtn",
                "group": "prompt_set",
                "windows": {
                    "w0": entry(0.0500, 0.8, bits=4),
                    "w1": entry(0.0700, 0.7, bits=4),
                },
            },
        },
    }


def test_aggregate_only_computes_mean_over_real_windows_and_derives_the_gate(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    assert main(["--aggregate-only", str(path)]) == 0
    out = json.loads(path.read_text(encoding="utf-8"))

    full = out["arms"]["w8a16_attn_experts_g128"]
    agg = full["aggregate"]
    # (0.0010+0.0020+0.0030+0.0040+0.0000)/5 = 0.0020，ones 的 0.5 **不在**里面
    assert agg["n_windows"] == 5
    assert abs(agg["kl_mean"] - 0.0020) < 1e-12
    assert agg["kl_min"] == 0.0 and agg["kl_max"] == 0.0040
    assert agg["ones_control_kl"] == 0.5
    # 门必须由实测值 + τ 推出来（w8a16 → 0.01）
    assert full["gate"]["weight_bits"] == 8
    assert full["gate"]["kl_mean_pass"] is True
    assert full["gate"]["all_windows_pass"] is True
    assert full["gate"]["ones_control_pass"] is False, "ones 对照也要按同一 τ 判（它不参与 mean）"

    thin = out["arms"]["w4a16_attn_experts_g128"]
    assert thin["aggregate"]["n_windows"] == 2
    assert thin["gate"]["weight_bits"] == 4 and thin["gate"]["kl_max"] == 0.05
    assert thin["gate"]["kl_mean_pass"] is False  # (0.05+0.07)/2 = 0.06 > 0.05


def test_aggregate_only_records_provenance_and_flags_thin_arms(tmp_path):
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")

    assert main(["--aggregate-only", str(path)]) == 0
    out = json.loads(path.read_text(encoding="utf-8"))

    prov = out["_aggregate_provenance"]
    assert "aggregate-only" in prov["by"]
    assert prov["source"] == str(path)
    # ★ 不完整臂必须被点名 —— 否则读工件的人会把 <5 窗口的 mean 当完整实测
    assert prov["arms_with_fewer_than_5_real_windows"] == ["w4a16_attn_experts_g128"]
    assert prov["n_arms_fewer_than_5"] == 1


def test_aggregate_only_can_write_elsewhere_without_touching_the_source(tmp_path):
    src = tmp_path / "probe.json"
    src.write_text(json.dumps(_artifact()), encoding="utf-8")
    dst = tmp_path / "out.json"

    assert main(["--aggregate-only", str(src), "--json", str(dst)]) == 0
    assert (
        "aggregate"
        not in json.loads(src.read_text(encoding="utf-8"))["arms"]["w8a16_attn_experts_g128"]
    )
    assert json.loads(dst.read_text(encoding="utf-8"))["arms"]["w8a16_attn_experts_g128"][
        "aggregate"
    ]
