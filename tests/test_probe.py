"""test_probe — 机器探测的两条纪律。

1. **无 GPU 时产出完整结构**：所有字段都在，未测的写 `null`（**不是 `false`**）。
   消费端（Gate-0 脚本、report）依赖这一点，否则会误把"没测"读成"不支持"。
2. **架构特性表来自 LLVM 谓词**，必须与 docs/reference/SPECS.md §2.1 一致：
   40/50 系都**没有** wgmma；只有 sm_120 有 MX block-scale。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tools.probe_machine import SCHEMA_VERSION, arch_expectations, bandwidth_gbps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_schema_version_is_2():
    assert SCHEMA_VERSION == 2


def test_arch_table_rejects_unknown():
    out = arch_expectations("sm_999")
    assert out["known"] is False
    assert "cp_async_supported" not in out


def test_cp_async_requires_sm80():
    assert arch_expectations("sm_70")["cp_async_supported"] is False
    assert arch_expectations("sm_75")["cp_async_supported"] is False
    assert arch_expectations("sm_80")["cp_async_supported"] is True
    assert arch_expectations("sm_89")["cp_async_supported"] is True
    assert arch_expectations("sm_120")["cp_async_supported"] is True


def test_wgmma_is_never_expected_on_consumer_cards():
    """★ wgmma 门在 SM90a；40/50 系消费卡都拿不到。

    这是反复被搞错的一点（"租 40/50 卡就能写 Hopper 式流水"是错的）。
    """
    for sm in ("sm_70", "sm_80", "sm_86", "sm_89", "sm_120", "sm_121"):
        assert arch_expectations(sm)["wgmma_supported"] is False, sm


def test_fp8_requires_sm89_or_sm90():
    assert arch_expectations("sm_70")["fp8_conversion_supported"] is False
    assert arch_expectations("sm_80")["fp8_conversion_supported"] is False
    assert arch_expectations("sm_86")["fp8_conversion_supported"] is False  # A6000
    assert arch_expectations("sm_89")["fp8_conversion_supported"] is True  # 4090
    assert arch_expectations("sm_120")["fp8_conversion_supported"] is True


def test_mx_block_scale_is_sm120_only():
    """5090 独有 —— 这是选 5090 而不是 4090 的一个技术理由。"""
    assert arch_expectations("sm_120")["mx_block_scale_supported"] is True
    assert arch_expectations("sm_89")["mx_block_scale_supported"] is False
    assert arch_expectations("sm_90")["mx_block_scale_supported"] is False


def test_tma_requires_sm90():
    assert arch_expectations("sm_89")["tma_supported"] is False
    assert arch_expectations("sm_120")["tma_supported"] is True


def test_bf16_requires_sm80():
    assert arch_expectations("sm_70")["bf16_tensor_core_available"] is False
    assert arch_expectations("sm_80")["bf16_tensor_core_available"] is True


def test_arch_table_cites_its_source():
    """先验必须标明出处，否则与"编的"无法区分。"""
    src = arch_expectations("sm_120")["source"]
    assert "NVPTX" in src


# ---------------------------------------------------------------------------
# 端到端：dry-run 必须产出**完整**结构
# ---------------------------------------------------------------------------
def _run_dry_run(tmp_path):
    out = tmp_path / "probe.json"
    proc = subprocess.run(
        [sys.executable, "-m", "tools.probe_machine", "--dry-run", "--json", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


def test_dry_run_produces_full_structure(tmp_path):
    r = _run_dry_run(tmp_path)
    assert r["schema_version"] == 2
    assert r["dry_run"] is True
    for key in (
        "host",
        "driver",
        "torch",
        "bandwidth",
        "arch_features",
        "triton",
        "profiler",
        "compile",
        "disk",
    ):
        assert key in r, "缺少 section: %s" % key
    # 无 GPU 时字段必须存在
    assert "cuda_available" in r["torch"]
    assert "host_cpu" in r["host"]
    assert r["disk"]["mounts"]


def test_dry_run_uses_null_not_false_for_unmeasured(tmp_path):
    """★ 纪律断言：`null` = 未测；`false` = 测到的是"不支持"。两者不可混。"""
    r = _run_dry_run(tmp_path)
    for key in ("measured_gbps", "read_only_gbps"):
        assert r["bandwidth"][key] is None
    for key in (
        "cp_async_supported",
        "tma_supported",
        "fp8_conversion_supported",
        "mx_block_scale_supported",
        "wgmma_supported",
    ):
        assert r["arch_features"][key] is None, "%s 应为 null（未测）" % key
    assert r["arch_features"]["verified"] is False
    assert r["compile"]["cpp_extension_compiles"] is None
    assert r["triton"]["triton_works"] is None


def test_dry_run_carries_method_and_uncertainty_fields(tmp_path):
    """`docs/reference/SPECS.md` §4：每一项都要有测量方法与不确定性，否则视为未测。"""
    r = _run_dry_run(tmp_path)
    for section in ("bandwidth", "arch_features", "compile", "disk"):
        assert r[section].get("method"), "%s 缺少 method" % section
    assert "median" in r["bandwidth"]["method"]


def test_profiler_probe_reports_booleans_not_null(tmp_path):
    """ncu/nsys 的存在性是**可直接检测**的，所以这里应当是 true/false。"""
    r = _run_dry_run(tmp_path)
    assert isinstance(r["profiler"]["ncu_available"], bool)
    assert isinstance(r["profiler"]["nsys_available"], bool)


def test_gate0_script_reads_fields_that_exist(tmp_path):
    """Gate-0 脚本读的每个键都必须在 dry-run 结构里存在。"""
    r = _run_dry_run(tmp_path)
    assert r["torch"].get("cuda_available") is not None or "devices" in r["torch"]
    assert r["arch_features"].get("triton_works", "MISSING") != "MISSING"
    assert r["profiler"].get("ncu_available", "MISSING") != "MISSING"
    assert r["compile"].get("cpp_extension_compiles", "MISSING") != "MISSING"
    assert r["bandwidth"].get("measured_gbps", "MISSING") != "MISSING"
    assert r["bandwidth"].get("read_only_gbps", "MISSING") != "MISSING"


# ---------------------------------------------------------------------------
# 带宽换算 —— 那次的 bug 就在这里，必须有无 GPU 也能跑的回归测试
# ---------------------------------------------------------------------------
def test_bandwidth_gbps_uses_per_iteration_time():
    """★ 回归测试：`total_ms` 是**整段循环**的耗时，必须除以迭代数。

    第一版漏了这一步：5090 上真值 ~1520 GB/s 被报成 76 GB/s（恰好差 timed=20 倍），
    而且数值稳定、CoV 0.1%，看起来完全可信。
    """
    n_bytes = 512 * 1024 * 1024
    per_iter_ms = n_bytes / (1520e9) * 1000.0  # 真值 1520 GB/s 下单次迭代的毫秒
    got = bandwidth_gbps(n_bytes, per_iter_ms * 20, iters=20)
    assert abs(got - 1520.0) < 1.0, "得到 %.1f，应为 1520" % got


def test_bandwidth_gbps_would_catch_the_original_bug():
    """把 bug 的算法写出来，确认它给出的是 1/20 —— 说明这个测试真的在测东西。"""
    n_bytes = 512 * 1024 * 1024
    per_iter_ms = n_bytes / (1520e9) * 1000.0
    total_ms = per_iter_ms * 20
    buggy = n_bytes / (total_ms / 1000.0) / 1e9  # 漏了 /20
    correct = bandwidth_gbps(n_bytes, total_ms, iters=20)
    assert abs(buggy - correct / 20) < 0.5
    assert buggy < 100.0 and correct > 1500.0


def test_bandwidth_gbps_defaults_to_single_iteration():
    assert abs(bandwidth_gbps(1000_000_000, 1000.0) - 1.0) < 1e-9


def test_bandwidth_gbps_rejects_bad_inputs():
    import pytest as _pytest

    with _pytest.raises(ValueError):
        bandwidth_gbps(1000, 10.0, iters=0)
    assert bandwidth_gbps(1000, 0.0) == 0.0


def test_plausibility_flags_an_order_of_magnitude_error():
    """合理性告警必须真的会响 —— 它就是这次抓出 bug 的机制。"""
    from tools.probe_machine import _plausibility_check

    bad = _plausibility_check("NVIDIA GeForce RTX 5090", 78.0)
    assert bad["warning"] and "investigate" in bad["warning"].lower()
    ok = _plausibility_check("NVIDIA GeForce RTX 5090", 1641.0)
    assert ok["warning"] is None
    assert ok["spec_is_informational_only"] is True


def test_plausibility_is_silent_for_unknown_gpus():
    from tools.probe_machine import _plausibility_check

    assert _plausibility_check("Some Future GPU", 10.0)["warning"] is None
