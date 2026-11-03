"""test_claims — 工件反算与文档数字一致性（2026-11-03）。

这组测试把"每个数字都追得到工件"从口号变成机械检查：
    - `tools.e2e_report` 从四个 canonical payload 反算 TTFT + 32/128-token E2E；
    - `tools.check_claims` 复算 canonical 1.4602/1.6075、2.4421/4.2506、E7 1.2253/1.1025，
      并扫描未标历史的旧值（1.1911 / 1.1974 / 1.69× / 1.31× / ...）。

干净克隆没有 `results/`：payload 检查自动跳过，文档扫描仍执行。
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HAS_RESULTS = (ROOT / "results").exists()

from tools import check_claims, e2e_report  # noqa: E402


def test_e2e_report_reconstructs_32_token_ratios():
    if not HAS_RESULTS:
        pytest.skip("results/ 不在干净克隆里")
    rows, missing = e2e_report.build_rows()
    assert not missing
    by_ctx = {row["ctx"]: row for row in rows}
    assert abs(by_ctx[1024]["ratio_32"] - 1.3824) < 2e-3
    assert abs(by_ctx[4096]["ratio_32"] - 1.7190) < 2e-3
    # E2E 必须 > decode-step ratio（TTFT 被摊薄后仍 >= 1）
    assert by_ctx[1024]["ratio_32"] > 1.0
    assert by_ctx[4096]["ratio_32"] > 1.0


def test_e2e_report_check_exit_code():
    if not HAS_RESULTS:
        pytest.skip("results/ 不在干净克隆里")
    assert e2e_report.main(["--check"]) == 0


def test_check_claims_passes_on_current_tree():
    assert check_claims.main(["--check"]) == 0
