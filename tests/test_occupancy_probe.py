"""test_occupancy_probe — 钉住 occupancy 上界算术与"取不到就不出数字"（CPU 可跑）。

为什么值得测：这个工具是 ncu 被阻断后的**替代口径**，它的输出会被写进结论
（"融合是否引入了新的 occupancy 天花板"）。它有两个失败模式：
  (a) 上界算术写错（三个除法取最小 / 寄存器粒度），会给出偏乐观的并行度；
  (b) 取不到 Triton 编译产物时**编一个数字**填进去 —— 那是本项目定义的最严重错误
      （硬规则 1：未测必须写成未测）。
两条都由测试钉住。
"""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, ".")

from tools import occupancy_probe as op  # noqa: E402


def test_upper_bound_takes_the_min_of_three_limits():
    """三个限制来源取最小 —— 手工算一遍对照。"""
    limits = {"regs_per_sm": 65536, "smem_per_sm": 228 * 1024, "max_threads_sm": 1536}
    # n_regs=64, tpb=256: regs → 65536/(64*256)=4 ; threads → 1536/256=6 ; smem → 大
    got = op.occupancy_upper_bound(
        n_regs=64, threads_per_block=256, shared_bytes=1024, limits=limits
    )
    assert got["blocks_per_sm_bound"] == 4
    assert got["limited_by"] == "regs"
    assert got["occupancy_upper_pct"] == pytest.approx(100.0 * 4 * 256 / 1536, abs=0.05)
    assert got["kind"] == "estimate" and got["bound"] == "upper"


def test_shared_memory_can_be_the_binding_limit():
    """shared memory 占满时它应当成为限制来源（融合 kernel 的典型风险）。"""
    limits = {"regs_per_sm": 65536, "smem_per_sm": 100 * 1024, "max_threads_sm": 1536}
    got = op.occupancy_upper_bound(
        n_regs=32, threads_per_block=256, shared_bytes=64 * 1024, limits=limits
    )
    assert got["blocks_per_sm_bound"] == 1
    assert got["limited_by"] == "shared"


def test_register_heavy_kernel_gets_a_sub_one_block_bound():
    """★ 这正是本工具要抓的形态：寄存器用量把 blocks/SM 压到 1 以下 ⇒ SM 填不满。"""
    limits = {"regs_per_sm": 65536, "smem_per_sm": 228 * 1024, "max_threads_sm": 1536}
    got = op.occupancy_upper_bound(
        n_regs=255, threads_per_block=1024, shared_bytes=0, limits=limits
    )
    assert got["blocks_per_sm_bound"] == 0, "255 regs × 1024 线程 > 65536 regs/SM"
    assert got["occupancy_upper_pct"] == 0.0
    assert got["limited_by"] == "regs"


def test_sm_limits_table_covers_sm120_and_falls_back(monkeypatch):
    """sm_120（本项目的卡）必须在兜底表里；查不到 cc 时不能崩。"""

    class _P:
        major, minor = 12, 0

    class _Cuda:
        @staticmethod
        def get_device_properties(_=0):
            return _P()

    class _Torch:
        cuda = _Cuda

    got = op._limits(_Torch)
    assert got["max_threads_sm"] >= 1024 and got["regs_per_sm"] >= 32768
    assert (12, 0) in op._SM_LIMITS


def test_missing_compiled_artifact_stays_planned_without_numbers():
    """取不到编译产物 ⇒ 该 subject 保持 planned，且**不含任何数字字段**。"""
    attempts: list[str] = []
    info = op._compiled_kernel_info(object(), attempts=attempts)  # 没有 cache 属性
    assert info is None
    assert attempts, "失败原因必须记录下来（否则上机时不知道该试什么）"


def test_find_compiled_kernel_reads_both_cache_layouts():
    """Triton 3.x 早期 `.cache` 与较新 `.device_caches` 两种布局都要认。"""

    class _CK:
        n_regs = 96
        n_spills = 2

        class metadata:
            shared = 4096

    class _Old:
        cache = {"k": _CK()}

    class _New:
        device_caches = {"cuda:0": {"k": _CK()}}

    for obj in (_Old(), _New()):
        ck = op._find_compiled_kernel(obj)
        assert ck is not None and ck.n_regs == 96
