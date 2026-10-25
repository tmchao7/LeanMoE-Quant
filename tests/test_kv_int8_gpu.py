"""test_kv_int8_gpu — E6 写路径 kernel 的 K 门（GPU；骨架预置于 2026-10-24）。

★ `kv_int8_write` 尚未实现（10-25 窗落地）——本文件按 EXP-p5-01 记录 §K门API 预置，
kernel 落地**同名 API** 后自动激活。钉住的判据（预注册，记录 §Method）：
    - write 侧 kernel vs 参考数学（`tests/test_kv_int8.py` 的 CPU 链）**逐位**：
      codes int8 逐位 + scale fp16 逐位（同一 fp32 舍入链：amax → /127 →
      round half-to-even → clamp ±127 → fp16 就近；kernel 必须复现参考舍入，
      对不齐是实现问题不是判据问题——F4 先例）；
    - 全零行：codes=0、scale=1.0；
    - roundtrip：参考 dequant(kernel 输出) 与原行的逐元素 |err| ≤ 0.6·scale；
    - 守卫把不适用形态显式拒掉（CPU/坏形状已在 CPU 测试；这里补 CUDA 上的
      D 非 2 幂 / dtype 错配）。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

try:  # ★ kernel 未落地时整文件 skip；落地同名 API 后自动激活（骨架语义）
    from lemoq.kernels.kv_int8 import (  # noqa: F401
        dequant_int8,
        kv_int8_rows_applicable,
        kv_int8_write,
        quantize_per_token_int8,
    )

    _KERNEL_READY = True
except ImportError:
    _KERNEL_READY = False

pytestmark = [pytest.mark.gpu]

HEADS = 4
HEAD_DIM = 8  # 2 的幂（守卫要求）
SEED = 7


def _requires_gpu():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    try:
        import triton  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        pytest.skip("需要 triton")


def _rows(h: int = HEADS, d: int = HEAD_DIM, seed: int = SEED, zeros: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, h, 1, d, generator=g, dtype=torch.float32).to(torch.float16)
    if zeros:
        x = torch.zeros_like(x)
    return x.cuda()


@pytest.mark.skipif(
    not _KERNEL_READY, reason="E6 kv_int8_write kernel 未落地（10-25 窗）——K 门骨架"
)
def test_write_kernel_matches_reference_bitwise():
    """K 门 ①：kernel 写出的 codes/scale 与参考数学逐位相同。"""
    assert _requires_gpu()
    k, v = _rows(seed=SEED), _rows(seed=SEED + 1)
    k_codes, k_scale, v_codes, v_scale = kv_int8_write(k, v)
    ref_kc, ref_ks = quantize_per_token_int8(k)
    ref_vc, ref_vs = quantize_per_token_int8(v)
    assert torch.equal(k_codes, ref_kc), "K codes 与参考不逐位"
    assert torch.equal(k_scale, ref_ks), "K scale 与参考不逐位"
    assert torch.equal(v_codes, ref_vc), "V codes 与参考不逐位"
    assert torch.equal(v_scale, ref_vs), "V scale 与参考不逐位"


@pytest.mark.skipif(
    not _KERNEL_READY, reason="E6 kv_int8_write kernel 未落地（10-25 窗）——K 门骨架"
)
def test_write_kernel_zero_row_convention():
    """K 门 ②：全零行 scale=1.0、codes=0（除零约定与参考一致）。"""
    assert _requires_gpu()
    z = _rows(zeros=True)
    k_codes, k_scale, _, _ = kv_int8_write(z, z)
    assert (k_codes == 0).all() and (k_scale == 1.0).all()


@pytest.mark.skipif(
    not _KERNEL_READY, reason="E6 kv_int8_write kernel 未落地（10-25 窗）——K 门骨架"
)
def test_write_kernel_roundtrip_bound():
    """K 门 ③：参考 dequant(kernel 输出) 的逐元素误差 ≤ 0.6·scale。"""
    assert _requires_gpu()
    for seed in (SEED, SEED + 1):
        x = _rows(seed=seed)
        codes, scale, _, _ = kv_int8_write(x, x)
        rec = dequant_int8(codes, scale.unsqueeze(-1))
        err = (rec - x.float()).abs()
        assert (err <= 0.6 * scale.float().unsqueeze(-1)).all(), f"seed={seed} 超界"


@pytest.mark.skipif(
    not _KERNEL_READY, reason="E6 kv_int8_write kernel 未落地（10-25 窗）——K 门骨架"
)
def test_guard_rejects_bad_shapes_on_cuda():
    """K 门 ④：CUDA 上的坏形态（D 非 2 幂 / dtype 错配）显式拒绝。"""
    assert _requires_gpu()
    good = _rows()
    assert kv_int8_rows_applicable(good, good.clone()) is True
    bad_d = torch.randn(1, HEADS, 1, 12, dtype=torch.float16).cuda()
    assert kv_int8_rows_applicable(bad_d, bad_d.clone()) is False
    assert kv_int8_rows_applicable(good, good.float()) is False
