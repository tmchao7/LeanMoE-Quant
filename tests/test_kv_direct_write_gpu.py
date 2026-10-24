"""test_kv_direct_write_gpu — F4 kernel 的 K 门（GPU；`tests/test_kv_direct_write.py` 的真核版）。

钉住的口径（HANDOFF_2026-10-25 §3）：
    - kernel 直写后的 cache 必须与 HF `StaticCache.update()` **逐位**相同（k、v、
      全部槽位——写入槽逐位相同 + 未填充槽原样）；
    - `kv_direct_write_applicable` 的守卫必须把不适用形态**显式拒到回退**
      （非连续 / dtype 不一致 / D 非 2 幂 / 形状不匹配），绝不静默算错；
    - 开关（`LEMOQ_KV_DIRECT_WRITE`）默认 False：`fusion_states` 与 payload 口径一致。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

from lemoq.kernels.fusion_switches import fusion_states  # noqa: E402
from lemoq.kernels.kv_direct import (  # noqa: E402
    kv_direct_write,
    kv_direct_write_applicable,
)

pytestmark = pytest.mark.gpu

LAYERS = 2
HEADS = 4
HEAD_DIM = 8  # 2 的幂（守卫要求）；也是 tiny-model 的形状
MAX_LEN = 16
SENTINEL = 777.0


def _requires_gpu():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    try:
        import triton  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        pytest.skip("需要 triton")


def _new_cache(device="cuda"):
    from transformers import StaticCache

    from tests.test_kv_direct_write import _config  # 复用 tiny 配置

    cache = StaticCache(config=_config(), max_cache_len=MAX_LEN, device=device, dtype=torch.float16)
    dummy = torch.zeros(1, HEADS, 1, HEAD_DIM, dtype=torch.float16, device=device)
    for layer in cache.layers:
        if not layer.is_initialized:
            layer.lazy_initialization(dummy)
    return cache


def _step_kv(layer_idx: int, t: int, q_len: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed + 1009 * layer_idx + 31 * t)
    k = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=g, dtype=torch.float32).to(torch.float16)
    v = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=g, dtype=torch.float32).to(torch.float16)
    return k.cuda(), v.cuda()


def _poison(cache, from_pos: int):
    for layer in cache.layers:
        layer.keys[:, :, from_pos:, :] = SENTINEL
        layer.values[:, :, from_pos:, :] = SENTINEL


def test_kernel_direct_write_bitwise_matches_hf_update():
    """K 门：逐步流（pos=0..7）下 kernel 直写与 HF update() 逐位相同。"""
    _requires_gpu()
    hf_cache, direct_cache = _new_cache(), _new_cache()
    _poison(hf_cache, 0)
    _poison(direct_cache, 0)
    for t in range(8):
        for li in range(LAYERS):
            k, v = _step_kv(li, t)
            hf_cache.update(
                k.clone(), v.clone(), li, {"cache_position": torch.tensor([t], device="cuda")}
            )
            layer = direct_cache.layers[li]
            assert kv_direct_write_applicable(layer, k, v)
            kv_direct_write(layer, k, v, torch.tensor([t], device="cuda"))
    for li in range(LAYERS):
        assert torch.equal(hf_cache.layers[li].keys, direct_cache.layers[li].keys), li
        assert torch.equal(hf_cache.layers[li].values, direct_cache.layers[li].values), li
    # 尾部哨兵原样：写入未越界
    for cache in (hf_cache, direct_cache):
        for layer in cache.layers:
            assert torch.all(layer.keys[:, :, 8:, :] == SENTINEL)
            assert torch.all(layer.values[:, :, 8:, :] == SENTINEL)


def test_applicable_guard_rejects_bad_shapes_and_layouts():
    """守卫必须显式拒绝不适用形态（回退 update()，不静默算错）。"""
    _requires_gpu()
    cache = _new_cache()
    layer = cache.layers[0]
    k, v = _step_kv(0, 0)
    assert kv_direct_write_applicable(layer, k, v) is True

    # dtype 不一致（k 是 fp32）
    assert kv_direct_write_applicable(layer, k.float(), v.float()) is False
    # 形状不匹配（q_len=2）
    k2, v2 = _step_kv(0, 0, q_len=2)
    assert kv_direct_write_applicable(layer, k2, v2) is False
    # 非连续 cache
    layer.keys.data = layer.keys.data.transpose(2, 3).contiguous().transpose(2, 3)
    assert kv_direct_write_applicable(layer, k, v) is False


def test_switch_matches_code_default_and_payload_fields_cover_it():
    """★ 对着 FUSION_DEFAULTS 断言（N6 教训：写死 True/False 会让采纳那天测试变红）。"""
    from lemoq.kernels.fusion_switches import FUSION_DEFAULTS

    states = fusion_states(env={})
    assert states["kv_direct_write"] is FUSION_DEFAULTS["kv_direct_write"]
    assert set(states) == {
        "fused_rope",
        "fused_route",
        "skip_mask_fill",
        "packed_qkv",
        "kv_direct_write",
    }
