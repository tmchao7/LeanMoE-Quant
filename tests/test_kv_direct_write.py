"""test_kv_direct_write — F4（KV 直写）的离线 K 门（FUSION_OPPORTUNITIES §F4 验收）。

F4 验收判据原文："给定 filled_len 与 pos，断言 cache 的 [0:filled_len+1] 内容与
HF `StaticCache.update()` 逐位相同"。本文件把这条判据机械化，并按仓库测试哲学加了
判别力自检（写错位置 / 越界 / 换 buffer 都必须被抓到），F4 kernel 落地后在 GPU 上
**原样重跑**本文件即可作为它的 K 门。

锚定的 HF 内部布局（transformers 4.57.6）：`StaticCache.layers[i]` 是 `StaticLayer`，
`update()` 即 `keys.index_copy_(2, cache_position, …)` 后返回整段 buffer。上游改布局
这里会响——F4 风险清单（"最容易被上游破坏的一处"）要的正是这条绊线。

本文件不产生任何可进 `results/` 的数字，只验证语义。
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lemoq.kernels.kv_write import write_kv_slot  # noqa: E402

# 与 tests/test_static_cache_decode._tiny_model 同款的小配置（MHA：kv 头数 == q 头数，
# 这也是 attn_decode 拒绝 GQA 的现实约束）；fp16 对齐 decode 真实 dtype。
LAYERS = 2
HEADS = 4
HEAD_DIM = 8
MAX_CACHE_LEN = 16
SENTINEL = 777.0  # 未填充槽位的哨兵值：直写后必须原样保留


def _config():
    from transformers import OlmoeConfig

    return OlmoeConfig(
        hidden_size=HEADS * HEAD_DIM,
        intermediate_size=16,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=64,
    )


def _new_cache():
    from transformers import StaticCache

    cache = StaticCache(
        config=_config(), max_cache_len=MAX_CACHE_LEN, device="cpu", dtype=torch.float16
    )
    # StaticLayer 是惰性初始化（首次 update 才分配 buffer）：这里主动触发，
    # 否则 poison 阶段 layer.keys 还是 None。
    dummy = torch.zeros(1, HEADS, 1, HEAD_DIM, dtype=torch.float16)
    for layer in cache.layers:
        if not layer.is_initialized:
            layer.lazy_initialization(dummy)
    return cache


def _step_kv(layer_idx: int, t: int, q_len: int = 1, seed: int = 0):
    """确定性的逐步 KV：与步号/层号相关，写错位置必然露馅。"""
    g = torch.Generator().manual_seed(seed + 1009 * layer_idx + 31 * t)
    k = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=g, dtype=torch.float32).to(torch.float16)
    v = torch.randn(1, HEADS, q_len, HEAD_DIM, generator=g, dtype=torch.float32).to(torch.float16)
    return k, v


def _poison_tail(cache, from_pos: int):
    """把 [from_pos:] 打上哨兵，用来证明直写不越界。

    未写入区是**未初始化内存**，两块 cache 互不相等——所以对照测试里哨兵必须
    从 0 位铺满（写入会覆盖 0..t），否则逐位断言会被垃圾区误伤。
    """
    for layer in cache.layers:
        layer.keys[:, :, from_pos:, :] = SENTINEL
        layer.values[:, :, from_pos:, :] = SENTINEL


def _hf_update(cache, layer_idx: int, pos: int, k, v):
    """HF 官方路径（内部走 index_copy_，版本变了也算它 whatever-it-does）。"""
    return cache.update(k, v, layer_idx, {"cache_position": torch.tensor([pos], dtype=torch.long)})


def _assert_caches_bitwise_equal(a, b):
    for li in range(LAYERS):
        assert torch.equal(a.layers[li].keys, b.layers[li].keys), "layer %d keys 不逐位相同" % li
        assert torch.equal(a.layers[li].values, b.layers[li].values), (
            "layer %d values 不逐位相同" % li
        )


def test_direct_write_bitwise_matches_hf_update_over_stream():
    """逐步流（pos=0..7）双路对照：HF update vs write_kv_slot，全程逐位相同。"""
    hf_cache, direct_cache = _new_cache(), _new_cache()
    _poison_tail(hf_cache, 0)
    _poison_tail(direct_cache, 0)
    for t in range(8):
        for li in range(LAYERS):
            k, v = _step_kv(li, t)
            _hf_update(hf_cache, li, t, k, v)
            write_kv_slot(direct_cache, li, t, k, v)
        _assert_caches_bitwise_equal(hf_cache, direct_cache)
    # 尾部哨兵原样保留：写入从未越过 filled_len
    for cache in (hf_cache, direct_cache):
        for layer in cache.layers:
            assert torch.all(layer.keys[:, :, 8:, :] == SENTINEL), "未填充槽位被改写（越界）"
            assert torch.all(layer.values[:, :, 8:, :] == SENTINEL)


def test_direct_write_returns_the_same_buffer_objects():
    """CUDA Graph 前提：返回的必须是 layer.keys/values 本身（地址稳定，不新建 buffer）。"""
    cache = _new_cache()
    k, v = _step_kv(0, 0)
    keys, values = write_kv_slot(cache, 0, 3, k, v)
    assert keys is cache.layers[0].keys
    assert values is cache.layers[0].values


def test_direct_write_does_not_touch_neighbouring_slots():
    """单步写 pos=5：pos=4 与 pos=6 的哨兵都必须原样。"""
    cache = _new_cache()
    _poison_tail(cache, 0)
    k, v = _step_kv(0, 5)
    write_kv_slot(cache, 0, 5, k, v)
    layer = cache.layers[0]
    assert torch.all(layer.keys[:, :, 4, :] == SENTINEL), "写穿了前一个槽位"
    assert torch.all(layer.keys[:, :, 6:, :] == SENTINEL), "写穿了后一个槽位"


def test_multi_token_write_matches_hf_positions():
    """q_len=4 的段写（prefill 形态）也必须与 HF 的 arange positions 等价。"""
    hf_cache, direct_cache = _new_cache(), _new_cache()
    _poison_tail(hf_cache, 0)
    _poison_tail(direct_cache, 0)
    for li in range(LAYERS):
        k, v = _step_kv(li, 0, q_len=4)
        hf_cache.update(k, v, li, {"cache_position": torch.arange(4, dtype=torch.long)})
        write_kv_slot(direct_cache, li, 0, k, v)
    _assert_caches_bitwise_equal(hf_cache, direct_cache)


def test_wrong_position_is_caught():
    """判别力自检：若 B 路写错一格（t+1 而不是 t），对照必须**不再**相等。

    一个不可能失败的对照测试比没有测试更糟（仓库测试哲学，同
    test_kernel_parity 的自检 2）。
    """
    hf_cache, direct_cache = _new_cache(), _new_cache()
    t = 2
    k, v = _step_kv(0, t)
    _hf_update(hf_cache, 0, t, k, v)
    write_kv_slot(direct_cache, 0, t + 1, k, v)  # 故意写错位置
    with pytest.raises(AssertionError):
        _assert_caches_bitwise_equal(hf_cache, direct_cache)


def test_stale_tail_makes_caches_differ():
    """判别力自检：若直写实现漏了一层或漏了 v，对照也必须红。"""
    hf_cache, direct_cache = _new_cache(), _new_cache()
    k, v = _step_kv(0, 0)
    _hf_update(hf_cache, 0, 0, k, v)
    write_kv_slot(direct_cache, 0, 0, k, v)
    # 模拟"漏写 v"：把 direct 的 v 清回哨兵
    direct_cache.layers[0].values[:, :, 0, :] = SENTINEL
    with pytest.raises(AssertionError):
        _assert_caches_bitwise_equal(hf_cache, direct_cache)
