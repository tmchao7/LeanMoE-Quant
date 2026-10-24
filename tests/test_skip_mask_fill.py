"""test_skip_mask_fill — `LEMOQ_SKIP_MASK_FILL` 候选开关的契约（纯 CPU）。

钉住三件事（spec skip-mask-fill [S2]）：
  1. 默认（env 关 / skip_mask=False）：两次 mask 更新照做，与旧行为一致；
  2. skip_mask=True：mask 逐字节不动，其余四个 fill（token/position/filled_len）照做；
  3. 开关只挂在 replay 路径：eager `decode_eager` 与 `capture` 永不跳过。

不需要 CUDA：直接构造 session 实例的最小属性集（不走 __init__，不加载模型）。
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from benchmarks.cudagraph import (  # noqa: E402
    CudaGraphConfig,
    GraphBuffers,
    StaticCacheDecodeSession,
    _skip_mask_fill_enabled,
)

MAX_LEN = 16
CTX = 8
POSITION = 8


def _bare_session() -> StaticCacheDecodeSession:
    """不加载模型、不碰 CUDA 的最小 session（只够跑 `_set_step_inputs`）。"""
    sess = StaticCacheDecodeSession.__new__(StaticCacheDecodeSession)
    sess.torch = torch
    sess.dtype = torch.float16
    sess.config = CudaGraphConfig(ctx=CTX, max_new_tokens=4, batch=1)
    sess.buffers = GraphBuffers.allocate(
        torch.device("cpu"), batch=1, max_cache_len=MAX_LEN, dtype=torch.float16
    )
    sess.position = POSITION
    sess.current_token_id = 42
    sess._captured_mask_independent = True  # mock a captured, verified kernel backend
    return sess


def test_default_fills_mask_and_other_buffers():
    sess = _bare_session()
    sess._set_step_inputs()
    mask = sess.buffers.attention_mask
    min_dtype = torch.finfo(torch.float16).min
    # 前 position+1 列为 0，其余为 min
    assert torch.all(mask[..., : POSITION + 1] == 0)
    assert torch.all(mask[..., POSITION + 1 :] == min_dtype)
    # 其余四个 fill 照做
    assert int(sess.buffers.token.item()) == 42
    assert int(sess.buffers.cache_position.item()) == POSITION
    assert int(sess.buffers.position_ids.item()) == POSITION
    assert int(sess.buffers.filled_len.item()) == POSITION + 1


def test_skip_mask_leaves_mask_untouched_but_updates_others():
    sess = _bare_session()
    sentinel = torch.full_like(sess.buffers.attention_mask, 7.0)
    sess.buffers.attention_mask.copy_(sentinel)

    sess._set_step_inputs(skip_mask=True)

    assert torch.equal(sess.buffers.attention_mask, sentinel), "mask 必须逐字节不动"
    assert int(sess.buffers.token.item()) == 42
    assert int(sess.buffers.cache_position.item()) == POSITION
    assert int(sess.buffers.position_ids.item()) == POSITION
    assert int(sess.buffers.filled_len.item()) == POSITION + 1


def test_skip_mask_still_enforces_capacity():
    sess = _bare_session()
    sess.position = MAX_LEN  # == max_cache_len ⇒ 越界
    with pytest.raises(RuntimeError, match="exceeded StaticCache capacity"):
        sess._set_step_inputs(skip_mask=True)


def test_env_helper_default_matches_switches_and_honors_flag(monkeypatch):
    """无 env ⇒ 生效值 = `FUSION_DEFAULTS`（翻默认后此测试不需要改动，2026-10-25）。"""
    from lemoq.kernels.fusion_switches import FUSION_DEFAULTS

    monkeypatch.delenv("LEMOQ_SKIP_MASK_FILL", raising=False)
    assert _skip_mask_fill_enabled() is FUSION_DEFAULTS["skip_mask_fill"]
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "0")
    assert _skip_mask_fill_enabled() is False
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "1")
    assert _skip_mask_fill_enabled() is True


def test_replay_paths_wire_the_flag_eager_and_capture_do_not():
    """静态源码契约：env 经 `_replay_skip_mask()`（开关 × 捕获期能力）只挂 replay；
    decode_eager/capture 永远默认 False。"""
    replay_src = inspect.getsource(StaticCacheDecodeSession.replay)
    assert "_set_step_inputs(skip_mask=self._replay_skip_mask())" in replay_src

    replay_at_src = inspect.getsource(StaticCacheDecodeSession.replay_at_position)
    assert "_set_step_inputs(skip_mask=self._replay_skip_mask())" in replay_at_src

    eager_src = inspect.getsource(StaticCacheDecodeSession.decode_eager)
    assert "_set_step_inputs()" in eager_src
    assert "skip_mask" not in eager_src

    capture_src = inspect.getsource(StaticCacheDecodeSession.capture)
    assert "skip_mask" not in capture_src
    assert "_set_step_inputs()" in capture_src


def test_replay_skip_mask_gates_on_captured_capability(monkeypatch):
    """翻默认后的 no-op 语义（2026-10-25）：开关开了、但捕获期不是 mask-independent
    （非 kernel 图格，如 p4b 的 sdpa）⇒ 不跳过，而不是 raise 成整格 blocked。"""
    sess = _bare_session()  # _captured_mask_independent = True
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "1")
    assert sess._replay_skip_mask() is True
    sess._captured_mask_independent = False
    assert sess._replay_skip_mask() is False
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "0")
    assert sess._replay_skip_mask() is False


def test_set_step_inputs_default_is_off():
    """默认参数必须是 False —— 已录路径逐字节不变的前提。"""
    sig = inspect.signature(StaticCacheDecodeSession._set_step_inputs)
    param = sig.parameters.get("skip_mask")
    assert param is not None
    assert param.default is False


def test_skip_mode_leaves_mask_stale_contract():
    """P6（review 2026-10-22）：skip 模式下 mask 是陈旧数据 —— 契约被单测钉住。

    skip_mask=True 时 attention_mask 逐字节不动（含 position 前缀），filled_len 照更。
    契约：**skip 模式下 mask 未定义，禁止任何图外消费者读它**；eager 路径永远重填。
    """
    sess = _bare_session()
    before = sess.buffers.attention_mask.clone()
    sess.position = 10
    sess._set_step_inputs(skip_mask=True)
    assert torch.equal(sess.buffers.attention_mask, before)  # mask 陈旧
    assert int(sess.buffers.filled_len.item()) == 11  # 长度语义仍正确
    # eager 路径（默认参数）永远重填 —— 陈旧数据不会被带回 eager
    sess._set_step_inputs()
    assert not torch.equal(sess.buffers.attention_mask, before)


def test_skip_mask_refuses_unproven_consumer_before_mutation():
    sess = _bare_session()
    sess._captured_mask_independent = False
    before = sess.buffers.token.clone()
    with pytest.raises(RuntimeError, match="mask-independent"):
        sess._set_step_inputs(skip_mask=True)
    assert torch.equal(sess.buffers.token, before)


def test_mask_capability_requires_all_layers_and_real_forward():
    from types import MethodType, SimpleNamespace

    from lemoq.kernels.attn_decode import _decode_forward_impl

    sess = _bare_session()
    attn = SimpleNamespace(_lemoq_attn_backend="triton", num_key_value_groups=1)
    attn.forward = MethodType(_decode_forward_impl, attn)
    sess.model = SimpleNamespace(
        config=SimpleNamespace(model_type="olmoe", output_attentions=False),
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)]),
    )
    assert sess._decode_mask_independent()
    attn._lemoq_attn_backend = "oracle"
    assert not sess._decode_mask_independent()
    attn._lemoq_attn_backend = "triton"
    sess.model.model.layers.append(SimpleNamespace(self_attn=SimpleNamespace()))
    assert not sess._decode_mask_independent()
