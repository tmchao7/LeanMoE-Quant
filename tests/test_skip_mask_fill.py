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


def test_env_helper_default_off_and_honors_flag(monkeypatch):
    monkeypatch.delenv("LEMOQ_SKIP_MASK_FILL", raising=False)
    assert _skip_mask_fill_enabled() is False
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "0")
    assert _skip_mask_fill_enabled() is False
    monkeypatch.setenv("LEMOQ_SKIP_MASK_FILL", "1")
    assert _skip_mask_fill_enabled() is True


def test_replay_paths_wire_the_flag_eager_and_capture_do_not():
    """静态源码契约：env 只挂 replay；decode_eager/capture 永远默认 False。"""
    replay_src = inspect.getsource(StaticCacheDecodeSession.replay)
    assert "_set_step_inputs(skip_mask=_skip_mask_fill_enabled())" in replay_src

    replay_at_src = inspect.getsource(StaticCacheDecodeSession.replay_at_position)
    assert "_set_step_inputs(skip_mask=_skip_mask_fill_enabled())" in replay_at_src

    eager_src = inspect.getsource(StaticCacheDecodeSession.decode_eager)
    assert "_set_step_inputs()" in eager_src
    assert "skip_mask" not in eager_src

    capture_src = inspect.getsource(StaticCacheDecodeSession.capture)
    assert "skip_mask" not in capture_src
    assert "_set_step_inputs()" in capture_src


def test_set_step_inputs_default_is_off():
    """默认参数必须是 False —— 已录路径逐字节不变的前提。"""
    sig = inspect.signature(StaticCacheDecodeSession._set_step_inputs)
    param = sig.parameters.get("skip_mask")
    assert param is not None
    assert param.default is False
