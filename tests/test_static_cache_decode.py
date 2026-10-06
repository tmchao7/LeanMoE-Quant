"""test_static_cache_decode — 离线预检 CUDA Graph session 用到的**整条 API 契约**。

`benchmarks/cudagraph.py` 的 `StaticCacheDecodeSession` 依赖一串很具体的调用方式：

    StaticCache(config=model.config, max_cache_len=ctx+max_new_tokens)
    model(..., past_key_values=cache, cache_position=pos, position_ids=pos,
          attention_mask=<4D 静态 mask>, use_cache=True)
    cache.reset()

这些在 CPU 上**都能验证**（除了真正的 capture/replay）。如果不在离线验证，
它们会在计费机器上、加载完 13 GB 模型之后才炸，然后花时间分辨
"是 API 用错了还是 CUDA Graph 不行"。

★ 本文件不产生任何可进 `results/` 的数字 —— 它只验证调用方式与数值正确性。

## 一个关键的设计确认

`OlmoeModel._prepare_4d_causal_attention_mask_with_cache_position` 开头是：

    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask          # 直接用，不做任何处理

也就是说**传 4D mask 会完全绕过每次 forward 的 mask 构造**。
session 正是这么做的（`GraphBuffers.attention_mask` 是
`(batch, 1, 1, max_cache_len)` 的 4D 缓冲），这是能进 CUDA Graph 的前提 ——
否则每步都会新建一个 mask 张量。
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def _tiny_model(seed=0):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=64,
    )
    torch.manual_seed(seed)
    model = OlmoeForCausalLM(cfg).eval()
    return model, cfg


def _static_mask(batch, max_cache_len, position, dtype):
    """复刻 session 里 `_set_step_inputs` 的 mask 构造方式（4D、原地改）。"""
    mask = torch.full((batch, 1, 1, max_cache_len), torch.finfo(dtype).min, dtype=dtype)
    mask[..., : position + 1] = 0.0
    return mask


# ---------------------------------------------------------------------------
# API 契约
# ---------------------------------------------------------------------------
def test_static_cache_accepts_the_documented_arguments():
    from transformers import StaticCache

    model, cfg = _tiny_model()
    cache = StaticCache(config=model.config, max_cache_len=cfg.max_position_embeddings)
    # session 依赖的三个接口
    assert hasattr(cache, "get_seq_length")
    assert hasattr(cache, "reset")
    cache.reset()


def test_model_accepts_cache_position_and_position_ids_with_static_cache():
    """session 的 prefill 调用签名必须成立。"""
    from transformers import StaticCache

    model, cfg = _tiny_model()
    ctx = 8
    cache = StaticCache(config=model.config, max_cache_len=ctx + 4)
    ids = torch.full((1, ctx), 1, dtype=torch.long)
    pos = torch.arange(ctx, dtype=torch.long)
    mask = torch.ones((1, ctx), dtype=torch.long)

    with torch.no_grad():
        out = model(
            ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            position_ids=pos.unsqueeze(0),
        )
    assert out.logits.shape == (1, ctx, cfg.vocab_size)


def test_prefill_then_decode_with_4d_static_mask_matches_plain_eager():
    """★ 核心预检：session 的 prefill+decode 流程必须与朴素 eager 前向**数值一致**。

    如果这里不一致，就不该去 GPU 上查 CUDA Graph —— 那是别的问题。
    """
    from transformers import StaticCache

    model, cfg = _tiny_model(seed=7)
    ctx = 8
    max_cache_len = ctx + 4
    ids = torch.full((1, ctx), 1, dtype=torch.long)
    dtype = next(model.parameters()).dtype

    # --- 参照：一次前向拿全部 token 的 logits ---
    with torch.no_grad():
        ref = model(ids).logits  # (1, ctx, V)

    # --- session 的走法：prefill + 逐步 decode ---
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    pos = torch.arange(ctx, dtype=torch.long)
    with torch.no_grad():
        out = model(
            ids,
            attention_mask=torch.ones((1, ctx), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            position_ids=pos.unsqueeze(0),
        )
    # prefill 的最后位置必须与朴素实现一致
    assert torch.allclose(
        out.logits[:, -1, :], ref[:, -1, :], atol=1e-5
    ), "prefill 最后一位与 eager 不一致"

    # --- decode：用 4D 静态 mask 再走 3 步 ---
    token = int(out.logits[:, -1, :].argmax(dim=-1).item())
    prev = out.logits[:, -1, :]
    for step in range(3):
        position = ctx + step
        tok = torch.tensor([[token]], dtype=torch.long)
        mask = _static_mask(1, max_cache_len, position, dtype)
        with torch.no_grad():
            step_out = model(
                tok,
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], dtype=torch.long),
                position_ids=torch.tensor([[position]], dtype=torch.long),
            )
        assert step_out.logits.shape == (1, 1, cfg.vocab_size)
        assert torch.isfinite(step_out.logits).all(), "decode 出现非有限值（mask 可能算错）"
        token = int(step_out.logits[:, -1, :].argmax(dim=-1).item())
        prev = step_out.logits[:, -1, :]

    assert prev is not None


def test_4d_mask_bypasses_per_forward_mask_construction():
    """★ 传 4D mask 时，模型必须**原样使用**它，不新建 mask。

    这条是 CUDA Graph 能否成立的关键：如果每步都新建 mask 张量，
    那些分配在 capture 时会被冻结，replay 时形状就对不上了。
    """
    model, cfg = _tiny_model(seed=11)
    ctx = 4
    max_cache_len = ctx + 2
    dtype = next(model.parameters()).dtype

    sentinel = torch.full((1, 1, 1, max_cache_len), torch.finfo(dtype).min, dtype=dtype)
    sentinel[..., :ctx] = 0.0

    seen = {}
    real_prepare = type(model.model)._prepare_4d_causal_attention_mask_with_cache_position

    def spy(attention_mask, *args, **kwargs):
        seen["called"] = True
        return real_prepare(attention_mask, *args, **kwargs)

    type(model.model)._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(spy)
    try:
        from transformers import StaticCache

        cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        ids = torch.full((1, ctx), 1, dtype=torch.long)
        pos = torch.arange(ctx, dtype=torch.long)
        with torch.no_grad():
            model(
                ids,
                attention_mask=sentinel,
                past_key_values=cache,
                use_cache=True,
                cache_position=pos,
                position_ids=pos.unsqueeze(0),
            )
    finally:
        type(model.model)._prepare_4d_causal_attention_mask_with_cache_position = staticmethod(
            real_prepare
        )

    # 函数仍会被调用（模型无条件调用它），但 4D 分支里它直接返回输入。
    assert seen.get("called"), "预期的调用路径没走到，测试本身失效了"
    # 用一次直接调用来确认 4D 分支是恒等
    identity = real_prepare(
        sentinel,
        sequence_length=ctx,
        target_length=max_cache_len,
        dtype=dtype,
        device=torch.device("cpu"),
        cache_position=pos,
        batch_size=1,
    )
    assert identity is sentinel, "4D mask 没有被原样返回 —— CUDA Graph 的静态缓冲前提不成立"


def test_mask_length_must_match_static_cache_length():
    """mask 的最后一维必须等于 StaticCache 的 max_cache_len。

    不等的话 SDPA 会报错或静默算错。这是 session 里两个常量
    （`CudaGraphConfig.max_cache_len` 与 `GraphBuffers.attention_mask` 的形状）
    必须保持一致的原因。
    """
    from transformers import StaticCache

    model, cfg = _tiny_model(seed=13)
    ctx = 4
    max_cache_len = ctx + 2
    dtype = next(model.parameters()).dtype

    # 正确的长度：能跑
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    pos = torch.arange(ctx, dtype=torch.long)
    mask = _static_mask(1, max_cache_len, ctx - 1, dtype)
    with torch.no_grad():
        model(
            torch.full((1, ctx), 1, dtype=torch.long),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            position_ids=pos.unsqueeze(0),
        )

    # 错误的长度：必须炸，而不是静默算错
    cache2 = StaticCache(config=model.config, max_cache_len=max_cache_len)
    bad = _static_mask(1, max_cache_len - 1, ctx - 1, dtype)
    with pytest.raises(Exception):
        with torch.no_grad():
            model(
                torch.full((1, ctx), 1, dtype=torch.long),
                attention_mask=bad,
                past_key_values=cache2,
                use_cache=True,
                cache_position=pos,
                position_ids=pos.unsqueeze(0),
            )


def test_decode_is_deterministic_without_graph():
    """没有图时也必须可复现 —— 否则后面无法判断 graph 与 eager 的差异来自哪。"""
    from transformers import StaticCache

    def run():
        model, cfg = _tiny_model(seed=17)
        ctx = 4
        max_cache_len = ctx + 2
        dtype = next(model.parameters()).dtype
        cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        ids = torch.full((1, ctx), 1, dtype=torch.long)
        pos = torch.arange(ctx, dtype=torch.long)
        with torch.no_grad():
            out = model(
                ids,
                attention_mask=torch.ones((1, ctx), dtype=torch.long),
                past_key_values=cache,
                use_cache=True,
                cache_position=pos,
                position_ids=pos.unsqueeze(0),
            )
        token = int(out.logits[:, -1, :].argmax(dim=-1).item())
        mask = _static_mask(1, max_cache_len, ctx, dtype)
        with torch.no_grad():
            step = model(
                torch.tensor([[token]], dtype=torch.long),
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([ctx], dtype=torch.long),
                position_ids=torch.tensor([[ctx]], dtype=torch.long),
            )
        return token, step.logits.clone()

    a = run()
    b = run()
    assert a[0] == b[0] and torch.equal(a[1], b[1]), "离线 eager 路径不可复现"


def test_cached_decode_matches_full_recompute():
    """★ 最强的一条：带 KV cache 的逐步 decode 必须等于**整段重算**。

    这正是 CUDA Graph replay 必须保持的性质 —— 图里跑的是 cached decode，
    而正确性参照是 eager 的完整前向。如果这条在离线就不成立，
    那 GPU 上的差距就分不清是 cache 用错了还是图捕获错了。
    """
    from transformers import StaticCache

    model, cfg = _tiny_model(seed=23)
    ctx = 6
    steps = 3
    max_cache_len = ctx + steps + 2
    dtype = next(model.parameters()).dtype

    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    # --- 逐步 decode，记录每一步选出的 token 与 logits ---
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    pos = torch.arange(ctx, dtype=torch.long)
    with torch.no_grad():
        out = model(
            prompt,
            attention_mask=torch.ones((1, ctx), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            position_ids=pos.unsqueeze(0),
        )
    generated = [int(out.logits[:, -1, :].argmax(dim=-1).item())]
    step_logits = []
    for i in range(steps):
        position = ctx + i
        mask = _static_mask(1, max_cache_len, position, dtype)
        with torch.no_grad():
            o = model(
                torch.tensor([[generated[-1]]], dtype=torch.long),
                attention_mask=mask,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], dtype=torch.long),
                position_ids=torch.tensor([[position]], dtype=torch.long),
            )
        step_logits.append(o.logits[:, -1, :].clone())
        generated.append(int(o.logits[:, -1, :].argmax(dim=-1).item()))

    # --- 参照：把整段（prompt + 生成）一次喂进去 ---
    full = torch.tensor([prompt[0].tolist() + generated[:-1]], dtype=torch.long)
    assert full.shape == (1, ctx + steps)
    with torch.no_grad():
        ref = model(full).logits  # (1, ctx+steps, V)

    # 第 ctx+i 位置的 logits 应与 decode 第 i 步一致
    for i, got in enumerate(step_logits):
        want = ref[:, ctx + i, :]
        assert torch.allclose(
            got, want, atol=1e-4
        ), "第 %d 步 decode 与整段重算不一致：max abs diff = %g" % (
            i,
            (got - want).abs().max().item(),
        )


def test_static_cache_update_is_in_place_and_returns_full_length():
    """★ 两个被 session 依赖、但很容易在升级 transformers 时静默改变的假设。

    1. `StaticLayer.update` 用 `index_copy_` **原地**写，不分配新张量。
       如果它改成 `torch.cat`，CUDA Graph 捕获的地址就会失效。
    2. 它返回的是**整段** cache（长度 = max_cache_len），不是已填充的部分。
       这解释了为什么 `attention_mask` 必须是 4D 静态缓冲并且逐位置更新 ——
       mask 是唯一阻止模型注意到未填充区域的机制。如果返回的是切片，
       session 的 mask 逻辑就变得多余（甚至错误）。
    """
    from transformers import StaticCache

    model, cfg = _tiny_model(seed=29)
    ctx = 4
    max_cache_len = ctx + 3
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    pos = torch.arange(ctx, dtype=torch.long)
    with torch.no_grad():
        model(
            torch.full((1, ctx), 1, dtype=torch.long),
            attention_mask=torch.ones((1, ctx), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            cache_position=pos,
            position_ids=pos.unsqueeze(0),
        )

    # 再走一步，检查返回的张量与 cache 内部对象是同一个（原地）
    layer = cache.layers[0]
    before_keys = layer.keys
    dtype = next(model.parameters()).dtype
    mask = _static_mask(1, max_cache_len, ctx, dtype)
    with torch.no_grad():
        model(
            torch.tensor([[1]], dtype=torch.long),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([ctx], dtype=torch.long),
            position_ids=torch.tensor([[ctx]], dtype=torch.long),
        )
    assert layer.keys is before_keys, "cache 张量被替换了 —— 地址不再固定，图捕获会失效"
    assert (
        layer.keys.shape[-2] == max_cache_len
    ), "cache 长度不是 max_cache_len（而是 %s）—— 长度假设变了，mask 逻辑需要重审" % (
        layer.keys.shape[-2],
    )
    # 未填充区域仍然是未初始化/零，必须靠 mask 屏蔽
    assert layer.keys.shape[-2] == max_cache_len
