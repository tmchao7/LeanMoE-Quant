"""test_kv_int8 — E6 参考数学的 CPU 测试（kernel 的 K 门对照物，EXP-p5-01）。

钉住的是**预注册的舍入链与守卫语义**（记录 §Method）：
    - amax/127、round half-to-even、clamp ±127、全零行 scale=1.0；
    - roundtrip 逐元素 |err| ≤ 0.6·scale（s/2 + fp16 scale 舍入的余量）；
    - per-channel 变体（v2 阶梯的 K 粒度）同性质；
    - 守卫把 CPU/坏形状/dtype 错配**显式拒掉**（F4 教训：缺 is_cuda 检查炸过）；
    - 折叠代数恒等式（读路径"内层零解包"的依据）。
kernel（10-25 窗）的逐位检验在 `tests/test_kv_int8_gpu.py`（骨架已预置）。
"""

from __future__ import annotations

from conftest import requires_torch

pytestmark = requires_torch

import torch  # noqa: E402

from lemoq.kernels.kv_int8 import (  # noqa: E402
    dequant_int8,
    fold_scores_per_token_reference,
    kv_int8_rows_applicable,
    quantize_per_channel_int8,
    quantize_per_token_int8,
)


def _rows(h: int = 4, d: int = 16, seed: int = 0, zeros: bool = False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, h, 1, d, generator=g, dtype=torch.float32).to(torch.float16)
    if zeros:
        x = torch.zeros_like(x)
    return x


def test_per_token_scale_formula_and_zero_row_convention():
    x = _rows()
    codes, scale = quantize_per_token_int8(x)
    # scale 形状随输入 (1,H,1,D) → (1,H,1)；参考值同一转换链 ⇒ 逐位一致
    expected = (x.float().abs().amax(dim=-1) / 127.0).to(torch.float16)
    assert scale.shape == expected.shape
    assert torch.equal(scale, expected)
    # 全零行：scale=1.0、codes=0（mxfp4 零块先例，避开除零）
    z = _rows(zeros=True)
    zc, zs = quantize_per_token_int8(z)
    assert (zs == 1.0).all() and (zc == 0).all()
    # 非零行不会有 0 scale
    assert (scale > 0).all()


def test_per_token_codes_range_and_dtype():
    x = _rows(seed=1)
    codes, scale = quantize_per_token_int8(x)
    assert codes.dtype == torch.int8 and scale.dtype == torch.float16
    assert codes.min() >= -127 and codes.max() <= 127
    # 幅值最大的元素必然打到 ±127（scale 就是按它定的；x/scale 对 127 的偏离 ~1e-5，
    # round 后必为 127——随机行 amax>0，无零行逃逸）
    amax_pos = x.float().abs().argmax(dim=-1, keepdim=True)
    lead = torch.gather(codes.int(), -1, amax_pos.to(torch.int64).expand_as(codes))
    assert (lead.abs() == 127).all()


def test_per_token_roundtrip_error_bound():
    for seed in range(4):
        x = _rows(seed=seed)
        codes, scale = quantize_per_token_int8(x)
        rec = dequant_int8(codes, scale.unsqueeze(-1))
        err = (rec - x.float()).abs()
        # 量化上界 s/2 + fp16 scale 舍入项（127·2^-11·s ≈ 0.062s）→ 0.6s 余量充足
        bound = 0.6 * scale.float().unsqueeze(-1)
        assert (err <= bound).all(), f"seed={seed} 超界"


def test_per_channel_variant_shapes_and_bound():
    # (T, D) 的 token 块：scale 沿通道（D），对 T 个 token 共享
    g = torch.Generator().manual_seed(2)
    x = torch.randn(32, 16, generator=g, dtype=torch.float32).to(torch.float16)
    codes, scale = quantize_per_channel_int8(x)
    assert codes.shape == x.shape and scale.shape == (16,)
    assert scale.dtype == torch.float16
    amax = x.float().abs().amax(dim=0)
    assert torch.allclose(scale.float(), (amax / 127.0).to(torch.float16).float(), rtol=0, atol=0)
    rec = dequant_int8(codes, scale)
    err = (rec - x.float()).abs()
    assert (err <= 0.6 * scale.float().unsqueeze(0)).all()
    # 全零列
    x2 = x.clone()
    x2[:, 3] = 0.0
    _, s2 = quantize_per_channel_int8(x2)
    assert s2[3] == 1.0


def test_fold_scores_identity_algebraic_not_bitwise():
    """折叠恒等式：s_t·(q·k_t) == q·(k_t·s_t)——**代数**恒等，浮点下**不逐位**。

    ★ 这正是记录预注册"读侧 parity 用 fp64 oracle + PARITY_GATE 1e-3、不写逐位
    断言"的原因（结合序不同；mxfp4 的教训）。这里用紧 allclose 钉代数性。
    """
    g = torch.Generator().manual_seed(3)
    q = torch.randn(8, 16, generator=g, dtype=torch.float32)
    k = torch.randn(8, 16, generator=g, dtype=torch.float32)
    s = torch.rand(8, generator=g, dtype=torch.float32) + 0.5
    folded_after = fold_scores_per_token_reference(q @ k.T, s)  # 先点积后折（kernel 的做法）
    folded_before = q @ (k * s.unsqueeze(-1)).T  # 先折 k 再点积（先反量化的等价形态）
    # 代数恒等：偏差以**矩阵尺度**衡量（逐元素 rtol 会被相消行打爆——这正是读侧
    # parity 要用 fp64 oracle + 矩阵级容差的原因）
    rel = (folded_after - folded_before).abs().max() / folded_after.abs().max()
    assert rel < 1e-5, f"折叠代数偏差过大：{rel:.2e}"


def test_guard_rejects_cpu_and_bad_shapes():
    good = _rows()  # (1,4,1,16) fp16, CPU
    # CPU 张量一律回退（F4 教训）
    assert kv_int8_rows_applicable(good, good.clone()) is False
    # dtype 错配
    assert kv_int8_rows_applicable(good, good.float()) is False
    # 形状错配
    assert kv_int8_rows_applicable(good, good[:, :2]) is False
    assert kv_int8_rows_applicable(good.squeeze(0), good.squeeze(0)) is False
    # D 非 2 幂
    bad_d = torch.randn(1, 4, 1, 12, dtype=torch.float16)
    assert kv_int8_rows_applicable(bad_d, bad_d.clone()) is False
    # 非连续
    t = torch.randn(1, 4, 2, 16, dtype=torch.float16)[:, :, :1]
    assert not t.is_contiguous()
    assert kv_int8_rows_applicable(t, t.clone()) is False


def test_fold_attention_algebra_end_to_end():
    """E6 读路径的**代数**验证（torch 级，CPU）：折叠口径 vs 先反量化口径。

    kernel 的逐位/parity 检验在 GPU（tests/test_kv_int8_read_gpu.py）；这里钉的是
    读路径依赖的恒等式本身——含"V scale 不得折进分母"这个刚才差点写错的点。
    """
    g = torch.Generator().manual_seed(11)
    H, S, D = 2, 24, 16
    q = torch.randn(1, H, 1, D, generator=g, dtype=torch.float32)
    kc = torch.randint(-127, 128, (1, H, S, D), generator=g, dtype=torch.int8)
    vc = torch.randint(-127, 128, (1, H, S, D), generator=g, dtype=torch.int8)
    ks = torch.rand(1, H, S, generator=g, dtype=torch.float32) + 0.5
    vs = torch.rand(1, H, S, generator=g, dtype=torch.float32) + 0.5
    qk_scale = 1.0 / (D**0.5)

    # (a) 先反量化（真值 k/v = c·scale）再标准 attention
    k_true = kc.float() * ks[..., None]
    v_true = vc.float() * vs[..., None]
    p = torch.softmax(q.float() @ k_true.transpose(-1, -2) * qk_scale, dim=-1)
    out_dequant = p @ v_true

    # (b) 折叠口径（kernel 的代数）：score = (q·c)·scale·ks；acc = Σ(p·vs)·c；分母 = Σp
    # ★ ks/vs 按**每头一行**对齐（unsqueeze(-2)）——直接乘会广播成 (1,H,H,S) 串头；
    #   V 的加权求和同样要把 vc 对齐成 (1,H,1,S,D)（4 维 @ 4 维在 batch 维串头）
    scores = (q.float() @ kc.float().transpose(-1, -2)) * qk_scale * ks.unsqueeze(-2)
    p2 = torch.softmax(scores, dim=-1)
    w = p2 * vs.unsqueeze(-2)  # (1,H,1,S)：attention 权重 × V scale（只折分子）
    out_fold = (w.unsqueeze(2) @ vc.float().unsqueeze(2)).squeeze(2)  # vc 也对齐到 (1,H,1,S,D)

    rel = (out_dequant - out_fold).abs().max() / out_dequant.abs().max()
    assert rel < 1e-5, f"折叠代数与反量化口径偏差过大：{rel:.2e}"
    # 反证：若把 vs 也折进分母（错误写法），输出会系统性偏离——钉住这个区别
    wrong_l = (p2 * vs.unsqueeze(-2)).sum(dim=-1, keepdim=True)
    out_wrong = (w.unsqueeze(2) @ vc.float()).squeeze(2) / wrong_l
    rel_wrong = (out_dequant - out_wrong).abs().max() / out_dequant.abs().max()
    assert rel_wrong > 1e-3, "错误口径居然也对——这条反证断言失去意义"


def test_bulk_quantize_whole_cache_and_zero_slots():
    from lemoq.kernels.kv_int8 import dequant_int8, kv_int8_bulk_quantize

    g = torch.Generator().manual_seed(12)
    x = torch.randn(1, 2, 8, 16, generator=g, dtype=torch.float32).to(torch.float16)
    x[0, :, 6:, :] = 0.0  # 模拟未填充槽位（零行）
    bufs = kv_int8_bulk_quantize(x, x, filled=None)
    assert bufs["kc"].shape == (1, 2, 8, 16) and bufs["ks"].shape == (1, 2, 8)
    # 已填充行：roundtrip 有界
    rec = dequant_int8(bufs["kc"][0, :, :6], bufs["ks"][0, :, :6].unsqueeze(-1))
    err = (rec - x[0, :, :6].float()).abs()
    assert (err <= 0.6 * bufs["ks"][0, :, :6].float().unsqueeze(-1)).all()
    # 零行：scale=1.0、codes=0
    assert (bufs["ks"][0, :, 6:] == 1.0).all() and (bufs["kc"][0, :, 6:] == 0).all()


def test_kv_int8_bufs_staleness_is_cache_identity():
    """★ 跨会话污染守卫（2026-10-25 定稿语义）：按 cache **对象身份**判失效，不按 prefill。

    - prefill/decode 的回退路径**不**碰 buffer：同 session 重复 prefill = 同 prompt 重灌，
      数值等价 ⇒ 复用（按 prefill 作废会让 `capture()` 二次复位后的懒初始化撞上
      "不许在 capture 里分配"的守卫——A/B run1 全 blocked 的实测教训）；
    - 换 cache 对象（probe 换窗/换会话）⇒ `_kv_int8_bufs_stale` 判旧 ⇒ 重建；
    - 身份锚死了（session 被 GC）⇒ 判旧 ⇒ 重建（weakref 语义）。
    """
    import weakref
    from types import SimpleNamespace  # noqa: PLC0415

    from lemoq.kernels.attn_decode import (  # noqa: PLC0415
        _decode_forward_impl,
        _kv_int8_bufs_stale,
    )

    def _self_with(bufs):
        calls = []

        def fake_original(*a, **k):
            calls.append(len(a))
            return ("out", None)

        return (
            SimpleNamespace(_lemoq_original_forward=fake_original, _lemoq_kv_int8_bufs=bufs),
            calls,
        )

    # ① 回退路径不碰 buffer：prefill（q_len=4）与 DynamicCache decode（filled=None）都原样保留
    mod, calls = _self_with({"kc": "keep"})
    out = _decode_forward_impl(
        mod,
        torch.zeros(1, 4, 8),
        past_key_values=object(),
        position_embeddings=(torch.zeros(1), torch.zeros(1)),
    )
    assert out == ("out", None) and calls == [1]
    assert mod._lemoq_kv_int8_bufs == {"kc": "keep"}, "prefill 回退不应作废镜像（capture 安全）"

    mod2, _ = _self_with({"kc": "keep"})
    _decode_forward_impl(
        mod2,
        torch.zeros(1, 1, 8),
        past_key_values=object(),  # 无 _lemoq_filled_len
        position_embeddings=(torch.zeros(1), torch.zeros(1)),
    )
    assert mod2._lemoq_kv_int8_bufs == {"kc": "keep"}, "decode 回退不应作废镜像"

    # ② 身份矩阵：没 buffer ⇒ 旧；没锚（legacy）⇒ 旧；同对象 ⇒ 新；换对象 ⇒ 旧；锚死 ⇒ 旧
    #   （cache 替身用普通类：object()/SimpleNamespace 都不支持 weakref）
    class _Cache:
        pass

    cache_a, cache_b = _Cache(), _Cache()
    assert _kv_int8_bufs_stale(SimpleNamespace(_lemoq_kv_int8_bufs=None), cache_a)
    assert _kv_int8_bufs_stale(SimpleNamespace(_lemoq_kv_int8_bufs={"kc": 1}), cache_a)
    mod_live = SimpleNamespace(
        _lemoq_kv_int8_bufs={"kc": 1}, _lemoq_kv_int8_src_cache_ref=weakref.ref(cache_a)
    )
    assert not _kv_int8_bufs_stale(
        mod_live, cache_a
    ), "同 cache 对象必须复用（runner reset/capture）"
    assert _kv_int8_bufs_stale(mod_live, cache_b), "换 cache 对象必须重建（probe 换窗）"
    mod_dead = SimpleNamespace(_lemoq_kv_int8_bufs={"kc": 1})
    anchor = _Cache()
    mod_dead._lemoq_kv_int8_src_cache_ref = weakref.ref(anchor)
    del anchor
    assert _kv_int8_bufs_stale(mod_dead, _Cache()), "身份锚死（session GC）必须重建"
