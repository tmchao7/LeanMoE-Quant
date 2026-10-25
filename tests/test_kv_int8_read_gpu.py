"""test_kv_int8_read_gpu — E6 读路径的 K 门（GPU）。

钉住的判据（EXP-p5-01 记录 §Method 预注册）：
    - int8 split kernel（scale 折叠）vs **fp64 先反量化 dense attention**：
      PARITY_GATE 1e-3（代数等价、结合序不同——不写逐位断言，mxfp4 教训）；
    - cache 写 kernel 的散布：[pos] 槽 = 参考行量化（逐位），其余槽不动；
    - bulk 量化（整段、免 host 长度）与参考一致。
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
torch = pytest.importorskip("torch")

from lemoq.kernels.attn_decode import attn_decode_forward_int8  # noqa: E402
from lemoq.kernels.kv_int8 import (  # noqa: E402
    kv_int8_bulk_quantize,
    kv_int8_cache_write,
)

pytestmark = pytest.mark.gpu

HEADS = 4
HEAD_DIM = 8
SEQ = 32
QK_SCALE = 1.0 / (HEAD_DIM**0.5)


def _requires_gpu():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    try:
        import triton  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        pytest.skip("需要 triton")


def _int8_cache(seed: int = 0, filled: int | None = None):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, 1, HEAD_DIM, generator=g, dtype=torch.float32).to(torch.float16)
    kc = torch.randint(-100, 101, (1, HEADS, SEQ, HEAD_DIM), generator=g, dtype=torch.int8)
    vc = torch.randint(-100, 101, (1, HEADS, SEQ, HEAD_DIM), generator=g, dtype=torch.int8)
    ks = (torch.rand(1, HEADS, SEQ, generator=g, dtype=torch.float32) * 0.04 + 0.01).to(
        torch.float16
    )
    vs = (torch.rand(1, HEADS, SEQ, generator=g, dtype=torch.float32) * 0.04 + 0.01).to(
        torch.float16
    )
    n = filled if filled is not None else SEQ
    # 未填充槽位置零（真实布局由 bulk/写路径保证；oracle 只读 [0, filled)）
    kc[0, :, n:] = 0
    vc[0, :, n:] = 0
    ks[0, :, n:] = 1.0
    vs[0, :, n:] = 1.0
    return (
        q.cuda(),
        kc.cuda(),
        ks.cuda(),
        vc.cuda(),
        vs.cuda(),
        n,
    )


def _fp64_dequant_attention(q, kc, ks, vc, vs, n):
    """先反量化（fp64）再标准 attention——折叠口径的参考。"""
    k = kc[0].double() * ks[0].double().unsqueeze(-1)
    v = vc[0].double() * vs[0].double().unsqueeze(-1)
    p = torch.softmax(q[0].double() @ k.transpose(-1, -2) * QK_SCALE, dim=-1)
    return (p @ v).to(torch.float32).unsqueeze(0)


@pytest.mark.gpu
def test_attn_int8_kernel_matches_fp64_dequant_oracle():
    """K 门 ⑤：折叠 kernel vs fp64 反量化 oracle —— PARITY_GATE 1e-3。"""
    assert _requires_gpu()
    for seed in (0, 1, 2):
        q, kc, ks, vc, vs, n = _int8_cache(seed=seed)
        filled = torch.tensor([n], dtype=torch.int32, device=q.device)
        out = attn_decode_forward_int8(q, kc, ks, vc, vs, filled)
        ref = _fp64_dequant_attention(q, kc, ks, vc, vs, n)
        err = (out.float() - ref).abs().max()
        assert err <= 1e-3, f"seed={seed} parity 超界：{err:.4e}"


@pytest.mark.gpu
def test_attn_int8_kernel_partial_fill():
    """K 门 ⑥：filled < S 时只读前缀（部分填充与 oracle 一致）。"""
    assert _requires_gpu()
    n = 17
    q, kc, ks, vc, vs, _ = _int8_cache(seed=3, filled=n)
    filled = torch.tensor([n], dtype=torch.int32, device=q.device)
    out = attn_decode_forward_int8(q, kc, ks, vc, vs, filled)
    ref = _fp64_dequant_attention(q, kc, ks, vc, vs, n)
    err = (out.float() - ref).abs().max()
    assert err <= 1e-3, f"部分填充 parity 超界：{err:.4e}"


@pytest.mark.gpu
def test_cache_write_scatters_at_pos():
    """K 门 ⑦：cache 写 kernel 把新行量化散布到 [pos]，其余槽逐位不动。"""
    assert _requires_gpu()
    g = torch.Generator().manual_seed(4)
    S, D = 16, HEAD_DIM
    kc = torch.randint(-100, 101, (1, HEADS, S, D), generator=g, dtype=torch.int8).cuda()
    ks = (
        (torch.rand(1, HEADS, S, generator=g, dtype=torch.float32) * 0.04 + 0.01)
        .to(torch.float16)
        .cuda()
    )
    bufs = {"kc": kc.clone(), "ks": ks.clone(), "vc": kc.clone(), "vs": ks.clone()}
    new_k = torch.randn(1, HEADS, 1, D, generator=g, dtype=torch.float32).to(torch.float16).cuda()
    new_v = torch.randn(1, HEADS, 1, D, generator=g, dtype=torch.float32).to(torch.float16).cuda()
    pos = torch.tensor([5], dtype=torch.int64, device=kc.device)
    kv_int8_cache_write(bufs, new_k, new_v, pos)
    # [pos] 槽 = 参考行量化（逐位）
    amax_k = new_k[0, :, 0].float().abs().amax(dim=-1, keepdim=True)
    s_ref = torch.where(amax_k > 0, amax_k / 127.0, torch.ones_like(amax_k)).to(torch.float16)
    q_ref = torch.round(new_k[0, :, 0].float() / s_ref.float()).clamp(-127, 127).to(torch.int8)
    assert torch.equal(bufs["kc"][0, :, 5, :], q_ref)
    assert torch.equal(bufs["ks"][0, :, 5], s_ref)
    # 其余槽逐位不动
    mask = torch.ones(S, dtype=torch.bool, device=kc.device)
    mask[5] = False
    assert torch.equal(bufs["kc"][0][:, mask, :], kc[0][:, mask, :])
    assert torch.equal(bufs["ks"][0][:, mask], ks[0][:, mask])


@pytest.mark.gpu
def test_bulk_quantize_matches_reference_on_device():
    """K 门 ⑧：整段 bulk 量化（免 host 长度）与逐位置参考一致。"""
    assert _requires_gpu()
    x = torch.randn(1, HEADS, SEQ, HEAD_DIM, dtype=torch.float32).to(torch.float16)
    bufs = kv_int8_bulk_quantize(x, x, filled=None)
    amax = x[0].float().abs().amax(dim=-1, keepdim=True)
    s_ref = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax)).to(torch.float16)
    assert torch.equal(bufs["ks"][0], s_ref.squeeze(-1))
    q_ref = torch.round(x[0].float() / s_ref.float()).clamp(-127, 127).to(torch.int8)
    assert torch.equal(bufs["kc"][0], q_ref)
