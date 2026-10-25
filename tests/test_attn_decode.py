"""test_attn_decode — decode attention（split-KV + LSE 合并）的离线回归，**全部在 CPU 上跑**。

## 为什么这些测试必须先在这里跑，而不是上机再调

kernel 有两大类错误，代价完全不同：

    公式/索引错   分片区间算错、运行最大值/运行和的顺序错、合并时权重用错、
                  长度读成 max_cache_len —— 会算出一个**看起来合理**的错结果
    Triton 实现错 偏移/掩码/降级写错 —— 需要真机

第一类占绝大多数，而且**不需要 GPU 就能全部抓出来**。所以这里做四件事：

1. 用 fp64 稠密 attention 钉住 split-KV 的**数学**（分片数、块大小都不许改变答案）；
2. 把 host 侧的 grid / 缓冲区形状做成 `launch_plan`（纯 Python），用**手算值**钉住 ——
   CUDA Graph 只约束"形状静态"，不约束"形状对不对"；
3. 把 **K4（换长度）的判据本身**测掉，包括它的**判别力**：一个不可能失败的测试比
   没有测试更糟，因为它会被引用；
4. 用 tiny OLMoE + StaticCache 在 CPU 上把**集成**（前后处理复制 + 长度来源 + 退回原路）
   测掉 —— 这是在 Mac 上能覆盖到的最接近真机的一层。

真 kernel 的 K1（vs fp64 oracle）与 K4 在 `-m gpu` 标记下跑（见文件末尾）。
"""

from __future__ import annotations

import math
import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lemoq.kernels.attn_decode import (  # noqa: E402
    ATTN_CONFIGS,
    LENGTH_CHANGE_GATE,
    LOG2E,
    PARITY_GATE,
    attention_cores_replaced,
    attn_decode_available,
    attn_decode_forward,
    decode_call_count,
    dense_attention_oracle,
    judge_length_change,
    launch_plan,
    merge_splits_reference,
    replace_attention_cores,
    resolve_num_splits,
    restore_attention_cores,
    split_kv_oracle,
    split_kv_partials_reference,
)

HEADS = 16
HEAD_DIM = 128


def _case(seq=37, filled=33, seed=0, dtype=torch.float16, heads=HEADS, head_dim=HEAD_DIM):
    """随机 q/k/v。**尾部的 v 故意给大值**：这样"读错长度"不可能被噪声掩盖。"""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(1, heads, 1, head_dim, generator=g) * 0.1).to(dtype)
    k = (torch.randn(1, heads, seq, head_dim, generator=g) * 0.1).to(dtype)
    v = (torch.randn(1, heads, seq, head_dim, generator=g) * 0.1).to(dtype)
    return q, k, v


# ---------------------------------------------------------------------------
# 1) 数学：split-KV 必须等于稠密 fp64，且与分片方式无关
# ---------------------------------------------------------------------------
def test_split_oracle_matches_dense_fp64():
    """split-KV（fp32）与 fp64 稠密 attention 的差必须落在 fp16 输入的量级内。"""
    q, k, v = _case()
    want = dense_attention_oracle(q, k, v, 33)
    got = split_kv_oracle(q, k, v, 33, num_splits=8)
    err = float((got.double() - want).abs().max())
    assert err < PARITY_GATE["tol_atol"], "split-KV 与 fp64 oracle 的差 %.3e 超过门" % err


@pytest.mark.parametrize("num_splits", [1, 2, 3, 8, 17, 64])
def test_num_splits_does_not_change_the_answer(num_splits):
    """★ LSE 合并的核心测试：分片数不许改变答案（只允许改变累加顺序）。

    这里用 fp32 而不是 fp16，量到的**只有合并公式本身**。若合并漏了
    `exp2(m_j − M)` 权重或除错了 `Σ l_j`，1 片与 64 片的差会是 O(1)，不是 1e-6。
    """
    q, k, v = _case()
    want = split_kv_oracle(q, k, v, 33, num_splits=1)
    got = split_kv_oracle(q, k, v, 33, num_splits=num_splits)
    assert not torch.isnan(got).any(), "num_splits=%d 出现 NaN" % num_splits
    err = float((got - want).abs().max())
    assert err < 1e-5, "num_splits=%d 的答案与单分片差 %.3e" % (num_splits, err)


def test_more_splits_than_tokens_leaves_no_nan():
    """空分片（filled < num_splits）必须写出 m=-inf/l=0，并且**不产生 NaN**。

    `exp2(-inf − -inf)` 是 NaN，所以"所有分片都空"的退化情形必须显式挡住 ——
    这条测试就是那个挡板的回归。
    """
    q, k, v = _case(seq=8, filled=5)
    got = split_kv_oracle(q, k, v, 5, num_splits=17)
    want = dense_attention_oracle(q, k, v, 5)
    assert not torch.isnan(got).any()
    assert float((got.double() - want).abs().max()) < PARITY_GATE["tol_atol"]


def test_filled_zero_is_zero_not_nan():
    q, k, v = _case(seq=8, filled=5)
    got = split_kv_oracle(q, k, v, 0, num_splits=4)
    assert torch.equal(got, torch.zeros_like(got))


def test_partials_shape_and_merge_are_consistent():
    """`m/l/acc` 的形状契约 + 合并只依赖这三样（分开测，便于定位错误）。"""
    q, k, v = _case(seq=17, filled=17)
    m, l, acc = split_kv_partials_reference(q, k, v, 17, num_splits=5)
    assert tuple(m.shape) == (HEADS, 5)
    assert tuple(l.shape) == (HEADS, 5)
    assert tuple(acc.shape) == (HEADS, 5, HEAD_DIM)
    merged = merge_splits_reference(m, l, acc)
    direct = split_kv_oracle(q, k, v, 17, num_splits=5)
    assert torch.allclose(merged, direct, atol=0, rtol=0)


@pytest.mark.parametrize("block_n", [16, 32, 128])
def test_block_size_does_not_change_the_answer(block_n):
    """块大小是**性能**参数，不许影响数学（只影响累加顺序）。"""
    q, k, v = _case(seq=100, filled=97)
    a = split_kv_oracle(q, k, v, 97, num_splits=7, block_n=128)
    b = split_kv_oracle(q, k, v, 97, num_splits=7, block_n=block_n)
    assert float((a - b).abs().max()) < 1e-5


def test_padding_beyond_filled_is_completely_ignored():
    """[filled, seq) 里的任何值都不得影响结果 —— 这是"不要 mask"能成立的**全部前提**。

    做法：跑两次，第二次把尾部槽位换成 NaN。若实现读到尾部，答案必然变 NaN。
    """
    q, k, v = _case(seq=64, filled=33)
    base = split_kv_oracle(q, k, v, 33, num_splits=8)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 33:, :] = float("nan")
    v2[:, :, 33:, :] = float("nan")
    got = split_kv_oracle(q, k2, v2, 33, num_splits=8)
    assert torch.equal(base, got), "尾部未被填充的槽位影响了结果 —— mask 不能去掉"


# ---------------------------------------------------------------------------
# 2) host 侧算术（grid / 缓冲区形状）—— 用**手算值**钉住
# ---------------------------------------------------------------------------
def test_launch_plan_hand_computed():
    plan = launch_plan(heads=16, head_dim=128, num_splits=10, block_n=128)
    assert plan["kernel_count"] == 2
    assert plan["split"]["grid"] == (10, 16)
    assert plan["split"]["m_shape"] == (16, 10)
    assert plan["split"]["l_shape"] == (16, 10)
    assert plan["split"]["acc_shape"] == (16, 10, 128)
    assert plan["combine"]["grid"] == (16,)
    assert plan["combine"]["out_shape"] == (1, 16, 1, 128)


def test_launch_plan_rejects_illegal_shapes():
    with pytest.raises(ValueError):
        launch_plan(heads=16, head_dim=128, num_splits=10, block_n=100)  # 不是 2 的幂
    with pytest.raises(ValueError):
        launch_plan(heads=16, head_dim=100, num_splits=10, block_n=128)  # head_dim 不是 2 的幂
    with pytest.raises(ValueError):
        launch_plan(heads=16, head_dim=128, num_splits=0, block_n=128)


def test_resolve_num_splits_defaults_and_overrides():
    assert resolve_num_splits(7, heads=16) == 7
    # `None` = ceil(SM / heads)，SM 数**实测**自当前设备；没有 CUDA 时退回 1。
    # ★ 这条断言必须跟着设备走：第一版写成"没有 CUDA 时 == 1"，于是它在 Mac 上绿、
    #   在 5090 上是红的（11 != 1）—— 也就是"只在没有 GPU 的机器上验证过"。
    #   默认值既不许写死 170，也不许在没有设备时假装知道 SM 数。
    if torch.cuda.is_available():
        sm = int(torch.cuda.get_device_properties(0).multi_processor_count)
        expected = max(1, -(-sm // 16))
        assert resolve_num_splits(None, heads=16) == expected
    else:
        assert resolve_num_splits(None, heads=16) == 1
    with pytest.raises(ValueError):
        resolve_num_splits(0, heads=16)


def test_gates_are_frozen():
    """★ 门在**上机之前**冻结：任何放宽都必须是一个可见的测试改动 + 一个 commit。"""
    assert PARITY_GATE["tol_atol"] == 1e-3
    assert PARITY_GATE["tol_rtol"] == 1e-3
    assert PARITY_GATE["tol_slack"] == 1.5
    assert LENGTH_CHANGE_GATE["correct_err_max"] == 1e-2
    assert LENGTH_CHANGE_GATE["discriminant_min"] == 1.0
    assert ATTN_CONFIGS["num_splits"] is None  # None = ceil(SM/heads)，不是写死的常数


def test_log2e_is_the_exp2_scale():
    assert abs(LOG2E - 1.0 / math.log(2.0)) < 1e-15


# ---------------------------------------------------------------------------
# 3) K4 的判据本身（含判别力）
# ---------------------------------------------------------------------------
def test_length_change_and_its_discriminating_power():
    """换长度的判据必须能**失败**：陈旧答案离正确答案远，正确答案误差小。"""
    q, k, v = _case(seq=70, filled=33)
    v = v.clone()
    v[:, :, 33:, :] = 20.0  # 判别力保障：忽略新长度的实现会漏掉这一大块贡献
    want = dense_attention_oracle(q, k, v, 67).abs().max().item()  # 量级参考
    stale = split_kv_oracle(q, k, v, 33, num_splits=8)
    correct = split_kv_oracle(q, k, v, 67, num_splits=8)
    true = dense_attention_oracle(q, k, v, 67)

    correct_err = float((correct.double() - true).abs().max())
    stale_err = float((stale.double() - true).abs().max())
    verdict = judge_length_change(correct_err=correct_err, stale_err=stale_err)
    assert verdict["discriminating"] is True
    assert verdict["verdict"] == "PASS"
    assert stale_err > 1.0, "判别力不足：陈旧答案只差 %.3e（want 量级 %.3f）" % (stale_err, want)

    # 反向：把"陈旧答案"当答案报上来，判据必须判 FAIL，而不是放过
    bad = judge_length_change(correct_err=stale_err, stale_err=stale_err)
    assert bad["verdict"] == "FAIL"


def test_length_change_judgement_is_inconclusive_without_discriminant():
    """没有判别力时必须判 INCONCLUSIVE —— 这是"永不失败的测试"的防线。"""
    verdict = judge_length_change(correct_err=1e-6, stale_err=1e-6)
    assert verdict["verdict"] == "INCONCLUSIVE"
    assert verdict["discriminating"] is False


# ---------------------------------------------------------------------------
# 4) host 管线与输出契约（oracle backend，不需要 triton）
# ---------------------------------------------------------------------------
def test_oracle_backend_output_contract():
    q, k, v = _case()
    out = attn_decode_forward(q, k, v, 33, backend="oracle", config={"num_splits": 4})
    assert tuple(out.shape) == (1, HEADS, 1, HEAD_DIM)
    assert out.dtype == q.dtype


def test_oracle_backend_accepts_a_device_length_tensor():
    """设备端 scalar 是图内唯一允许的长度来源，oracle 路径也要能吃它。"""
    q, k, v = _case()
    filled = torch.tensor([33], dtype=torch.int32)
    out = attn_decode_forward(q, k, v, filled, backend="oracle", config={"num_splits": 4})
    want = dense_attention_oracle(q, k, v, 33).to(q.dtype).reshape(out.shape)
    assert torch.allclose(out, want, atol=PARITY_GATE["tol_atol"], rtol=PARITY_GATE["tol_rtol"])


def test_forward_rejects_wrong_shapes_and_dtypes():
    q, k, v = _case()
    q2 = torch.cat([q, q], dim=2)  # q_len = 2（prefill 形状）
    with pytest.raises(NotImplementedError):  # q_len != 1
        attn_decode_forward(q2, k, v, 33, backend="oracle")
    with pytest.raises(NotImplementedError):  # GQA
        attn_decode_forward(q, k[:, :8], v[:, :8], 33, backend="oracle")
    with pytest.raises(ValueError):  # filled 越界
        attn_decode_forward(q, k, v, 999, backend="oracle")
    with pytest.raises(ValueError):  # 最后一维不连续
        attn_decode_forward(q, k.transpose(2, 3), v.transpose(2, 3), 33, backend="oracle")
    with pytest.raises(ValueError):  # dtype 不一致
        attn_decode_forward(q.float(), k, v, 33, backend="oracle")


def test_kernel_backend_reports_missing_triton_loudly():
    """Mac 上没有 triton：`backend='auto'` 必须**报错**，不能静默换慢路径。"""
    if attn_decode_available():
        pytest.skip("本机有 triton，这条测试针对无 triton 环境")
    q, k, v = _case()
    with pytest.raises(RuntimeError):
        attn_decode_forward(q, k, v, 33, backend="auto")


# ---------------------------------------------------------------------------
# 5) 集成：tiny OLMoE + StaticCache（CPU）
# ---------------------------------------------------------------------------
def _tiny_model():
    transformers = pytest.importorskip("transformers")
    from transformers import OlmoeConfig

    torch.manual_seed(0)
    config = OlmoeConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = transformers.OlmoeForCausalLM(config).eval().float()
    return model


def _prefill(model, ctx=8, max_cache_len=16):
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    ids = torch.randint(0, 128, (1, ctx))
    with torch.no_grad():
        out = model(
            ids, attention_mask=torch.ones((1, ctx), dtype=torch.long), past_key_values=cache
        )
    return cache, int(out.logits[:, -1:, :].argmax(dim=-1).item()), ids


def _decode_step(model, cache, token, position, max_cache_len=16):
    with torch.no_grad():
        return model(
            torch.tensor([[token]], dtype=torch.long),
            attention_mask=torch.ones((1, max_cache_len), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position], dtype=torch.long),
            position_ids=torch.tensor([[position]], dtype=torch.long),
        ).logits


def test_replaced_attention_matches_hf_at_q_len_1(monkeypatch):
    """★ 集成的核心测试：替换后**只换 core**，其余前后处理必须与 HF 等价。

    用 fp32 + oracle backend（Mac 上无 triton），所以量到的差异只来自
    "core 的数学"与"有没有把 HF 的前后处理写错"。
    ★ E6 采纳（2026-10-25）后 kv_int8 默认 True，而 oracle 后端不带 int8 读路径——
      本测试量的是 fp16 core 数学，显式 env=0（N6：钉机制不钉默认值）。
    """
    monkeypatch.setenv("LEMOQ_KV_INT8", "0")
    model = _tiny_model()
    cache, token, _ = _prefill(model)
    position = 8
    want = _decode_step(model, cache, token, position).clone()

    assert replace_attention_cores(model, backend="oracle") == 2  # 2 层
    # 设备端长度张量：cache 上挂 filled_len（与 StaticCacheDecodeSession 同机制）
    cache._lemoq_filled_len = torch.tensor([position + 1], dtype=torch.int32)
    got = _decode_step(model, cache, token, position).clone()

    assert decode_call_count(model) == 2, "新 core 没有被走到（测试会变成空跑）"
    err = float((got - want).abs().max())
    assert err < 1e-4, "替换后与 HF eager 的差 %.3e" % err
    assert restore_attention_cores(model) == 2


def test_prefill_takes_the_original_path_unchanged():
    """prefill（q_len = ctx > 1）必须逐位等于**未替换**的实现 —— 本任务只作用于 T=1。"""
    model = _tiny_model()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        want = model(ids).logits.clone()
    replace_attention_cores(model, backend="oracle")
    with torch.no_grad():
        got = model(ids).logits.clone()
    assert torch.equal(got, want), "prefill 被改动了 —— §5.7 的边界被破坏"
    assert decode_call_count(model) == 0, "prefill 不应该走 decode core"


def test_decode_without_filled_len_falls_back_to_hf():
    """动态 cache（informational 参照）上没有 filled_len → 必须退回 HF 原路。

    这条保证 runner 里"与 HF 动态 cache 对比"那一项量的仍是 HF 的数值，不是我们的
    （否则跨实现差异会被算成两遍，口径就乱了）。
    """
    from transformers import DynamicCache

    model = _tiny_model()
    ids = torch.randint(0, 128, (1, 8))
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        want = model(torch.tensor([[1]]), past_key_values=past, use_cache=True).logits.clone()

    replace_attention_cores(model, backend="oracle")
    dynamic = DynamicCache()
    with torch.no_grad():
        first = model(ids, past_key_values=dynamic, use_cache=True)
        got = model(
            torch.tensor([[1]]), past_key_values=first.past_key_values, use_cache=True
        ).logits
    assert torch.equal(got, want), "没有 filled_len 时没有退回 HF 原路"
    assert decode_call_count(model) == 0
    assert attention_cores_replaced(model) == 2


def test_replaced_attention_respects_the_length_tensor(monkeypatch):
    """长度是**设备张量**：改它就能改答案 —— 集成层不许把长度烘进 shape。

    ★ 同上：oracle 后端 + kv_int8 显式 env=0（E6 采纳后默认 True）。
    """
    monkeypatch.setenv("LEMOQ_KV_INT8", "0")
    model = _tiny_model()
    cache, token, _ = _prefill(model)
    attn = model.model.layers[0].self_attn
    assert attn is not None
    replace_attention_cores(model, backend="oracle")

    cache._lemoq_filled_len = torch.tensor([9], dtype=torch.int32)
    long = _decode_step(model, cache, token, 8).clone()
    cache._lemoq_filled_len = torch.tensor([3], dtype=torch.int32)
    short = _decode_step(model, cache, token, 8).clone()
    # ★ 判别力：两次必须不同（否则这个测试证明不了任何事情）
    assert float((long - short).abs().max()) > 1e-3, "长度张量没有影响答案，测试无判别力"
    # 与"直接在 fp32 里只读前 3 个位置"的 HF 路径对齐
    cache._lemoq_filled_len = torch.tensor([9], dtype=torch.int32)
    restore_attention_cores(model)
    cache._lemoq_filled_len = torch.tensor([9], dtype=torch.int32)
    again = _decode_step(model, cache, token, 8).clone()
    assert float((again - long).abs().max()) < 1e-4


# ---------------------------------------------------------------------------
# 真机门（K1 / K4）—— 需要 CUDA + triton
# ---------------------------------------------------------------------------
@pytest.mark.gpu
def test_kernel_matches_fp64_oracle_and_length_change():
    """K1：kernel vs fp64 oracle（误差不差于 eager 基线 × 1.5）；K4：换长度重放。

    ★ K4 用的是**同一个** `filled` 张量、原地 `fill_` 改值 —— 也就是图内每步
    `_set_step_inputs` 干的那件事。换一块新张量测不出"kernel 真的从设备内存读"。
    """
    if not torch.cuda.is_available() or not attn_decode_available():
        pytest.skip("需要 CUDA + triton")
    device = "cuda"
    torch.manual_seed(0)
    seq, filled0 = 1056, 33
    q = (torch.randn(1, HEADS, 1, HEAD_DIM, device=device) * 0.1).half()
    k = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device) * 0.1).half()
    v = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device) * 0.1).half()
    v[:, :, filled0:, :] = 20.0  # 判别力保障（见 test_length_change_...）

    filled = torch.full((1,), filled0, dtype=torch.int32, device=device)
    got = attn_decode_forward(q, k, v, filled)
    want = dense_attention_oracle(q, k, v, filled0).to(torch.float32).reshape(HEADS, HEAD_DIM)
    kernel_err = float((got.float().reshape(HEADS, HEAD_DIM) - want.float()).abs().max())

    baseline = attn_decode_forward(q, k, v, filled, backend="oracle")
    baseline_err = float((baseline.float().reshape(HEADS, HEAD_DIM) - want.float()).abs().max())
    assert (
        kernel_err <= baseline_err * PARITY_GATE["tol_slack"]
    ), "K1: kernel err %.3e > baseline err %.3e × %s" % (
        kernel_err,
        baseline_err,
        PARITY_GATE["tol_slack"],
    )
    assert torch.allclose(
        got.float().reshape(HEADS, HEAD_DIM),
        want.float(),
        atol=PARITY_GATE["tol_atol"],
        rtol=PARITY_GATE["tol_rtol"],
    )

    filled1 = 67
    filled.fill_(filled1)  # ← 原地改设备内存，与图内每步 fill_ 完全同一机制
    correct = attn_decode_forward(q, k, v, filled)
    true1 = dense_attention_oracle(q, k, v, filled1).to(torch.float32).reshape(HEADS, HEAD_DIM)
    stale = want  # 旧长度的答案
    verdict = judge_length_change(
        correct_err=float((correct.float().reshape(HEADS, HEAD_DIM) - true1).abs().max()),
        stale_err=float((stale - true1).abs().max()),
    )
    assert verdict["verdict"] == "PASS", verdict


@pytest.mark.gpu
def test_kernel_never_reads_the_unfilled_tail():
    """★ "不需要 4D mask"的**真机**证明：把 [filled, seq) 的 k/v 全填 NaN。

    若 kernel 读到任何一个未填充槽位，输出必然是 NaN（NaN 会污染整条 softmax）。
    这是 kernel 与 mask 类实现的根本分界：mask 是"算完再遮"，kernel 是"根本不读"。
    """
    if not torch.cuda.is_available() or not attn_decode_available():
        pytest.skip("需要 CUDA + triton")
    device = "cuda"
    torch.manual_seed(0)
    seq, filled0 = 1056, 320
    q = (torch.randn(1, HEADS, 1, HEAD_DIM, device=device) * 0.1).half()
    k = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device) * 0.1).half()
    v = (torch.randn(1, HEADS, seq, HEAD_DIM, device=device) * 0.1).half()
    ft = torch.full((1,), filled0, dtype=torch.int32, device=device)
    clean = attn_decode_forward(q, k, v, ft).clone()

    k2, v2 = k.clone(), v.clone()
    k2[:, :, filled0:, :] = float("nan")
    v2[:, :, filled0:, :] = float("nan")
    got = attn_decode_forward(q, k2, v2, ft)
    assert not torch.isnan(got.float()).any(), "kernel 读到了未填充槽位（NaN 出现了）"
    assert torch.equal(got, clean), "尾部 NaN 改变了结果 —— mask 不能去掉"
