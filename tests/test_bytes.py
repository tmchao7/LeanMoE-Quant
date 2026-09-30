"""test_bytes — 字节预算是全项目的物理口径，因此必须**锁死**。

锚点（已核实，见 docs/MODELS.md §3.1–§3.2）：
    OLMoE-1B-7B-0924 总参数            6,919,028,736
    激活/token（不含 embedding）        1,178,861,568
    [W] fp16 权重字节/token             2,357,723,136
    专家占 [W]                          68.3%
    KV 读 @1K / @4K                     134,217,728 / 536,870,912
    KV 占 [T] @1K / @4K                 5.4% / 18.5%

**任何改动若打破这些数字，必须解释**——它们是把 PLAN.md / docs/MODELS.md /
docs/METRICS.md 串起来的唯一定量依据。
"""

from __future__ import annotations

import pytest

from benchmarks.bytes import (
    ModelSpec,
    QuantSpec,
    attainments,
    byte_budget,
    kv_budget,
    weight_budget,
)


def test_total_params_matches_verified_config(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config, name="OLMoE-1B-7B-0924")
    assert spec.total_params() == 6_919_028_736


def test_activated_params_excl_embedding(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    assert spec.activated_params_excl_embedding() == 1_178_861_568


def test_W_anchor_bytes(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    w = weight_budget(spec)
    assert w["total_bytes"] == 2_357_723_136


def test_W_breakdown_shares(olmoe_config):
    """专家 68.3% / attention 22.8% / lm_head 8.7% —— 量化该打哪里由它决定。"""
    spec = ModelSpec.from_hf_config(olmoe_config)
    w = weight_budget(spec)
    by_name = {it["name"]: it for it in w["items"]}
    assert by_name["experts(top-k)"]["share_pct"] == pytest.approx(68.3, abs=0.1)
    assert by_name["attention"]["share_pct"] == pytest.approx(22.8, abs=0.1)
    assert by_name["lm_head"]["share_pct"] == pytest.approx(8.7, abs=0.1)
    assert by_name["router"]["share_pct"] == pytest.approx(0.2, abs=0.1)


def test_embedding_row_not_counted_in_total(olmoe_config):
    """每 token 只 gather 一行（~4 KiB），它不能被算进 [W]。"""
    spec = ModelSpec.from_hf_config(olmoe_config)
    w = weight_budget(spec)
    embed = [it for it in w["items"] if it["name"] == "embedding_row"][0]
    assert embed["counted_in_total"] is False
    assert embed["bytes"] == 2048 * 2
    assert embed["bytes"] < 10_000


def test_kv_read_anchors(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    assert kv_budget(spec, ctx=1024)["read_bytes"] == 134_217_728
    assert kv_budget(spec, ctx=4096)["read_bytes"] == 536_870_912


def test_T_is_W_plus_kv_and_kv_share(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    b1 = byte_budget(spec, ctx=1024)
    b4 = byte_budget(spec, ctx=4096)
    assert b1["T"]["total_bytes"] == 2_357_723_136 + 134_217_728
    assert b1["T"]["kv_share_pct"] == pytest.approx(5.4, abs=0.1)
    assert b4["T"]["kv_share_pct"] == pytest.approx(18.5, abs=0.1)


def test_kv_scales_with_batch_and_ctx(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    a = kv_budget(spec, ctx=1024, batch=1)["read_bytes"]
    b = kv_budget(spec, ctx=1024, batch=4)["read_bytes"]
    c = kv_budget(spec, ctx=2048, batch=1)["read_bytes"]
    assert b == 4 * a
    assert c == 2 * a


def test_quantized_weight_shrinks_but_not_4x(olmoe_config):
    """W4A16 的真实收益 **小于 4x** —— 因为 scale、router、lm_head 仍是 fp16。

    这是必须诚实报出的数字：字节降到 1/4 只是专家的那部分。
    """
    spec = ModelSpec.from_hf_config(olmoe_config)
    fp16 = weight_budget(spec)["total_bytes"]
    w4 = weight_budget(spec, quant=QuantSpec(bits=4, group_size=128))["total_bytes"]
    ratio = fp16 / w4
    assert 2.5 < ratio < 4.0, "实际比值 %.2f 不在预期区间" % ratio


def test_lm_head_not_quantized_when_kept_fp16(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    w = weight_budget(spec, quant=QuantSpec(bits=4, group_size=128))
    lm = [it for it in w["items"] if it["name"] == "lm_head"][0]
    assert lm["bytes"] == 103_022_592 * 2  # 保持 fp16


def test_tied_embeddings_means_no_lm_head_stream():
    cfg = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "num_hidden_layers": 2,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "intermediate_size": 256,
        "vocab_size": 1000,
        "tie_word_embeddings": True,
    }
    spec = ModelSpec.from_hf_config(cfg)
    w = weight_budget(spec)
    lm = [it for it in w["items"] if it["name"] == "lm_head"][0]
    assert lm["bytes"] == 0
    assert "tied" in lm["formula"]


def test_dense_model_path():
    """非 MoE：FFN 不算 top-k，且没有 router。"""
    cfg = {
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    spec = ModelSpec.from_hf_config(cfg)
    assert spec.is_moe is False
    w = weight_budget(spec)
    by_name = {it["name"]: it for it in w["items"]}
    # 24 层 x 3 x 896 x 4864
    assert by_name["experts(top-k)"]["params"] == 24 * 3 * 896 * 4864
    # dense 模型没有 router，因此该项不应出现
    assert "router" not in by_name
    # tied embedding -> 不流 lm_head
    assert by_name["lm_head"]["bytes"] == 0


def test_attainments_state_their_denominator(olmoe_config):
    spec = ModelSpec.from_hf_config(olmoe_config)
    w = weight_budget(spec)
    b = byte_budget(spec, ctx=4096)
    out = attainments(
        w_bytes=w["total_bytes"],
        t_bytes=b["T"]["total_bytes"],
        measured_ms=3.0,
        measured_bw_gbps=1500.0,
    )
    assert out["status"] == "measured"
    # [T] > [W]  =>  floor_T > floor_W  =>  带宽达成率 > 量化达成率（同一个实测延迟作分母）
    assert out["floor_t_ms"] > out["floor_w_ms"]
    assert out["bw_attainment_pct"] > out["quant_attainment_pct"]
    assert "excludes KV" in out["denominator_note"]
    assert "[T]" in out["denominator_note"]


def test_attainments_refuse_to_divide_by_zero(olmoe_config):
    out = attainments(w_bytes=1, t_bytes=1, measured_ms=0.0, measured_bw_gbps=1500.0)
    assert out["status"] == "not_measurable"


def test_oltmoe_max_position_embeddings_is_4096(olmoe_config):
    """context 扫描上限据此确定 —— 防止有人再写 8K（docs/MODELS.md §3.3）。"""
    assert olmoe_config["max_position_embeddings"] == 4096
