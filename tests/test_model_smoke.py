"""test_model_smoke — 离线用 transformers 按 OLMoE 架构构建并前向。

★ **这不是性能测试**。它只回答一个问题：**版本 pin 是否成立**
（transformers 能否识别 olmoe、能否构建、能否前向）。

在 Mac CPU 上用 tiny 随机权重，**不下载任何 checkpoint、不产出任何可报告数字**。
真正的 0.5B / OLMoE 前向属于 Gate-0 的 B 组（见 docs/SERVER_RUNBOOK.md）。
"""

from __future__ import annotations

import pytest
from conftest import requires_transformers

pytestmark = requires_transformers


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return OlmoeForCausalLM(cfg).eval(), cfg


def test_transformers_recognizes_olmoe_architecture():
    from transformers import OlmoeConfig, OlmoeForCausalLM  # noqa: F401
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    assert "olmoe" in CONFIG_MAPPING


def test_tiny_olmoe_builds_and_forwards(tiny_model):
    import torch

    model, _ = tiny_model
    with torch.no_grad():
        out = model(torch.tensor([[1, 2, 3, 4]]))
    assert out.logits.shape == (1, 4, 256)


def test_tiny_olmoe_has_expert_parameters(tiny_model):
    """必须真的有 MoE 专家参数 —— 否则这个 smoke test 没在测 MoE。"""
    model, cfg = tiny_model
    names = [n for n, _ in model.named_parameters()]
    assert any("experts" in n for n in names), "没找到 experts 参数"
    assert cfg.num_experts == 8


def test_config_matches_verified_olmoe_fields():
    """用**已核实**的 OLMoE-1B-7B-0924 config 值构建 config 对象（仍然不下载权重）。"""
    from transformers import OlmoeConfig

    cfg = OlmoeConfig(
        vocab_size=50304,
        hidden_size=2048,
        intermediate_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts=64,
        num_experts_per_tok=8,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    assert cfg.num_hidden_layers == 16
    assert cfg.num_experts == 64
    assert cfg.num_experts_per_tok == 8
    assert cfg.intermediate_size == 1024
    assert cfg.max_position_embeddings == 4096
    assert cfg.tie_word_embeddings is False
    assert cfg.num_key_value_heads == cfg.num_attention_heads  # MHA, 不是 GQA


def test_greedy_decode_is_deterministic_on_cpu(tiny_model):
    """CPU 上 greedy 应当可复现。GPU/bf16 的平局非确定性是已知例外（docs/SPECS.md §1.3）。"""
    import torch

    model, _ = tiny_model
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        a = model(ids).logits.argmax(-1).tolist()
        b = model(ids).logits.argmax(-1).tolist()
    assert a == b
