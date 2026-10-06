"""test_moe_static — 固定形状 MoE 与 HF 原版的**数值等价性**，全部可在 CPU 上跑。

为什么这些测试必须在离线跑：

EXP-p0-03 要验证 CUDA Graph 能不能 capture。我（基于 profile 证据）预判 HF 的
`torch.where` 路径会失败，所以提前写了 `moe_static` 作为替代路径。
但"提前写好的替代实现"如果不先证明与原版**数值一致**，就会把
"CUDA Graph 能不能用"这个问题和"我的实现对不对"搅在一起 —— 那样在计费机器上
根本分不清失败原因。

所以这里在 CPU 上用真实的 `OlmoeSparseMoeBlock` 做 parity 对照。
"""

from __future__ import annotations

import os
import re
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from lemoq.kernels.moe_static import (  # noqa: E402
    fuse_experts,
    replace_moe_blocks,
    static_moe_forward,
)

MODULE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
    "lemoq",
    "kernels",
    "moe_static.py",
)


def _tiny_config(**over):
    from transformers import OlmoeConfig

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
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def _tiny_block(cfg=None, seed=0):
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    torch.manual_seed(seed)
    return OlmoeSparseMoeBlock(cfg or _tiny_config()).eval()


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------
def test_fuse_experts_stacks_into_two_tensors():
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    E = len(block.experts)
    ffn = block.experts[0].intermediate_size
    hidden = block.experts[0].hidden_size

    assert fused.gate_up.shape == (E, 2 * ffn, hidden)
    assert fused.down.shape == (E, hidden, ffn)
    assert fused.num_experts == E
    assert fused.nbytes() > 0


def test_fused_gate_up_matches_original_experts():
    """拼起来之后第 e 行必须逐元素等于原第 e 个 expert 的 gate/up。"""
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    ffn = block.experts[0].intermediate_size
    for e in range(len(block.experts)):
        assert torch.equal(fused.gate_up[e, :ffn], block.experts[e].gate_proj.weight)
        assert torch.equal(fused.gate_up[e, ffn:], block.experts[e].up_proj.weight)
        assert torch.equal(fused.down[e], block.experts[e].down_proj.weight)


# ---------------------------------------------------------------------------
# ★ 数值等价性 —— 这是本文件存在的理由
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("norm_topk_prob", [False, True])
def test_parity_with_hf_block(norm_topk_prob):
    """固定形状前向必须与 HF 原版数值一致。"""
    cfg = _tiny_config(norm_topk_prob=norm_topk_prob)
    block = _tiny_block(cfg)

    torch.manual_seed(1)
    x = torch.randn(1, 1, cfg.hidden_size)

    with torch.no_grad():
        ref_out, ref_logits = block(x)

        fused = fuse_experts(block.experts)
        got_out, got_logits = static_moe_forward(
            x,
            block.gate.weight,
            fused,
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )

    assert torch.equal(ref_logits, got_logits), "router logits 不一致"
    assert torch.allclose(ref_out, got_out, atol=1e-6), (
        "输出不一致：max abs diff = %g" % (ref_out - got_out).abs().max().item()
    )


def test_parity_with_multi_token_input():
    """prefill 时 token 数 > 1，也要一致。"""
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    torch.manual_seed(2)
    x = torch.randn(1, 5, cfg.hidden_size)
    with torch.no_grad():
        ref_out, _ = block(x)
        got_out, _ = static_moe_forward(
            x,
            block.gate.weight,
            fuse_experts(block.experts),
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )
    assert torch.allclose(ref_out, got_out, atol=1e-6)


def test_parity_holds_for_many_different_routings():
    """★ 关键：parity 不能只对一组输入成立。

    原版走 `torch.where` 分支、我们走 gather+sum，两条路径在不同路由分布下
    （全都在同一个 expert / 每个 token 分到不同 expert）都应当等价。
    这正是 CUDA Graph replay 时会发生的情况 —— 数据变了，捕获的图不变。
    """
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    fused = fuse_experts(block.experts)
    for seed in range(12):
        torch.manual_seed(100 + seed)
        x = torch.randn(1, 1, cfg.hidden_size)
        with torch.no_grad():
            ref, _ = block(x)
            got, _ = static_moe_forward(
                x,
                block.gate.weight,
                fused,
                top_k=block.top_k,
                norm_topk_prob=block.norm_topk_prob,
                act_fn=block.experts[0].act_fn,
            )
        assert torch.allclose(ref, got, atol=1e-6), "seed=%d 不一致" % seed


def test_parity_with_extreme_routing():
    """把 router 权重推到极端，强制所有 token 挤到同一个 expert。

    这是 `torch.where` 路径最容易出问题的情形（输出长度从 0 变到满），
    也是 CUDA Graph 冻结形状后最可能静默算错的情形。
    """
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    with torch.no_grad():
        block.gate.weight.zero_()
        block.gate.weight[3, :] = 10.0  # expert 3 通吃
    fused = fuse_experts(block.experts)
    x = torch.randn(1, 4, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = block(x)
        got, _ = static_moe_forward(
            x,
            block.gate.weight,
            fused,
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )
    assert torch.allclose(ref, got, atol=1e-6)


# ---------------------------------------------------------------------------
# ★ 形状不变性 —— CUDA Graph 真正依赖的性质
# ---------------------------------------------------------------------------
def test_output_shape_is_data_independent():
    """输出形状绝不能依赖路由结果。"""
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    fused = fuse_experts(block.experts)
    shapes = set()
    with torch.no_grad():
        for seed in range(6):
            torch.manual_seed(seed)
            x = torch.randn(1, 1, cfg.hidden_size)
            out, logits = static_moe_forward(
                x,
                block.gate.weight,
                fused,
                top_k=block.top_k,
                norm_topk_prob=block.norm_topk_prob,
                act_fn=block.experts[0].act_fn,
            )
            shapes.add((tuple(out.shape), tuple(logits.shape)))
    assert len(shapes) == 1, "输出形状随数据变化：%s" % shapes


def test_intermediate_shapes_are_static():
    """逐个检查中间张量的形状都由 (T, K) 决定，与数据无关。

    直接断言"长度为 top_k"而不是"等于实际非零专家数" —— 后者才是数据依赖。
    """
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    fused = fuse_experts(block.experts)
    seen = set()
    with torch.no_grad():
        for seed in range(8):
            torch.manual_seed(seed)
            x = torch.randn(1, 1, cfg.hidden_size)
            logits = torch.nn.functional.linear(x.reshape(1, -1), block.gate.weight)
            w = torch.softmax(logits, dim=-1, dtype=torch.float)
            top_w, top_i = torch.topk(w, block.top_k, dim=-1)
            ids = top_i.reshape(-1)
            seen.add(
                (
                    tuple(top_w.shape),
                    tuple(top_i.shape),
                    tuple(ids.shape),
                    tuple(fused.gate_up[ids].shape),
                    tuple(fused.down[ids].shape),
                )
            )
    assert len(seen) == 1, "中间形状随数据变化：%s" % seen


# ---------------------------------------------------------------------------
# ★ 静态源码检查 —— 防止以后有人把敌对算子加回来
# ---------------------------------------------------------------------------
def test_module_contains_no_host_synchronizing_ops():
    """模块里不得出现会触发 device->host 同步的算子。

    这些是 CUDA Graph capture 的直接杀手。数值测试抓不到它们，
    因为它们在 CPU 上"能跑"、在 eager GPU 上也"能跑"，只在 capture 时炸。
    """
    with open(MODULE, encoding="utf-8") as fh:
        src = fh.read()
    # 去掉 docstring 与注释，避免把说明文字误判成代码
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = re.sub(r"#.*", "", code)
    forbidden = {
        "torch.where": r"torch\.where\s*\(",
        ".nonzero(": r"\.nonzero\s*\(",
        ".item()": r"\.item\s*\(\s*\)",
        ".cpu()": r"\.cpu\s*\(\s*\)",
        ".tolist()": r"\.tolist\s*\(\s*\)",
        ".numpy()": r"\.numpy\s*\(\s*\)",
    }
    for label, pat in forbidden.items():
        assert not re.search(pat, code), "%s 出现在代码里 —— 它会阻塞 CUDA Graph capture" % label


def test_module_has_no_per_expert_python_loop():
    """不得出现 `for ... in range(num_experts)` 这种随模型规模增长的循环。

    原版正是这个循环产生了 8000 个 kernel。
    """
    with open(MODULE, encoding="utf-8") as fh:
        src = fh.read()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = re.sub(r"#.*", "", code)
    assert not re.search(r"for\s+\w+\s+in\s+range\s*\(\s*(self\.)?num_experts", code)


# ---------------------------------------------------------------------------
# 整模型替换
# ---------------------------------------------------------------------------
def test_replace_moe_blocks_swaps_forward_and_keeps_output():
    from transformers import OlmoeForCausalLM

    cfg = _tiny_config()
    torch.manual_seed(3)
    model = OlmoeForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3, 4]])

    with torch.no_grad():
        ref = model(ids).logits

    n = replace_moe_blocks(model)
    assert n == cfg.num_hidden_layers, "替换了 %d 个，应为 %d" % (n, cfg.num_hidden_layers)

    with torch.no_grad():
        got = model(ids).logits

    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5), (
        "整模型输出不一致：max abs diff = %g" % (ref - got).abs().max().item()
    )


def test_replace_moe_blocks_frees_original_experts():
    """替换后原 ModuleList 必须被释放，否则显存翻倍（16 层 = +12 GiB）。"""
    from transformers import OlmoeForCausalLM

    cfg = _tiny_config()
    torch.manual_seed(4)
    model = OlmoeForCausalLM(cfg).eval()
    replace_moe_blocks(model)
    for layer in model.model.layers:
        assert not hasattr(layer.mlp, "experts"), "原始 experts 没被释放"
        assert hasattr(layer.mlp, "_static_experts")


def test_replace_moe_blocks_is_idempotent_in_effect():
    """重复调用不应改变结果（第二次已经没有 experts 可换）。"""
    from transformers import OlmoeForCausalLM

    cfg = _tiny_config()
    torch.manual_seed(5)
    model = OlmoeForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3]])
    assert replace_moe_blocks(model) == cfg.num_hidden_layers
    with torch.no_grad():
        first = model(ids).logits
    assert replace_moe_blocks(model) == 0
    with torch.no_grad():
        second = model(ids).logits
    assert torch.allclose(first, second, atol=0)


# ---------------------------------------------------------------------------
# 与 runner / 实验矩阵的接线（离线可测，不需要 GPU）
# ---------------------------------------------------------------------------
def test_runner_exposes_static_moe_flag():
    """`--static-moe` 必须能被解析。"""
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.runner", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "--static-moe" in proc.stdout


def test_run_cudagraph_accepts_static_moe_parameter():
    """run_cudagraph 必须显式接受 static_moe，而不是偷偷从全局读。"""
    import inspect

    from benchmarks.runner import run_cudagraph

    params = inspect.signature(run_cudagraph).parameters
    assert "static_moe" in params
    assert params["static_moe"].default is False, "默认必须是 False（p0-03 行为不变）"


def test_exp_p0_04_declared_with_static_moe():
    """矩阵里必须有 p0-04，且它声明 moe: static。

    没有这一步的话，`--static-moe` 的结果会写进 p0-03 的 EXP-ID，
    两条**不同假设**的实验就混在同一个结果文件里了。
    """
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "experiments.yaml"), encoding="utf-8") as fh:
        matrix = yaml.safe_load(fh)
    by_id = {e["id"]: e for e in matrix["experiments"]}
    assert "EXP-p0-04-cudagraph-static-moe" in by_id
    exp = by_id["EXP-p0-04-cudagraph-static-moe"]
    assert exp["config"]["moe"] == "static"
    assert exp["config"]["execution"] == "cudagraph"
    assert exp["ctx"] == [1024]


def test_p0_03_stays_on_the_hf_path():
    """p0-03 必须仍然是 HF 原版路径 —— 它是"原版能不能进图"的证据。"""
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "experiments.yaml"), encoding="utf-8") as fh:
        matrix = yaml.safe_load(fh)
    exp = {e["id"]: e for e in matrix["experiments"]}["EXP-p0-03-baseline-cudagraph"]
    assert exp["config"].get("moe", "hf") == "hf"
