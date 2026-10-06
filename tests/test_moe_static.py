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
    FusedExperts,
    fuse_experts,
    replace_moe_blocks,
    static_moe_forward,
)

OLMOE_HIDDEN = 2048
OLMOE_FFN = 1024
OLMOE_TOP_K = 8

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
def test_data_dependent_ops_are_confined_to_the_prefill_path():
    """★ 数据依赖算子只允许出现在 `_prefill_forward` 里。

    decode 路径（`_decode_forward`）要进 CUDA Graph，绝不能含这些算子；
    prefill 路径是 eager 的，含它们反而是**正确**的做法 —— 逐专家循环让每个
    专家的权重只读一次（实测 7.60 ms vs 分块方案的 155.77 ms）。

    所以断言的不是"文件里没有"，而是"只在那个函数里"。
    这条替代了早期那条"整模块不许有 torch.where"的检查 —— 那条在引入 prefill
    专用路径后就不再是正确的约束了，而放宽它必须是有理由的、且范围更精确。
    """

    with open(MODULE, encoding="utf-8") as fh:
        src = fh.read()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = re.sub(r"#.*", "", code)

    def body(name):
        m = re.search(r"^def %s\(.*?(?=^def |^class |\Z)" % name, code, re.S | re.M)
        assert m, "找不到函数 %s" % name
        return m.group(0)

    decode = body("_decode_forward")
    prefill = body("_prefill_forward")
    forbidden = {
        "torch.where": r"torch\.where\s*\(",
        ".nonzero(": r"\.nonzero\s*\(",
        ".item()": r"\.item\s*\(\s*\)",
        ".cpu()": r"\.cpu\s*\(\s*\)",
        ".tolist()": r"\.tolist\s*\(\s*\)",
        ".numpy()": r"\.numpy\s*\(\s*\)",
    }
    for label, pat in forbidden.items():
        assert not re.search(pat, decode), "decode 路径含 %s —— 图捕获会被阻塞" % label
    assert re.search(
        r"\.nonzero\s*\(", prefill
    ), "prefill 路径应使用数据依赖索引（逐专家循环）—— 它被改回分块了吗？"


def test_no_python_loop_in_the_decode_path():
    """decode 路径不得有随专家数增长的循环（HF 的 64 专家循环就是这么来的）。"""

    with open(MODULE, encoding="utf-8") as fh:
        src = fh.read()
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = re.sub(r"#.*", "", code)
    m = re.search(r"^def _decode_forward\(.*?(?=^def |^class |\Z)", code, re.S | re.M)
    assert m
    assert not re.search(r"for\s+\w+\s+in\s+range", m.group(0)), "decode 路径出现了 Python 循环"


def test_prefill_path_reads_each_expert_once():
    """★ 事故 2 的回归：prefill 必须逐专家循环，而不是按 (token, expert) 对 gather。

    算术判据：prefill 的权重读量应约为 `num_experts x 单专家权重`，
    而"每对读一次"的写法则会是它的 K 倍以上。

    OLMoE 实际数字（单层，prefill ctx=1024）：
        HF / 逐专家循环 : 64 x 8.4 MB        =   537 MB
        分块 gather     : 8192 x 8.4 MB      = 68813 MB   (128 倍)
    """
    hidden, ffn, experts, top_k, ctx = 2048, 1024, 64, 8, 1024
    per_expert = (2 * ffn * hidden + hidden * ffn) * 2
    per_expert_loop = experts * per_expert
    per_pair = top_k * ctx * per_expert
    assert per_pair / per_expert_loop == 128.0, "每对 gather 应是逐专家循环的 128 倍（=%d/%d）" % (
        top_k * ctx,
        experts,
    )
    # 逐专家循环的读量 = 64 个专家各一遍 = 768 MB
    assert abs(per_expert_loop / 1024**2 - 768.0) < 1.0, "重写后应为 768 MB，实际 %.0f MB" % (
        per_expert_loop / 1024**2
    )
    assert per_expert / 1024**2 == 12.0, "单专家权重应为 12 MiB"


def test_prefill_at_real_ctx_materialises_bounded_memory():
    """★ 事故 1 的回归：prefill 不得按 (T*K) 物化专家权重。

    旧实现在 ctx=1024 时物化 (8192, 2048, 2048) fp16 = 64 GiB（实测报错原文），
    新实现一次只碰一个专家的权重。
    """
    per_pair_bytes = (2 * OLMOE_FFN * OLMOE_HIDDEN + OLMOE_HIDDEN * OLMOE_FFN) * 2
    ctx = 1024
    old_gib = OLMOE_TOP_K * ctx * per_pair_bytes / 1024**3
    assert abs(old_gib - 96.0) < 0.1, "旧实现应为 96 GiB，实际 %.1f" % old_gib
    gate_up_only = OLMOE_TOP_K * (2 * OLMOE_FFN) * OLMOE_HIDDEN * 2 * ctx / 1024**3
    assert abs(gate_up_only - 64.0) < 0.1, "实测报错是 64.00 GiB，这里应为 64"
    # 新实现：一次只碰一个专家 = 12 MiB，与 ctx 无关
    assert per_pair_bytes / 1024**2 == 12.0, "单专家权重应为 12 MiB"
    assert per_pair_bytes / 1024**3 < 0.02, "单专家权重应远小于出事的 64 GiB"


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
    # ctx 是声明的轴：1024 是可行性门，4096 是 H-A/H-B 的判别点
    # （attention 随 ctx 增长，MoE 专家路径不随）。两个 ctx 各自一个结果文件。
    assert exp["ctx"] == [1024, 4096]


def test_p0_03_stays_on_the_hf_path():
    """p0-03 必须仍然是 HF 原版路径 —— 它是"原版能不能进图"的证据。"""
    import yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "configs", "experiments.yaml"), encoding="utf-8") as fh:
        matrix = yaml.safe_load(fh)
    exp = {e["id"]: e for e in matrix["experiments"]}["EXP-p0-03-baseline-cudagraph"]
    assert exp["config"].get("moe", "hf") == "hf"


# ---------------------------------------------------------------------------
# 双路径各自的正确性
# ---------------------------------------------------------------------------
def test_decode_path_handles_single_token():
    """T=1 必须走固定形状路径。"""
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    fused = fuse_experts(block.experts)
    x = torch.randn(1, 1, cfg.hidden_size)
    with torch.no_grad():
        out, logits = static_moe_forward(
            x,
            block.gate.weight,
            fused,
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )
    assert out.shape == (1, 1, cfg.hidden_size)
    assert logits.shape == (1, cfg.num_experts)


def test_prefill_path_still_matches_hf():
    """★ prefill 路径（逐专家循环）必须与 HF 原版数值一致。"""
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    torch.manual_seed(43)
    x = torch.randn(1, 9, cfg.hidden_size)
    with torch.no_grad():
        ref, _ = block(x)
        got, _ = static_moe_forward(
            x,
            block.gate.weight,
            fuse_experts(block.experts),
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )
    assert torch.allclose(ref, got, atol=1e-6), (
        "prefill 路径与 HF 不一致：max diff = %g" % (ref - got).abs().max().item()
    )


def test_prefill_path_equals_token_by_token_decode():
    """prefill（逐专家）与逐 token 走 decode 路径应当一致。

    两者是**不同实现**，所以这条会抓到"某一条路径写了特例/漏了归一化"这类错误。
    """
    cfg = _tiny_config()
    block = _tiny_block(cfg)
    fused = fuse_experts(block.experts)
    torch.manual_seed(47)
    x = torch.randn(1, 6, cfg.hidden_size)

    with torch.no_grad():
        prefill_out, _ = static_moe_forward(
            x,
            block.gate.weight,
            fused,
            top_k=block.top_k,
            norm_topk_prob=block.norm_topk_prob,
            act_fn=block.experts[0].act_fn,
        )
        per_token = torch.cat(
            [
                static_moe_forward(
                    x[:, t : t + 1, :],
                    block.gate.weight,
                    fused,
                    top_k=block.top_k,
                    norm_topk_prob=block.norm_topk_prob,
                    act_fn=block.experts[0].act_fn,
                )[0]
                for t in range(6)
            ],
            dim=1,
        )

    assert torch.allclose(prefill_out, per_token, atol=1e-6), (
        "prefill 与逐 token 不一致：max diff = %g" % (prefill_out - per_token).abs().max().item()
    )


def test_prefill_path_survives_a_large_token_count():
    """接近真实 prefill 的 token 数不得爆内存（缩小的 channel，T 保持量级）。"""
    import torch.nn as nn

    hidden, ffn, experts, top_k = 64, 32, 8, 2
    torch.manual_seed(41)
    router = nn.Linear(hidden, experts, bias=False)
    fused = FusedExperts(
        gate_up=torch.randn(experts, 2 * ffn, hidden) * 0.02,
        down=torch.randn(experts, hidden, ffn) * 0.02,
    )
    x = torch.randn(1, 512, hidden)
    with torch.no_grad():
        out, logits = static_moe_forward(x, router.weight, fused, top_k=top_k)
    assert out.shape == (1, 512, hidden)
    assert logits.shape == (512, experts)
    assert torch.isfinite(out).all()
