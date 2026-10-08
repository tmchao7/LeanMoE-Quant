"""test_moe_grouped — grouped（无 gather）路径的离线回归，**全部在 CPU 上跑**。

## 为什么这些测试必须先在这里跑，而不是上机再调

grouped 路径有两类错误，代价完全不同：

    公式/索引错   gather 的 slot 与 expert 配对错位、gate 与 up 读错段、
                  `top_w` 乘错位置 —— 会算出一个**看起来合理**的结果
    kernel 实现错  Triton 里偏移写错 —— 需要真机

第一类占了绝大多数，而且**不需要 GPU 就能全部抓出来**。所以这里做两件事：

1. 用 fp32 把 grouped 公式与 gather 路径对齐到 1e-5 —— 这样量到的只有"公式"
   本身，不含 fp16 舍入；
2. 把 host 侧的 grid / 缓冲区形状做成 `launch_plan`（纯 Python），用**手算值**钉住。

第 2 条是上一轮的教训：prefill 的 64 GiB 与 128× 重复读，小模型 parity 测试一个
都没抓到，最后是靠纯算术断言抓到的。CUDA Graph 只约束"形状静态"，不约束"形状
对不对" —— grid 少算一维不会报错，只会静默算错。
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lemoq.kernels.moe_grouped import (  # noqa: E402
    GROUPED_CONFIGS,
    PARITY_GATE,
    _check_act_fn,
    grouped_available,
    grouped_down_sum_reference,
    grouped_gate_up_act_reference,
    grouped_moe_forward,
    launch_plan,
    static_moe_forward_grouped_reference,
)
from lemoq.kernels.moe_static import (  # noqa: E402
    VALID_MOE_IMPLS,
    fuse_experts,
    replace_moe_blocks,
    static_moe_forward,
)

TRITON_KERNELS = os.path.join(ROOT, "src", "lemoq", "kernels", "_grouped_triton.py")
OLMOE_HIDDEN = 2048
OLMOE_FFN = 1024
OLMOE_TOP_K = 8
OLMOE_LAYERS = 16
BW = 1559.2


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
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def _tiny_block(cfg=None, seed=0):
    from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

    torch.manual_seed(seed)
    return OlmoeSparseMoeBlock(cfg or _tiny_config()).eval()


def _olmoe_fused(dtype=torch.float32):
    """An OLMoE-shaped, zero-filled FusedExperts — for arithmetic only, never a claim."""

    class _Holder:
        pass

    holder = _Holder()
    holder.gate_up = torch.zeros(64, 2 * OLMOE_FFN, OLMOE_HIDDEN, dtype=dtype)
    return holder


# ---------------------------------------------------------------------------
# 1) 公式等价：grouped 与 gather 必须是同一个函数
# ---------------------------------------------------------------------------
def test_grouped_matches_gather_in_float32():
    """★ 核心测试：fp32 下 grouped 公式与 gather 路径一致到 1e-5。

    用 fp32 而不是 fp16 是刻意的 —— 这样量到的差异**只来自公式/索引**，
    不掺 fp16 舍入。fp16 的差异另有单独一条测试，量的才是真实的数值尺度。
    """
    block = _tiny_block().to(torch.float32)
    fused = fuse_experts(block.experts).to(torch.float32)
    torch.manual_seed(7)
    x = torch.randn(1, 1, block.hidden_dim if hasattr(block, "hidden_dim") else 32)

    gather, logits_g = static_moe_forward(
        x, block.gate.weight.detach(), fused, top_k=block.top_k, impl="gather"
    )
    grouped, logits_d = static_moe_forward_grouped_reference(
        x, block.gate.weight.detach(), fused, top_k=block.top_k
    )
    # 路由必须逐位相同（两边复用 _route）—— 否则差异无法归因
    assert torch.equal(logits_g, logits_d)
    assert torch.allclose(gather, grouped, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
def test_grouped_matches_gather_across_random_routings_float32(seed):
    """12 组不同随机初始化 —— 每一组都会给出不同的 top-k 选择。

    单组通过可能是"碰巧这对专家权重长得像"，12 组才说明配对关系是对的。
    """
    block = _tiny_block(seed=seed).to(torch.float32)
    fused = fuse_experts(block.experts).to(torch.float32)
    torch.manual_seed(100 + seed)
    x = torch.randn(1, 1, 32)
    gather, _ = static_moe_forward(x, block.gate.weight.detach(), fused, top_k=block.top_k)
    grouped, _ = static_moe_forward_grouped_reference(
        x, block.gate.weight.detach(), fused, top_k=block.top_k
    )
    assert torch.allclose(gather, grouped, atol=1e-5, rtol=1e-5), "seed=%d" % seed


def test_grouped_handles_degenerate_routing_all_slots_same_expert():
    """退化路由：top_k 个槽位全落在同一个专家上（专家数与 top_k 相等时必然发生）。

    这正是分组最容易写错的情形：`act` 的 K 行都在用同一份权重，
    而 `top_w` 的归一化让权重和为 1。索引错位在这里**不会**被随机路由暴露。
    """
    cfg = _tiny_config(num_experts=2, num_experts_per_tok=2)
    block = _tiny_block(cfg).to(torch.float32)
    fused = fuse_experts(block.experts).to(torch.float32)
    torch.manual_seed(3)
    x = torch.randn(1, 1, 32)
    gather, _ = static_moe_forward(x, block.gate.weight.detach(), fused, top_k=2)
    grouped, _ = static_moe_forward_grouped_reference(x, block.gate.weight.detach(), fused, top_k=2)
    assert torch.allclose(gather, grouped, atol=1e-5, rtol=1e-5)


def test_grouped_matches_gather_in_float16_within_fp16_scale():
    """fp16 下两边的差异必须落在 fp16 自身的量级内（记录真实尺度）。

    ★ 这条测试同时是"为什么 K1 门比 fp64 oracle 而不是比 gather 路径"的证据：
    grouped 在 fp32 里累加、并且把 top_k 维一起吃进 down kernel，
    而 gather 路径每步都落回 fp16。所以差异是**方向明确的更准**，
    但数值上不等 —— 拿 gather 路径当门会把它误判成失败。
    """
    block = _tiny_block().to(torch.float16)
    fused = fuse_experts(block.experts).to(torch.float16)
    torch.manual_seed(11)
    x = torch.randn(1, 1, 32).to(torch.float16)
    gather, _ = static_moe_forward(x, block.gate.weight.detach(), fused, top_k=block.top_k)
    grouped, _ = static_moe_forward_grouped_reference(
        x, block.gate.weight.detach(), fused, top_k=block.top_k
    )
    oracle = static_moe_forward_grouped_reference(
        x.to(torch.float32),
        block.gate.weight.detach().to(torch.float32),
        fuse_experts(block.experts).to(torch.float32),
        top_k=block.top_k,
    )[0]
    err_gather = float((gather.to(torch.float32) - oracle).abs().max())
    err_grouped = float((grouped.to(torch.float32) - oracle).abs().max())
    # grouped 的 fp16 输出应当**不差于** gather 路径
    assert err_grouped <= err_gather + 1e-6, (err_grouped, err_gather)
    assert torch.allclose(gather, grouped, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# 2) 形状不随数据变化 —— CUDA Graph 真正依赖的性质
# ---------------------------------------------------------------------------
def test_grouped_shapes_are_data_independent():
    """输出与中间形状只由 (top_k, ffn, hidden) 决定，不随路由变化。

    这就是 grouped 路径能进图的原因：expert id 只进指针算术，不进控制流。
    """
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    shapes = set()
    for seed in range(8):
        torch.manual_seed(seed)
        x = torch.randn(1, 1, 32)
        out, logits = static_moe_forward_grouped_reference(
            x, block.gate.weight.detach(), fused, top_k=block.top_k
        )
        shapes.add((tuple(out.shape), tuple(logits.shape)))
        x16 = x.to(torch.float16)
        act = grouped_gate_up_act_reference(
            x16,
            fused.gate_up,
            torch.zeros(block.top_k, dtype=torch.long),
            top_k=block.top_k,
            ffn=16,
        )
        shapes.add(("act", tuple(act.shape)))
    assert len({s for s in shapes if not isinstance(s[0], str)}) == 1


def test_grouped_and_gather_agree_on_output_shape():
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    x = torch.randn(3, 1, 32)  # 多 token：两边都必须接受同一形状
    a, _ = static_moe_forward(x, block.gate.weight.detach(), fused, top_k=block.top_k)
    b, _ = static_moe_forward_grouped_reference(
        x.view(1, 3, 32), block.gate.weight.detach(), fused, top_k=block.top_k
    )
    assert a.shape == (3, 1, 32)
    assert b.shape == (1, 3, 32)


# ---------------------------------------------------------------------------
# 3) launch plan 的算术 —— 纯 Python，手算可核对
# ---------------------------------------------------------------------------
def test_launch_plan_grids_for_olmoe_decode():
    """OLMoE decode 的 grid 必须等于手算值。grid 少算一维不会报错，只会算错。"""
    fused = _olmoe_fused()
    plan = launch_plan(fused, top_k=OLMOE_TOP_K)
    assert plan["hidden"] == OLMOE_HIDDEN
    assert plan["ffn"] == OLMOE_FFN
    # gate_up: (top_k, cdiv(ffn, BLOCK_N)=1024/32=32, SPLIT_K=1)
    assert plan["gate_up_act"]["grid"] == (8, 32, 1)
    # down: (cdiv(hidden, BLOCK_N)=2048/64=32, SPLIT_K=4) = 128 program
    assert plan["down_sum"]["grid"] == (32, 4)
    assert plan["gate_up_act"]["act_shape"] == (8, OLMOE_FFN)
    assert plan["down_sum"]["out_shape"] == (OLMOE_HIDDEN,)


def test_default_config_is_the_measured_winner_not_a_guess():
    """★ 默认配置必须保持是实测赢家 —— 参数敏感性在这里是 3.2x。

    这个断言的存在理由是第一版脚本的教训：默认配置落在整张网格**最慢的一角**
    （0.3868 vs 0.1204 ms/层），于是单配置的正确性+性能门判定 NULL，
    按计划就该放弃整条路线 —— 而实际上调好参数它比基线快 1.48x。

    所以宁可让这条测试在有人"顺手改个默认值"时断掉。
    """
    from lemoq.kernels.moe_grouped import GROUPED_CONFIGS

    assert GROUPED_CONFIGS["gate_up_act"] == {
        "BLOCK_N": 32,
        "BLOCK_K": 128,
        "SPLIT_K": 1,
        "num_warps": 4,
    }
    assert GROUPED_CONFIGS["down_sum"] == {
        "BLOCK_N": 64,
        "BLOCK_K": 128,
        "SPLIT_K": 4,
        "num_warps": 8,
    }


def test_grid_size_must_be_enough_to_fill_the_gpu():
    """★ 机制回归：`down_sum` 的 program 数必须够铺满 170 个 SM。

    第一版默认是 BLOCK_N=128 / SPLIT_K=1 → grid (16, 1) = **16 个 program**，
    9% occupancy，比基线慢 2.17x。这个量级的问题不需要 GPU 就能算出来：
    grid 的第 0 维 < SM 数时，SM 大部分是空的。
    """
    import math

    sms = 170  # RTX 5090（硬编码的是**这台卡**，判据本身是比例）
    # ★ 真正的机制判据：program 数要占到 SM 数的相当比例。
    #   以前这条只断言了 `programs == 128`（硬编码）和"慢配置 < 170"——
    #   而 128 < 170，所以测试名声称的"够铺满 170 个 SM"既没被断言、也并不成立；
    #   真正兜住默认配置的是那个硬编码 128，不是机制。
    min_fraction = 0.7

    def programs_of(block_n: int, split: int) -> int:
        plan = launch_plan(
            _olmoe_fused(),
            top_k=OLMOE_TOP_K,
            config={"down_sum": {"BLOCK_N": block_n, "SPLIT_K": split}},
        )
        return math.prod(plan["down_sum"]["grid"])

    # 已知慢的形状：SPLIT_K=1 时 grid 只有 cdiv(hidden, BLOCK_N) 个 program
    for block_n in (32, 64, 128):
        assert programs_of(block_n, 1) < min_fraction * sms, (
            "SPLIT_K=1 / BLOCK_N=%d 的 program 数不足以铺满 SM —— 这正是第一版默认配置"
            "（BLOCK_N=128,SPLIT_K=1: 16 个 program, 9%% occupancy, 比基线慢 2.17x）" % block_n
        )
    # 默认与切分配置必须达标
    assert programs_of(64, 4) >= min_fraction * sms
    assert programs_of(64, 8) >= min_fraction * sms

    # plan 的算术仍要钉住（默认配置 down_sum 的 grid）
    plan = launch_plan(_olmoe_fused(), top_k=OLMOE_TOP_K)
    assert math.prod(plan["down_sum"]["grid"]) == 128
    assert math.prod(plan["gate_up_act"]["grid"]) == OLMOE_TOP_K * (
        plan["ffn"] // plan["gate_up_act"]["BLOCK_N"]
    )


def test_split_k_adds_the_reduction_kernels_and_sizes_the_partials():
    """SPLIT_K > 1 必须多出归约 kernel，且部分和缓冲区形状要对。

    部分和写小了 = 越界写（图里不会报错）；写大了 = 白占显存。
    """
    plan = launch_plan(
        _olmoe_fused(),
        top_k=OLMOE_TOP_K,
        config={"gate_up_act": {"SPLIT_K": 2}, "down_sum": {"SPLIT_K": 4}},
    )
    assert plan["kernel_count"] == 4
    # 默认 BLOCK_N=32 → cdiv(1024, 32) = 32；SPLIT_K 覆盖成 2 → 第 2 维是 2
    assert plan["gate_up_act"]["grid"] == (8, 32, 2)
    assert plan["gate_up_act"]["partial_shape"] == (2, 8, OLMOE_FFN)
    assert plan["down_sum"]["grid"] == (32, 4)
    assert plan["down_sum"]["partial_shape"] == (4, OLMOE_HIDDEN)
    assert plan["act_combine"]["grid"] == ((8 * OLMOE_FFN + 1023) // 1024,)
    assert plan["out_reduce"]["grid"] == ((OLMOE_HIDDEN + 1023) // 1024,)


def test_launch_plan_is_pure_and_does_not_mutate_the_shared_config():
    """`launch_plan` 必须是纯函数：改调用点的配置不能污染模块级默认值。"""
    before = {k: dict(v) for k, v in GROUPED_CONFIGS.items()}
    launch_plan(_olmoe_fused(), top_k=8, config={"gate_up_act": {"SPLIT_K": 3}})
    assert {k: dict(v) for k, v in GROUPED_CONFIGS.items()} == before
    assert GROUPED_CONFIGS["gate_up_act"]["SPLIT_K"] == 1


def test_gather_tax_arithmetic_regression():
    """★ 纯算术回归：gather 路径每 token 的额外流量 = 2 × 理想量。

    OLMoE fp16：gate_up 67,108,864 B/层、down 33,554,432 B/层，
    16 层下的税 = 3,221,225,472 B → 2.066 ms @1559.2 GB/s。
    与 `benchmarks/attribution.py` 的核算必须一致（两处独立算，互相钉住）。
    """
    gate_up = OLMOE_TOP_K * 2 * OLMOE_FFN * OLMOE_HIDDEN * 2
    down = OLMOE_TOP_K * OLMOE_HIDDEN * OLMOE_FFN * 2
    assert gate_up == 67_108_864
    assert down == 33_554_432
    ideal = (gate_up + down) * OLMOE_LAYERS
    assert ideal == 1_610_612_736
    tax = 2 * ideal
    assert tax == 3_221_225_472
    assert tax / (BW * 1e9) * 1000.0 == pytest.approx(2.065953, abs=1e-5)


# ---------------------------------------------------------------------------
# 4) host 管线（不需要 triton，用 oracle backend 跑同一条管线）
# ---------------------------------------------------------------------------
def test_oracle_backend_exercises_the_host_pipeline():
    """`backend="oracle"` 走同一套形状检查 / launch_plan / 输出契约（无 GPU）。"""
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    x = torch.randn(32)
    ids = torch.tensor([0, 3], dtype=torch.long)
    w = torch.tensor([0.5, 0.5])
    out = grouped_moe_forward(x, fused, ids, w, backend="oracle")
    assert out.shape == (32,)
    assert out.dtype == fused.gate_up.dtype
    expect = grouped_down_sum_reference(
        grouped_gate_up_act_reference(x, fused.gate_up, ids, top_k=2, ffn=16),
        fused.down,
        ids,
        w,
        top_k=2,
    ).to(out.dtype)
    assert torch.allclose(out, expect, atol=1e-5)


def test_grouped_moe_forward_rejects_bad_inputs():
    """形状/布局错误必须在 host 侧就报出来，不要等到 kernel 里越界。"""
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    ids = torch.tensor([0, 1])
    w = torch.tensor([0.5, 0.5])
    with pytest.raises(ValueError):
        grouped_moe_forward(torch.randn(31), fused, ids, w, backend="oracle")  # x 长度不对
    with pytest.raises(ValueError):
        grouped_moe_forward(torch.randn(32), fused, ids, w, backend="nope")
    # down 的 ffn 维不一致
    bad = type(fused)(gate_up=fused.gate_up, down=torch.zeros(8, 32, 15))
    with pytest.raises(ValueError):
        grouped_moe_forward(torch.randn(32), bad, ids, w, backend="oracle")


def test_grouped_moe_forward_requires_triton_when_backend_is_auto():
    """没有 triton 时必须给可操作的错误信息，而不是 AttributeError。"""
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    if grouped_available():
        pytest.skip("本机有 triton：auto 会真的跑 kernel，这条只针对无 triton 的机器")
    with pytest.raises(RuntimeError) as excinfo:
        grouped_moe_forward(torch.randn(32), fused, torch.tensor([0, 1]), torch.tensor([0.5, 0.5]))
    assert "triton" in str(excinfo.value)


def test_grouped_kernel_refuses_a_non_fp16_weight_on_the_triton_path():
    """★ kernel 的 `tl.store` 硬编码 fp16，非 fp16 权重必须**响亮拒绝**。

    host 侧按 `gate_up.dtype` 分配缓冲区，kernel 却恒按 fp16 写 —— fp32/bf16 权重
    会收到 fp16 的位模式而**不报错**。守卫只约束 triton 路径：oracle 是纯 PyTorch、
    dtype 无关（CPU 上的 oracle 测试就用 fp32，必须仍然能跑）。
    """
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    assert fused.gate_up.dtype != torch.float16, "这条要用非 fp16 权重，前置条件变了"

    # triton 路径：必须拒绝（守卫在 _kernels() 之前，所以没装 triton 也测得）
    with pytest.raises(NotImplementedError, match="fp16"):
        grouped_moe_forward(
            torch.randn(32),
            fused,
            torch.tensor([0, 1]),
            torch.tensor([0.5, 0.5]),
            backend="triton",
        )

    # oracle 路径：dtype 无关，必须照常工作
    out = grouped_moe_forward(
        torch.randn(32),
        fused,
        torch.tensor([0, 1]),
        torch.tensor([0.5, 0.5]),
        backend="oracle",
    )
    assert out.shape == (32,)


def test_grouped_quant_kernel_refuses_a_non_fp16_scale():
    """量化 grouped kernel 的 store 同样硬编码 fp16 —— 守的是 scale 的 dtype。"""
    from lemoq.kernels.moe_grouped_quant import (
        grouped_moe_forward_quant,
        quantize_fused_experts,
    )

    fused = fuse_experts(_tiny_block().experts)
    # tiny block: hidden=32 / ffn=16 —— group_size 必须同时整除两者（量化路径不补零）
    q = quantize_fused_experts(fused, bits=8, group_size=8)
    assert q.scale_gate_up.dtype == torch.float16
    q_fp32 = type(q)(  # 同结构、scale 换成 fp32 —— 模拟"哪天把 scale 存成 fp32"
        packed_gate_up=q.packed_gate_up,
        scale_gate_up=q.scale_gate_up.float(),
        packed_down=q.packed_down,
        scale_down=q.scale_down.float(),
        bits=q.bits,
        group_size=q.group_size,
    )
    with pytest.raises(NotImplementedError, match="fp16"):
        grouped_moe_forward_quant(
            torch.zeros(int(q.hidden), dtype=torch.float16),
            q_fp32,
            torch.tensor([0, 1]),
            torch.tensor([0.5, 0.5], dtype=torch.float16),
            backend="triton",
        )


def test_check_act_fn_refuses_a_non_silu_activation():
    """kernel 里融合的是 silu。别的激活必须显式拒绝，不能静默算错。"""
    import torch.nn.functional as F

    _check_act_fn(None)
    _check_act_fn(F.silu)
    with pytest.raises(NotImplementedError):
        _check_act_fn(F.gelu)


# ---------------------------------------------------------------------------
# 5) 分派：grouped 只作用于 decode，prefill 永远走逐专家循环
# ---------------------------------------------------------------------------
def _spy_on_grouped_kernels(monkeypatch) -> dict[str, int]:
    """把 grouped 的两条 kernel 入口换成计数器，返回调用次数。

    ★ 为什么必须用 spy：旧写法靠"没有 triton 时会抛 RuntimeError"来反证"没走
      grouped"，于是它 `if grouped_available(): pytest.skip(...)` —— 也就是说
      这条防 128× 重读 / 慢 20 倍那个事故的测试，**在 5090 上从来没跑过**。
      计数器不依赖 triton 是否存在，两边都能断言。
    """
    import lemoq.kernels.moe_grouped as mg
    import lemoq.kernels.moe_grouped_quant as mgq

    calls = {"fp16": 0, "quant": 0}
    orig_f, orig_q = mg.grouped_moe_forward, mgq.grouped_moe_forward_quant

    def spy_f(*a, **kw):
        calls["fp16"] += 1
        return orig_f(*a, **kw)

    def spy_q(*a, **kw):
        calls["quant"] += 1
        return orig_q(*a, **kw)

    monkeypatch.setattr(mg, "grouped_moe_forward", spy_f)
    monkeypatch.setattr(mgq, "grouped_moe_forward_quant", spy_q)
    return calls


def test_prefill_never_uses_the_grouped_kernel(monkeypatch):
    """★ T > 1 时 `impl="grouped"` 仍然走逐专家循环 —— **有 triton 也要成立**。

    为什么必须这样：grouped kernel 是"每 token 读 top-k 个专家"的写法，
    T=1024 时会对每个 token 重读一遍权重（上一轮实测多读 128 倍、慢 20 倍）。
    这条现在用调用计数断言，不再靠"没抛异常"反证。
    """
    calls = _spy_on_grouped_kernels(monkeypatch)
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    x = torch.randn(4, 1, 32)  # 4 个 token → prefill
    out, _ = static_moe_forward(
        x, block.gate.weight.detach(), fused, top_k=block.top_k, impl="grouped"
    )
    assert out.shape == (4, 1, 32)
    reference, _ = static_moe_forward(
        x, block.gate.weight.detach(), fused, top_k=block.top_k, impl="gather"
    )
    assert torch.allclose(out, reference, atol=1e-4, rtol=1e-4)
    # ★ 真正的判据：grouped 的 kernel 一次都没被调用（无论本机有没有 triton）
    assert calls == {"fp16": 0, "quant": 0}, calls

    # 判别力自检：spy 必须真的装上了 —— 否则 `calls == 0` 是个天然成立的空转断言
    # （monkeypatch 没生效时计数器本来就不会动）。
    import lemoq.kernels.moe_grouped as mg

    mg.grouped_moe_forward(
        torch.randn(32),
        fused,
        torch.tensor([0, 1]),
        torch.tensor([0.5, 0.5]),
        backend="oracle",
    )
    assert calls["fp16"] == 1, "spy 没生效 —— 上面的 calls==0 等于什么都没断言"


def test_decode_with_grouped_impl_calls_the_grouped_kernel(monkeypatch):
    """T == 1 且 impl="grouped" 时必须**真的**走到 grouped kernel。

    有 triton：计数必须为 1（不能静默退回 gather —— 那样"grouped 臂"就成了假臂）。
    无 triton：必须给可操作的 RuntimeError，而不是 AttributeError。
    """
    calls = _spy_on_grouped_kernels(monkeypatch)
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    if not grouped_available():
        with pytest.raises(RuntimeError, match="triton"):
            static_moe_forward(
                torch.randn(1, 1, 32),
                block.gate.weight.detach(),
                fused,
                top_k=block.top_k,
                impl="grouped",
            )
        return
    static_moe_forward(
        torch.randn(1, 1, 32),
        block.gate.weight.detach(),
        fused,
        top_k=block.top_k,
        impl="grouped",
    )
    assert calls["fp16"] == 1, calls


def test_unknown_impl_is_rejected():
    block = _tiny_block()
    fused = fuse_experts(block.experts)
    assert VALID_MOE_IMPLS == ("gather", "grouped")
    with pytest.raises(ValueError):
        static_moe_forward(
            torch.randn(1, 1, 32),
            block.gate.weight.detach(),
            fused,
            top_k=block.top_k,
            impl="magic",
        )


def test_replace_moe_blocks_records_the_impl_and_keeps_prefill_working():
    """整模型替换后：impl 被记录，prefill（多 token）仍然可用且与 gather 一致。"""
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=64,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(5)
    model = OlmoeForCausalLM(cfg).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        reference = model(ids).logits
    assert replace_moe_blocks(model, impl="grouped") == 1
    block = model.model.layers[0].mlp
    assert block._static_impl == "grouped"
    with torch.no_grad():
        after = model(ids).logits
    assert torch.allclose(reference, after, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6) Triton 源码的图安全约束 + 冻结的门
# ---------------------------------------------------------------------------
def test_triton_kernels_avoid_everything_that_breaks_capture():
    """用 **AST** 静态检查 `_grouped_triton.py` 的图安全约束。

    为什么用 AST 而不是字符串匹配：kernel 的 docstring 里**解释**了为什么
    不能用 `@triton.autotune`，字符串匹配会把这段解释本身当成违规
    （第一版就是这样误报的）。AST 只看真正的代码。

    被禁的三类：
      1. `@triton.autotune` —— 它会在 capture 期间真的跑多个 config 来计时，
         等于往图里塞一堆计划外的 kernel（最容易犯、最难从报错里看出来）；
      2. host 同步：`.item()` / `.cpu()` / `nonzero` / `torch.where`；
      3. `while` —— 循环上界通常来自数据。

    ★ 允许的"数据依赖"只有一种：expert id 进**指针算术**（kernel 里的
    `e = tl.load(ids_ptr + slot)`）。它不控制控制流，所以图是安全的；
    所有 `range(...)` 的上界只来自 `program_id` 与 constexpr。
    """
    import ast

    src = open(TRITON_KERNELS, encoding="utf-8").read()
    tree = ast.parse(src)

    decorators = [
        ast.unparse(dec)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for dec in node.decorator_list
    ]
    assert not [d for d in decorators if "autotune" in d], decorators

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call = ast.unparse(node.func)
            if call.endswith((".item", ".cpu", ".numpy", ".tolist")):
                offenders.append(call)
            if any(bad in call for bad in ("nonzero", "where", "sort", "unique")):
                offenders.append(call)
        if isinstance(node, ast.While):
            offenders.append("while-loop")
    assert offenders == [], offenders

    # 循环上界必须只来自 program_id / constexpr —— 逐条点名，改动可见
    assert "for k0 in range(k_start, k_end, BLOCK_K)" in src
    assert "for slot in range(K_TOKENS)" in src
    assert "tl.program_id" in src
    # expert id 只能以 load 的形式进指针算术
    assert "e = tl.load(ids_ptr + slot)" in src


def test_parity_gate_constants_are_frozen():
    """K1 的容差在上机之前冻结；改动必须可见。"""
    assert PARITY_GATE["tol_atol"] == 1e-3
    assert PARITY_GATE["tol_rtol"] == 1e-3
    assert PARITY_GATE["tol_slack"] == 1.5
    # gate 是自校准的（相对基线误差），不是拍一个绝对魔数
    assert "fp64" in PARITY_GATE["note"]
