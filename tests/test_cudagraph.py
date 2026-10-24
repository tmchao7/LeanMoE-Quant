"""Unit contracts for the fp16 CUDA Graph feasibility path."""

from __future__ import annotations

import pytest
from conftest import requires_cuda

from benchmarks.cudagraph import CudaGraphConfig, GraphBuffers, normalize_execution
from benchmarks.schema import validate_result


def test_cudagraph_static_cache_capacity_includes_decode_window():
    cfg = CudaGraphConfig(ctx=1024, max_new_tokens=32, batch=1)
    assert cfg.max_cache_len == 1056
    assert cfg.q_len == 1


def test_execution_mode_rejects_unknown_values():
    assert normalize_execution("eager") == "eager"
    assert normalize_execution("cudagraph") == "cudagraph"
    with pytest.raises(ValueError, match="unknown execution"):
        normalize_execution("paged")


def test_graph_buffers_update_values_without_replacing_addresses():
    torch = pytest.importorskip("torch")
    buffers = GraphBuffers.allocate(torch.device("cpu"), batch=1)
    addresses = buffers.addresses()
    buffers.set_step(token=7, position=1024)
    assert buffers.addresses() == addresses
    assert int(buffers.token.item()) == 7
    assert int(buffers.cache_position.item()) == 1024
    assert int(buffers.position_ids.item()) == 1024


def test_blocked_result_keeps_performance_unmeasured():
    from benchmarks.cudagraph import blocked_result

    payload = blocked_result(
        exp_id="EXP-p0-03-baseline-cudagraph__ctx=1024",
        config={
            "ctx": 1024,
            "batch": 1,
            "max_new_tokens": 32,
            "warmup": 5,
            "timed": 20,
            "dtype": "float16",
        },
        failure={"stage": "capture", "type": "RuntimeError", "message": "dynamic shape"},
    )
    assert payload["status"] == "blocked"
    assert payload["latency"]["per_token_p50_ms"] is None
    assert payload["correctness"]["gate"] == "FAIL"
    assert validate_result(payload) == []


def test_blocked_and_failed_results_record_effective_fusion_switches(monkeypatch):
    """A blocked cell must still say which arm it was *attempted* with.

    ★ 2026-10-23：刷新轮的自检在 blocked 格上读到 `fused_route=None`，把"格被阻塞"
    误报成"开关没生效"。blocked/failed payload 走的是另一条组装路径，必须也带
    `fusion_payload_fields()` 的四个键（请求状态 = env 覆盖 + 代码默认）。

    ★ 2026-10-24：断言改成对着 `FUSION_DEFAULTS` 比 —— 采纳那天只改默认，测试不用改
    （旧写法把"默认全 False"写死，翻默认就会红）。另加 `packed_qkv_effective`：
    它有两个前置（attn=kernel + 量化），blocked 的格**生效**恒为 False。
    """
    from benchmarks.cudagraph import blocked_result, failed_result
    from lemoq.kernels import fusion_switches as fs

    # ★ 键集从 `ENV_VARS` 派生（不是手抄开关名）：加第 N 个开关时这条测试不用改，
    #   与"断言对着 FUSION_DEFAULTS 比"是同一个意图（2026-10-24 加 lm_head_w8 时验证）。
    for env in fs.ENV_VARS.values():
        monkeypatch.delenv(env, raising=False)
    cfg = {"ctx": 1024, "batch": 1, "max_new_tokens": 32, "warmup": 5, "timed": 20}
    keys = tuple(fs.ENV_VARS)

    default = blocked_result(
        exp_id="EXP-x__ctx=1024",
        config=cfg,
        failure={"stage": "capture", "type": "RuntimeError", "message": "x"},
    )
    assert {k: default["config"][k] for k in keys} == dict(fs.FUSION_DEFAULTS)
    assert default["config"]["packed_qkv_effective"] is False
    assert validate_result(default) == []

    monkeypatch.setenv("LEMOQ_FUSED_ROPE", "1")
    monkeypatch.setenv("LEMOQ_PACKED_QKV", "1")
    armed = blocked_result(
        exp_id="EXP-x__ctx=1024",
        config=cfg,
        failure={"stage": "capture", "type": "RuntimeError", "message": "x"},
    )
    assert armed["config"]["fused_rope"] is True
    assert armed["config"]["packed_qkv"] is True
    assert armed["config"]["packed_qkv_effective"] is False
    assert armed["config"]["fused_route"] is fs.FUSION_DEFAULTS["fused_route"]  # 未被覆盖

    failed = failed_result(
        exp_id="EXP-x__ctx=1024",
        config=cfg,
        correctness={"kl": 0.1, "top1_agreement": 0.5},
        note="gate failed",
    )
    assert failed["status"] == "failed"
    assert {k: failed["config"][k] for k in keys} == {k: armed["config"][k] for k in keys}
    assert failed["config"]["packed_qkv_effective"] is False
    assert validate_result(failed) == []


def test_compare_logits_accepts_cpu_tensors_and_reports_graph_gate_metrics():
    """CPU 张量要能过（cuda:0 / cpu 混设备的那个真实 bug），而且**必须能分辨差异**。

    ★ 旧写法把 reference 和 candidate 写成**逐字相同**的两个张量，于是
    `kl == 0` / `top1 == 1.0` 是输入恒等的必然后果 —— 把 `compare_logits` 改坏
    （比如让 got_logp 直接用 ref_logp、或 agreement 恒返回 1.0）这条也全绿。
    现在两组：一组恒等（应判 0 / 1.0），一组把两个位置对调（应判明显有差异）。
    """
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import compare_logits

    reference = [torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]])]

    # 1) 恒等：kl 必须为 0、top-1 必须满分（CPU 张量能跑通 = 混设备 bug 没回来）
    same = [torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]])]
    rep_same = compare_logits(reference, same)
    assert rep_same["top1_agreement"] == 1.0
    assert rep_same["kl"] == pytest.approx(0.0, abs=1e-7)

    # 2) 对调：每一步的 argmax 都变了，kl 必须显著为正
    swapped = [torch.tensor([[[0.0, 2.0]]]), torch.tensor([[[2.0, 0.0]]])]
    rep_sw = compare_logits(reference, swapped)
    assert rep_sw["top1_agreement"] == 0.0, "argmax 全变了，一致率必须是 0"
    assert rep_sw["kl"] > 0.5, rep_sw  # 实测约 0.76
    assert rep_sw["kl"] != pytest.approx(0.0, abs=1e-7)


def test_runner_selects_cudagraph_arm_from_exp_id_and_rejects_unknown():
    from benchmarks.runner import _execution_for_exp

    assert _execution_for_exp("EXP-p0-03-baseline-cudagraph", None) == "cudagraph"
    assert _execution_for_exp("EXP-p0-02-baseline-fp16", None) == "eager"
    with pytest.raises(ValueError, match="unknown execution"):
        _execution_for_exp("EXP-p0-03-baseline-cudagraph", "paged")


def test_correctness_failure_result_keeps_performance_unmeasured():
    from benchmarks.cudagraph import failed_result

    payload = failed_result(
        exp_id="EXP-p0-03-baseline-cudagraph__ctx=1024",
        config={"ctx": 1024, "batch": 1, "max_new_tokens": 32, "warmup": 5, "timed": 20},
        correctness={"kl": 0.1, "top1_agreement": 0.5},
        note="synthetic correctness failure",
    )
    assert payload["status"] == "failed"
    assert payload["latency"]["per_token_p50_ms"] is None
    assert validate_result(payload) == []


def test_every_payload_states_that_service_performance_is_out_of_scope():
    """★ 适用范围必须**写进每一份 result.json**，不能只写在 README 里。

    本项目是 batch=1 单流测量：无并发度、无 req/s、无吞吐-延迟曲线。如果这句话
    只活在 README，那么任何一份被单独引用出去的 payload 都会看起来像"服务性能"。
    """
    from benchmarks.cudagraph import blocked_result, failed_result
    from benchmarks.schema import SCOPE_NOT_APPLICABLE

    assert "batch=1" in SCOPE_NOT_APPLICABLE and "req/s" in SCOPE_NOT_APPLICABLE

    cfg = {"ctx": 1024, "batch": 1, "max_new_tokens": 32, "warmup": 5, "timed": 20}
    blocked = blocked_result(
        exp_id="EXP-scope-01", config=cfg, failure={"stage": "capture", "type": "X", "message": "m"}
    )
    failed = failed_result(
        exp_id="EXP-scope-02",
        config=cfg,
        correctness={"kl": 0.1, "top1_agreement": 0.5},
        note="synthetic",
    )
    for payload in (blocked, failed):
        assert "batch=1" in payload["applicability"]["does_not_apply_to"], payload["exp_id"]


def test_graph_correctness_gate_requires_kl_and_top1_and_sequence():
    from benchmarks.cudagraph import graph_correctness_passed

    assert graph_correctness_passed(
        {"kl": 1e-6, "top1_agreement": 1.0, "seed_token_match": True, "token_sequence_match": True}
    )
    assert not graph_correctness_passed(
        {"kl": 1e-6, "top1_agreement": 0.5, "seed_token_match": True, "token_sequence_match": True}
    )


def test_cudagraph_config_rejects_timing_beyond_cache_horizon():
    cfg = CudaGraphConfig(ctx=8, max_new_tokens=20, batch=1)
    with pytest.raises(ValueError, match="timed"):
        CudaGraphConfig(ctx=8, max_new_tokens=19, batch=1).validate_timing(4)
    cfg.validate_timing(4)


def test_blocked_result_preserves_execution_metadata():
    from benchmarks.cudagraph import blocked_result

    payload = blocked_result(
        exp_id="EXP-p0-03-baseline-cudagraph__ctx=1024",
        config={"ctx": 1024, "batch": 1, "max_new_tokens": 32, "warmup": 5, "timed": 20},
        failure={"stage": "capture", "type": "RuntimeError", "message": "dynamic shape"},
        metadata={
            "model": {"repo": "allenai/OLMoE-1B-7B-0924", "dtype": "float16"},
            "hardware": {"gpu": "RTX 5090"},
            "runtime": {"torch": "2.8.0+cu128"},
        },
    )
    assert payload["model"]["repo"] == "allenai/OLMoE-1B-7B-0924"
    assert payload["hardware"]["gpu"] == "RTX 5090"
    assert payload["runtime"]["torch"] == "2.8.0+cu128"


@pytest.mark.gpu
@requires_cuda
def test_cuda_graph_toy_capture_replay_uses_fixed_buffer():
    """Real CUDA Graph smoke; skipped on Mac, explicit on the 5090."""
    import torch

    x = torch.zeros((1,), device="cuda")
    y = torch.zeros_like(x)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y.copy_(x + 1)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        y.copy_(x + 1)
    x.fill_(7)
    graph.replay()
    stream.synchronize()
    assert float(y.item()) == 8.0


@pytest.fixture
def tiny_olmoe():
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    torch.manual_seed(123)
    config = tf.OlmoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return tf.OlmoeForCausalLM(config).eval()


def test_real_static_cache_matches_dynamic_cache_and_reset(tiny_olmoe):
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import StaticCacheDecodeSession, compare_logits
    from benchmarks.runner import _eager_reference_logits

    ids = torch.tensor([[1, 2, 3, 4]])
    reference = _eager_reference_logits(tiny_olmoe, ids, max_new_tokens=4)
    session = StaticCacheDecodeSession(tiny_olmoe, ids, ctx=4, max_new_tokens=4, device="cpu")
    addresses = session.buffer_addresses()
    assert session.seed_token_id == reference["seed_token"]
    candidate = []
    for _ in range(4):
        candidate.append(session.decode_eager().clone())
    assert session.position == 8
    assert session.buffer_addresses() == addresses
    report = compare_logits(reference["logits"], candidate)
    assert report["kl"] <= 1e-5
    assert report["top1_agreement"] == 1.0
    first = candidate[0]
    session.reset()
    assert session.position == 4
    assert session.buffer_addresses() == addresses
    assert all(torch.count_nonzero(layer.keys[:, :, 4:]) == 0 for layer in session.cache.layers)
    torch.testing.assert_close(session.decode_eager(), first)
    with pytest.raises(RuntimeError, match="CUDA"):
        session.capture()


def test_graph_cli_on_mac_does_not_write_results(tmp_path):
    import torch

    from benchmarks.runner import main

    if torch.cuda.is_available():
        pytest.skip("this is the non-CUDA guard test")
    assert main(["--exp", "EXP-p0-03-baseline-cudagraph", "--results-dir", str(tmp_path)]) != 0
    assert list(tmp_path.iterdir()) == []


def test_runner_cli_rejects_graph_config_even_in_dry_run(tmp_path):
    """cudagraph 臂必须拒绝 batch != 1 与 ctx < 1 —— 这是图的静态形状契约。

    ★ 这里原来钉的是"ctx != 1024 就拒绝"（p0-03 的范围限制）。放开 ctx 是刻意的：
    `EXP-p0-04` 要用 ctx=4096 做 H-A/H-B 的判别（attention 随 ctx 增长，
    MoE 专家路径不随）。ctx 是**声明的实验轴**（configs/experiments.yaml），
    batch 不是 —— 把范围限制钉成测试会把"我还没跑过"变成"结论的一部分"。
    """
    from benchmarks.runner import main

    def run(extra):
        return main(
            ["--exp", "EXP-p0-03-baseline-cudagraph", "--results-dir", str(tmp_path), "--dry-run"]
            + extra
        )

    assert run(["--batch", "2"]) != 0
    assert run(["--ctx", "0"]) != 0
    assert run(["--ctx", "-1"]) != 0
    # 拒绝的路径不得留下任何结果文件
    assert list(tmp_path.iterdir()) == []
    # 而 ctx=4096 必须被接受 —— 否则 p0-04 的判别实验根本跑不了
    assert run(["--ctx", "4096"]) == 0
    assert list(tmp_path.iterdir()) == []


def test_graph_gate_rejects_nonfinite_logits_and_negative_kl():
    from benchmarks.cudagraph import graph_correctness_passed

    for kl in (float("nan"), float("inf"), -1.0):
        assert not graph_correctness_passed(
            {
                "kl": kl,
                "top1_agreement": 1.0,
                "seed_token_match": True,
                "token_sequence_match": True,
            }
        )


@pytest.mark.gpu
@requires_cuda
def test_tiny_olmoe_hf_moe_cannot_be_captured(tiny_olmoe):
    """**冻结的前提**：HF 的 `OlmoeSparseMoeBlock` 无法进 CUDA Graph。

    ★ 这条测试原来断言的是相反的事（capture 应当成功），所以在 `EXP-p0-03`
    判定为 `blocked` 之后，gate 脚本的 smoke 步骤**永远失败**，p0-04 根本跑不到。
    错的是测试，不是结果 —— 实测（RTX 5090，tiny OLMoE，1 层 4 专家）：

        hf-moe      capture FAILED: operation failed due to a previous error during capture
        static-moe  capture_success=True

    机制：`torch.where(expert_mask[e])` → `nonzero`，输出长度取决于数据，
    capture 期间 host 同步被禁止。transformers 自己在 modeling_olmoe.py 里写了
    `_can_compile_fullgraph = False  # ... (torch.where(condition) not supported)`。

    **这条测试的价值在于它会先断**：上游若把 routing 改成固定形状，
    它会失败并提醒我们"p0-03 的结论已过期"，而不是让一个过期的结论继续被引用。
    """
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import CudaGraphCaptureError, StaticCacheDecodeSession

    model = tiny_olmoe.to("cuda")
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    session = StaticCacheDecodeSession(model, ids, ctx=4, max_new_tokens=4, device="cuda")
    with pytest.raises(CudaGraphCaptureError) as excinfo:
        session.capture(warmup=2)
    # 必须在 capture 阶段就**硬失败**。静默成功才是危险的（见 MOE_CUDAGRAPH §2.4）。
    assert (excinfo.value.stage or "").startswith("capture")


@pytest.mark.gpu
@requires_cuda
def test_tiny_olmoe_static_moe_captures_and_replays(tiny_olmoe):
    """smoke：换成固定形状 MoE 之后，同一张小图应当 capture + replay 且忠实。

    ★ 参照必须是**同路径 eager**（`_static_eager_correctness`），不是
    `_eager_reference_logits`。后者走动态 KV cache，量到的是 cache 口径差
    （实测 1.698e-04），而不是"图回放是否忠实"。这是本会话修掉的测量 bug，
    这条测试是它的 GPU 侧守卫。
    """
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import StaticCacheDecodeSession
    from benchmarks.runner import (
        _static_eager_correctness,
        _static_graph_correctness,
        compare_logits,
    )
    from lemoq.kernels.moe_static import replace_moe_blocks

    model = tiny_olmoe.to("cuda")
    assert replace_moe_blocks(model) >= 1
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    session = StaticCacheDecodeSession(model, ids, ctx=4, max_new_tokens=4, device="cuda")
    assert session.capture(warmup=2)["capture_success"] is True

    reference = _static_eager_correctness(session, max_new_tokens=4)
    candidate = _static_graph_correctness(session, max_new_tokens=4)
    report = compare_logits(reference["logits"], candidate["logits"])
    assert reference["seed_token"] == candidate["seed_token"]
    assert reference["output_tokens"] == candidate["output_tokens"]
    assert report["kl"] <= 1e-5
    assert report["top1_agreement"] == 1.0


def test_timed_replay_advances_autoregressive_state():
    from benchmarks.cudagraph import replay_timed_step

    class FakeSession:
        def __init__(self):
            self.calls = []

        def replay(self, *, synchronize):
            self.calls.append(synchronize)
            return "logits"

    session = FakeSession()
    assert replay_timed_step(session) == "logits"
    assert session.calls == [True]


# ---------------------------------------------------------------------------
# ★ 跨设备比较 —— 一个直到 capture 成功才暴露出来的 bug
# ---------------------------------------------------------------------------
def test_compare_logits_accepts_tensors_on_different_devices():
    """★ 回归测试：参照在 GPU、候选在 CPU，是**正常**情况，不是错误。

    实际布局：eager 参照 logits 留在 GPU 上，而 graph 侧经
    `session.logits_cpu()` 回到 CPU。原来的实现只做 `.to(torch.float32)`
    （只改 dtype、不改设备），于是 `torch.cat` 报
    "Expected all tensors to be on the same device, but found at least two
     devices, cuda:0 and cpu!"。

    这个 bug 之所以长期潜伏：在此之前**没有任何一次 CUDA Graph capture 成功过**，
    所以正确性门从来没被执行到。它只在阻塞点被解掉之后才会出现。

    在 Mac 上用 MPS 复现同样的"两个设备"条件（没有 MPS 就跳过）。
    """
    import pytest as _pytest

    torch = _pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        _pytest.skip("需要第二个设备（MPS）才能复现跨设备场景")

    from benchmarks.cudagraph import compare_logits

    gen = torch.Generator(device="cpu").manual_seed(3)
    ref_cpu = [torch.randn(1, 32, generator=gen) for _ in range(4)]
    got_cpu = [t.clone() for t in ref_cpu]

    reference = [t.to("mps") for t in ref_cpu]  # GPU 侧
    candidate = [t.to("cpu") for t in got_cpu]  # CPU 侧

    report = compare_logits(reference, candidate)
    assert report["kl"] == 0.0 or report["kl"] < 1e-6
    assert report["top1_agreement"] == 1.0


def test_compare_logits_mixed_device_is_not_bitwise_sensitive():
    """跨设备后 KL 仍应约为 0（搬设备不应改变数值到越过门槛）。"""
    import pytest as _pytest

    torch = _pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        _pytest.skip("需要第二个设备（MPS）")

    from benchmarks.cudagraph import compare_logits

    gen = torch.Generator(device="cpu").manual_seed(5)
    base = [torch.randn(1, 64, generator=gen) for _ in range(3)]
    # ★ 必须加**逐元素**噪声，不能加常数：给所有 logits 加同一个常数在 softmax
    #   之后是完全等价的，KL 恒为 0 —— 那样这条测试就没有区分度了。
    noise = [torch.randn(1, 64, generator=gen) * 0.05 for _ in range(3)]
    perturbed = [b + n for b, n in zip(base, noise)]

    report = compare_logits([t.to("mps") for t in base], [t.to("cpu") for t in perturbed])
    assert report["kl"] > 0.0, "微扰后 KL 应大于 0，否则测试没有区分度"


def test_compare_logits_rejects_length_mismatch():
    import pytest as _pytest

    torch = _pytest.importorskip("torch")
    from benchmarks.cudagraph import compare_logits

    a = [torch.randn(1, 8)]
    b = [torch.randn(1, 8), torch.randn(1, 8)]
    with _pytest.raises(ValueError):
        compare_logits(a, b)


def test_blocked_result_does_not_relabel_a_quantized_cell_as_fp16():
    """★ P3 的第二处同类坑：被阻塞的**量化格**不得以 fp16 的名义落盘。

    实测过：`w8a16 × cudagraph` 的一次 capture OOM 写出的 payload 里 `bit_width=fp16`
    —— 那会让"这一格为什么没有数字"变成一条错的记录（它明明是 w8a16）。
    """
    from benchmarks.cudagraph import blocked_result
    from benchmarks.schema import validate_result

    payload = blocked_result(
        exp_id="EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024",
        config={
            "bit_width": "w8a16",
            "group_size": 128,
            "dequant": "fused",
            "execution": "cudagraph",
            "ctx": 1024,
            "batch": 1,
            "max_new_tokens": 32,
            "warmup": 5,
            "timed": 20,
            "dtype": "float16",
        },
        failure={"stage": "unknown", "type": "OutOfMemoryError", "message": "CUDA out of memory"},
    )
    assert payload["config"]["bit_width"] == "w8a16"
    assert payload["config"]["group_size"] == 128
    assert payload["config"]["dequant"] == "fused"
    assert "w8a16" in payload["hypothesis"]["statement"]
    assert payload["latency"]["per_token_p50_ms"] is None
    assert validate_result(payload) == []

    # fp16 的行为逐字不变
    fp16 = blocked_result(
        exp_id="EXP-p0-03-baseline-cudagraph__ctx=1024",
        config={"ctx": 1024, "batch": 1, "max_new_tokens": 32, "dtype": "float16"},
        failure={"stage": "capture", "type": "RuntimeError", "message": "x"},
    )
    assert fp16["config"]["bit_width"] == "fp16"
    assert fp16["config"]["group_size"] is None
    assert fp16["config"]["dequant"] == "none"
