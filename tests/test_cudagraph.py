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


def test_compare_logits_accepts_cpu_tensors_and_reports_graph_gate_metrics():
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import compare_logits

    reference = [torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]])]
    candidate = [torch.tensor([[[2.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]])]
    report = compare_logits(reference, candidate)
    assert report["top1_agreement"] == 1.0
    assert report["kl"] == pytest.approx(0.0, abs=1e-7)


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
    from benchmarks.runner import main

    assert (
        main(
            [
                "--exp",
                "EXP-p0-03-baseline-cudagraph",
                "--ctx",
                "4096",
                "--results-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        != 0
    )


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
def test_tiny_olmoe_static_cache_graph_capture_and_replay(tiny_olmoe):
    """Exercise the real session with a tiny model before loading the 13 GB checkpoint."""
    torch = pytest.importorskip("torch")
    from benchmarks.cudagraph import StaticCacheDecodeSession, compare_logits
    from benchmarks.runner import _eager_reference_logits, _static_graph_correctness

    model = tiny_olmoe.to("cuda")
    ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    session = StaticCacheDecodeSession(model, ids, ctx=4, max_new_tokens=4, device="cuda")
    capture = session.capture(warmup=2)
    assert capture["capture_success"] is True
    reference = _eager_reference_logits(model, ids, max_new_tokens=4)
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
