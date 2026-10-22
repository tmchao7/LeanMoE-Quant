"""CPU preparation contract for the new GPU correctness diagnostic."""

from types import SimpleNamespace

import pytest
import torch

from tools import check_fusion_candidate as fc


def test_dry_run_cannot_load_model_or_create_results(monkeypatch, tmp_path, capsys):
    def bomb(*args):
        raise AssertionError("dry run must not execute GPU work")

    monkeypatch.setattr(fc, "_execute", bomb)
    out = tmp_path / "new.json"
    assert (
        fc.main(
            [
                "--model",
                "not-present",
                "--prompt-cache",
                "not-present.pt",
                "--candidate",
                "rope",
                "--json",
                str(out),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not out.exists()
    assert '"performance_measured": false' in capsys.readouterr().out


def test_prompt_windows_are_full_distinct_and_identical_for_both_cache_shapes():
    ids = torch.arange(32)
    a = fc.prompt_windows(ids, ctx=4, windows=5, offset=3)
    b = fc.prompt_windows(ids.view(1, -1), ctx=4, windows=5, offset=3)
    assert len(a) == 5 and all(torch.equal(x, y) for x, y in zip(a, b))
    assert a[0].tolist() == [[3, 4, 5, 6]] and a[-1].tolist() == [[19, 20, 21, 22]]


@pytest.mark.parametrize(
    "ids",
    [
        torch.ones(32, dtype=torch.long),
        torch.arange(5),
        torch.arange(32).float(),
        torch.arange(32).view(2, 16),
    ],
)
def test_prompt_cache_rejects_degenerate_short_or_invalid_data(ids):
    with pytest.raises(ValueError):
        fc.prompt_windows(ids, ctx=4, windows=5)


def test_candidate_env_restores_all_switches_after_exception(monkeypatch):
    for key in fc.SWITCHES.values():
        monkeypatch.setenv(key, "original")
    with pytest.raises(RuntimeError):
        with fc.candidate_env("rope"):
            assert [fc.os.environ[k] for k in fc.SWITCHES.values()] == ["0", "1", "0"]
            raise RuntimeError("failed capture")
    assert all(fc.os.environ[k] == "original" for k in fc.SWITCHES.values())


def test_teacher_forcing_uses_hf_inputs_even_if_candidate_predictions_differ():
    class Session:
        def prefill(self):
            self.current_token_id = 0
            self.seen = []
            return SimpleNamespace(logits=torch.tensor([[[1.0, 0.0, 0.0]]]))

        def decode_eager(self):
            self.seen.append(self.current_token_id)
            self.current_token_id = 2
            return torch.tensor([[[0.0, 0.0, 3.0]]])

    session = Session()
    got = fc.collect(session, steps=3, graph=False, forced=[0, 1, 0])
    assert session.seen == got["inputs"] == [0, 1, 0]
    assert got["tokens"] == [2, 2, 2] and len(got["logits"]) == 4


def test_independent_comparison_rejects_changed_logits():
    a = {"logits": [torch.tensor([[[5.0, 0.0]]])], "seed": 0, "tokens": [0]}
    b = {"logits": [torch.tensor([[[0.0, 5.0]]])], "seed": 0, "tokens": [1]}
    assert not fc._same_path(a, b)["passed"]


def test_collect_and_mask_capability_with_real_tiny_static_cache():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    from benchmarks.cudagraph import StaticCacheDecodeSession
    from lemoq.kernels.attn_decode import replace_attention_cores

    config = OlmoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(17)
    model = OlmoeForCausalLM(config).float().eval()
    session = StaticCacheDecodeSession(
        model, torch.tensor([[1, 2, 3, 4]]), ctx=4, max_new_tokens=3, device="cpu"
    )
    a = fc.collect(session, steps=3, graph=False)
    b = fc.collect(session, steps=3, graph=False, forced=a["inputs"])
    assert fc._same_path(a, b)["passed"]
    assert session.position == 7
    assert not session._decode_mask_independent()
    replace_attention_cores(model, backend="triton")  # inspect only; no CUDA launch on Mac
    assert session._decode_mask_independent()
