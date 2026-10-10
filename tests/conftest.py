"""conftest — 让测试在**两种环境**下都能跑。

    - Mac（系统 python3.9.6，无 torch，无 pyyaml）：只能跑纯 Python 的测试；
    - Mac/服务器 venv（python3.12 + torch）：全部非 GPU 测试；
    - GPU 机器：加跑 `-m gpu`。

因此这里显式把 `src/` 与仓库根加入 sys.path（不依赖 editable install），
并提供 `requires_*` 跳过标记，让每个测试自己声明依赖。
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


requires_torch = pytest.mark.skipif(not _has("torch"), reason="需要 torch")
requires_numpy = pytest.mark.skipif(not _has("numpy"), reason="需要 numpy")
requires_yaml = pytest.mark.skipif(not _has("yaml"), reason="需要 pyyaml")
requires_transformers = pytest.mark.skipif(
    not (_has("torch") and _has("transformers")), reason="需要 torch + transformers"
)


def _cuda_available() -> bool:
    if not _has("torch"):
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="需要 CUDA 设备")


@pytest.fixture(scope="session")
def olmoe_config() -> dict:
    """The **verified** OLMoE-1B-7B-0924 config values (see docs/reference/MODELS.md §3.1)."""
    return {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "num_hidden_layers": 16,
        "num_experts": 64,
        "num_experts_per_tok": 8,
        "intermediate_size": 1024,
        "vocab_size": 50304,
        "tie_word_embeddings": False,
        "max_position_embeddings": 4096,
    }
