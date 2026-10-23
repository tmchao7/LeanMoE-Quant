"""The four decode-path candidate switches: one source of truth for defaults.

`RESULTS.md` only ever publishes numbers for the **default** implementation, so the
defaults must live in code (not in a shell export) and must be recorded in every
payload — exactly like `config.rmsnorm` after P2i.

    name            env var                  default   what it switches
    fused_rope      LEMOQ_FUSED_ROPE         False     fused RoPE candidate
    fused_route     LEMOQ_FUSED_ROUTE        False     fused routing candidate
    skip_mask_fill  LEMOQ_SKIP_MASK_FILL     False     mask-fill skip candidate
    packed_qkv      LEMOQ_PACKED_QKV         False     F2 packed q/k/v projection

Adoption is a one-line change to `FUSION_DEFAULTS` per candidate **after** a valid
round (F1: three same-round A/Bs; F2: a compliant A/B), followed by the graph-cell
refresh. Env vars stay as an explicit override for A/B arms and bisection; the
payload records the *effective* state, so a flipped default is visible without
the env.
"""

from __future__ import annotations

import os
from typing import Any

ENV_VARS: dict[str, str] = {
    "fused_rope": "LEMOQ_FUSED_ROPE",
    "fused_route": "LEMOQ_FUSED_ROUTE",
    "skip_mask_fill": "LEMOQ_SKIP_MASK_FILL",
    "packed_qkv": "LEMOQ_PACKED_QKV",
}

FUSION_DEFAULTS: dict[str, bool] = {
    "fused_rope": False,
    "fused_route": False,
    "skip_mask_fill": False,
    "packed_qkv": False,
}


def fusion_enabled(name: str, env: dict[str, str] | None = None) -> bool:
    """Effective state: env override if set, else the code default."""
    if name not in ENV_VARS:
        raise KeyError("未知融合开关 %r；已声明：%s" % (name, sorted(ENV_VARS)))
    source = os.environ if env is None else env
    raw = source.get(ENV_VARS[name])
    return FUSION_DEFAULTS[name] if raw is None else raw == "1"


def fusion_states(env: dict[str, str] | None = None) -> dict[str, bool]:
    """Effective state of every switch, for the payload `config` block."""
    return {name: fusion_enabled(name, env) for name in ENV_VARS}


def fusion_payload_fields(env: dict[str, str] | None = None) -> dict[str, Any]:
    """`config` fields; one key per switch, so a cell can never silently change arm."""
    return fusion_states(env)
