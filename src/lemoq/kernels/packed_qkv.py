"""F2 candidate — packed q/k/v projection: **one** fused dequant-GEMM launch.

`LEMOQ_PACKED_QKV=1` (default off) installs, per OLMoE attention module, one
concatenation of the three w8 `QuantLinear` projections along the output
dimension. Decode (`q_len=1`, `attn=kernel`) then computes q/k/v with a single
kernel call instead of three.

Why this is the documented fix for the worst operator
(`docs/FUSION_OPPORTUNITIES.md` §F2): `_dequant_gemm_kernel` reaches only 54.0%
of its byte floor in-graph (0.3233 ms/step, 64 calls) because `BLOCK_N=4` gives
grid 512 × 2 warps on 170 SMs ≈ 6 warp/SM (~10–12% occupancy) — too few warps
in flight to hide memory latency. Packing q/k/v along N triples the grid
(2048 → 6144 rows ⇒ 512 → 1536 programs) while each program still reads one
row-local group of weights, so the per-row arithmetic is unchanged.

Parity: the grouped w8 layout is **row-local** (codes and scale are indexed per
output row and input group), so every packed output row must equal the row the
single-projection call produced — bitwise, not merely close. The GPU test
asserts exactly that (`tests/test_packed_qkv.py`).

Scope of this first round — w8 only, `dequant=fused`, symmetric, bias-free,
`attn=kernel` decode. `o_proj` is untouched.

★ Two preconditions, hence "requested" ≠ "effective" (2026-10-24). Unlike the
other three switches (whose only precondition is `attn=kernel`), this one also
needs the cell to be **quantized**: fp16 q/k/v are `nn.Linear` and there is
nothing to pack. When the candidate is adopted as a code default it is requested
on every cell while only the quantized `attn=kernel` cells can apply it, so the
runner installs it only where `packed_qkv_applicable()` holds, records the
request (`config.packed_qkv`) and the outcome (`config.packed_qkv_effective`)
separately, and never refuses a cell just because the default is on.
Non-applicable ⇒ **no buffers, no error, the ordinary three-projection path**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SWITCH = "LEMOQ_PACKED_QKV"


def packed_qkv_enabled() -> bool:
    """Runtime switch; read at install time (the buffers are built once)."""
    from lemoq.kernels.fusion_switches import fusion_enabled  # noqa: PLC0415

    return fusion_enabled("packed_qkv")


def packed_qkv_applicable(attn: Any, quantized: bool) -> bool:
    """Both preconditions: `attn == "kernel"` **and** the cell is quantized.

    ★ "not applicable" must mean *no-op*, not *refuse*: adoption turns this
    switch into a default, and only 2 of the 23 recorded graph cells satisfy
    both preconditions. Refusing the other 21 (and every eager baseline cell)
    on a default would make the adoption refresh round unrunnable.
    """
    return str(attn) == "kernel" and bool(quantized)


def _attention_modules(model: Any) -> list[Any]:
    """Same selector as `attn_decode.replace_attention_cores` (OLMoE attention only)."""
    from lemoq.kernels.attn_decode import _attention_modules as _select  # noqa: PLC0415

    return list(_select(model))


@dataclass
class PackedQKV:
    """One concatenated w8 buffer for the q/k/v projections of one attention module."""

    packed: Any
    scale: Any
    in_features: int
    out_features: int
    group_size: int

    def project(self, x: Any) -> tuple[Any, Any, Any]:
        """Return `(q, k, v)` for `x` with one fused dequant-GEMM call."""
        from lemoq.kernels._dequant_triton import dequant_gemm_fused  # noqa: PLC0415

        flat = x.reshape(-1, self.in_features)
        if not flat.is_contiguous():
            flat = flat.contiguous()
        y = dequant_gemm_fused(
            flat,
            self.packed,
            self.scale,
            None,
            group_size=self.group_size,
            bits=8,
            symmetric=True,
            out_dtype=x.dtype,
        )
        y = y.reshape(*x.shape[:-1], 3, self.out_features)
        return y[..., 0, :], y[..., 1, :], y[..., 2, :]

    def combined(self, x: Any) -> Any:
        """Return the raw `[..., 3*out]` output (tests/debug)."""
        q, k, v = self.project(x)
        import torch  # noqa: PLC0415

        return torch.cat((q, k, v), dim=-1)


def _validate(layer: Any, q: Any, k: Any, v: Any) -> None:
    from lemoq.quant.linear import QuantLinear  # noqa: PLC0415

    for role, module in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
        if not isinstance(module, QuantLinear):
            raise ValueError(
                "packed q/k/v 需要已量化的 %s（layer=%r 得到 %s）；"
                "显式拒绝而不是静默退回三次调用" % (role, layer, type(module).__name__)
            )
        if int(module.weight_bits) != 8:
            raise NotImplementedError(
                "packed q/k/v 第一轮只支持 w8（%s 是 %d-bit）" % (role, int(module.weight_bits))
            )
        if not bool(module.spec.symmetric) or module.zero is not None:
            raise NotImplementedError("packed q/k/v 第一轮只支持对称量化（%s）" % role)
        if module.bias is not None:
            raise NotImplementedError("packed q/k/v 不支持带 bias 的投影（%s）" % role)
    shape = {
        (
            int(m.in_features),
            int(m.out_features),
            int(m.group_size),
            int(m.padded_in),
            int(m.n_groups),
        )
        for m in (q, k, v)
    }
    if len(shape) != 1:
        raise ValueError(
            "packed q/k/v 要求三个投影的 (in, out, group, padded_in, n_groups) 完全一致，得到 %r"
            % (sorted(shape),)
        )


def install_packed_qkv(model: Any) -> int:
    """Build and attach the packed q/k/v state on every attention module.

    Must run **after** quantization and **before** CUDA-Graph capture (it creates
    new buffers). Returns the number of modules installed; raises if none match.
    """
    import torch  # noqa: PLC0415

    installed = 0
    for module in _attention_modules(model):
        q = getattr(module, "q_proj", None)
        k = getattr(module, "k_proj", None)
        v = getattr(module, "v_proj", None)
        _validate(getattr(module, "layer_idx", "?"), q, k, v)
        module._lemoq_packed_qkv = PackedQKV(
            packed=torch.cat((q.packed, k.packed, v.packed), dim=0).contiguous(),
            scale=torch.cat((q.scale, k.scale, v.scale), dim=0).contiguous(),
            in_features=int(q.in_features),
            out_features=int(q.out_features),
            group_size=int(q.group_size),
        )
        installed += 1
    if not installed:
        raise RuntimeError("没有找到任何 OLMoE attention 模块；模型结构可能变了")
    return installed


def packed_qkv_installed(model: Any) -> int:
    """How many attention modules carry a packed-q/k/v state (for payload recording)."""
    return sum(
        1
        for module in _attention_modules(model)
        if getattr(module, "_lemoq_packed_qkv", None) is not None
    )


def remove_packed_qkv(model: Any) -> int:
    """Drop the packed state (A/B arms in one process, tests)."""
    removed = 0
    for module in _attention_modules(model):
        if getattr(module, "_lemoq_packed_qkv", None) is not None:
            delattr(module, "_lemoq_packed_qkv")
            removed += 1
    return removed
