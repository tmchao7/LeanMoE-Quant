"""Opt-in GPTQ-style reconstruction in the existing symmetric grouped layout.

No act-order/permutation, online scaling or new inference kernel. Hessians and
error feedback use FP32; feedback uses the FP16 scale that will actually ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class CalibrationError(ValueError):
    """Insufficient coverage or unusable numerical evidence; never silently use RTN."""


@dataclass
class HessianStats:
    gram: Any
    samples: int = 0
    seen: int = 0
    max_samples: int = 2048

    @classmethod
    def empty(cls, width: int, *, device: Any = "cpu", max_samples: int = 2048):
        if type(width) is not int or width < 1 or type(max_samples) is not int or max_samples < 1:
            raise CalibrationError("positive width/max_samples required")
        return cls(
            torch.zeros(width, width, device=device, dtype=torch.float32), max_samples=max_samples
        )

    def add(self, inputs: Any) -> None:
        if (
            not torch.is_tensor(inputs)
            or inputs.ndim < 2
            or inputs.shape[-1] != self.gram.shape[0]
            or not inputs.is_floating_point()
        ):
            raise CalibrationError("activation shape/dtype mismatch")
        x = inputs.detach().reshape(-1, inputs.shape[-1])
        if not bool(torch.isfinite(x).all()):
            raise CalibrationError("non-finite calibration activations")
        self.seen += x.shape[0]
        x = x[: max(0, self.max_samples - self.samples)].to(
            device=self.gram.device, dtype=torch.float32
        )
        if x.shape[0]:
            self.gram.add_(x.T @ x)
            self.samples += x.shape[0]

    def hessian(self, *, min_samples: int = 256):
        if (
            type(min_samples) is not int
            or min_samples < 1
            or type(self.samples) is not int
            or type(self.seen) is not int
            or type(self.max_samples) is not int
            or self.samples < min_samples
            or self.samples > self.max_samples
            or self.seen < self.samples
        ):
            raise CalibrationError("insufficient or invalid calibration coverage")
        g = self.gram
        if (
            not torch.is_tensor(g)
            or g.ndim != 2
            or g.shape[0] != g.shape[1]
            or g.dtype != torch.float32
            or not bool(torch.isfinite(g).all())
            or not torch.allclose(g, g.T, atol=1e-5, rtol=1e-5)
        ):
            raise CalibrationError("invalid FP32 Gram matrix")
        h = g * (2.0 / self.samples)
        if not bool((h.diagonal() >= 0).all()) or h.diagonal().mean().item() <= 0:
            raise CalibrationError("zero-information or negative Hessian")
        return h

    def summary(self) -> dict[str, int]:
        return {"samples": self.samples, "seen_tokens": self.seen, "sample_cap": self.max_samples}


def quantize_gptq(
    weight: Any,
    *,
    calibration: HessianStats,
    bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
    damp: float = 0.01,
    min_samples: int = 256,
) -> dict[str, Any]:
    """Blocked sequential feedback; scales are static per original weight group."""
    if not isinstance(calibration, HessianStats):
        raise CalibrationError("scheme=gptq requires explicit HessianStats")
    if (
        not torch.is_tensor(weight)
        or weight.ndim != 2
        or not weight.is_floating_point()
        or not bool(torch.isfinite(weight).all())
        or min(weight.shape) < 1
    ):
        raise CalibrationError("finite non-empty 2-D floating weight required")
    if bits != 4 or type(group_size) is not int or group_size < 2 or group_size % 2:
        raise CalibrationError("v1 supports symmetric INT4 with even group size")
    rows, width = weight.shape
    if width % group_size or type(block_size) is not int or block_size < 1:
        raise CalibrationError("group must divide input width; positive block size required")
    if type(damp) not in (int, float) or not 0 < damp <= 1:
        raise CalibrationError("damp must be finite in (0,1]")
    h = calibration.hessian(min_samples=min_samples).to(weight.device).clone()
    if h.shape != (width, width):
        raise CalibrationError("Hessian/weight dimensions differ")
    h.diagonal().add_(float(damp) * h.diagonal().mean())
    try:
        lower = torch.linalg.cholesky(h)
        upper = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    except RuntimeError as exc:
        raise CalibrationError("regularized Hessian is not positive definite") from exc
    w = weight.detach().float().clone()
    absmax = w.reshape(rows, -1, group_size).abs().amax(-1)
    scale = torch.where(absmax > 0, absmax / 7.0, torch.ones_like(absmax)).half()
    scale = torch.where(scale == 0, torch.full_like(scale, 2.0**-24), scale)
    if not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all()):
        raise CalibrationError("scales cannot be represented as positive finite FP16")
    codes = torch.empty_like(w, dtype=torch.int32)
    for start in range(0, width, block_size):
        end = min(start + block_size, width)
        local = w[:, start:end].clone()
        errors = torch.zeros_like(local)
        for j in range(end - start):
            col = start + j
            s = scale[:, col // group_size].float()
            code = (local[:, j] / s).round().clamp(-8, 7)
            codes[:, col] = code.int()
            err = (local[:, j] - code * s) / upper[col, col]
            errors[:, j] = err
            local[:, j:] -= err[:, None] * upper[col, col:end][None, :]
        if not bool(torch.isfinite(errors).all()):
            raise CalibrationError("non-finite reconstruction feedback")
        if end < width:
            w[:, end:] -= errors @ upper[start:end, end:]
    if not bool(torch.isfinite(w).all()):
        raise CalibrationError("non-finite reconstruction feedback")
    return {
        "codes": codes,
        "codes_padded": codes,
        "scale": scale,
        "zero": torch.zeros_like(scale),
        "bits": 4,
        "group_size": group_size,
        "symmetric": True,
        "shape": (rows, width),
        "n_groups": width // group_size,
        "padded_in": width,
        "device": str(weight.device),
        "calibration": calibration.summary(),
        "algorithm": {
            "name": "gptq",
            "block_size": block_size,
            "damp": damp,
            "act_order": False,
            "static_groups": True,
            "scale_dtype": "float16",
        },
    }
