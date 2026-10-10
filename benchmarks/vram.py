"""vram — 显存**分列**测量（`docs/reference/METRICS.md` §4）。

规则：
    - **不得用 `nvidia-smi` 的总量当唯一数字**——它包含缓存与其它进程，无法定位是哪一项爆的；
    - 必须分列 `weights / KV / activations / workspace`，用 allocator 峰值；
    - 用 `torch.cuda.max_memory_allocated()`（allocator 视角），不是 `memory_reserved`。

用法：

    tr = VramTracker()
    tr.mark("after_torch_import")
    model = load(...);            tr.mark("after_weights")
    kv = alloc_kv(...);           tr.mark("after_kv")
    with tr.phase("forward"):
        model(...)
    tr.finalize()
"""

from __future__ import annotations

from typing import Any


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


class VramTracker:
    """Record named allocator checkpoints and derive the four-way split."""

    def __init__(self, reset_peak: bool = True) -> None:
        self.torch = _torch()
        self.available = bool(self.torch is not None and self.torch.cuda.is_available())
        self.points: list[dict[str, Any]] = []
        self.phases: list[dict[str, Any]] = []
        if self.available and reset_peak:
            self.torch.cuda.reset_peak_memory_stats()

    # ---------- 记录 ----------
    def _allocated(self) -> int:
        return int(self.torch.cuda.memory_allocated()) if self.available else 0

    def _peak(self) -> int:
        return int(self.torch.cuda.max_memory_allocated()) if self.available else 0

    def mark(self, name: str) -> dict[str, Any]:
        pt = {"name": name, "allocated_bytes": self._allocated(), "peak_bytes": self._peak()}
        self.points.append(pt)
        return pt

    def phase(self, name: str) -> "_Phase":
        return _Phase(self, name)

    # ---------- 汇总 ----------
    def _point(self, name: str) -> int | None:
        for p in self.points:
            if p["name"] == name:
                return int(p["allocated_bytes"])
        return None

    def _phase_peak(self, name: str) -> int | None:
        for p in self.phases:
            if p["name"] == name:
                return int(p["peak_bytes"])
        return None

    def finalize(self) -> dict[str, Any]:
        if not self.available:
            return {
                "status": "not_measured",
                "reason": "no CUDA device or no torch",
                "weights_gib": None,
                "kv_gib": None,
                "activations_gib": None,
                "workspace_gib": None,
                "peak_total_gib": None,
                "method": "torch.cuda allocator checkpoints",
            }

        base = self._point("after_torch_import") or 0
        after_w = self._point("after_weights")
        after_kv = self._point("after_kv")
        fwd_peak = self._phase_peak("forward")
        peak_total = self._peak()

        def gib(n: int | None) -> float | None:
            return None if n is None else round(n / 1024**3, 4)

        weights = (after_w - base) if after_w is not None else None
        kv = (after_kv - after_w) if (after_kv is not None and after_w is not None) else None
        acts = (fwd_peak - (after_kv or after_w)) if fwd_peak is not None else None
        accounted = sum(x for x in (weights, kv, acts) if x is not None) + base
        workspace = max(peak_total - accounted, 0)

        return {
            "status": "measured",
            "method": "torch.cuda allocator checkpoints (max_memory_allocated)",
            "weights_bytes": weights,
            "kv_bytes": kv,
            "activations_bytes": acts,
            "workspace_bytes": workspace,
            "peak_total_bytes": peak_total,
            "baseline_bytes": base,
            "weights_gib": gib(weights),
            "kv_gib": gib(kv),
            "activations_gib": gib(acts),
            "workspace_gib": gib(workspace),
            "peak_total_gib": gib(peak_total),
            "checkpoints": self.points,
            "phases": self.phases,
            "note": (
                "workspace 是 peak_total 减去已归因部分的残差；"
                "它同时包含框架缓冲与 kernel 工作区，无法进一步细分"
            ),
        }


class _Phase:
    def __init__(self, tracker: VramTracker, name: str) -> None:
        self.tr = tracker
        self.name = name

    def __enter__(self) -> "_Phase":
        self._before_peak = self.tr._peak()
        return self

    def __exit__(self, *exc: Any) -> bool:
        tr = self.tr
        tr.phases.append(
            {
                "name": self.name,
                "peak_bytes": tr._peak(),
                "peak_delta_bytes": max(tr._peak() - self._before_peak, 0),
            }
        )
        return False


def split_report(tracker: VramTracker) -> dict[str, Any]:
    """Convenience wrapper: finalize and add a human-readable share breakdown."""
    rep = tracker.finalize()
    if rep.get("status") != "measured":
        return rep
    total = rep["peak_total_bytes"] or 1
    for key in ("weights", "kv", "activations", "workspace"):
        b = rep.get("%s_bytes" % key)
        rep["%s_share_pct" % key] = round(100.0 * (b or 0) / total, 1)
    return rep
