"""quant_gate — `EXP-p1-01` 的精度门（只谈精度，不谈性能）。

## 为什么不用现有的 free-running 参照

`docs/reference/METRICS.md` §1 的定义是"**逐位置** KL" —— 同一个位置上，量化配置与 fp16
参考的 logits 分布之差。

而 `benchmarks.runner._eager_reference_logits` 是 **free-running greedy**：它喂给
模型的 token 来自它**自己**的 argmax。两条臂一旦在某一步选了不同的 token，
后续位置比的就不是同一个上下文了 —— 那时算出来的 KL 已经不是"逐位置"，而是
"两条不同轨迹"的差，数值可以任意大，也不能归因给量化。

所以这里用 **teacher forcing**：

    1. 用 fp16 模型 free-running 贪心出一串 token（这一步只需要一次）；
    2. 把**同一串 token** 强制喂给两条臂，逐位置收集 logits；
    3. 逐位置比 KL / top-1。

这样唯一的变量就是量化本身。

## 门槛来自文档，不是这里发明的

`docs/reference/METRICS.md` §1：W8 KL < 0.01、W4 KL < 0.05、top-1 > 0.95（**建议值**）。
本模块**照抄**这三个数，一个都不放宽 —— 不达标就记 FAIL 并触发
`lemoq.quant.REMEDIATION_LADDER`。**agent 不得放宽正确性门。**
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "QUANT_THRESHOLDS",
    "cell_name",
    "kl_threshold_for",
    "quant_accuracy_report",
    "teacher_forced_logits",
]


# docs/reference/METRICS.md §1 的原话：W8 ≈ <0.01；W4 ≈ <0.05；top-1 >0.95。
# ★ 这三个数是**文档里的建议值**，不是本文件调的。改它们要改文档并说明理由。
QUANT_THRESHOLDS: dict[str, Any] = {
    "kl_w8_max": 0.01,
    "kl_w4_max": 0.05,
    "top1_min": 0.95,
    "source": "docs/reference/METRICS.md §1（建议值，需按实测调整；调整必须单独成 commit）",
}


def kl_threshold_for(scheme: str) -> float:
    """`"w8a8"` / `"w8a16"` -> 0.01；`"w4a16"` -> 0.05。"""
    bits, _ = _bits_of(scheme)
    if bits == 8:
        return float(QUANT_THRESHOLDS["kl_w8_max"])
    if bits == 4:
        return float(QUANT_THRESHOLDS["kl_w4_max"])
    raise ValueError("scheme %r 的位宽不在门槛表里" % (scheme,))


def _bits_of(scheme: str) -> tuple[int, int | None]:
    from lemoq.quant.linear import parse_scheme  # noqa: PLC0415

    return parse_scheme(scheme)


def cell_name(scheme: str, group_size: int) -> str:
    """与 `configs/experiments.yaml` 的 factor 展开保持同一种命名。"""
    return "EXP-p1-01-w8-w4-accuracy__bit_width=%s_group_size=%d" % (scheme, group_size)


def teacher_forced_logits(model: Any, ids: Any, tokens: list[int], *, device: str = "cuda") -> Any:
    """在**给定的** token 序列上逐位置收集 logits（teacher forcing）。

    返回 `(1 + len(tokens), ...)` 形状的 logits 列表：第一个是 prompt 末尾那一步，
    之后每一步对应 `tokens[i-1]` 被喂进去之后的输出。

    ★ 两条臂必须传**同一个** `tokens`。自由贪心会让轨迹分叉，那时逐位置 KL
    就不再是逐位置了。
    """
    import torch  # noqa: PLC0415

    out = []
    with torch.no_grad():
        step = model(ids, use_cache=True)
        past = step.past_key_values
        out.append(step.logits[:, -1:, :].detach())
        for token in tokens:
            t = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            step = model(t, past_key_values=past, use_cache=True)
            past = step.past_key_values
            out.append(step.logits[:, -1:, :].detach())
    return out


def quant_accuracy_report(
    reference_logits: list[Any],
    candidate_logits: list[Any],
    *,
    scheme: str,
    group_size: int,
    quantize_report: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """逐位置比 KL / top-1，套用文档门槛，返回可直接落盘的判定。"""
    from benchmarks.cudagraph import compare_logits  # noqa: PLC0415

    th = dict(QUANT_THRESHOLDS)
    th.update(thresholds or {})
    if len(reference_logits) != len(candidate_logits):
        raise ValueError(
            "两条臂的位置数必须相同（teacher forcing）：%d vs %d"
            % (len(reference_logits), len(candidate_logits))
        )
    report = compare_logits(reference_logits, candidate_logits)
    kl = float(report["kl"])
    top1 = float(report["top1_agreement"])
    kl_max = float(th["kl_w8_max"]) if _bits_of(scheme)[0] == 8 else float(th["kl_w4_max"])
    kl_ok = kl < kl_max
    top1_ok = top1 >= float(th["top1_min"])
    return {
        "scheme": scheme,
        "weight_bits": _bits_of(scheme)[0],
        "act_bits": _bits_of(scheme)[1],
        "group_size": group_size,
        "method": "teacher forcing（两条臂在同一串 token 上逐位置比较）",
        "n_positions": len(reference_logits),
        "kl": kl,
        "top1_agreement": top1,
        "thresholds": {
            "kl_max": kl_max,
            "top1_min": float(th["top1_min"]),
            "source": th["source"],
        },
        "kl_pass": kl_ok,
        "top1_pass": top1_ok,
        "gate": "PASS" if (kl_ok and top1_ok) else "FAIL",
        "quantize_report": quantize_report,
        "note": (
            "FAIL 是结果，不是事故。按 lemoq.quant.REMEDIATION_LADDER 逐档重试，"
            "每一档都要重新过这道门，并标注它改动了哪些变量。"
        ),
    }
