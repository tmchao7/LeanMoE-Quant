"""schema — `results/<EXP-ID>/result.json` 的形状与校验器。

约定（**这是关键**）：

    - 字段与 `experiments/TEMPLATE.md` 一一对应。模板是给人读的，本模块是给机器查的。
    - **`None` 表示"未测"**，而 `0` / `False` 表示"测到的是 0/否"。
      校验器**允许** None，但要求每个 section 都存在——缺 section 说明这次实验
      没有按模板记录，那么"对本项目来说这个结果不存在"（AGENTS.md 规则 7）。
    - 纯 Python，无第三方依赖：Mac 上默认 `python3`（3.9.6）也能跑，CI 也能跑。

用法：

    from benchmarks.schema import validate_result, empty_result
    problems = validate_result(payload)      # [] 表示通过
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = 1

# section -> {field: 允许的 Python 类型（None 总是允许）}
RESULT_SCHEMA: dict[str, dict[str, tuple[type, ...]]] = {
    "hypothesis": {
        "statement": (str,),
        "refutation_signal": (str,),
    },
    "model": {
        "repo": (str,),
        "arch": (str,),
        "dtype": (str,),
        "layers": (int,),
        "hidden": (int,),
        "num_experts": (int,),
        "top_k": (int,),
        "expert_ffn_dim": (int,),
        "total_params": (int,),
        "activated_params": (int,),
    },
    "hardware": {
        "gpu": (str,),
        "compute_capability": (str,),
        "vram_total_gb": (float, int),
        "gpu_count_used": (int,),
        "host_cpu": (str,),
        "measured_bw_gbps": (float, int),
        "measured_read_bw_gbps": (float, int),
        "driver": (str,),
        "clocks": (str,),
    },
    "runtime": {
        "python": (str,),
        "torch": (str,),
        "cuda_build": (str,),
        "transformers": (str,),
        "triton": (str,),
        "commit": (str,),
    },
    "config": {
        "bit_width": (str,),
        "group_size": (int, str),
        "dequant": (str,),
        "fusion": (str,),
        "execution": (str,),
        "kv_cache_dtype": (str,),
        "ctx": (int,),
        "batch": (int,),
        "max_new_tokens": (int,),
        "warmup": (int,),
        "timed": (int,),
    },
    "correctness": {
        "kl": (float, int),
        "top1_agreement": (float, int),
        "ppl_fp16": (float, int),
        "ppl_this": (float, int),
        "ppl_delta": (float, int),
        "gate": (str,),
        "thresholds": (dict,),
    },
    "bytes": {
        "W_total_bytes": (int,),
        "T_total_bytes": (int,),
        "kv_read_bytes": (int,),
        "breakdown": (list, dict),
    },
    "latency": {
        "ttft_p50_ms": (float, int),
        "ttft_p99_ms": (float, int),
        "per_token_p50_ms": (float, int),
        "per_token_p99_ms": (float, int),
        "tok_per_s": (float, int),
        "samples": (dict, list),
    },
    "attribution": {
        "gpu_p50_ms": (float, int),
        # ★ GPU 真正在跑 kernel 的时间（profiler），不是 event 窗口宽度。
        #   在 launch-bound 时两者差一个数量级，见 benchmarks/hostgap.py。
        "gpu_busy_ms": (float, int),
        "host_p50_ms": (float, int),
        "host_share_pct": (float, int),
        "host_share_pct_residual": (float, int),
        "kernels_per_call": (float, int),
        "dequant_share_pct": (float, int),
        "routing_share_pct": (float, int),
    },
    # 瓶颈解剖：roofline 下限与实测的差距，以及差距落在哪些 kernel 上。
    # 这是回答"量化到底能不能兑现"的唯一依据，所以是必填而非可选。
    "bottleneck": {
        "roofline_floor_T_ms": (float, int),
        "measured_p50_ms": (float, int),
        "slack_factor": (float, int),
        "gpu_total_ms_per_step": (float, int),
        "kernels_per_step": (int,),
        "bucket_ms": (dict,),
        "bucket_share_pct": (dict,),
        "top_kernels": (list,),
        "moe_block": (dict,),
        "conclusion": (str,),
    },
    "attainment": {
        "floor_w_ms": (float, int),
        "floor_t_ms": (float, int),
        "quant_attainment_pct": (float, int),
        "bw_attainment_pct": (float, int),
        "denominator_note": (str,),
    },
    "vram": {
        "weights_gib": (float, int),
        "kv_gib": (float, int),
        "activations_gib": (float, int),
        "workspace_gib": (float, int),
        "peak_total_gib": (float, int),
        "method": (str,),
    },
    "verdict": {
        "verdict": (str,),
        "outcome": (str,),
        "notes": (str,),
    },
    "applicability": {
        "applies_to": (str,),
        "does_not_apply_to": (str,),
        "why": (str,),
    },
}

REQUIRED_TOP = (
    "schema_version",
    "exp_id",
    "status",
    "hypothesis",
    "model",
    "hardware",
    "runtime",
    "config",
    "correctness",
    "bytes",
    "latency",
    "attribution",
    "bottleneck",
    "attainment",
    "vram",
    "verdict",
    "applicability",
)

VALID_STATUS = ("planned", "running", "done", "blocked", "failed")
VALID_VERDICT = ("PASS", "CASCADE", "FAIL", "planned")
VALID_OUTCOME = ("supported", "refuted", "inconclusive")


def _type_name(types: tuple[type, ...]) -> str:
    return "/".join(t.__name__ for t in types)


def validate_result(payload: Any) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["result.json 顶层必须是 object，得到 %s" % type(payload).__name__]

    for key in REQUIRED_TOP:
        if key not in payload:
            problems.append("缺少顶层字段: %s" % key)
    if problems:
        return problems

    if payload.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            "schema_version 应为 %s，得到 %r" % (SCHEMA_VERSION, payload.get("schema_version"))
        )
    if not isinstance(payload.get("exp_id"), str) or not payload["exp_id"]:
        problems.append("exp_id 必须是非空字符串")
    if payload.get("status") not in VALID_STATUS:
        problems.append(
            "status 必须是 %s 之一，得到 %r" % (list(VALID_STATUS), payload.get("status"))
        )

    for section, fields in RESULT_SCHEMA.items():
        value = payload.get(section)
        if not isinstance(value, dict):
            problems.append("section %s 必须是 object，得到 %s" % (section, type(value).__name__))
            continue
        for field, types in fields.items():
            if field not in value:
                problems.append("%s 缺少字段 %s" % (section, field))
                continue
            v = value[field]
            if v is None:
                continue  # None == 未测，允许
            if not isinstance(v, types):
                problems.append(
                    "%s.%s 类型应为 %s，得到 %s"
                    % (section, field, _type_name(types), type(v).__name__)
                )

    verdict = payload.get("verdict") or {}
    if verdict.get("verdict") not in VALID_VERDICT:
        problems.append(
            "verdict.verdict 必须是 %s 之一，得到 %r"
            % (list(VALID_VERDICT), verdict.get("verdict"))
        )
    if verdict.get("outcome") not in VALID_OUTCOME:
        problems.append(
            "verdict.outcome 必须是 %s 之一，得到 %r"
            % (list(VALID_OUTCOME), verdict.get("outcome"))
        )

    # 项目硬规则 2：正确性门槛不过，不得报性能数字
    gate = (payload.get("correctness") or {}).get("gate")
    if gate == "FAIL":
        lat = payload.get("latency") or {}
        reported = [
            k
            for k in ("per_token_p50_ms", "per_token_p99_ms", "tok_per_s")
            if lat.get(k) is not None
        ]
        if reported:
            problems.append(
                "correctness.gate == FAIL 却报了性能数字 %s —— 违反项目硬规则 2" % reported
            )
    return problems


def empty_result(exp_id: str, *, status: str = "planned") -> dict[str, Any]:
    """A template-valid skeleton. Every measured value starts as None (= not measured,
    never 0, never False)."""

    def fields(section: str) -> dict[str, Any]:
        return dict.fromkeys(RESULT_SCHEMA[section])

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "exp_id": exp_id,
        "status": status,
    }
    for section in RESULT_SCHEMA:
        payload[section] = fields(section)
    payload["verdict"] = {"verdict": "planned", "outcome": "inconclusive", "notes": None}
    payload["correctness"] = dict(payload["correctness"], gate=None, thresholds=None)
    payload["vram"] = dict(payload["vram"], method=None)
    payload["applicability"] = {"applies_to": None, "does_not_apply_to": None, "why": None}
    payload["bytes"]["breakdown"] = []
    payload["latency"]["samples"] = {}
    return payload
