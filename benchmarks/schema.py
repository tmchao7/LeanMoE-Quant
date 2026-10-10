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
        # ★ `ppl_note` **刻意不**放在 RESULT_SCHEMA 的必需字段表里：
        #   放进去会让已经落盘的 69 份历史工件全部判"缺少字段 ppl_note"。
        #   它是一条**条件性**要求（只有 ppl_* 全 null 时才需要），由
        #   `validate_ppl_declaration` 单独校验；`empty_result` 会为**新** payload
        #   创建这个键。
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

# ★ 适用范围的**固定后缀**：本项目是 batch=1 单流课题，所有格子都没有并发度、
#   没有 req/s、没有吞吐-延迟曲线。这句话必须出现在**每一份** result.json 的
#   applicability 里，而不是只写在 README 里 —— 否则任何一个单独被引用出去的
#   payload 都会看起来像"服务性能"。
#   放在 schema 而不是 runner，是因为 `benchmarks.cudagraph` 也要用它，而 runner
#   单向依赖 cudagraph（放 runner 会造成循环 import）。
SCOPE_NOT_APPLICABLE = (
    "服务/并发场景（单请求 batch=1 单流测量：无并发度、无 req/s、无吞吐-延迟曲线）—— "
    "本项目的数字不能用来推断线上服务能力；吞吐随 batch 的衰减未测"
    "（见 docs/reference/METRICS.md §6 G2）"
)

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


# PPL 声明清单（离线生成；`tools/ppl_status.py`）。存在它是为了让**已经落盘的** 98 份
# result.json 不必重写就能满足"ppl_* 要么有值、要么有声明"。
PPL_STATUS_PATH = "results/PPL_STATUS.json"

# 新 payload 自带的声明（与 `tools/ppl_status.REASON_NOT_MEASURED` 同一句）。
PPL_NOT_MEASURED_NOTE = (
    "未测：PPL 尚未写入本工件（`tools/quant_ppl.py` 已实现，见 docs/reference/METRICS.md §6 待办 G5）。"
    "现有 PPL 数字只存在于 EXP-p1-03 记录 §2 的表格里，不是工件级数据。"
)


def load_ppl_index(path: str = PPL_STATUS_PATH) -> dict[str, Any]:
    """读 PPL 声明清单。文件不存在 => 空索引（那时校验只认 payload 自带的 ppl_note）。"""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001
        return {}
    return (data.get("cells") or {}) if isinstance(data, dict) else {}


def validate_ppl_declaration(payload: Any, ppl_index: dict[str, Any] | None = None) -> list[str]:
    """`ppl_*` 三个全 null 时，必须有声明（本 payload 的 `ppl_note` 或清单里的记录）。

    ★ 为什么需要这条：`docs/reference/METRICS.md` §1 把 PPL 列为带门槛的指标，而历史上 98 份
      result.json 的 `ppl_*` **全为 null 且没有任何说明** —— 那既可能是"未测"，也
      可能是"漏记"，读者分不出来。硬规则 1 要求未测的值显式写成未测。
    """
    if not isinstance(payload, dict):
        return []
    # ★ `planned` 骨架**整份**都是 null（status=planned 就是"这个格子还没跑"），
    #   它的声明就是它自己的 status，不该再要求一句 PPL 说明 —— 否则
    #   `harness --dry-run` 产出的骨架会全部不合法。
    if payload.get("status") == "planned":
        return []
    corr = payload.get("correctness") or {}
    if not isinstance(corr, dict):
        return []
    if any(corr.get(k) is not None for k in ("ppl_fp16", "ppl_this", "ppl_delta")):
        return []  # 有值 => 不需要声明
    if isinstance(corr.get("ppl_note"), str) and corr["ppl_note"].strip():
        return []
    exp_id = payload.get("exp_id")
    if ppl_index and isinstance(exp_id, str) and exp_id in ppl_index:
        entry = ppl_index[exp_id] or {}
        if str(entry.get("status") or "").strip() and str(entry.get("reason") or "").strip():
            return []
    return [
        "correctness.ppl_* 全为 null 且无声明：必须写明'未测'及其原因"
        "（写 ppl_note，或登记到 %s）—— 否则读者分不清是未测还是漏记" % PPL_STATUS_PATH
    ]


# 项目硬规则 2 的**完整**字段清单：`correctness.gate == FAIL` 时这些字段必须为 None。
# 实测值先由 `sanitize_gate_failed_perf` 挪进 `_measurement.gate_failed_evidence`。
# 历史校验只盯 latency 三个字段；2026-10-10 评审用真实 FAIL payload 实证
# `bottleneck.measured_p50_ms` / `attribution.host_share_pct` / `attainment.*`
# 同样可被当"性能数字"引用，故一并钉死（字段名跨 section 唯一，便于平铺暂存）。
GATE_FAIL_MUST_BE_NULL: dict[str, tuple[str, ...]] = {
    "latency": (
        "ttft_p50_ms",
        "ttft_p99_ms",
        "per_token_p50_ms",
        "per_token_p99_ms",
        "tok_per_s",
        "samples",
    ),
    "attribution": (
        "gpu_p50_ms",
        "gpu_busy_ms",
        "host_p50_ms",
        "host_share_pct",
        "host_share_pct_residual",
        "kernels_per_call",
        "dequant_share_pct",
        "routing_share_pct",
    ),
    "bottleneck": (
        "measured_p50_ms",
        "slack_factor",
        "gpu_total_ms_per_step",
        "kernels_per_step",
        "bucket_ms",
        "bucket_share_pct",
        "top_kernels",
        "moe_block",
    ),
    "attainment": ("quant_attainment_pct", "bw_attainment_pct"),
}


def _reports_perf(value: Any) -> bool:
    """None = 未测；空 dict/list（如 latency.samples 的默认 {}）同样 = 没有数据。
    只有非空容器或标量（含 0.0）才算"报了性能数字"。"""
    if value is None:
        return False
    if isinstance(value, (dict, list)) and len(value) == 0:
        return False
    return True


def sanitize_gate_failed_perf(payload: Any) -> dict[str, Any]:
    """硬规则 2 的写盘执行点：gate == FAIL 时把实测性能值挪进
    `_measurement.gate_failed_evidence` 并在原位置置 None（= 未测，不得引用）。

    只在 gate == FAIL 时动作；已由 runner 暂存过的字段（setdefault）不覆盖。
    返回被挪走的 `{字段: 值}`（空 dict = 无需动作）。
    """
    if not isinstance(payload, dict) or (payload.get("correctness") or {}).get("gate") != "FAIL":
        return {}
    moved: dict[str, Any] = {}
    for section, fields in GATE_FAIL_MUST_BE_NULL.items():
        sec = payload.get(section)
        if not isinstance(sec, dict):
            continue
        for field in fields:
            value = sec.get(field)
            if _reports_perf(value):
                moved[field] = value
                sec[field] = None
    if moved:
        measurement = payload.setdefault("_measurement", {})
        if isinstance(measurement, dict):
            evidence = measurement.setdefault("gate_failed_evidence", {})
            if isinstance(evidence, dict):
                for field, value in moved.items():
                    evidence.setdefault(field, value)
    return moved


def validate_result(payload: Any, ppl_index: dict[str, Any] | None = None) -> list[str]:
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

    # 项目硬规则 2：正确性门槛不过，不得报性能数字（全字段清单见
    # GATE_FAIL_MUST_BE_NULL；写盘前用 sanitize_gate_failed_perf 挪值即可通过）
    gate = (payload.get("correctness") or {}).get("gate")
    if gate == "FAIL":
        reported = [
            "%s.%s" % (section, field)
            for section, fields in GATE_FAIL_MUST_BE_NULL.items()
            for field in fields
            if isinstance(payload.get(section), dict) and _reports_perf(payload[section].get(field))
        ]
        if reported:
            problems.append(
                "correctness.gate == FAIL 却报了性能数字 %s —— 违反项目硬规则 2"
                "（实测值只能存在于 _measurement.gate_failed_evidence）" % reported
            )
    problems.extend(validate_ppl_declaration(payload, ppl_index))
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
    payload["correctness"] = dict(payload["correctness"], gate=None, thresholds=None, ppl_note=None)
    payload["vram"] = dict(payload["vram"], method=None)
    payload["applicability"] = {"applies_to": None, "does_not_apply_to": None, "why": None}
    payload["bytes"]["breakdown"] = []
    payload["latency"]["samples"] = {}
    return payload
