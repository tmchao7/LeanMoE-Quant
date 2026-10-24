"""runner — 真正执行一个实验并把结果填进 `result.json`。

`harness.py` 只做**编排**（展开矩阵、arch 裁剪、断点续跑、写骨架）；真正的测量在这里。
这个分工是刻意的：harness 在 Mac 上就能跑通，runner 必须在有 GPU 的机器上跑。

用法：

    # 第一条基线：fp16 加载 + 前向 + 首个性级延迟/显存/归因数字
    python -m benchmarks.runner --exp EXP-p0-02-baseline-fp16 --ctx 1024

    # 看一眼要跑什么，不实际执行
    python -m benchmarks.runner --exp EXP-p0-02-baseline-fp16 --dry-run

设计约束：
    - **只填补得到的字段**，其余保持 `None`（= 未测）。绝不为了"填满"而编数字。
    - 正确性门槛没过时，**不报性能数字**（项目硬规则 2，schema 校验器会拦）。
    - 每个数字都带口径：ctx、batch、warmup/timed、是否含 launch 开销。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.bytes import ModelSpec, QuantSpec, byte_budget, weight_budget
from benchmarks.cudagraph import (
    CudaGraphCaptureError,
    CudaGraphConfig,
    CudaGraphCorrectnessError,
    StaticCacheDecodeSession,
    blocked_result,
    compare_logits,
    failed_result,
    failure_record,
    graph_correctness_passed,
    measure_graph_gap,
    normalize_execution,
)
from benchmarks.harness import is_done, write_result
from benchmarks.schema import (
    PPL_NOT_MEASURED_NOTE,
    SCOPE_NOT_APPLICABLE,
    empty_result,
    sanitize_gate_failed_perf,
)
from benchmarks.timing import summarize

PROBE_PATH = "results/machine_probe.json"


# ---------------------------------------------------------------------------
# 环境事实
# ---------------------------------------------------------------------------
def _load_probe(path: str = PROBE_PATH) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return {}


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def _hardware_from_probe(probe: dict[str, Any]) -> dict[str, Any]:
    devs = (probe.get("torch") or {}).get("devices") or []
    dev = devs[0] if devs else {}
    bw = probe.get("bandwidth") or {}
    smi = (probe.get("driver") or {}).get("nvidia_smi") or {}
    driver = ""
    if smi.get("stdout"):
        parts = [p.strip() for p in smi["stdout"].split(",")]
        if len(parts) >= 2:
            driver = parts[1]
    return {
        "gpu": dev.get("name"),
        "compute_capability": dev.get("compute_capability"),
        "vram_total_gb": dev.get("vram_total_gb"),
        "gpu_count_used": 1,
        "host_cpu": (probe.get("host") or {}).get("host_cpu"),
        # ★ 只读带宽才是 decode 的相关上限；两者都记下来
        "measured_bw_gbps": bw.get("read_only_gbps"),
        "measured_read_bw_gbps": bw.get("read_only_gbps"),
        "measured_copy_bw_gbps": bw.get("measured_gbps"),
        "driver": driver,
        "clocks": None,
    }


def _runtime() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": None,
        "cuda_build": None,
        "transformers": None,
        "triton": None,
        "commit": _git_commit(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_build"] = getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    try:
        import triton

        out["triton"] = getattr(triton, "__version__", None)
    except Exception:  # noqa: BLE001
        pass
    return out


def _model_spec(model_cfg: Any, name: str) -> ModelSpec:
    cfg = dict(model_cfg.to_dict()) if hasattr(model_cfg, "to_dict") else dict(model_cfg)
    return ModelSpec.from_hf_config(cfg, name=name)


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------
def _decode_step(model, tok, past, device, dtype):
    """One autoregressive decode step. Returns (logits_last, new_past)."""
    import torch

    with torch.no_grad():
        out = model(tok, past_key_values=past, use_cache=True)
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return nxt, out.past_key_values


def _matrix_entry(exp_id: str) -> dict[str, Any]:
    """The experiment's declaration from `configs/experiments.yaml`, or `{}`."""
    try:
        import yaml  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return {}
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "experiments.yaml"
    )
    try:
        with open(path, encoding="utf-8") as fh:
            matrix = yaml.safe_load(fh)
    except Exception:  # noqa: BLE001
        return {}
    for exp in matrix.get("experiments") or []:
        if exp.get("id") == exp_id:
            return exp
    return {}


def _execution_for_exp(exp_id: str, requested: str | None) -> str:
    """Resolve the execution arm **from the matrix**, never from the name.

    ★ 这条原来是从 EXP-ID 里找子串 "cudagraph"。新加的 `EXP-p2-02-moe-grouped-gemm`
    里没有这个子串，于是它**静默跑成了 eager 臂**（`run_baseline`），并且在没有
    外网的机器上去连 huggingface —— 报错说的是网络，真正的原因是臂选错了。
    如果那台机器有外网，它还会写出一个 `execution=eager` 却被当成 grouped 证据的
    结果文件。臂决定了结果是哪条假设的证据，只能读声明。
    """
    if requested is not None:
        return normalize_execution(requested)
    declared = (_matrix_entry(exp_id).get("config") or {}).get("execution")
    if declared is not None:
        return normalize_execution(str(declared))
    # 矩阵里没有这个实验：保留老的启发式，但只作为最后手段
    if "cudagraph" in exp_id:
        return "cudagraph"
    return "eager"


def _config_for_exp(exp_id: str, key: str) -> Any:
    """Read one key from the experiment's declared `config`, or None."""
    return (_matrix_entry(exp_id).get("config") or {}).get(key)


def _moe_for_exp(exp_id: str) -> str:
    """`config.moe` from the matrix; default to the HF path (no replacement).

    ★ 与 `_execution_for_exp` 同理：moe 决定了结果是哪一条假设的证据，
    绝不能从 EXP-ID 猜。矩阵里没有这个实验时返回 "hf" —— "没有替换 MoE"
    是唯一安全的默认。
    """
    return str((_matrix_entry(exp_id).get("config") or {}).get("moe", "hf"))


# ---------------------------------------------------------------------------
# 量化格（EXP-p2-01 / p4b-01）的三个纯函数：**全部离线可测**
# ---------------------------------------------------------------------------
def _matrix_axes(exp_id: str) -> list[str]:
    """这个实验声明的**轴**，按 yaml 里的顺序：`factors` 的键依次展开，然后 `ctx`。

    cell id 的后缀必须与 `results/` 里已有的骨架 id **逐字相同**，否则 runner 会
    写出一个新文件名、`is_done()` 也永远判不了这个格。顺序不是风格问题：
    `EXP-p4b-01-truncation__bit_width=..._execution=..._ctx=...` 的顺序就是
    factors 的声明顺序。没有轴的实验（EXP-p4-01）返回空 —— 它的 id 不带后缀。
    """
    entry = _matrix_entry(exp_id)
    axes: list[str] = []
    for key, value in entry.items():
        if key == "factors":
            axes.extend(str(k) for k in (value or {}).keys())
        elif key == "ctx":
            axes.append("ctx")
    return axes


def build_exp_id(exp_id: str, values: dict[str, Any]) -> str:
    """把解析后的轴取值按 `_matrix_axes` 的顺序拼成 cell id。

    矩阵**不认识**的实验（`_matrix_entry` 返回 `{}`）没有可声明的轴 —— 保留老的
    `__ctx=%d` 口径，因为无法区分"它本来就不带轴"和"矩阵里没有它"。
    """
    if not _matrix_entry(exp_id):
        ctx = values.get("ctx")
        return "%s__ctx=%s" % (exp_id, ctx) if ctx is not None else exp_id
    axes = _matrix_axes(exp_id)
    if not axes:
        return exp_id
    missing = [a for a in axes if values.get(a) is None]
    if missing:
        raise ValueError("cell id 缺轴取值: %s" % ", ".join(missing))
    return exp_id + "__" + "_".join("%s=%s" % (a, values[a]) for a in axes)


# ★ 正确性门的**引用表**（硬规则 2：没过门的配置不许报性能数字）。
#   数值全部来自 EXP-p1-04 的 5 窗口真实 prompt 集实测（32 步 teacher forcing）：
#       ctx=1024: results/EXP-p1-04-prompt-set-gate/probe_prompt_set.json
#       ctx=4096: results/EXP-p1-04-prompt-set-gate/probe_prompt_set_ctx4096.json
#   两张工件都由 `tools/quant_probe.py --prompt-windows 5 --include-ones-control` 产出，
#   后者是 EXP-p4b-01 记录 §3 的 **P2**（ctx 不同 → prompt 不同 → KL 是**新测量**）。
#   key = (bit_width, group_size, ctx)；targets 恒为 ("attention", "experts")（与 p1-04 逐字一致）。
#   换任何一项（g=64、ctx=2048、加 router…）就**没有**门实测 → runner 直接拒绝。
#
# ★ 门是**两个判据的合取**（docs/reference/METRICS.md §1：「KL 或 top-1 不达标 → 不报性能数字」），
#   聚合口径 = p1-04 记录自己写明的「mean_pass（METRICS 的口径）」：
#       gate = PASS  ⟺  KL 均值 < 门限  且  top-1 **均值** ≥ 0.95
#   `gate` 由 `gate_from_metrics()` 从实测值**推导**，不在表里手写 —— 手写会漂移。
#   这条规则**不改动任何已有判定**（w8a16@1024 两个判据都过；w8a8/w4a16@1024 的 KL 本就超门），
#   但它挡住了一个真实的放松：`w8a8@ctx=4096` 的 KL 均值 0.0046 过了 W8 门，
#   而 top-1 均值 0.9455 < 0.95 —— 只看 KL 就会让它发表延迟，那是放宽门。
#   （top-1 的**逐窗口最小值**只作为额外信息保留在工件里：p1-04 的 w8a16 也有一个窗口
#    0.9394 < 0.95，若按下限判门，过门的配置一个都不剩 —— 与 p1-04 已记录的 PASS 矛盾。）
GATE_CITATIONS: dict[tuple[str, int, int], dict[str, Any]] = {
    # ★ P2i（2026-10-08）：门在**融合 RMSNorm** 实现下重测过，ctx=1024 三行随之更新。
    #   "同一配置"必须包含"同一实现" —— 所以每一行都带 `rmsnorm`，工件名也与实现对应：
    #   规范名 `probe_prompt_set.json` 现在是 fused 版；HF 原链的版本归档为
    #   `probe_prompt_set_v1_hf_norm.json`（数值见 EXP-p1-04 记录的 v2 小节）。
    #   τ 阈值**一个字没动**（`KL_MAX_W8/W4`、`TOP1_MIN` 依旧），判定仍由 gate_from_metrics 推。
    # ctx=1024 —— EXP-p1-04 的五窗口实测（v2：融合 norm）
    ("w8a16", 128, 1024): {"kl": 0.001361, "top1": 0.975758, "rmsnorm": "fused"},
    ("w8a8", 128, 1024): {"kl": 0.033304, "top1": 0.963636, "rmsnorm": "fused"},
    ("w4a16", 128, 1024): {"kl": 0.093560, "top1": 0.866667, "rmsnorm": "fused"},
    # ctx=4096 —— P2 补测（EXP-p4b-01-truncation 记录 §3）；v2（融合 norm，P2i 复测）
    # ★★ 注意 `w8a8@4096` 这一行：v1 的 top-1 是 0.9455（<0.95 ⇒ FAIL，曾被当成
    #   "门是合取"的关键证据）；v2 是 0.9636（⇒ PASS）。两版都合法 —— 差别只有 3/165 个
    #   位置上的一致，而且新 norm 还改变了 fp16 参考的贪心轨迹（w2/w4 两窗口）。
    #   所以这一格的判定**不可分辨**，见 EXP-p1-04 记录 §v2 与 EXP-p4b-01 §3 的警示；
    #   表里填的是"当前实现"的实测值（规则：引用必须与跑的实现一致），不是"更好的那个"。
    ("w8a16", 128, 4096): {"kl": 0.000842, "top1": 0.975758, "rmsnorm": "fused"},
    ("w8a8", 128, 4096): {"kl": 0.004259, "top1": 0.963636, "rmsnorm": "fused"},
    ("w4a16", 128, 4096): {"kl": 0.059331, "top1": 0.866667, "rmsnorm": "fused"},
}

# docs/reference/METRICS.md §1 的门槛（**门不动**；本表只引用实测，不改门槛）
KL_MAX_W8 = 0.01
KL_MAX_W4 = 0.05
TOP1_MIN = 0.95


def gate_from_metrics(bit_width: str, kl_mean: float, top1_mean: float) -> str:
    """由**实测聚合值**推导门判定。唯一真值来源，禁止在表里手写 PASS/FAIL。"""
    kl_max = KL_MAX_W4 if str(bit_width).startswith("w4") else KL_MAX_W8
    return "PASS" if (kl_mean < kl_max and top1_mean >= TOP1_MIN) else "FAIL"


def gate_for_cell(bit_width: str, group_size: int, ctx: int) -> dict[str, Any] | None:
    """该量化配置有没有**实测过**的正确性门。None = 没测过，不许跑。

    `gate` 是推导出来的：表里只存实测的 `kl` / `top1`，判定不可能与数值漂移。
    """
    cite = GATE_CITATIONS.get((bit_width, int(group_size), int(ctx)))
    if cite is None:
        return None
    return dict(cite, gate=gate_from_metrics(bit_width, cite["kl"], cite["top1"]))


def quant_spec_for_cfg(cfg: dict[str, Any]) -> Any:
    """cell 的量化 spec（fp16 返回 None）。

    targets 必须显式写 `("attention", "experts")`：`SchemeSpec.from_flags` 的默认值
    多一个 `router`，而 EXP-p1-04 的门是**不含 router** 测的 —— 用默认值就等于
    换了一个没过门的配置（p1-01 实测：只量 router 就能让 W8 的 KL 动 14-74%）。

    `dequant` 是 `EXP-p2-01` 的轴。★ 历史兼容：老格子（p0/p4b 的量化格）没有传
    `--dequant`，落盘的是 `"none"`，而它们实际跑的**就是** separate 臂。`"none"`
    在这里归一成 `"separate"`（不改变行为），cfg 里写下来的 dequant 保持逐字不变。
    """
    bit_width = cfg.get("bit_width") or "fp16"
    if bit_width == "fp16":
        return None
    from lemoq.quant.linear import SchemeSpec  # noqa: PLC0415

    dequant = str(cfg.get("dequant") or "separate")
    if dequant == "none":
        dequant = "separate"
    return SchemeSpec.from_flags(
        str(bit_width),
        int(cfg.get("group_size") or 128),
        targets=("attention", "experts"),
        dequant=dequant,
    )


def fused_unavailable_reason() -> str | None:
    """fused 臂在这台机器上不可用的可读原因；可用时 None。

    ★ 这是 runner 的**加载前拒绝**要问的问题：一台没有 triton/CUDA 的机器上，
    `--dequant fused` 必须在加载 13.4 GB 模型之前失败。测试用 monkeypatch 覆盖它。
    """
    try:
        from lemoq.kernels._dequant_triton import unavailable_reason  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return "kernel 模块 import 失败（%s）" % type(exc).__name__
    return unavailable_reason()


def _dequant_arms(model: Any, qspec: Any) -> tuple[str | None, ...]:
    """本格要测的 dequant 臂。量化格 → `("separate", "fused")`（同进程交错）。

    ★ 即使这一格属于 separate 臂，也要**两臂都测**：K3.1 的判据是两臂的**差**，
      单臂顺序跑的数字不采信（机器状态会漂移）。fp16 格没有 dequant 语义 → `(None,)`。
    """
    if qspec is None:
        return (None,)
    from lemoq.quant.linear import supported_dequant_modes  # noqa: PLC0415

    modes = supported_dequant_modes(model)
    if "fused" not in modes:
        if qspec.dequant == "fused":
            raise RuntimeError(
                "这一格要求 fused，但本机 fused 臂不可用 —— 静默退回 separate 会让结果"
                "记成一次并不存在的融合测量"
            )
        return ("separate",)
    return ("separate", "fused")


# 同进程 round-robin 的轮数（每轮每臂各 `max(timed,20)` 步）。
DEQUANT_AB_ROUNDS = 3


def _median(xs: list[float]) -> float:
    from benchmarks.timing import percentile  # noqa: PLC0415

    return percentile(sorted(float(x) for x in xs), 50)


def _interleaved_ab_stats(
    per_arm: dict[str, list[float]],
    arms: tuple[str, ...],
    *,
    first_call_ms: dict[str, float],
    rounds: int,
) -> dict[str, Any]:
    """两臂交错测量的统计（`docs/reference/METRICS.md` §2 的口径 + MAD）。

    ★ 配对是**按步号**的：两臂交替走同一条 KV 流（第 2i 步给第一臂、2i+1 步给第二臂），
      所以第 i 对看到的 KV 长度只差 1 个 token —— 上下文增长与顺序效应都被平掉，
      可以直接做配对差。单臂顺序跑会把这部分漂移算成方案差异。
    """
    stats = {
        a: {
            "n": len(per_arm[a]),
            "p50_ms": round(_median(per_arm[a]), 4),
            "min_ms": round(min(per_arm[a]), 4),
            "max_ms": round(max(per_arm[a]), 4),
        }
        for a in arms
    }
    out: dict[str, Any] = {
        "protocol": (
            "同进程、逐步 round-robin（两臂交替，共享同一条 KV 流）；每条臂 %d 轮 × "
            ">=20 步。单臂顺序跑的数字不采信（机器漂移会被算成方案差异）。" % rounds
        ),
        "rounds": rounds,
        "arms": stats,
        "first_call_ms_excluded": first_call_ms,
        "caveat": "可分辨的最小差异约等于 MAD；比 MAD 小的差值不得当成结论",
    }
    if len(arms) == 2:
        first, second = arms
        n = min(len(per_arm[first]), len(per_arm[second]))
        diffs = [per_arm[first][i] - per_arm[second][i] for i in range(n)]
        median_delta = _median(diffs)
        mad = _median([abs(d - median_delta) for d in diffs])
        p_first, p_second = stats[first]["p50_ms"], stats[second]["p50_ms"]
        out["paired"] = {
            "n_pairs": n,
            "median_delta_ms": round(median_delta, 4),
            "mad_ms": round(mad, 4),
            "wins": {
                "%s_faster" % second: sum(1 for d in diffs if d > 0),
                "%s_faster" % first: sum(1 for d in diffs if d < 0),
            },
            "note": "delta = p50(%s) - p50(%s)，逐对相减" % (first, second),
        }
        out["ratio_fused_over_separate"] = (
            round(stats["fused"]["p50_ms"] / stats["separate"]["p50_ms"], 4)
            if "fused" in stats and "separate" in stats and stats["separate"]["p50_ms"]
            else None
        )
        out["speedup_separate_over_fused"] = (
            round(p_first / p_second, 4) if first == "separate" and p_second else None
        )
        out["dequant_share_pct"] = (
            round(100.0 * (p_first - p_second) / p_first, 2)
            if first == "separate" and p_first
            else None
        )
    return out


def quantize_for_cfg(model: Any, cfg: dict[str, Any]) -> Any:
    """按 cell 的 cfg **就地**量化；fp16 返回 None。返回 `QuantizeReport`。CPU 可测。"""
    spec = quant_spec_for_cfg(cfg)
    if spec is None:
        return None
    from lemoq.quant.linear import quantize_model  # noqa: PLC0415

    _, report = quantize_model(model, spec)
    return report


def correctness_for_cell(cfg: dict[str, Any], *, consistency_passed: bool) -> dict[str, Any]:
    """这个格的 `correctness` 段。两条互斥的口径，**绝不混用**：

    fp16     ：本配置即参考，KL/top-1 无意义 → 全 null，把守的是自洽性（consistency）。
    量化配置  ：KL **引用** GATE_CITATIONS 里同配置的 EXP-p1-04 实测（不是本格实测，
                必须写明），gate 同时要求"引用门 PASS"且"自洽性 PASS"。
                引用门 FAIL 的格子照跑（它们是 evidence），但 verdict 只能是 FAIL
                —— 它的延迟**不得**被当成性能结论引用。
    """
    bit_width = cfg.get("bit_width") or "fp16"
    base: dict[str, Any] = {
        "kl": None,
        "top1_agreement": None,
        "ppl_fp16": None,
        "ppl_this": None,
        "ppl_delta": None,
        # ★ ppl_* 三个都是 null，就必须带一句"为什么是 null"—— 否则读者分不清
        #   是"未测"还是"漏记"（schema 会校验这一条，见 validate_ppl_declaration）。
        "ppl_note": PPL_NOT_MEASURED_NOTE,
    }
    if bit_width == "fp16":
        return dict(
            base,
            gate="PASS" if consistency_passed else "FAIL",
            thresholds={
                "note": (
                    "fp16 基线：KL / top-1 对参考实现无意义（本配置即参考），故为 null。"
                    "可证伪的门槛是自洽性 —— 见 consistency。"
                ),
                "kl_max_w8": KL_MAX_W8,
                "kl_max_w4": KL_MAX_W4,
                "top1_min": TOP1_MIN,
            },
        )
    cite = gate_for_cell(bit_width, int(cfg.get("group_size") or 0), int(cfg.get("ctx") or 0))
    if cite is None:  # 主流程会先拒绝；这里是最后一道闸
        return dict(
            base,
            gate="FAIL",
            thresholds={"note": "该量化配置没有 EXP-p1-04 的门实测 —— 不得报性能数字。"},
        )
    cited_pass = cite["gate"] == "PASS"
    ctx = int(cfg.get("ctx") or 0)
    artifact = {
        1024: "EXP-p1-04-prompt-set-gate/probe_prompt_set.json",
        4096: "EXP-p1-04-prompt-set-gate/probe_prompt_set_ctx4096.json",
    }.get(ctx)
    if artifact is None:
        # 与 cite is None 同款的最后一道闸：该 ctx 没有门实测引用文件，
        # 不得静默套用 1024 的工件（2026-10-10 评审发现的映射缺口）。
        return dict(
            base,
            gate="FAIL",
            thresholds={"note": "ctx=%d 没有 EXP-p1-04 的门实测引用文件 —— 不得报性能数字。" % ctx},
        )
    return dict(
        base,
        kl=cite["kl"],
        top1_agreement=cite["top1"],
        gate="PASS" if (cited_pass and consistency_passed) else "FAIL",
        thresholds={
            "kl_max_w8": KL_MAX_W8,
            "kl_max_w4": KL_MAX_W4,
            "top1_min": TOP1_MIN,
            "note": (
                "★ kl / top-1 都是**引用** EXP-p1-04（5 窗口真实 prompt 集、ctx=%d、同 scheme/"
                "同 group_size/targets=attention+experts）的实测**均值**，不是本格实测；"
                "引用值 kl=%.6f、top-1=%.6f。门是合取（METRICS §1）：KL < %.2f 且 top-1 ≥ %.2f"
                " → %s。本格只测性能。"
                % (ctx, cite["kl"], cite["top1"], KL_MAX_W8, TOP1_MIN, cite["gate"])
            ),
            "source": artifact,
        },
    )


def _load_model(
    model_id: str,
    dtype_name: str,
    device: str,
    attn: str | None = None,
    rmsnorm: str = "fused",
):
    """Load the model, optionally overriding the attention implementation.

    ★ `attn` 是必须显式记录的实验变量，不是环境细节：OLMoE 加载后默认
    `_attn_implementation == "sdpa"`（`OlmoeSdpaAttention`），而 `EXP-p2-03`
    实测它的 FMHA 内核用 64x128 的 tile，在 q_len=1 时浪费 98.4% ——
    换成 `"eager"` 在 ctx=4096 下值 5.80x。换实现会改变数值（多项式求和顺序），
    所以它必须进 `config` 并出现在结果文件里。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]
    kwargs = {"dtype": dtype}
    if attn:
        kwargs["attn_implementation"] = attn
    AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()
    # ★ 2026-10-07（P2i）：decode 的 RMSNorm 默认走本仓库的融合 kernel。
    #   HF 的 `OlmoeRMSNorm` 每个 norm 7 个 kernel（2 cast + pow + mean + rsqrt + 2 mul），
    #   每步 65 个 norm ⇒ ~455 kernel/步、约 0.43 ms。融合后 1 个 kernel、1.26 µs/次：
    #   图内 A/B 实测全步 **1.2358×**（EXP-p4c-02 §5.6）。数值 KL 5.261e-04（vs 同一实现
    #   的 HF 公式），所以引用门必须在新实现下重测（P2i 就是这么做的）。
    #   `rmsnorm="hf"` 保留为显式退路（复现 v1/v2 的格子、做对照）。
    from lemoq.kernels.rmsnorm import replace_rms_norms  # noqa: PLC0415

    n_norm = replace_rms_norms(model, rmsnorm)
    model._lemoq_rmsnorm_impl = rmsnorm
    model._lemoq_rms_norms_replaced = int(n_norm)
    return model, dtype


def _weight_bytes_actual(model, qreport: Any = None) -> int:
    """加载后**驻留**的权重字节（区别于 `[W]`：那是每 token 实际流过的字节）。

    三类会被 `model.parameters()` 漏掉的权重，都必须补上：

    1. `QuantLinear`：packed/scale 是 **buffer**，权重参数在量化时被吃掉
       —— 不补，量化格会少算 96%（experts 占大头）；
    2. `replace_moe_blocks` 之后的 experts：`block._static_experts` 是**普通张量**，
       不是 parameter —— 不补，MoE 替换过的格子会少算 96%。
       实测：EXP-p0-04 / EXP-p2-02 记的是 953,421,824 B（≈ 只剩 attention+embedding），
       比预算的 2,357,723,136 B（每 token 流过的）还低 59.6%。
       这两个格子是**已落盘的历史**，本函数只保证**今后**的格子不再犯；
    3. tie_word_embeddings 的共享（少算一次）。
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    if qreport is not None:
        from lemoq.quant.linear import QuantLinear  # noqa: PLC0415

        total += sum(
            b.numel() * b.element_size()
            for m in model.modules()
            if isinstance(m, QuantLinear)
            for b in m.buffers()
        )
    for m in model.modules():
        fused = getattr(m, "_static_experts", None)
        if fused is None:
            continue
        # ★ `fuse_experts` 返回的是 `FusedExperts(gate_up=…, down=…)` **dataclass**，
        #   不是张量 —— 只判 hasattr(fused, "numel") 会静默漏掉整个 experts（实测
        #   漏 393,216 B，恰好是 tiny 模型全部专家权重）。所以要拆容器。
        if hasattr(fused, "numel"):
            parts = (fused,)
        elif hasattr(fused, "__dataclass_fields__"):
            parts = tuple(getattr(fused, k) for k in fused.__dataclass_fields__)
        elif isinstance(fused, (tuple, list)):
            parts = tuple(fused)
        else:
            parts = ()
        total += sum(int(t.numel()) * int(t.element_size()) for t in parts if hasattr(t, "numel"))
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        total -= (
            model.get_input_embeddings().weight.numel()
            * model.get_input_embeddings().weight.element_size()
        )
    return total


def _record_fused_expert_quantization(model: Any, qreport: Any) -> None:
    """把"在 fused 张量上量化"的 experts 计入 `QuantizeReport`（P1）。

    为什么必须补：`quantize_model` 只看得到 attention —— experts 在
    `replace_moe_blocks` 里已经被 `del block.experts` 了。不补这一笔，
    报告会说"experts 量化了 0 个"，而实际上一整条量化 expert 路径都在跑，
    于是"量化了哪些东西 / 多少个字节"这条信息是**错的**。
    """
    from lemoq.kernels.moe_grouped_quant import QuantFusedExperts  # noqa: PLC0415

    blocks = 0
    packed = 0
    scale = 0
    n_params = 0
    n_experts = 0
    for module in model.modules():
        fused = getattr(module, "_static_experts", None)
        if not isinstance(fused, QuantFusedExperts):
            continue
        blocks += 1
        n_experts += int(fused.num_experts)
        for tensor in (fused.packed_gate_up, fused.packed_down):
            packed += int(tensor.numel()) * int(tensor.element_size())
            n_params += int(tensor.numel())
        for tensor in (fused.scale_gate_up, fused.scale_down):
            scale += int(tensor.numel()) * int(tensor.element_size())
    if not blocks:
        return
    qreport.replaced["experts"] = qreport.replaced.get("experts", 0) + blocks
    bucket = qreport.realized_by_target.setdefault("experts", {})
    for key, value in (
        ("packed_bytes", packed),
        ("scale_bytes", scale),
        ("zero_bytes", 0),
        ("bias_bytes", 0),
    ):
        bucket[key] = bucket.get(key, 0) + int(value)
    bucket["total_bytes"] = bucket.get("total_bytes", 0) + packed + scale
    bucket["n_linear"] = bucket.get("n_linear", 0) + blocks
    bucket["n_params"] = bucket.get("n_params", 0) + n_params
    bucket["n_params_padded"] = bucket.get("n_params_padded", 0) + n_params
    bucket["n_moe_blocks"] = blocks
    bucket["n_experts"] = n_experts
    bucket["quantized_in"] = "fused_experts(kernel-side dequant, P1)"
    qreport.spec["experts_quantization_path"] = (
        "fuse_experts → 在 (E,2F,H)/(E,H,F) 上做 per-group RTN → packed int8 + fp16 scale；"
        "decode 由量化 grouped kernel 在 kernel 内解包（不物化 fp16 权重）。"
        "见 EXP-p4b-01-truncation 记录 §3 的 P1。"
    )


def _eager_reference_logits(model, ids, *, max_new_tokens: int) -> dict[str, Any]:
    """Collect eager logits and greedy token sequence for the graph gate."""
    import torch

    with torch.no_grad():
        output = model(ids, use_cache=True)
        seed_token = int(output.logits[:, -1:, :].argmax(dim=-1).item())
        token = output.logits[:, -1:, :].argmax(dim=-1)
        past = output.past_key_values
        logits: list[Any] = []
        output_tokens: list[int] = []
        for _ in range(max_new_tokens):
            output = model(token, past_key_values=past, use_cache=True)
            logits.append(output.logits[:, -1:, :].detach())
            token = output.logits[:, -1:, :].argmax(dim=-1)
            output_tokens.append(int(token.item()))
            past = output.past_key_values
    return {"seed_token": seed_token, "output_tokens": output_tokens, "logits": logits}


def _static_eager_correctness(
    session: StaticCacheDecodeSession, *, max_new_tokens: int
) -> dict[str, Any]:
    """Same graph path, executed **eagerly** — the reference the gate actually needs.

    ★ 为什么不能用 `_eager_reference_logits` 当参照：

    那个函数走的是**动态 KV cache**，而 graph 走的是 **StaticCache**。两者
    数值上本来就不一样（实测 KL = 1.698e-04，top-1 仍然 1.0），因为：

      - StaticCache 每步 attend 全部 max_cache_len（1056）个位置，靠 mask 屏蔽；
        动态 cache 只 attend 已填充的 ~33 个位置 —— 归约顺序与项数都不同；
      - mask 用 `finfo.min`（-65504）而不是 -inf，softmax 的分母项数也不同。

    拿它当参照，量到的是"两种 cache 的口径差"，而不是"图回放是否忠实"。

    实测四种组合的 KL（RTX 5090 / OLMoE fp16 / ctx=1024 / 32 步）：

        HF+动态      vs 固定形状+动态        = 7.481e-04   (MoE 实现差异)
        固定形状+动态 vs 固定形状+StaticCache = 1.698e-04   (cache 类型差异)
        graph        vs 固定形状+StaticCache = 0.000e+00   <-- 图本身零误差
        graph        vs HF+动态              = 7.199e-04

    所以门必须比 `decode_eager()`（同 StaticCache、同 4D mask、不进图）。
    **这是修测量口径，不是放宽门槛** —— 图回放的容差反而从 1e-5 收紧到"必须完全一致"。
    """
    import torch

    candidate: list[Any] = []
    output_tokens: list[int] = []
    session.reset()
    seed_token = session.seed_token_id
    for _ in range(max_new_tokens):
        logits = session.decode_eager()
        last = logits[:, -1:, :]
        output_tokens.append(int(last.argmax(dim=-1).item()))
        candidate.append(last.detach().to("cpu", torch.float32).clone())
    return {"seed_token": seed_token, "output_tokens": output_tokens, "logits": candidate}


def _static_graph_correctness(
    session: StaticCacheDecodeSession, *, max_new_tokens: int
) -> dict[str, Any]:
    """Run a graph sequence and return CPU logits/tokens for the correctness gate."""
    candidate: list[Any] = []
    output_tokens: list[int] = []
    session.reset()
    seed_token = session.seed_token_id
    for _ in range(max_new_tokens):
        session.replay(synchronize=True)
        candidate.append(session.logits_cpu())
        output_tokens.append(session.next_token_cpu())
    return {"seed_token": seed_token, "output_tokens": output_tokens, "logits": candidate}


def _cudagraph_metadata(model_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {"repo": model_id, "dtype": cfg.get("dtype")},
        "hardware": _hardware_from_probe(_load_probe()),
        "runtime": _runtime(),
        "cudagraph": {
            "static_cache": {
                "max_cache_len": (cfg.get("ctx") or 0) + (cfg.get("max_new_tokens") or 0),
                "layout": "continuous",
            }
        },
    }


def _graph_res(
    *,
    model: Any,
    model_id: str,
    ctx: int,
    batch: int,
    max_new_tokens: int,
    warmup: int,
    timed: int,
    session: StaticCacheDecodeSession,
    gap: dict[str, Any],
    vram: dict[str, Any],
    tr: Any,
    measured_bw_gbps: float | None,
    qreport: Any = None,
) -> dict[str, Any]:
    """这个格的 res 字典。★ `qreport` 决定字节账走 fp16 还是**量化后**的分母。

    量化格若用 fp16 的 `[W]`/`[T]` 算达成率，分母会大一倍 —— 那不是"保守"，
    是错的口径（`docs/reference/METRICS.md` §3.2）。所以 `quant` 必须透传到两个预算函数。
    """
    spec = _model_spec(model.config, model_id)
    steps = gap["stats"]["wall"]
    quant = _quant_spec_arg(qreport)
    budget = byte_budget(spec, ctx=ctx, batch=batch, quant=quant)
    weights = weight_budget(spec, quant=quant)
    attainment: dict[str, Any] = {
        "floor_w_ms": None,
        "floor_t_ms": None,
        "quant_attainment_pct": None,
        "bw_attainment_pct": None,
        "denominator_note": "measured bandwidth from results/machine_probe.json",
    }
    if measured_bw_gbps:
        floor_w = (weights["total_bytes"] / (measured_bw_gbps * 1e9)) * 1000.0
        floor_t = (budget["T"]["total_bytes"] / (measured_bw_gbps * 1e9)) * 1000.0
        attainment.update(
            floor_w_ms=round(floor_w, 4),
            floor_t_ms=round(floor_t, 4),
            quant_attainment_pct=round(100.0 * floor_w / steps["p50"], 1),
            bw_attainment_pct=round(100.0 * floor_t / steps["p50"], 1),
        )
    breakdown = {
        "gpu_total_ms_per_step": gap.get("gpu_busy_ms"),
        "kernels_per_step": int(round(gap.get("kernels_per_call", 0))),
        "bucket_ms": {"cudagraph_replay": gap.get("gpu_busy_ms")},
        "bucket_share_pct": {"cudagraph_replay": 100.0},
        # ★ 实测的 per-kernel 归因表。p0-04 的 1.59 ms 残差有两个互斥解释
        #   （额外的 DRAM 流量 vs kernel 数 × 最短执行时间），这张表是唯一的判据。
        #   之前这里是硬编码的空列表，等于把结论建立在散文上。
        "top_kernels": gap.get("top_kernels", []),
    }
    return {
        "spec": spec,
        "weight_bytes_actual": _weight_bytes_actual(model, qreport),
        "breakdown": breakdown,
        "moe_block": None,
        "ttft": {"p50": None, "p99": None, "samples": []},
        "steps": steps,
        "gap": gap,
        "vram": vram,
        "budget": budget,
        "weight": weights,
        "attainment": attainment,
        "measured_bw_gbps": measured_bw_gbps,
        "capture": dict(session.capture_info),
        "tracker": tr,
    }


def _quant_spec_arg(qreport: Any) -> Any:
    """`QuantizeReport.spec`（一个 dict）-> `bytes.QuantSpec`；没有量化时 None。"""
    if qreport is None:
        return None
    spec = qreport.spec
    return QuantSpec(
        bits=spec.get("weight_bits"),
        group_size=spec.get("group_size"),
        targets=tuple(spec.get("targets") or ()),
        keep_lm_head_fp16=bool(spec.get("keep_lm_head_fp16", True)),
    )


def _measure_static_prefill(session: StaticCacheDecodeSession, *, warmup: int, timed: int):
    """Measure StaticCache prefill separately from single-token replay."""
    import time

    import torch

    warmup = max(warmup, 2)
    timed = max(timed, 5)
    for _ in range(warmup):
        session.reset()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(timed):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.reset()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    session.reset()
    return summarize(samples)


def run_cudagraph(
    *,
    model_id: str,
    ctx: int,
    batch: int,
    max_new_tokens: int,
    warmup: int,
    timed: int,
    dtype_name: str,
    device: str = "cuda",
    moe: str = "hf",
    attn: str | None = None,
    quant_cfg: dict[str, Any] | None = None,
    rmsnorm: str = "fused",
) -> dict[str, Any]:
    """Run the StaticCache + one-token CUDA Graph path.

    `moe` 选 decode 的 MoE 实现（三种，对应三个不同的假设）：

        "hf"      HF 的逐专家 Python 循环 —— EXP-p0-03，实测 blocked
        "gather"  固定形状 gather + bmm（moe_static；**旧名 static**，EXP-p0-04 的矩阵
                  与 --static-moe 别名都写 static，main 会归一到 gather）—— EXP-p0-04
        "grouped" Triton grouped GEMV，无 gather 拷贝（moe_grouped）—— EXP-p2-02

    `quant_cfg` 非 None 时在同一进程里就地量化（P3）。顺序**不能反**，且都在 capture 之前：

        replace_moe_blocks(impl=moe, quant=qspec)   # experts：在融合张量上量化（P1）
        quantize_for_cfg(model, quant_cfg)          # attention：QuantLinear（separate/fused）

    ★ 为什么必须在 capture 之前：替换会释放原 experts 并新建 packed 张量；量化会新建
      buffer。任何一次张量新建若发生在图内，都会破坏已捕获的指针。

    ★ 两臂（separate / fused）都能 capture：fused kernel 无 autotune、grid 是 host 常量；
      首调 JIT 编译发生在 `capture()` 自己的 warmup 相位（在专用 stream 上、capture 之前）。

    prefill 与 `moe` 无关：永远走逐专家循环（理由见 `moe_static.static_moe_forward`）。
    """
    import torch

    from benchmarks.vram import VramTracker, split_report
    from lemoq.kernels.moe_static import VALID_MOE_IMPLS

    valid = ("hf",) + tuple(VALID_MOE_IMPLS)
    if moe not in valid:
        raise ValueError("unknown moe %r; expected one of %s" % (moe, valid))
    if dtype_name != "float16":
        raise ValueError("EXP-p0-03 supports dtype=float16 only")
    if batch != 1 or ctx < 1:
        raise ValueError("the CUDA Graph arm supports batch=1 and ctx >= 1 only")
    CudaGraphConfig(ctx=ctx, max_new_tokens=max_new_tokens, batch=batch).validate_timing(timed)

    # ★ F2 候选（packed q/k/v）有两个前置：`attn=kernel` **且** 该格已量化。
    #   不适用时**不装**（no-op），而不是拒绝 —— 采纳后默认就是 True，23 个图格里
    #   只有 2 格同时满足两个前置；若仍按第一轮那样 raise，刷新轮会直接失败
    #   （2026-10-24 复核）。"请求/生效"的区分见 payload 里的两个字段。
    from lemoq.kernels.packed_qkv import packed_qkv_applicable, packed_qkv_enabled  # noqa: PLC0415

    packed_qkv_requested = packed_qkv_enabled()
    packed_qkv_modules = 0

    qspec = quant_spec_for_cfg(quant_cfg) if quant_cfg else None
    packed_qkv_eligible = packed_qkv_applicable(attn, qspec is not None)

    tr = VramTracker()
    tr.mark("after_torch_import")
    # ★ `attn="kernel"` 不是 HF 的一个实现名，是我们自己的替换：绝不能把它传给
    #   `from_pretrained(attn_implementation=...)`（那会直接报错）。模型按默认实现
    #   （sdpa）加载，然后在图**外**把 attention core 换成 kernel —— prefill 仍然走
    #   HF 的原路（q_len > 1 自动退回），所以 TTFT 不受这个替换影响。
    load_attn = None if attn == "kernel" else attn
    model, _dtype = _load_model(model_id, dtype_name, device, attn=load_attn, rmsnorm=rmsnorm)
    tr.mark("after_weights")

    moe_replaced = 0
    qreport = None
    kernel_warm: dict[str, Any] | None = None
    attn_warm: dict[str, Any] | None = None
    attn_replaced = 0
    if moe != "hf":
        # 必须在 capture 之前完成：替换会释放原 experts 并新建 fused 张量，
        # 若在图内发生就会破坏已捕获的指针。
        from lemoq.kernels.moe_static import replace_moe_blocks

        # ★ P1/P3：量化 × grouped 时 experts 在替换**内部**被量化（融合张量上），
        #   所以 spec 要传下去；fp16 臂传 None，那条路径逐字不变。
        moe_replaced = replace_moe_blocks(model, impl=moe, quant=qspec)
        torch.cuda.synchronize()
        tr.mark("after_static_moe")
        if moe == "grouped" and qspec is None:
            # ★ Triton 的首次调用会 JIT 编译。在 capture 期间做这件事会报错，
            #   而且报错信息通常指向别处。所以这里显式 warm 一次。
            #   （量化路径不需要：量化 grouped kernel 在 capture 的 warmup 相位里
            #     自己完成首调编译，见 quant_launch_plan 的静态形状约定。）
            from lemoq.kernels.moe_grouped import warm as warm_kernels

            block = model.model.layers[0].mlp
            kernel_warm = warm_kernels(
                block._static_experts,
                hidden=int(block._static_experts.gate_up.shape[2]),
                top_k=int(block._static_top_k),
            )
            torch.cuda.synchronize()
            tr.mark("after_grouped_warm")

    if qspec is not None:
        # attention（experts 已在 replace_moe_blocks 里量化掉，这里找不到它们）
        qreport = quantize_for_cfg(model, quant_cfg)
        if moe != "hf":
            _record_fused_expert_quantization(model, qreport)
        torch.cuda.synchronize()
        tr.mark("after_quantize")

    if attn == "kernel":
        # 同样的两步，同样的理由：先替换（图外），再 warm（图外），然后才 capture。
        from lemoq.kernels.attn_decode import (
            replace_attention_cores,
        )
        from lemoq.kernels.attn_decode import (
            warm as warm_attn,
        )

        attn_replaced = replace_attention_cores(model, backend="triton")
        torch.cuda.synchronize()
        tr.mark("after_attn_replace")
        if packed_qkv_eligible:
            # F2 候选：q/k/v 打包成一份 buffer（在 capture 之前建好，见 packed_qkv.py）。
            from lemoq.kernels.packed_qkv import install_packed_qkv  # noqa: PLC0415

            packed_qkv_modules = install_packed_qkv(model)
            torch.cuda.synchronize()
            tr.mark("after_packed_qkv")
        attn_warm = warm_attn(
            heads=int(model.config.num_attention_heads),
            head_dim=int(
                getattr(model.config, "head_dim", 0)
                or (model.config.hidden_size // model.config.num_attention_heads)
            ),
            kv_len=ctx + max_new_tokens,
            filled=ctx + max_new_tokens,
            dtype=torch.float16,
            device=device,
        )
        torch.cuda.synchronize()
        tr.mark("after_attn_warm")

    ids = torch.full((1, ctx), 1, dtype=torch.long, device=device)
    try:
        session = StaticCacheDecodeSession(
            model,
            ids,
            ctx=ctx,
            max_new_tokens=max_new_tokens,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001
        raise CudaGraphCaptureError("prefill", exc) from exc
    torch.cuda.synchronize()
    tr.mark("after_kv")

    try:
        ttft = _measure_static_prefill(session, warmup=warmup, timed=timed)
    except Exception as exc:  # noqa: BLE001
        raise CudaGraphCaptureError("prefill", exc) from exc

    capture = session.capture(warmup=warmup)
    with tr.phase("forward"):
        # ★ 门用**同路径 eager**做参照（见 _static_eager_correctness 的说明）。
        #   HF+动态 cache 的对比另算，作为 informational 记录 —— 那个 1.7e-4
        #   是已知的 cache 口径差，不是失败。
        try:
            reference = _static_eager_correctness(session, max_new_tokens=max_new_tokens)
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("correctness_reference", exc) from exc
        try:
            candidate = _static_graph_correctness(session, max_new_tokens=max_new_tokens)
        except CudaGraphCaptureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("replay", exc) from exc
        # informational：与 HF 原版 + 动态 cache 的差异（跨实现、跨 cache 口径）
        try:
            hf_reference = _eager_reference_logits(model, ids, max_new_tokens=max_new_tokens)
            cross = compare_logits(hf_reference["logits"], candidate["logits"])
        except Exception:  # noqa: BLE001
            cross = {"kl": None, "top1_agreement": None}
        try:
            correctness = compare_logits(reference["logits"], candidate["logits"])
            correctness["informational_vs_hf_dynamic_cache"] = {
                "kl": cross.get("kl"),
                "top1_agreement": cross.get("top1_agreement"),
                "note": (
                    "不是门槛：这一项同时含 MoE 实现差异(~7.5e-4)与 cache 口径差异"
                    "(~1.7e-4)，两者都是预期的数值差，不表示图有问题"
                ),
            }
            correctness.update(
                seed_token_match=reference["seed_token"] == candidate["seed_token"],
                token_sequence_match=reference["output_tokens"] == candidate["output_tokens"],
                eager_seed_token=reference["seed_token"],
                graph_seed_token=candidate["seed_token"],
                eager_output_tokens=reference["output_tokens"],
                graph_output_tokens=candidate["output_tokens"],
            )
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("correctness_compare", exc) from exc
        if not graph_correctness_passed(correctness):
            raise CudaGraphCorrectnessError(correctness)
        try:
            gap = measure_graph_gap(session, warmup=warmup, timed=timed, repeats=3)
        except CudaGraphCaptureError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise CudaGraphCaptureError("measurement", exc) from exc
    vram = split_report(tr)
    probe = _load_probe()
    bw = (probe.get("bandwidth") or {}).get("read_only_gbps")
    res = _graph_res(
        model=model,
        model_id=model_id,
        ctx=ctx,
        batch=batch,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        timed=timed,
        session=session,
        gap=gap,
        vram=vram,
        tr=tr,
        measured_bw_gbps=bw,
        qreport=qreport,
    )
    res["capture"] = capture
    res["correctness"] = correctness
    res["ttft"] = ttft
    res["moe"] = str(moe)
    res["attn"] = attn or str(getattr(model.config, "_attn_implementation", "unknown"))
    # ★ norm 实现必须进 payload：cell id 里没有这个轴，所以"这一格用的是哪个 norm"
    #   只能靠记录 —— 否则同一格会被悄悄量成两个东西（P2i）。
    from lemoq.kernels.rmsnorm import rms_norms_replaced  # noqa: PLC0415

    res["rmsnorm"] = str(getattr(model, "_lemoq_rmsnorm_impl", rmsnorm))
    res["rms_norms_replaced"] = int(rms_norms_replaced(model))
    res["packed_qkv_effective"] = bool(packed_qkv_modules)
    res["packed_qkv_modules"] = int(packed_qkv_modules)
    res["packed_qkv_requested"] = bool(packed_qkv_requested)
    res["kernel_warm"] = kernel_warm
    res["moe_blocks_replaced"] = int(moe_replaced)
    res["quantize_report"] = None if qreport is None else qreport.to_dict()
    if qreport is not None:
        # ★ P3 的证据：这一格**实际**走的 dequant 臂。capture 之后 replay 不再经过
        #   Python，所以这些计数来自 capture 之前的相位（prefill / warmup / eager 参照）
        #   —— 它们证明"被捕获的那段前向里用的是哪条臂"，而不是给 replay 计数。
        from lemoq.quant.linear import dequant_activity  # noqa: PLC0415

        res["dequant_activity"] = dequant_activity(model)
        res["dequant_arm"] = str(qspec.dequant)
    res["attn_kernel"] = {
        "warm": attn_warm,
        "cores_replaced": int(attn_replaced),
        "note": (
            "实例级替换：只作用于 q_len=1 的 decode core；prefill（q_len>1）退回 HF 原路。"
            "长度来自 StaticCache 上挂的 `_lemoq_filled_len` 设备张量。"
        ),
    }
    return res


def run_baseline(
    *,
    model_id: str,
    ctx: int,
    batch: int,
    max_new_tokens: int,
    warmup: int,
    timed: int,
    dtype_name: str,
    device: str = "cuda",
    quant_cfg: dict[str, Any] | None = None,
    moe: str = "hf",
    rmsnorm: str = "fused",
) -> dict[str, Any]:
    """Load the model, measure prefill / per-token latency / VRAM / host-gap.

    `quant_cfg` 非 None 时在测量前就地量化（`separate` 臂的全部语义：QuantLinear 每次
    forward 物化 fp16 权重）。`moe` 是 decode 的 MoE 实现（与 `run_cudagraph` 同名参数
    —— **E7 的两臂必须同一个 MoE 路径**，否则 execution 轴与 MoE 实现混在一起）。
    fp16 + `moe="hf"` 路径**逐字不变**。

    ★ **量化 × grouped** 现在是存在的（`EXP-p4b-01` 的 **P1**）：experts 在**融合后的
    (E,2F,H) 张量上**被量化（`moe_grouped_quant`），decode 走量化 grouped kernel
    （读 packed int8 + scale，kernel 内解包，不物化 fp16 权重）；attention 仍由
    `quantize_for_cfg` 换成 `QuantLinear`。两条路互不重叠 —— experts 在替换时已被
    `del block.experts`，`quantize_model` 找不到它们，所以不会重复量化。
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from benchmarks.hostgap import measure_host_gap
    from benchmarks.timing import summarize
    from benchmarks.vram import VramTracker, split_report
    from lemoq.kernels.packed_qkv import packed_qkv_enabled  # noqa: PLC0415

    # ★ eager 基线**没有** packed 路径（F2 只挂在 attn=kernel 的 decode 分支）。
    #   采纳后该开关默认 True ⇒ 这里必须是"请求但不生效"，不能再 raise：否则默认一翻，
    #   全部 eager 格（含 E7 的 8 格与所有基线）都跑不了（2026-10-24 复核）。
    packed_qkv_requested = packed_qkv_enabled()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype_name
    ]

    tr = VramTracker()
    tr.mark("after_torch_import")

    # tokenizer 先加载：确认它是可取的（本基线用合成的 token id，不需要它编码）
    AutoTokenizer.from_pretrained(model_id)

    # ★ `dtype=` 而不是 `torch_dtype=`：transformers 4.57 里 `torch_dtype` 会打
    #   deprecation 警告。**不要**传 `device_map=`：那会拉进 accelerate 依赖，
    #   而且会让权重分散到不同设备从而污染显存账。这里显式 `.to(device)`。
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model = model.to(device).eval()
    # ★ P2i：eager 臂也必须用**同一个** norm 实现，否则 execution 轴会与实现混在一起
    #   （E7 两臂的可比性同理）。默认 fused，与 `_load_model` 一致。
    from lemoq.kernels.rmsnorm import replace_rms_norms  # noqa: PLC0415

    _n_norm = replace_rms_norms(model, rmsnorm)
    model._lemoq_rmsnorm_impl = rmsnorm
    model._lemoq_rms_norms_replaced = int(_n_norm)
    tr.mark("after_weights")

    # MoE 替换：必须在量化**之前**，而且两者互斥（docstring 与 main 的拒绝）。
    # eager 臂也要能换 —— 否则 E7 的两臂只能一头 grouped 一头 hf，
    # execution 轴会与 MoE 实现混淆（硬规则 1）。
    moe_replaced = 0
    qspec = quant_spec_for_cfg(quant_cfg) if quant_cfg else None
    if moe != "hf":
        from lemoq.kernels.moe_static import replace_moe_blocks  # noqa: PLC0415

        # ★ P1：量化 × grouped 时 experts 在替换**内部**被量化（在融合后的张量上），
        #   所以 spec 要传下去。fp16 臂传 None，那条路径逐字不变。
        moe_replaced = replace_moe_blocks(model, impl=moe, quant=qspec)
    qreport = quantize_for_cfg(model, quant_cfg) if quant_cfg else None
    if qreport is not None and moe != "hf" and qspec is not None:
        _record_fused_expert_quantization(model, qreport)

    spec = _model_spec(model.config, model_id)

    # 权重实际占用（区别于 checkpoint 里的字节数）。这一步很关键：
    # OLMoE 的 checkpoint 自述是 bf16，我们按 fp16 加载，两者都是 2 字节所以应当接近，
    # 但如果差得远，说明加载路径没按预期走 —— 那样后面所有字节账都是错的。
    # ★ 三类会被 parameters() 漏掉的权重由 helper 统一补（QuantLinear buffer、
    #   _static_experts、tie），理由见 `_weight_bytes_actual` 的 docstring。
    weight_bytes_actual = _weight_bytes_actual(model, qreport)

    # 用重复的 token 构造一个长度为 ctx 的 prompt（内容对延迟无影响）
    prompt_ids = [1] * ctx
    ids = torch.tensor([prompt_ids], device=device)

    # ---- prefill (TTFT) ----
    for _ in range(max(warmup, 2)):
        _decode_step(model, ids, None, device, dtype)  # 预热
    torch.cuda.synchronize()
    tr.mark("after_kv")

    ttft_ms: list[float] = []
    for _ in range(max(warmup, 5)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(ids, use_cache=True)
            _ = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        ttft_ms.append((time.perf_counter() - t0) * 1000.0)
    ttft = summarize(ttft_ms)

    # ---- decode: per-token, 从 ctx 继续 ----
    with torch.no_grad():
        out = model(ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    with tr.phase("forward"):
        for _ in range(3):  # 打满 allocator 峰值
            nxt, past = _decode_step(model, nxt, past, device, dtype)

    def _timed_block(samples: list[float], n: int) -> None:
        """Wall-clock 单步计时（每步 synchronize），与既有的 fp16 口径逐字一致。"""
        nonlocal nxt, past
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt, past = _decode_step(model, nxt, past, device, dtype)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    # ---- 量化格：两臂**同进程交错**（EXP-p2-01 的 K3.1）----
    # ★ 为什么不能一条臂跑完再跑另一条：机器状态会漂移，漂移会被算成方案差异。
    #   这里两臂交替走**同一条 KV 流**（按步号取模），配对差可直接用。
    #   fp16 格 arms == (None,)，整段行为与历史**逐字不变**。
    dequant_ab: dict[str, Any] | None = None
    dequant_activity: dict[str, Any] | None = None
    arms = _dequant_arms(model, qspec)
    if len(arms) > 1:
        from lemoq.quant.linear import QuantLinear, reset_dequant_counters  # noqa: PLC0415
        from lemoq.quant.linear import dequant_activity as _activity  # noqa: PLC0415

        qmodules = [m for m in model.modules() if isinstance(m, QuantLinear)]

        def _set_arm(arm: str) -> None:
            for m in qmodules:
                m.dequant_mode = arm

        # 首调（Triton JIT 编译）必须与稳态分开记：几秒的编译混进 p50 就是造假。
        first_call_ms: dict[str, float] = {}
        for arm in arms:
            _set_arm(arm)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            nxt, past = _decode_step(model, nxt, past, device, dtype)
            torch.cuda.synchronize()
            first_call_ms[arm] = round((time.perf_counter() - t0) * 1000.0, 3)

        per_arm: dict[str, list[float]] = {a: [] for a in arms}
        reset_dequant_counters(model)
        block = max(int(timed), 20)
        for _ in range(DEQUANT_AB_ROUNDS):
            for i in range(block * len(arms)):
                arm = arms[i % len(arms)]
                _set_arm(arm)
                _timed_block(per_arm[arm], 1)

        dequant_activity = _activity(model)
        # ★ 硬检查：fused 臂一步都没进 kernel 的话，这个"对照"根本不是融合对照。
        #   宁可响亮失败，也不写一个会被误读成"融合无收益"的结果。
        if "fused" in arms and int(dequant_activity["fused_calls"]) == 0:
            raise RuntimeError(
                "fused 臂 fused_calls=0（fallbacks=%r）—— 这一步没有走 kernel，"
                "拒绝把 separate-vs-separate 记成融合对照" % (dequant_activity["fallbacks"],)
            )
        dequant_ab = _interleaved_ab_stats(
            per_arm, ("separate", "fused"), first_call_ms=first_call_ms, rounds=DEQUANT_AB_ROUNDS
        )
        steps = summarize(per_arm[qspec.dequant])
        arm_used = qspec.dequant
    else:
        step_ms: list[float] = []
        _timed_block(step_ms, max(timed, 20))
        steps = summarize(step_ms)
        arm_used = arms[0]

    # ---- host / kernel 拆分（固定状态下重复同一步） ----
    gap = measure_host_gap(
        lambda: _decode_step(model, nxt, past, device, dtype),
        warmup=5,
        timed=20,
        repeats=3,
    )

    # ---- 瓶颈解剖（不是可选的美化）----
    # ★ OLMoE 的 HF MoE 块对 64 个专家**逐个 Python 循环**，每步产生约 8000 个 kernel。
    #   实测单步 ~105 ms，而 roofline 下限 1.52 ms —— 差 ~70 倍。不做归因的话，
    #   会把它误当成"MoE 在 batch=1 本来就慢"，从而彻底错过真正的原因。
    breakdown = _breakdown(model, nxt, past, device, dtype)
    # ★ 原名 `moe = _bench_moe_block(...)` —— 它把**形参** moe（"hf"/"gather"/
    #   "grouped"）覆盖掉了，于是 eager 臂的 res["moe"] 变成 str(dict)，而
    #   cudagraph 臂（run_cudagraph）的 res["moe"] 是实现名字符串 —— 同一个字段
    #   两种语义。今天没人读它（build_payload 走 cfg.get("moe")），但
    #   build_cudagraph_payload 读 res.get("moe")，是地雷。改成不遮蔽。
    moe_block_bench = _bench_moe_block(model, device, dtype)

    vram = split_report(tr)

    # ---- 字节预算 ----
    # 量化格必须把 QuantSpec 传进去：否则 [W] 用 fp16 的分母算"达成率"，
    # 而量化格的分母本来就该是**打包后**的字节（docs/reference/METRICS.md §3.2）。
    qspec = quant_spec_for_cfg(quant_cfg) if quant_cfg else None
    quant = (
        QuantSpec(
            bits=qspec.weight_bits,
            group_size=qspec.group_size,
            targets=qspec.targets,
            keep_lm_head_fp16=qspec.keep_lm_head_fp16,
        )
        if qspec
        else None
    )
    budget = byte_budget(spec, ctx=ctx, batch=batch, quant=quant)
    w = weight_budget(spec, quant=quant)

    # ---- 达成率（用实测带宽，绝不用规格值） ----
    probe = _load_probe()
    bw = (probe.get("bandwidth") or {}).get("read_only_gbps")
    attainment: dict[str, Any] = {
        "floor_w_ms": None,
        "floor_t_ms": None,
        "quant_attainment_pct": None,
        "bw_attainment_pct": None,
        "denominator_note": "measured bandwidth from results/machine_probe.json",
    }
    if bw:
        floor_w = (w["total_bytes"] / (bw * 1e9)) * 1000.0
        floor_t = (budget["T"]["total_bytes"] / (bw * 1e9)) * 1000.0
        attainment.update(
            floor_w_ms=round(floor_w, 4),
            floor_t_ms=round(floor_t, 4),
            quant_attainment_pct=round(100.0 * floor_w / steps["p50"], 1),
            bw_attainment_pct=round(100.0 * floor_t / steps["p50"], 1),
        )

    return {
        "spec": spec,
        "weight_bytes_actual": weight_bytes_actual,
        "moe": str(moe),
        "moe_blocks_replaced": int(moe_replaced),
        "rmsnorm": str(getattr(model, "_lemoq_rmsnorm_impl", rmsnorm)),
        "rms_norms_replaced": int(getattr(model, "_lemoq_rms_norms_replaced", 0)),
        # F2 候选只挂在 attn=kernel 的量化 cudagraph 路径；eager 臂**生效**恒为 False
        # （请求状态记在 payload 的 `packed_qkv`，见 `fusion_payload_fields()`）。
        "packed_qkv_effective": False,
        "packed_qkv_requested": bool(packed_qkv_requested),
        "breakdown": breakdown,
        "moe_block": moe_block_bench,
        "ttft": ttft,
        "steps": steps,
        # `EXP-p2-01` 的证据：这一格**实际**走的反量化臂 + 同进程交错的 A/B。
        # fp16 格两者都是 None（没有 dequant 语义）。
        "dequant_arm": arm_used,
        "dequant_activity": dequant_activity,
        "dequant_ab": dequant_ab,
        "dequant_share_pct": (dequant_ab or {}).get("dequant_share_pct") if dequant_ab else None,
        "gap": gap,
        "vram": vram,
        "budget": budget,
        "weight": w,
        "attainment": attainment,
        "measured_bw_gbps": bw,
    }


def _gpu_time_ms(fn, *, warmup: int = 5, iters: int = 30) -> dict[str, Any]:
    """Pure GPU time via CUDA events (host launch latency excluded).

    与 wall time 的差就是"host 是否在拖后腿"。两者必须都报：这个项目的论点是
    batch=1 受带宽限制，而如果实测显示 GPU 忙等 host，那论点在这一段就不成立。
    """
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    xs: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    xs.sort()
    return {"p50": xs[len(xs) // 2], "min": xs[0], "max": xs[-1], "n": len(xs)}


def _breakdown(model, tok, past, device, dtype) -> dict[str, Any]:
    """Profile one decode step and attribute GPU time to kernels.

    `indexFuncSmallIndex` / `cub::DeviceSelectSweep` 这些不是"MoE 的代价"，
    而是 `torch.where` + 逐专家 Python 循环的代价。把它们与真正的 GEMM 分开，
    才能判断量化能救什么、不能救什么。
    """
    import torch

    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}

    state = {"tok": tok, "past": past}

    def one():
        with torch.no_grad():
            o = model(state["tok"], past_key_values=state["past"], use_cache=True)
        state["tok"] = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        state["past"] = o.past_key_values

    for _ in range(3):
        one()
    torch.cuda.synchronize()

    steps = 5
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(steps):
            one()
        torch.cuda.synchronize()

    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    evts.sort(key=lambda e: -e.self_device_time_total)
    total_ms = sum(e.self_device_time_total for e in evts) / steps / 1000.0
    kernel_count = sum(e.count for e in evts) / steps

    def _bucket(name: str) -> str:
        n = name.lower()
        if "gemvx" in n or "gemm" in n or "cutlass" in n:
            return "gemm"
        if "flash" in n or "attention" in n:
            return "attention"
        if "cub::" in n or "indexfunc" in n or "index_elementwise" in n or "write_indices" in n:
            return "routing_pack"
        if "memcpy" in n or "dtoh" in n:
            return "memcpy"
        return "other"

    buckets: dict[str, float] = {}
    for e in evts:
        buckets[_bucket(e.key)] = (
            buckets.get(_bucket(e.key), 0.0) + e.self_device_time_total / steps / 1000.0
        )
    for k in buckets:
        buckets[k] = round(buckets[k], 3)

    return {
        "gpu_total_ms_per_step": round(total_ms, 3),
        "kernels_per_step": int(kernel_count),
        "bucket_ms": dict(sorted(buckets.items(), key=lambda kv: -kv[1])),
        "top_kernels": [
            {
                "name": e.key[:120],
                "per_step_ms": round(e.self_device_time_total / steps / 1000.0, 3),
                "bucket": _bucket(e.key),
            }
            for e in evts[:8]
        ],
        "interpretation": (
            "kernel 数（~8000/步）本身就说明问题：batch=1 decode 的每一层只需要"
            "几十个 GEMM，多出来的全是 routing/pack 的逐算子开销。"
        ),
    }


def _bench_moe_block(model, device, dtype) -> dict[str, Any]:
    """单独测一个 MoE 块在 batch=1 下的耗时，作为可独立复核的最小证据。

    把它摘出来是因为整个项目的论点是围绕 MoE routing 的固定成本展开的，
    而这个成本必须能在**不加载整个模型**的情况下被验证。
    """
    import torch

    try:
        blk = model.model.layers[2].mlp
        h = int(model.config.hidden_size)
        x = torch.randn(1, 1, h, device=device, dtype=dtype)
        per_step_ms = _gpu_time_ms(lambda: blk(x), warmup=5, iters=20)
        # 纯 GEMM 部分的对照：64 个专家全算一遍需要读多少字节
        spec = ModelSpec.from_hf_config(
            model.config.to_dict(), name=str(getattr(model.config, "_name_or_path", ""))
        )
        all_experts_bytes = spec.num_experts * 3 * spec.hidden * spec.expert_ffn_dim * 2
        topk_bytes = spec.top_k * 3 * spec.hidden * spec.expert_ffn_dim * 2
        return {
            "per_call_ms": round(per_step_ms["p50"], 3),
            "called_per_token": 1,
            "n_experts_available": spec.num_experts,
            "n_experts_activated": spec.top_k,
            "bytes_if_all_experts_streamed": all_experts_bytes,
            "bytes_if_only_topk_streamed": topk_bytes,
            "note": (
                "HF 实现对全部 64 个专家都做了 gather/scatter（见 OlmoeSparseMoeBlock），"
                "而每 token 只激活 8 个。这个差值就是 EXP-p4-01 要量化的目标。"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def _consistency_gate(res: dict[str, Any]) -> dict[str, Any]:
    """The falsifiable check available at the fp16 baseline stage.

    没有量化时，"与参考比"是同义反复。此时唯一能证伪的是**自洽性**：

        floor_T = T_bytes / measured_bandwidth   <=   measured_time

    若 floor > 实测时间，说明要么字节账少算了、要么带宽高估了。两者都会污染之后
    每一个达成率，所以这一步必须**在**做量化之前就通过。
    """
    a = res.get("attainment") or {}
    p50 = (res.get("steps") or {}).get("p50")
    floor_t = a.get("floor_t_ms")
    checks = [
        {
            "name": "floor_T <= measured_p50",
            "floor_t_ms": floor_t,
            "measured_p50_ms": p50,
            "passed": bool(floor_t and p50 and floor_t <= p50),
            "why": "理论下限若高于实测时间，则字节账或带宽有误",
        },
        {
            "name": "attainment_T <= 100%",
            "bw_attainment_pct": a.get("bw_attainment_pct"),
            "passed": bool(
                a.get("bw_attainment_pct") is not None and a["bw_attainment_pct"] <= 100.0
            ),
            "why": ">100% 意味着比'只能读这么多字节'还快 —— 物理上不可能",
        },
        {
            "name": "bandwidth was measured, not assumed",
            "measured_bw_gbps": res.get("measured_bw_gbps"),
            "passed": bool(res.get("measured_bw_gbps")),
            "why": "硬规则 4：roofline 必须用实测带宽",
        },
    ]
    passed = all(c["passed"] for c in checks)
    return {
        "passed": passed,
        "checks": checks,
        "note": (
            "这是 fp16 阶段能做的唯一可证伪检查。量化阶段（EXP-p1-01 起）"
            "还会额外要求 KL / top-1 对 fp16 参考达标。"
        ),
    }


# ---------------------------------------------------------------------------
# 组装 result.json
# ---------------------------------------------------------------------------
def latency_section_for_gate(
    gate_passed: bool, measured: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """硬规则 2 的机械执行点：`gate == FAIL` 的格子不得在 `latency` 段报性能数字。

    返回 `(latency 段, gate_failed_evidence 或 None)`。

    ★ 为什么不是"FAIL 格干脆不跑"：`w8a8` / `w4a16` 的格子是"融合/执行方式**没有**改变
      正确性判定"的 evidence，必须跑（`EXP-p2-01` 记录 §4 第 3 条）。但它们的延迟
      **不得**被引用为性能结论 —— 所以原始值不删除（删掉就变成"没测过"），而是移到
      `_measurement.gate_failed_evidence`，`latency` 段全 null。
      `benchmarks/schema.py` 的校验器就是这个判定的回归测试。
    """
    if gate_passed:
        return measured, None
    null_latency = {
        "ttft_p50_ms": None,
        "ttft_p99_ms": None,
        "per_token_p50_ms": None,
        "per_token_p99_ms": None,
        "tok_per_s": None,
        "samples": {},
    }
    evidence = dict(measured)
    evidence["note"] = (
        "★ 这一格的正确性门 FAIL（见 correctness.gate）：以上是**实测原始值**，"
        "按硬规则 2 **不得**被引用为性能结论，latency 段因此为 null。"
        "它存在的意义只有一条 —— 作为'融合/执行方式不改变正确性判定'的证据。"
    )
    return null_latency, evidence


def _rmsnorm_payload_fields(res: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """P2i：decode 的 norm 实现必须进 payload。

    ★ cell id 里**没有** norm 轴，可追溯性只能靠记录；而 payload 里有**两处**
      `payload["config"] = {…}`（eager 与 cudagraph 各一处，后者会覆盖前者）——
      把字段写成一个函数、两处都展开，才不会出现"加在会被覆盖的那一块"这种事
      （第一版就是这么漏的：v3 轮第一遍的 payload 里 `config.rmsnorm` 是 None）。
    """
    return {
        "rmsnorm": res.get("rmsnorm") or cfg.get("rmsnorm") or "hf",
        "rms_norms_replaced": res.get("rms_norms_replaced"),
    }


def build_payload(
    exp_id: str, res: dict[str, Any], cfg: dict[str, Any], hypothesis: dict[str, str]
) -> dict[str, Any]:
    spec: ModelSpec = res["spec"]
    probe = _load_probe()
    steps = res["steps"]
    ttft = res["ttft"]
    gap = res["gap"]
    vram = res["vram"]

    payload = empty_result(exp_id, status="done")
    payload["hypothesis"] = hypothesis

    payload["model"] = {
        "repo": spec.name,
        "arch": "olmoe" if spec.is_moe else "dense",
        "dtype": cfg.get("dtype"),
        "layers": spec.layers,
        "hidden": spec.hidden,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "expert_ffn_dim": spec.expert_ffn_dim,
        "total_params": spec.total_params(),
        "activated_params": spec.activated_params_excl_embedding(),
    }
    payload["hardware"] = _hardware_from_probe(probe)
    payload["runtime"] = _runtime()
    from lemoq.kernels.fusion_switches import fusion_payload_fields  # noqa: PLC0415

    payload["config"] = {
        "bit_width": cfg.get("bit_width") or "fp16",
        "group_size": cfg.get("group_size"),
        "dequant": cfg.get("dequant") or "none",
        "fusion": cfg.get("fusion") or "none",
        "execution": cfg.get("execution") or "eager",
        # ★ eager 臂也要记 MoE 实现：E7 的两臂必须同一路径，不记就没法核对
        "moe": cfg.get("moe") or "hf",
        "moe_blocks_replaced": res.get("moe_blocks_replaced"),
        # ★ P2i：decode 的 RMSNorm 实现（fused = 本仓库 kernel / hf = transformers 原链）
        **_rmsnorm_payload_fields(res, cfg),
        # ★ 融合候选的**有效**状态（env 覆盖 + 代码默认）：采纳 = 改默认，记录自动跟随
        **fusion_payload_fields(),
        # ★ F2 有两个前置（attn=kernel **且** 量化）⇒ "请求"与"生效"分开记。上面那块是
        #   **请求**（与另外三个开关对称）；这里补**生效**。采纳后 21/23 个图格是
        #   "请求 True / 生效 False"——只记一个字段会把 no-op 读成"已经用上了"。
        "packed_qkv_effective": bool(res.get("packed_qkv_effective")),
        "kv_cache_dtype": "fp16",
        "ctx": cfg.get("ctx"),
        "batch": cfg.get("batch"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "warmup": cfg.get("warmup"),
        "timed": cfg.get("timed"),
    }

    # 两条互斥的口径（细节在 `correctness_for_cell`）：
    #   fp16   ：本配置**就是**参考实现，KL=0 冒充"过门"是同义反复 → 全 null，
    #            把守的是自洽性：由实测带宽和字节账推出的理论下限必须 <= 实测时间。
    #   量化格 ：KL **引用** EXP-p1-04 同配置的 5 窗口 prompt 集实测（不是本格实测），
    #            gate = 引用门 PASS 且自洽性 PASS。
    #            引用门 FAIL 的格子照跑（它们是"融合没有改变正确性判定"的 evidence），
    #            但 verdict 只能是 FAIL —— 其延迟不得作为性能结论引用。
    consistency = _consistency_gate(res)
    payload["correctness"] = correctness_for_cell(
        cfg, consistency_passed=bool(consistency["passed"])
    )
    payload["correctness"]["consistency"] = consistency
    # ★ 硬规则 2 的机械执行点：schema 只允许 **gate == PASS** 的格子往 `latency` 里写
    #   性能数字。gate FAIL 的格子（w8a8 / w4a16）**照跑**（它们是"融合没有改变正确性
    #   判定"的 evidence），但实测延迟只能进 `_measurement.gate_failed_evidence`，
    #   不得出现在 latency 段 —— 否则 `validate_result` 会直接报"违反项目硬规则 2"。
    gate_pass = payload["correctness"]["gate"] == "PASS"

    budget = res["budget"]
    payload["bytes"] = {
        "W_total_bytes": budget["W"]["total_bytes"],
        "T_total_bytes": budget["T"]["total_bytes"],
        "kv_read_bytes": budget["KV"]["read_bytes"],
        "breakdown": budget["W"]["items"],
        # ★ 预算 vs 实测：两者必须接近。差得远说明"读多少字节"这个前提错了，
        #   而那正是本项目全部结论的地基（docs/reference/METRICS.md §3）。
        "weight_bytes_actual": res.get("weight_bytes_actual"),
        "budget_vs_actual_delta_pct": (
            round(
                100.0
                * (res["weight_bytes_actual"] - budget["W"]["total_bytes"])
                / budget["W"]["total_bytes"],
                3,
            )
            if res.get("weight_bytes_actual") and budget["W"]["total_bytes"]
            else None
        ),
        "budget_vs_actual_note": (
            "W 是**每 token 实际读到的权重字节**（top-k 专家），故略小于 checkpoint 里"
            "所有专家之和；weight_bytes_actual 是加载后驻留的全部参数。两者不等是预期的，"
            "但差距应能由 expert 未激活 + lm_head 是否计入来解释。"
        ),
    }

    latency_block = {
        "ttft_p50_ms": round(ttft["p50"], 4),
        "ttft_p99_ms": round(ttft["p99"], 4),
        "per_token_p50_ms": round(steps["p50"], 4),
        "per_token_p99_ms": round(steps["p99"], 4),
        "tok_per_s": round(1000.0 / steps["p50"], 2) if steps["p50"] else None,
        "samples": {"ttft_ms": ttft["samples"], "per_token_ms": steps["samples"]},
    }
    payload["latency"], _gate_failed_evidence = latency_section_for_gate(gate_pass, latency_block)
    if _gate_failed_evidence is not None:
        payload["_measurement"] = {"gate_failed_evidence": _gate_failed_evidence}

    bd = res.get("breakdown") or {}
    bucket = bd.get("bucket_ms") or {}
    gpu_total = bd.get("gpu_total_ms_per_step")
    payload["attribution"] = {
        "gpu_p50_ms": gap.get("gpu_p50_ms"),
        "gpu_busy_ms": gap.get("gpu_busy_ms"),
        "host_p50_ms": gap.get("host_p50_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "kernels_per_call": gap.get("kernels_per_call"),
        "host_share_pct_residual": gap.get("host_share_pct_residual"),
        # ★ EXP-p2-01 的 dequant 占比：现在**是实测的** —— 来自同进程交错 A/B 的
        #   (p50_separate − p50_fused) / p50_separate，即"物化 fp16 权重"吃掉的那部分
        #   单步时间。gate FAIL 的格子不给（同上：那是性能量）。
        "dequant_share_pct": (res.get("dequant_share_pct") if gate_pass else None),
        "dequant_share_source": (
            "同进程 round-robin A/B（separate vs fused 的 p50 差 / separate p50）；"
            "不是 ncu 的 dram 计数"
            if res.get("dequant_ab")
            else None
        ),
        "routing_share_pct": (
            round(100.0 * bucket.get("routing_pack", 0.0) / gpu_total, 1) if gpu_total else None
        ),
    }

    # ★ 瓶颈解剖必须进 payload：roofline 下限与实测差约 70 倍是这条基线最重要的
    #   发现，它决定后续所有优化的顺序（先消 routing/launch，再谈量化）。
    payload["bottleneck"] = {
        "roofline_floor_T_ms": (res.get("attainment") or {}).get("floor_t_ms"),
        "measured_p50_ms": steps["p50"],
        "slack_factor": (
            round(steps["p50"] / res["attainment"]["floor_t_ms"], 1)
            if (res.get("attainment") or {}).get("floor_t_ms")
            else None
        ),
        "gpu_total_ms_per_step": gpu_total,
        "kernels_per_step": bd.get("kernels_per_step"),
        "bucket_ms": bucket,
        "bucket_share_pct": (
            {k: round(100.0 * v / gpu_total, 1) for k, v in bucket.items()} if gpu_total else {}
        ),
        "top_kernels": bd.get("top_kernels"),
        "moe_block": res.get("moe_block"),
        "conclusion": (
            "batch=1 decode 的实测时间远高于 roofline 下限，而差的不是带宽 —— "
            "是逐专家 Python 循环产生的上千个 kernel。说明本项目里"
            "**量化收益会被 kernel/launch 开销截断**（对应 PLAN.md 推论 4）。"
        ),
    }

    payload["attainment"] = res["attainment"]

    payload["vram"] = {
        "weights_gib": vram.get("weights_gib"),
        "kv_gib": vram.get("kv_gib"),
        "activations_gib": vram.get("activations_gib"),
        "workspace_gib": vram.get("workspace_gib"),
        "peak_total_gib": vram.get("peak_total_gib"),
        "method": vram.get("method"),
    }

    _gate_pass = gate_pass
    if (cfg.get("bit_width") or "fp16") == "fp16":
        _notes = (
            "第一条系统级基线，不构成任何量化结论；它建立的是测量管线与绝对数字。"
            "正确性栏位里的 KL/top-1 为 null（对本配置无意义），"
            "实际把守的是 consistency 自洽性检查。"
        )
        _applies = "fp16、单卡、batch=1、ctx=%s、%s" % (
            cfg.get("ctx"),
            payload["hardware"].get("compute_capability"),
        )
        _not_applies = "%s；%s" % (
            "量化配置（见 EXP-p1-*）、batch>1、其它架构",
            SCOPE_NOT_APPLICABLE,
        )
    else:
        _arm = res.get("dequant_arm") or cfg.get("dequant")
        _notes = (
            "量化格（%s group=%s dequant=%s，实际走的臂 = %s）。KL 引用 EXP-p1-04 的"
            "同配置实测；本格测的是性能。`separate` 臂 = QuantLinear 每次 forward 物化"
            "fp16 权重；`fused` 臂 = decode 单 token 在 kernel 内解包（prefill/多 token"
            "退回 separate，分派计数见 _measurement.dequant）。"
            % (cfg.get("bit_width"), cfg.get("group_size"), cfg.get("dequant"), _arm)
        )
        if not _gate_pass:
            _notes += (
                " ★ 正确性门 FAIL：本格的延迟**只允许**被引用为'融合/执行方式不改变"
                "正确性判定'的证据，不得作为性能结论（硬规则 2）；实测值在 "
                "_measurement.gate_failed_evidence。"
            )
        else:
            _notes += " 性能结论只对引用的这个配置成立，不能外推到其它 group_size/ctx。"
        _applies = "%s group=%s dequant=%s、单卡、batch=1、ctx=%s、%s" % (
            cfg.get("bit_width"),
            cfg.get("group_size"),
            cfg.get("dequant"),
            cfg.get("ctx"),
            payload["hardware"].get("compute_capability"),
        )
        _not_applies = "%s；%s" % (
            "其它 group_size / ctx / targets（门没有实测）、batch>1、其它架构",
            SCOPE_NOT_APPLICABLE,
        )
    payload["verdict"] = {
        "verdict": "PASS" if _gate_pass else "FAIL",
        "outcome": "supported" if _gate_pass else "refuted",
        "notes": _notes,
    }
    payload["applicability"] = {
        "applies_to": _applies,
        "does_not_apply_to": _not_applies,
        "why": "所有数字都绑定在实测带宽 %s GB/s 与该卡的 launch 开销上" % res["measured_bw_gbps"],
    }
    # gate FAIL 的格子已经写过 `_measurement`（证据块），不能在这里被覆盖掉。
    measurement = payload.setdefault("_measurement", {})
    measurement.update(
        {
            "timing_method": "perf_counter wall, synchronized per step (includes launch overhead)",
            "ctx_window": (
                "prefill at ctx=%s, then %s decode steps (KV grows by that many tokens)"
                % (cfg.get("ctx"), cfg.get("max_new_tokens"))
            ),
            "host_gap_caveats": gap.get("caveats"),
            "vram_split_note": vram.get("note"),
        }
    )
    if res.get("dequant_ab") or res.get("dequant_activity"):
        # ★ EXP-p2-01 的证据块：这一格**实际**走了哪条反量化臂（计数），以及同进程
        #   交错 A/B 的原始统计。没有它，"融合有没有收益"无法与"根本没跑到 kernel"区分。
        measurement["dequant"] = {
            "arm_recorded": res.get("dequant_arm"),
            "activity": res.get("dequant_activity"),
            "interleaved_ab": res.get("dequant_ab"),
        }
    # 硬规则 2 的 builder 出口：gate FAIL 格子的 attribution/bottleneck/attainment
    # 实测值挪进 evidence（latency 段上面已单独处理；这里是全字段收口）。
    sanitize_gate_failed_perf(payload)
    return payload


def build_cudagraph_payload(
    exp_id: str, res: dict[str, Any], cfg: dict[str, Any], hypothesis: dict[str, str]
) -> dict[str, Any]:
    """Build a result for a successful fp16 StaticCache CUDA Graph run."""
    spec: ModelSpec = res["spec"]
    probe = _load_probe()
    steps = res["steps"]
    ttft = res["ttft"]
    gap = res["gap"]
    vram = res["vram"]
    bd = res["breakdown"]
    consistency = _consistency_gate(res)
    payload = empty_result(exp_id, status="done")
    payload["hypothesis"] = hypothesis
    payload["model"] = {
        "repo": spec.name,
        "arch": "olmoe" if spec.is_moe else "dense",
        "dtype": cfg.get("dtype"),
        "layers": spec.layers,
        "hidden": spec.hidden,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
        "expert_ffn_dim": spec.expert_ffn_dim,
        "total_params": spec.total_params(),
        "activated_params": spec.activated_params_excl_embedding(),
    }
    payload["hardware"] = _hardware_from_probe(probe)
    payload["runtime"] = _runtime()
    # ★ P3：量化 × cudagraph 的 `bit_width` **必须来自 cfg**。原实现硬编码 "fp16"，
    #   于是量化格会以 fp16 的名义落盘（并且跳过引用门）—— 那违反硬规则 2，
    #   见 EXP-p4b-01-truncation 记录 §2.7 的「P3 的真实工作量」。
    bit_width = str(cfg.get("bit_width") or "fp16")
    quantized = bit_width != "fp16"
    from lemoq.kernels.fusion_switches import fusion_payload_fields  # noqa: PLC0415

    payload["config"] = {
        "bit_width": bit_width,
        "group_size": cfg.get("group_size") if quantized else None,
        "dequant": (cfg.get("dequant") if quantized else "none"),
        "fusion": "none",
        "execution": "cudagraph",
        "moe": res.get("moe", "hf"),
        "attn": res.get("attn"),
        "moe_blocks_replaced": res.get("moe_blocks_replaced"),
        # ★ 这一块**覆盖**上面那块 —— 字段必须在这里也出现（见 `_rmsnorm_payload_fields`）
        **_rmsnorm_payload_fields(res, cfg),
        # ★ 融合候选的**有效**状态（env 覆盖 + 代码默认）：采纳 = 改默认，记录自动跟随
        **fusion_payload_fields(),
        # ★ F2 有两个前置（attn=kernel **且** 量化）⇒ "请求"与"生效"分开记。上面那块是
        #   **请求**（与另外三个开关对称）；这里补**生效**。采纳后 21/23 个图格是
        #   "请求 True / 生效 False"——只记一个字段会把 no-op 读成"已经用上了"。
        "packed_qkv_effective": bool(res.get("packed_qkv_effective")),
        "kv_cache_dtype": "fp16",
        "ctx": cfg.get("ctx"),
        "batch": cfg.get("batch"),
        "max_new_tokens": cfg.get("max_new_tokens"),
        "warmup": cfg.get("warmup"),
        "timed": cfg.get("timed"),
    }
    correctness = res["correctness"]
    graph_pass = graph_correctness_passed(correctness)
    graph_block = {
        "kl": correctness["kl"],
        "top1_agreement": correctness["top1_agreement"],
        "thresholds": {
            "kl_max_fp16_graph": 1e-5,
            "top1_min_fp16_graph": 1.0,
            "note": (
                "与**同路径 eager**（同 StaticCache、同 4D mask、同 MoE，只是不进图）"
                "比较 —— 这才是'图回放是否忠实'。跨 cache / 跨实现的差异另见 "
                "informational_vs_hf_dynamic_cache，它不作为门槛。"
            ),
        },
        "passed": bool(graph_pass),
        "seed_token_match": correctness.get("seed_token_match"),
        "token_sequence_match": correctness.get("token_sequence_match"),
        "eager_seed_token": correctness.get("eager_seed_token"),
        "graph_seed_token": correctness.get("graph_seed_token"),
        "eager_output_tokens": correctness.get("eager_output_tokens"),
        "graph_output_tokens": correctness.get("graph_output_tokens"),
    }
    if quantized:
        # ★ 量化格的正确性门 = **引用门（p1-04 的 KL/top-1）∧ 图回放门**（合取）。
        #   与 eager 量化格同一条口径（`correctness_for_cell`）：KL 是**引用**的实测值，
        #   图回放忠实性进 `consistency`。原实现只用图回放门 —— 那会放行 w8a8。
        consistency = {
            "passed": bool(graph_pass),
            "checks": [dict(graph_block, name="graph_replay_is_faithful")],
            "note": (
                "本条把守的是「被 capture 的那段前向是否忠实于同路径 eager」；"
                "引用门（KL/top-1）另计，两者合取才是 gate。"
            ),
        }
        payload["correctness"] = correctness_for_cell(cfg, consistency_passed=bool(graph_pass))
        payload["correctness"]["consistency"] = consistency
        payload["correctness"]["graph_replay"] = graph_block
        payload["correctness"]["informational_vs_hf_dynamic_cache"] = correctness.get(
            "informational_vs_hf_dynamic_cache"
        )
    else:
        payload["correctness"] = {
            "kl": correctness["kl"],
            "top1_agreement": correctness["top1_agreement"],
            "ppl_fp16": None,
            "ppl_this": None,
            "ppl_delta": None,
            # ★ ppl_* 全 null 必须带声明（schema 会校验）
            "ppl_note": PPL_NOT_MEASURED_NOTE,
            "gate": "PASS" if graph_pass else "FAIL",
            "thresholds": {
                "kl_max_fp16_graph": 1e-5,
                "top1_min_fp16_graph": 1.0,
                "note": (
                    "与**同路径 eager**（同 StaticCache、同 4D mask、同 MoE，只是不进图）"
                    "比较 —— 这才是'图回放是否忠实'。跨 cache / 跨实现的差异另见 "
                    "informational_vs_hf_dynamic_cache，它不作为门槛。"
                ),
            },
            # ★ 这些字段原来被白名单丢掉了。门通过时它们是唯一的证据链：
            #   没有 token 序列与 informational KL，事后无法复核"图到底忠不忠实"。
            "seed_token_match": correctness.get("seed_token_match"),
            "token_sequence_match": correctness.get("token_sequence_match"),
            "eager_seed_token": correctness.get("eager_seed_token"),
            "graph_seed_token": correctness.get("graph_seed_token"),
            "graph_output_tokens": correctness.get("graph_output_tokens"),
            "informational_vs_hf_dynamic_cache": correctness.get(
                "informational_vs_hf_dynamic_cache"
            ),
        }
    budget = res["budget"]
    payload["bytes"] = {
        "W_total_bytes": budget["W"]["total_bytes"],
        "T_total_bytes": budget["T"]["total_bytes"],
        "kv_read_bytes": budget["KV"]["read_bytes"],
        "breakdown": budget["W"]["items"],
        "weight_bytes_actual": res["weight_bytes_actual"],
        "budget_vs_actual_delta_pct": round(
            100.0
            * (res["weight_bytes_actual"] - budget["W"]["total_bytes"])
            / budget["W"]["total_bytes"],
            3,
        ),
        "budget_vs_actual_note": "W is per-token streamed weight bytes; resident model bytes include inactive experts.",
    }
    payload["latency"] = {
        "ttft_p50_ms": round(ttft["p50"], 4),
        "ttft_p99_ms": round(ttft["p99"], 4),
        "per_token_p50_ms": round(steps["p50"], 4),
        "per_token_p99_ms": round(steps["p99"], 4),
        "tok_per_s": round(1000.0 / steps["p50"], 2) if steps["p50"] else None,
        "samples": {"ttft_ms": ttft["samples"], "per_token_ms": steps["samples"]},
    }
    # ★ 硬规则 2 同样管 cudagraph 臂：gate FAIL 的量化格（w8a8/w4a16）照跑，
    #   但延迟只能进 `_measurement.gate_failed_evidence`，`latency` 段全 null。
    payload["attribution"] = {
        "gpu_p50_ms": gap.get("gpu_p50_ms"),
        "gpu_busy_ms": gap.get("gpu_busy_ms"),
        "host_p50_ms": gap.get("host_p50_ms"),
        "host_share_pct": gap.get("host_share_pct"),
        "host_share_pct_residual": None,
        # ★ cudagraph 臂**没有** dequant 占比：图里没有 Python，separate/fused 的差
        #   要靠"两张图交错重放"才测得出（那是另一次实验）。写 None 而不是 0 ——
        #   0 会被读成"反量化在 graph 里不花时间"，那是没测过的话。
        "dequant_share_pct": None,
        "dequant_share_source": (
            None
            if not quantized
            else "未测（graph 臂的 per-call Python 开销为 0，A/B 需要两张图交错重放）"
        ),
        "routing_share_pct": None,
        "kernels_per_call": gap.get("kernels_per_call"),
    }
    payload["bottleneck"] = {
        "roofline_floor_T_ms": res["attainment"].get("floor_t_ms"),
        "measured_p50_ms": steps["p50"],
        "slack_factor": (
            round(steps["p50"] / res["attainment"]["floor_t_ms"], 1)
            if res["attainment"].get("floor_t_ms")
            else None
        ),
        "gpu_total_ms_per_step": bd.get("gpu_total_ms_per_step"),
        "kernels_per_step": bd.get("kernels_per_step"),
        "bucket_ms": bd.get("bucket_ms", {}),
        "bucket_share_pct": bd.get("bucket_share_pct", {}),
        "top_kernels": bd.get("top_kernels", []),
        "moe_block": None,
        "conclusion": (
            "这张图只包含一次固定形状的 decode 前向；量化要在这里兑现，靠的是"
            "权重字节（[W]）下降。★ 图里没有 Python —— 这正是 E7 要问的"
            "「把 host 开销拿掉之后量化还剩多少收益」。"
            if quantized
            else "This result isolates one captured CUDA Graph replay; it does not measure quantization."
        ),
    }
    payload["attainment"] = res["attainment"]
    payload["vram"] = {
        "weights_gib": vram.get("weights_gib"),
        "kv_gib": vram.get("kv_gib"),
        "activations_gib": vram.get("activations_gib"),
        "workspace_gib": vram.get("workspace_gib"),
        "peak_total_gib": vram.get("peak_total_gib"),
        "method": vram.get("method"),
    }
    gate_pass = payload["correctness"]["gate"] == "PASS"
    payload["latency"], _cg_failed_evidence = latency_section_for_gate(
        gate_pass, payload["latency"]
    )
    payload["_measurement"] = {}
    if _cg_failed_evidence is not None:
        payload["_measurement"]["gate_failed_evidence"] = _cg_failed_evidence
    if quantized:
        _arm = res.get("dequant_arm") or cfg.get("dequant")
        _notes = (
            "量化格（%s group=%s dequant=%s，实际走的臂 = %s）× **cudagraph**。"
            "KL/top-1 引用 EXP-p1-04 的同配置实测；本格把守的两件事是"
            "（a）引用门 PASS、（b）图回放忠实于**同路径 eager**。"
            % (bit_width, cfg.get("group_size"), cfg.get("dequant"), _arm)
        )
        if not gate_pass:
            _notes += (
                " ★ 正确性门 FAIL：本格延迟**只允许**作为'图回放/融合不改变正确性判定'"
                "的证据，不得作为性能结论（硬规则 2）；实测值在 "
                "_measurement.gate_failed_evidence。"
            )
        _applies = "%s group=%s dequant=%s × cudagraph、单卡、batch=1、ctx=%s、%s" % (
            bit_width,
            cfg.get("group_size"),
            cfg.get("dequant"),
            cfg.get("ctx"),
            payload["hardware"].get("compute_capability"),
        )
        _not_applies = (
            "其它 group_size / ctx / targets（门没有实测）、batch>1、其它架构、"
            "PagedAttention；dequant 占比未测（graph 臂需要两张图交错）；%s" % SCOPE_NOT_APPLICABLE
        )
    else:
        _notes = "fp16 CUDA Graph feasibility and correctness gate; no quantization conclusion."
        _applies = "fp16、单卡、batch=1、ctx=%s、%s" % (
            cfg.get("ctx"),
            payload["hardware"].get("compute_capability"),
        )
        _not_applies = "%s；%s" % (
            "quantized execution、PagedAttention、ctx>4096、other GPUs",
            SCOPE_NOT_APPLICABLE,
        )
    payload["verdict"] = {
        "verdict": "PASS" if gate_pass else "FAIL",
        "outcome": "supported" if gate_pass else "refuted",
        "notes": _notes,
    }
    payload["applicability"] = {
        "applies_to": _applies,
        "does_not_apply_to": _not_applies,
        "why": "StaticCache and graph addresses/shapes are fixed to this feasibility gate.",
    }
    payload["_measurement"].update(
        {
            "timing_method": (
                "perf_counter wall over CUDA Graph replay; CUDA event and profiler "
                "self-device time"
            ),
            "ctx_window": (
                "eager StaticCache prefill at ctx=%s, then fixed q_len=1 graph replays"
                % cfg.get("ctx")
            ),
            "host_gap_caveats": gap.get("caveats"),
            "vram_split_note": vram.get("note"),
        }
    )
    if quantized and (res.get("dequant_activity") or res.get("quantize_report")):
        # ★ 这一格实际走了哪条 dequant 臂：计数来自 capture **之前**的相位
        #   （prefill / warmup / eager 参照），replay 不再经过 Python —— 它证明的是
        #   "被捕获的那段前向用的是哪条臂"，不是给 replay 计数。
        payload["_measurement"]["dequant"] = {
            "arm_recorded": res.get("dequant_arm"),
            "activity": res.get("dequant_activity"),
            "quantize_report": res.get("quantize_report"),
            "note": "计数只覆盖 capture 之前 + eager 参照相位；graph replay 不经过 Python。",
        }
    payload["_cudagraph"] = {
        "capture": res.get("capture"),
        "correctness": correctness,
        "static_cache": {
            "max_cache_len": cfg.get("ctx") + cfg.get("max_new_tokens"),
            "layout": "continuous",
        },
    }
    if res.get("attn_kernel"):
        # ★ 这条替换是**本仓库自己的** attention core，其中"长度来自设备内存"这件事
        #   是全部正确性的来源。warm 的配置与替换的层数必须进结果文件，否则事后
        #   无法判断这个数字是在哪个实现、哪个容量下测的。
        payload["_cudagraph"]["attn_kernel"] = res["attn_kernel"]
        payload["_cudagraph"]["length_source"] = {
            "buffer": "GraphBuffers.filled_len (int32, shape (1,), device)",
            "updated_in": "_set_step_inputs()，在图外、每步 fill_",
            "reached_kernel_via": "StaticCacheDecodeSession → cache._lemoq_filled_len → kernel 的循环上界",
        }
        payload["_measurement"]["attention_mask_note"] = (
            "4D mask 仍然每步更新（eager 臂需要它），但 kernel 不读它："
            "kernel 只读 [0, filled_len)，未填充槽位根本不参与计算。"
        )
    # 硬规则 2 的 builder 出口（与 build_payload 同款收口）。
    sanitize_gate_failed_perf(payload)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="EXP-p0-02-baseline-fp16")
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument(
        "--execution",
        default=None,
        help="execution arm: eager or cudagraph (default inferred from --exp)",
    )
    ap.add_argument(
        "--moe",
        default=None,
        choices=["hf", "static", "gather", "grouped"],
        help=(
            "decode 的 MoE 实现（默认从 --exp 的 config.moe 推；grouped 需要 triton）。"
            "★ 'static' 是旧名，归一到实现名 'gather'（VALID_MOE_IMPLS = gather|grouped）"
        ),
    )
    ap.add_argument(
        "--static-moe",
        action="store_true",
        help="等价于 --moe static（保留这个别名，scripts/run_cudagraph_gate.sh 还在用）",
    )
    ap.add_argument(
        "--attn",
        default=None,
        choices=["sdpa", "eager", "flash_attention_2", "kernel"],
        help=(
            "attention 实现覆盖（默认从实验矩阵的 config.attn 读，再退回模型默认）。"
            "kernel = 本仓库的 split-KV decode attention（实例级替换 attention core，"
            "只作用于 q_len=1；prefill 仍走 HF 原路）"
        ),
    )
    ap.add_argument(
        "--bit-width",
        default="fp16",
        choices=["fp16", "w8a16", "w8a8", "w4a16"],
        help="量化位宽（EXP-p2-01 / p4b-01 的 bit_width 轴）。fp16 = 不量化。",
    )
    ap.add_argument(
        "--dequant",
        default="none",
        choices=["none", "separate", "fused"],
        help=(
            "反量化方式（EXP-p2-01 的 dequant 轴）。fused = decode 单 token 走融合 kernel"
            "（需要 triton + CUDA，否则加载前拒绝）；prefill / 多 token 退回 separate 并计数。"
        ),
    )
    ap.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="量化 group_size。门只实测过 128（EXP-p1-04），其它值会被拒绝。",
    )
    ap.add_argument(
        "--rmsnorm",
        default="fused",
        choices=["fused", "hf"],
        help=(
            "decode 每层的 RMSNorm 实现（默认 fused = 本仓库融合 kernel；需要 triton+CUDA，"
            "否则加载前拒绝）。hf = transformers 的 OlmoeRMSNorm 原链（7 kernel/norm），"
            "保留它是为了复现 P2i 之前的格子与做对照。★ cell id 里没有这个轴，"
            "实现名写进 payload 的 config.rmsnorm。"
        ),
    )
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    execution = _execution_for_exp(args.exp, args.execution)
    # ★ attention 实现必须在**拼 cell id 之前**解析：`attn` 从 EXP-p4c-01 起是**声明轴**
    #   （{sdpa, kernel}），cell id 里必须带上它，否则两个臂会写成同一个文件。
    #   缺轴取值时 `build_exp_id` 会直接报错，**不静默退回默认臂** —— 那正是这里要的：
    #   一个静默跑错臂的格子会写出一份"看起来是 kernel 证据"的 sdpa 结果。
    attn_mode = args.attn if getattr(args, "attn", None) else _config_for_exp(args.exp, "attn")
    # ★ cell id 必须由**矩阵声明的轴**拼出（顺序 = yaml 里 factors 键的顺序，然后 ctx）。
    #   老代码手拼 `__ctx=%d`：EXP-p2-01 的格子是 `__bit_width=..._dequant=...`，
    #   于是会写出一个与骨架对不上的文件名 —— `is_done()` 永远判不了它。
    exp_id = build_exp_id(
        args.exp,
        {
            "ctx": args.ctx,
            "bit_width": args.bit_width,
            "dequant": args.dequant,
            "execution": execution,
            "attn": attn_mode,
        },
    )
    quantized = args.bit_width != "fp16"
    citation = None if not quantized else gate_for_cell(args.bit_width, args.group_size, args.ctx)

    # decode 的 MoE 实现。优先级：显式 --moe > 兼容别名 --static-moe > 实验矩阵里
    # 声明的 config.moe > "hf"。从矩阵读是关键：结果文件里的 moe 字段必须与矩阵
    # 声明一致，否则 p0-03 / p0-04 / p2-02 三条**不同假设**会混进同一个口径。
    # ★ 必须在拒绝逻辑之前解析：第四条拒绝（量化 × 非 hf MoE）要用它。
    if args.moe:
        moe_mode = args.moe
    elif getattr(args, "static_moe", False):
        moe_mode = "static"
    else:
        moe_mode = _moe_for_exp(args.exp)
    # ★ "static" 是 p0-04 时代的旧名，实现现在叫 "gather"
    #   （`VALID_MOE_IMPLS = ("gather", "grouped")`，随 grouped kernel 一起改的）。
    #   **不归一就全线崩溃**：run_cudagraph 的 valid 检查会 raise
    #   `unknown moe 'static'` —— 矩阵里 p0-04 的声明、`--static-moe` 别名、
    #   以及 scripts/run_cudagraph_gate.sh 今天都会撞上它（p0-04 的格子是 done，
    #   所以一直没人触发）。归一到实现名，记录里写真实跑的 impl。
    moe_mode = {"static": "gather"}.get(moe_mode, moe_mode)

    if not args.dry_run:
        # 每条拒绝都发生在**任何模型加载之前** —— 干净失败，且离线（无 GPU）可测。
        # ★ `--dequant fused` 的"kernel 尚未实现"拒绝已于 2026-10-07 删除：
        #   `lemoq.kernels._dequant_triton` 已落地（P2-01 的 fused 臂）。现在拒绝它的
        #   唯一理由是**这台机器跑不了**（没有 triton / 没有 CUDA）—— Mac 上仍然拒绝，
        #   而不是加载 13.4 GB 模型之后才炸。
        if args.dequant == "fused":
            if not quantized:
                print(
                    "runner 拒绝: --dequant fused 只在量化格（bit_width != fp16）有意义，"
                    "fp16 没有反量化语义。",
                    file=sys.stderr,
                )
                return 2
            _fused_reason = fused_unavailable_reason()
            if _fused_reason is not None:
                print(
                    "runner 拒绝: --dequant fused 需要 triton + CUDA（%s）。"
                    "本机只能跑 separate 臂 —— 见 src/lemoq/kernels/_dequant_triton.py。"
                    % _fused_reason,
                    file=sys.stderr,
                )
                return 2
        if quantized and citation is None:
            print(
                "runner 拒绝: %s / group=%d / ctx=%d 没有 EXP-p1-04 的门实测，"
                "不许在没有正确性门的配置上跑性能（硬规则 2）。先补测精度。"
                % (args.bit_width, args.group_size, args.ctx),
                file=sys.stderr,
            )
            return 2
        # ★ 2026-10-07：量化 × cudagraph 的"未接线"拒绝已删除（P3 落地）。
        #   量化在 capture **之前**就地完成（`run_cudagraph(quant_cfg=…)`），所以图里
        #   只有权重表示与 kernel 选择的差别 —— 这正是 E7 要量的事情。
        #   仍然不存在的组合由下面的四条（gather / 4-bit grouped / 无门实测 /
        #   fused 本机不可用）在**加载模型之前**拒绝。
        # ★ P1 已落地：量化 × grouped **存在**（moe_grouped_quant：在 fused 张量上量化，
        #   kernel 内解包）。仍然不存在的只有两条，且都必须在加载模型之前拒绝：
        #     (a) 量化 × gather —— gather 路径吃 fp16 融合张量，没有量化版；
        #     (b) 4-bit —— fused **expert** 的 nibble 解包尚未实现（注意：dense 侧的
        #         `_dequant_triton` 已经支持 4-bit，所以 `--dequant fused` + w4a16 在
        #         **hf** 路径上是能跑的；被拒的是量化 expert 的 grouped 路径）。
        if quantized and moe_mode == "gather":
            print(
                "runner 拒绝: 量化 × gather 的组合不存在 —— gather 路径直接索引 fp16 融合"
                "张量（gate_up[ids]），没有量化版。量化 expert 路径只接在 grouped 上，"
                "见 EXP-p4b-01-truncation 记录 §3。",
                file=sys.stderr,
            )
            return 2
        # ★ 2026-10-07：量化 grouped 的 4-bit 拒绝已删除 —— nibble 解包在
        #   `moe_grouped_quant` + `_grouped_triton._expand_int4` 里落地了（K1 门在 5090 上 PASS）。
        #   ★ 但 w4a16 / g128 的**引用门 FAIL**（KL 0.0933 > 0.05）：它照跑，只是按硬规则 2
        #   标成 `verdict=FAIL`、延迟只进 `_measurement.gate_failed_evidence`。
        #   4-bit 真正的约束在下面这条：group_size 必须是偶数（nibble 成对打包）。
        if quantized and moe_mode == "grouped" and args.group_size % 2 != 0:
            print(
                "runner 拒绝: 4/8-bit 量化 grouped 的 group_size=%d 不是偶数 —— "
                "nibble 成对打包要求偶数（4-bit），且 kernel 的 k 块必须整除 group_size。"
                % args.group_size,
                file=sys.stderr,
            )
            return 2

        # ★ P2i 的第六条拒绝：融合 norm 需要 triton + CUDA。
        #   顺序刻意排在**配置合法性拒绝之后**：那几条是**与机器无关**的
        #   （无门实测 / 量化×gather / 奇数 group_size），必须保持最高优先级 ——
        #   否则在 Mac 上，那四条测试会先撞到"本机没有 triton"，测不到它们本来的对象。
        #   仍然在**加载 13.4 GB 模型之前**。
        if args.rmsnorm == "fused":
            from lemoq.kernels.rmsnorm import unavailable_reason as _norm_reason  # noqa: PLC0415

            _r = _norm_reason()
            if _r is not None:
                print(
                    "runner 拒绝: --rmsnorm fused 需要 triton + CUDA（%s）。"
                    "本机请用 --rmsnorm hf（见 src/lemoq/kernels/rmsnorm.py）。" % _r,
                    file=sys.stderr,
                )
                return 2

    # ★ `--dry-run` **不受 `is_done` 短路**：它是只读查询（不加载模型、不写文件），
    #   必须永远答得出"这一格叫什么、门是什么状态"。`results/` 是**机器本地**的
    #   （gitignored）：同一格在 GPU 机器上跑过之后，dry-run 若被短路成
    #   `skip (done)`，测试和人就都问不到它了 —— `test_dry_run_p4b_cell_id_matches_skeleton`
    #   在 workspace 干净的 Mac 上过、在刚跑过该格的服务器上红，就是这个原因（实测红过）。
    if not args.dry_run and is_done(exp_id, args.results_dir) and not args.force:
        print("skip (done): %s" % exp_id)
        return 0

    if execution == "cudagraph" and args.dtype != "float16":
        print("runner 失败: EXP-p0-03 只支持 --dtype float16", file=sys.stderr)
        return 2

    # attn_mode 已在上面解析（cell id 需要它；见那里的注释）。

    # ★ 假设优先取**矩阵声明**（harness 写骨架时就是这么取的）。只有矩阵没写、
    #   且本实验是 fp16 基线时，才用下面这段内置文本 —— 否则 p2-01 的记录里
    #   会被塞进一句与它无关的 fp16 假设。
    _mx = _matrix_entry(args.exp)
    hypothesis = {
        "statement": _mx.get("hypothesis")
        or (
            "fp16 StaticCache 能让固定 batch=1/q_len=1 的 OLMoE decode capture/replay；"
            "graph 与 eager 的 logits/top-1 通过正确性门"
            if execution == "cudagraph"
            else "fp16 能在目标卡上加载并前向；测量管线产出的延迟/显存/归因自洽"
        ),
        "refutation_signal": _mx.get("refutation_signal")
        or (
            "prefill、capture、replay 或 graph/eager 正确性门失败；必须记录阻塞阶段，"
            "不填性能数字"
            if execution == "cudagraph"
            else "加载失败、OOM、或达成率 >100%（说明字节账或带宽错了）"
        ),
    }
    cfg = {
        "bit_width": args.bit_width,
        "dtype": args.dtype,
        "ctx": args.ctx,
        "batch": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "warmup": args.warmup,
        "timed": args.timed,
        "execution": execution,
        "moe": moe_mode,
        "attn": attn_mode or "model-default",
        # fp16 格的 dequant/group_size 必须保持 "none"/None —— 已落盘的 p0/p2 结果
        # 就是这两个值，改它们会让重跑的 payload 与历史不一致。
        "dequant": args.dequant if quantized else "none",
        "rmsnorm": args.rmsnorm,
        "fusion": "none",
        "group_size": args.group_size if quantized else None,
    }

    if execution == "cudagraph":
        # ★ 这里原来把 ctx 钉死成 1024（p0-03 的范围限制）。放开是刻意的：
        #   ctx 是**声明的实验轴**（见 configs/experiments.yaml 的 p0-04），
        #   而且 ctx=4096 不只是多一个数据点 —— 它是 H-A/H-B 的判别实验：
        #   attention 扫的槽位随 ctx 线性增长，而 MoE 专家路径与 ctx 无关。
        #   残差若随 ctx 长大，它是 attention；若持平，它是 kernel 数地板。
        if args.batch != 1 or args.ctx < 1:
            print("runner 失败: cudagraph 臂只支持 batch=1 且 ctx >= 1", file=sys.stderr)
            return 2
        try:
            CudaGraphConfig(
                ctx=args.ctx, max_new_tokens=args.max_new_tokens, batch=args.batch
            ).validate_timing(args.timed)
        except ValueError as exc:
            print("runner 失败: %s" % exc, file=sys.stderr)
            return 2
        if not args.dry_run:
            try:
                import torch

                if not torch.cuda.is_available():
                    print(
                        "runner 失败: EXP-p0-03 requires CUDA; Mac runs must not create result files",
                        file=sys.stderr,
                    )
                    return 2
            except Exception as exc:  # noqa: BLE001
                print("runner 失败: cannot inspect CUDA availability: %s" % exc, file=sys.stderr)
                return 2

    if args.dry_run:
        print("would run: %s" % exp_id)
        print(
            "  model=%s ctx=%d batch=%d dtype=%s" % (args.model, args.ctx, args.batch, args.dtype)
        )
        print(
            "  bit_width=%s group_size=%s dequant=%s execution=%s moe=%s rmsnorm=%s"
            % (
                cfg["bit_width"],
                cfg["group_size"],
                cfg["dequant"],
                cfg["execution"],
                cfg["moe"],
                cfg.get("rmsnorm"),
            )
        )
        if quantized:
            print(
                "  gate: %s"
                % (
                    "%s (引用 EXP-p1-04, kl=%.6f, top-1=%.6f)"
                    % (citation["gate"], citation["kl"], citation["top1"])
                    if citation
                    else "无门实测 → 真跑会被拒绝"
                )
            )
        return 0

    try:
        if execution == "cudagraph":
            res = run_cudagraph(
                model_id=args.model,
                ctx=args.ctx,
                batch=args.batch,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                timed=args.timed,
                dtype_name=args.dtype,
                moe=moe_mode,
                attn=attn_mode,
                quant_cfg=(
                    {
                        "bit_width": args.bit_width,
                        "group_size": args.group_size,
                        "ctx": args.ctx,
                        "dequant": args.dequant,
                    }
                    if quantized
                    else None
                ),
                rmsnorm=args.rmsnorm,
            )
        else:
            res = run_baseline(
                model_id=args.model,
                ctx=args.ctx,
                batch=args.batch,
                max_new_tokens=args.max_new_tokens,
                warmup=args.warmup,
                timed=args.timed,
                dtype_name=args.dtype,
                quant_cfg=(
                    {
                        "bit_width": args.bit_width,
                        "group_size": args.group_size,
                        "ctx": args.ctx,
                        # ★ dequant 必须传进去：它决定 QuantLinear 的 fused 分支，
                        #   也决定 payload 里写的"这一格实际走哪条臂"。
                        "dequant": args.dequant,
                    }
                    if quantized
                    else None
                ),
                moe=moe_mode,
                rmsnorm=args.rmsnorm,
            )
    except CudaGraphCorrectnessError as exc:
        payload = failed_result(
            exp_id=exp_id,
            config=cfg,
            correctness=exc.report,
            note=str(exc),
            metadata=_cudagraph_metadata(args.model, cfg),
        )
        path, problems = write_result(payload, root=args.results_dir)
        print("CUDA Graph correctness gate failed; wrote %s" % path, file=sys.stderr)
        for problem in problems:
            print("  - %s" % problem, file=sys.stderr)
        return 1 if problems else 0
    except CudaGraphCaptureError as exc:
        payload = blocked_result(
            exp_id=exp_id,
            config=cfg,
            failure=failure_record(exc),
            metadata=_cudagraph_metadata(args.model, cfg),
        )
        path, problems = write_result(payload, root=args.results_dir)
        print("CUDA Graph feasibility blocked; wrote %s" % path, file=sys.stderr)
        for problem in problems:
            print("  - %s" % problem, file=sys.stderr)
        return 1 if problems else 0
    except Exception as exc:  # noqa: BLE001
        if execution == "cudagraph":
            payload = blocked_result(
                exp_id=exp_id,
                config=cfg,
                failure={
                    "stage": "unknown",
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                },
                metadata=_cudagraph_metadata(args.model, cfg),
            )
            path, problems = write_result(payload, root=args.results_dir)
            print("CUDA Graph feasibility blocked; wrote %s" % path, file=sys.stderr)
            for problem in problems:
                print("  - %s" % problem, file=sys.stderr)
            return 1 if problems else 0
        import traceback

        print("runner 失败: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        traceback.print_exc()
        return 1

    payload = (
        build_cudagraph_payload(exp_id, res, cfg, hypothesis)
        if execution == "cudagraph"
        else build_payload(exp_id, res, cfg, hypothesis)
    )
    path, problems = write_result(payload, root=args.results_dir)

    s = res["steps"]
    print("=== %s ===" % exp_id)
    print("  measured bw      : %s GB/s (read-only)" % res["measured_bw_gbps"])
    print("  TTFT p50/p99     : %.3f / %.3f ms" % (res["ttft"]["p50"], res["ttft"]["p99"]))
    print("  per-token p50/p99: %.4f / %.4f ms" % (s["p50"], s["p99"]))
    print("  host share       : %s%%" % res["gap"].get("host_share_pct"))
    print("  VRAM peak        : %s GiB" % res["vram"].get("peak_total_gib"))
    print(
        "  [W] / [T]        : %d / %d bytes"
        % (res["budget"]["W"]["total_bytes"], res["budget"]["T"]["total_bytes"])
    )
    a = res["attainment"]
    print("  floor_W / floor_T: %s / %s ms" % (a["floor_w_ms"], a["floor_t_ms"]))
    print(
        "  attainment       : quant %s%% | bandwidth %s%%"
        % (a["quant_attainment_pct"], a["bw_attainment_pct"])
    )
    if a["quant_attainment_pct"] and a["quant_attainment_pct"] > 100:
        print(
            "  ⚠️ 达成率 >100% —— 字节账或带宽有问题，必须解释（docs/reference/METRICS.md §3）",
            file=sys.stderr,
        )
    print("\n  wrote %s" % path)
    print("  schema problems: %d" % len(problems))
    for p in problems:
        print("    - %s" % p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
