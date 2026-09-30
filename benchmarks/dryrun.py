"""dryrun — **离线**把整条流水线跑通一遍（无 GPU，无网络）。

它验证什么（这些正是"上机前该准备好的东西"）：

    1. `transformers` 能按 OLMoE 架构构建模型并前向（版本 pin 是否成立）
    2. `benchmarks.bytes` 能从 config 算出 `[W]`/`[T]`，且与已核实的锚点一致
    3. `lemoq.quant` 的量化/反量化往返正确，且 oracle 与 torch 路径一致
    4. `benchmarks.timing` / `vram` / `hostgap` 的接口可用（CPU 上走降级路径）
    5. `benchmarks.schema` 的校验器接受产出的 payload
    6. 结果能落盘并被 `report.py` 读到

★ **它产出的每一个数字都不可报告**。Mac 没有 NVIDIA GPU
（`AGENTS.md`：任何 Mac 产生的数字都不得进入 `results/<EXP-ID>/` 或任何结论）。
因此本模块写到 `results/_dryrun/`，并在 payload 里显式标 `_reportable: false`。

用法：
    python -m benchmarks.dryrun              # 需要 torch + transformers（.venv-mac 即可）
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.bytes import ModelSpec, QuantSpec, byte_budget, weight_budget
from benchmarks.schema import validate_result

DRYRUN_DIR = "results/_dryrun"
DRYRUN_EXP_ID = "DRYRUN-mac-cpu-pipeline"


def _tiny_olmoe_config():
    """A tiny randomly-initialised OLMoE-shaped model. No download, no network."""
    from transformers import OlmoeConfig

    return OlmoeConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )


def run(*, ctx: int = 64, bits: int = 4, group_size: int = 128) -> dict[str, Any]:
    import torch
    from transformers import OlmoeForCausalLM

    steps: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    # ---- 1. 构建模型 + 前向 ----
    cfg = _tiny_olmoe_config()
    model = OlmoeForCausalLM(cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        logits = model(ids).logits
    steps.append(
        {
            "step": "forward_cpu",
            "ok": True,
            "params": n_params,
            "logits_shape": list(logits.shape),
            "device": "cpu",
        }
    )

    # ---- 2. 字节预算（用真实 OLMoE config 值，不是 tiny 模型） ----
    real_olmoe = {
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
    }
    spec = ModelSpec.from_hf_config(real_olmoe, name="OLMoE-1B-7B-0924")
    w_fp16 = weight_budget(spec)
    w_w4 = weight_budget(spec, quant=QuantSpec(bits=4, group_size=128))
    budget = byte_budget(spec, ctx=4096)
    anchor_ok = w_fp16["total_bytes"] == 2357723136
    steps.append(
        {
            "step": "byte_budget",
            "ok": anchor_ok,
            "W_fp16_bytes": w_fp16["total_bytes"],
            "W_w4a16_g128_bytes": w_w4["total_bytes"],
            "T_at_4k_bytes": budget["T"]["total_bytes"],
            "anchor_expected": 2357723136,
            "anchor_ok": anchor_ok,
        }
    )

    # ---- 3. 量化往返 + oracle 一致性 ----
    from lemoq.quant import numpy_oracle as no
    from lemoq.quant import pack
    from lemoq.quant import rtn as tr

    # 选最大的 2-D 权重；tiny 模型的 in_features 可能小于 128，所以 group_size 要收缩
    weight = None
    weight_name = ""
    for name, p in model.named_parameters():
        if p.dim() == 2 and (weight is None or p.numel() > weight.numel()):
            weight, weight_name = p.detach(), name
    if weight is None:  # pragma: no cover - tiny model always has one
        raise RuntimeError("找不到可量化的 2-D 权重")

    effective_gs = min(group_size, int(weight.shape[1]))
    if effective_gs < 2:
        raise RuntimeError("权重内维太小（%d），无法分组量化" % weight.shape[1])

    qn = no.quantize_rtn(weight.numpy(), bits=bits, group_size=effective_gs)
    qt = tr.quantize_rtn(weight, bits=bits, group_size=effective_gs)
    import numpy as np

    oracle_match = np.array_equal(qn["codes"], qt["codes"].numpy())
    err = no.quantization_error(weight.numpy(), qn)
    packed = pack.pack_4bit(qn["codes"])
    unpacked = pack.unpack_4bit(packed, signed=True)
    pack_ok = np.array_equal(qn["codes"], unpacked.astype(np.int32))
    steps.append(
        {
            "step": "quant_roundtrip",
            "ok": bool(oracle_match and pack_ok),
            "weight": weight_name,
            "shape": list(weight.shape),
            "group_size_used": effective_gs,
            "oracle_matches_torch": bool(oracle_match),
            "pack_roundtrip_exact": bool(pack_ok),
            "snr_db": err["snr_db"],
        }
    )

    # ---- 4. 计时降级路径（CPU） ----
    from benchmarks.timing import measure_wall

    def one_step() -> None:
        with torch.no_grad():
            model(ids)

    wall = measure_wall(one_step, warmup=5, timed=20, repeats=3)
    steps.append(
        {
            "step": "timing_degraded_cpu",
            "ok": True,
            "timing_method": wall["timing_method"],
            "p50_ms": wall["stats"]["p50"],
            "note": "CPU wall time —— 不可报告，仅证明接口可用",
        }
    )

    # ---- 5. hostgap 在无 CUDA 时应如实返回 not_measured ----
    from benchmarks.hostgap import measure_host_gap

    gap = measure_host_gap(one_step, warmup=5, timed=10, repeats=1)
    steps.append(
        {
            "step": "hostgap_on_cpu",
            "ok": gap["status"] == "not_measured",
            "status": gap["status"],
            "reason": gap.get("reason"),
            "note": "无 CUDA 时必须返回 not_measured，不得编造数字",
        }
    )

    # ---- 6. vram 在无 CUDA 时应如实返回 not_measured ----
    from benchmarks.vram import VramTracker

    vr = VramTracker().finalize()
    steps.append(
        {
            "step": "vram_on_cpu",
            "ok": vr["status"] == "not_measured",
            "status": vr["status"],
        }
    )

    elapsed = time.perf_counter() - t_start
    all_ok = all(s["ok"] for s in steps)

    return {
        "steps": steps,
        "all_ok": all_ok,
        "elapsed_s": round(elapsed, 2),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
    }


def to_result_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Wrap the dry-run report as a schema-valid `result.json` skeleton.

    Every measured-looking field stays None: this run **measures nothing reportable**.
    """
    from benchmarks.schema import empty_result

    payload = empty_result(DRYRUN_EXP_ID, status="done")
    payload["hypothesis"] = {
        "statement": "离线流水线（config -> 字节预算 -> 量化 -> 计时接口 -> schema -> 落盘）可用",
        "refutation_signal": "任一步骤失败，或 schema 校验不通过",
    }
    payload["verdict"] = {
        "verdict": "PASS" if report["all_ok"] else "FAIL",
        "outcome": "supported" if report["all_ok"] else "refuted",
        "notes": "离线自检；不包含任何可报告的测量值",
    }
    payload["applicability"] = {
        "applies_to": "Mac CPU 上的流程验证（无 GPU）",
        "does_not_apply_to": "任何性能或精度结论",
        "why": "Mac 无 NVIDIA GPU：AGENTS.md 规定 Mac 产生的数字不得进入结论",
    }
    payload["_reportable"] = False
    payload["_reason_not_reportable"] = (
        "Mac / CPU 运行：没有 NVIDIA GPU，也没有实测可达带宽，"
        "因此任何延迟或达成率都无意义。仅用于验证流水线连通。"
    )
    payload["_dryrun"] = report
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out-dir", default=DRYRUN_DIR)
    args = ap.parse_args(argv)

    try:
        report = run(ctx=args.ctx, bits=args.bits, group_size=args.group_size)
    except Exception as exc:  # noqa: BLE001
        print("dryrun 失败: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    print("=== 离线流水线自检 ===")
    for s in report["steps"]:
        flag = "ok  " if s["ok"] else "FAIL"
        extra = ""
        if s["step"] == "byte_budget":
            extra = " [W]=%d B (anchor %s)" % (
                s["W_fp16_bytes"],
                "match" if s["anchor_ok"] else "MISMATCH",
            )
        elif s["step"] == "quant_roundtrip":
            extra = " oracle==torch: %s, pack exact: %s, SNR %.1f dB" % (
                s["oracle_matches_torch"],
                s["pack_roundtrip_exact"],
                s["snr_db"],
            )
        elif s["step"] == "hostgap_on_cpu":
            extra = " status=%s" % s["status"]
        elif s["step"] == "vram_on_cpu":
            extra = " status=%s" % s["status"]
        elif s["step"] == "forward_cpu":
            extra = " params=%d logits=%s" % (s["params"], s["logits_shape"])
        print("  %s %-24s%s" % (flag, s["step"], extra))
    print("  elapsed: %.2fs" % report["elapsed_s"])
    print(
        "  versions: torch %s | transformers %s"
        % (report["host"]["torch"], report["host"]["transformers"])
    )

    payload = to_result_payload(report)
    problems = validate_result(payload)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "result.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print("\n  schema problems: %d" % len(problems))
    for p in problems:
        print("    - %s" % p, file=sys.stderr)
    print("  wrote %s  (reportable=false)" % path)

    return 0 if (report["all_ok"] and not problems) else 1


if __name__ == "__main__":
    raise SystemExit(main())
