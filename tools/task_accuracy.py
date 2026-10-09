"""task_accuracy — 任务级精度评估（**GPU 待执行**，见 docs/METRICS.md §6 G1）。

## 为什么要它

PPL 是 token 级聚合指标，`docs/METRICS.md` §1 自己就写了它对低比特区分度弱；
KL 与 top-1 量的是"分布偏了多少"。三者都回答不了"这个模型还能不能用"。
任务级精度（MMLU / GSM8K）才是那句话的直接证据。本仓库此前**零数据**。

## 口径（★ 严格沿用 `benchmarks/quant_gate.py:72` 的 teacher forcing）

两条臂必须比**同一串** token，否则量到的是"两条分叉轨迹的差"，不是量化本身的差。
所以本脚本跑两种口径，都要报：

    A. **端到端（现实意义）**：每条臂各自 greedy 生成答案 → 任务 accuracy。
       这是"能不能用"的直接答案，但两臂轨迹可能分叉。
    B. **teacher forcing（隔离变量）**：把 **fp16 那条轨迹**的答案 token 强制喂给
       量化臂，量化这些 token 上的一致率与"抽出的答案是否相同"。
       这才是"分布偏了多少"落在任务上的对应物（与 KL 同一口径）。

★ 两者都要报，缺一不可：只报 A 无法与 KL/top-1 对上账；只报 B 不能回答能不能用。

## 规模（成本可控，且必须写明是子集）

    MMLU 5-shot，200 题（分层抽样的子集，不是全量 —— 数字必须带"子集"标注）
    GSM8K 100 题（子集）

只对**已过正确性门**的配置跑（默认 w8a16 / g128）；门 FAIL 的配置跑它是浪费机器，
而且按硬规则 2 不得发性能/精度结论。

## 用法

    python -m tools.task_accuracy --dry-run          # 打印计划（离线可跑，不产出数字）
    python -m tools.task_accuracy --model <M> --data-dir <dir> --json results/task_accuracy.json

**离线只做 --dry-run。** 本脚本在无 CUDA 的机器上不会写任何结果文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

PLANNED_SUBSETS = {
    "mmlu": {
        "n": 200,
        "shots": 5,
        "note": "分层抽样子集（不是全量 MMLU）；5-shot 示例取自 dev 分片",
    },
    "gsm8k": {
        "n": 100,
        "shots": 8,
        "note": (
            "子集（不是全量 GSM8K）。★ 计划原本写 0-shot，上机时改为 8-shot（示例取自 "
            "train 前 8 条）：本仓库的 OLMoE 是**基座**模型，0-shot 直答 GSM8K 会贴地板，"
            "delta 失去意义。这一改动发生在上机时，在此显式记录。"
        ),
    },
}

# 只跑已过门的配置（GATE_CITATIONS 里 gate == PASS 的）
DEFAULT_ARMS = ["w8a16"]

REQUIRED_DATA_FILES = {
    "mmlu": "mmlu_subset.jsonl   (每行: {question, choices:[4], answer:int, subject:str})",
    "gsm8k": "gsm8k_subset.jsonl  (每行: {question, answer:str, answer_int:int})",
}


def _plan(args: argparse.Namespace) -> dict[str, Any]:
    """上机计划。**不含任何测量结果**。"""
    return {
        "generated_by": "tools/task_accuracy.py --dry-run",
        "status": "planned",
        "measured": False,
        "note": "这是**计划**，不是结果。未测的值一律 planned（硬规则 1）。",
        "arms": list(args.arms),
        "reference": "fp16（同模型、同权重、未量化）",
        "subsets": PLANNED_SUBSETS,
        "protocol": {
            "A_end_to_end": "各臂各自 greedy 生成 → 任务 accuracy（现实意义，但轨迹可能分叉）",
            "B_teacher_forcing": (
                "把 fp16 轨迹的答案 token 强制喂给量化臂（沿用 "
                "benchmarks/quant_gate.py:72 的口径）→ token 一致率 + 抽出答案是否相同"
                "（隔离变量，与 KL/top-1 同一口径）",
            )[0],
            "why_both": "只报 A 无法与 KL/top-1 对账；只报 B 回答不了能不能用。",
        },
        "required_data_files": REQUIRED_DATA_FILES,
        "data_dir": args.data_dir,
        "output": args.json or "results/task_accuracy.json",
        "acceptance": {
            "metric": "accuracy + 与 fp16 的 delta（百分点）",
            "gate": "下降 < 2 个点（docs/METRICS.md §1；**自定建议值**）",
            "must_report": [
                "样本量 n（子集，不是全量）",
                "两口径 A 与 B 都报",
                "delta 的样本量推导误差棒（二项分布，不能只给点估计）",
                "gate 是否 PASS；FAIL 时不得发性能结论（硬规则 2）",
            ],
        },
        "estimated_cost": {
            "note": "**估算**，上机后必须用实测时长覆盖；不得当作实测值引用。",
            "per_arm": "2 个数据集 × (200 + 100) 题 × greedy 生成",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="本地 jsonl 目录（避免依赖 HF 下载；见 REQUIRED_DATA_FILES）",
    )
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--json", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="每数据集只跑 2 题，验证通路。smoke 的结果不是测量，不得引用。",
    )
    args = ap.parse_args(argv)

    args.arms = [a.strip() for a in str(args.arms).split(",") if a.strip()]

    if args.dry_run:
        print(json.dumps(_plan(args), ensure_ascii=False, indent=2))
        print(
            "\n[dry-run] 未做任何测量。上机前请确认：\n"
            "  1. --data-dir 里有 %s；\n"
            "  2. 只对**已过门**的配置跑（当前 %s）；\n"
            "  3. 结果必须两种口径都报，且标注是子集。" % (sorted(REQUIRED_DATA_FILES), args.arms),
            file=sys.stderr,
        )
        return 0

    # 真跑：必须有 CUDA + 模型 + 数据
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("需要 torch。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print(
            "task_accuracy 需要 CUDA：GPU 上的数字不得在无 GPU 的机器上产生。\n"
            "离线请用 --dry-run 打印计划。",
            file=sys.stderr,
        )
        return 2
    if not args.model:
        print("--model 必填（或用 --dry-run 打印计划）。", file=sys.stderr)
        return 2
    if not args.data_dir or not os.path.isdir(args.data_dir):
        print(
            "--data-dir 必须指向含 %s 的目录（准备方法见 --dry-run）。"
            % sorted(REQUIRED_DATA_FILES),
            file=sys.stderr,
        )
        return 2

    # ★ 评测循环（2026-10-09 于 5090 上机时补齐并验证：先 --smoke 每数据集 2 题，
    #   再跑完整子集。快照/恢复复用 tools/quant_ppl 的实现，保证与 PPL/KL 同一套
    #   扰动机制。）
    return _run(args)


# ---------------------------------------------------------------------------
# 评测循环
# ---------------------------------------------------------------------------

MMLU_PROMPT_HEADER = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)
LETTERS = ("A", "B", "C", "D")


def _fmt_mmlu_q(ex: dict[str, Any]) -> str:
    lines = ["Question: " + ex["question"].strip()]
    for letter, choice in zip(LETTERS, ex["choices"]):
        lines.append("%s. %s" % (letter, str(choice).strip()))
    lines.append("Answer:")
    return "\n".join(lines)


def _mmlu_prompt(ex: dict[str, Any], dev: list[dict[str, Any]]) -> str:
    """5-shot：取**同 subject** 的 dev 示例（dev 分片恰好每 subject 5 条）。

    ★ 上机实测抓到的 bug：最初直接把整个 dev 列表（285 条）传进来当 shots，
      prompt 膨胀到 ~29k token（模型上限 4096），生成被完全截断、parse_fail 全满。
    """
    shots = [s for s in dev if s["subject"] == ex["subject"]][:5]
    parts = [MMLU_PROMPT_HEADER.format(subject=ex["subject"])]
    for s in shots:
        parts.append(_fmt_mmlu_q(s))
        parts.append(" " + LETTERS[int(s["answer"])] + "\n\n")
    parts.append(_fmt_mmlu_q(ex))
    return "".join(parts)


def _gsm8k_prompt(ex: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    parts = []
    for s in shots:
        parts.append(
            "Question: " + s["question"].strip() + "\nSolution: " + s["solution"].strip() + "\n\n"
        )
    parts.append("Question: " + ex["question"].strip() + "\nSolution:")
    return "".join(parts)


def _parse_mmlu(text: str) -> str | None:
    t = text.strip()
    for ch in t:
        if ch in LETTERS:
            return ch
    return None


def _parse_gsm8k(text: str) -> str | None:
    """优先 `#### N`；否则取最后一个数（含小数/负数/逗号）。"""
    if "####" in text:
        tail = text.split("####")[-1]
        for tok in tail.replace(",", "").split():
            tok = tok.strip(".")
            if tok.lstrip("-").replace(".", "", 1).isdigit():
                return tok
    import re

    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    return nums[-1].replace(",", "").strip(".")


def _num_eq(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


def _gen(model: Any, tok: Any, prompt: str, max_new: int) -> list[int]:
    """greedy 生成，返回**新生成的** token id 列表。"""
    import torch

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new,
            pad_token_id=tok.eos_token_id or tok.pad_token_id or 0,
        )
    return out[0][ids.shape[1] :].tolist()


def _teacher_forcing_agreement(
    model: Any, tok: Any, prompt: str, gen_ids: list[int]
) -> dict[str, Any] | None:
    """口径 B：把 fp16 的生成 token 强制喂给本臂，逐位置比较 argmax。

    一个前向拿全部位置的 logits（teacher forcing = KV 对齐的并行打分）。
    返回 {n_positions, n_agree, first_token_agree}；生成序列为空时返回 None。
    """
    import torch

    if not gen_ids:
        return None
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    full = torch.cat([ids, torch.tensor([gen_ids], device=model.device)], dim=1)
    with torch.no_grad():
        logits = model(full).logits  # (1, L, V)
    # 位置 i 的 logits 预测第 i+1 个 token → 生成段从 len(ids)-1 起读
    start = ids.shape[1] - 1
    seg = logits[0, start : start + len(gen_ids)]
    pred = seg.argmax(dim=-1)
    gold = torch.tensor(gen_ids, device=model.device)
    n_agree = int((pred == gold).sum())
    first_agree = bool(pred[0].item() == gold[0].item())
    return {
        "n_positions": len(gen_ids),
        "n_agree": n_agree,
        "agreement_rate": n_agree / len(gen_ids),
        "first_token_agree": first_agree,
    }


def _run_arm(
    model: Any,
    tok: Any,
    arm: str,
    data: dict[str, list[dict[str, Any]]],
    max_new: dict[str, int],
    fp16_traces: dict[str, list[dict[str, Any]]] | None = None,
    traces_out: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """一条臂的完整评测。

    fp16 臂先跑，传 `traces_out` 顺手把每题的 (prompt, gen) 存下来 —— 轨迹**必须**
    在量化前收集，量化之后再生成的就不是 fp16 的轨迹了。量化臂传 `fp16_traces`
    做口径 B。
    """
    import time

    out: dict[str, Any] = {"arm": arm, "tasks": {}}
    for task in ("mmlu", "gsm8k"):
        rows = data[task]
        n = len(rows)
        correct = 0
        parse_fail = 0
        agreements: list[float] = []
        first_agree = 0
        first_total = 0
        t0 = time.time()
        for i, ex in enumerate(rows):
            if task == "mmlu":
                prompt = _mmlu_prompt(ex, data["mmlu_dev"])
            else:
                prompt = _gsm8k_prompt(ex, data["gsm8k_shots"])
            gen = _gen(model, tok, prompt, max_new[task])
            if traces_out is not None:
                traces_out[task].append({"prompt": prompt, "gen": gen})
            text = tok.decode(gen, skip_special_tokens=True)
            if task == "mmlu":
                pred = _parse_mmlu(text)
                ok = pred is not None and pred == LETTERS[int(ex["answer"])]
            else:
                pred = _parse_gsm8k(text)
                ok = _num_eq(pred, ex["answer"])
            if pred is None:
                parse_fail += 1
            correct += int(ok)
            # 口径 B：只对量化臂（fp16 轨迹已存）
            if fp16_traces is not None and arm != "fp16":
                trace = fp16_traces[task][i]
                rep = _teacher_forcing_agreement(model, tok, trace["prompt"], trace["gen"])
                if rep is not None:
                    agreements.append(rep["agreement_rate"])
                    first_total += 1
                    first_agree += int(rep["first_token_agree"])
            if (i + 1) % 25 == 0 or i + 1 == n:
                print(
                    "    [%s/%s] %d/%d  acc=%.3f (%.0fs)"
                    % (arm, task, i + 1, n, correct / (i + 1), time.time() - t0),
                    flush=True,
                )
        entry: dict[str, Any] = {
            "n": n,
            "n_correct": correct,
            "accuracy": correct / n,
            "binomial_se": (correct / n * (1 - correct / n) / n) ** 0.5,
            "parse_fail": parse_fail,
            "subset_note": PLANNED_SUBSETS[task]["note"],
        }
        if fp16_traces is not None and arm != "fp16" and agreements:
            entry["protocol_B"] = {
                "口径": "fp16 轨迹 teacher forcing（quant_gate.py:72 的任务级对应物）",
                "n_questions_with_trace": len(agreements),
                "token_agreement_mean": sum(agreements) / len(agreements),
                "first_token_agreement": first_agree / first_total if first_total else None,
            }
        out["tasks"][task] = entry
    return out


def _run(args: argparse.Namespace) -> int:

    from benchmarks.runner import _load_model
    from tools.quant_ppl import _candidate_names, _restore, _snapshot

    # ---- 数据 ----
    def _jsonl(name: str) -> list[dict[str, Any]]:
        path = os.path.join(args.data_dir, name)
        if not os.path.exists(path):
            raise SystemExit("缺 %s（用 tools.prep_task_data 生成）" % path)
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    data = {
        "mmlu": _jsonl("mmlu_subset.jsonl"),
        "gsm8k": _jsonl("gsm8k_subset.jsonl"),
        "mmlu_dev": _jsonl("mmlu_dev.jsonl"),
        "gsm8k_shots": _jsonl("gsm8k_shots.jsonl"),
    }
    if args.smoke:
        data["mmlu"] = data["mmlu"][:2]
        data["gsm8k"] = data["gsm8k"][:2]
        print("★ SMOKE：每数据集只跑 2 题 —— 验证通路，不是测量结果。", flush=True)

    max_new = {"mmlu": 8, "gsm8k": args.max_new_tokens}

    # ---- 模型 ----
    model, _ = _load_model(args.model, "float16", "cuda")
    model.eval()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    # fp16 先跑（参照 + 口径 B 的轨迹来源），然后量化臂
    print("=== arm fp16（参照）===", flush=True)
    quant_arms = [a for a in args.arms if a != "fp16"]
    fp16_traces: dict[str, list[dict[str, Any]]] | None = (
        {"mmlu": [], "gsm8k": []} if quant_arms else None
    )
    fp16_result = _run_arm(model, tok, "fp16", data, max_new, traces_out=fp16_traces)

    # 轨迹要重存一次：_run_arm 里没外存 prompt/gen —— 为了口径 B，fp16 跑完直接量化，
    # 口径 B 在量化臂循环内重算 fp16 的 prompt + gen。因此 fp16 臂要存轨迹。
    result: dict[str, Any] = {
        "tool": "tools/task_accuracy.py",
        "model": args.model,
        "status": "done",
        "smoke": bool(args.smoke),
        "subsets": PLANNED_SUBSETS,
        "protocol": {
            "A_end_to_end": "各臂各自 greedy 生成 → accuracy",
            "B_teacher_forcing": "fp16 生成的 token 强制喂给量化臂 → 逐位置 argmax 一致率",
        },
        "gate": {"threshold": "<2 个点（METRICS §1，自定建议值）", "per_task": {}},
        "arms": {"fp16": fp16_result},
    }

    # ---- 量化臂 ----
    if quant_arms:
        # 快照 fp16 权重（量化臂之间共享一次快照）
        all_names: list[str] = []
        for arm in quant_arms:
            all_names.extend(_candidate_names(model, targets=["attention", "experts"], layers=None))
        all_names = sorted(set(all_names))
        snapshot = _snapshot(model, all_names)
        from lemoq.quant.linear import SchemeSpec, quantize_model

        for arm in quant_arms:
            _restore(model, snapshot)
            bits = 4 if arm.startswith("w4") else 8
            spec = SchemeSpec.from_flags("w%da16" % bits, 128, targets=("attention", "experts"))
            print("=== arm %s（quantize in-place）===" % arm, flush=True)
            quantize_model(model, spec, names=None)
            res = _run_arm(model, tok, arm, data, max_new, fp16_traces=fp16_traces)
            result["arms"][arm] = res
            for task in ("mmlu", "gsm8k"):
                a = fp16_result["tasks"][task]["accuracy"]
                b = res["tasks"][task]["accuracy"]
                result["gate"]["per_task"]["%s@%s" % (arm, task)] = {
                    "fp16": a,
                    "arm": b,
                    "delta_pp": (b - a) * 100,
                    "pass": (b - a) * 100 > -2.0,
                }

    out_path = args.json or "results/task_accuracy.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    print("\nwrote %s" % out_path)
    if args.smoke:
        print("★ SMOKE 结果只证明通路可用，不得引用任何数字（硬规则 1）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
