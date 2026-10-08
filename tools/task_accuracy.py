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
    "mmlu": {"n": 200, "shots": 5, "note": "分层抽样子集（不是全量 MMLU）"},
    "gsm8k": {"n": 100, "shots": 0, "note": "子集（不是全量 GSM8K）"},
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

    # ★ 到这一步仍**没有实现**生成/评分循环：这是**故意的**。
    #   在没有 GPU 的环境里写一个跑不了的评测循环，只会产出一个看起来能跑、
    #   实际上从未被验证过的脚本 —— 那正是本项目反对的"未测装成已测"。
    #   上机第一步就是补这里，并且第一件事是跑 smoke（每数据集 2 题）验证通路。
    print(
        "task_accuracy 的评测循环**尚未实现**（刻意留白）。\n"
        "理由：无 GPU 环境下写出来的循环无法验证，等价于未测装成已测。\n"
        "上机后的第一步：补 `_run_arm()`（greedy 生成 + 答案抽取 + 计分），\n"
        "先跑 --smoke（每数据集 2 题）验证通路，再跑完整子集。\n"
        "在那之前，本项的 status 一律是 **planned**。",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
