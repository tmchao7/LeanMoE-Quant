"""prep_task_data — 生成 `tools/task_accuracy.py` 需要的本地 jsonl 子集。

为什么需要这个工具：上机机直连 huggingface.co 通常不通（本仓 `scripts/preflight.sh`
的结论），数据要经 hf-mirror 拉；且任务级精度（METRICS §6 G1）只对**固定的子集**跑，
子集本身必须可复现 —— 所以抽样是确定性的（固定 seed + 分层），并把选出的
例子的 id 落盘到 jsonl，谁在哪个实例上重新生成都得到同一批题。

产出（`--out-dir` 下）：
  mmlu_subset.jsonl   每行 {question, choices:[4], answer:int, subject:str, source_id}
  gsm8k_subset.jsonl  每行 {question, answer:str, answer_int:int, source_id}

★ 这是**数据准备**，不是测量：本脚本不做任何推理、不产出精度数字。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SEED = 42
MMLU_N = 200  # 与 tools/task_accuracy.SUBSETS 对齐
GSM8K_N = 100


def prep_mmlu(out_path: str, n: int) -> str:
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("cais/mmlu", "all", split="test")
    # 分层：每个 subject 至少 1 题、按 subject 大小比例分配名额（确定性）。
    subjects = sorted(set(ds["subject"]))
    by_subj: dict[str, list[int]] = {s: [] for s in subjects}
    for i, s in enumerate(ds["subject"]):
        by_subj[s].append(i)
    import random

    rng = random.Random(SEED)
    quota = {s: max(1, round(n * len(idx) / len(ds))) for s, idx in by_subj.items()}
    # 比例取整可能超出 n，按 quota 降序裁掉
    while sum(quota.values()) > n:
        s = max(quota, key=lambda k: quota[k])
        quota[s] -= 1
    rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in sorted(quota):
            idx = sorted(rng.sample(by_subj[s], quota[s]))
            for i in idx:
                ex = ds[i]
                f.write(
                    json.dumps(
                        {
                            "question": ex["question"],
                            "choices": list(ex["choices"]),
                            "answer": int(ex["answer"]),
                            "subject": s,
                            "source_id": "mmlu-all-test-%d" % i,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
    return "mmlu: %d subjects, %d rows -> %s" % (len(subjects), rows, out_path)


def prep_gsm8k(out_path: str, n: int) -> str:
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset("openai/gsm8k", "main", split="test")
    import random

    rng = random.Random(SEED)
    idx = sorted(rng.sample(range(len(ds)), n))
    rows = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in idx:
            ex = ds[i]
            # gsm8k 的 answer 形如 "...推算...\n#### 42"
            final = ex["answer"].split("####")[-1].strip().replace(",", "")
            f.write(
                json.dumps(
                    {
                        "question": ex["question"],
                        "answer": final,
                        "answer_int": int(final),
                        "source_id": "gsm8k-main-test-%d" % i,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rows += 1
    return "gsm8k: %d rows -> %s" % (rows, out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, help="jsonl 输出目录")
    ap.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"),
        help="直连不通时用镜像（默认 hf-mirror.com）",
    )
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        print("HF_ENDPOINT=%s" % args.hf_endpoint)

    out_mmlu = os.path.join(args.out_dir, "mmlu_subset.jsonl")
    out_gsm = os.path.join(args.out_dir, "gsm8k_subset.jsonl")
    print(prep_mmlu(out_mmlu, MMLU_N))
    print(prep_gsm8k(out_gsm, GSM8K_N))
    print("done. 记得把 out-dir 传给 tools/task_accuracy --data-dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
