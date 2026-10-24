"""quant_ppl — 用 **PPL（wikitext-2）+ 配对 ΔNLL** 做量化精度的第二个指标。

## 为什么需要第二个指标（这是 EXP-p1-02 的直接产物）

`EXP-p1-02` 实测：逐位置 KL 在 0.03 量级上**同一个误差量级、换一个扰动方向**就能差
**19×**，而 group_size 之间的差只有 1.73×。也就是说 KL 在这个量级上分不出方案好坏。

`docs/reference/METRICS.md` §1 早就把 PPL 列为指标（门槛：相对退化 <1%），但仓库里**从来没有实现过**
（`result.json` 里的 `ppl_fp16/ppl_this/ppl_delta` 一直是 null）。本工具补上它，
并且刻意报**配对**统计：

    所有臂在**同一串 token** 上算 NLL（同一份文本、同样的切窗），
    于是"文本本身的难易"在臂间相减时抵消，剩下的是**配对差值**：
        ΔNLL/token = mean_i [ logp_fp16(t_i) − logp_arm(t_i) ]
        SE          = std(diff_i) / sqrt(n)
    PPL = exp(mean NLL)，同时报 ΔPPL。

    ★ 为什么不只报 PPL：PPL 的绝对值受文本难度主导，两个方案的 PPL 差 0.5% 时，
      逐 token 的噪声会把结论淹没。配对差值把这个噪声消掉，并给出一个**可判断的**误差棒
      —— "差 0.5% ± 0.1%" 是可判断的，"PPL 6.41 vs 6.44" 不是。

## 口径（必须写清楚，否则 PPL 不可比）

- 数据：`wikitext-2-raw-v1` 的 **test** split（HF `datasets`），非空行用 `\\n` 连接，
  取**前 `--tokens` 个 token**（不挑样本）。
- 切窗：**不重叠**的 `--ctx` 长窗口；每个窗口只对"有左文的下一个 token"计 NLL。
  不重叠是刻意的：每个 token 只被计一次，臂间的分母完全相同。
- 这是**固定文本上的 NLL**，天然是 teacher forcing，不涉及采样策略。

## 臂的定义与 `tools/quant_probe.py` **共用**

`--arms` 用的是同一个 `build_arms()`：所以 PPL 与 KL 可以在**完全相同的扰动**下比较，
这才是"两个指标互相校验"该有的样子。恢复机制也是同一套（CPU 快照 + `noop` 自检）。

用法：

    export HF_ENDPOINT=https://hf-mirror.com        # huggingface.co 在服务器上不可达
    python -m tools.quant_ppl --model M --arms repro_attn,noise,mixed --tokens 8192
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.quant_probe import (  # noqa: E402
    _candidate_names,
    _inject_noise,
    _restore,
    _sigma_table,
    _snapshot,
    build_arms,
)

WIKITEXT = ("wikitext", "wikitext-2-raw-v1")


def load_wikitext_tokens(tokenizer: Any, n_tokens: int, *, cache: str | None = None) -> Any:
    """wikitext-2 test 的前 `n_tokens` 个 token（可缓存到磁盘，避免每次联网）。"""
    import torch

    if cache and os.path.exists(cache):
        ids = torch.load(cache, weights_only=True)
        return ids[:n_tokens] if ids.numel() >= n_tokens else ids
    from datasets import load_dataset

    ds = load_dataset(*WIKITEXT, split="test")
    text = "\n".join(line for line in ds["text"] if line.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        torch.save(ids, cache)
    return ids[:n_tokens] if ids.numel() >= n_tokens else ids


def ppl_and_nll(model: Any, ids: Any, *, ctx: int) -> dict[str, Any]:
    """**逐 token** 的 NLL（不重叠窗口）。返回 PPL、总 NLL、逐 token NLL 列表。

    逐 token 的 NLL 是配对统计的基础：两条臂的差要在**同一个 token 上**相减。
    """
    import torch

    per_token: list[float] = []
    total = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, int(ids.numel()) - 1, ctx):
            # ids 缓存在 CPU（省显存、缓存可复用），进模型前再搬
            chunk = ids[start : start + ctx].to(device)
            if chunk.numel() < 2:
                break
            logits = model(chunk.view(1, -1)).logits[0, :-1, :].float()
            targets = chunk[1:]
            logp = torch.log_softmax(logits, dim=-1)
            picked = logp.gather(1, targets.view(-1, 1)).squeeze(1)
            per_token.extend(float(x) for x in picked.cpu())
            total += float(-picked.sum())
    n = len(per_token)
    return {
        "n_tokens": n,
        "nll_total": total,
        "nll_per_token": total / n if n else float("nan"),
        "ppl": math.exp(total / n) if n else float("nan"),
        # ★ 名字里有 logp 而不是 nll：这个列表装的是 log_softmax 取出来的值
        #   （恒 ≤ 0），即 **−NLL**。旧名叫 `_per_token_nll`，于是 `paired_delta`
        #   的差值方向被读反过一次（见那里的注释）。
        "_per_token_logp": per_token,
    }


def paired_delta(reference: list[float], candidate: list[float]) -> dict[str, Any]:
    """配对差值：**正 = 本臂更差**，负 = 本臂更好。

    ★ 口径（与 `EXP-p1-03` 记录 §2 逐字一致，改这里必须先改记录）：

        delta_i = logp_fp16(t_i) − logp_arm(t_i) = NLL_arm(t_i) − NLL_fp16(t_i)

    因为 logp = −NLL，所以 `delta > 0` 意味着**本臂的 NLL 更高**，也就是**更差**。
    旧注释写的是「参考的 NLL 减去本臂的 NLL，正 = 本臂更好」—— 那是这个量的
    **相反数**，会把表格里每一行读反（`w4a16_attn_g32` 的 +0.019973 配合 PPL
    退化 +2.02%，正是"更差"）。数值本身一直是对的，错的是这段说明。
    """
    if len(reference) != len(candidate) or not reference:
        raise ValueError(
            "配对比较要求逐 token 数量相同：%d vs %d" % (len(reference), len(candidate))
        )
    diffs = [a - b for a, b in zip(reference, candidate)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    return {
        "delta_nll_per_token": mean,
        "paired_se": se,
        "t_stat": (mean / se) if se else None,
        "n_pairs": n,
        # ★ 方向：正 = 本臂 NLL 更高 = 更差（见 docstring）。旧文案写反了。
        "note": (
            "delta = mean(logp_fp16 − logp_arm) = mean(NLL_arm − NLL_fp16)；"
            "正 = 本臂 NLL 更高（更差），负 = 本臂更好；|t| < 2 视为不可分辨"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="repro_attn,noise,mixed")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument(
        "--tokens", type=int, default=8192, help="wikitext 上取的 token 数（不重叠窗口）"
    )
    ap.add_argument("--cache", default=None, help="token id 缓存路径（避免重复联网）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer

    from benchmarks.runner import _load_model

    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2

    include = {a.strip() for a in args.arms.split(",") if a.strip()}
    # ★ noop 臂保留：它同样依赖 CPU 快照 + 恢复，而"恢复不干净"在配对指标上表现为
    #   noop 的 ΔNLL ≠ 0。丢掉自检等于在最需要它的地方放弃验证。
    arms = build_arms(include=include)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = load_wikitext_tokens(tokenizer, args.tokens, cache=args.cache)
    print("wikitext-2 test: %d tokens, 窗口 %d" % (int(ids.numel()), args.ctx), flush=True)

    model, _ = _load_model(args.model, "float16", "cuda")
    result: dict[str, Any] = {
        "tool": "tools/quant_ppl.py",
        "model": args.model,
        "dataset": "%s/%s test" % WIKITEXT,
        "n_tokens": int(ids.numel()),
        "ctx_window": args.ctx,
        "windowing": "不重叠窗口；每个 token 只计一次；每个窗口第一个 token 不计",
        "metric": "PPL + 与 fp16 的**配对** ΔNLL/token（±paired SE）",
        "arms": {},
    }

    # 参照臂：fp16（不扰动）
    ref = ppl_and_nll(model, ids, ctx=args.ctx)
    result["fp16"] = {k: v for k, v in ref.items() if not k.startswith("_")}
    print(
        "fp16: ppl=%.4f nll/token=%.6f (%d tokens)"
        % (ref["ppl"], ref["nll_per_token"], ref["n_tokens"]),
        flush=True,
    )

    all_names: list[str] = []
    for arm in arms:
        targets = arm.get("targets") or ["attention"]
        if arm["kind"] == "mixed":
            targets = sorted({t for e in arm["plan"] for t in e["targets"]})
        for name in _candidate_names(model, targets=targets, layers=arm.get("layers")):
            if name not in all_names:
                all_names.append(name)
    snapshot = _snapshot(model, all_names)
    from lemoq.quant.linear import SchemeSpec, quantize_model

    sigma_table = _sigma_table(model, _candidate_names(model, targets=["attention"], layers=None))

    def _write() -> None:
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    for arm in arms:
        t0 = time.perf_counter()
        _restore(model, snapshot)
        arm_targets = arm.get("targets") or ["attention"]
        if arm["kind"] == "mixed":
            arm_targets = sorted({t for e in arm["plan"] for t in e["targets"]})
        arm_names = _candidate_names(model, targets=arm_targets, layers=arm.get("layers"))
        if arm.get("filter"):
            arm_names = [n for n in arm_names if arm["filter"] in n]
        entry: dict[str, Any] = {"kind": arm["kind"], "group": arm["group"]}
        try:
            if arm["kind"] == "noop":
                entry["note"] = "恢复但不扰动：ΔNLL 必须**恰好** 0.0"
            elif arm["kind"] == "rtn":
                spec = SchemeSpec.from_flags(
                    arm["scheme"],
                    arm["group_size"],
                    targets=tuple(arm["targets"]),
                    # ★ F5（EXP-p1-06）：臂要量 lm_head 就必须显式关掉这个保护。
                    #   2026-10-25 实测：漏传它会让 lmhead_only 臂**静默替换 0 个模块**
                    #   （realized_bytes={}、ΔNLL 与 fp16 逐位相同——那次 PPL 因此作废）。
                    keep_lm_head_fp16=bool(arm.get("keep_lm_head_fp16", True)),
                )
                _, qreport = quantize_model(
                    model,
                    spec,
                    layers=arm.get("layers"),
                    names=arm_names if arm.get("filter") else None,
                )
                entry["realized_bytes_total"] = qreport.to_dict()["realized_bytes_total"]
                entry["realized_bytes_by_target"] = qreport.to_dict()["realized_bytes_by_target"]
            elif arm["kind"] == "noise":
                entry.update(
                    _inject_noise(
                        model,
                        arm_names,
                        sigma_table,
                        lam=float(arm["lam"]),
                        direction=int(arm["direction"]),
                    )
                )
                entry["lam"] = arm["lam"]
                entry["direction"] = arm["direction"]
            elif arm["kind"] == "mixed":
                total = 0
                plan = []
                for sub in arm["plan"]:
                    spec = SchemeSpec.from_flags(
                        sub["scheme"], sub["group_size"], targets=tuple(sub["targets"])
                    )
                    names = [n for n in arm_names if sub["filter"] in n]
                    if not names:
                        raise ValueError("mixed 子计划 %s 没匹配到张量" % sub)
                    _, report = quantize_model(model, spec, names=names)
                    buckets = report.to_dict()["realized_bytes_by_target"]
                    total += sum(b["total_bytes"] for b in buckets.values())
                    plan.append(
                        {"filter": sub["filter"], "scheme": sub["scheme"], "n_names": len(names)}
                    )
                entry["plan"] = plan
                entry["realized_bytes_total"] = int(total)
            else:
                raise ValueError("quant_ppl 不支持 kind=%r" % arm["kind"])

            measured = ppl_and_nll(model, ids, ctx=args.ctx)
            entry.update({k: v for k, v in measured.items() if not k.startswith("_")})
            entry["paired_vs_fp16"] = paired_delta(
                ref["_per_token_logp"], measured["_per_token_logp"]
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        result["arms"][arm["id"]] = entry
        _write()
        if arm["kind"] == "noop":
            delta = (entry.get("paired_vs_fp16") or {}).get("delta_nll_per_token")
            if delta is None or abs(delta) > 1e-9:
                result["restore_check_failed"] = {"arm": arm["id"], "delta_nll": delta}
                _write()
                raise RuntimeError("恢复自检失败：noop 的 ΔNLL=%r（应为 0.0）" % (delta,))
        print(
            "  %-34s ppl=%-10.4f dNLL/token=%+.6f ±%.6f (t=%s)  bytes=%.3f GB (%.1fs)"
            % (
                arm["id"],
                entry.get("ppl", float("nan")),
                (entry.get("paired_vs_fp16") or {}).get("delta_nll_per_token", float("nan")),
                (entry.get("paired_vs_fp16") or {}).get("paired_se", float("nan")),
                round((entry.get("paired_vs_fp16") or {}).get("t_stat") or 0.0, 2),
                (entry.get("realized_bytes_total") or 0) / 1e9,
                entry["seconds"],
            ),
            flush=True,
        )

    _restore(model, snapshot)
    _write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
