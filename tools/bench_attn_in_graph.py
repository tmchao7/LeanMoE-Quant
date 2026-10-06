"""bench_attn_in_graph — **图内**的 decode-attention 臂对照（本任务 D 步的闸门）。

## 为什么靶子必须来自图内（`EXP-p2-04` 的教训）

`tools/bench_decode_attention.py` 把每个臂 eager 地跑在紧循环里，而它们最终要被
capture 进图。进图后每算子的分配与启动开销消失，**多算子实现拿到一笔"图补贴"**：
隔离预测 ctx=1024 省 0.049 ms，实测省 **0.978 ms**（差 5.7 倍）。

所以这里的做法是：**两条臂各自捕获一张全模型 decode 图**，然后逐轮交错重放。
交织的目的（见 `_interleaved` 的注释）：这台机器是共享的，`pareto_offload.py`
会随时抢 GPU，顺序测会把争用当成臂差异（实测同一个 config 两次差 29%）。
报 `wins`（逐轮最快次数）与 MAD，不只报中位数。

## 两条臂

    eager   HF 的 eager 数学实现（`attn_implementation="eager"`）= `EXP-p2-04` 的基线
    kernel  本仓库的 split-KV decode attention（实例级替换 attention core）

★ 捕获顺序是**有意义**的：`eager` 必须先捕获（那时模型还没被替换）。被替换的是
实例属性上的 `forward`，而图里记录的是**内核**，所以在替换之后重放 eager 臂仍然是
原来的内核 —— 这是实例级 patch（而不是类级）带来的可推理性质。

两条臂的 host 侧每步开销**相同**（同样的 `_set_step_inputs`：token/position/mask/filled），
token 序列也由同一份脚本驱动，所以量到的差异只可能来自 attention core。

## 判据

    K1  kernel vs fp64 oracle（误差不差于基线 × 1.5）        —— 见 tests/test_attn_decode.py
    K2  graph+kernel vs 同路径 eager+kernel：KL 必须恰好 0
    K3  两条臂之间的 KL：**只记录，不当门槛**（跨实现差异是预期的）
    K4  在 ctx 捕获 → 改成 M 重放：必须给出 M 的答案（err < 1e-2），
        且"陈旧答案"必须离正确答案足够远，否则判 INCONCLUSIVE 而不是 PASS
    D   图内 per-token：kernel 更快、逐轮 wins 占多数、attention 那几行消失

用法：

    python -m tools.bench_attn_in_graph --model "$M" --ctx 1024 --json results/.../attn_in_graph.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.cudagraph import (  # noqa: E402
    StaticCacheDecodeSession,
    _profiler_rows,
    compare_logits,
    kernel_table,
)
from benchmarks.runner import (  # noqa: E402
    _load_model,
    _static_eager_correctness,
    _static_graph_correctness,
)

# §2 的图内 attention 核心行口径：这些名字 + **每层出现一次**（occurrences == heads）。
# gemvx::kernel 同时被 MoE 路径用到（occ 81），所以"出现次数"是必需的过滤器 ——
# 不是装饰：`§2` 里那张表就是按"出现 16 次"筛出来的。
EAGER_ATTN_PATTERNS = (
    "gemvNSP_kernel",
    "cunn_SoftMaxForward",
    "softmax_warp_forward",
    "gemvx::kernel",
    "CUDAFunctor_add",
)
KERNEL_ATTN_PATTERNS = ("_attn_split_kernel", "_attn_combine_kernel")
_ATTN_OCC_TOL = 1e-6


def _layered_rows(table: list[dict[str, Any]], *, heads: int) -> dict[tuple[str, float], float]:
    """每层恰好出现一次的行（`occurrences == heads`）。

    图内的 per-layer 算子只可能是 attention / 路由 / 残差 —— 用"出现次数"过滤是
    §2 定下的口径（那里就是从 `occurrences_per_token == 16` 的行里取 attention 的）。
    """
    rows: dict[tuple[str, float], float] = {}
    for row in table:
        occ = float(row.get("occurrences_per_token") or 0.0)
        if abs(occ - heads) > _ATTN_OCC_TOL:
            continue
        rows[(str(row.get("name") or ""), occ)] = float(row["self_ms_per_token"])
    return rows


def _attention_delta(
    eager_table: list[dict[str, Any]], kernel_table: list[dict[str, Any]], *, heads: int
) -> dict[str, Any]:
    """两臂的**集合差**，而不是我手挑一份名字列表。

    为什么这样做：`softmax_warp_forward` / `gemvx::kernel` / `CUDAFunctor_add` 这些
    名字**同时**出现在 routing、残差、投影里（它们在两臂都出现，属于共享行）。
    手挑一个名字列表会把它们算进 attention 的核心代价里，那正是 §2 那张表最容易
    被误读的地方。集合差只报"这条臂有、那条臂没有"的行 —— 于是：
        消失的行（eager 有、kernel 没有）= 被换掉的 attention 核心
        新增的行（kernel 有、eager 没有）= 换上去的 kernel
        交集 = 与本次替换无关的共享行
    """
    eager_layered = _layered_rows(eager_table, heads=heads)
    kernel_layered = _layered_rows(kernel_table, heads=heads)
    vanished = {k: v for k, v in eager_layered.items() if k not in kernel_layered}
    appeared = {k: v for k, v in kernel_layered.items() if k not in eager_layered}
    shared = {k: v for k, v in eager_layered.items() if k in kernel_layered}

    def _fmt(rows: dict[tuple[str, float], float]) -> list[dict[str, Any]]:
        return [
            {"name": name[:80], "occurrences_per_token": occ, "self_ms_per_token": ms}
            for (name, occ), ms in sorted(rows.items(), key=lambda kv: -kv[1])
        ]

    vanished_ms = round(sum(vanished.values()), 5)
    appeared_ms = round(sum(appeared.values()), 5)
    return {
        "per_layer_occurrences_expected": heads,
        "vanished_rows": _fmt(vanished),
        "new_rows": _fmt(appeared),
        "shared_rows": _fmt(shared),
        "vanished_ms_per_token": vanished_ms,
        "new_ms_per_token": appeared_ms,
        "net_saving_ms_per_token": round(vanished_ms - appeared_ms, 5),
        "eager_attention_signature_vanished": bool(
            any(any(p in name for p in EAGER_ATTN_PATTERNS) for name, _ in vanished)
        ),
        "kernel_signature_appeared": bool(
            any(any(p in name for p in KERNEL_ATTN_PATTERNS) for name, _ in appeared)
        ),
        "note": (
            "消失/新增按 (name, occurrences) 集合差算；共享行（路由/残差/投影）不进 attention 口径"
        ),
    }


def _row_delta_all(
    eager_table: list[dict[str, Any]], kernel_table: list[dict[str, Any]]
) -> dict[str, Any]:
    """**不按出现次数过滤**的整表集合差 —— 用来解释总 busy 的差值从哪来。

    为什么还要这一个：`per-token` 的收益（ctx=1024 实测 0.115 ms）大于"每层一次的
    attention 行"的净差（0.038 ms）。差额来自那些**出现次数也会变**的行（fp16 拷贝、
    elementwise、launch 数）。只报 occ==16 那一层会让人以为收益只有 0.038。
    """
    eager_rows = {str(r["name"]): r for r in eager_table}
    kernel_rows = {str(r["name"]): r for r in kernel_table}

    def _fmt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "name": str(r["name"])[:80],
                "occurrences_per_token": r["occurrences_per_token"],
                "self_ms_per_token": r["self_ms_per_token"],
            }
            for r in sorted(rows, key=lambda r: -float(r["self_ms_per_token"]))
        ]

    vanished = [eager_rows[n] for n in eager_rows if n not in kernel_rows]
    appeared = [kernel_rows[n] for n in kernel_rows if n not in eager_rows]
    return {
        "vanished_rows": _fmt(vanished),
        "new_rows": _fmt(appeared),
        "vanished_ms_per_token": round(sum(float(r["self_ms_per_token"]) for r in vanished), 5),
        "new_ms_per_token": round(sum(float(r["self_ms_per_token"]) for r in appeared), 5),
        "note": "按 name 集合差（不看出现次数）；两张表都取 limit=60 行，避免截断造成假差异",
    }


def _script_tokens(session: Any, n: int) -> list[int]:
    """从一条臂自己的贪心链上取 n 个**输入** token。

    ★ 为什么要有 token 脚本：如果不固定 token，两条臂各自解码，一旦某步 top-1 不同，
    后面就在处理不同的 token —— 那时量到的差异就不只是 attention core 了。
    固定脚本 + 固定 position，使"一次一个变量"成立。
    """
    session.reset()
    script = [int(session.seed_token_id)]
    for _ in range(n - 1):
        session.replay(synchronize=True)
        script.append(int(session.current_token_id))
    session.reset()
    return script


def _run_scripted(session: Any, script: list[int], *, offset: int, count: int) -> None:
    for i in range(offset, offset + count):
        session.current_token_id = int(script[i])
        session.replay(synchronize=True)
    session.stream.synchronize()


def _timed_round(session: Any, script: list[int], *, warmup: int, timed: int) -> dict[str, float]:
    """一轮计时：warmup 在窗口外，timed 段用 wall + CUDA event 两种口径。"""
    torch = session.torch
    session.reset()
    _run_scripted(session, script, offset=0, count=warmup)
    session.reset()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(session.stream)
    t0 = time.perf_counter()
    _run_scripted(session, script, offset=0, count=timed)
    wall = (time.perf_counter() - t0) * 1000.0 / timed
    end.record(session.stream)
    session.stream.synchronize()
    return {"wall_ms": wall, "event_ms": start.elapsed_time(end) / timed}


def _profile_arm(
    session: Any, script: list[int], *, warmup: int, timed: int, limit: int = 60
) -> dict[str, Any]:
    """图内 kernel 归因：与 runner 的 `bottleneck` 表**同一个函数**（可直接引用）。"""
    torch = session.torch
    session.reset()
    _run_scripted(session, script, offset=0, count=warmup)
    session.reset()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        _run_scripted(session, script, offset=0, count=timed)
    events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    busy = sum(e.self_device_time_total for e in events) / timed / 1000.0
    kernels = sum(e.count for e in events) / float(timed)
    return {
        "gpu_busy_ms": round(busy, 5),
        "kernels_per_call": round(kernels, 1),
        # limit=60 而不是 runner 的 25：下面的集合差**不能**建立在被截断的表上
        # （第一版用 25 行，把 kernel 臂里挤出榜的 routing softmax 误报成"新增行"）。
        "top_kernels": kernel_table(_profiler_rows(prof), timed=timed, busy_ms=busy, limit=limit),
    }


def _interleaved(
    arms: dict[str, Any], scripts: dict[str, list[int]], *, rounds: int, warmup: int, timed: int
) -> dict[str, Any]:
    """逐轮交错：每一轮里两条臂各测一次，轮内顺序交替，避免"谁先谁后"变成系统偏差。"""
    names = list(arms)
    samples: dict[str, list[float]] = {n: [] for n in names}
    events: dict[str, list[float]] = {n: [] for n in names}
    ranks: list[dict[str, int]] = []
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        round_ms: dict[str, float] = {}
        for name in order:
            s = _timed_round(arms[name], scripts[name], warmup=warmup, timed=timed)
            samples[name].append(s["wall_ms"])
            events[name].append(s["event_ms"])
            round_ms[name] = s["wall_ms"]
        ranking = sorted(round_ms, key=lambda n: round_ms[n])
        ranks.append({n: ranking.index(n) + 1 for n in names})
    report: dict[str, Any] = {}
    for name in names:
        med = statistics.median(samples[name])
        report[name] = {
            "per_token_wall_p50_ms": round(med, 5),
            "per_token_wall_min_ms": round(min(samples[name]), 5),
            "per_token_wall_mad_ms": round(
                statistics.median([abs(v - med) for v in samples[name]]), 6
            ),
            "per_token_event_p50_ms": round(statistics.median(events[name]), 5),
            "wins": sum(1 for r in ranks if r[name] == 1),
            "rounds": rounds,
            "samples_ms": [round(v, 5) for v in samples[name]],
        }
    noise = statistics.median([v["per_token_wall_mad_ms"] for v in report.values()])
    speedup = (
        report[names[0]]["per_token_wall_p50_ms"] / report[names[1]]["per_token_wall_p50_ms"]
        if len(names) == 2
        else None
    )
    return {
        "method": "interleaved round-robin over two separately captured full-model decode graphs",
        "rounds": rounds,
        "warmup": warmup,
        "timed": timed,
        "per_arm": report,
        "speedup_first_over_second": round(speedup, 4) if speedup else None,
        "noise_floor_ms": round(noise, 6),
        "caveat": "可分辨的最小差异约等于 MAD；比 MAD 小的差值不要当成结论",
    }


def _length_change(session: Any, *, target_position: int, capture_position: int) -> dict[str, Any]:
    """K4：在 capture_position 捕获的图上，用 target_position 重放。

    ★ 两个曾经的 bug（`tools/bench_decode_attention.py` 的第一版）在这里被显式避开：
      1. 只改 host 变量、没改**设备端**状态 —— 这里改的是 `filled_len` buffer（设备内存），
         也就是 kernel 真正读的那个东西；
      2. 两种长度的答案趋同 → 判据永不失败 —— 这里除了 err，还必须报 `stale_err`
         （陈旧答案离正确答案多远），由 `judge_length_change` 决定 PASS/FAIL/INCONCLUSIVE。
    """
    want = None
    got = None
    stale = None
    session.reset()
    session.set_position(target_position)
    want = session.decode_eager().detach().clone()
    session.reset()
    session.set_position(target_position)
    got = session.replay_at_position(target_position).detach().clone()
    session.reset()
    stale = session.decode_eager().detach().clone()
    session.reset()

    correct_err = float((got.float() - want.float()).abs().max())
    stale_err = float((stale.float() - want.float()).abs().max())
    from lemoq.kernels.attn_decode import judge_length_change  # noqa: PLC0415

    verdict = judge_length_change(correct_err=correct_err, stale_err=stale_err)
    verdict.update(
        {
            "captured_at_position": int(capture_position),
            "replayed_at_position": int(target_position),
            "correct_answer_is_target_length": True,
            "same_path_kl": compare_logits([want], [got]),
            "note": (
                "correct_err 是图重放结果与**同路径 eager**（同长度）的差；"
                "stale_err 是旧长度答案与新长度正确答案的距离 —— 后者是判别力，不是噪声"
            ),
        }
    )
    return verdict


def _kernel_vs_oracle(model: Any, config: dict[str, Any] | None) -> dict[str, Any]:
    """K1：真 kernel vs fp64 oracle，与 fp16 基线（oracle 路径）比误差。"""
    import torch

    from lemoq.kernels.attn_decode import (
        PARITY_GATE,
        attn_decode_forward,
        dense_attention_oracle,
    )

    heads = int(model.config.num_attention_heads)
    dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // heads))
    torch.manual_seed(0)
    seq, filled = 1056, 33
    device = "cuda"
    q = (torch.randn(1, heads, 1, dim, device=device) * 0.1).half()
    k = (torch.randn(1, heads, seq, dim, device=device) * 0.1).half()
    v = (torch.randn(1, heads, seq, dim, device=device) * 0.1).half()
    v[:, :, filled:, :] = 20.0  # 判别力：读到尾部就会差 O(10)
    ft = torch.full((1,), filled, dtype=torch.int32, device=device)
    want = dense_attention_oracle(q, k, v, filled).float()
    kernel = attn_decode_forward(q, k, v, ft, config=config).float().reshape(heads, dim)
    baseline = attn_decode_forward(q, k, v, ft, config=config, backend="oracle")
    baseline = baseline.float().reshape(heads, dim)
    kernel_err = float((kernel - want).abs().max())
    baseline_err = float((baseline - want).abs().max())
    return {
        "kernel_err": kernel_err,
        "baseline_err": baseline_err,
        "tol_slack": PARITY_GATE["tol_slack"],
        "slack_ratio": round(kernel_err / baseline_err, 4) if baseline_err else None,
        "allclose_atol_rtol_1e-3": bool(
            torch.allclose(kernel, want, atol=PARITY_GATE["tol_atol"], rtol=PARITY_GATE["tol_rtol"])
        ),
        "verdict": "PASS" if kernel_err <= baseline_err * PARITY_GATE["tol_slack"] else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=20)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="StaticCache 容量（ctx + 这个数），与 EXP-p2-04 的 runner 口径保持一致",
    )
    ap.add_argument(
        "--split-grid",
        default="1,2,4,8,11,16,32",
        help="num_splits 扫描网格（**报全部**，不只报最好的那个）",
    )
    ap.add_argument("--no-sweep", action="store_true", help="跳过 num_splits 扫描（更快）")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch

    from lemoq.kernels.attn_decode import (
        attn_decode_available,
        decode_call_count,
        replace_attention_cores,
        resolve_num_splits,
    )
    from lemoq.kernels.attn_decode import (
        warm as warm_attn,
    )

    if not torch.cuda.is_available():
        print("需要 CUDA", file=sys.stderr)
        return 2
    if not attn_decode_available():
        print("需要 triton", file=sys.stderr)
        return 2

    # ★ 容量必须与 `EXP-p2-04` 的 runner 口径一致（ctx + max_new_tokens = 1024 + 32 = 1056）：
    #   `max_cache_len` 决定 cache 的 head stride，而 cuBLAS 的 batched-GEMV 选择**依赖 stride**。
    #   我第一版把它设成 ctx + (warmup + timed + 2) = 1051，于是 eager 臂用了另一个 GEMV
    #   （gemvx 0.1498 而不是 gemvNSP 0.0928），基线被抬高到 3.1852 ms —— 那会**虚报**收益。
    capacity = int(args.max_new_tokens)
    steps_needed = args.warmup + args.timed
    if steps_needed > capacity:
        print(
            "warmup+timed=%d 超过 --max-new-tokens=%d（StaticCache 会溢出）"
            % (steps_needed, capacity),
            file=sys.stderr,
        )
        return 2
    model, _dtype = _load_model(args.model, args.dtype, "cuda", attn="eager")
    heads = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", 0) or (model.config.hidden_size // heads))
    sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
    default_splits = resolve_num_splits(None, heads=heads, device=torch.device("cuda"))
    ids = torch.full((1, args.ctx), 1, dtype=torch.long, device="cuda")

    # ★ MoE 也必须在图**外**换成 grouped：HF 的逐专家循环里有
    #   `torch.where(expert_mask[expert_idx])` 这种数据依赖的 host 侧操作，
    #   capture 时直接报 "operation not permitted when stream is capturing"。
    #   `EXP-p2-04` 的图内靶子也是在 `moe=grouped` 下测的 —— 本工具保持它不变，
    #   两条臂的差异才只剩 attention core（一次一个变量）。
    from lemoq.kernels.moe_grouped import warm as warm_grouped  # noqa: PLC0415
    from lemoq.kernels.moe_static import replace_moe_blocks  # noqa: PLC0415

    moe_replaced = replace_moe_blocks(model, impl="grouped")
    torch.cuda.synchronize()
    block = model.model.layers[0].mlp
    moe_warm = warm_grouped(
        block._static_experts,
        hidden=int(block._static_experts.gate_up.shape[2]),
        top_k=int(block._static_top_k),
    )
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "tool": "tools/bench_attn_in_graph.py",
        "shape": {
            "batch": 1,
            "q_len": 1,
            "heads": heads,
            "head_dim": head_dim,
            "ctx": args.ctx,
            "dtype": args.dtype,
            "sm_count": sm_count,
        },
        "held_fixed": {
            "moe": "grouped",
            "moe_blocks_replaced": moe_replaced,
            "moe_warm": moe_warm,
        },
        "num_splits_default": default_splits,
        "num_splits_default_rule": "ceil(sm_count / heads)",
        "arms": {},
    }

    # ---- 臂 A：eager（必须在替换之前捕获）----
    session_eager = StaticCacheDecodeSession(
        model, ids, ctx=args.ctx, max_new_tokens=capacity, device="cuda"
    )
    session_eager.capture(warmup=args.warmup)
    result["arms"]["eager"] = {"class": type(model.model.layers[0].self_attn).__name__}

    # ---- 臂 B：本仓库 kernel（实例级替换，图里记录内核，所以臂 A 不受影响）----
    replaced = replace_attention_cores(model, backend="triton")
    warm_attn(
        heads=heads,
        head_dim=head_dim,
        kv_len=args.ctx + capacity,
        filled=args.ctx + capacity,
        dtype=torch.float16,
        device="cuda",
    )
    session_kernel = StaticCacheDecodeSession(
        model, ids, ctx=args.ctx, max_new_tokens=capacity, device="cuda"
    )
    session_kernel.capture(warmup=args.warmup)
    result["arms"]["kernel"] = {
        "replaced_layers": replaced,
        "decode_calls_during_setup": decode_call_count(model),
    }

    arms = {"eager": session_eager, "kernel": session_kernel}
    scripts = {
        "eager": _script_tokens(session_eager, steps_needed),
        "kernel": _script_tokens(session_kernel, steps_needed),
    }
    result["token_script_matches"] = scripts["eager"] == scripts["kernel"]

    # ---- 图内交错计时 ----
    inter = _interleaved(arms, scripts, rounds=args.rounds, warmup=args.warmup, timed=args.timed)
    result["interleaved"] = inter

    # ---- 图内 kernel 归因 ----
    for name, session in arms.items():
        result["arms"][name]["profile"] = _profile_arm(
            session, scripts[name], warmup=args.warmup, timed=args.timed
        )
    delta = _attention_delta(
        result["arms"]["eager"]["profile"]["top_kernels"],
        result["arms"]["kernel"]["profile"]["top_kernels"],
        heads=heads,
    )
    result["attention_core"] = delta
    result["row_delta_all_occurrences"] = _row_delta_all(
        result["arms"]["eager"]["profile"]["top_kernels"],
        result["arms"]["kernel"]["profile"]["top_kernels"],
    )

    # ---- 判据 ----
    # ★ 顺序在这里是有意义的：`decode_eager()` 跑的是**模型当前**的 forward，
    #   而两条臂共用同一个模型实例。所以 kernel 臂的判据必须在 restore **之前**做，
    #   eager 臂的判据必须在 restore **之后**做 —— 否则"eager 对照"会悄悄变成 kernel。
    #   图不受影响：捕获之后图里记录的是内核，替换/还原 forward 都改不动它。
    horizon_corr = max(8, min(args.timed, capacity))
    ref = _static_eager_correctness(session_kernel, max_new_tokens=horizon_corr)
    cand = _static_graph_correctness(session_kernel, max_new_tokens=horizon_corr)
    k2 = compare_logits(ref["logits"], cand["logits"])
    k2.update(
        seed_token_match=ref["seed_token"] == cand["seed_token"],
        token_sequence_match=ref["output_tokens"] == cand["output_tokens"],
        note="K2：graph+kernel vs 同路径 eager+kernel；期望 KL 恰好 0",
    )
    cand_e = _static_graph_correctness(session_eager, max_new_tokens=horizon_corr)
    k3 = compare_logits(cand_e["logits"], cand["logits"])
    k3["note"] = "K3：跨实现差异，**只记录不当门槛**"
    k4_kernel = _length_change(session_kernel, target_position=32, capture_position=args.ctx)

    from lemoq.kernels.attn_decode import (  # noqa: PLC0415
        attention_cores_replaced,
        restore_attention_cores,
    )

    restored = restore_attention_cores(model)
    ref_e = _static_eager_correctness(session_eager, max_new_tokens=horizon_corr)
    cand_e2 = _static_graph_correctness(session_eager, max_new_tokens=horizon_corr)
    k2_eager = compare_logits(ref_e["logits"], cand_e2["logits"])
    k2_eager.update(
        seed_token_match=ref_e["seed_token"] == cand_e2["seed_token"],
        token_sequence_match=ref_e["output_tokens"] == cand_e2["output_tokens"],
        note="K2 对照：graph+eager vs 同路径 eager（未替换）—— 验证这套判据本身是好的",
    )
    k4_eager = _length_change(session_eager, target_position=32, capture_position=args.ctx)

    result["correctness"] = {
        "K1_kernel_vs_fp64_oracle": _kernel_vs_oracle(model, None),
        "K2_graph_vs_same_path_eager_kernel": k2,
        "K2_graph_vs_same_path_eager_control": k2_eager,
        "K3_informational_eager_vs_kernel": k3,
        "K4_length_change_kernel": k4_kernel,
        "K4_length_change_eager_control": k4_eager,
        "restored_layers": restored,
        "layers_still_replaced_after_restore": attention_cores_replaced(model),
    }

    # ---- num_splits 扫描（参数敏感性本身就是结果；**报全部**，含默认值的位置）----
    sweep: list[dict[str, Any]] = []
    if not args.no_sweep:
        grid = [int(x) for x in str(args.split_grid).split(",") if x.strip()]
        if default_splits not in grid:
            grid.append(default_splits)
        for ns in sorted(set(grid)):
            replace_attention_cores(model, backend="triton", config={"num_splits": ns})
            warm_attn(
                heads=heads,
                head_dim=head_dim,
                kv_len=args.ctx + capacity,
                filled=args.ctx + capacity,
                dtype=torch.float16,
                device="cuda",
                config={"num_splits": ns},
            )
            s = StaticCacheDecodeSession(
                model, ids, ctx=args.ctx, max_new_tokens=capacity, device="cuda"
            )
            s.capture(warmup=args.warmup)
            script = _script_tokens(s, steps_needed)
            timing = _timed_round(s, script, warmup=args.warmup, timed=args.timed)
            sweep.append(
                {
                    "num_splits": ns,
                    "is_default": ns == default_splits,
                    "per_token_wall_ms": round(timing["wall_ms"], 5),
                    "per_token_event_ms": round(timing["event_ms"], 5),
                }
            )
            del s
        fastest = min(sweep, key=lambda r: r["per_token_wall_ms"])
        default_row = next(r for r in sweep if r["is_default"])
        spread = max(r["per_token_wall_ms"] for r in sweep) / min(
            r["per_token_wall_ms"] for r in sweep
        )
        result["num_splits_sweep"] = {
            "grid": sweep,
            "fastest": fastest,
            "default_row": default_row,
            "spread_x": round(spread, 3),
            "caveat": "单轮扫描，只用于定位量级与默认值的位置；结论用交错对照的那一组",
        }
        # 扫描会改动模型上的 patch 配置，恢复到默认
        replace_attention_cores(model, backend="triton")

    # ---- 闸门（D）----
    per_arm = inter["per_arm"]
    faster = per_arm["kernel"]["per_token_wall_p50_ms"] < per_arm["eager"]["per_token_wall_p50_ms"]
    wins_majority = per_arm["kernel"]["wins"] > args.rounds / 2
    k4_ok = result["correctness"]["K4_length_change_kernel"]["verdict"] == "PASS"
    k2_ok = k2.get("kl") == 0.0 and k2.get("top1_agreement") == 1.0
    result["gate_D"] = {
        "faster": bool(faster),
        "wins_majority": bool(wins_majority),
        "kernel_wins": per_arm["kernel"]["wins"],
        "rounds": args.rounds,
        "attention_rows_replaced": bool(
            result["attention_core"]["eager_attention_signature_vanished"]
            and result["attention_core"]["kernel_signature_appeared"]
        ),
        "K1_pass": result["correctness"]["K1_kernel_vs_fp64_oracle"]["verdict"] == "PASS",
        "K2_pass": bool(k2_ok),
        "K4_pass": bool(k4_ok),
        "verdict": "PASS" if (faster and wins_majority and k4_ok and k2_ok) else "FAIL",
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    print(payload)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        print("\nwrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
