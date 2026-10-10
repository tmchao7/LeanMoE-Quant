"""attribution — 把 per-token 时间拆到 **kernel 组** 上，并判定瓶颈归属。

## 为什么需要这个模块

`EXP-p0-04` 达到 5.2547 ms/token（floor_T 1.5982 ms，达成率 28.8%）。剩下的
3.66 ms 被记录拆成"gather 2.07 ms + 其它 1.59 ms"。但那两句话里只有前半句
有算术支撑；"其它 1.59 ms"是散文，而且**现有的解释（attention 扫 1056 槽位）
对不上流量账**：

    KV 读 = 1056 槽 × 131 KB = 138 MB  →  仅 0.089 ms
    实测 busy 5.2315 ms × 1559.2 GB/s = 8.16 GB 等效流量
    可核算流量 = [T] 2.49 GB + gather 税 3.22 GB = 5.71 GB → 3.66 ms
                                             差额 ≈ 2.45 GB，无处安放

同一份数据里还有第二个线索：1,073 kernels / 5.2315 ms = **平均每个 kernel
4.88 µs**。于是有两条互斥假设：

    H-A（流量）  残差是 DRAM 流量 —— gather 拷贝与小 kernel 的低效访存
                 → 修法是 grouped GEMM（kernel 内算指针，不物化副本）
    H-B（地板）  残差是 kernel 数 × 单 kernel 最短执行时间
                 （~1000 个小 kernel × 2-4 µs）
                 → 修法主要是**融合/减少 kernel 数**，grouped GEMM 只解决一半

两者的 GPU 花法完全不同，所以**先判定再动手**。本模块只做两件事：

    1. 把 MoE 专家路径的字节与下界算清楚（`expert_path_accounting`）；
    2. 把实测 kernel 表按"是否 DRAM 载荷"分组，套用**跑之前冻结**的判据
       （`resolve_hypothesis`）。

**纯 Python，不 import torch。** 这样判定逻辑可以在 Mac 上用手算值做回归 ——
单位错（µs/ms、总计/每调用）会让 GPU 预算花到错误的方向上，而那种错误
在 GPU 上很难被发现。

## 已知局限（必须随结论一起报）

- kernel 名到分组的映射是**模式匹配**，不是语义识别。匹配不中时本模块报
  `unmatched` 并列出实际 kernel 名，**不猜**。
- 单 kernel 的字节数只能对我们**已知**的那几个算子算（gather / bmm），
  通用 kernel 的 DRAM 流量需要 `ncu dram__bytes.sum`。
- `torch.profiler` 看到的是 replay 内部逐个 kernel 的 self time（实测
  `kernels_per_call` 非零即为证据），但它**不提供**每个 kernel 的字节数。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# ★ 判据阈值在测量之前冻结（plan Stage A）。
#   改这些数字必须单独成一个 commit 并写明理由 —— 否则就是事后调判据。
# ---------------------------------------------------------------------------
HYPO_THRESHOLDS = {
    # H-A：MoE 专家路径（gather + 专家 GEMM）实测之和 ≥ 此值 → 流量假设成立
    "dram_group_ms_min": 1.6,
    # H-B：均值 < latency_floor_us 的 kernel 之和 ≥ 此值 → 地板假设成立
    "latency_bucket_ms_min": 1.0,
    # "最短执行时间"的量级：sm_120 上一个空 kernel 的实测时间约 2-4 µs
    "latency_floor_us": 4.0,
    # H-C：attention 实测 ≥ 此值 → 值得为它写 kernel
    #
    # ★ 这一条是**事后补的**，必须说清楚：判据的原始版本只有 gather/GEMM 与
    #   地板两个桶，**根本没有 attention 桶**，所以它无法对最大的那一项排序。
    #   补它的理由是"漏了一个桶"，不是"门槛太高"：阈值取 1.0，与 H-B 同一条线，
    #   没有按实测的 4.48 ms 去凑。原有的两个阈值一个都没动。
    "attention_ms_min": 1.0,
    # --- ctx 差分（H-C 的判别）---
    # MoE decode 的专家路径与 ctx **无关**（每 token 只读 top-k 个专家）。
    # 所以它的时间随 ctx 增长超过此倍数 = 有 bug，不是优化机会。
    "ctx_moe_growth_max": 1.25,
    # 两个 ctx 之间总时间差小于此值 → 残差与 ctx 无关
    "ctx_flat_delta_ms_max": 0.5,
    # 若 ctx 带来的增量有 >= 此比例落在 attention 上 → 残差是 unfilled slots
    "ctx_attention_delta_share_pct_min": 60.0,
}

# kernel 名模式 → 组。宁可少匹配（报 unmatched）也不要错匹配。
#
# ★ 顺序是有意义的：第一个命中的模式获胜。这一条不是形式主义 ——
#   实测的 attention kernel 叫
#       fmha_cutlassF_f16_aligned_64x128_rf_sm80(PyTorchMemEfficientAttention...)
#   它同时含 "cutlass"。如果 bmm 排在 attention 前面，这 4.48 ms（ctx=4096 下
#   占总时间的 53%）会被算进"专家 GEMM 组"，于是判据会得出
#   "dram group >= 1.6 ms → H-A：去写 grouped GEMM"，而真正的答案是 attention。
#   这正是本模块 caveat 里说的"模式匹配不是语义识别"。
#   下面每个 needle 都对着**实测出现过的 kernel 名**，并有回归测试钉住
#   （tests/test_attribution.py 用 ctx=4096 那一跑的原始名字）。
KERNEL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # ★ 自己写的 kernel 必须**单独成桶**。把它们丢进 unmatched 的后果实测到了：
    #   EXP-p2-02 里 `_gate_up_act_kernel`(0.78) + `_down_sum_kernel`(0.34) = 1.13 ms
    #   （占 busy 的 28%）全部落进 unmatched，于是"我的 kernel 花了多少时间"
    #   这个最该看的东西在归因表里消失了。工具不能把自己加的东西藏起来。
    ("custom_kernel", ("gate_up_act", "down_sum", "act_combine", "out_reduce", "triton_")),
    # 专家权重 gather —— 刻意**不**用裸 "gather"：at::native::sbtopk::gatherTopK
    # 是路由的 top-k，不是专家拷贝，用它会把 routing 算成 expert_gather。
    ("expert_gather", ("indexfunc", "vectorized_gather", "index_select", "index_copy")),
    # attention 必须排在 expert_gemm 之前（见上）
    ("attention", ("fmha", "attention", "flash", "scaled_dot", "memefficient")),
    ("routing", ("topk", "sortkv", "bitonic", "softmax")),
    ("expert_gemm", ("gemm", "gemv", "cublas", "cutlass", "matmul", "bmm")),
    ("elementwise", ("elementwise", "reduce", "cub::")),
    ("copy_fill", ("copy", "fill", "cast", "cat", "contiguous", "memcpy")),
)


def _spec_get(spec: Any, field: str, default: int = 0) -> int:
    """Read a `ModelSpec` field whether it is a dataclass or a plain dict."""
    if isinstance(spec, dict):
        value = spec.get(field, default)
    else:
        value = getattr(spec, field, default)
    return int(value or 0)


def expert_path_accounting(
    spec: Any, *, dtype_bytes: int = 2, bw_gbps: float, layers: int | None = None
) -> dict[str, Any]:
    """MoE 专家路径的字节与 DRAM 下界，逐层与逐 token 都给出。

    对齐 `docs/reference/MOE_CUDAGRAPH.md` §3.3 与 p0-04 记录的附录：

        gate_up        (top_k, 2*ffn, hidden)
        down           (top_k, hidden, ffn)
        理想            每个激活专家的权重**只读一次**
        gather 路径    读 + 写一份副本 + bmm 再读一遍 = 理想量的 3 倍
        因此税 = 2 × 理想量

    OLMoE-1B-7B-0924 / fp16 的锚点（回归测试断言这些数字）：
        gate_up/layer   67,108,864 B
        down/layer      33,554,432 B
        理想/token      1,610,612,736 B      = [W] 的 68.3%
        税/token        3,221,225,472 B      → 2.066 ms @1559.2 GB/s
    """
    n_layers = int(layers if layers is not None else _spec_get(spec, "layers"))
    top_k = _spec_get(spec, "top_k")
    hidden = _spec_get(spec, "hidden")
    ffn = _spec_get(spec, "expert_ffn_dim")
    if min(n_layers, top_k, hidden, ffn) <= 0:
        raise ValueError("spec 缺字段：需要 layers / top_k / hidden / expert_ffn_dim 均 > 0")
    if bw_gbps <= 0:
        raise ValueError("需要实测带宽 bw_gbps > 0（不得用规格书数字）")

    gate_up = top_k * 2 * ffn * hidden * dtype_bytes
    down = top_k * hidden * ffn * dtype_bytes
    ideal = gate_up + down
    gather_read = gate_up + down  # 读原始权重
    gather_write = gate_up + down  # 写副本
    gather_reread = gate_up + down  # bmm 再读副本
    driven = gather_read + gather_write + gather_reread
    tax = driven - ideal

    def ms(nbytes: int) -> float:
        return round(nbytes / (bw_gbps * 1e9) * 1000.0, 6)

    return {
        "layers": n_layers,
        "top_k": top_k,
        "dtype_bytes": dtype_bytes,
        "per_layer": {
            "gate_up_bytes": gate_up,
            "down_bytes": down,
            "ideal_bytes": ideal,
        },
        "per_token": {
            "ideal_bytes": ideal * n_layers,
            "gather_read_bytes": gather_read * n_layers,
            "gather_write_bytes": gather_write * n_layers,
            "gather_reread_bytes": gather_reread * n_layers,
            "driven_bytes": driven * n_layers,
            "tax_bytes": tax * n_layers,
        },
        "ms": {
            "ideal": ms(ideal * n_layers),
            "driven": ms(driven * n_layers),
            "tax": ms(tax * n_layers),
            "note": "tax = 去掉 gather 后应省下的时间（下界估计）",
        },
        "four_groups": _four_groups(gate_up, down, n_layers, ms),
        "bw_gbps": bw_gbps,
    }


def _four_groups(gate_up: int, down: int, layers: int, ms) -> list[dict[str, Any]]:
    """The four kernels the gather path costs, each with its own DRAM floor.

    Sizes follow `moe_static._decode_forward`: two gathers, two `bmm`.  These are
    the only kernels in the graph whose bytes are known exactly, which makes them
    the ones the H-A test can actually be run against.
    """
    rows = [
        ("gate_up_gather", 2 * gate_up, "read + write the (K, 2F, H) copy"),
        ("gate_up_bmm", gate_up, "read the copy once"),
        ("down_gather", 2 * down, "read + write the (K, H, F) copy"),
        ("down_bmm", down, "read the copy once"),
    ]
    out = []
    for name, per_layer, why in rows:
        total = per_layer * layers
        out.append(
            {
                "name": name,
                "bytes_per_layer": per_layer,
                "bytes_per_token": total,
                "floor_ms_per_token": ms(total),
                "note": why,
            }
        )
    return out


def group_kernels(
    rows: list[dict[str, Any]], *, latency_floor_us: float | None = None
) -> dict[str, Any]:
    """Bucket a measured kernel table by name pattern and by duration.

    `rows` are `top_kernels` entries from `result.json`
    (`{name, self_ms_per_token, occurrences_per_token, mean_us_per_occurrence}`).

    Returns `{buckets, by_pattern, latency_floor, unmatched_names, sums}`.
    `self_ms_per_token` is the per-token contribution, so it sums to `busy_ms`.
    """
    floor_us = float(
        latency_floor_us if latency_floor_us is not None else HYPO_THRESHOLDS["latency_floor_us"]
    )
    by_pattern: dict[str, dict[str, Any]] = {}
    latency_ms = 0.0
    big_ms = 0.0
    latency_n = 0
    unmatched: list[str] = []
    for row in rows:
        name = str(row.get("name") or "?")
        ms = float(row.get("self_ms_per_token") or 0.0)
        occ = float(row.get("occurrences_per_token") or 0.0)
        mean_us = row.get("mean_us_per_occurrence")
        mean_us = float(mean_us) if mean_us is not None else None
        bucket = _pattern_of(name)
        if bucket is None:
            unmatched.append(name)
            bucket = "unmatched"
        entry = by_pattern.setdefault(
            bucket, {"self_ms_per_token": 0.0, "n_kernels": 0, "names": []}
        )
        entry["self_ms_per_token"] = round(entry["self_ms_per_token"] + ms, 5)
        entry["n_kernels"] += 1
        if len(entry["names"]) < 5:
            entry["names"].append(name)
        if mean_us is not None and mean_us < floor_us:
            latency_ms += ms
            latency_n += 1
        else:
            big_ms += ms
        if occ <= 0:
            unmatched.append("%s (occurrences_per_token=0)" % name)
    return {
        "by_pattern": by_pattern,
        "latency_floor": {
            "threshold_us": floor_us,
            "n_kernels": latency_n,
            "self_ms_per_token": round(latency_ms, 5),
            "note": "均值低于空 kernel 量级的那些 kernel 贡献的时间",
        },
        "above_floor_ms_per_token": round(big_ms, 5),
        "unmatched_names": unmatched,
        "sums": {
            "pattern_total_ms_per_token": round(
                sum(v["self_ms_per_token"] for v in by_pattern.values()), 5
            )
        },
    }


def _pattern_of(name: str) -> str | None:
    low = name.lower()
    for bucket, needles in KERNEL_PATTERNS:
        for needle in needles:
            if needle.lower() in low:
                return bucket
    return None


def resolve_hypothesis(
    measured: dict[str, Any], *, thresholds: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Apply the frozen decision rule and name which bucket deserves the next GPU hour.

    `measured` may be either the output of `group_kernels` or the `top_kernels`
    list itself.  The conditions are **independent** (time from three different
    causes), and any combination may hold:

        H-A   gather + expert GEMM 之和 >= 1.6 ms   -> gather 拷贝是真的浪费
        H-B   均值 < 4 µs 的 kernel 之和 >= 1.0 ms  -> 地板是真的（减 kernel 数）
        H-C   attention 之和          >= 1.0 ms     -> 槽位扫描是真的（写 attention kernel）

    Reporting a single verdict would hide the case that needs more than one fix.
    `dominant` names the largest **actionable** term, which is what orders the
    work.  Note the size of the H-A opportunity is the gather alone, not
    `dram_group`: a grouped GEMM removes the copies, not the expert GEMMs — the
    trigger threshold is on the group (frozen before measuring) but the ranking
    must use what the fix actually removes.
    """
    th = dict(HYPO_THRESHOLDS)
    th.update(thresholds or {})
    if isinstance(measured, list):
        groups = group_kernels(measured)
    else:
        groups = measured
    pattern = groups.get("by_pattern", {})

    def bucket_ms(name: str) -> float:
        return round(float(pattern.get(name, {}).get("self_ms_per_token", 0.0)), 5)

    gather_ms = bucket_ms("expert_gather")
    gemm_ms = bucket_ms("expert_gemm")
    attention_ms = bucket_ms("attention")
    routing_ms = bucket_ms("routing")
    custom_ms = bucket_ms("custom_kernel")
    other_ms = round(
        sum(
            float(v.get("self_ms_per_token", 0.0))
            for k, v in pattern.items()
            if k not in ("expert_gather", "expert_gemm", "attention", "custom_kernel")
        ),
        5,
    )
    dram_ms = round(gather_ms + gemm_ms, 5)
    latency_ms = float(groups.get("latency_floor", {}).get("self_ms_per_token", 0.0))

    flags = {
        "H-A": dram_ms >= th["dram_group_ms_min"],
        "H-B": latency_ms >= th["latency_bucket_ms_min"],
        "H-C": attention_ms >= th["attention_ms_min"],
    }
    triggered = [key for key in ("H-A", "H-B", "H-C") if flags[key]]
    verdict = "+".join(triggered) if triggered else "neither"
    # 排序用的是"这个修法真正能省掉的部分"，不是触发判据用的总量
    opportunity = {"H-A": gather_ms, "H-B": round(latency_ms, 5), "H-C": attention_ms}
    dominant = max(triggered, key=lambda k: opportunity[k]) if triggered else "neither"
    actions = {
        "H-A": "Stage C：Triton grouped GEMM —— kernel 内由 expert id 算指针，去掉 gather 拷贝",
        "H-B": "D-2a：融合 router+softmax+topk 与小 kernel，减少 kernel 数",
        "H-C": (
            "D-2b：写 attention kernel，把循环上界改成**设备端读出**的已填充长度 —— "
            "launch 形状保持静态（图安全），但不再扫未填充槽位"
        ),
        "neither": "三条假设都被否证 —— 按实测归因重新定方向，不要硬凑",
    }
    plan_order = [key for key in ("H-C", "H-A", "H-B") if key in triggered]
    return {
        "verdict": verdict,
        "dominant": dominant,
        "dominant_opportunity_ms": opportunity.get(dominant),
        "opportunity_ms": dict(opportunity, dram_group=dram_ms),
        "attention_ms_per_token": attention_ms,
        "expert_gather_ms_per_token": gather_ms,
        "expert_gemm_ms_per_token": gemm_ms,
        "routing_ms_per_token": routing_ms,
        "custom_kernel_ms_per_token": custom_ms,
        "other_ms_per_token": other_ms,
        "unmatched_ms_per_token": bucket_ms("unmatched"),
        "unmatched_names": groups.get("unmatched_names", []),
        "dram_group_ms_per_token": dram_ms,
        "latency_bucket_ms_per_token": round(latency_ms, 5),
        "thresholds": th,
        "action": actions[dominant] if dominant != "neither" else actions["neither"],
        "action_all": [actions[key] for key in ("H-A", "H-B", "H-C") if key in triggered],
        "suggested_order": plan_order or ["neither: 先重新归因"],
        "caveats": [
            "分组是 kernel 名模式匹配，不是语义识别；unmatched_names 必须一起看",
            "H-B 的 4 µs 是空 kernel 量级，不是本卡实测的最小 kernel 时间",
            "单 kernel 的 DRAM 字节需 ncu dram__bytes.sum，本模块不提供",
            "attention 是否随 ctx 增长要用 ctx_differential 复核 —— "
            "它是'扫未填充槽位'与'本来就该算这么多'的区别",
        ],
    }


def _by_name(rows: list[dict[str, Any]]) -> dict[str, float]:
    """name -> self_ms_per_token, summing duplicate names."""
    out: dict[str, float] = {}
    for row in rows:
        name = str(row.get("name") or "?")
        out[name] = out.get(name, 0.0) + float(row.get("self_ms_per_token") or 0.0)
    return out


def ctx_differential(
    low: list[dict[str, Any]],
    high: list[dict[str, Any]],
    *,
    ctx_low: int,
    ctx_high: int,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diff two CUDA Graph runs at different ctx to place the ctx-driven cost.

    ★ Why this is the decisive measurement for the residual:

    `StaticCache` attends all `max_cache_len` slots every step, so the attention
    sweep grows with ctx (1056 -> 4128 slots at ctx=1024/4096).  The MoE expert
    path, by (the project's) construction, reads only `top_k` experts per token
    and is therefore **ctx-independent**.  So:

        residual grows with ctx   -> it is the attention over unfilled slots
        residual stays flat       -> it is the kernel-count floor / DRAM traffic

    which is exactly the fork that decides the next GPU hour.  The MoE group
    growing is not an optimisation opportunity but a **regression**: it means the
    fixed-shape decode path started depending on ctx.

    Returns per-pattern ms at both ctx, the delta, and a verdict in
    `attention_scales` / `flat` / `moe_regression` / `mixed`.
    """
    th = dict(HYPO_THRESHOLDS)
    th.update(thresholds or {})
    if ctx_high <= ctx_low:
        raise ValueError("ctx_high 必须大于 ctx_low")
    low_by = _by_name(low)
    high_by = _by_name(high)
    per_pattern: dict[str, dict[str, float]] = {}
    for name in set(low_by) | set(high_by):
        pattern = _pattern_of(name) or "unmatched"
        entry = per_pattern.setdefault(pattern, {"ctx_low_ms": 0.0, "ctx_high_ms": 0.0})
        entry["ctx_low_ms"] = round(entry["ctx_low_ms"] + low_by.get(name, 0.0), 5)
        entry["ctx_high_ms"] = round(entry["ctx_high_ms"] + high_by.get(name, 0.0), 5)
    for entry in per_pattern.values():
        entry["delta_ms"] = round(entry["ctx_high_ms"] - entry["ctx_low_ms"], 5)
        entry["growth_x"] = (
            round(entry["ctx_high_ms"] / entry["ctx_low_ms"], 3) if entry["ctx_low_ms"] else None
        )
    total_delta = round(sum(e["delta_ms"] for e in per_pattern.values()), 5)
    moe_growth = None
    if per_pattern.get("expert_gather") and per_pattern.get("expert_gemm"):
        moe_low = (
            per_pattern["expert_gather"]["ctx_low_ms"] + per_pattern["expert_gemm"]["ctx_low_ms"]
        )
        moe_high = (
            per_pattern["expert_gather"]["ctx_high_ms"] + per_pattern["expert_gemm"]["ctx_high_ms"]
        )
        moe_growth = round(moe_high / moe_low, 3) if moe_low else None
    attn_delta = per_pattern.get("attention", {}).get("delta_ms", 0.0)
    attn_share = round(100.0 * attn_delta / total_delta, 1) if total_delta else None

    if moe_growth is not None and moe_growth > th["ctx_moe_growth_max"]:
        verdict = "moe_regression"
        finding = (
            "MoE 专家组随 ctx 增长了 %.2fx —— 它与 ctx 无关，所以这是 bug（或图里混进了"
            " ctx 相关的访存），不是优化机会。先查这个。" % moe_growth
        )
    elif total_delta < th["ctx_flat_delta_ms_max"]:
        verdict = "flat"
        finding = (
            "残差与 ctx 无关（增量 %.4f ms < %.2f）—— 所以它不是 attention 扫槽位。"
            "按 H-A/H-B 继续：grouped GEMM 与减 kernel 数。"
            % (total_delta, th["ctx_flat_delta_ms_max"])
        )
    elif attn_share is not None and attn_share >= th["ctx_attention_delta_share_pct_min"]:
        verdict = "attention_scales"
        finding = (
            "ctx 增量有 %.1f%% 落在 attention 上（%.4f ms，%d -> %d 槽位）—— 这与 "
            "StaticCache 每步 attend 全部槽位一致：扫的**主要是未填充的**那些。"
            % (attn_share, attn_delta, ctx_low, ctx_high)
        )
    else:
        verdict = "mixed"
        finding = (
            "ctx 增量 %.4f ms 分散在多个组（attention 占 %s%%）—— 不足以指向单一原因，"
            "不要据此下结论。" % (total_delta, attn_share)
        )
    return {
        "verdict": verdict,
        "ctx_low": ctx_low,
        "ctx_high": ctx_high,
        "slot_ratio": round(ctx_high / ctx_low, 3),
        "per_pattern": per_pattern,
        "total_delta_ms": total_delta,
        "moe_growth_x": moe_growth,
        "attention_delta_share_pct": attn_share,
        "thresholds": th,
        "finding": finding,
        "unmatched": _unmatched_between(low_by, high_by),
        "caveats": [
            "两个 ctx 的 kernel 名集合可能不同；只在一侧出现的 name 按 0 处理并列入 unmatched",
            "ctx 变化的还有 prefill 时间与 KV 占用；本函数只看 decode 的 per-token 增量",
        ],
    }


def _unmatched_between(low_by: dict[str, float], high_by: dict[str, float]) -> dict[str, list[str]]:
    return {
        "only_at_low": sorted(n for n in low_by if n not in high_by),
        "only_at_high": sorted(n for n in high_by if n not in low_by),
    }
