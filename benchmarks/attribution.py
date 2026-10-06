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
    # H-A：MoE 专家路径四组（gather + bmm）实测之和 ≥ 此值 → 流量假设成立
    "dram_group_ms_min": 1.6,
    # H-B：均值 < latency_floor_us 的 kernel 之和 ≥ 此值 → 地板假设成立
    "latency_bucket_ms_min": 1.0,
    # "最短执行时间"的量级：sm_120 上一个空 kernel 的实测时间约 2-4 µs
    "latency_floor_us": 4.0,
}

# kernel 名模式 → 组。宁可少匹配（报 unmatched）也不要错匹配。
KERNEL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gather", ("indexFunc", "index_select", "index_copy", "gather", "take")),
    ("bmm", ("gemm", "gemv", "cublas", "cutlass", "matmul", "bmm")),
    ("attention", ("attention", "flash", "softmax", "scaled_dot", "efficient")),
    ("elementwise", ("elementwise", "vectorized", "reduce", "cub::", "triton_")),
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

    对齐 `docs/MOE_CUDAGRAPH.md` §3.3 与 p0-04 记录的附录：

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
    """Apply the frozen Stage-A decision rule and name the bottleneck attribution.

    `measured` may be either the output of `group_kernels` or the `top_kernels`
    list itself.  The two conditions are **independent** (amounts of time from
    two different causes), and both may hold:

        H-A   MoE 四组（gather + bmm）实测之和 >= 1.6 ms   -> gather 税是真的
        H-B   均值 < 4 µs 的 kernel 之和      >= 1.0 ms   -> 地板是真的

    Reporting them as one verdict would hide the case that actually needs *both*
    fixes.  `dominant` then says which is currently larger, which is what orders
    the work; it never overrides the verdict.
    """
    th = dict(HYPO_THRESHOLDS)
    th.update(thresholds or {})
    if isinstance(measured, list):
        groups = group_kernels(measured)
    else:
        groups = measured
    pattern = groups.get("by_pattern", {})
    dram_ms = round(
        float(pattern.get("gather", {}).get("self_ms_per_token", 0.0))
        + float(pattern.get("bmm", {}).get("self_ms_per_token", 0.0)),
        5,
    )
    latency_ms = float(groups.get("latency_floor", {}).get("self_ms_per_token", 0.0))
    h_a = dram_ms >= th["dram_group_ms_min"]
    h_b = latency_ms >= th["latency_bucket_ms_min"]
    verdict = {(True, True): "H-A+H-B", (True, False): "H-A", (False, True): "H-B"}.get(
        (h_a, h_b), "neither"
    )
    if dram_ms > latency_ms:
        dominant = "H-A" if h_a else "neither"
    elif latency_ms > dram_ms:
        dominant = "H-B" if h_b else "neither"
    else:
        dominant = verdict
    actions = {
        "H-A": "Stage C：Triton grouped GEMM（去掉 gather 拷贝）",
        "H-B": "D-2：融合 router+softmax+topk 与小 kernel，减少 kernel 数（地板是真的）",
        "H-A+H-B": "Stage C 与 D-2 都要做：先 grouped GEMM 消掉确定性浪费，再按实测残差决定 D-2 的规模",
        "neither": "两条假设都被否证 —— 重写 p0-04 的 Follow-ups，按实测归因重新定方向",
    }
    return {
        "verdict": verdict,
        "dominant": dominant,
        "dram_group_ms_per_token": dram_ms,
        "latency_bucket_ms_per_token": round(latency_ms, 5),
        "thresholds": th,
        "action": actions[verdict],
        "caveats": [
            "分组是 kernel 名模式匹配，不是语义识别；unmatched_names 必须一起看",
            "H-B 的 4 µs 是空 kernel 量级，不是本卡实测的最小 kernel 时间",
            "单 kernel 的 DRAM 字节需 ncu dram__bytes.sum，本模块不提供",
        ],
    }
