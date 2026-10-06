"""bench_grouped_gemm — isolated microbenchmark: grouped GEMV vs gather + bmm.

## 这个脚本是 Stage C 的**闸门**

approved plan 里的判据是明确的：如果 grouped kernel 在**同形状**上不快于
`gather + bmm`，就停下、记录 null 结果，不要把 kernel 接进模型。

为什么必须有这一步，而不是直接跑 `EXP-p2-02`：

- 接进模型要加载 13 GB 权重、走完整 correctness 门、跑 25 次 prefill —— 一次
  10 分钟量级。而"kernel 本身有没有比 cuBLAS 快"这个问题**只需要 805 MB 的
  合成权重**（单层：gate_up 537 MB + down 268 MB），几十秒就有答案。
  （同一份权重在 fp16 下每 token 的**理想读量**是 1.61 GB —— 那是 8 个激活
  专家 × 16 层，与这里的单层存储量不是一回事。）
- 更重要的：模型级数字混了 router、attention、launch 开销。如果 grouped 变快了
  而 E2E 没变，你需要知道是哪一层的问题；这个脚本给出的是**单层专家路径**的
  干净对照。

## 两条臂的形状完全一致

    gate_up  (E, 2*ffn, hidden)   fp16
    down     (E, hidden, ffn)     fp16
    x        (hidden,)            fp16   —— decode 时 T=1
    ids      (top_k,)             int64
    top_w    (top_k,)             fp16

    臂 A  gather : gate_up[ids] -> bmm -> silu*up -> down[ids] -> bmm -> 加权求和
    臂 B  grouped: kernel 内算指针，无副本

两者的 DRAM 下界**不一样**，这正是要比的东西：

    理想（每个激活专家读一次）  = ideal_bytes
    臂 A 的驱动流量              = 3 x ideal_bytes   （读 + 写副本 + bmm 再读）

所以结果必须报**两个达成率**，而不是一个"加速比" —— 否则"臂 B 比臂 A 快 1.4x"
和"臂 A 本来就在浪费"这两件事会混在一起。

## 实测提醒（会影响怎么读这个脚本的输出）

`ctx=1024` 那一跑的归因显示：gather + bmm 实测 2.1807 ms/token，而"读一遍"的
DRAM 下界是 1.033 ms/token。两个 gather 的 kernel 各自只花 104.6 µs/层，
**低于** 129.1 µs/层的 DRAM 下界 —— 说明副本的写与 bmm 的重读有相当一部分
命中了 L2（5090 约 96 MB，而 gate_up 副本 67 MB）。

含义：可省的**不是**"2 x ideal = 2.07 ms"，而是"实测 2.18 ms 减掉不读副本时
必须付的 DRAM 代价"。所以这个脚本的闸门判据看的是**实测的两臂时间差**，
不是流量差。

    python -m tools.bench_grouped_gemm --json results/kernel_microbench.json
    python -m tools.bench_grouped_gemm --sweep          # 参数敏感性（autotune 轴）
    python -m tools.bench_grouped_gemm --print-ncu      # 打印 ncu 命令
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

# OLMoE-1B-7B-0924 的已核实形状（docs/MODELS.md §3.1）
OLMOE = {"num_experts": 64, "hidden": 2048, "ffn": 1024, "top_k": 8, "layers": 16}

# ncu 指标：kernel 级证据（见 lemoq.kernels.autotune 的清单）
NCU_METRICS = [
    "dram__bytes.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "lts__t_sector_hit_rate.pct",
]

_SWEEP = {
    "gate_up_act": [
        {"BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
        {"BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
        {"BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 8},
        {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 8},
        {"BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 2, "num_warps": 4},
        {"BLOCK_N": 32, "BLOCK_K": 128, "SPLIT_K": 4, "num_warps": 4},
    ],
    "down_sum": [
        {"BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
        {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
        {"BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 8},
        {"BLOCK_N": 256, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 8},
        {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 2, "num_warps": 4},
        {"BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 4, "num_warps": 4},
    ],
}


def _find_tool(name: str) -> str | None:
    """同 `tools.probe_machine._find_tool`：ncu 常常不在 PATH 上。"""
    import glob
    import shutil

    found = shutil.which(name)
    if found:
        return found
    for pattern in (
        "/usr/local/cuda/bin/%s" % name,
        "/opt/nvidia/nsight-compute/*/%s" % name,
        "/usr/local/bin/%s" % name,
    ):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def _summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"p50": None, "p99": None, "min": None, "n": 0, "samples": []}
    ordered = sorted(samples)
    return {
        "p50": round(statistics.median(ordered), 5),
        "p99": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 5),
        "min": round(ordered[0], 5),
        "n": len(ordered),
        "samples": [round(s, 5) for s in samples],
    }


def _measured_bw_gbps() -> tuple[float | None, str]:
    """实测带宽**必须**来自 machine_probe.json —— 不得用规格书数字。"""
    for path in (
        os.environ.get("LEMOQ_PROBE", ""),
        "results/machine_probe.json",
    ):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                probe = json.load(fh)
            bw = (probe.get("bandwidth") or {}).get("read_only_gbps")
            if bw:
                return float(bw), path
        except Exception:  # noqa: BLE001
            continue
    return None, "results/machine_probe.json (not found)"


def _build(shape: dict[str, int], device: str, dtype_name: str, seed: int):
    import torch

    from lemoq.kernels.moe_static import FusedExperts

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    e, hidden, ffn = shape["num_experts"], shape["hidden"], shape["ffn"]

    def randn(*size):
        # 小幅度权重：与真实初始化同量级，避免 fp16 溢出把差异放大
        return (torch.randn(*size, generator=gen) * 0.02).to(dtype).to(device)

    fused = FusedExperts(
        gate_up=randn(e, 2 * ffn, hidden).contiguous(),
        down=randn(e, hidden, ffn).contiguous(),
    )
    router = randn(e, hidden)
    x = (torch.randn(1, 1, hidden, generator=gen) * 0.5).to(dtype).to(device)
    return fused, router, x


def _time_arm(fn, *, warmup: int, iters: int, repeats: int, torch) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1000.0 / iters)
        end.record()
        torch.cuda.synchronize()
    return _summarize(wall)


def _ncu_run(target: str, output: str) -> dict[str, Any] | None:
    """用 ncu 取 kernel 级 DRAM 证据。失败要记录，不能假装成功。"""
    ncu = _find_tool("ncu")
    if not ncu:
        return {"status": "unavailable", "reason": "ncu 不在 PATH 也不在常见位置"}
    cmd = [
        ncu,
        "--kernel-name",
        "regex:%s" % target,
        "--launch-count",
        "1",
        "--metrics",
        ",".join(NCU_METRICS),
        "--csv",
        "--target-processes",
        "all",
        # ★ 必须用 sys.executable：容器里的 `python` 可能是另一个解释器（没装 torch）
        sys.executable,
        __file__,
        "--no-ncu",
        "--iters",
        "1",
        "--warmup",
        "1",
        "--repeats",
        "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "reason": "%s: %s" % (type(exc).__name__, exc)}
    if proc.returncode != 0:
        return {
            "status": "failed",
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr or "")[-800:],
        }
    parsed = {}
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) >= 3 and any(metric in parts[1] for metric in NCU_METRICS):
            parsed[parts[1]] = parts[2]
    return {"status": "measured", "metrics": parsed, "raw_tail": (proc.stdout or "")[-2000:]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name, default in OLMOE.items():
        ap.add_argument("--%s" % name.replace("_", "-"), type=int, default=default)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="扫参数空间并报每个 config 的时间")
    ap.add_argument("--ncu", action="store_true", help="用 ncu 取 kernel 级 DRAM 证据")
    ap.add_argument("--no-ncu", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--print-ncu", action="store_true", help="只打印 ncu 命令")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    import torch

    shape = {
        "num_experts": args.num_experts,
        "hidden": args.hidden,
        "ffn": args.ffn,
        "top_k": args.top_k,
        "layers": args.layers,
    }
    bw, bw_src = _measured_bw_gbps()

    if args.print_ncu:
        ncu = _find_tool("ncu") or "ncu"
        print(
            "%s --kernel-name regex:gate_up_act|down_sum --launch-count 1 \\\n"
            "  --metrics %s \\\n"
            "  --csv %s %s --no-ncu --iters 1 --warmup 1 --repeats 1"
            % (ncu, ",".join(NCU_METRICS), sys.executable, os.path.abspath(__file__))
        )
        return 0

    if not torch.cuda.is_available():
        print("需要 CUDA；这个脚本的数字不得在无 GPU 的机器上产生。", file=sys.stderr)
        return 2

    from lemoq.kernels.moe_grouped import grouped_available, launch_plan, warm
    from lemoq.kernels.moe_static import static_moe_forward

    if not grouped_available():
        print("triton 不可用，无法测 grouped 臂。", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    fused, router, x = _build(shape, args.device, args.dtype, args.seed)

    # 两条臂用**同一组** ids —— 否则测的不是同一个问题。
    # 这里直接用 router 的 top-k 算一遍（与 `_route` 同一条路径：softmax 后的
    # topk 与 logits 的 topk 在 argmax 集合上一致，而两臂都只用 ids 做专家选择）。
    with torch.no_grad():
        logits = x.reshape(1, -1).float() @ router.t().float()
        ids = torch.topk(logits, shape["top_k"], dim=-1).indices.reshape(-1).long()

    plan = launch_plan(fused, top_k=shape["top_k"])
    result: dict[str, Any] = {
        "shape": shape,
        "dtype": args.dtype,
        "measured_bw_gbps": bw,
        "measured_bw_source": bw_src,
        "plan": plan,
        "timing": {"warmup": args.warmup, "iters": args.iters, "repeats": args.repeats},
        "ids": [int(i) for i in ids.tolist()],
        "caveats": [
            "合成权重（幅度 0.02）—— 只用于时间对照，不是精度证据",
            "两条臂测的都是**完整 MoE 前向**（含 router，两臂相同），所以差值就是专家路径",
            "单层。模型级结论要看 EXP-p2-02",
            "实测带宽取自 machine_probe.json，不得用规格书数字",
            "本机若同时有别的进程占 GPU，数字不可用（跑前查 nvidia-smi）",
        ],
    }

    warm(fused, hidden=shape["hidden"], top_k=shape["top_k"])

    def arm(impl, moe_config=None):
        def run():
            with torch.no_grad():
                static_moe_forward(
                    x,
                    router,
                    fused,
                    top_k=shape["top_k"],
                    impl=impl,
                    moe_config=moe_config,
                )

        return run

    result["gather"] = {
        "wall_ms": _time_arm(
            arm("gather"), warmup=args.warmup, iters=args.iters, repeats=args.repeats, torch=torch
        )
    }
    result["grouped"] = {
        "wall_ms": _time_arm(
            arm("grouped"), warmup=args.warmup, iters=args.iters, repeats=args.repeats, torch=torch
        )
    }

    # ★ 时间之前先确认两条臂算的是**同一个东西**。
    #   一个快的错 kernel 不是结果 —— 没有这一步，1.4x 的"加速"可能只是它少算了。
    with torch.no_grad():
        y_gather = static_moe_forward(x, router, fused, top_k=shape["top_k"], impl="gather")[
            0
        ].float()
        y_grouped = static_moe_forward(x, router, fused, top_k=shape["top_k"], impl="grouped")[
            0
        ].float()
        # fp64 oracle：K1 门的参照只能是它，不是另一条 fp16 路径
        from lemoq.kernels.moe_grouped import static_moe_forward_grouped_reference

        y_oracle = static_moe_forward_grouped_reference(
            x.float(),
            router.float(),
            fused.to(torch.float64),
            top_k=shape["top_k"],
        )[0]
    err_g = float((y_gather - y_oracle).abs().max())
    err_q = float((y_grouped - y_oracle).abs().max())
    span = float(y_oracle.abs().max()) or 1.0
    from lemoq.kernels.moe_grouped import PARITY_GATE

    result["correctness"] = {
        "note": "两条臂都必须先和 fp64 oracle 对齐；K1 门用的是相对基线的误差，不是魔数",
        "max_abs_err_vs_fp64_gather": err_g,
        "max_abs_err_vs_fp64_grouped": err_q,
        "oracle_absmax": span,
        "rel_err_gather": round(err_g / span, 6),
        "rel_err_grouped": round(err_q / span, 6),
        "grouped_within_slack": bool(err_q <= err_g * PARITY_GATE["tol_slack"] + 1e-6),
        "gather_grouped_max_abs_diff": float((y_gather - y_grouped).abs().max()),
    }
    if not result["correctness"]["grouped_within_slack"]:
        result["verdict_preliminary"] = (
            "grouped 相对 fp64 的误差超过基线的 %.1fx —— 时间数字不可用，先修正确性"
            % PARITY_GATE["tol_slack"]
        )

    # 字节账（纯算术，与 benchmarks/attribution.py 同源，这里独立再算一遍以便自洽）
    gate_up_bytes = shape["top_k"] * 2 * shape["ffn"] * shape["hidden"] * 2
    down_bytes = shape["top_k"] * shape["hidden"] * shape["ffn"] * 2
    ideal = gate_up_bytes + down_bytes
    result["bytes"] = {
        "per_layer_ideal": ideal,
        "per_layer_driven_gather_path": 3 * ideal,
        "gate_up_bytes": gate_up_bytes,
        "down_bytes": down_bytes,
    }
    if bw:
        result["floors_ms"] = {
            "ideal": round(ideal / (bw * 1e9) * 1000.0, 5),
            "gather_path_driven": round(3 * ideal / (bw * 1e9) * 1000.0, 5),
        }
        # ★ 变量名不能叫 `arm` —— 那会遮蔽上面的 `arm()` 闭包，随后 sweep 调用
        #   `arm("grouped", cfg)` 就会报 "str object is not callable"（实测踩到）。
        for arm_name in ("gather", "grouped"):
            p50 = result[arm_name]["wall_ms"]["p50"]
            result[arm_name]["attainment"] = {
                "vs_ideal_pct": (
                    round(100.0 * result["floors_ms"]["ideal"] / p50, 1) if p50 else None
                ),
                "vs_gather_path_floor_pct": (
                    round(100.0 * result["floors_ms"]["gather_path_driven"] / p50, 1)
                    if p50
                    else None
                ),
            }
        g = result["gather"]["wall_ms"]["p50"]
        q = result["grouped"]["wall_ms"]["p50"]
        result["verdict"] = {
            "gather_p50_ms": g,
            "grouped_p50_ms": q,
            "speedup_x": round(g / q, 3) if q else None,
            "grouped_faster": bool(q < g),
            "gate": (
                "PASS：grouped 更快，可以把 kernel 接进模型（EXP-p2-02）"
                if q < g
                else "NULL：grouped 没有更快 —— 记录这个结果，不要接进模型（转 D-2）"
            ),
            "correctness_ok": result.get("correctness", {}).get("grouped_within_slack"),
        }
        # ★ 时间差必须建立在"算的是同一个东西"之上
        if not result["verdict"]["correctness_ok"]:
            result["verdict"][
                "gate"
            ] = "INVALID：correctness 未过（见 result.correctness）—— 时间对照无意义"

    if args.sweep:
        default_cfg = {
            "gate_up_act": {"BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
            "down_sum": {"BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1, "num_warps": 4},
        }
        sweep = []
        for ga in _SWEEP["gate_up_act"]:
            for ds in _SWEEP["down_sum"]:
                cfg = {"gate_up_act": dict(ga), "down_sum": dict(ds)}
                stats = _time_arm(
                    arm("grouped", cfg),
                    warmup=args.warmup,
                    iters=args.iters,
                    repeats=args.repeats,
                    torch=torch,
                )
                sweep.append({"config": cfg, "wall_ms": stats})
        sweep.sort(key=lambda row: row["wall_ms"]["p50"])
        result["sweep"] = {
            "note": (
                "参数敏感性本身就是结果（lemoq.kernels.autotune 的要求）；"
                "报全部与最好，不只报最好的那个"
            ),
            "rows": sweep,
            "best": sweep[0] if sweep else None,
            "default": next((r for r in sweep if r["config"] == default_cfg), None),
            "spread_ms": (
                round(sweep[-1]["wall_ms"]["p50"] - sweep[0]["wall_ms"]["p50"], 5)
                if len(sweep) > 1
                else None
            ),
        }

    if args.ncu and not args.no_ncu:
        result["ncu"] = {
            "gate_up_act": _ncu_run("gate_up_act", "results/ncu/grouped"),
            "metrics": NCU_METRICS,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        print("\nwrote %s" % args.json, file=sys.stderr)
    return 0 if result.get("verdict", {}).get("grouped_faster", True) else 0


if __name__ == "__main__":
    raise SystemExit(main())
