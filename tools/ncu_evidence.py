#!/usr/bin/env python3
"""ncu_evidence — 四个自研 kernel 族的 **kernel 级**证据（DRAM 字节 / 吞吐 / stall）。

## 为什么需要它

`PLAN.md` §5 把 "ncu 证据" 列为 Phase 2 的交付物，而在此之前**全部**性能结论只有步级
证据（CUDA event + torch profiler 的桶级归因）。这件事在记录里是写明了的 ——
`EXP-p4-01` 的 payload 里就挂着

    "dequant_share_source": "...不是 ncu 的 dram 计数"

所以本工具把"我认为它是带宽受限的"变成三行可核对的数字：

    解析字节（本工具算）        vs  ncu 实测 `dram__bytes_read.sum`
    到峰值的比例                    `dram__throughput.avg.pct_of_peak_sustained_elapsed`
    为什么没到峰值                  long/short scoreboard stall、achieved occupancy

★ 解析字节是**手算的**（top_k 个专家 × 形状 × 位宽），不是从测量反推的；两者的比值就是
"这个 kernel 有多少读是必要的、有多少是浪费的"。4-bit 那一条尤其值得看：
`l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` 会告诉我们
packed 权重有没有被合并访问 —— 这是 4-bit 加载最容易翻车的地方。

## 用法

```bash
# 不占卡：打印将要执行的 ncu 命令行
python -m tools.ncu_evidence --print-commands

# 逐个 subject 取证据（自己调 ncu，不需要手工包一层）
python -m tools.ncu_evidence --json results/ncu_evidence/summary.json

# 只跑一个
python -m tools.ncu_evidence --only grouped_w4a16_gate_up

# 内部模式：在 ncu 下执行某个 subject（由上面的驱动方式调用）
python -m tools.ncu_evidence --subject grouped_w4a16_gate_up
```

## 纪律（与仓库其余部分一致）

- 指标清单的**契约**是 `lemoq.kernels.NCU_METRICS`（AGENTS.md §6）；本工具只额外加
  "字节数" 这一类，因为"实测读了多少字节"是本章要回答的第一个问题；
- ncu 失败要**记录失败**（`status=failed` + stderr 尾部），缺证据不许当通过；
- 一致性：`--cache-control all`（每次 replay 前清 L2）是**必须**的 —— 不清 L2 的字节数
  会把"第二次读命中了 L2"算成 DRAM 流量，那正是本项目最容易被自己骗到的地方；
- 这里测的是**访存行为**，不是正确性。正确性由 K1 门 / 引用门各自钉住，本工具不重复。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT_DIR = os.path.join("results", "ncu_evidence")

# 除 NCU_METRICS 之外要额外取的：字节数是本章的第一个论断（"读的量对不对"），
# sm__throughput 用来与 dram__throughput 对照 —— 两个都低才叫"既不带宽也不算力受限"。
EXTRA_METRICS = (
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
)

# 每个 subject 的固定形状（OLMoE-1B-7B-0924 的单层 decode 形状）。
OLMOE = {
    "num_experts": 64,
    "hidden": 2048,
    "ffn": 1024,
    "top_k": 8,
    "heads": 16,
    "head_dim": 128,
}


# --------------------------------------------------------------------------------------
# builders：每个返回 (可调用, 解析字节, 元信息)
#   ★ 解析字节必须是**手算**的必要流量，不许从任何测量反推
# --------------------------------------------------------------------------------------


def _build_grouped(args: Any, *, bits: int | None, field: str):
    """grouped MoE GEMV：fp16（`bits=None`）或 packed 8/4-bit。

    `field`：`gate_up` / `down`（per-kernel 的 ncu subject）/ `layer`（整层一次调用，
    `--isolated` 用）。它只影响**字节归属**，不影响被测的可调用对象。
    """
    import torch  # noqa: PLC0415

    from lemoq.kernels.moe_static import FusedExperts  # noqa: PLC0415

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    e, hidden, ffn, top_k = args.num_experts, args.hidden, args.ffn, args.top_k

    def randn(*size):
        # 与真实初始化同量级，避免 fp16 溢出把访存模式以外的东西变成变量
        return (torch.randn(*size, generator=gen) * 0.02).to(torch.float16).to(args.device)

    fused = FusedExperts(
        gate_up=randn(e, 2 * ffn, hidden).contiguous(),
        down=randn(e, hidden, ffn).contiguous(),
    )
    x = (torch.randn(1, 1, hidden, generator=gen) * 0.5).to(torch.float16).to(args.device)
    # ★ 固定取前 top_k 个专家：形状/字节数与真实路由一致（真实路由只换行号，不换流量）。
    #   记录里要写明这一点，否则"随机专家"会变成不可复现的第三个变量。
    ids = torch.arange(top_k, dtype=torch.int64, device=args.device)
    top_w = torch.full((top_k,), 1.0 / top_k, dtype=torch.float16, device=args.device)

    gate_up_elems = top_k * 2 * ffn * hidden
    down_elems = top_k * hidden * ffn
    meta: dict[str, Any] = {
        "impl": "grouped_moe",
        "bits": 16 if bits is None else bits,
        "group_size": None if bits is None else args.group_size,
        "expert_ids": "0..%d（固定）" % (top_k - 1),
        "weights": "gate_up (E,2F,H) + down (E,H,F) 单层",
        "scope": field,
        "launched_kernels": (
            "整层调用（gate_up_act + act_combine + down_sum [+ out_reduce]）"
            if field == "layer"
            else "整层调用；字节只算 %s 那一段" % field
        ),
    }

    # ★ 字节必须按"这条 subject 到底看哪个 kernel / 哪次调用"来算。
    #   第一版把整层字节给了 per-kernel 的 subject：于是 gate_up 的"达成率"里
    #   混进了 down 的字节，数字好看到 97% 却是假的。per-kernel 与整层必须分开。
    if bits is None:
        per_unit = 2
        scale_gu = scale_dn = 0
    else:
        per_unit = bits / 8.0
        # scale 沿 K 维分组：gate_up 每组 2F 行、hidden/g 组；down 每组 H 行、ffn/g 组
        scale_gu = int(top_k * 2 * ffn * (hidden // args.group_size)) * 2
        scale_dn = int(top_k * hidden * (ffn // args.group_size)) * 2

    bytes_by_field = {
        "gate_up": int(gate_up_elems * per_unit) + scale_gu,
        "down": int(down_elems * per_unit) + scale_dn,
        "layer": int((gate_up_elems + down_elems) * per_unit) + scale_gu + scale_dn,
    }
    if field not in bytes_by_field:
        raise ValueError("field 必须是 gate_up / down / layer，得到 %r" % (field,))

    if bits is None:
        from lemoq.kernels.moe_grouped import grouped_moe_forward  # noqa: PLC0415

        def fn():
            return grouped_moe_forward(x, fused, ids, top_w)

        meta["bytes_note"] = "top_k × 对应权重元素数 × 2 B（fp16）"
    else:
        from lemoq.kernels.moe_grouped_quant import (  # noqa: PLC0415
            grouped_moe_forward_quant,
            quantize_fused_experts,
        )

        q = quantize_fused_experts(fused, bits=bits, group_size=args.group_size)
        del fused  # 别让 fp16 与 packed 同时占显存（在计费机器上这是实打实的钱）
        meta["bytes_note"] = "top_k 个专家的 packed codes（%d-bit）+ per-group fp16 scale" % bits

        def fn():
            return grouped_moe_forward_quant(x, q, ids, top_w)

    return fn, {"read": int(bytes_by_field[field]), "write": 0}, meta


def _build_dequant(args: Any):
    """dense 融合反量化 GEMM（`_dequant_gemm_kernel`）。"""
    import torch  # noqa: PLC0415

    from lemoq.kernels._dequant_triton import dequant_gemm_fused  # noqa: PLC0415

    n, k, g = args.dense_n, args.dense_k, args.group_size
    if k % g != 0:
        raise SystemExit("--dense-k 必须被 --group-size 整除，得到 %d / %d" % (k, g))
    n_groups = k // g
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (torch.randn(1, k, generator=gen) * 0.5).to(torch.float16).to(args.device)
    # 只为访存行为服务：codes 用满量程随机，避免出现全 0 这种"退化但很快"的输入
    qweight = torch.randint(-127, 127, (n, k), generator=gen, dtype=torch.int8, device="cpu").to(
        args.device
    )
    scale = (torch.rand(n, n_groups, generator=gen) * 0.01).to(torch.float16).to(args.device)

    def fn():
        return dequant_gemm_fused(
            x, qweight, scale, None, group_size=g, bits=8, out_dtype=torch.float16
        )

    read = n * k + n * n_groups * 2 + k * 2
    write = n * 2
    meta = {
        "impl": "dequant_gemm_fused",
        "bits": 8,
        "group_size": g,
        "shape": "(M=1, K=%d) x (N=%d, K=%d)" % (k, n, k),
        "bytes_note": "N·K int8 + N·(K/g)·2 B scale + M·K·2 B 激活；写 N·2 B",
    }
    return fn, {"read": int(read), "write": int(write)}, meta


def _build_attn(args: Any):
    """split-KV decode attention（`_attn_split_kernel`）。"""
    import torch  # noqa: PLC0415

    from lemoq.kernels.attn_decode import attn_decode_forward  # noqa: PLC0415

    b, h, d, length = 1, args.heads, args.head_dim, args.kv_len
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    def randn(*size):
        return (torch.randn(*size, generator=gen) * 0.1).to(torch.float16).to(args.device)

    q = randn(b, h, 1, d)
    k = randn(b, h, length, d)
    v = randn(b, h, length, d)
    # ★ 契约要求 int32（`attn_decode._filled_tensor` 会显式拒绝 int64）
    filled = torch.tensor(length, dtype=torch.int32, device=args.device)

    def fn():
        return attn_decode_forward(q, k, v, filled)

    read = 2 * h * length * d * 2 + h * d * 2
    write = h * d * 2
    meta = {
        "impl": "attn_decode (split-KV)",
        "ctx": length,
        "shape": "q (1,%d,1,%d), k/v (1,%d,%d,%d)" % (h, d, h, length, d),
        "bytes_note": "读整段 k+v（split kernel 各 split 读自己那一段）+ q；写 partials 与输出",
        "note": "partials 的写（num_splits·H·D·3）与 combine kernel 不在本条内",
    }
    return fn, {"read": int(read), "write": int(write)}, meta


def _build_rmsnorm(args: Any):
    """自研融合 RMSNorm（`_rms_norm_kernel`）。"""
    import torch  # noqa: PLC0415

    from lemoq.kernels.rmsnorm import rms_norm_fused  # noqa: PLC0415

    hidden = args.hidden
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    x = (torch.randn(1, hidden, generator=gen) * 0.5).to(torch.float16).to(args.device)
    w = torch.ones(hidden, dtype=torch.float16, device=args.device)

    def fn():
        return rms_norm_fused(x, w, 1e-5, out_dtype=torch.float16)

    read = hidden * 2 + hidden * 2
    write = hidden * 2
    meta = {
        "impl": "rms_norm_fused",
        "shape": "(M=1, H=%d)" % hidden,
        "bytes_note": "读 M·H·2 B（x）+ H·2 B（weight）；写 M·H·2 B",
        "note": (
            "单次调用只有 8 KB 流量 ⇒ 预期它是**延迟受限**的。它的收益来自"
            "把 7 个 kernel 变成 1 个（EXP-p4c-02 §5.6：994→539 kernel/步），不是来自字节 —— "
            "本表就是这句话的证据。"
        ),
    }
    return fn, {"read": int(read), "write": int(write)}, meta


SUBJECTS: dict[str, dict[str, Any]] = {
    # --- ncu 臂：per-kernel（`--kernel-name` 精确匹配一个 kernel，字节只算它那一段） ---
    "grouped_fp16_gate_up": {
        "kernel": "_gate_up_act_kernel",
        "builder": lambda a: _build_grouped(a, bits=None, field="gate_up"),
        "note": "grouped MoE GEMV 第一段：gate/up 投影（top_k 个专家）",
        "in_ncu": True,
        "in_iso": False,
    },
    "grouped_w8a16_gate_up": {
        "kernel": "_gate_up_act_kernel_q",
        "builder": lambda a: _build_grouped(a, bits=8, field="gate_up"),
        "note": "同上，权重 8-bit packed + per-group scale 在 kernel 内解包",
        "in_ncu": True,
        "in_iso": False,
    },
    "grouped_w4a16_gate_up": {
        "kernel": "_gate_up_act_kernel_q",
        "builder": lambda a: _build_grouped(a, bits=4, field="gate_up"),
        "note": "同上，4-bit —— 看 sector/request 是否退化（nibble 展开最容易不合并）",
        "in_ncu": True,
        "in_iso": False,
    },
    "grouped_fp16_down": {
        "kernel": "_down_sum_kernel",
        "builder": lambda a: _build_grouped(a, bits=None, field="down"),
        "note": "grouped MoE GEMV 第二段：down 投影 + top_k 加权求和",
        "in_ncu": True,
        "in_iso": False,
    },
    # --- isolated 臂：per-call（整层调用，字节就是整层；不需要计数器） ---
    "grouped_fp16_layer": {
        "kernel": "-（整层调用）",
        "builder": lambda a: _build_grouped(a, bits=None, field="layer"),
        "note": "grouped MoE **整层**一次调用（fp16）：4 个 kernel 的合计字节与端到端时间",
        "in_ncu": False,
        "in_iso": True,
    },
    "grouped_w8a16_layer": {
        "kernel": "-（整层调用）",
        "builder": lambda a: _build_grouped(a, bits=8, field="layer"),
        "note": "同上，8-bit packed 权重（不物化 fp16）",
        "in_ncu": False,
        "in_iso": True,
    },
    "grouped_w4a16_layer": {
        "kernel": "-（整层调用）",
        "builder": lambda a: _build_grouped(a, bits=4, field="layer"),
        "note": "同上，4-bit packed 权重",
        "in_ncu": False,
        "in_iso": True,
    },
    "dequant_lm_head_w8a16": {
        "kernel": "_dequant_gemm_kernel",
        "builder": _build_dequant,
        "note": "dense 融合反量化 GEMM，形状取 lm_head（单次调用字节量最大的 dense 层）",
        "extra": {"dense_n": 50304, "dense_k": 2048},
        "in_ncu": True,
        "in_iso": True,
    },
    "attn_split_kv_1024": {
        "kernel": "_attn_split_kernel",
        "builder": _build_attn,
        "note": "split-KV decode attention，ctx=1024",
        "in_ncu": True,
        "in_iso": True,
    },
    "rmsnorm_h2048": {
        "kernel": "_rms_norm_kernel",
        "builder": _build_rmsnorm,
        "note": "自研融合 RMSNorm，H=2048（每层 4 个 norm 里的 input/post_attention）",
        "in_ncu": True,
        "in_iso": True,
    },
}

NCU_SUBJECTS: tuple[str, ...] = tuple(s for s, spec in SUBJECTS.items() if spec["in_ncu"])
ISO_SUBJECTS: tuple[str, ...] = tuple(s for s, spec in SUBJECTS.items() if spec["in_iso"])

# 默认跳过 2 次 launch 再取 1 次：第 1 次是 Triton JIT 编译 + cubin 加载，
# 第 2 次是"首次真跑"（此时 L2 状态与后续不同）。第 3 次才是稳态。
DEFAULT_SKIP = 2
DEFAULT_ITERS = 6


def _declared_metrics() -> list[str]:
    """契约来自 `lemoq.kernels.NCU_METRICS`（AGENTS.md §6）。"""
    from lemoq.kernels.autotune import NCU_METRICS  # noqa: PLC0415

    names: list[str] = []
    for spec in NCU_METRICS.values():
        # 这个 dict 里混着非指标项（"how_to_run" 是 list、"caveat" 是 str）——
        # 只取"指标名 → 说明"的那几组，否则说明文字会被按字符拆成"指标名"
        if isinstance(spec, dict):
            names.extend(spec)
    return names


def _metric_list() -> list[str]:
    seen: list[str] = []
    for name in _declared_metrics() + list(EXTRA_METRICS):
        if name not in seen:
            seen.append(name)
    return seen


def _ncu_command(subject: str, ncu: str, metrics: list[str], *, python: str) -> list[str]:
    spec = SUBJECTS[subject]
    return [
        ncu,
        "--kernel-name",
        "regex:^%s$" % spec["kernel"],
        "--launch-skip",
        str(spec.get("skip", DEFAULT_SKIP)),
        "--launch-count",
        "1",
        "--metrics",
        ",".join(metrics),
        # ★ 每次 replay 前清 L2：不清的话，第二遍读会命中 L2，DRAM 字节数会低到没有意义
        "--cache-control",
        "all",
        # 不锁时钟：否则 duration 是基频下的值，与 CUDA event 的交叉验证对不上
        "--clock-control",
        "none",
        "--csv",
        "--target-processes",
        "all",
        python,
        "-m",
        "tools.ncu_evidence",
        "--subject",
        subject,
        "--iters",
        str(spec.get("iters", DEFAULT_ITERS)),
    ]


# --------------------------------------------------------------------------------------
# 内部模式：在 ncu 之下执行一个 subject
# --------------------------------------------------------------------------------------


def _run_subject_inline(subject: str, args: Any) -> int:
    spec = SUBJECTS[subject]
    for key, value in (spec.get("extra") or {}).items():
        setattr(args, key, value)
    fn, byte_info, meta = spec["builder"](args)
    for _ in range(max(0, args.warmup)):
        fn()
    import torch  # noqa: PLC0415

    torch.cuda.synchronize()
    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()
    payload = {
        "subject": subject,
        "kernel": spec["kernel"],
        "note": spec["note"],
        "analytical_bytes": byte_info,
        "meta": meta,
        "device": torch.cuda.get_device_name(0),
        "iters": args.iters,
        "warmup": args.warmup,
    }
    print("NCU_EVIDENCE_META=" + json.dumps(payload, ensure_ascii=False))
    return 0


# --------------------------------------------------------------------------------------
# 驱动模式：调 ncu，解析 CSV，落盘
# --------------------------------------------------------------------------------------


def _parse_csv(text: str) -> dict[str, dict[str, str]]:
    """ncu 的 `--csv` 输出按 kernel 名归组：{kernel: {metric: value}}。"""
    out: dict[str, dict[str, str]] = {}
    reader = csv.reader(io.StringIO(text))
    idx: dict[str, int] = {}
    for row in reader:
        if not row:
            continue
        if not idx:
            if "Metric Name" in row and "Metric Value" in row:
                idx = {name: i for i, name in enumerate(row)}
            continue
        need = ("Metric Name", "Metric Value")
        if any(k not in idx or len(row) <= idx[k] for k in need):
            continue
        name_i, val_i = idx["Metric Name"], idx["Metric Value"]
        kname = (
            row[idx["Kernel Name"]]
            if "Kernel Name" in idx and len(row) > idx["Kernel Name"]
            else "<unknown>"
        )
        entry = out.setdefault(kname, {})
        entry[row[name_i]] = row[val_i]
        if "Metric Unit" in idx and len(row) > idx["Metric Unit"]:
            unit = row[idx["Metric Unit"]]
            if unit:
                entry[row[name_i] + " [unit]"] = unit
    return out


_SCALE = {
    "nsecond": 1e-9,
    "usecond": 1e-6,
    "msecond": 1e-3,
    "second": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "Tbyte": 1e12,
    "%": 1.0,
    "": 1.0,
}


def _as_float(value: str, unit: str) -> float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number * _SCALE.get(unit, 1.0)


def _bytes_to_gb(value: Any, unit: str) -> float | None:
    scaled = _as_float(value, unit)
    return None if scaled is None else scaled / 1e9


def _summarize(kernels: dict[str, dict[str, str]], byte_info: dict[str, int]) -> dict[str, Any]:
    """把 ncu 的原始字符串整理成可引用的数字 + 与解析字节的比值。"""
    rows: dict[str, Any] = {}
    for kname, metrics in kernels.items():
        unit = lambda key: metrics.get(key + " [unit]", "")  # noqa: E731
        read_b = _bytes_to_gb(metrics.get("dram__bytes_read.sum", ""), unit("dram__bytes_read.sum"))
        write_b = _bytes_to_gb(
            metrics.get("dram__bytes_write.sum", ""), unit("dram__bytes_write.sum")
        )
        analytic = byte_info.get("read", 0) / 1e9 if byte_info.get("read") else None
        rows[kname] = {
            "duration_ms": (
                None
                if _as_float(
                    metrics.get("gpu__time_duration.sum", ""), unit("gpu__time_duration.sum")
                )
                is None
                else _as_float(
                    metrics.get("gpu__time_duration.sum", ""), unit("gpu__time_duration.sum")
                )
                * 1e3
            ),
            "dram_read_gb": read_b,
            "dram_write_gb": write_b,
            "analytical_read_gb": analytic,
            "read_vs_analytical": (None if not read_b or not analytic else read_b / analytic),
            "dram_throughput_pct": _as_float(
                metrics.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", ""),
                unit("dram__throughput.avg.pct_of_peak_sustained_elapsed"),
            ),
            "sm_throughput_pct": _as_float(
                metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", ""),
                unit("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            ),
            "achieved_occupancy_pct": _as_float(
                metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", ""),
                unit("sm__warps_active.avg.pct_of_peak_sustained_active"),
            ),
            "stall_long_scoreboard_pct": _as_float(
                metrics.get("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", ""),
                unit("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"),
            ),
            "stall_short_scoreboard_pct": _as_float(
                metrics.get("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct", ""),
                unit("smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"),
            ),
            "sectors_per_request": _as_float(
                metrics.get(
                    "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio", ""
                ),
                unit("l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"),
            ),
            "l2_hit_rate_pct": _as_float(
                metrics.get("lts__t_sector_hit_rate.pct", ""), unit("lts__t_sector_hit_rate.pct")
            ),
            "registers_per_thread": _as_float(
                metrics.get("launch__registers_per_thread", ""),
                unit("launch__registers_per_thread"),
            ),
            "raw": dict(metrics),
        }
    return rows


def _run_with_ncu(subject: str, args: Any, metrics: list[str], ncu: str) -> dict[str, Any]:
    spec = SUBJECTS[subject]
    cmd = _ncu_command(subject, ncu, metrics, python=args.python or sys.executable)
    record: dict[str, Any] = {
        "subject": subject,
        "kernel_regex": "^%s$" % spec["kernel"],
        "note": spec["note"],
        "ncu_command": " ".join(cmd),
        "metrics": metrics,
        "cache_control": "all",
        "clock_control": "none",
    }
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=args.timeout
        )
    except subprocess.TimeoutExpired:
        record.update({"status": "failed", "reason": "ncu 超时 (>%ds)" % args.timeout})
        return record
    record["ncu_returncode"] = proc.returncode
    meta = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("NCU_EVIDENCE_META="):
            try:
                meta = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                meta = None
            break
    if meta:
        record.update(
            {
                "analytical_bytes": meta["analytical_bytes"],
                "meta": meta["meta"],
                "device": meta["device"],
                "iters": meta["iters"],
            }
        )
    if proc.returncode != 0:
        record.update(
            {
                "status": "failed",
                "stderr_tail": (proc.stderr or "")[-2000:],
                "stdout_tail": (proc.stdout or "")[-800:],
            }
        )
        return record
    parsed = _parse_csv(proc.stdout or "")
    if not parsed:
        record.update(
            {
                "status": "failed",
                "reason": "ncu 成功退出但没有解析出任何 metric（kernel 名没匹配上？）",
                "stdout_tail": (proc.stdout or "")[-2000:],
            }
        )
        return record
    record.update(
        {
            "status": "measured",
            "kernels": _summarize(parsed, (meta or {}).get("analytical_bytes", {})),
        }
    )
    return record


def _print_commands(metrics: list[str], python: str, ncu: str) -> int:
    print("# ncu：%s" % ncu)
    print("# metrics（%d 个，来自 lemoq.kernels.NCU_METRICS + 字节数）：" % len(metrics))
    for name in metrics:
        print("#   %s" % name)
    print()
    for subject in SUBJECTS:
        print(" \\\n  ".join(_ncu_command(subject, ncu, metrics, python=python)))
        print()
    return 0


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return ("%%.%df" % digits) % value
    return str(value)


def _print_table(summary: dict[str, Any]) -> None:
    header = "%-24s %-24s %9s %9s %7s %7s %7s %7s %7s" % (
        "subject",
        "kernel",
        "read(GB)",
        "解析(GB)",
        "读/解析",
        "dram%",
        "sm%",
        "occ%",
        "lngsb%",
    )
    print(header)
    print("-" * len(header))
    for subject, record in summary["subjects"].items():
        if record.get("status") != "measured":
            print(
                "%-24s %-24s %s" % (subject, record.get("kernel_regex", "?"), record.get("status"))
            )
            continue
        for kname, row in record["kernels"].items():
            print(
                "%-24s %-24s %9s %9s %7s %7s %7s %7s %7s"
                % (
                    subject,
                    kname[:24],
                    _fmt(row.get("dram_read_gb"), 3),
                    _fmt(row.get("analytical_read_gb"), 3),
                    _fmt(row.get("read_vs_analytical"), 2),
                    _fmt(row.get("dram_throughput_pct"), 1),
                    _fmt(row.get("sm_throughput_pct"), 1),
                    _fmt(row.get("achieved_occupancy_pct"), 1),
                    _fmt(row.get("stall_long_scoreboard_pct"), 1),
                )
            )
    print()
    failed = [s for s, r in summary["subjects"].items() if r.get("status") != "measured"]
    print(
        "subjects: %d   measured: %d   failed: %d"
        % (
            len(summary["subjects"]),
            len(summary["subjects"]) - len(failed),
            len(failed),
        )
    )
    if failed:
        print("失败（按纪律记录，不重试到「好看」为止）：%s" % ", ".join(failed))


# --------------------------------------------------------------------------------------
# 隔离计时回退臂：拿不到性能计数器时，仍然能回答"这个 kernel 到没到带宽墙"
#
# ★ 为什么这条不是"退而求其次还算过得去"：判据本来就是
#       attained = 解析字节 / 测量时间        （解析字节手算，时间用 CUDA event）
#   再除以**实测可达带宽**（`tools/probe_machine` 标定过的 1561.9 GB/s，不是规格值）。
#   这与记录里步级"带宽达成率"是同一个口径，只是从一步缩到一个 kernel。
#   计数器能多给的只有"为什么"（stall/occupancy），给不了"是多少"。
#
# ★ 已知偏差（必须写进结论）：
#   - 小于约 10 µs 的 kernel，批内 wall time 含 launch 间隔 ⇒ 达成率被低估，
#     这时正确的读法是"它是延迟受限的"（自研 RMSNorm 就属于这一类）；
#   - L2：单次调用里权重会被读多遍吗？不会 —— 单次调用读一遍。但**连续多次调用**时
#     下一遍可能命中 L2 中的残页。所以本模式报 **min 与 p50**，并用
#     "DRAM 有效带宽上限 = 实测可达带宽" 保守判读，不把 L2 命中当成 DRAM 成就。
# --------------------------------------------------------------------------------------


def _measured_bw_gbps(path: str = "results/machine_probe.json") -> tuple[float | None, str]:
    """实测可达带宽（硬规则 4：不许用规格值）。"""
    full = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
    try:
        with open(full, encoding="utf-8") as fh:
            payload = json.load(fh)
        value = payload["bandwidth"]["value"]
        return float(value), full
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, "%s 读不到（%s）" % (full, type(exc).__name__)


def _foreign_processes() -> list[dict[str, Any]]:
    """记录同卡的他人进程（协议：记录，**绝不 kill**）。

    ★ 为什么隔离计时也必须记录：本模式的 p50 里含 launch 间隔，宿主 CPU 被压住时
      GPU 会在两次 launch 之间空转 ⇒ p50 被抬高。所以同时报 min（受宿主影响最小）
      与 p50，并把同租户快照写进工件 —— 让读者自己看到"这份数字是在什么环境下取的"。
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"status": "unavailable", "reason": type(exc).__name__}]
    out: list[dict[str, Any]] = []
    self_pid = str(os.getpid())
    for line in (proc.stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0]:
            continue
        pid = parts[0]
        if pid == self_pid:
            # ★ 别把自己写进"他人进程"：第一版的 foreign_processes 里就是本工具自己，
            #   读起来像"有共租户"，其实是空场。自己的 pid 单独记（见 payload.self_pid）。
            continue
        entry: dict[str, Any] = {
            "pid": pid,
            "process_name": parts[1] if len(parts) > 1 else "",
            "used_memory": parts[2] if len(parts) > 2 else "",
        }
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as fh:
                entry["cmdline"] = fh.read().decode("utf-8", "replace").replace("\0", " ").strip()
        except OSError:
            entry["cmdline"] = "<unreadable>"
        try:
            stat = subprocess.run(
                ["ps", "-o", "pcpu=", "-p", pid], capture_output=True, text=True, timeout=15
            )
            entry["cpu_pct"] = (stat.stdout or "").strip()
        except (OSError, subprocess.TimeoutExpired):
            entry["cpu_pct"] = None
        out.append(entry)
    return out


def _time_subject(fn: Any, args: Any, *, flush_mb: int) -> dict[str, Any]:
    """两种计时，缺一不可：

    1. **warm batch**：R 轮 × 每轮 I 次，事件包住整批。接近真实推理（权重留在 L2 里），
       但工作集 < L2 的 subject 会得到"带宽 > 实测可达带宽"的假相（L2 不是 DRAM）；
    2. **cold single**：每次调用前先 memset 一块 256 MB 的 buffer 把 L2 冲掉，
       再单独计时一次。**这一列才代表 DRAM**。

    ★ 事件计时本身有固定开销（一次 record 约 µs 量级），小 kernel 上不能忽略。
      所以显式测一个"空调用"的开销并记录，读者可以自己减。
    """
    import statistics  # noqa: PLC0415
    import time  # noqa: PLC0415

    import torch  # noqa: PLC0415

    for _ in range(max(0, args.warmup)):
        fn()
    torch.cuda.synchronize()

    def _pair():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        return start, end

    # 空调用 = 事件地板（不许当成 kernel 时间）
    floor_samples: list[float] = []
    for _ in range(20):
        start, end = _pair()
        start.record()
        end.record()
        torch.cuda.synchronize()
        # ★ 方向：`start.elapsed_time(end)`（self 先记、参数后记）。写反了会得到一个
        #   "恰好等于真值但带负号"的数 —— 它看起来像环境坏了，其实是调用写反了。
        floor_samples.append(start.elapsed_time(end))
    event_floor_ms = min(floor_samples)

    warm: list[float] = []
    for _ in range(args.repeats):
        start, end = _pair()
        start.record()
        for _ in range(args.iso_iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        warm.append(start.elapsed_time(end) / args.iso_iters)

    flush = None
    cold: list[float] = []
    if flush_mb > 0:
        flush = torch.empty(flush_mb * 1024 * 1024, dtype=torch.uint8, device=args.device)
        for _ in range(args.cold_iters):
            flush.zero_()  # 冲掉 L2（256 MB > 96 MB）
            start, end = _pair()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            cold.append(start.elapsed_time(end))

    def _stats(values: list[float]) -> dict[str, Any]:
        ordered = sorted(values)
        median = statistics.median(ordered)
        return {
            "n": len(values),
            "p50_ms": round(median, 6),
            "min_ms": round(ordered[0], 6),
            "max_ms": round(ordered[-1], 6),
            "mad_ms": round(statistics.median([abs(v - median) for v in ordered]), 6),
            "samples_ms": [round(v, 5) for v in values],
        }

    out: dict[str, Any] = {
        "warm_batch": _stats(warm),
        "cold_single": _stats(cold) if cold else None,
        "event_floor_ms": round(event_floor_ms, 6),
        "flush_mb": flush_mb if flush is not None else 0,
        "method": (
            "warm = 事件包住每轮 %d 次调用；cold = 每次调用前 memset %d MB 冲 L2 后单独计时"
            % (args.iso_iters, flush_mb)
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return out


def _run_isolated(args: Any) -> int:
    import torch  # noqa: PLC0415

    bw, bw_src = _measured_bw_gbps(args.machine_probe)
    if bw is None:
        print("需要实测可达带宽：%s" % bw_src, file=sys.stderr)
        return 2
    chosen = list(args.only) if args.only else list(ISO_SUBJECTS)
    unknown = [s for s in chosen if s not in SUBJECTS]
    if unknown:
        print("未知 subject: %s" % ", ".join(unknown), file=sys.stderr)
        return 2
    not_iso = [s for s in chosen if not SUBJECTS[s]["in_iso"]]
    if not_iso:
        print(
            "--isolated 只支持按「一次调用」计时的 subject；这些是 per-kernel 的，"
            "请用 ncu 臂：%s" % ", ".join(not_iso),
            file=sys.stderr,
        )
        return 2

    # ★ `ncu_attempted` 与 `ncu_blocked_by` 都必须是**本轮实测**的，不能抄。
    #   旧代码把 `ncu_attempted` 写死 True，而 `--isolated` 这一段**根本不调用
    #   ncu**（它只在 `_run_subject_inline` 里被动被测）；`ncu_blocked_by` 那串
    #   CapEff（0xa80425fb）是另一台机器上另一次手动尝试读到的 —— 换台机器（例如
    #   V100）重跑，产物里仍然写着一句从没在该机验证过的断言。违反"每个结论都要
    #   有出处"。现在：available 实测，blocked_by 只在**本轮真的试过且失败**时才有值。
    ncu_status = {
        "ncu_attempted": False,
        "ncu_available": _find_tool(args.ncu or "ncu") is not None,
        "ncu_blocked_by": None,
        "note": (
            "--isolated 这一段不用 ncu 计时（隔离计时取代了它），所以本轮既没有尝试"
            "也没有被阻塞。计数器被宿主策略阻断的那条实测结论见 docs/reference/NCU_EVIDENCE.md §2，"
            "它是**在那台机器上**量出来的，不随本产物外推。"
        ),
    }

    results: dict[str, Any] = {}
    order = list(chosen)
    canary_subject = None
    if order and order[0] not in args.canary_skip:
        order = order + [order[0]]  # 首尾同格金丝雀（与步级协议一致）
        canary_subject = order[0]

    def build_one(subject: str):
        spec = SUBJECTS[subject]
        for key, value in (spec.get("extra") or {}).items():
            setattr(args, key, value)
        fn, byte_info, meta = spec["builder"](args)
        return fn, byte_info, meta

    for index, subject in enumerate(order):
        # ★ `is_canary` 原来只看"是不是最后一个"，不看"最后一个是不是真的金丝雀副本"。
        #   当 --canary-skip 正好跳过 chosen[0] 时 order == chosen，于是**最后一个
        #   subject 被贴上 canary: 前缀**，results 里丢掉它的原名键（下游按 subject
        #   名读 payload["subjects"][X] 会 KeyError）。现在按身份判，不再按位置判。
        is_canary = canary_subject is not None and index == len(order) - 1
        label = "canary:" + subject if is_canary else subject
        fn, byte_info, meta = build_one(subject)
        timing = _time_subject(fn, args, flush_mb=args.flush_mb)
        total_bytes = byte_info.get("read", 0) + byte_info.get("write", 0)
        floor = timing["event_floor_ms"]

        def _rate(ms: float, subtract_floor: bool) -> float | None:
            effective = ms - floor if subtract_floor else ms
            if not effective or effective <= 0:
                return None
            return total_bytes / (effective * 1e-3) / 1e9

        # 主判据：cold（冲过 L2）− 事件地板。减去地板是必须的：不加的话
        # 8 KB 的 RMSNorm 会把 ~2 µs 的事件开销当成自己的时间。
        cold = timing["cold_single"] or {}
        cold_min = cold.get("min_ms")
        gbps_cold = None if cold_min is None else _rate(cold_min, True)
        gbps_cold_p50 = None if cold.get("p50_ms") is None else _rate(cold["p50_ms"], True)
        warm = timing["warm_batch"]
        gbps_warm = _rate(warm["p50_ms"], False)
        warm_ws_mb = total_bytes / 1e6
        record = {
            "subject": subject,
            "label": label,
            "kernel": SUBJECTS[subject]["kernel"],
            "note": SUBJECTS[subject]["note"],
            "analytical_bytes": byte_info,
            "working_set_mb": round(warm_ws_mb, 1),
            "l2_mb": args.l2_mb,
            "working_set_fits_l2": warm_ws_mb < args.l2_mb,
            "meta": meta,
            "timing": timing,
            "achieved_gbps_cold_best": None if gbps_cold is None else round(gbps_cold, 1),
            "achieved_gbps_cold_p50": (None if gbps_cold_p50 is None else round(gbps_cold_p50, 1)),
            "achieved_gbps_warm_p50": None if gbps_warm is None else round(gbps_warm, 1),
            "measured_achievable_gbps": bw,
            "attainment_pct_cold_best": (
                None if gbps_cold is None else round(100 * gbps_cold / bw, 1)
            ),
            "attainment_pct_cold_p50": (
                None if gbps_cold_p50 is None else round(100 * gbps_cold_p50 / bw, 1)
            ),
            "attainment_pct_warm_p50": (
                None if gbps_warm is None else round(100 * gbps_warm / bw, 1)
            ),
            "evidence_kind": (
                "kernel 级带宽达成率 = 解析字节 / (cold 计时 − 事件地板) / 实测可达带宽；"
                "**不是** ncu 计数器的 dram__bytes"
            ),
        }
        results[label] = record
        print(
            "%-26s cold %8.4f ms  %8.1f GB/s  %5.1f%% of wall   (warm %5.1f%%)"
            % (
                label,
                cold_min if cold_min is not None else float("nan"),
                gbps_cold or 0.0,
                100 * gbps_cold / bw if gbps_cold else 0.0,
                100 * gbps_warm / bw if gbps_warm else 0.0,
            ),
            flush=True,
        )

    def _cold_ms(label: str) -> float | None:
        cold = results[label]["timing"]["cold_single"]
        return None if not cold else cold["min_ms"]

    drift = None
    if len(order) > len(chosen):
        head = _cold_ms(chosen[0])
        tail = _cold_ms("canary:" + chosen[0])
        if head and tail:
            drift = 100.0 * (tail - head) / head
            print(
                "\n金丝雀（%s，cold min）：%.4f → %.4f ms，漂移 %s%% ⇒ 本轮%s"
                % (
                    chosen[0],
                    head,
                    tail,
                    _fmt(drift, 2),
                    "**作废**" if abs(drift) > 5.0 else "有效",
                )
            )

    payload = {
        "generated_by": "tools/ncu_evidence.py --isolated",
        "kind": "kernel/call-level bandwidth attainment (no performance counters)",
        "device": torch.cuda.get_device_name(0),
        "measured_achievable_gbps": bw,
        "measured_achievable_source": bw_src,
        "event_floor_ms": results[chosen[0]]["timing"]["event_floor_ms"] if chosen else None,
        "flush_mb": args.flush_mb,
        "clock_note": "不锁时钟；同租户快照见 foreign_processes（协议：记录，绝不 kill）",
        "self_pid": os.getpid(),
        "foreign_processes": _foreign_processes(),
        "ncu": ncu_status,
        "canary_drift_pct": None if drift is None else round(drift, 3),
        "subjects": results,
        "caveat": (
            "主判据是 cold（每次调用前 memset 256 MB 冲掉 L2）− 事件地板：不冲 L2 的话，"
            "工作集小于 96 MB 的 subject（8/4-bit 的权重、attention 的 k/v）会命中 L2，"
            "算出超过实测可达带宽的假达成率；不减去事件地板的话，8 KB 的 RMSNorm 会把 "
            "µs 级的事件开销算成自己的时间。cold 计时含**单次 launch** 的间隔，"
            "所以 <10 µs 的 kernel 达成率仍是下界（正确的读法是「延迟受限」）。"
            "DRAM 上限一律用实测可达带宽，不用规格值（硬规则 4）。"
        ),
    }
    out = args.json or os.path.join(args.out_dir, "isolated.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    print("isolated -> %s" % out)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", choices=sorted(SUBJECTS), default=None, help=argparse.SUPPRESS)
    ap.add_argument("--only", action="append", default=None, help="只跑指定 subject（可重复）")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--json", default=None, help="summary 的落盘路径")
    ap.add_argument("--print-commands", action="store_true", help="只打印 ncu 命令行")
    ap.add_argument("--metrics", default=None, help="覆盖指标清单（逗号分隔）")
    ap.add_argument("--ncu", default=None, help="ncu 路径（默认自动查找）")
    ap.add_argument("--python", default=None, help="子进程解释器（默认 sys.executable）")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--entries", type=int, default=OLMOE["num_experts"])
    ap.add_argument("--hidden", type=int, default=OLMOE["hidden"])
    ap.add_argument("--ffn", type=int, default=OLMOE["ffn"])
    ap.add_argument("--top-k", type=int, default=OLMOE["top_k"])
    ap.add_argument("--heads", type=int, default=OLMOE["heads"])
    ap.add_argument("--head-dim", type=int, default=OLMOE["head_dim"])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--kv-len", type=int, default=1024)
    ap.add_argument("--dense-n", type=int, default=50304)
    ap.add_argument("--dense-k", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    ap.add_argument(
        "--isolated",
        action="store_true",
        help="拿不到性能计数器时的回退臂：隔离计时 + 解析字节算 kernel 级带宽达成率",
    )
    ap.add_argument("--iso-iters", type=int, default=50, help="--isolated 每轮调用次数")
    ap.add_argument("--repeats", type=int, default=5, help="--isolated 的 warm 轮数")
    ap.add_argument(
        "--cold-iters", type=int, default=15, help="--isolated 的 cold 单次计时次数（每次前冲 L2）"
    )
    ap.add_argument("--flush-mb", type=int, default=256, help="--isolated 冲 L2 的 buffer 大小")
    ap.add_argument(
        "--l2-mb", type=int, default=96, help="本卡 L2 容量（5090 = 96 MB），用于标注是否 L2 驻留"
    )
    ap.add_argument("--machine-probe", default="results/machine_probe.json")
    ap.add_argument(
        "--canary-skip",
        action="append",
        default=[],
        help="--isolated 不把某个 subject 再跑一遍作金丝雀（可重复）",
    )
    args = ap.parse_args(argv)
    # `num_experts` 这个名字在 builder 里用得更多，两个别名指向同一个值
    args.num_experts = args.entries

    if args.isolated:
        return _run_isolated(args)

    metrics = [m for m in args.metrics.split(",") if m] if args.metrics else _metric_list()
    ncu = args.ncu or _find_tool("ncu") or "ncu"
    python = args.python or sys.executable

    if args.subject:
        return _run_subject_inline(args.subject, args)

    if args.print_commands:
        return _print_commands(metrics, python, ncu)

    if not os.path.exists(ncu):
        print(
            "找不到 ncu（%s）。本工具需要 Nsight Compute；"
            "在服务器上它通常在 /usr/local/cuda/bin/ncu。" % ncu,
            file=sys.stderr,
        )
        return 2

    chosen = list(args.only) if args.only else list(NCU_SUBJECTS)
    unknown = [s for s in chosen if s not in SUBJECTS]
    if unknown:
        print("未知 subject: %s（可选：%s）" % (", ".join(unknown), ", ".join(sorted(SUBJECTS))))
        return 2
    not_ncu = [s for s in chosen if not SUBJECTS[s]["in_ncu"]]
    if not_ncu:
        print(
            "ncu 臂只支持 per-kernel 的 subject；这些请用 --isolated：%s" % ", ".join(not_ncu),
            file=sys.stderr,
        )
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    subjects: dict[str, Any] = {}
    for subject in chosen:
        print("[ncu] %s (%s) ..." % (subject, SUBJECTS[subject]["kernel"]), flush=True)
        record = _run_with_ncu(subject, args, metrics, ncu)
        subjects[subject] = record
        status = record.get("status")
        if status == "measured":
            for kname, row in record["kernels"].items():
                print(
                    "      %s: read %s GB / 解析 %s GB（x%s）, dram %s%%, %sms"
                    % (
                        kname,
                        _fmt(row.get("dram_read_gb")),
                        _fmt(row.get("analytical_read_gb")),
                        _fmt(row.get("read_vs_analytical"), 2),
                        _fmt(row.get("dram_throughput_pct"), 1),
                        _fmt(row.get("duration_ms"), 4),
                    ),
                    flush=True,
                )
        else:
            print(
                "      %s: %s"
                % (status, record.get("reason", record.get("stderr_tail", "")[-200:])),
                flush=True,
            )
        with open(os.path.join(args.out_dir, subject + ".json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2, default=str)

    summary = {
        "generated_by": "tools/ncu_evidence.py",
        "ncu": ncu,
        "python": python,
        "metrics": metrics,
        "subjects": subjects,
        "caveat": (
            "本条是**访存行为**证据（DRAM 字节 / 吞吐 / stall），不是正确性证据；"
            "解析字节是手算的必要流量，比值 >1 的部分就是非必要的读。"
        ),
    }
    summary_path = args.json or os.path.join(args.out_dir, "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print()
    _print_table(summary)
    print("summary -> %s" % summary_path)
    return 0 if all(r.get("status") == "measured" for r in subjects.values()) else 1


def _find_tool(name: str) -> str | None:
    """与 `tools.probe_machine._find_tool` 同策略：ncu 常常不在 PATH 上。"""
    import shutil  # noqa: PLC0415

    found = shutil.which(name)
    if found:
        return found
    for path in (
        "/usr/local/cuda/bin/%s" % name,
        "/usr/local/cuda-12.8/bin/%s" % name,
        "/opt/nvidia/nsight-compute/%s" % name,
        os.path.expanduser("~/ncu/%s" % name),
    ):
        if os.path.exists(path):
            return path
    return None


if __name__ == "__main__":
    raise SystemExit(main())
