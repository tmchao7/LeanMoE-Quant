"""quant_error_audit — 纯**权重域**的量化误差审计（不加载模型、不碰 GPU）。

## 为什么需要它（以及它和 KL 的分工）

`EXP-p1-01` 的两个信号都是关于 **KL** 的，而 KL 是**间接**证据：从权重误差到
逐位置 KL 之间隔着 16 层非线性（RMSNorm、softmax、MoE 路由）。如果权重误差本身
非单调，那是实现缺陷；如果权重误差单调而 KL 非单调，那是**放大**，两级必须分开测。

所以本工具只回答一个问题：

    ||W − Ŵ|| 随 group↓ 是否**严格**单调下降？W4/W8 的比值是否等于步长的算术比？

它做的检查（每条都对应一个可能被误当成"模型敏感"的实现缺陷）：

1. **分组嵌套**：g32 的组是 g64 的组的细分，所以每个元素的 absmax 必须满足
   `s32 ≤ s64 ≤ s128`。违反它 = 分组划分与声称的不一致。
2. **逐元素误差界**：对称 RTN 下 `|err| ≤ s/2`（s 是该组步长）。违反 = 反量化
   用错了 scale，或 codes 与 scale 错位。
3. **实现误差 vs 步长算术**：W4/W8 的**界**之比必须恰好 `(2^7−1)/(2^3−1) = 127/7 = 18.143`；
   实测 max|err| 之比应落在同一量级（界是紧的，比值才可信）。
4. **离群主导度**（RTN 的已知失效模式）：每组的 `absmax/RMS`、是否被单个元素主导。
   这个量决定 `REMEDIATION_LADDER` 的 AWQ 档**有没有依据** —— 没有离群就不需要 AWQ。
5. **误差集中度**：误差平方和中"最大 1% 权重"占的份额，以及**逐输入通道**的相对误差
   （AWQ 保护的正是输入通道）。

## 两种 scale 都报

生产路径（`QuantLinear`）把 scale 存 **fp16**（与 `bytes.QuantSpec.scale_dtype_bytes=2`
的记账一致）；`rtn.dequantize_rtn` 是 fp32 scale。对 W8，fp16 scale 的舍入
（相对 ~1e-3）与量化误差（~5e-3）同量级，**不可忽略**，所以两个都算、都报。

用法：

    python -m tools.quant_error_audit --model <ckpt> --layers 0 --scopes attention,experts
    python -m tools.quant_error_audit --synthetic --json /tmp/audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 与 linear.TARGET_RULES 同一套命名；这里按**全名**匹配层内张量。
SCOPE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "attention": (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ),
    "router": ("mlp.gate.weight",),
    "experts": (
        "mlp.experts.{e}.gate_proj.weight",
        "mlp.experts.{e}.up_proj.weight",
        "mlp.experts.{e}.down_proj.weight",
    ),
    "lm_head": ("lm_head.weight",),
}

# docs/METRICS.md 的门槛不在这里 —— 本工具只谈**权重域**，不谈 KL 门。
QMAX = {4: 7, 8: 127}


def qmax_for(bits: int) -> int:
    return (2 ** (bits - 1)) - 1


def error_stats(
    w: Any, *, bits: int, group_size: int, symmetric: bool = True, scale_in_fp16: bool = False
) -> dict[str, Any]:
    """一个张量的权重域误差统计。`w` 必须已经是**要比较的那个** dtype（通常是 fp16）。

    `scale_in_fp16=True` 复现生产路径（`QuantLinear` 存 fp16 scale）。
    """
    from lemoq.quant import rtn as _rtn

    if w.ndim != 2:
        raise ValueError("只支持 2-D 权重，得到 ndim=%d" % w.ndim)
    wf = w.float()
    q = _rtn.quantize_rtn(wf, bits=bits, group_size=group_size, symmetric=symmetric)
    codes = q["codes_padded"].float()
    scale = q["scale"].float()
    if scale_in_fp16:
        scale = scale.half().float()
    out_features, padded_in = wf.shape
    n_groups = q["n_groups"]
    wh = codes.reshape(out_features, n_groups, group_size) * scale.reshape(
        out_features, n_groups, 1
    )
    wh = wh.reshape(out_features, padded_in)[:, : wf.shape[1]]

    err = wf - wh
    grouped_w = wf.reshape(out_features, n_groups, group_size)
    grouped_e = err.reshape(out_features, n_groups, group_size)
    step = scale.reshape(out_features, n_groups, 1)
    total_sq = float((err**2).sum())
    return {
        "bits": int(bits),
        "group_size": int(group_size),
        "shape": [int(wf.shape[0]), int(wf.shape[1])],
        "n_groups": int(n_groups),
        "padded_in": int(padded_in),
        "mse": float((err**2).mean()),
        "rmse": float((err**2).mean().sqrt()),
        "rel_fro": float(err.norm() / wf.norm()),
        "max_abs_err": float(err.abs().max()),
        "mean_abs_err": float(err.abs().mean()),
        "step_max": float(step.max()),
        # ★ step_max 与 group_size **无关**（全局 absmax 总在某个组里）；
        #   真正随分组变的是 step 的**分布**，所以均值才是单调性该看的量。
        "step_mean": float(step.mean()),
        "bound_max": float(step.max() / 2.0),
        # 界必须成立：|err| ≤ s/2。留 1e-6 的相对余量给 fp32 运算。
        "bound_violation": float((grouped_e.abs() - step / 2.0 - 1e-6).clamp(min=0.0).max()),
        "error_top1pct_share": _top_share(grouped_w, grouped_e, 0.01),
        "single_element_dominated_frac": _dominated_frac(grouped_w),
        "dynamic_range_p50": _dyn_range(grouped_w, 0.5),
        "dynamic_range_p95": _dyn_range(grouped_w, 0.95),
        "channel_rel_err_max": _channel_rel_err(wf, err)[0],
        "channel_rel_err_p50": _channel_rel_err(wf, err)[1],
        "zero_code_frac": float((q["codes_padded"] == 0).float().mean()),
        "total_sq_err": total_sq,
        "scale_dtype": "fp16" if scale_in_fp16 else "fp32",
    }


def _top_share(grouped_w: Any, grouped_e: Any, frac: float) -> float:
    """误差平方和中，|w| 最大的 `frac` 份额元素贡献的占比。

    ★ 用 `kthvalue` + 掩码而不是 `topk`：两者都精确，但 topk 在这个形状上
    （每张量 200 万元素）实测占掉 `error_stats` 总时间的 **43%**（0.025 s/call，
    cProfile）。审计要跑 (bits × group × 张量) 上千次 —— 那是十几分钟与两分钟的区别。
    """
    import torch

    a = grouped_w.abs().reshape(-1)
    e2 = (grouped_e**2).reshape(-1)
    total = float(e2.sum())
    if total <= 0:
        return 0.0
    k = min(a.numel(), max(1, int(a.numel() * frac)))
    if k >= a.numel():
        return 1.0
    threshold = torch.kthvalue(a, a.numel() - k + 1).values
    return float(e2[a >= threshold].sum() / total)


def _dominated_frac(grouped_w: Any) -> float:
    """被**单个元素**主导的组占比：absmax ≥ 3 × 第二大 |w|（RTN 离群失效的形态）。"""
    import torch

    a = grouped_w.abs()
    if a.shape[-1] < 2:
        return 0.0
    top2 = torch.topk(a, 2, dim=-1).values
    dominated = top2[..., 0] >= 3.0 * torch.clamp(top2[..., 1], min=1e-12)
    return float(dominated.float().mean())


def _dyn_range(grouped_w: Any, quantile: float) -> float:
    """每组的 `absmax / RMS` 分布的分位数。高斯权重下约 2.5–3.5。"""
    import torch

    a = grouped_w.abs()
    rms = a.pow(2).mean(dim=-1).sqrt().clamp(min=1e-12)
    dyn = a.amax(dim=-1) / rms
    return float(torch.quantile(dyn.reshape(-1).float(), quantile))


def _channel_rel_err(w: Any, err: Any) -> tuple[float, float]:
    """逐**输入通道**（列）的相对误差：AWQ 保护的正是这些通道。"""
    import torch

    num = err.pow(2).sum(dim=0).sqrt()
    den = w.pow(2).sum(dim=0).sqrt().clamp(min=1e-12)
    rel = num / den
    return float(rel.max()), float(torch.quantile(rel, 0.5))


def nested_scale_check(
    w: Any, bits: int, groups: tuple[int, ...] = (32, 64, 128)
) -> dict[str, Any]:
    """分组必须是**嵌套**的：s32 ≤ s64 ≤ s128（逐元素）。

    这条是"group↓ ⇒ 误差界↓"的**唯一**理由。它不成立时，非单调性就是实现问题。
    """
    from lemoq.quant import rtn as _rtn

    scales = {}
    for g in groups:
        q = _rtn.quantize_rtn(w.float(), bits=bits, group_size=g, symmetric=True)
        scales[g] = q["scale"].float()
    checks = {}
    ordered = list(groups)
    for fine, coarse in zip(ordered, ordered[1:]):
        if coarse % fine != 0:
            continue
        ratio = coarse // fine
        n_coarse = int(scales[coarse].shape[1])
        # 细组在同一个粗组内的最大值 ≤ 粗组的 absmax（等价于 s_fine ≤ s_coarse）
        fine_max = scales[fine].reshape(int(w.shape[0]), n_coarse, ratio).amax(dim=2)
        checks["%d<=%d" % (fine, coarse)] = bool((fine_max <= scales[coarse] + 1e-12).all())
    return {"groups": list(groups), "nested": checks, "all_nested": all(checks.values())}


def mse_monotonicity(w: Any, bits: int, groups: tuple[int, ...] = (32, 64, 128)) -> dict[str, Any]:
    """实测 MSE 随 group 的走向（这是**证据**，不是证明；界才是证明）。"""
    mses = {}
    for g in groups:
        mses[g] = error_stats(w, bits=bits, group_size=g)["mse"]
    ordered = list(groups)
    monotone = all(mses[a] <= mses[b] for a, b in zip(ordered, ordered[1:]))
    ratios = [round(mses[b] / mses[a], 4) for a, b in zip(ordered, ordered[1:])]
    return {
        "bits": int(bits),
        "mse_by_group": mses,
        "monotone": bool(monotone),
        "step_ratios": ratios,
    }


def checkpoint_weight(model_dir: str, name: str) -> Any:
    """**按名字惰性读一个张量**，不加载整个模型（safetensors 分片 + index）。"""
    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json"), encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    if name not in weight_map:
        raise KeyError("checkpoint 里没有 %s" % name)
    path = os.path.join(model_dir, weight_map[name])
    with safe_open(path, framework="pt", device="cpu") as fh:
        return fh.get_tensor(name).clone()


def layer_tensor_names(
    layer: int, scopes: tuple[str, ...], *, n_experts: int | None = None
) -> list[str]:
    names: list[str] = []
    for scope in scopes:
        for suffix in SCOPE_SUFFIXES[scope]:
            if "{e}" in suffix:
                if n_experts is None:
                    raise ValueError("experts 需要 n_experts")
                for e in range(n_experts):
                    names.append("model.layers.%d.%s" % (layer, suffix.format(e=e)))
            else:
                names.append("model.layers.%d.%s" % (layer, suffix))
    if "lm_head" in scopes:
        names = [n for n in names if not n.startswith("model.layers.")]
        names.append("lm_head.weight")
    return names


def audit_tensors(
    tensors: dict[str, Any],
    *,
    bits_list=(8, 4),
    groups=(32, 64, 128),
    scale_variants=("fp16",),
) -> dict[str, Any]:
    """对一组 (name -> weight) 做全量审计。返回可直接落盘的 dict。"""
    out: dict[str, Any] = {
        "tensors": {},
        "bits_list": list(bits_list),
        "groups": list(groups),
        "scale_variants": list(scale_variants),
    }
    for name, w in tensors.items():
        entry: dict[str, Any] = {"shape": [int(x) for x in w.shape], "dtype": str(w.dtype)}
        for bits in bits_list:
            per_g: dict[str, Any] = {}
            for g in groups:
                if int(w.shape[1]) < g:
                    per_g[str(g)] = {"skipped": "in_features < group_size"}
                    continue
                # 生产路径用 fp16 scale；fp32 scale 作为"量化器本身的误差"参照
                variants: dict[str, Any] = {}
                for variant in scale_variants:
                    variants["%s_scale" % variant] = error_stats(
                        w, bits=bits, group_size=g, scale_in_fp16=(variant == "fp16")
                    )
                per_g[str(g)] = variants
            # 单调性用**已经算出来的** mse，不再重复跑 error_stats
            mses = {
                int(g): per_g[str(g)]["fp16_scale"]["mse"]
                for g in groups
                if "fp16_scale" in per_g.get(str(g), {})
            }
            ordered = sorted(mses)
            entry["w%d" % bits] = per_g
            entry["nested_w%d" % bits] = nested_scale_check(w, bits, tuple(groups))
            entry["mse_monotone_w%d" % bits] = {
                "bits": int(bits),
                "mse_by_group": mses,
                "monotone": all(mses[a] <= mses[b] for a, b in zip(ordered, ordered[1:])),
                "step_ratios": [round(mses[b] / mses[a], 4) for a, b in zip(ordered, ordered[1:])],
            }
        out["tensors"][name] = entry
    return out


def _summarize(result: dict[str, Any]) -> list[dict[str, Any]]:
    """把逐张量结果压成一行一条的摘要（用于人读与记录）。"""
    rows: list[dict[str, Any]] = []
    for name, entry in result["tensors"].items():
        row: dict[str, Any] = {"name": name}
        for bits in result["bits_list"]:
            key = "w%d" % bits
            if key not in entry:
                continue
            stats = entry[key]
            row["w%d_rel_fro" % bits] = {
                g: round(float(stats[g]["fp16_scale"]["rel_fro"]), 6)
                for g in stats
                if isinstance(stats[g], dict) and "fp16_scale" in stats[g]
            }
            row["w%d_mse" % bits] = {
                g: float(stats[g]["fp16_scale"]["mse"])
                for g in stats
                if isinstance(stats[g], dict) and "fp16_scale" in stats[g]
            }
            row["w%d_monotone" % bits] = entry["mse_monotone_w%d" % bits]["monotone"]
            row["w%d_nested" % bits] = entry["nested_w%d" % bits]["all_nested"]
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model", default=None, help="HF checkpoint 目录（含 model.safetensors.index.json）"
    )
    ap.add_argument("--layers", default="0", help="逗号分隔的层号")
    ap.add_argument("--scopes", default="attention", help="attention,experts,router,lm_head")
    ap.add_argument("--n-experts", type=int, default=64)
    ap.add_argument("--max-experts", type=int, default=None, help="专家只审前 N 个（省时间）")
    ap.add_argument("--bits", default="8,4")
    ap.add_argument("--groups", default="32,64,128")
    ap.add_argument(
        "--synthetic", action="store_true", help="不读 checkpoint，用合成权重（Mac 冒烟用）"
    )
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--dtype",
        default="float16",
        help=(
            "checkpoint 权重转成什么 dtype 再量化。**必须与 runner 一致**：OLMoE 的 "
            "checkpoint 存的是 bf16，而 `_load_model(..., 'float16', ...)` 会先转 fp16 —— "
            "审计不跟着转，量的就是另一个张量。"
        ),
    )
    ap.add_argument(
        "--scale-variants",
        default="fp16",
        help="fp16 / fp32 / fp16,fp32（默认只算生产路径的 fp16 scale）",
    )
    args = ap.parse_args(argv)

    import torch

    bits_list = tuple(int(b) for b in str(args.bits).split(","))
    groups = tuple(int(g) for g in str(args.groups).split(","))
    scopes = tuple(s.strip() for s in str(args.scopes).split(",") if s.strip())

    if args.synthetic:
        # 合成权重：高斯 + 一个**植入的离群通道**，这样离群统计也有东西可测。
        torch.manual_seed(0)
        tensors = {}
        for scope in scopes:
            shape = {"attention": (2048, 2048), "router": (64, 2048), "experts": (2048, 1024)}[
                scope
            ]
            w = (torch.randn(*shape) * 0.02).half()
            w[:, 7] *= 40.0  # 一个输入通道离群（AWQ 的动机）
            tensors["synthetic.%s" % scope] = w
    else:
        if not args.model:
            print("需要 --model 或 --synthetic", file=sys.stderr)
            return 2
        tensors = {}
        n_experts = args.n_experts
        for layer in (int(x) for x in str(args.layers).split(",")):
            names = layer_tensor_names(layer, scopes, n_experts=n_experts)
            kept = 0
            for name in names:
                if ".mlp.experts." in name and args.max_experts is not None:
                    idx = int(name.split(".experts.")[1].split(".")[0])
                    if idx >= args.max_experts:
                        continue
                tensor = checkpoint_weight(args.model, name)
                if str(tensor.dtype) != "torch." + args.dtype:
                    tensor = tensor.to(getattr(torch, args.dtype))  # bf16 -> fp16，与 runner 同口径
                tensors[name] = tensor
                kept += 1
            print("layer %d: read %d tensors" % (layer, kept), file=sys.stderr)

    result = audit_tensors(
        tensors,
        bits_list=bits_list,
        groups=groups,
        scale_variants=tuple(s.strip() for s in str(args.scale_variants).split(",") if s.strip()),
    )
    result["summary"] = _summarize(result)
    result["meta"] = {
        "model": args.model,
        "synthetic": bool(args.synthetic),
        "scopes": list(scopes),
        "note": (
            "纯权重域审计：不含模型、不含 KL。fp16_scale = 生产路径（QuantLinear 存 fp16 scale），"
            "fp32_scale = rtn.dequantize_rtn 的参照。"
        ),
    }

    print(
        "  %-52s %-26s %-26s %s"
        % ("tensor", "w8 rel_fro (g32/64/128)", "w4 rel_fro (g32/64/128)", "mono8/mono4")
    )
    for row in result["summary"][:400]:

        def _fmt(key):
            d = row.get(key)
            return "/".join("%.2e" % v for v in d.values()) if d else "-"

        print(
            "  %-52s %-26s %-26s %s/%s"
            % (
                row["name"][:52],
                _fmt("w8_rel_fro"),
                _fmt("w4_rel_fro"),
                row.get("w8_monotone"),
                row.get("w4_monotone"),
            )
        )
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
        print("\nwrote %s" % args.json, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
