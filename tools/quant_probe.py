"""quant_probe — 把 `EXP-p1-01` 的两个非单调性信号**归因**（GPU + teacher forcing）。

## 三个层次，缺一不可

    cause    每个 arm 真实造成的**权重域**误差（与 tools/quant_error_audit 同一套函数）
    profile  每一层输出的相对误差 ||h_l − h_l^fp16|| / ||h_l^fp16|| —— 误差**在哪一层放大**
    KL       end-to-end，与 EXP-p1-01 同一套 teacher forcing + compare_logits，
             另外报**逐位置** KL（看是不是被一两个位置主导）

只报 KL 的话，"W8 的 g64 比 g128 差 1.4×"既可能是模型敏感、也可能是实现缺陷、
也可能只是这个指标的噪声。三层一起看才能区分。

## 噪声臂（本条实验的核心设计）

分组从 32 变到 128，只让权重 MSE 动约 1.2×（`tools.quant_error_audit` 实测），
而 KL 动了 1.4× 且非单调。要判断"这 1.4× 是不是信号"，必须知道
**同一误差量级下换一个扰动方向，KL 会动多少**。所以：

    δ = λ · σ₈ ⊙ ε        σ₈ = W8A16 g128 的**逐张量** RMS 误差（实测，不是假设）

  - 固定方向 ε、扫 λ ∈ {0.5, 1, 1.2, 2, 4, 8, 18.143}
    → 误差量级单调变化的曲线。λ = 18.143 正是 W4/W8 的**步长比**，
      即"把 W8 的误差放大到 W4 的量级但**不给它 W4 的结构**"。
      拿它与真实 W4 的 KL 比，就能把"量级"与"结构（离群塌缩）"分开。
  - 固定 λ = 1、换 4 个方向 → **同量级下的方向散布** = 这个指标的噪声地板。

## 空臂（noop）是自检，不是装饰

每个 arm 之前都会从 CPU 快照恢复权重。`noop` = "恢复但不扰动"，它的 KL 必须是
**恰好 0.0**；否则说明恢复不干净，后面的 arm 全部不可信。它按固定间隔插入。

用法：

    python -m tools.quant_probe --model M --arms repro_attn,noise --json out.json
    python -m tools.quant_probe --model M --dry-run          # 只列臂，不占 GPU
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks.quant_gate import teacher_forced_logits  # noqa: E402

# W8→W4 的**步长**比：(2^7−1)/(2^3−1)。噪声臂用它把 W8 的误差放大到 W4 的量级。
STEP_RATIO_W4_OVER_W8 = 127.0 / 7.0
DEFAULT_LAMBDAS = (0.5, 1.0, 1.2, 2.0, 4.0, 8.0, STEP_RATIO_W4_OVER_W8)
DEFAULT_DIRECTIONS = 4


# ---------------------------------------------------------------------------
# 臂的定义
# ---------------------------------------------------------------------------
def build_arms(
    *,
    lambdas=DEFAULT_LAMBDAS,
    directions=DEFAULT_DIRECTIONS,
    include: set[str] | None = None,
) -> list[dict[str, Any]]:
    """全部臂的清单。`include` 非空时只保留这些组。"""
    arms: list[dict[str, Any]] = []

    def add(group: str, arm: dict[str, Any]) -> None:
        arm = dict(arm)
        arm["group"] = group
        arms.append(arm)

    for scheme in ("w8a16", "w4a16"):
        for g in (32, 64, 128):
            add(
                "repro_attn",
                {
                    "id": "%s_attn_g%d" % (scheme, g),
                    "kind": "rtn",
                    "scheme": scheme,
                    "group_size": g,
                    "targets": ["attention"],
                    "layers": None,
                },
            )
    for scheme in ("w8a16", "w4a16"):
        for g in (32, 64, 128):
            add(
                "repro_doc",
                {
                    "id": "%s_attn_experts_g%d" % (scheme, g),
                    "kind": "rtn",
                    "scheme": scheme,
                    "group_size": g,
                    "targets": ["attention", "experts"],
                    "layers": None,
                },
            )
    for scheme in ("w8a16", "w4a16"):
        add(
            "experts_only",
            {
                "id": "%s_experts_g128" % scheme,
                "kind": "rtn",
                "scheme": scheme,
                "group_size": 128,
                "targets": ["experts"],
                "layers": None,
            },
        )
    for g in (32, 64, 128):
        add(
            "layer0",
            {
                "id": "w8a16_layer0_attn_experts_g%d" % g,
                "kind": "rtn",
                "scheme": "w8a16",
                "group_size": g,
                "targets": ["attention", "experts"],
                "layers": [0],
            },
        )
    for lam in lambdas:
        add(
            "noise",
            {
                "id": "noise_lam%s_dir0" % _fmt_lam(lam),
                "kind": "noise",
                "lam": float(lam),
                "direction": 0,
                "targets": ["attention"],
                "layers": None,
            },
        )
    for d in range(1, max(1, directions)):
        add(
            "noise",
            {
                "id": "noise_lam1.0_dir%d" % d,
                "kind": "noise",
                "lam": 1.0,
                "direction": int(d),
                "targets": ["attention"],
                "layers": None,
            },
        )
    # 逐投影类型：方向散布（在同量级下 KL 差 19×）到底集中在哪个张量类型上？
    # 这直接决定"混合位宽该保护谁"（REMEDIATION_LADDER 第 4 档）。
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        for direction in (0, 2):
            add(
                "sensitivity",
                {
                    "id": "noise_%s_lam1_dir%d" % (proj, direction),
                    "kind": "noise",
                    "lam": 1.0,
                    "direction": direction,
                    "targets": ["attention"],
                    "layers": None,
                    # ★ 只扰动这一类张量：σ 仍取该张量自己实测的 W8 RMS，
                    #   所以这里的变量是"扰动落在哪个张量类型上"
                    "filter": proj,
                },
            )
    # 逐投影的**真实 RTN**（不是噪声）：混合位宽该把哪一类投影留在高精度？
    # W8 那组由 noise_*_proj_lam1_* 回答（同量级），这里补 W4，看 q/k 在 W4 是否也崩。
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        add(
            "perproj",
            {
                "id": "w4a16_%s_g128" % proj,
                "kind": "rtn",
                "scheme": "w4a16",
                "group_size": 128,
                "targets": ["attention"],
                "layers": None,
                "filter": proj,
            },
        )

    # ---- 混合位宽（REMEDIATION_LADDER 第 4 档）----
    # 每格只改**一个**子计划，且都在同一道门（EXP-p1-01 的 KL 门槛）下比。
    # 输入来自 EXP-p1-02 的逐投影结果：q/k 在 W4 只有 0.0017–0.0019，v/o 各 4.05。
    def _plan(entries):
        return [dict(e) for e in entries]

    add(
        "mixed",
        {
            "id": "mixed_baseline_w8",
            "kind": "mixed",
            "plan": _plan(
                [
                    {
                        "filter": "self_attn",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "mlp.experts",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["experts"],
                    },
                ]
            ),
        },
    )
    add(
        "mixed",
        {
            "id": "mixed_qk_w4_vo_w8_exp_w8",
            "kind": "mixed",
            "plan": _plan(
                [
                    {
                        "filter": "q_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "k_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "v_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "o_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "mlp.experts",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["experts"],
                    },
                ]
            ),
        },
    )
    add(
        "mixed",
        {
            "id": "mixed_qk_w8_vo_w4_exp_w8",
            "kind": "mixed",
            "plan": _plan(
                [
                    {
                        "filter": "q_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "k_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "v_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "o_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "mlp.experts",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["experts"],
                    },
                ]
            ),
        },
    )
    add(
        "mixed",
        {
            "id": "mixed_qk_w4_vo_w8_exp_w4",
            "kind": "mixed",
            "plan": _plan(
                [
                    {
                        "filter": "q_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "k_proj",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "v_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "o_proj",
                        "scheme": "w8a16",
                        "group_size": 128,
                        "targets": ["attention"],
                    },
                    {
                        "filter": "mlp.experts",
                        "scheme": "w4a16",
                        "group_size": 128,
                        "targets": ["experts"],
                    },
                ]
            ),
        },
    )
    # 专家的 96% 字节在哪一类投影上最敏感？（attention 的答案是 v/o）
    for proj_name, proj_filter in (("gate_up", ("gate_proj", "up_proj")), ("down", ("down_proj",))):
        other = ("down_proj",) if proj_name == "gate_up" else ("gate_proj", "up_proj")
        add(
            "mixed",
            {
                "id": "mixed_exp_%s_w4" % proj_name,
                "kind": "mixed",
                "plan": _plan(
                    [
                        {"filter": f, "scheme": "w4a16", "group_size": 128, "targets": ["experts"]}
                        for f in proj_filter
                    ]
                    + [
                        {"filter": f, "scheme": "w8a16", "group_size": 128, "targets": ["experts"]}
                        for f in other
                    ]
                    + [
                        {
                            "filter": "self_attn",
                            "scheme": "w8a16",
                            "group_size": 128,
                            "targets": ["attention"],
                        }
                    ]
                ),
            },
        )
    add("noop", {"id": "noop_a", "kind": "noop", "targets": [], "layers": None})
    add("noop", {"id": "noop_b", "kind": "noop", "targets": [], "layers": None})
    add("noop", {"id": "noop_final", "kind": "noop", "targets": [], "layers": None})

    if include:
        # ★ noop 臂是自检，**永远保留**：选子集时把自检一起丢掉，
        #   等于在最需要它的时候（长时间、多臂的增量运行）失去恢复验证。
        arms = [a for a in arms if a["group"] in include or a["kind"] == "noop"]
    return _interleave_noops(arms)


def _fmt_lam(lam: float) -> str:
    return ("%.3f" % lam).rstrip("0").rstrip(".")


def _interleave_noops(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 noop 夹在真实臂之间（首个 noop 放在第 3 个臂之后，之后每 4 个臂一个）。

    为什么不是只放最后一个：如果只在最后检查，一次坏的恢复会污染它**前面**的所有臂，
    而检查只告诉你"最后是坏的"。夹在中间能把污染**定位**到具体区段。
    """
    real = [a for a in arms if a["kind"] != "noop"]
    noops = [a for a in arms if a["kind"] == "noop"]
    if not real or not noops:
        return real + noops
    # 均匀插在 1/4、1/2、末尾：一处坏恢复能被定位到具体区段，而不是只告诉你"最后是坏的"
    positions = sorted({max(0, len(real) // 4), max(0, len(real) // 2), len(real) - 1})
    out: list[dict[str, Any]] = []
    nxt = 0
    for i, arm in enumerate(real):
        out.append(arm)
        if nxt < len(noops) and i == positions[min(nxt, len(positions) - 1)]:
            out.append(noops[nxt])
            nxt += 1
    out.extend(noops[nxt:])
    return out


# ---------------------------------------------------------------------------
# 权重快照 / 恢复（**必须干净**，noop 臂就是它的检验）
# ---------------------------------------------------------------------------
def _candidate_names(model: Any, *, targets: Any, layers: Any) -> list[str]:
    import torch.nn as nn  # noqa: PLC0415

    from lemoq.quant.linear import _layer_index, classify_linear  # noqa: PLC0415

    targets = set(targets)
    layer_filter = None if layers is None else {int(x) for x in layers}
    names: list[str] = []
    for name, module in model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        target = classify_linear(name)
        if target is None or target not in targets:
            continue
        if layer_filter is not None and _layer_index(name) not in layer_filter:
            continue
        names.append(name)
    return names


def _snapshot(model: Any, names: list[str]) -> dict[str, Any]:
    """把原模块**搬到 CPU** 并保留对象引用。恢复时放回去就是逐位相同。"""
    snap: dict[str, Any] = {}
    for name in names:
        module = model.get_submodule(name)
        snap[name] = module
        module.to("cpu")
    return snap


def _restore(model: Any, snap: dict[str, Any]) -> int:
    moved = 0
    for name, module in snap.items():
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if getattr(parent, leaf, None) is not module:
            setattr(parent, leaf, module)
            moved += 1
        if module.weight.device.type != "cuda":
            module.to("cuda")
    return moved


# ---------------------------------------------------------------------------
# 扰动
# ---------------------------------------------------------------------------
def _sigma_table(
    model: Any, names: list[str], *, bits: int = 8, group_size: int = 128
) -> dict[str, Any]:
    """每个张量在 (bits, group_size) 下的**实测** RMS 误差（fp16 scale = 生产路径）。"""
    from tools.quant_error_audit import error_stats  # noqa: PLC0415

    table: dict[str, Any] = {}
    for name in names:
        w = model.get_submodule(name).weight.detach().float().cpu()
        stats = error_stats(w.half(), bits=bits, group_size=group_size, scale_in_fp16=True)
        table[name] = {
            "rmse": float(stats["rmse"]),
            "rel_fro": float(stats["rel_fro"]),
            "numel": int(w.numel()),
            "fro": float(w.norm()),
        }
    return table


def _inject_noise(
    model: Any, names: list[str], table: dict[str, Any], *, lam: float, direction: int
) -> dict[str, Any]:
    """`w ← w + λ·σ·ε`（ε 用 CPU 生成，便于事后**精确**复算权重域统计）。"""
    import torch  # noqa: PLC0415

    if not names:
        raise ValueError("noise: 没有任何张量")
    sq = 0.0
    fro2 = 0.0
    for name in names:
        module = model.get_submodule(name)
        w = module.weight
        gen = torch.Generator().manual_seed(
            int(zlib.crc32(name.encode("utf-8"))) + 1000 * int(direction)
        )
        eps = torch.randn(w.shape, generator=gen, dtype=torch.float32)
        sig = float(table[name]["rmse"])
        # ★ 不许 `w.data.copy_()` 原地改：快照里存的是**同一个模块对象**，原地改会
        #   把快照一起改掉 → 恢复变成空操作 → 噪声臂**累积**。实测就是这样丢掉一版的：
        #   noop_b 的 KL 恰好等于前一个噪声臂的 KL（0.0648），而 noop 自检本就是为抓它。
        #   做法与 quantize_model 一致：新建模块挂上去，原模块保持不动。
        perturbed = (w.detach().float() + (lam * sig * eps).to(w.device)).half()
        clone = torch.nn.Linear(
            module.in_features, module.out_features, bias=module.bias is not None
        )
        clone = clone.to(device=w.device, dtype=w.dtype)
        with torch.no_grad():
            clone.weight.copy_(perturbed)
            if module.bias is not None:
                clone.bias.copy_(module.bias)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, clone)
        sq += float((lam * sig) ** 2) * int(table[name]["numel"])
        fro2 += float(table[name]["fro"]) ** 2
    return {"rel_fro_perturbation": float((sq**0.5) / (fro2**0.5)) if fro2 else 0.0}


def _weight_domain_stats(
    snapshot: dict[str, Any], names: list[str], *, bits: int, group_size: int, max_tensors: int = 32
) -> dict[str, Any]:
    """RTN 臂的权重域误差（与 `error_stats` 同一套函数），从 **CPU 快照** 算。

    只取 `max_tensors` 个张量做均匀子采样：attn+experts 臂有 3136 个 Linear，
    逐个算要 9 分钟，而 32 个张量（≈1 亿个元素）的估计已经足够确定。
    子采样事实**写进结果**（`subsampled`），不假装是全量。
    """
    from tools.quant_error_audit import error_stats  # noqa: PLC0415

    if not names:
        raise ValueError("weight_domain: 没有任何张量 —— 名字必须在**量化之前**收集")
    stride = max(1, len(names) // max_tensors)
    picked = names[::stride]
    total_sq = 0.0
    total_fro2 = 0.0
    numel = 0
    per_tensor: dict[str, float] = {}
    for name in picked:
        w = snapshot[name].weight.detach().float().cpu().half()
        stats = error_stats(w, bits=bits, group_size=group_size, scale_in_fp16=True)
        total_sq += float(stats["total_sq_err"])
        total_fro2 += float(w.float().norm()) ** 2
        numel += int(w.numel())
        if len(picked) <= 8:
            per_tensor[name] = round(float(stats["rel_fro"]), 6)
    return {
        "rel_fro": float((total_sq**0.5) / (total_fro2**0.5)) if total_fro2 else 0.0,
        "rmse": float((total_sq / max(numel, 1)) ** 0.5),
        "numel": int(numel),
        "subsampled": len(picked) < len(names),
        "tensors_used": len(picked),
        "tensors_total": len(names),
        "per_tensor_rel_fro": per_tensor or "omitted",
    }


# ---------------------------------------------------------------------------
# 测量
# ---------------------------------------------------------------------------
def _per_position_kl(reference: list[Any], candidate: list[Any]) -> list[float]:
    import torch  # noqa: PLC0415

    out: list[float] = []
    for ref, cand in zip(reference, candidate):
        r = ref.detach().reshape(1, -1).to(dtype=torch.float32, device="cpu")
        c = cand.detach().reshape(1, -1).to(dtype=torch.float32, device="cpu")
        rl = torch.log_softmax(r, dim=-1)
        cl = torch.log_softmax(c, dim=-1)
        out.append(float((rl.exp() * (rl - cl)).sum()))
    return out


def _capture_hidden(model: Any, ids: Any, *, n_layers: int) -> list[Any]:
    """prefill 一次，抓每一层的输出 hidden state（用于逐层放大曲线）。"""
    import torch  # noqa: PLC0415

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[idx] = tensor.detach().float()

        return hook

    for idx in range(n_layers):
        handles.append(model.model.layers[idx].register_forward_hook(make_hook(idx)))
    try:
        with torch.no_grad():
            model(ids, use_cache=True)
    finally:
        for handle in handles:
            handle.remove()
    return [captured[i] for i in range(n_layers)]


def _layer_profile(reference_hidden: list[Any], current_hidden: list[Any]) -> list[float]:
    prof: list[float] = []
    for ref, cur in zip(reference_hidden, current_hidden):
        denom = float(ref.norm())
        prof.append(float((cur - ref).norm()) / denom if denom else 0.0)
    return prof


def _measure(
    model: Any,
    *,
    ids: Any,
    tokens: list[int],
    reference_logits: list[Any],
    reference_hidden: list[Any],
    device: str,
    n_layers: int,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    from benchmarks.cudagraph import compare_logits  # noqa: PLC0415

    hidden = _capture_hidden(model, ids, n_layers=n_layers)
    candidate = teacher_forced_logits(model, ids, tokens, device=device)
    torch.cuda.synchronize()
    report = compare_logits(reference_logits, candidate)
    per_pos = _per_position_kl(reference_logits, candidate)
    order = sorted(range(len(per_pos)), key=lambda i: -per_pos[i])
    return {
        "kl": float(report["kl"]),
        "top1_agreement": float(report["top1_agreement"]),
        "per_position_kl": [round(v, 8) for v in per_pos],
        "per_position_kl_max": round(per_pos[order[0]], 8),
        "per_position_kl_max_index": int(order[0]),
        "per_position_kl_median": round(sorted(per_pos)[len(per_pos) // 2], 8),
        "kl_share_top3_positions": round(
            sum(per_pos[i] for i in order[:3]) / max(sum(per_pos), 1e-12), 4
        ),
        "layer_output_rel_err": [round(v, 8) for v in _layer_profile(reference_hidden, hidden)],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--arms",
        default="all",
        help="逗号分隔的组名（repro_attn,repro_doc,experts_only,layer0,noise,noop）或 all",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--published-summary",
        default=None,
        help="EXP-p1-01 的 summary.json（用于核对轨迹是否一致）",
    )
    ap.add_argument(
        "--prompt-source",
        default="ones",
        choices=["ones", "wikitext"],
        help=(
            "teacher forcing 的 prompt。★ `ones` 是 EXP-p1-01/02 用的（1024 个 token id=1）——"
            "那是一段**退化输入**，不是 METRICS.md 说的'固定 prompt 集'。"
            "`wikitext` 用真实文本的前 ctx 个 token，用来检验 KL 是不是 prompt 的产物。"
        ),
    )
    ap.add_argument(
        "--prompt-cache", default=None, help="wikitext token id 缓存（与 tools/quant_ppl 共用）"
    )
    ap.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="从缓存 token 流的哪个位置取 prompt（换一个窗口 = 换一个 prompt，用于检查结论不是单个窗口的巧合）",
    )
    args = ap.parse_args(argv)

    include = (
        None
        if args.arms.strip() == "all"
        else {a.strip() for a in args.arms.split(",") if a.strip()}
    )
    arms = build_arms(include=include)

    if args.dry_run:
        print("%d arms:" % len(arms))
        for arm in arms:
            print(
                "  %-34s %-6s %-10s targets=%-22s layers=%-6s filter=%s"
                % (
                    arm["id"],
                    arm["kind"],
                    arm.get("scheme", arm.get("lam", "")),
                    ",".join(arm.get("targets") or []) or "-",
                    arm.get("layers"),
                    arm.get("filter", "-"),
                )
            )
        return 0

    import torch

    from benchmarks.cudagraph import compare_logits
    from benchmarks.runner import _load_model
    from lemoq.quant.linear import SchemeSpec, quantize_model

    if not torch.cuda.is_available():
        print("需要 CUDA；GPU 上的数字不得在无 GPU 的机器上产生。", file=sys.stderr)
        return 2

    device = "cuda"
    model, _ = _load_model(args.model, "float16", device)
    n_layers = int(model.config.num_hidden_layers)
    if args.prompt_source == "wikitext":
        if not args.prompt_cache or not os.path.exists(args.prompt_cache):
            print(
                "--prompt-source wikitext 需要 --prompt-cache 指向已缓存的 token id",
                file=sys.stderr,
            )
            return 2
        cached = torch.load(args.prompt_cache, weights_only=True)
        if cached.numel() < args.ctx:
            print("缓存的 token 数 %d < ctx %d" % (int(cached.numel()), args.ctx), file=sys.stderr)
            return 2
        start = int(args.prompt_offset)
        if cached.numel() < start + args.ctx:
            print(
                "缓存 token 不够：%d < offset %d + ctx %d" % (int(cached.numel()), start, args.ctx),
                file=sys.stderr,
            )
            return 2
        ids = cached[start : start + args.ctx].view(1, -1).to(device)
    else:
        # EXP-p1-01/02 的口径：1024 个 token id=1。保留它以便逐位复现历史数字。
        ids = torch.full((1, args.ctx), 1, dtype=torch.long, device=device)

    print("=== 取轨迹（fp16 贪心 %d 步）===" % args.max_new_tokens, flush=True)
    tokens: list[int] = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        past = out.past_key_values
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        for _ in range(args.max_new_tokens):
            tokens.append(int(tok.item()))
            out = model(tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
    reference_logits = teacher_forced_logits(model, ids, tokens, device=device)
    reference_hidden = _capture_hidden(model, ids, n_layers=n_layers)
    torch.cuda.synchronize()

    result: dict[str, Any] = {
        "tool": "tools/quant_probe.py",
        "model": args.model,
        "ctx": args.ctx,
        "max_new_tokens": args.max_new_tokens,
        "n_layers": n_layers,
        "prompt_source": args.prompt_source,
        "prompt_first_tokens": [int(x) for x in ids[0, :8].tolist()],
        "trajectory_tokens": tokens,
        "step_ratio_w4_over_w8": STEP_RATIO_W4_OVER_W8,
        # 自检：参考与自己的 KL 必须是 0 —— 这是"指标实现没写错"的最小断言
        "reference_self_kl": float(compare_logits(reference_logits, reference_logits)["kl"]),
        "arms": {},
    }
    if args.published_summary and os.path.exists(args.published_summary):
        with open(args.published_summary, encoding="utf-8") as fh:
            published = json.load(fh)
        ref_tokens = published.get("trajectory_tokens")
        result["published_summary"] = args.published_summary
        result["trajectory_matches_published"] = ref_tokens == tokens
        result["published_w8a16_kl"] = {
            k.split("__")[-1]: v.get("kl") for k, v in (published.get("cells") or {}).items()
        }

    # 所有臂可能用到的 Linear：快照一次，供全部 arm 恢复
    all_names: list[str] = []
    for arm in arms:
        targets = arm.get("targets") or ["attention"]
        if arm["kind"] == "mixed":
            # ★ 混合臂的子计划可能覆盖 experts：不把它们纳入快照，恢复就是空操作，
            #   后面所有臂都会被污染（noop 自检会抓到，但那是浪费一整轮 GPU）。
            targets = sorted({t for entry in arm["plan"] for t in entry["targets"]})
        for name in _candidate_names(model, targets=targets, layers=arm.get("layers")):
            if name not in all_names:
                all_names.append(name)
    print("snapshot %d Linear -> CPU" % len(all_names), flush=True)
    t_snap = time.perf_counter()
    snapshot = _snapshot(model, all_names)
    print("  %.1f s" % (time.perf_counter() - t_snap), flush=True)

    sigma_names = _candidate_names(model, targets=["attention"], layers=None)
    print("sigma table for %d attention tensors ..." % len(sigma_names), flush=True)
    sigma_table = _sigma_table(model, sigma_names)
    _sig = sorted(v["rmse"] for v in sigma_table.values())
    _fro2 = sum(float(v["fro"]) ** 2 for v in sigma_table.values())
    _numel = sum(int(v["numel"]) for v in sigma_table.values())
    _sq = sum(float(v["rmse"]) ** 2 * int(v["numel"]) for v in sigma_table.values())
    result["sigma_table_summary"] = {
        "source": "W8A16 group=128, fp16 scale（生产路径）",
        "n_tensors": len(sigma_table),
        "rmse_median": _sig[len(_sig) // 2],
        "rmse_min": _sig[0],
        "rmse_max": _sig[-1],
        "attention_rel_fro": (float(_sq**0.5) / float(_fro2**0.5)) if _fro2 else None,
        "numel": _numel,
    }

    def _write() -> None:
        if args.json:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    for arm in arms:
        t0 = time.perf_counter()
        moved = _restore(model, snapshot)
        entry: dict[str, Any] = {
            "kind": arm["kind"],
            "group": arm["group"],
            "restored_modules": moved,
        }
        # ★ 必须在扰动**之前**收集名字：`quantize_model` 会把 nn.Linear 换成
        #   QuantLinear（**不是** nn.Linear 的子类），之后再按类型找就一个也找不到 ——
        #   那样 weight_domain 会静默变成 0，而 0 看起来像"没有误差"。
        arm_targets = arm.get("targets") or ["attention"]
        if arm["kind"] == "mixed":
            arm_targets = sorted({t for entry in arm["plan"] for t in entry["targets"]})
        arm_names = _candidate_names(model, targets=arm_targets, layers=arm.get("layers"))
        if arm.get("filter"):
            arm_names = [n for n in arm_names if arm["filter"] in n]
            entry["filter"] = arm["filter"]
        entry["n_target_tensors"] = len(arm_names)
        try:
            if arm["kind"] == "rtn":
                spec = SchemeSpec.from_flags(
                    arm["scheme"], arm["group_size"], targets=tuple(arm["targets"])
                )
                _, qreport = (
                    quantize_model(model, spec, layers=arm.get("layers"), names=arm_names or None)
                    if arm.get("filter")
                    else quantize_model(model, spec, layers=arm.get("layers"))
                )
                entry["spec"] = qreport.spec
                entry["replaced"] = qreport.replaced
                entry["realized_bytes_total"] = qreport.to_dict()["realized_bytes_total"]
                entry["realized_bytes_by_target"] = qreport.to_dict()["realized_bytes_by_target"]
                entry["weight_domain"] = _weight_domain_stats(
                    snapshot,
                    arm_names,
                    bits=int(spec.weight_bits),
                    group_size=int(arm["group_size"]),
                )
            elif arm["kind"] == "mixed":
                sub_reports = []
                for entry in arm["plan"]:
                    spec = SchemeSpec.from_flags(
                        entry["scheme"], entry["group_size"], targets=tuple(entry["targets"])
                    )
                    names = [n for n in arm_names if entry["filter"] in n]
                    if not names:
                        raise ValueError("mixed 子计划 %s 没匹配到张量" % entry)
                    _, sub = quantize_model(model, spec, names=names)
                    sub_reports.append(
                        {
                            "filter": entry["filter"],
                            "scheme": entry["scheme"],
                            "group_size": entry["group_size"],
                            "replaced": sub.replaced,
                            "bytes": sub.to_dict()["realized_bytes_by_target"],
                            "n_names": len(names),
                        }
                    )
                entry["plan"] = sub_reports
                entry["realized_bytes_total"] = int(
                    sum(
                        bucket["total_bytes"]
                        for report in sub_reports
                        for bucket in report["bytes"].values()
                    )
                )
            elif arm["kind"] == "noise":
                entry["weight_domain"] = _inject_noise(
                    model,
                    arm_names,
                    sigma_table,
                    lam=float(arm["lam"]),
                    direction=int(arm["direction"]),
                )
                entry["lam"] = float(arm["lam"])
                entry["direction"] = int(arm["direction"])
                entry["sigma_source"] = "W8A16 g128 rmse per attention tensor (measured)"
            elif arm["kind"] == "noop":
                entry["note"] = "恢复但不扰动：KL 必须**恰好** 0.0，否则恢复不干净"
            else:
                raise ValueError("unknown kind %r" % arm["kind"])

            entry.update(
                _measure(
                    model,
                    ids=ids,
                    tokens=tokens,
                    reference_logits=reference_logits,
                    reference_hidden=reference_hidden,
                    device=device,
                    n_layers=n_layers,
                )
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        entry["seconds"] = round(time.perf_counter() - t0, 2)
        result["arms"][arm["id"]] = entry
        _write()
        print(
            "  %-36s kl=%-12s top1=%-6s layer0_rel=%-10s (%.1f s)"
            % (
                arm["id"],
                round(entry.get("kl", float("nan")), 6) if entry.get("kl") is not None else "ERR",
                entry.get("top1_agreement"),
                (
                    round(entry.get("layer_output_rel_err", [0.0])[0], 6)
                    if entry.get("layer_output_rel_err")
                    else "-"
                ),
                entry["seconds"],
            ),
            flush=True,
        )

    _restore(model, snapshot)
    final_noop = next((a["id"] for a in arms if a["id"] == "noop_final"), None)
    if final_noop:
        result["restore_verified_by"] = {
            "arm": final_noop,
            "kl": result["arms"].get(final_noop, {}).get("kl"),
            "is_exact": result["arms"].get(final_noop, {}).get("kl") == 0.0,
        }
    _write()
    print("\nwrote %s" % args.json if args.json else "", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
