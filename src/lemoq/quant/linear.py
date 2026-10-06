"""linear — 把量化真正**装到模型上**（Phase 1 的缺失环节）。

## 为什么需要这个模块

`lemoq.quant` 一直只到 RTN + 位打包 + oracle 为止：能对**一个**权重张量量化，
但没有"把量化装进模型"的东西 —— 没有 `QuantLinear`、没有 `quantize_model`。
于是 `EXP-p1-01`（W8/W4 精度门）**跑不了**，`benchmarks/bytes.py` 里那套
`[W]` 字节账也从来没有被一个**真实的实现**验证过。

本模块补上的就是这一段，并且刻意**只做精度、不做性能**（PLAN §5 Phase 1）：

- `QuantLinear` 持有**打包后**的 codes（真的按位存，不是为了记账而假装），
  forward 里反量化回 fp16 再走 `F.linear`。这是 `EXP-p2-01` 的 `separate` 臂。
- 激活量化（`w8a8`）用**伪量化**（quantize 后立刻 dequantize）。
  它测的是**数值影响**，不是速度 —— 真 int8 GEMM 属于 Phase 2。
  这一点必须在结果里写明，否则"W8A8"会被误读成"已经做了 int8 计算"。

## 两个必须显式报告的量

1. **padding 膨胀**：`quantize_rtn` 会把输入维补齐到 `group_size` 的整数倍，
   补齐的部分**也要占字节**。而 `bytes._quantized_bytes` 用的是未补齐的
   `n_params`。OLMoE 的形状恰好都整除（2048 / 1024 / 50304 对 32/64/128），
   所以差为 0 —— 但这是**测出来的**，不是假设的。
2. **实数 vs 账**：`realized_bytes()` 把真正存下来的张量加总，
   与 `bytes._quantized_bytes` 的预测对比。项目的 `[W]` 全部建立在那个公式上，
   这是它第一次被对照。

## 与 grouped MoE 路径的关系（不要混）

`moe_static.replace_moe_blocks` 会把 experts 融合成 `gate_up`/`down` **并删掉
`experts` ModuleList**。所以：

- `EXP-p1-01`（精度）**不需要**动 MoE 路径 —— 纯 eager + `QuantLinear` 就够；
- 让 grouped kernel 直接吃 packed codes 是 `EXP-p2-01`（E2）的事。
两者顺序不能反：先量化再融合，融合会把量化结果反量化掉。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ★ 必须在本模块顶层拿到 torch。原因不是懒：`QuantLinear` 要**继承 nn.Module**，
#   而基类必须在类定义时存在。用 `__new__` 去动态造基类（第一版就是这么写的）
#   会绕过 nn.Module 的 `__init_subclass__`，_buffers/_modules 的初始化顺序变脆。
#
#   代价：没有 torch 的机器 import 不了本模块。所以 `lemoq/quant/__init__.py`
#   **刻意不 import 它** —— 纯 numpy 的 oracle 路径仍然可用。
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401
except Exception as exc:  # noqa: BLE001
    raise ImportError(
        "lemoq.quant.linear 需要 torch（CPU 版即可）。"
        "只需要 RTN/oracle 的话请用 lemoq.quant.rtn 或 lemoq.quant.numpy_oracle。"
    ) from exc

__all__ = [
    "QuantLinear",
    "SchemeSpec",
    "TARGET_RULES",
    "classify_linear",
    "parse_scheme",
    "quantize_model",
    "realized_bytes",
]


# scheme 名字 -> (weight bits, activation bits 或 None)
#
# ★ a16 / a8 的区别是**真的**：a8 会把激活也量化，测出来的精度差异来自两处。
#   报告时必须分开写，不能笼统说"W8"。
_SCHEMES = {
    "w8a8": (8, 8),
    "w8a16": (8, None),
    "w4a16": (4, None),
    "fp16": (None, None),
}


def parse_scheme(name: str) -> tuple[int | None, int | None]:
    """`"w4a16"` -> `(4, None)`；`"w8a8"` -> `(8, 8)`；`"fp16"` -> `(None, None)`。"""
    key = str(name).lower()
    if key not in _SCHEMES:
        raise ValueError(
            "未知 scheme %r；已实现：%s（兜底阶梯见 lemoq.quant.REMEDIATION_LADDER）"
            % (name, sorted(_SCHEMES))
        )
    return _SCHEMES[key]


@dataclass(frozen=True)
class SchemeSpec:
    """一个实验臂的完整量化配置。"""

    name: str
    weight_bits: int | None
    act_bits: int | None
    group_size: int | None
    targets: tuple[str, ...] = ("attention", "experts", "router")
    keep_lm_head_fp16: bool = True
    symmetric: bool = True
    scheme: str = "rtn"

    @classmethod
    def from_flags(
        cls,
        name: str,
        group_size: int,
        *,
        targets: tuple[str, ...] | None = None,
        keep_lm_head_fp16: bool = True,
    ) -> "SchemeSpec":
        bits, act = parse_scheme(name)
        return cls(
            name=name,
            weight_bits=bits,
            act_bits=act,
            group_size=group_size,
            targets=targets or ("attention", "experts", "router"),
            keep_lm_head_fp16=keep_lm_head_fp16,
        )

    @property
    def enabled(self) -> bool:
        return self.weight_bits is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.name,
            "weight_bits": self.weight_bits,
            "act_bits": self.act_bits,
            "group_size": self.group_size,
            "targets": list(self.targets),
            "keep_lm_head_fp16": self.keep_lm_head_fp16,
            "symmetric": self.symmetric,
            "rtn_scheme": self.scheme,
            "activation_quantization": (
                "fake-quant（quantize 后立刻 dequantize）：只测数值影响，不是 int8 GEMM"
                if self.act_bits is not None
                else "none"
            ),
        }


# 模块名 -> `bytes.weight_budget` 用的 target key。
# 顺序有意义：先匹配更具体的。
TARGET_RULES: tuple[tuple[str, str], ...] = (
    (r"^lm_head$", "lm_head"),
    (r"\.self_attn\.", "attention"),
    (r"\.experts\.\d+\.", "experts"),
    (r"\.mlp\.gate$", "router"),
    (r"\.shared_expert", "shared"),
)


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(module_name: str) -> int | None:
    """`model.layers.7.mlp...` -> 7；顶层（`lm_head`）-> None。"""
    match = _LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def classify_linear(module_name: str) -> str | None:
    """把一个 `nn.Linear` 的**全名**归到 `[W]` 的某一项，或 None（不量化）。

    名字来自 `model.named_modules()`，例如
    `model.layers.3.mlp.experts.17.gate_proj` / `model.layers.0.self_attn.q_proj` /
    `lm_head`。归不到类的返回 None —— **不猜**，宁可保持 fp16 并在报告里列出来。
    """
    for pattern, target in TARGET_RULES:
        if re.search(pattern, module_name):
            return target
    return None


class QuantLinear(nn.Module):
    """一个被量化的 `nn.Linear`（drop-in 替换，只有 forward 与权重表示不同）。

    权重放在 **buffer** 里而不是 Parameter：packed codes 是整数、scale/zero 是 fp16，
    都不该被优化器或 `requires_grad` 逻辑碰到。这样 `.to(device)` / `.eval()` /
    `state_dict()` 都照常工作。
    """

    def __init__(
        self,
        linear: Any,
        spec: SchemeSpec,
        *,
        name: str = "",
    ) -> None:
        from lemoq.quant import api as qapi  # noqa: PLC0415
        from lemoq.quant import pack as qpack  # noqa: PLC0415

        super().__init__()
        if spec.weight_bits is None:
            raise ValueError("SchemeSpec.weight_bits 为 None：不该构造 QuantLinear")
        self.spec = spec
        self.name = name
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        weight = linear.weight.detach()

        q = qapi.quantize_linear(
            weight,
            bits=int(spec.weight_bits),
            group_size=int(spec.group_size or 0),
            scheme=spec.scheme,
            symmetric=spec.symmetric,
        )
        # ★ 真的打包，而不是留着 int32 codes。这样 realized_bytes() 才是**测量**。
        #
        # `api.QuantizedLinear.codes` 是**未补齐**的 `[:, :in]` 切片，而打包必须按
        # 补齐后的 (out, padded_in) 来做 —— 直接用 q.codes 在 `in % group_size != 0`
        # 时会 reshape 失败（实测：shape [32,32] invalid for input of size 512）。
        # 复用 api 自己的 `_repad`，保证与 `dequantize_reference` 用同一套补齐规则。
        codes_padded = qapi._repad(q)
        if int(spec.weight_bits) == 4:
            packed = qpack.pack_4bit(codes_padded)
        elif int(spec.weight_bits) == 8:
            packed = codes_padded.to(torch.int8)
        else:
            raise ValueError("只支持 4 / 8 bit 权重")

        self.register_buffer("packed", packed.contiguous(), persistent=True)
        # scale / zero 存成 fp16 —— 与 bytes.QuantSpec.scale_dtype_bytes=2 的记账一致。
        # 存 fp32 会让实数比账大 2 倍，那是"账没有对照过"才看不出来的错。
        self.register_buffer("scale", q.scale.contiguous().half(), persistent=True)
        # ★ 对称量化**不需要** zero（rtn 里 zero 恒为 0）。第一版无条件存了一份，
        #   于是实数比 `[W]` 的账多出 `out * n_groups * 2` 字节 —— 实测 (2048,2048),
        #   group=32 时多 262,144 B（正好是 2048*64*2）。既白占显存，也让
        #   "实数 vs 账"这条核对永远对不上，从而失去意义。
        if spec.symmetric:
            self.register_buffer("zero", None)
        else:
            self.register_buffer("zero", q.zero.contiguous().half(), persistent=True)
        self.register_buffer(
            "bias",
            (linear.bias.detach().clone() if linear.bias is not None else None),
            persistent=True,
        )
        self.weight_bits = int(spec.weight_bits)
        self.group_size = int(spec.group_size or 0)
        if self.weight_bits == 4 and self.group_size % 2 != 0:
            raise ValueError("4-bit 打包要求 group_size 为偶数，得到 %d" % self.group_size)
        self.padded_in = int(q.meta["padded_in"])
        self.n_groups = int(q.meta["n_groups"])

    # ---------- 反量化（separate 臂的核心动作） ----------
    def dequantized_weight(self, dtype: Any = None) -> Any:
        """还原成 (out, in) 的浮点权重。**每次 forward 都会调用它** —— 那就是 separate 臂。

        ⚠️ 与 `api.dequantize_reference` **不是逐位相同**：那个参考用 fp32 scale，
        这里按 `bytes.QuantSpec.scale_dtype_bytes=2` 存 fp16。差异量级是 fp16 的
        scale 舍入（相对 ~1e-3），是**刻意的**：存 fp32 会让实数比 `[W]` 的账大一倍。
        要逐位对齐就得让参考也用 fp16 scale（见 tests/test_quant_linear.py）。
        """
        from lemoq.quant import pack as qpack  # noqa: PLC0415

        if self.weight_bits == 4:
            codes = qpack.unpack_4bit(self.packed, signed=True).to(torch.float32)
        else:
            codes = self.packed.to(torch.float32)
        codes = codes.reshape(self.out_features, self.n_groups, self.group_size)
        scale = self.scale.to(torch.float32).reshape(self.out_features, self.n_groups, 1)
        if self.spec.symmetric:
            w = codes * scale
        else:
            zero = self.zero.to(torch.float32).reshape(self.out_features, self.n_groups, 1)
            w = (codes - zero) * scale
        w = w.reshape(self.out_features, self.padded_in)[:, : self.in_features]
        return w.to(dtype) if dtype is not None else w

    # ---------- 激活伪量化（w8a8） ----------
    def _fake_quant_activation(self, x: Any) -> Any:
        """Per-token 对称 int8 伪量化。返回**反量化后**的浮点张量。

        ★ 这是 fake-quant：它复现数值影响，不产生 int8 计算。
        真的 int8 GEMM 属于 Phase 2；把它当成已完成的性能优化是错的。
        """
        if self.spec.act_bits is None:
            return x
        dt = x.dtype
        xf = x.float()
        qmax = float(2 ** (int(self.spec.act_bits) - 1) - 1)  # 127 for int8
        absmax = xf.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        q = (xf / scale).round().clamp(-qmax - 1, qmax)
        return (q * scale).to(dt)

    def forward(self, x: Any) -> Any:
        xq = self._fake_quant_activation(x)
        w = self.dequantized_weight(dtype=xq.dtype)
        return F.linear(xq, w, self.bias.to(xq.dtype) if self.bias is not None else None)

    def extra_repr(self) -> str:
        return "%d->%d %s group=%d" % (
            self.in_features,
            self.out_features,
            self.spec.name,
            self.group_size,
        )


@dataclass
class QuantizeReport:
    """量化后必须一起报的量。缺任何一项，`[W]` 的账就不可核对。"""

    spec: dict[str, Any]
    replaced: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    realized_by_target: dict[str, dict[str, int]] = field(default_factory=dict)
    padding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        total = sum(v["total_bytes"] for v in self.realized_by_target.values())
        return {
            "spec": self.spec,
            "replaced_per_target": self.replaced,
            "realized_bytes_total": total,
            "realized_bytes_by_target": self.realized_by_target,
            "padding": self.padding,
            "skipped_linear_count": len(self.skipped),
            "skipped_linear_examples": self.skipped[:8],
        }


def realized_bytes(module: Any) -> dict[str, int]:
    """真正存下来的字节：packed codes + scale + zero（+ bias）。"""
    packed = int(module.packed.numel() * module.packed.element_size())
    scale = int(module.scale.numel() * module.scale.element_size())
    zero = 0 if module.zero is None else int(module.zero.numel() * module.zero.element_size())
    bias = 0 if module.bias is None else int(module.bias.numel() * module.bias.element_size())
    return {
        "packed_bytes": packed,
        "scale_bytes": scale,
        "zero_bytes": zero,
        "bias_bytes": bias,
        "total_bytes": packed + scale + zero + bias,
    }


def quantize_model(
    model: Any,
    spec: SchemeSpec,
    *,
    verbose: bool = False,
    layers: Any = None,
    names: Any = None,
) -> tuple[Any, QuantizeReport]:
    """把模型里的 `nn.Linear` 就地换成 `QuantLinear`（按 `spec.targets` 选择）。

    返回 `(model, report)`。未被任何规则归类的 Linear 保持 fp16，并在
    `report.skipped` 里逐个列出 —— 不静默跳过。

    `layers`：只量化这些层号（`None` = 全部）。存在的理由不是省事，而是
    **归因**：`EXP-p1-01` 的结论是"整模型上 W8/W4 过不了门"，但那没有回答
    "误差是在哪一层放大的"。把量化限制在 `layers=[0]` 就能把"哪一层"变成
    受控变量（`REMEDIATION_LADDER` 的第 4 档也需要它）。

    `names`：更一般的显式白名单（完整模块名集合）。`EXP-p1-02` 用它做**逐张量类型**
    归因（只量化 q_proj / v_proj ...）—— 实测这个区分极其重要：W8 量级下扰动
    v_proj 造成的 KL 比扰动 q_proj 大三个数量级。两个限制都写进 report。
    """
    report = QuantizeReport(spec=spec.to_dict())
    if not spec.enabled:
        return model, report
    layer_filter = None if layers is None else {int(x) for x in layers}
    name_filter = None if names is None else set(names)
    report.spec["layers"] = None if layer_filter is None else sorted(layer_filter)
    report.spec["names"] = None if name_filter is None else sorted(name_filter)

    # ★ 只收集**名字**（字符串），不要 `list(model.named_modules())`。
    #
    #   `named_modules()` 返回的是模块对象，把它们物化进一个 list 会让**全部**
    #   原始 fp16 权重在整轮循环期间都无法释放 —— 于是峰值变成
    #   "fp16 全部 + 量化后全部"（OLMoE 上约 13.8 + 5.5 = 19.3 GB），而不是
    #   "未处理的 fp16 + 已处理的量化"（滑动窗口，峰值≈13.8 GB）。
    #   实测环境里 GPU 已被另一个进程占了 13.8 GB，这 5 GB 的差就是能不能跑的区别。
    #
    #   也不能直接遍历 `named_modules()` 生成器 + 就地替换：那会在迭代中改
    #   模块树（RuntimeError）。收集名字两全。
    candidates: list[str] = [
        name for name, module in model.named_modules() if name and isinstance(module, nn.Linear)
    ]
    for module_name in candidates:
        module = model.get_submodule(module_name)
        if not isinstance(module, nn.Linear):
            continue
        target = classify_linear(module_name)
        if target is None:
            report.skipped.append(module_name)
            continue
        if target not in spec.targets:
            report.skipped.append("%s (target=%s 未在 spec.targets 内)" % (module_name, target))
            continue
        if target == "lm_head" and spec.keep_lm_head_fp16:
            report.skipped.append("%s (keep_lm_head_fp16=True)" % module_name)
            continue
        if name_filter is not None and module_name not in name_filter:
            report.skipped.append("%s (不在 names= 白名单内)" % module_name)
            continue
        if layer_filter is not None and _layer_index(module_name) not in layer_filter:
            report.skipped.append(
                "%s (layer=%s 不在 layers=%s 内)"
                % (module_name, _layer_index(module_name), sorted(layer_filter))
            )
            continue

        parent_name, _, leaf = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        ql = QuantLinear(module, spec, name=module_name)
        setattr(parent, leaf, ql)

        report.replaced[target] = report.replaced.get(target, 0) + 1
        bucket = report.realized_by_target.setdefault(target, {})
        for key, value in realized_bytes(ql).items():
            bucket[key] = bucket.get(key, 0) + int(value)
        bucket["n_linear"] = bucket.get("n_linear", 0) + 1
        bucket["n_params"] = bucket.get("n_params", 0) + ql.out_features * ql.in_features
        bucket["n_params_padded"] = (
            bucket.get("n_params_padded", 0) + ql.out_features * ql.padded_in
        )
        if verbose:
            print("  quantized %s" % module_name)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    padded = sum(v["n_params_padded"] for v in report.realized_by_target.values())
    unpadded = sum(v["n_params"] for v in report.realized_by_target.values())
    report.padding = {
        "n_params_padded": padded,
        "n_params_unpadded": unpadded,
        "inflation_pct": round(100.0 * (padded - unpadded) / unpadded, 4) if unpadded else 0.0,
        "note": (
            "padding 的字节也是真占的；bytes._quantized_bytes 用的是未补齐的 n_params。"
            "差不为 0 时 `[W]` 会低估，必须一起报。"
        ),
    }
    return model, report
