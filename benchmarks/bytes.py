"""bytes — 每-token 字节预算 `[W]` 与 `[T]` 的计算（**纯 Python，无 torch 依赖**）。

口径定义见 docs/reference/METRICS.md §3.2。这是全项目的物理口径，所以：

    - 本模块**不 import torch**，因此可以在 Mac 的最小环境、CI 或任何地方跑；
    - 每个分项都带 provenance（参数量/公式），因为"别人无法复核的数字等于没测"；
    - 用**已核实的锚点**做回归：OLMoE-1B-7B-0924 的 `[W]` 必须等于 2,357,723,136 B
      （见 tests/test_bytes.py）。任何改动若打破它，必须解释。

两个分母：

    [W] 权重字节，与 context 无关 —— 量化的直接杠杆
    [T] = [W] + KV 读(ctx)        —— 决定实测单步延迟
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


@dataclass
class ModelSpec:
    """Everything the byte budget needs. Derived from a HF `config.json`."""

    name: str
    layers: int
    hidden: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    expert_ffn_dim: int
    vocab_size: int
    tie_word_embeddings: bool
    # 可选：某些 MoE（DeepSeek 系）有 shared expert；OLMoE 没有
    shared_expert_ffn_dim: int = 0

    # ---------- 构造 ----------
    @classmethod
    def from_hf_config(cls, cfg: dict[str, Any], name: str = "") -> "ModelSpec":
        hidden = int(cfg["hidden_size"])
        heads = int(cfg.get("num_attention_heads", 1))
        head_dim = int(cfg.get("head_dim") or hidden // max(heads, 1))
        return cls(
            name=name or str(cfg.get("_name_or_path", "unknown")),
            layers=int(cfg["num_hidden_layers"]),
            hidden=hidden,
            num_attention_heads=heads,
            num_key_value_heads=int(cfg.get("num_key_value_heads", heads)),
            head_dim=head_dim,
            num_experts=int(cfg.get("num_experts") or 0),
            top_k=int(cfg.get("num_experts_per_tok") or 0),
            expert_ffn_dim=int(cfg.get("intermediate_size") or 0),
            vocab_size=int(cfg.get("vocab_size") or 0),
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
            shared_expert_ffn_dim=int(cfg.get("shared_expert_intermediate_size") or 0),
        )

    # ---------- 参数量 ----------
    @property
    def is_moe(self) -> bool:
        return self.num_experts > 1

    def attn_params_per_layer(self) -> int:
        """q/k/v/o 投影参数量（无 bias）。"""
        q = self.hidden * (self.num_attention_heads * self.head_dim)
        kv = self.hidden * (self.num_key_value_heads * self.head_dim)
        o = (self.num_attention_heads * self.head_dim) * self.hidden
        return q + 2 * kv + o

    def router_params_per_layer(self) -> int:
        return self.hidden * self.num_experts if self.is_moe else 0

    def ffn_params_per_layer_activated(self) -> int:
        """每 token **实际流经**的 FFN 参数量（MoE 只算 top-k）。"""
        if self.is_moe:
            routed = self.top_k * 3 * self.hidden * self.expert_ffn_dim
            shared = (
                3 * self.hidden * self.shared_expert_ffn_dim if self.shared_expert_ffn_dim else 0
            )
            return routed + shared
        return 3 * self.hidden * self.expert_ffn_dim

    def ffn_params_per_layer_total(self) -> int:
        """该层 FFN 的**全部**参数量（MoE 是 num_experts 倍）。"""
        if self.is_moe:
            routed = self.num_experts * 3 * self.hidden * self.expert_ffn_dim
            shared = (
                3 * self.hidden * self.shared_expert_ffn_dim if self.shared_expert_ffn_dim else 0
            )
            return routed + shared
        return 3 * self.hidden * self.expert_ffn_dim

    def lm_head_params(self) -> int:
        """untied 时才需要每 token 完整读取。tied 时为 0（用 embedding 行，可忽略）。"""
        return 0 if self.tie_word_embeddings else self.vocab_size * self.hidden

    def total_params(self) -> int:
        """全量参数量（含 embedding 表一次）。"""
        embed = self.vocab_size * self.hidden
        per_layer = (
            self.attn_params_per_layer()
            + self.router_params_per_layer()
            + self.ffn_params_per_layer_total()
        )
        return self.layers * per_layer + embed + (0 if self.tie_word_embeddings else embed)

    def activated_params_excl_embedding(self) -> int:
        """每 token 激活的参数量（不含 embedding 表；含 untied lm_head）。"""
        per_layer = (
            self.attn_params_per_layer()
            + self.router_params_per_layer()
            + self.ffn_params_per_layer_activated()
        )
        return self.layers * per_layer + self.lm_head_params()


@dataclass
class QuantSpec:
    """量化配置。`None` 表示该部分保持 fp16。"""

    bits: int | None = None
    group_size: int | None = None
    # 只量化这几类；未列出的保持 fp16
    targets: tuple[str, ...] = ("attention", "experts")
    keep_lm_head_fp16: bool = True
    scale_dtype_bytes: int = 2  # fp16 scale

    @property
    def enabled(self) -> bool:
        return self.bits is not None


def _bytes_for_params(n_params: int, dtype_bytes: int) -> int:
    return n_params * dtype_bytes


def _quantized_bytes(
    n_params: int,
    bits: int,
    group_size: int | None,
    scale_dtype_bytes: int,
    *,
    in_features: int | None = None,
) -> int:
    """Packed weights + per-group scales.

    ★ 组数沿**输入维、每行一组**数：`rows × ceil(in / group_size)`。
    旧写法 `ceil(n_params / group_size)` 只在每个矩阵的输入维都被 `group_size`
    整除时等于它 —— 否则**低估** scale 开销（实测 `out=8, in=300, g=128`：
    真实 24 组，旧写法 19 组）。

    所以 `in_features` 必须传。`in_features=None` 保留旧近似，只为兼容不了解
    形状的调用点，且**不**假装它总是对的。
    """
    packed = n_params * bits // 8
    if group_size is None or group_size <= 0:
        return packed
    if in_features is None:
        n_groups = -(-n_params // group_size)  # ceil（近似口径）
    else:
        in_features = int(in_features)
        if in_features <= 0:
            raise ValueError("in_features 必须 > 0，得到 %r" % (in_features,))
        if n_params % in_features != 0:
            raise ValueError(
                "n_params=%d 不能被 in_features=%d 整除：组数按行算，形状不自洽就没法数"
                % (n_params, in_features)
            )
        n_groups = (n_params // in_features) * -(-in_features // group_size)
    return packed + n_groups * scale_dtype_bytes


def _item(name: str, params: int, byte_count: int, formula: str) -> dict[str, Any]:
    return {"name": name, "params": params, "bytes": byte_count, "formula": formula}


def weight_budget(
    spec: ModelSpec, quant: QuantSpec | None = None, dtype_bytes: int = 2
) -> dict[str, Any]:
    """Compute `[W]`: per-token weight bytes, with a per-item breakdown.

    MoE: only top-k experts are streamed per token — that is the whole point.
    """
    quant = quant or QuantSpec()
    items: list[dict[str, Any]] = []

    # ★ 按 **(rows, in_features)** 逐块算，而不是只给一个 n_params：
    #   per-group scale 是"沿输入维、每行一组"，组数 = rows × ceil(in / group_size)。
    #   只知道总参数量时这个数算不出来（旧写法 `ceil(n_params/gs)` 在非整除时低估），
    #   所以每个 bucket 必须把它由哪些 (rows, in) 组成说清楚。
    def maybe_quant(name: str, shapes: tuple[tuple[int, int], ...], dtype_b: int) -> int:
        params = sum(r * c for r, c in shapes)
        if quant.enabled and name in quant.targets:
            if name == "lm_head" and quant.keep_lm_head_fp16:
                return _bytes_for_params(params, dtype_b)
            return sum(
                _quantized_bytes(
                    rows * in_f,
                    int(quant.bits),
                    quant.group_size,
                    quant.scale_dtype_bytes,
                    in_features=in_f,
                )
                for rows, in_f in shapes
            )
        return _bytes_for_params(params, dtype_b)

    def params_of(shapes: tuple[tuple[int, int], ...]) -> int:
        return sum(r * c for r, c in shapes)

    # ★ o_proj 的输入维是 heads*head_dim（不是 hidden）—— 四块投影的 (rows, in)
    #   不一样，混成一个 n_params 会数错组数（改这里若算错，[W] 的 fp16 锚点
    #   2,357,723,136 会立刻不对，tests/test_bytes.py 钉着它）。
    q_rows = spec.layers * spec.num_attention_heads * spec.head_dim
    kv_rows = spec.layers * spec.num_key_value_heads * spec.head_dim
    attn_shapes = (
        (q_rows, spec.hidden),  # q_proj:  (heads*hd, hidden)
        (kv_rows, spec.hidden),  # k_proj
        (kv_rows, spec.hidden),  # v_proj
        (spec.layers * spec.hidden, spec.num_attention_heads * spec.head_dim),  # o_proj
    )
    items.append(
        _item(
            "attention",
            params_of(attn_shapes),
            maybe_quant("attention", attn_shapes, dtype_bytes),
            "layers x (q + 2*kv + o)",
        )
    )

    router = spec.layers * spec.router_params_per_layer()
    if router:
        items.append(
            _item(
                "router",
                router,
                maybe_quant(
                    "router", ((spec.layers * spec.num_experts, spec.hidden),), dtype_bytes
                ),
                "layers x hidden x num_experts",
            )
        )

    # ★ experts 必须拆成 gate_up（in = hidden）与 down（in = ffn）两块：
    #   它们的输入维不同，混成一个 n_params 就数不出正确的组数。
    mult = spec.top_k if spec.is_moe else 1
    gu_shapes = ((spec.layers * mult * 2 * spec.expert_ffn_dim, spec.hidden),)
    dn_shapes = ((spec.layers * mult * spec.hidden, spec.expert_ffn_dim),)
    experts = params_of(gu_shapes) + params_of(dn_shapes)
    items.append(
        _item(
            "experts(top-k)",
            experts,
            maybe_quant("experts", gu_shapes + dn_shapes, dtype_bytes),
            (
                "layers x top_k x 3 x hidden x expert_ffn_dim (MoE streams only top-k)"
                if spec.is_moe
                else "layers x 3 x hidden x ffn_dim"
            ),
        )
    )

    if spec.shared_expert_ffn_dim:
        sffn = spec.shared_expert_ffn_dim
        shared_shapes = (
            (spec.layers * 2 * sffn, spec.hidden),
            (spec.layers * spec.hidden, sffn),
        )
        items.append(
            _item(
                "shared_expert",
                params_of(shared_shapes),
                maybe_quant("shared", shared_shapes, dtype_bytes),
                "layers x 3 x hidden x shared_ffn_dim",
            )
        )

    lm = spec.lm_head_params()
    if lm:
        items.append(
            _item(
                "lm_head",
                lm,
                maybe_quant("lm_head", ((spec.vocab_size, spec.hidden),), dtype_bytes),
                "vocab x hidden (untied -> streamed every token)",
            )
        )
    else:
        items.append(_item("lm_head", 0, 0, "tied with embedding -> 0 (row gather ~4 KiB)"))

    total_params = sum(it["params"] for it in items)
    total_bytes = sum(it["bytes"] for it in items)
    for it in items:
        it["share"] = (it["bytes"] / total_bytes) if total_bytes else 0.0
        it["share_pct"] = round(100.0 * it["share"], 1)

    # embedding lookup: 1 row per token, negligible but stated for completeness
    embed_row_params = spec.hidden
    embed_row_bytes = embed_row_params * dtype_bytes
    embed_item = _item(
        "embedding_row",
        embed_row_params,
        embed_row_bytes,
        "hidden x dtype_bytes (one row gathered per token)",
    )
    embed_item["share"] = (embed_row_bytes / total_bytes) if total_bytes else 0.0
    embed_item["share_pct"] = round(100.0 * embed_item["share"], 4)
    embed_item["counted_in_total"] = False
    items.append(embed_item)

    return {
        "kind": "W",
        "definition": "per-token weight bytes, context-independent (quantization lever)",
        "items": items,
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 4),
        "dtype_bytes": dtype_bytes,
        "quant": {
            "bits": quant.bits,
            "group_size": quant.group_size,
            "targets": list(quant.targets),
            "keep_lm_head_fp16": quant.keep_lm_head_fp16,
        },
        "includes_embedding_row": True,
        "note": "总字节**不含** embedding 查表（每 token 只取 1 行，约 hidden x dtype_bytes）",
    }


def kv_budget(spec: ModelSpec, ctx: int, batch: int = 1, dtype_bytes: int = 2) -> dict[str, Any]:
    """KV cache: pool size AND per-token read (they are equal only at batch=1)."""
    per_token_per_seq = 2 * spec.layers * spec.num_key_value_heads * spec.head_dim * dtype_bytes
    read_bytes = per_token_per_seq * ctx * batch
    pool_bytes = read_bytes  # 单步 decode 读完整 context
    return {
        "kind": "KV",
        "formula": "2 x layers x num_kv_heads x head_dim x ctx x batch x dtype_bytes",
        "components": {
            "layers": spec.layers,
            "num_kv_heads": spec.num_key_value_heads,
            "head_dim": spec.head_dim,
            "ctx": ctx,
            "batch": batch,
            "dtype_bytes": dtype_bytes,
        },
        "per_token_per_seq_bytes": per_token_per_seq,
        "read_bytes": read_bytes,
        "pool_bytes": pool_bytes,
        "read_gib": round(read_bytes / 1024**3, 4),
    }


def byte_budget(
    spec: ModelSpec, ctx: int, batch: int = 1, dtype_bytes: int = 2, quant: QuantSpec | None = None
) -> dict[str, Any]:
    """Compute both `[W]` and `[T] = [W] + KV read(ctx)`.

    Both denominators are returned, because `docs/reference/METRICS.md` §3.2 requires reporting
    which one an attainment figure used.
    """
    w = weight_budget(spec, quant=quant, dtype_bytes=dtype_bytes)
    kv = kv_budget(spec, ctx=ctx, batch=batch, dtype_bytes=dtype_bytes)
    t_total = w["total_bytes"] + kv["read_bytes"]
    return {
        "W": w,
        "KV": kv,
        "T": {
            "kind": "T",
            "definition": "W + KV read(ctx) -- determines measured step latency",
            "total_bytes": t_total,
            "total_gib": round(t_total / 1024**3, 4),
            "kv_share_pct": round(100.0 * kv["read_bytes"] / t_total, 1) if t_total else 0.0,
        },
        "ctx": ctx,
        "batch": batch,
    }


def attainments(
    *, w_bytes: int, t_bytes: int, measured_ms: float, measured_bw_gbps: float
) -> dict[str, Any]:
    """Compute the two attainment figures, each with its denominator stated.

    Hard rule (METRICS §3.2 rule 5): an attainment must always carry ctx, which
    denominator was used, and the [W] breakdown. This function returns the numbers;
    the caller is responsible for persisting the context.
    """
    if measured_bw_gbps <= 0 or measured_ms <= 0:
        return {
            "status": "not_measurable",
            "reason": "need measured_bw_gbps > 0 and measured_ms > 0",
        }
    floor_w_ms = (w_bytes / (measured_bw_gbps * 1e9)) * 1000.0
    floor_t_ms = (t_bytes / (measured_bw_gbps * 1e9)) * 1000.0
    return {
        "status": "measured",
        "measured_bw_gbps": measured_bw_gbps,
        "floor_w_ms": round(floor_w_ms, 4),
        "floor_t_ms": round(floor_t_ms, 4),
        "quant_attainment_pct": round(100.0 * floor_w_ms / measured_ms, 1),
        "bw_attainment_pct": round(100.0 * floor_t_ms / measured_ms, 1),
        "denominator_note": (
            "quant_attainment uses [W] (excludes KV); bw_attainment uses [T] (includes KV) "
            "at the stated ctx. Never mix them."
        ),
    }
