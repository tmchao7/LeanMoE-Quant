"""moe_grouped — 无 gather 的 MoE decode 前向（Phase 2 的第一步）。

## 它要解决的问题：gather 税

`moe_static._decode_forward` 用"固定形状"换来了 CUDA Graph 的可行性，代价是一次
**真实的副本拷贝**（`gate_up[ids]`）。OLMoE fp16、batch=1 下每 token：

    理想（只读 8 个激活专家）      1.6106 GB
    gather 读 + 写副本 + bmm 再读  4.8318 GB
    税                             3.2212 GB  ->  2.066 ms @1559.2 GB/s

实测单 token 5.2547 ms 里约 **39%** 是这个税（逐项核算见
`benchmarks/attribution.py::expert_path_accounting`，有离线回归测试钉住）。

## 消掉的办法：kernel 内由 expert id 算指针

    out[i, n] = sum_k W[ids[i], n, k] * x[k]

expert id 只参与**指针算术**，不参与控制流 —— 所以没有数据依赖的形状，
CUDA Graph 依然安全。副本消失了，税也就消失了。

## 与 gather 路径的数值关系（**必须说清楚**）

两条路径**不再逐位相同**，而且差异的方向是分组路径更准：

    gather 路径   gate/up 落成 fp16 -> 激活 -> down 落成 fp16 -> 乘 top_w -> fp16 求和
    本模块        gate/up 保持 fp32 -> 激活 -> down 在 fp32 里累加并吃掉 top_k 维

所以：

- **K1 门（kernel 级）必须比 fp64 oracle**，不能比 gather 路径。比 gather 路径
  只会量到"谁更准"，而那不是 kernel 正确性。
- **K2 门（端到端）仍与同路径 eager 比**，因为两边都跑这个 kernel，
  确定性一致 → 仍然要求 KL 恰好为 0。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PARITY_GATE",
    "GROUPED_CONFIGS",
    "grouped_available",
    "grouped_gate_up_act_reference",
    "grouped_down_sum_reference",
    "static_moe_forward_grouped_reference",
    "grouped_moe_forward",
    "static_moe_forward_grouped",
    "warm",
]


# ---------------------------------------------------------------------------
# ★ 容差在**上机之前**冻结，与测得的数字无关。
#
# K1 是自校准的：门不写一个魔数，而是拿 **cuBLAS/fp16 路径自己的误差**当基准，
# 要求手写 kernel 不比它差超过 `tol_slack`。理由：fp16 输入 + fp32 累加下，
# 归约顺序本来就决定了 1e-3 量级的差异；"绝对等于参考"是不可能满足的要求，
# 而"不许比基线更差"是能检验、且不会随观察结果调整的。
#
# 放宽这两个数必须单独成一个 commit 并写明理由（见 AGENTS.md 规则 2）。
# ---------------------------------------------------------------------------
PARITY_GATE = {
    "tol_atol": 1e-3,
    "tol_rtol": 1e-3,
    "tol_slack": 1.5,
    "note": (
        "err_kernel <= err_baseline * tol_slack 且 allclose(atol=tol_atol, rtol=tol_rtol)；"
        "err_* 都是相对 fp64 oracle 的最大绝对误差"
    ),
}

# 默认配置。**刻意不用 @triton.autotune**：autotune 会真的跑 kernel 来计时，
# 在 CUDA Graph capture 期间等于往图里塞一堆计划外的 kernel。
# 参数敏感性本身是要报告的结果（见 `lemoq.kernels.autotune`），
# 但扫描在隔离微基准里做（`tools/bench_grouped_gemm.py`），不在模型里做。
GROUPED_CONFIGS: dict[str, dict[str, int]] = {
    # ★ 这两个不是猜的，是 `tools/bench_grouped_gemm.py --compare` 在 5090 上
    #   交错对照测出来的赢家（7 轮，逐轮排名 6/7，MAD 0.0004 ms）。
    #
    #   原来的默认（BLOCK_N=64/BLOCK_K=64/SPLIT_K=1 + down BLOCK_N=128/SK=1）
    #   是整张 36 点网格里**最慢的一角**：0.3868 vs 0.1204 ms/层 —— 差 3.2x。
    #   原因可以直接算出来：down_sum 的 grid 是 (cdiv(hidden, BLOCK_N), SPLIT_K)，
    #   BLOCK_N=128、SK=1 时只有 16 个 program 落在 170 个 SM 上（9% occupancy）。
    #   同一个 kernel 在默认配置下比 gather 基线慢 2.17x，调好之后快 1.48x。
    #
    #   ⚠️ 这条教训值得留着：**单配置的正确性+性能门不是合格的门**。
    #   第一版脚本只测默认配置，闸门判定 NULL、按计划就该放弃整条路线了。
    "gate_up_act": {"BLOCK_N": 32, "BLOCK_K": 128, "SPLIT_K": 1, "num_warps": 4},
    "down_sum": {"BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 4, "num_warps": 8},
    # SPLIT_K > 1 时这两个 kernel 各自需要一次归约，配置单独给
    "act_combine": {"BLOCK": 1024, "num_warps": 4},
    "out_reduce": {"BLOCK": 1024, "num_warps": 4},
}

# int32 指针算术的安全上限（见 `grouped_moe_forward` 的检查）
_MAX_OFFSET = 2**31


def _torch():
    import torch  # noqa: PLC0415

    return torch


def grouped_available() -> bool:
    """Triton 是否可用。Mac 上永远是 False —— 那时只有 oracle 能跑。"""
    try:
        from lemoq.kernels import _grouped_triton  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def _kernels():
    """惰性取 kernel 模块，失败时给出可操作的错误信息。"""
    try:
        import triton  # noqa: PLC0415

        from lemoq.kernels import _grouped_triton as k  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "grouped kernels 需要 triton（本机不可用：%s）。"
            "oracle 路径（*_reference）不需要 triton，可以在 CPU 上跑。" % type(exc).__name__
        ) from exc
    return triton, k


def _check_act_fn(act_fn: Any) -> None:
    """kernel 里融合的是 silu。别的东西必须**显式报错**，不能静默算错。"""
    import torch.nn.functional as F  # noqa: PLC0415

    if act_fn is None or act_fn is F.silu:
        return
    if getattr(act_fn, "__name__", "") == "silu":
        return
    raise NotImplementedError(
        "grouped kernels 融合了 silu；act_fn=%r 不是 silu。"
        "换激活需要改 kernel —— 这里拒绝静默用别的激活算出一个错的结果。" % (act_fn,)
    )


# ---------------------------------------------------------------------------
# 参考实现（纯 PyTorch fp32，**不需要 triton / GPU**）
# ---------------------------------------------------------------------------
def grouped_gate_up_act_reference(x: Any, gate_up: Any, ids: Any, *, top_k: int, ffn: int) -> Any:
    """`act[i, n] = silu(W_gate[e_i, n, :] . x_i) * (W_up[e_i, n, :] . x_i)`，fp32。

    `gate_up` 是 `(E, 2*ffn, H)` 连续张量（`fuse_experts` 的产物）。

    ★ `ids` 的顺序约定必须与 `moe_static._decode_forward` 一致：
    `ids = top_i.reshape(-1)` 是 **(token, slot) 行优先**，所以把每个 token 的
    hidden state 重复 `top_k` 次就能与 ids 对齐（`repeat_interleave`）。
    这个配对写错**不会报错**，只会算出一个看起来合理的结果 ——
    所以 `top_k` 是必填参数，T 由 `ids.numel() // top_k` 推出，不许猜。
    （第一版把 x 直接 flatten、假定 T=1，被 `test_grouped_and_gather_agree_on_output_shape` 抓到。）
    """
    import torch  # noqa: PLC0415

    if top_k <= 0:
        raise ValueError("top_k 必须 > 0")
    ids = ids.reshape(-1)
    if int(ids.numel()) % top_k != 0:
        raise ValueError("ids 的长度必须是 top_k 的整数倍")
    tokens = int(ids.numel()) // top_k
    xf = x.reshape(-1, x.shape[-1]).to(torch.float32)  # (T, H)
    if int(xf.shape[0]) != tokens:
        raise ValueError("x 有 %d 个 token，但 ids 推出 %d 个" % (int(xf.shape[0]), tokens))
    xr = xf.repeat_interleave(top_k, dim=0)  # (T*K, H) —— 与 ids 一一对应
    w = gate_up[ids].to(torch.float32)  # (T*K, 2F, H)
    gate = (w[:, :ffn, :] @ xr.unsqueeze(-1)).squeeze(-1)  # (T*K, F)
    up = (w[:, ffn:, :] @ xr.unsqueeze(-1)).squeeze(-1)  # (T*K, F)
    return gate * torch.sigmoid(gate) * up


def grouped_down_sum_reference(act: Any, down: Any, ids: Any, top_w: Any, *, top_k: int) -> Any:
    """`out[t, n] = sum_i top_w[t, i] * (W_down[e_{t,i}, n, :] . act[t, i, :])`，fp32。

    返回 `(T, H)`。对比 `_decode_forward`：它把 `(T*K, H)` 落成 fp16、乘上
    routing 权重、再 `reshape(T, K, H).sum(1)`。这里在 fp32 里做同一件事，
    所以两边**数学等价但不逐位相同** —— 这正是 K1 门要比 fp64 oracle 的原因。
    """
    import torch  # noqa: PLC0415

    ids = ids.reshape(-1)
    if top_k <= 0 or int(ids.numel()) % top_k != 0:
        raise ValueError("ids 的长度必须是 top_k 的整数倍")
    tokens = int(ids.numel()) // top_k
    af = act.to(torch.float32)  # (T*K, F)
    w = down[ids].to(torch.float32)  # (T*K, H, F)
    hidden = int(w.shape[1])
    y = torch.matmul(w, af.unsqueeze(-1)).squeeze(-1)  # (T*K, H)
    y = y * top_w.reshape(-1, 1).to(torch.float32)  # (T*K, H)
    return y.reshape(tokens, top_k, hidden).sum(dim=1)  # (T, H)


def static_moe_forward_grouped_reference(
    hidden_states: Any,
    router_weight: Any,
    fused: Any,
    *,
    top_k: int,
    norm_topk_prob: bool = False,
    act_fn: Any = None,
) -> tuple[Any, Any]:
    """整条 grouped 路径的 fp32 oracle，返回与 `static_moe_forward` 相同的 `(out, logits)`。

    路由**复用** `moe_static._route` —— 这样两边的 top-k 选择逐位相同，
    任何数值差异都只能来自专家计算部分，不会和路由差异混在一起。
    """
    from lemoq.kernels.moe_static import _route  # noqa: PLC0415

    shape = tuple(hidden_states.shape)
    x = hidden_states.reshape(-1, shape[-1])
    logits, top_w, top_i = _route(x, router_weight, top_k=top_k, norm_topk_prob=norm_topk_prob)
    ffn = int(fused.gate_up.shape[1]) // 2
    ids = top_i.reshape(-1)
    act = grouped_gate_up_act_reference(x, fused.gate_up, ids, top_k=top_k, ffn=ffn)
    out = grouped_down_sum_reference(act, fused.down, ids, top_w.reshape(-1), top_k=top_k)
    return out.to(x.dtype).reshape(*shape), logits


# ---------------------------------------------------------------------------
# GPU 路径
# ---------------------------------------------------------------------------
def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def merged_config(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """`GROUPED_CONFIGS` 的一张深拷贝，按调用点覆盖合并。"""
    cfg = {name: dict(values) for name, values in GROUPED_CONFIGS.items()}
    for key, value in (config or {}).items():
        cfg.setdefault(key, {}).update(value)
    return cfg


def launch_plan(fused: Any, *, top_k: int, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """把 kernel 的 grid / 缓冲区形状**算清楚**，不碰 GPU。

    ★ 为什么把这段算术单独拎出来（纯 Python、可离线测试）：

    CUDA Graph 只约束"形状静态"，不约束"形状对不对"。grid 少算一维、
    部分和缓冲区写错大小，都不会报错 —— 只会**静默算错**，而这类错误在
    GPU 上要用 KL 门才能发现，代价是加载 13 GB 模型 + 几分钟。
    这里把它变成 Mac 上可以手算核对的东西。

    `kernel_count` 是这条路径的 kernel 数：`SPLIT_K == 1` 时是 **2**
    （gate_up+act、down+加权求和），否则 4。对照：gather 路径每层约 10 个，
    其中 2 个是纯浪费的拷贝。
    """
    gate_up = fused.gate_up
    ffn = int(gate_up.shape[1]) // 2
    hidden = int(gate_up.shape[2])
    cfg = merged_config(config)

    ga = cfg["gate_up_act"]
    gsplit = int(ga["SPLIT_K"])
    gbn = int(ga["BLOCK_N"])
    # 注意 grid 的第 0 维是**槽位数（top_k）**，不是 hidden —— 每 token 恰好
    # top_k 个 (token, expert) 对，形状与数据无关，这就是它能进图的原因。
    gate_grid = (top_k, _cdiv(ffn, gbn), gsplit)

    ds = cfg["down_sum"]
    dsplit = int(ds["SPLIT_K"])
    dbn = int(ds["BLOCK_N"])
    down_grid = (_cdiv(hidden, dbn), dsplit)

    plan: dict[str, Any] = {
        "hidden": hidden,
        "ffn": ffn,
        "top_k": top_k,
        "gate_up_act": {
            "grid": gate_grid,
            "BLOCK_N": gbn,
            "BLOCK_K": int(ga["BLOCK_K"]),
            "SPLIT_K": gsplit,
            "num_warps": int(ga["num_warps"]),
            "act_shape": (top_k, ffn),
            "partial_shape": (gsplit, top_k, ffn) if gsplit > 1 else None,
        },
        "act_combine": None,
        "down_sum": {
            "grid": down_grid,
            "BLOCK_N": dbn,
            "BLOCK_K": int(ds["BLOCK_K"]),
            "SPLIT_K": dsplit,
            "num_warps": int(ds["num_warps"]),
            "out_shape": (hidden,),
            "partial_shape": (dsplit, hidden) if dsplit > 1 else None,
        },
        "out_reduce": None,
        "kernel_count": 2,
    }
    if gsplit > 1:
        cb = cfg["act_combine"]
        plan["act_combine"] = {
            "grid": (_cdiv(top_k * ffn, int(cb["BLOCK"])),),
            "BLOCK": int(cb["BLOCK"]),
            "num_warps": int(cb["num_warps"]),
        }
        plan["kernel_count"] += 1
    if dsplit > 1:
        orr = cfg["out_reduce"]
        plan["out_reduce"] = {
            "grid": (_cdiv(hidden, int(orr["BLOCK"])),),
            "BLOCK": int(orr["BLOCK"]),
            "num_warps": int(orr["num_warps"]),
        }
        plan["kernel_count"] += 1
    return plan


def grouped_moe_forward(
    x: Any,
    fused: Any,
    ids: Any,
    top_w: Any,
    *,
    config: dict[str, Any] | None = None,
    backend: str = "auto",
) -> Any:
    """`(H,)` hidden state + `(K,)` 专家 id + `(K,)` 路由权重 → `(H,)` 输出。

    单次调用 = 2 个 kernel（`SPLIT_K == 1`）或 4 个（见 `launch_plan`）。

    `backend="oracle"` 用纯 PyTorch 参考实现跑**同一条 host 管线**
    （同一个 `launch_plan`、同样的形状检查与输出契约）。这样 host 侧的形状/
    dtype/reshape 逻辑在没有 GPU 的机器上也能被测到；只有 kernel 本体需要在
    真机上过 K1 门。
    """
    torch = _torch()

    gate_up = fused.gate_up
    down = fused.down
    for name, tensor in (("gate_up", gate_up), ("down", down)):
        if not tensor.is_contiguous():
            raise ValueError("%s 必须连续（kernel 里的偏移按连续布局算）" % name)
        if tensor.numel() >= _MAX_OFFSET:
            raise ValueError("%s 的元素数超过 int32 偏移上限" % name)

    if backend not in ("auto", "triton", "oracle"):
        raise ValueError("unknown backend %r" % (backend,))
    use_oracle = backend == "oracle"
    if backend == "auto" and not grouped_available():
        raise RuntimeError(
            "grouped kernels 需要 triton；本机不可用。"
            "用 backend='oracle' 可以只跑 host 管线 + 参考实现。"
        )
    if not use_oracle:
        triton, k = _kernels()

    xf = x.reshape(-1).contiguous()
    if xf.dtype != gate_up.dtype:
        xf = xf.to(gate_up.dtype)
    idx = ids.reshape(-1).contiguous()
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    wsum = top_w.reshape(-1).contiguous()
    if wsum.dtype != gate_up.dtype:
        wsum = wsum.to(gate_up.dtype)

    # ★ 单 token 契约：这里 `top_k = len(ids)`，即 grid 第 0 维就是槽位数。
    #   T > 1 不会走到这里 —— `static_moe_forward` 按 token 数分派，prefill 永远
    #   走逐专家循环（grouped 在 T=1024 上会对每 token 重读一遍权重）。
    top_k = int(idx.numel())
    ffn = int(gate_up.shape[1]) // 2
    hidden = int(gate_up.shape[2])
    if int(xf.numel()) != hidden:
        raise ValueError(
            "grouped_moe_forward 只处理单 token：x 有 %d 个元素，hidden 是 %d"
            % (int(xf.numel()), hidden)
        )
    if int(wsum.numel()) != top_k:
        raise ValueError("top_w 的长度 %d 与槽位数 %d 不符" % (int(wsum.numel()), top_k))
    if int(down.shape[2]) != ffn:
        raise ValueError("down 的 ffn 维与 gate_up 不一致")

    plan = launch_plan(fused, top_k=top_k, config=config)
    dev = xf.device
    dt = gate_up.dtype

    # ---- gate_up + activation ----
    ga = plan["gate_up_act"]
    act = torch.empty(plan["gate_up_act"]["act_shape"], dtype=dt, device=dev)
    if use_oracle:
        act = grouped_gate_up_act_reference(xf, gate_up, idx, top_k=top_k, ffn=ffn).to(dt)
    else:
        split = int(ga["SPLIT_K"])
        if split == 1:
            part_g = act  # 占位指针：SPLIT_K == 1 时不会被写入
            part_u = act
        else:
            part_g = torch.empty(ga["partial_shape"], dtype=torch.float32, device=dev)
            part_u = torch.empty(ga["partial_shape"], dtype=torch.float32, device=dev)
        k._gate_up_act_kernel[ga["grid"]](
            xf,
            gate_up,
            idx,
            act,
            part_g,
            part_u,
            F=ffn,
            H=hidden,
            BLOCK_N=int(ga["BLOCK_N"]),
            BLOCK_K=int(ga["BLOCK_K"]),
            SPLIT_K=split,
            K_TOKENS=top_k,
            num_warps=int(ga["num_warps"]),
        )
        if plan["act_combine"] is not None:
            cb = plan["act_combine"]
            k._act_combine_kernel[cb["grid"]](
                part_g,
                part_u,
                act,
                F=ffn,
                K_TOKENS=top_k,
                SPLIT_K=split,
                BLOCK=int(cb["BLOCK"]),
                num_warps=int(cb["num_warps"]),
            )

    # ---- down + top_k 加权求和 ----
    ds = plan["down_sum"]
    out = torch.empty(plan["down_sum"]["out_shape"], dtype=dt, device=dev)
    if use_oracle:
        # 参考实现返回 (T, H)；本函数是单 token 契约，所以压回 (H,)
        out = grouped_down_sum_reference(act, down, idx, wsum, top_k=top_k).to(dt).reshape(hidden)
    else:
        dsplit = int(ds["SPLIT_K"])
        part = (
            out
            if dsplit == 1
            else torch.empty(ds["partial_shape"], dtype=torch.float32, device=dev)
        )
        k._down_sum_kernel[ds["grid"]](
            act,
            down,
            idx,
            wsum,
            out,
            part,
            H=hidden,
            F=ffn,
            K_TOKENS=top_k,
            BLOCK_N=int(ds["BLOCK_N"]),
            BLOCK_K=int(ds["BLOCK_K"]),
            SPLIT_K=dsplit,
            num_warps=int(ds["num_warps"]),
        )
        if plan["out_reduce"] is not None:
            orr = plan["out_reduce"]
            k._out_reduce_kernel[orr["grid"]](
                part,
                out,
                H=hidden,
                SPLIT_K=dsplit,
                BLOCK=int(orr["BLOCK"]),
                num_warps=int(orr["num_warps"]),
            )
    return out


def static_moe_forward_grouped(
    hidden_states: Any,
    router_weight: Any,
    fused: Any,
    *,
    top_k: int,
    norm_topk_prob: bool,
    act_fn: Any,
    config: dict[str, Any] | None = None,
) -> tuple[Any, Any]:
    """`OlmoeSparseMoeBlock.forward` 的 grouped 版本，契约与 `static_moe_forward` 相同。

    `config` 是 kernel 参数覆盖（见 `GROUPED_CONFIGS`）—— 只有参数扫描会传它，
    模型路径用默认值。
    """
    _check_act_fn(act_fn)
    from lemoq.kernels.moe_static import _route  # noqa: PLC0415

    shape = tuple(hidden_states.shape)
    hidden = shape[-1]
    tokens = 1
    for s in shape[:-1]:
        tokens *= s
    x = hidden_states.reshape(tokens, hidden)
    logits, top_w, top_i = _route(x, router_weight, top_k=top_k, norm_topk_prob=norm_topk_prob)
    out = grouped_moe_forward(
        x,
        fused,
        top_i.reshape(-1),
        top_w.reshape(-1),
        config=config,
    )
    return out.reshape(*shape), logits


def warm(
    fused: Any,
    *,
    hidden: int,
    top_k: int,
    dtype: Any = None,
    device: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在 capture **之前**把 kernel 编译好。

    ★ 为什么必须有这一步：Triton 的首次调用会 JIT 编译并加载 cubin。
    在 CUDA Graph capture 期间做这件事会直接报错（而且那时的报错信息
    通常指向别处）。所以 session 的 `capture()` 之前必须调用它。

    返回被 warm 的配置，便于记录进结果文件。
    """
    torch = _torch()
    dev = device if device is not None else fused.gate_up.device
    dt = dtype or fused.gate_up.dtype
    x = torch.zeros(hidden, dtype=dt, device=dev)
    ids = torch.zeros(top_k, dtype=torch.int64, device=dev)
    wsum = torch.full((top_k,), 1.0 / max(top_k, 1), dtype=dt, device=dev)
    grouped_moe_forward(x, fused, ids, wsum, config=config)
    if getattr(dev, "type", None) == "cuda":
        torch.cuda.synchronize()
    return {"hidden": hidden, "top_k": top_k, "dtype": str(dt), "device": str(dev)}
