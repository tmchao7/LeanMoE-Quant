"""rmsnorm — 自研融合 RMSNorm（**decode 形状**：每行 128 或 2048 个元素）。

## 为什么需要它（`EXP-p4c-02` §5.5）

HF 的 `OlmoeRMSNorm.forward` 每个 norm 要 **7 个 kernel**：

    to(float32) → pow(2) → mean(-1) → rsqrt/mul → to(fp16) → weight * (fp16)

OLMoE 每层 4 个 norm（q_norm / k_norm / input_layernorm / post_attention_layernorm）+
最后 1 个 = **65 个/步** ⇒ 约 455 kernel/步，占 elementwise/copy 长尾（0.90 ms）里的
≈0.43 ms（含 130 次 dtype cast = 0.185 ms）。

**torch 自带的 `F.rms_norm` 已经试过并被否证**（同记录 §5.5）：它保留同一个 `MeanOps`
reduce，却把"乘 weight"挪到 fp32 里做（`BinaryFunctor<float>` ×65 = 0.189 ms），
而 HF 是**先舍入到 fp16、再用 fp16 的 weight 乘**（0.066 ms）⇒ 整体慢 5%。

所以本 kernel 的契约里有一条**硬要求**：

    ★ 先算 `normed = (x * rstd).to(fp16)`，**再**在 fp16 里乘 `w`（w 也是 fp16）。
      不是在 fp32 里一次乘完再舍入 —— 那不是 HF 的语义，也正是 `F.rms_norm` 慢的原因。

## 形状与 launch

`R` = 每个 program 处理的行数，`BLOCK` = 行宽（2 的幂）。host 侧取
`BLOCK = next_power_of_2(H)`、`R = max(1, TARGET_ELEMS // BLOCK)`（TARGET_ELEMS = 2048），
于是两种真实形状各一个 program：

    H = 2048（layer norm，M=1）  → BLOCK=2048, R=1
    H = 128 （q/k norm，M=16）   → BLOCK=128,  R=16

行宽与 R 都是 host 侧常量 ⇒ 图安全（无 `@triton.autotune`、grid 是常量）。
"""

from __future__ import annotations

from typing import Any

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
    _TRITON_IMPORT_ERROR: str | None = None
except Exception as _exc:  # noqa: BLE001
    triton = None
    tl = None
    _HAS_TRITON = False
    _TRITON_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

__all__ = [
    "RMSNORM_CONFIG",
    "fused_available",
    "rms_norm_fused",
    "unavailable_reason",
    "warm",
]

# 每个 program 处理的目标元素数（源自行宽 2048 的实测形状；不 autotune，声明在这里）
TARGET_ELEMS = 2048

RMSNORM_CONFIG: dict[str, int] = {
    "num_warps": 4,
    "num_stages": 1,
}


def fused_available() -> bool:
    """本机能不能用（Mac 上永远 False，**不抛异常**）。"""
    if not _HAS_TRITON:
        return False
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def unavailable_reason() -> str | None:
    if _HAS_TRITON and _TRITON_IMPORT_ERROR is None:
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return "torch 不可用（%s）" % type(exc).__name__
        if not torch.cuda.is_available():
            return "本机没有可用的 CUDA 设备"
        return None
    return "triton 不可用（%s）" % (_TRITON_IMPORT_ERROR or "unknown")


def rms_norm_fused(
    x: Any,
    weight: Any,
    eps: float,
    *,
    out_dtype: Any = None,
    config: dict[str, Any] | None = None,
) -> Any:
    """`y = (x * rsqrt(mean(x^2) + eps)).to(fp16) * weight`，逐行归一化（沿最后一维）。

    契约（违反即**响亮报错**，不静默算错）：

        x        (..., H)  fp16，**连续**；H = 最后一维（128 / 2048）
        weight   (H,)      fp16，连续
        返回     与 x 同形状（`out_dtype=None` 时同 dtype）
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "fused RMSNorm 需要 triton（本机不可用：%s）" % (_TRITON_IMPORT_ERROR or "unknown")
        )
    import torch  # noqa: PLC0415

    if x.dim() < 1:
        raise ValueError("x 至少要有一维")
    if not x.is_contiguous():
        raise ValueError("x 必须是连续的（kernel 用行宽推 stride）")
    if not weight.is_contiguous():
        raise ValueError("weight 必须是连续的")
    if x.dtype != torch.float16:
        raise NotImplementedError("x 只支持 fp16，得到 %s" % x.dtype)
    if weight.dtype != torch.float16:
        raise NotImplementedError("weight 只支持 fp16，得到 %s" % weight.dtype)
    H = int(x.shape[-1])
    if int(weight.numel()) != H or weight.dim() != 1:
        raise ValueError("weight 必须是 (H,) 且 H=%d，得到 %r" % (H, tuple(weight.shape)))
    if H & (H - 1) != 0:
        raise NotImplementedError("H 必须是 2 的幂（本 kernel 用整块掩码），得到 %d" % H)
    if H > 8192:
        raise NotImplementedError("H=%d 超过本 kernel 的块上限 8192" % H)

    cfg = dict(RMSNORM_CONFIG)
    if config:
        cfg.update({k: int(v) for k, v in (config.get("rmsnorm") or config).items()})
    M = int(x.numel()) // H
    BLOCK = H
    R = max(1, TARGET_ELEMS // BLOCK)
    out_dtype = x.dtype if out_dtype is None else out_dtype
    y = torch.empty_like(x, dtype=out_dtype)
    grid = (triton.cdiv(M, R),)
    _rms_norm_kernel[grid](
        x,
        weight,
        y,
        M,
        H,
        float(eps),
        R=R,
        BLOCK=BLOCK,
        num_warps=int(cfg["num_warps"]),
        num_stages=int(cfg["num_stages"]),
    )
    return y


def warm(x: Any, weight: Any, eps: float, **kwargs: Any) -> None:
    """JIT 编译一次 + 真实 launch（capture 之前必须调用）。"""
    rms_norm_fused(x, weight, eps, **kwargs)


if _HAS_TRITON:

    @triton.jit
    def _rms_norm_kernel(
        x_ptr,
        w_ptr,
        y_ptr,
        M,
        H,
        EPS,
        R: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """逐行：`y[r, :] = (x[r, :] * rstd).to(fp16) * w`。

        ★ 舍入点与 HF 对齐：先在 fp32 里乘 `rstd`、**舍入到 fp16**，再在 fp16 里乘 `w`。
          改成在 fp32 里一次乘完再舍入会改变数值，而且实测更慢（见模块 docstring）。
        """
        pid = tl.program_id(0)
        rows = pid * R + tl.arange(0, R)
        cols = tl.arange(0, BLOCK)
        rmask = rows < M
        cmask = cols < H
        mask = rmask[:, None] & cmask[None, :]

        x = tl.load(x_ptr + rows[:, None] * H + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / H  # fp32 累加（与 HF 的 mean 同精度，顺序不同）
        rstd = 1.0 / tl.sqrt(var + EPS)
        normed = (x * rstd[:, None]).to(tl.float16)
        w = tl.load(w_ptr + cols, mask=cmask, other=0.0).to(tl.float16)
        y = normed * w[None, :]
        tl.store(y_ptr + rows[:, None] * H + cols[None, :], y, mask=mask)
