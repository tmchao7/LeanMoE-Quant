"""_dequant_triton — dense 反量化融合 GEMV 的 Triton kernel（`EXP-p2-01` 的 fused 臂）。

## 这个文件为什么单独存在

与 `_attn_triton.py` / `_grouped_triton.py` 同一条理由：`triton` 在 Mac 上装不了
（没有 CUDA 后端）。把 `@triton.jit` 放进 `lemoq/quant/linear.py` 会让整个量化模块在
Mac 上 import 失败 —— 而离线测试（回退逻辑、形状检查）正是发现错误的地方。所以：

    lemoq/quant/linear.py      QuantLinear（host 逻辑 + separate 臂 + fused 分派）
    _dequant_triton.py         本文件：`@triton.jit` kernel + host wrapper

★ 本文件的 `import triton` 是**可失败的**（`_HAS_TRITON`）：没有 triton 时模块仍可
import，但 `fused_available()` 返回 False、`dequant_gemm_fused()` 抛出可读的错误。
这样 `QuantLinear` 的"没有 kernel 就退回 separate 并**如实记录**"才是一条能被测试的代码
路径，而不是一个只能在 GPU 机器上发现的 ImportError。

## 语义（必须与 `lemoq.kernels.reference` 逐字对应）

    separate : W_hat = dequant(packed)         # 物化 [out,in] 的 fp16 权重（写+读一遍）
               y     = x @ W_hat.T
    fused    : y     = x @ dequant(packed).T   # 读 packed 时当场解压，不落中间张量（本文件）

**数学结果相同**，差别只在多不多一次权重的写+读流量 —— 这就是 E2 的全部内容。
所以 K1 门就是"本 kernel 的输出 == `dequant_gemm_separate_reference`"
（`atol/rtol 1e-3`，容差不得为了通过而放宽）。

## 数值口径（与 separate 臂一致，不是"差不多"）

1. **累加在 fp32，且不用 `tl.dot`**：fp32 的 `tl.dot` 默认走 tf32，分辨力低于 1e-3，
   会与"相对 fp64 oracle 的 1e-3 门"打架。M=1 的 decode 是访存受限，FFMA 不是瓶颈 ——
   正确性门优先（AGENTS.md 规则 2）。这里用 `tl.sum(w * x, axis=1)` 的逐元素 fp32 乘加。
2. **scale/zero 按原 dtype 读入再转 fp32**：`QuantLinear` 存 fp16 scale（字节账要求），
   参考实现也吃同一份 fp16 tensor → 两边逐位相同。
3. **4-bit 的 nibble 在寄存器里展开**（`pack.py` 的约定：低半字节在前），
   不物化 int8 codes 中间张量 —— 否则"融合"就名不副实。

## CUDA Graph 约束（P3 会 capture 这条路径）

1. **不得用 `@triton.autotune`**：它会真的跑多个 config 计时，capture 期间等于往图里塞
   计划外的 kernel。配置由 host 显式传入（`DEQUANT_GEMM_CONFIGS`）。
2. **grid 是 host 侧常量**：`cdiv(N, BLOCK_N)`。M/N/K 进的是核函数参数（decode 期间为常量），
   与 attn 的"循环上界来自设备内存"不同 —— 那里必须设备端取值，这里不必。
3. 首次调用会 JIT 编译 —— 必须在 capture **之前** warm 过（`warm()`）。

## 只支持单 token（decode）

`QuantLinear` 只在 `q_len == 1` 的 decode core 上分派到本 kernel；prefill / 多 token
退回 separate 臂。kernel 的 `M` 是行循环，所以对任意 M 都算得对（K1 测试用 m=4 交叉验证），
但生产路径只用 M=1。
"""

from __future__ import annotations

from typing import Any

# ★ triton 的 import 允许失败：本模块必须在没有 triton 的机器上也能 import。
try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR: str | None = None
except Exception as _exc:  # noqa: BLE001
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = "%s: %s" % (type(_exc).__name__, _exc)

_HAS_TRITON = triton is not None and tl is not None

__all__ = [
    "DEQUANT_GEMM_CONFIGS",
    "dequant_gemm_fused",
    "fused_available",
    "warm",
]


# ---------------------------------------------------------------------------
# 默认配置
#
# ★ 不用 @triton.autotune（理由见模块 docstring）。参数敏感性是**要报告的结果**
#   （`lemoq.kernels.autotune.DEQUANT_GEMM_AUTOTUNE_SPACE`），扫描在隔离微基准里做。
# ---------------------------------------------------------------------------
DEQUANT_GEMM_CONFIGS: dict[str, dict[str, int]] = {
    # BLOCK_N 小 = 更多 program（M=1 的 GEMV 里 grid = cdiv(N, BLOCK_N)，OLMoE 的
    # q_proj N=2048）；BLOCK_K 大 = 每个 program 一次读更多 packed 权重。
    # ★ 本默认值是 2026-10-07 在 5090 上按声明空间的**子集**扫出来的赢家，
    #   不是全网格 autotune；扫描数据见 experiments/EXP-p2-01-dequant-fusion.md 结果节。
    "dequant_gemm": {"BLOCK_N": 4, "BLOCK_K": 256, "num_warps": 2, "num_stages": 3},
}


def fused_available() -> bool:
    """fused 臂在这台机器上能不能用（Mac 上永远 False。**不抛异常**）。"""
    if not _HAS_TRITON:
        return False
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def unavailable_reason() -> str | None:
    """fused 臂不可用的**可读原因**（可用时 None）。runner 的加载前拒绝要用它。"""
    if _HAS_TRITON and _TRITON_IMPORT_ERROR is None:
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            return "torch 不可用（%s）" % type(exc).__name__
        if not torch.cuda.is_available():
            return "本机没有可用的 CUDA 设备"
        return None
    return "triton 不可用（%s）" % (_TRITON_IMPORT_ERROR or "unknown")


# 解析结果缓存：`dequant_gemm_fused` 在 decode 里**每次前向都会被调用**（OLMoE 上每
# token 上千次），每次都重算 config 是在热点路径上烧 μs。key 里带上全局默认值，
# 所以扫描时改 `DEQUANT_GEMM_CONFIGS` 不会读到陈旧条目。
_CFG_CACHE: dict[Any, dict[str, int]] = {}


def _resolve_config(config: dict[str, Any] | None, group_size: int = 128) -> dict[str, int]:
    override = None if not config else (config.get("dequant_gemm") or config)
    key = (
        tuple(sorted(DEQUANT_GEMM_CONFIGS["dequant_gemm"].items())),
        None if override is None else tuple(sorted(override.items())),
        int(group_size),
    )
    cached = _CFG_CACHE.get(key)
    if cached is not None:
        return cached
    cfg = dict(DEQUANT_GEMM_CONFIGS["dequant_gemm"])
    if override:
        cfg.update({k: int(v) for k, v in override.items()})
    for name in ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages"):
        if name not in cfg:
            raise ValueError("dequant_gemm 配置缺少 %s" % name)
    if cfg["BLOCK_N"] < 1 or cfg["BLOCK_K"] < 16:
        raise ValueError("非法 dequant_gemm 配置：%r" % cfg)
    if cfg["BLOCK_K"] % 2 != 0:
        raise ValueError("BLOCK_K 必须是偶数（4-bit 每两个 k 共用一个 uint8）：%r" % cfg)
    # ★ scale 的装载方式要求"一个 k 块要么含整数个组、要么落在一个组内"（见 kernel 注释）。
    #   不满足就响亮报错 —— 静默退回逐元素 gather 会让性能悄悄掉一半。
    if cfg["BLOCK_K"] % int(group_size) != 0 and int(group_size) % cfg["BLOCK_K"] != 0:
        raise ValueError(
            "dequant_gemm 配置非法：BLOCK_K=%d 与 group_size=%d 必须有一个整除另一个"
            "（否则一个 k 块会跨组边界、scale 不能按组广播）" % (cfg["BLOCK_K"], group_size)
        )
    cfg = {k: int(v) for k, v in cfg.items()}
    _CFG_CACHE[key] = cfg
    return cfg


def dequant_gemm_fused(
    x: Any,
    qweight: Any,
    scale: Any,
    zero: Any = None,
    *,
    group_size: int,
    bits: int | None = None,
    symmetric: bool = True,
    out_dtype: Any = None,
    config: dict[str, Any] | None = None,
    mxfp4: bool = False,
) -> Any:
    """`y = x @ dequant(qweight, scale, zero).T`，反量化在 kernel 内完成。

    参数（与 `dequant_gemm_separate_reference` 同一个契约）：

        x         (M, K)  fp16 / fp32，K = 未补齐的输入维
        qweight   (N, P)  int8（bits=8，P = padded_in）或 uint8（bits=4，P = padded_in/2）
        scale     (N, n_groups)  fp16 / fp32；n_groups 沿 **输入维**（与 rtn 一致）
        zero      对称量化传 None
        group_size  与量化时一致；padded_in = n_groups * group_size

    返回 (M, N)；`out_dtype=None` 时是 fp32（与参考实现一致），否则按 `out_dtype`。
    边界一律**显式报错**，不静默算错。

    ★ 契约：x / qweight / scale / zero **必须连续**（非连续会静默算错，所以显式报错）——
      这样 kernel 签名里一个 stride 都不用传，实测每次 launch 省约 10 µs。
    """
    if not _HAS_TRITON:
        raise RuntimeError(
            "fused dequant-GEMM 需要 triton（本机不可用：%s）。"
            "separate 臂（QuantLinear 每次 forward 反量化）不需要 triton。"
            % (_TRITON_IMPORT_ERROR or "unknown")
        )
    import torch  # noqa: PLC0415

    cfg = _resolve_config(config, int(group_size))
    if x.dim() != 2:
        raise ValueError("x 必须是 (M, K) 二维，得到 %r" % (tuple(x.shape),))
    if qweight.dim() != 2:
        raise ValueError("qweight 必须是 (N, P) 二维，得到 %r" % (tuple(qweight.shape),))
    if scale.dim() != 2:
        raise ValueError("scale 必须是 (N, n_groups) 二维，得到 %r" % (tuple(scale.shape),))

    M, K = int(x.shape[0]), int(x.shape[1])
    N, packed_in = int(qweight.shape[0]), int(qweight.shape[1])
    n_groups = int(scale.shape[1])
    if int(scale.shape[0]) != N:
        raise ValueError("scale 的行数 %d 与 qweight 的 %d 不一致" % (int(scale.shape[0]), N))
    if bits is None:
        bits = 4 if qweight.dtype == torch.uint8 else 8
    bits = int(bits)
    if bits not in (4, 8):
        raise NotImplementedError("fused dequant-GEMM 只实现 4/8-bit，得到 %d" % bits)
    padded_in = int(n_groups) * int(group_size)
    expect_packed = padded_in if bits == 8 else padded_in // 2
    if packed_in != expect_packed:
        raise ValueError(
            "qweight 的第二维 %d 与 n_groups*group_size(/bits) 推出的 %d 不一致"
            % (packed_in, expect_packed)
        )
    if K > padded_in:
        raise ValueError("K=%d 超过 padded_in=%d：x 比量化权重还宽" % (K, padded_in))
    if not symmetric and zero is None:
        raise ValueError("非对称量化必须给 zero")
    if bits == 4 and int(group_size) % 2 != 0:
        raise ValueError("4-bit 要求 group_size 为偶数，得到 %d" % int(group_size))
    expected_q_dtype = torch.uint8 if bits == 4 else torch.int8
    if qweight.dtype != expected_q_dtype:
        raise ValueError(
            "%d-bit 的 qweight 应是 %s，得到 %s（打包格式见 lemoq.quant.pack）"
            % (bits, expected_q_dtype, qweight.dtype)
        )
    # MXFP4（EXP-p3-02）：E2M1 码（无符号 0..15，自带符号位）+ E8M0 尺度字节（uint8）。
    # 语义契约 = lemoq/quant/mxfp4.py（主机侧量化器）；侦察依据见
    # docs/reference/QUANT_FORMAT_RECON_2026-10-25（向量路径，无 tl.dot）。
    if mxfp4:
        if bits != 4:
            raise ValueError("mxfp4 只支持 4-bit E2M1，得到 bits=%r" % bits)
        if int(group_size) != 32:
            raise ValueError(
                "mxfp4 的块大小固定 32（OCP MX），得到 group_size=%d" % int(group_size)
            )
        if scale.dtype != torch.uint8:
            raise ValueError("mxfp4 的 scale 必须是 uint8 E8M0 字节，得到 %s" % scale.dtype)
        if not symmetric or zero is not None:
            raise ValueError(
                "mxfp4 无 zero-point（E2M1 自带符号位）：symmetric 必须为 True 且 zero=None"
            )
    if x.dtype not in (torch.float16, torch.float32):
        raise NotImplementedError("x 只支持 fp16/fp32，得到 %s" % x.dtype)
    if out_dtype is None:
        out_dtype = torch.float32
    if out_dtype not in (torch.float32, torch.float16):
        raise NotImplementedError("out_dtype 只支持 fp32/fp16，得到 %s" % out_dtype)

    # ★ 契约：全部连续。**不是偷懒** —— 实测（5090 + triton 3.4）launcher 每个额外参数
    #   约 1 µs（3 参数 16 µs，20 参数 32 µs），而 M=1 的 decode 每次前向都要过一遍
    #   launcher。把 10 个 stride 从签名里去掉，是这条路径上最直接的一刀。
    #   `.contiguous()` 在已经连续时是常数开销（返回自身），但仍要显式检查：
    #   非连续输入会**静默算错**，必须响亮失败。
    for name, tensor in (("x", x), ("qweight", qweight), ("scale", scale), ("zero", zero)):
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError("%s 必须是连续的（fused kernel 用形状推 stride）" % name)

    z_arg = scale if zero is None else zero
    y = torch.empty((M, N), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(N, cfg["BLOCK_N"]),)
    _dequant_gemm_kernel[grid](
        x,
        qweight,
        scale,
        z_arg,
        y,
        M,
        N,
        K,
        padded_in,
        GROUP_SIZE=int(group_size),
        BITS=bits,
        SYMMETRIC=bool(symmetric),
        MXFP4=bool(mxfp4),
        OUT_FP16=bool(out_dtype == torch.float16),
        BLOCK_N=int(cfg["BLOCK_N"]),
        BLOCK_K=int(cfg["BLOCK_K"]),
        num_warps=int(cfg["num_warps"]),
        num_stages=int(cfg["num_stages"]),
    )
    return y


def warm(
    x: Any,
    qweight: Any,
    scale: Any,
    zero: Any = None,
    **kwargs: Any,
) -> None:
    """JIT 编译一次并做一次真实 launch（capture 之前必须调用）。

    ★ 首次编译时间**不得**计入稳态数字（`docs/reference/METRICS.md` §2）——
    `benchmarks.timing.record_compile_time` 就是为它准备的。
    """
    dequant_gemm_fused(x, qweight, scale, zero, **kwargs)


if _HAS_TRITON:

    @triton.jit
    def _dequant_gemm_kernel(
        x_ptr,  # (M, K)          fp16/fp32，连续
        q_ptr,  # (N, P)          int8 (BITS=8) / uint8 (BITS=4)，连续
        s_ptr,  # (N, n_groups)   scale，连续
        z_ptr,  # (N, n_groups)   zero（SYMMETRIC 时不读，传 s_ptr 占位）
        y_ptr,  # (M, N)          输出，连续
        M,
        N,
        K,
        PADDED_K,
        GROUP_SIZE: tl.constexpr,
        BITS: tl.constexpr,
        SYMMETRIC: tl.constexpr,
        MXFP4: tl.constexpr,
        OUT_FP16: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """`y[m, n] = sum_k x[m, k] * dequant(q[n, k])`（per-group scale，沿 k）。

        ★ 契约：五个张量**全部连续**，所以 stride 全部由形状推出（见 host 侧注释：
        launcher 每参数 ~1 µs，而这条路径每次都过 launcher）。
        ★ 为什么不用 `tl.dot`：fp32 的 `tl.dot` 默认 tf32，分辨力低于 1e-3 的 K1 门。
        M=1 时是访存受限，逐元素 fp32 乘加不会成为瓶颈（模块 docstring 有完整理由）。
        """
        pid_n = tl.program_id(0)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        n_groups = PADDED_K // GROUP_SIZE

        for m in range(0, M):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for k0 in range(0, PADDED_K, BLOCK_K):
                ks = k0 + tl.arange(0, BLOCK_K)
                mask_g = ks < PADDED_K  # padded 区域（codes 恒为 0）的边界
                if BITS == 8:
                    codes = tl.load(
                        q_ptr + offs_n[:, None] * PADDED_K + ks[None, :],
                        mask=mask_n[:, None] & mask_g[None, :],
                        other=0,
                    ).to(tl.float32)
                else:
                    # 低半字节在前：byte j -> code[2j]（低）、code[2j+1]（高）
                    offs_j = tl.arange(0, BLOCK_K // 2)
                    kb = k0 // 2 + offs_j
                    mask_b = kb < (PADDED_K // 2)
                    packed = tl.load(
                        q_ptr + offs_n[:, None] * (PADDED_K // 2) + kb[None, :],
                        mask=mask_n[:, None] & mask_b[None, :],
                        other=0,
                    ).to(tl.int32)
                    if MXFP4:
                        # E2M1 码无符号（bit3=符号、bit2..0=幅值），不做符号扩展。
                        # 幅值 = e==0 ? 0.5·m : 2^(e−1)·(1+0.5·m) —— 全部是二次幂精确值，
                        # 与主机侧 LUT（mxfp4._E2M1_MAG）逐位一致（K1 的前提）。
                        lo = packed & 0xF
                        hi = (packed >> 4) & 0xF
                        ci = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
                        e = (ci >> 1) & 3
                        m_bit = (ci & 1).to(tl.float32)
                        mag = tl.where(
                            e == 0,
                            0.5 * m_bit,
                            tl.exp2((e - 1).to(tl.float32)) * (1.0 + 0.5 * m_bit),
                        )
                        sg = ((ci >> 3) * (-2.0)) + 1.0  # 0→+1，1→−1
                        codes = sg * mag
                    else:
                        # int4 符号扩展：`((x & 0xF) ^ 8) - 8`（0..15 -> -8..7），纯算术
                        lo = ((packed & 0xF) ^ 8) - 8
                        hi = (((packed >> 4) & 0xF) ^ 8) - 8
                        # join + reshape 按 (byte, which) 展平成 k 序：k = 2j + which
                        codes = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K)).to(tl.float32)

                # ★ scale 一次只装**每组的那个数**（(BLOCK_N, n_in_block)），再广播到
                #   (BLOCK_N, BLOCK_K)。老写法 `g = ks // GROUP_SIZE` 逐元素 gather：
                #   一个 k 块里 128 个元素共用同一个值，却发了 128 次 gather
                #   （BLOCK_K=256 / BLOCK_N=4 / 2 warps 时约 128 条 gather/线程/k 块）。
                #   这是 EXP-p4c-01 §5.4 里 attention 投影那 0.52 ms 的主因：
                #   8.14 µs/次 vs 4.26 MB 的字节地板 2.7 µs。
                #   ★ 分支条件是 constexpr（GROUP_SIZE/BLOCK_K 都是 constexpr），
                #     所以图安全、签名不变（launcher 参数数不变）。
                if BLOCK_K % GROUP_SIZE == 0:
                    n_in_block: tl.constexpr = BLOCK_K // GROUP_SIZE
                    gs = k0 // GROUP_SIZE + tl.arange(0, n_in_block)
                    gmask = mask_n[:, None] & (gs[None, :] < n_groups)
                    sc_t = tl.load(
                        s_ptr + offs_n[:, None] * n_groups + gs[None, :],
                        mask=gmask,
                        other=0.0,
                    ).to(tl.float32)
                    if MXFP4:
                        sc_t = tl.exp2(sc_t - 127.0)  # E8M0 字节 → 2^(X−127)
                    sc = tl.reshape(
                        tl.broadcast_to(sc_t[:, :, None], (BLOCK_N, n_in_block, GROUP_SIZE)),
                        (BLOCK_N, BLOCK_K),
                    )
                else:
                    # 整块落在**一个**组内（host 侧已保证 GROUP_SIZE % BLOCK_K == 0）
                    g_one = k0 // GROUP_SIZE
                    sc1 = tl.load(s_ptr + offs_n * n_groups + g_one, mask=mask_n, other=0.0).to(
                        tl.float32
                    )
                    sc = tl.broadcast_to(sc1[:, None], (BLOCK_N, BLOCK_K))
                if SYMMETRIC:
                    w = codes * sc
                else:
                    if BLOCK_K % GROUP_SIZE == 0:
                        n_in_block2: tl.constexpr = BLOCK_K // GROUP_SIZE
                        gs2 = k0 // GROUP_SIZE + tl.arange(0, n_in_block2)
                        gmask2 = mask_n[:, None] & (gs2[None, :] < n_groups)
                        z_t = tl.load(
                            z_ptr + offs_n[:, None] * n_groups + gs2[None, :],
                            mask=gmask2,
                            other=0.0,
                        ).to(tl.float32)
                        zz = tl.reshape(
                            tl.broadcast_to(z_t[:, :, None], (BLOCK_N, n_in_block2, GROUP_SIZE)),
                            (BLOCK_N, BLOCK_K),
                        )
                    else:
                        z1 = tl.load(
                            z_ptr + offs_n * n_groups + (k0 // GROUP_SIZE),
                            mask=mask_n,
                            other=0.0,
                        ).to(tl.float32)
                        zz = tl.broadcast_to(z1[:, None], (BLOCK_N, BLOCK_K))
                    w = (codes - zz) * sc

                # x 在 padded 区域是 0（掩码用未补齐的 K）—— 与 separate 臂的
                # `_dequant_tile` 把权重尾部切成 [:, :in_features] 等价。
                xv = tl.load(x_ptr + m * K + ks, mask=ks < K, other=0.0).to(tl.float32)
                acc += tl.sum(w * xv[None, :], axis=1)

            if OUT_FP16:
                tl.store(y_ptr + m * N + offs_n, acc.to(tl.float16), mask=mask_n)
            else:
                tl.store(y_ptr + m * N + offs_n, acc, mask=mask_n)
