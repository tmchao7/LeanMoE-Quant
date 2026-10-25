"""E6（EXP-p5-01）KV INT8 的参考数学、守卫与**写路径 kernel**。

模块结构（K 门对照物在前，kernel 在后——判据与实现分得清）：

    1. 参考量化数学（K 门的对照物）——kernel 必须逐位复现它的舍入链：
           amax(fp32) → scale = amax/127 → q = round(x/scale)（half-to-even）
           → clamp ±127；scale 存 fp16（fp32→fp16 就近舍入是确定的 ⇒ 逐位可比）。
       全零行/列的约定照 mxfp4 的零块先例：scale = 1.0、codes = 0（避开除零，
       反量化仍是 0，数值无害）。
    2. 守卫（`kv_int8_rows_applicable`）——不满足就**显式回退** fp16 cache，
       绝不静默算错（照 `kv_direct_write_applicable` 的教训：缺 is_cuda 检查
       让盒子上的 CPU 测试崩过一次）。
    3. 折叠代数的参考恒等式（读路径"内层零解包"的依据，RECON §3.2）。
    4. `kv_int8_write` Triton kernel（grid=(2H,)：K 行×H + V 行×H；行内 amax 归约
       → int8 行 + fp16 scale）。★ 舍入用 `libdevice.rint`（half-to-even）——
       `tl.round` 是 half-away-from-zero，与 torch.round 逐位不齐（.5 值上必炸，
       K 门是逐位判据）。

舍入约定是**预注册**的（EXP-p5-01 记录 §Method）：kernel 对不齐是实现问题，
不是判据问题——与 F4 的 K 门同一立场。
"""

from __future__ import annotations

from typing import Any

_QMAX = 127.0


def quantize_per_token_int8(x: Any) -> tuple[Any, Any]:
    """沿**最后一维**（head_dim）做 per-token 对称 int8 量化。

    输入 (..., D)（kernel 场景是 (1, H, 1, D) 的一批行）；返回
    `(codes int8 (..., D), scale fp16 (...,))`。全零行 scale=1.0。
    """
    torch = _torch()
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))
    q = torch.round(xf / scale).clamp_(-int(_QMAX), int(_QMAX))
    return q.to(torch.int8), scale.to(torch.float16).squeeze(-1)


def quantize_per_channel_int8(x: Any) -> tuple[Any, Any]:
    """沿**倒数第二维**（token 维）做 per-channel 对称 int8 量化（v2 阶梯的 K 粒度）。

    输入 (..., T, D)（一个头里连续 T 个 token 的行块）；KIVI/KVQuant 的经验是
    K 的离群在**通道**维——per-channel（每通道一个 scale，对 T 个 token 共享）
    比 per-token 稳。返回 `(codes int8 (..., T, D), scale fp16 (..., D))`。
    """
    torch = _torch()
    xf = x.float()
    amax = xf.abs().amax(dim=-2, keepdim=True)
    scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))
    q = torch.round(xf / scale).clamp_(-int(_QMAX), int(_QMAX))
    return q.to(torch.int8), scale.to(torch.float16).squeeze(-2)


def dequant_int8(codes: Any, scale: Any) -> Any:
    """反量化参考：`codes.float() * scale.float()`（scale fp16 → fp32 与 kernel 读回一致）。"""
    return codes.float() * scale.float()


def kv_int8_rows_applicable(k_new: Any, v_new: Any) -> bool:
    """写路径的行级守卫：不满足就**显式回退** fp16 cache 写，绝不静默算错。

    要求（全部满足才走 int8 写）：
        - k/v 都在 **CUDA**（CPU 张量即使形状匹配也回退——F4 的教训）；
        - dtype fp16 且两者一致（kernel 只做量化转换，不做 dtype 猜测）；
        - 形状 (1, H, 1, D) 且 k/v 同形；
        - head_dim D 是 2 的幂（`tl.arange` 需要精确 BLOCK，无 mask 路径）；
        - 连续。

    ★ cache 侧 int8/scale buffer 的形状检查在集成时加（buffer 是 10-25 窗建的，
      本守卫先钉住行级必要条件；集成守卫必须**包含**本函数）。
    """
    torch = _torch()
    if torch is None:
        return False
    if not (k_new.is_cuda and v_new.is_cuda):
        return False
    if k_new.dtype is not torch.float16 or v_new.dtype is not torch.float16:
        return False
    if k_new.shape != v_new.shape or k_new.dim() != 4 or k_new.shape[0] != 1:
        return False
    if not (k_new.is_contiguous() and v_new.is_contiguous()):
        return False
    dim = k_new.shape[-1]
    if dim < 1 or dim & (dim - 1) != 0:
        return False
    return True


def fold_scores_per_token_reference(scores: Any, k_scale: Any) -> Any:
    """K per-token 的**分数折叠**参考：`s_t·(q·k_t)`（KIVI 式代数，读路径内层零解包）。

    这是恒等式的参考形态（fp32 下逐元素相等，见 CPU 测试）——kernel 把它做成
    logits 上的逐 token 标量乘；V 侧同理折进 attention 权重。预注册结论：
    折叠路径 vs 先反量化再点积是**代数等价、浮点结合序不同** ⇒ 读侧 parity 用
    fp64 oracle + PARITY_GATE 1e-3，不写逐位断言（mxfp4 的教训）。
    """
    return scores * k_scale


def _torch() -> Any:
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# cache 布局（E6 读路径）：per-layer int8 镜像 (1,H,S,D) codes + (1,H,S) scale，
# fp16 StaticCache 保留为正确性网；bulk 量化在初始化时做一次（eager 相位，可同步），
# 之后每 decode 步镜像写新行（pos 从设备张量读——图安全，照 F4）。
# ---------------------------------------------------------------------------

_TRITON_CACHE_WRITE: Any = None


def kv_int8_bulk_quantize(cache_keys: Any, cache_values: Any, filled: Any = None) -> dict[str, Any]:
    """把 fp16 cache 按 per-token 对称 int8 量化（torch 参考数学，全设备侧、免 host 长度）。

    `filled=None`（初始化路径）量化**整段** S——未填充槽位是零行 ⇒ scale=1.0/codes=0，
    attention 不读它们，值确定无害。返回 dict：kc/ks/vc/vs——形状 (1,H,S,D) int8 与
    (1,H,S) fp16（S = cache 容量）。
    """
    torch = _torch()
    H, S, D = cache_keys.shape[1], cache_keys.shape[2], cache_keys.shape[3]
    dev = cache_keys.device
    out: dict[str, Any] = {}
    for name, tensor in (("kc", cache_keys), ("vc", cache_values)):
        codes = torch.ones((1, H, S, D), dtype=torch.int8, device=dev)
        scales = torch.ones((1, H, S), dtype=torch.float16, device=dev)
        if filled is not None and filled > 0:
            region = tensor[0, :, :filled, :].float()  # (H, filled, D)
            amax = region.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))
            q = torch.round(region / scale).clamp_(-int(_QMAX), int(_QMAX))
            codes[0, :, :filled, :] = q.to(torch.int8)
            scales[0, :, :filled] = scale.squeeze(-1).to(torch.float16)
        else:
            region = tensor[0].float()  # (H, S, D) 整段
            amax = region.abs().amax(dim=-1, keepdim=True)
            scale = torch.where(amax > 0, amax / _QMAX, torch.ones_like(amax))
            q = torch.round(region / scale).clamp_(-int(_QMAX), int(_QMAX))
            codes[0] = q.to(torch.int8)
            scales[0] = scale.squeeze(-1).to(torch.float16)
        out[name] = codes
        out[name.replace("c", "s")] = scales
    return out


def kv_int8_init_buffers(layer_keys: Any, layer_values: Any, filled: Any = None) -> dict[str, Any]:
    """初始化一层的 int8 镜像 buffer（bulk 量化；filled=None 量化整段）。"""
    bufs = kv_int8_bulk_quantize(layer_keys, layer_values, filled)
    bufs["initialized"] = True
    return bufs


def _get_cache_write_kernel() -> Any:
    """惰性定义（CPU 导入安全）：grid=(2H,)，把 (1,H,1,D) 的新行量化写入 cache 布局。"""
    global _TRITON_CACHE_WRITE
    if _TRITON_CACHE_WRITE is not None:
        return _TRITON_CACHE_WRITE
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415
    from triton.language.extra import libdevice  # noqa: PLC0415

    @triton.jit
    def _kv_int8_cache_write_kernel(
        k_ptr,  # (1,H,1,D) fp16 新行
        v_ptr,
        kc_ptr,  # (1,H,S,D) int8
        ks_ptr,  # (1,H,S) fp16
        vc_ptr,
        vs_ptr,
        pos_ptr,  # (1,) int 设备张量（图安全，照 F4 的 pos_ptr）
        kv_stride_h,  # codes 的 head stride = S*D
        s_len,  # scale 的序列长度 = S
        HEADS: tl.constexpr,
        D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pos = tl.load(pos_ptr).to(tl.int32)
        offs_d = tl.arange(0, D)
        head = pid % HEADS
        code_base = head * kv_stride_h + pos * D
        scale_off = head * s_len + pos
        if pid < HEADS:
            x = tl.load(k_ptr + head * D + offs_d).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            s = tl.where(amax > 0, amax / 127.0, 1.0)
            q = libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(kc_ptr + code_base + offs_d, q.to(tl.int8))
            tl.store(ks_ptr + scale_off, s.to(tl.float16))
        else:
            x = tl.load(v_ptr + head * D + offs_d).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            s = tl.where(amax > 0, amax / 127.0, 1.0)
            q = libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(vc_ptr + code_base + offs_d, q.to(tl.int8))
            tl.store(vs_ptr + scale_off, s.to(tl.float16))

    _TRITON_CACHE_WRITE = _kv_int8_cache_write_kernel
    return _TRITON_CACHE_WRITE


def kv_int8_cache_write(bufs: dict[str, Any], k_new: Any, v_new: Any, pos: Any) -> None:
    """把新 token 的 k/v 行量化写入 int8 镜像 cache 的 [pos] 槽（每步一次，图安全）。

    `pos` 是设备张量（HF 的 cache_position）。守卫与 `kv_int8_rows_applicable`
    同源（行形状/dtype/CUDA）；buffer 形状不匹配直接 RuntimeError（静默写错槽位
    比回退更危险——读端会读到旧值，tokens 会"看起来正常"地错）。
    """
    if not kv_int8_rows_applicable(k_new, v_new):
        raise RuntimeError("kv_int8_cache_write: 行守卫不满足——调用方应先回退 fp16 读")
    kc = bufs["kc"]
    if (
        kc.shape[1] != k_new.shape[1]
        or kc.shape[3] != k_new.shape[3]
        or kc.shape[0] != 1
        or not kc.is_contiguous()
    ):
        raise RuntimeError(
            "kv_int8_cache_write: buffer 形状/布局与行不匹配 %s vs %s"
            % (tuple(kc.shape), tuple(k_new.shape))
        )
    S = kc.shape[2]
    _get_cache_write_kernel()[(2 * k_new.shape[1],)](
        k_new,
        v_new,
        kc,
        bufs["ks"],
        bufs["vc"],
        bufs["vs"],
        pos,
        int(kc.stride(1)),
        S,
        HEADS=k_new.shape[1],
        D=k_new.shape[3],
        num_warps=1,
    )


# ---------------------------------------------------------------------------
# 写路径 kernel（惰性定义：CPU 导入安全，照 kv_direct._get_kernel 模式）
# ---------------------------------------------------------------------------

_TRITON_KERNEL: Any = None


def _get_kernel() -> Any:
    """惰性编译入口（CPU 导入安全；CUDA 上首次调用才 import triton）。"""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415
    from triton.language.extra import libdevice  # noqa: PLC0415

    @triton.jit
    def _kv_int8_write_kernel(
        k_ptr,
        v_ptr,
        kc_ptr,
        ks_ptr,
        vc_ptr,
        vs_ptr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
    ):
        """grid=(2*HEADS,)：pid < HEADS 量 k 行，否则量 v 行。

        参考数学的逐位复刻（EXP-p5-01 §Method 预注册舍入链）：fp32 amax → /127
        → **rint（half-to-even）** → clamp ±127 → int8；scale 存 fp16。全零行
        scale=1.0。★ 不许用 `tl.round`——它是 half-away-from-zero，与 torch.round
        在 .5 值上逐位不齐，而 K 门是逐位判据。偏移 int32（H·D ≪ 2^31）。
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, D)
        if pid < HEADS:
            x = tl.load(k_ptr + pid * D + offs).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            s = tl.where(amax > 0, amax / 127.0, 1.0)
            q = libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(kc_ptr + pid * D + offs, q.to(tl.int8))
            tl.store(ks_ptr + pid, s.to(tl.float16))
        else:
            base = (pid - HEADS) * D
            x = tl.load(v_ptr + base + offs).to(tl.float32)
            amax = tl.max(tl.abs(x), axis=0)
            s = tl.where(amax > 0, amax / 127.0, 1.0)
            q = libdevice.rint(x / s)
            q = tl.minimum(tl.maximum(q, -127.0), 127.0)
            tl.store(vc_ptr + base + offs, q.to(tl.int8))
            tl.store(vs_ptr + (pid - HEADS), s.to(tl.float16))

    _TRITON_KERNEL = _kv_int8_write_kernel
    return _TRITON_KERNEL


def kv_int8_write(
    k_new: Any,
    v_new: Any,
    *,
    k_codes: Any = None,
    k_scale: Any = None,
    v_codes: Any = None,
    v_scale: Any = None,
) -> tuple[Any, Any, Any, Any]:
    """写路径入口：把 (1,H,1,D) 的 k/v 行量化为 int8 codes + fp16 per-token scale。

    out 四件套缺省时就地分配（**K 门/测试便捷**）；decode 集成时传 cache 的
    **静态地址**切片（图捕获兼容，照 F4）。守卫不满足 ⇒ RuntimeError——调用方
    必须先查 `kv_int8_rows_applicable`，不满足就走 fp16 写（回退不是本函数职责）。
    """
    torch = _torch()
    if not kv_int8_rows_applicable(k_new, v_new):
        raise RuntimeError(
            "kv_int8_write: 守卫不满足——先查 kv_int8_rows_applicable，不满足走 fp16 写"
        )
    heads, dim = k_new.shape[1], k_new.shape[-1]
    if k_codes is None:
        k_codes = torch.empty((1, heads, 1, dim), dtype=torch.int8, device=k_new.device)
        k_scale = torch.empty((1, heads, 1), dtype=torch.float16, device=k_new.device)
        v_codes = torch.empty((1, heads, 1, dim), dtype=torch.int8, device=v_new.device)
        v_scale = torch.empty((1, heads, 1), dtype=torch.float16, device=v_new.device)
    _get_kernel()[(2 * heads,)](
        k_new,
        v_new,
        k_codes,
        k_scale,
        v_codes,
        v_scale,
        HEADS=heads,
        D=dim,
        num_warps=1,
    )
    return k_codes, k_scale, v_codes, v_scale
