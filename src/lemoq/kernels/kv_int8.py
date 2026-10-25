"""E6（EXP-p5-01）KV INT8 的**参考数学**与守卫 —— kernel（10-25 窗落地）的对照物。

本模块今天（2026-10-24）只落三样**CPU 可测**的东西：

    1. 参考量化数学（K 门的对照物）——kernel 必须逐位复现它的舍入链：
           amax(fp32) → scale = amax/127 → q = round(x/scale)（half-to-even）
           → clamp ±127；scale 存 fp16（fp32→fp16 就近舍入是确定的 ⇒ 逐位可比）。
       全零行/列的约定照 mxfp4 的零块先例：scale = 1.0、codes = 0（避开除零，
       反量化仍是 0，数值无害）。
    2. 守卫（`kv_int8_rows_applicable`）——不满足就**显式回退** fp16 cache，
       绝不静默算错（照 `kv_direct_write_applicable` 的教训：缺 is_cuda 检查
       让盒子上的 CPU 测试崩过一次）。
    3. 折叠代数的参考恒等式（读路径"内层零解包"的依据，RECON §3.2）。

舍入约定是**预注册**的（EXP-p5-01 记录 §Method）：kernel 对不齐是实现问题，
不是判据问题——与 F4 的 K 门同一立场。

    name                状态        说明
    kv_int8_write       10-25 窗    Triton 写 kernel（grid=(2H,)：K 行×H + V 行×H；
                                    行内 amax 归约 → int8 行 + fp16 scale 写静态 buffer）。
                                    `tests/test_kv_int8_gpu.py` 已按名预置，落地自动激活。
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
