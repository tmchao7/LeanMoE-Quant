"""kv_direct — F4 候选：KV 直写 kernel（默认关，S2 预注册选定，见 HANDOFF_2026-10-24 §2）。

把 rope 后的 k 与 v **逐位**写进 `StaticCache` 的 `[pos]` 槽，替代 HF
`StaticLayer.update()` 的 `index_copy_`（每步 32 次 aten launch，实测 0.0560 ms、
gap 55.9 µs——S2 账本第一名）。语义契约由 `kv_write.write_kv_slot` 钉死，
K 门 = `tests/test_kv_direct_write.py` 的 kernel 用例与 HF 路径**逐位**对照。

设计依据（2026-10-24 调研，`docs/reference/KERNEL_MICROOPT_PLAYBOOK_2026-10-24.md`）：

    - vLLM `reshape_and_cache_flash`（csrc/cache_kernels.cu）：NHD 布局下写 cache
      是**平铺 coalesced copy**；本仓 (1,H,1,D)→(1,H,max,D) 同为平铺、无转置。
    - 小拷贝 kernel：grid=(2H,)（k/v 各 H 个 head）、BLOCK=head_dim、num_warps=1
      （256 B/head 是 latency-bound，多 warp 只加同步）；D 为 2 的幂 ⇒ 无 mask 精确拷贝。
    - int32 偏移（max_len·D ≪ 2^31）；**默认 cache 策略**——写入的 KV 下一步就被
      attention 读，96 MB L2 应留住它（勿用 evict_first）。
    - 图安全：无 autotune、pos 从**设备端** `cache_position` 张量读（capture 烘焙指针，
      值每步在图外填充）；dtype 不做转换（守卫保证 k/v 与 cache 同 dtype ⇒ 纯位拷贝）。
"""

from __future__ import annotations

from typing import Any

_TRITON_KERNEL = None


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # noqa: BLE001
        return None


def kv_direct_write_available(*, device: Any = None) -> bool:
    """真 kernel 的可用性守卫（CPU/Mac ⇒ False，走显式回退；与 fused_route 同款）。"""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: PLC0415,F401

        return True
    except Exception:  # noqa: BLE001
        return False


def kv_direct_write_applicable(cache_layer: Any, key_states: Any, value_states: Any) -> bool:
    """形状/布局守卫：不满足就**显式回退** `update()`，绝不静默算错。

    要求（全部满足才走 kernel）：
        - cache 的 keys/values 是 (1, H, max_len, D) 的**连续**张量且同形状；
        - dtype 与 k/v 一致（kernel 是纯位拷贝，不做转换）；
        - k/v 形状 (1, H, 1, D)；
        - head_dim D 是 2 的幂（`tl.arange` 需要精确 BLOCK，无 mask 路径）；
        - 全部张量在 **CUDA** 上（CPU 张量即使形状匹配也回退——盒子上有 triton，
          默认翻转后 CPU 测试必须走 update()，2026-10-24 实测踩中）。
    """
    torch = _torch()
    if torch is None:
        return False
    if not (key_states.is_cuda and value_states.is_cuda):
        return False
    keys = getattr(cache_layer, "keys", None)
    values = getattr(cache_layer, "values", None)
    if keys is None or values is None:
        return False
    if not (keys.is_cuda and values.is_cuda):
        return False
    if keys.shape != values.shape or keys.dim() != 4 or keys.shape[0] != 1:
        return False
    if not (keys.is_contiguous() and values.is_contiguous()):
        return False
    if key_states.dtype is not keys.dtype or value_states.dtype is not values.dtype:
        return False
    _, heads, _, dim = keys.shape
    if key_states.shape != (1, heads, 1, dim) or value_states.shape != (1, heads, 1, dim):
        return False
    return dim >= 2 and (dim & (dim - 1)) == 0


def _get_kernel():
    """惰性编译入口（CPU 导入安全；CUDA 上首次调用时才 import triton）。"""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton  # noqa: PLC0415
    import triton.language as tl  # noqa: PLC0415

    @triton.jit
    def _kv_direct_write_kernel(
        k_src_ptr,
        v_src_ptr,
        k_dst_ptr,
        v_dst_ptr,
        pos_ptr,
        stride_src_head,
        stride_dst_head,
        HEADS: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """grid=(2*HEADS,)：pid < HEADS 写 k，否则写 v。每程序平铺拷贝一个 head 行。

        偏移 int32（max_len·D ≪ 2^31）；pos 从设备张量读（图安全）；
        纯 LDG→STG（dtype 由守卫保证两侧一致），默认 cache 策略（KV 下一步被 attention 读）。
        """
        pid = tl.program_id(0)
        head = pid % HEADS
        pos = tl.load(pos_ptr).to(tl.int32)
        offs = tl.arange(0, BLOCK_DIM)
        if pid < HEADS:
            val = tl.load(k_src_ptr + head * stride_src_head + offs)
            tl.store(k_dst_ptr + head * stride_dst_head + pos * BLOCK_DIM + offs, val)
        else:
            val = tl.load(v_src_ptr + head * stride_src_head + offs)
            tl.store(v_dst_ptr + head * stride_dst_head + pos * BLOCK_DIM + offs, val)

    _TRITON_KERNEL = _kv_direct_write_kernel
    return _TRITON_KERNEL


def kv_direct_write(layer: Any, key_states: Any, value_states: Any, cache_position: Any) -> None:
    """把 (1,H,1,D) 的 k/v 写进 `layer.keys/values` 的 `[pos]` 槽（逐位，同 `write_kv_slot`）。

    前置（调用方用 `kv_direct_write_applicable` 保证）：形状匹配、连续、D 为 2 的幂、
    CUDA + triton 可用。`cache_position` 是设备端 int 张量（HF `cache_kwargs` 原样传入），
    decode 形状下 numel==1。
    """
    torch = _torch()
    assert torch is not None and torch.cuda.is_available(), "kv_direct_write 需要 CUDA"
    pos_ptr = cache_position
    if pos_ptr.dim() != 0:
        assert pos_ptr.numel() == 1, "kv_direct_write 只支持 decode 的单槽写（q_len=1）"
        pos_ptr = pos_ptr.reshape(-1)
    kernel = _get_kernel()
    _, heads, _, dim = layer.keys.shape
    max_len = layer.keys.shape[2]
    kernel[(2 * heads,)](
        key_states,
        value_states,
        layer.keys,
        layer.values,
        pos_ptr,
        key_states.stride(1),
        max_len * dim,
        HEADS=heads,
        BLOCK_DIM=dim,
        num_warps=1,
    )
