"""moe_grouped_quant — **量化权重上**的 grouped MoE decode（`EXP-p4b-01` 的 P1）。

## 它解决什么问题

`EXP-p4b-01-truncation` 记录 §3 把 E7（本项目的头号结论）拆成 P0–P3，其中 **P1** 是
"量化 × 可 capture 的 MoE 路径" —— 在此之前它**不存在**：

    replace_moe_blocks → fuse_experts 读 `experts[i].gate_proj.weight`
    → QuantLinear **没有** `.weight`（权重在 packed buffer 里）→ 崩；
    反过来先 fuse 再 quantize 又会 `del block.experts` → 静默只量化 attention。

本模块给出第三条路（记录 §3 的候选 1）：**融合成 stacked 张量之后，直接在那个张量上做
per-group RTN，把 codes 与 scale 留在 kernel 里解包**。这样：

- 不需要 `nn.Linear.weight` —— 量化发生在 fused 的 (E, 2F, H) 上；
- **不物化** fp16 权重 → experts 的 HBM 流量按位宽下降（8-bit = 减半），
  这才是 E7 要测的"量化收益"，而不是"先反量化再算同一件事"；
- 形状全静态、无数据依赖控制流 → 与 fp16 grouped 路径一样**可进 CUDA Graph**（P3 的前提）。

★ 与"量化每个 expert 再融合"的等价性：RTN 是**逐行、逐组**的（组沿最后一维），
而融合只是把专家的行堆起来 / 在 dim=1 拼接 —— 两者都不跨行。所以
"先融合再量化" 与 "先量化再融合" 逐位相同，这有测试钉住
（`test_quantized_fuse_matches_per_expert_quantization`），不是口头声明。

## 与 fp16 路径的数值关系

和 `moe_grouped` 的说明一致：kernel 在 fp32 里累加，比 gather 路径（fp16 中间张量）更准，
所以 **K1 门必须比 fp64/fp32 oracle**，不能比 gather 路径。
oracle 就在本模块（`*_reference`），不需要 triton，可在 CPU 上跑。

## 支持的位宽

**8-bit 与 4-bit**（2026-10-07 起）。4-bit 的 nibble 在 kernel 里展开
（`pack.py` 的约定：低半字节在前），与 dense 侧 `_dequant_triton` 用同一个把戏
（`tl.join` + `tl.reshape`）。★ 4-bit 的意义**只到 evidence 层**：`w4a16 / g128`
的引用门 FAIL（5 窗口 mean KL 0.0933 > 0.05），所以它的延迟按硬规则 2 不得作为
性能结论 —— 跑它是为了把 p4b 声明的 w4a16 行补成实测（记录 §2.9）。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from lemoq.kernels.moe_grouped import (  # noqa: F401
    _MAX_OFFSET,
    PARITY_GATE,
    launch_plan,
    merged_config,
)

__all__ = [
    "QuantFusedExperts",
    "dequantize_fused_experts",
    "grouped_down_sum_quant_reference",
    "grouped_gate_up_act_quant_reference",
    "grouped_moe_forward_quant",
    "quant_launch_plan",
    "quantize_fused_experts",
    "quantized_grouped_available",
    "static_moe_forward_grouped_quant",
]


def _torch():
    import torch  # noqa: PLC0415

    return torch


@dataclass
class QuantFusedExperts:
    """量化后的 expert 权重：packed codes + per-group scale，形状与 fp16 版一一对应。

        packed_gate_up (E, 2F, H) int8 [8-bit] / (E, 2F, H//2) uint8 [4-bit]
        packed_down    (E, H, F)  int8 [8-bit] / (E, H, F//2)  uint8 [4-bit]
        scale_gate_up (E, 2F, H//g) fp16      scale_down (E, H, F//g) fp16

    组沿**最后一维**（kernel 的 K 维），与 `lemoq.quant.rtn` / `bytes._quantized_bytes`
    的口径一致。是 dataclass 而不是自定义类，是为了让
    `benchmarks.runner._weight_bytes_actual` 的通用拆容器逻辑能数到这些张量。
    """

    packed_gate_up: Any
    scale_gate_up: Any
    packed_down: Any
    scale_down: Any
    bits: int
    group_size: int

    @property
    def num_experts(self) -> int:
        return int(self.packed_gate_up.shape[0])

    @property
    def ffn(self) -> int:
        return int(self.packed_gate_up.shape[1]) // 2

    @property
    def hidden(self) -> int:
        """**未打包**的 hidden（4-bit 时 packed 的最后一维是 H//2）。"""
        return self._unpacked_last(self.packed_gate_up)

    def _unpacked_last(self, packed: Any) -> int:
        """packed 张量最后一维对应的未打包宽度：8-bit ×1、4-bit ×2。"""
        return int(packed.shape[-1]) * (8 // int(self.bits))

    def nbytes(self) -> int:
        total = 0
        for t in (self.packed_gate_up, self.scale_gate_up, self.packed_down, self.scale_down):
            total += int(t.numel()) * int(t.element_size())
        return total

    def to(self, device: Any = None, dtype: Any = None) -> "QuantFusedExperts":
        """搬到目标设备。**只搬设备、不转 dtype。**

        ★ `dtype` 显式拒绝（`FusedExperts.to` 接受它，所以这里更容易误用）：
        packed 必须保持 int8、scale 必须保持 fp16 —— 把它们转成浮点就等于把量化拆掉
        （而且 fp16 存不下 int8 的整数域）。宁可报错。
        与 `Tensor.to` 的习惯兼容：`to(torch.float32)` 会被识别成 dtype 请求并拒绝。
        """
        torch = _torch()

        if isinstance(device, torch.dtype):
            device, dtype = None, device
        if dtype is not None:
            raise ValueError(
                "QuantFusedExperts 不支持 dtype 转换（packed 必须 int8/uint8、scale 必须 fp16）："
                "得到 dtype=%r。要改精度请重新量化。" % (dtype,)
            )
        if device is None:
            return self

        return QuantFusedExperts(
            packed_gate_up=self.packed_gate_up.to(device=device).contiguous(),
            scale_gate_up=self.scale_gate_up.to(device=device).contiguous(),
            packed_down=self.packed_down.to(device=device).contiguous(),
            scale_down=self.scale_down.to(device=device).contiguous(),
            bits=self.bits,
            group_size=self.group_size,
        )

    # ---------- 反量化（oracle / prefill 用；decode 路径**不**走这里） ----------
    def _dequant_2d(self, codes: Any, scale: Any, out_rows: int, in_features: int, dtype):
        torch = _torch()
        ng = in_features // self.group_size
        if int(self.bits) == 4:
            # ★ packed 的最后一维是 in/2 个 uint8；先解包成 int8 的 codes。
            #   符号约定与 `dequantize_rtn` 一致（对称量化 → int4）。
            from lemoq.quant import pack as qpack  # noqa: PLC0415

            codes = qpack.unpack_4bit(codes, signed=True).to(torch.int32)
        c = codes.reshape(out_rows, in_features).to(torch.float32)
        s = scale.reshape(out_rows, ng, 1).to(torch.float32)
        w = (c.reshape(out_rows, ng, self.group_size) * s).reshape(out_rows, in_features)
        return w.to(dtype) if dtype is not None else w

    def dequant_gate_up(self, dtype: Any = None) -> Any:
        e, rows = int(self.packed_gate_up.shape[0]), int(self.packed_gate_up.shape[1])
        hidden = self.hidden
        return self._dequant_2d(
            self.packed_gate_up, self.scale_gate_up, e * rows, hidden, dtype
        ).reshape(e, rows, hidden)

    def dequant_down(self, dtype: Any = None) -> Any:
        e, rows = int(self.packed_down.shape[0]), int(self.packed_down.shape[1])
        ffn = self._unpacked_last(self.packed_down)
        return self._dequant_2d(self.packed_down, self.scale_down, e * rows, ffn, dtype).reshape(
            e, rows, ffn
        )

    def dequant_gate_up_expert(self, index: int, dtype: Any = None) -> Any:
        """**单个**专家的 gate_up（(2F, H)）。

        prefill 用：逐专家反量化。★ 千万不要用 `dequant_gate_up()[index]` ——
        那会先把全部专家物化成 fp32（OLMoE 上 1.07 GB），把省下来的字节又还回去。
        """
        codes = self.packed_gate_up[index]
        scale = self.scale_gate_up[index]
        return self._dequant_2d(codes, scale, int(codes.shape[0]), self.hidden, dtype)

    def dequant_down_expert(self, index: int, dtype: Any = None) -> Any:
        codes = self.packed_down[index]
        scale = self.scale_down[index]
        return self._dequant_2d(
            codes, scale, int(codes.shape[0]), self._unpacked_last(self.packed_down), dtype
        )


def quantize_fused_experts(fused: Any, *, bits: int = 8, group_size: int = 128):
    """把 `FusedExperts`（fp16 融合权重）就地量化成 `QuantFusedExperts`。

    调用方负责在之后释放 `fused` 的 fp16 张量（`replace_moe_blocks` 会随
    `del block.experts` 一起释放），否则显存会短暂翻倍。
    """
    import torch  # noqa: PLC0415

    from lemoq.quant import pack as qpack  # noqa: PLC0415
    from lemoq.quant import rtn as qrtn  # noqa: PLC0415

    if bits not in (4, 8):
        raise NotImplementedError("量化 grouped 路径只实现 4 / 8-bit，得到 %r" % (bits,))
    if group_size <= 0:
        raise ValueError("group_size 必须 > 0")
    gate_up = fused.gate_up
    down = fused.down
    hidden = int(gate_up.shape[2])
    ffn = int(down.shape[2])
    # ★ 不补零：kernel 的 k 循环没有掩码，补零列会让 scale 的组数对不上。
    #   OLMoE 的 H=2048 / F=1024 对 g=128 整除，所以这条约束不损失什么。
    for name, dim in (("hidden", hidden), ("ffn", ffn)):
        if dim % group_size != 0:
            raise NotImplementedError(
                "%s=%d 不能被 group_size=%d 整除：量化 grouped 路径不补零（kernel 无 k 掩码）。"
                % (name, dim, group_size)
            )

    q_gu = qrtn.quantize_rtn(gate_up.reshape(-1, hidden).float(), bits=bits, group_size=group_size)
    q_d = qrtn.quantize_rtn(down.reshape(-1, ffn).float(), bits=bits, group_size=group_size)

    def _codes_scale(q, rows_shape, in_features):
        padded = q["codes_padded"].reshape(
            int(q["codes_padded"].numel()) // in_features, in_features
        )
        if bits == 4:
            # 低半字节在前；pack_4bit 要求最后一维为偶数（group_size 偶数 ⇒ 成立）
            packed = qpack.pack_4bit(padded)
        else:
            packed = padded.to(torch.int8)
        codes = packed.reshape(*rows_shape, packed.shape[-1])
        scale = q["scale"].half().reshape(*rows_shape, in_features // group_size)
        return codes.contiguous(), scale.contiguous()

    packed_gu, scale_gu = _codes_scale(q_gu, tuple(gate_up.shape[:2]), hidden)
    packed_d, scale_d = _codes_scale(q_d, tuple(down.shape[:2]), ffn)
    return QuantFusedExperts(
        packed_gate_up=packed_gu,
        scale_gate_up=scale_gu,
        packed_down=packed_d,
        scale_down=scale_d,
        bits=int(bits),
        group_size=int(group_size),
    )


def dequantize_fused_experts(q: QuantFusedExperts, dtype: Any = None) -> Any:
    """`QuantFusedExperts` → `FusedExperts`（fp32 或缺省 dtype）。**只给 oracle / prefill 用。**"""
    from lemoq.kernels.moe_static import FusedExperts  # noqa: PLC0415

    return FusedExperts(
        gate_up=q.dequant_gate_up(dtype=dtype),
        down=q.dequant_down(dtype=dtype),
    )


# ---------------------------------------------------------------------------
# host 逻辑
# ---------------------------------------------------------------------------
def quantized_grouped_available() -> bool:
    """Triton 是否可用（Mac 上永远 False —— 那时只有 oracle 能跑）。"""
    try:
        from lemoq.kernels import _grouped_triton  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return False
    return True


def _kernels():
    try:
        import triton  # noqa: PLC0415

        from lemoq.kernels import _grouped_triton as k  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "量化 grouped kernels 需要 triton（本机不可用：%s）。"
            "oracle 路径（*_reference）不需要 triton，可在 CPU 上跑。" % type(exc).__name__
        ) from exc
    return triton, k


class _ShapeShim:
    """只带 `.shape` 的壳：复用 `moe_grouped.launch_plan` 的 grid 算术（不重复实现）。"""

    def __init__(self, shape: tuple) -> None:
        self.shape = tuple(shape)


def quant_launch_plan(
    q: QuantFusedExperts,
    *,
    top_k: int,
    config: dict[str, Any] | None = None,
    for_kernel: bool = True,
) -> dict[str, Any]:
    """量化路径的 grid/缓冲区形状 + 约束检查（违反就报错，不静默算错）。

    复用 `moe_grouped.launch_plan`（那里的 36 点网格扫描结论不该被复制一份），
    在其上补量化特有的检查：

        1. bits ∈ {4, 8}（4-bit 需要 group_size 为偶数）—— **总是**检查；
        2. hidden/ffn 被 group_size 整除（不补零）—— **总是**检查；
        3. `BLOCK_K` 整除 group_size（一个 k 块不跨组边界 → scale 是标量广播）；
        4. `dim % (SPLIT_K * BLOCK_K) == 0`（k 循环无掩码）。

    `for_kernel=False` 时**跳过 3/4**：它们只对"真的要启动 kernel"成立。
    oracle 路径（纯 PyTorch）不该被 kernel 的 grid 约束绑住 —— 否则一个小模型
    连 oracle 都跑不了（默认 BLOCK_K=128 是按生产 group_size=128 调的）。
    """
    # ★ 必须喂**未打包**的形状：`launch_plan` 用 (E, 2F, H)/(E, H, F) 推 hidden/ffn 与 grid，
    #   4-bit 的 packed 最后一维是 H//2 / F//2，直接用会静默算错 grid。
    e = int(q.packed_gate_up.shape[0])
    ffn = int(q.packed_gate_up.shape[1]) // 2
    hidden = q.hidden
    plan = launch_plan(
        SimpleNamespace(
            gate_up=_ShapeShim((e, 2 * ffn, hidden)),
            down=_ShapeShim((e, hidden, ffn)),
        ),
        top_k=top_k,
        config=config,
    )
    if int(q.bits) not in (4, 8):
        raise NotImplementedError("量化 grouped 路径只实现 4/8-bit，得到 %d" % int(q.bits))
    if int(q.bits) == 4 and int(q.group_size) % 2 != 0:
        # nibble 成对打包：组大小必须是偶数，否则一对 code 会跨组边界
        raise NotImplementedError("4-bit 要求 group_size 为偶数，得到 %d" % int(q.group_size))

    for name, tensor in (
        ("packed_gate_up", q.packed_gate_up),
        ("scale_gate_up", q.scale_gate_up),
        ("packed_down", q.packed_down),
        ("scale_down", q.scale_down),
    ):
        if not tensor.is_contiguous():
            raise ValueError("%s 必须连续（kernel 的偏移按连续布局算）" % name)
        if int(tensor.numel()) >= _MAX_OFFSET:
            raise ValueError("%s 的元素数超过 int32 偏移上限" % name)

    group = int(q.group_size)
    for name, dim in (("hidden", plan["hidden"]), ("ffn", plan["ffn"])):
        if dim % group != 0:
            raise ValueError("%s=%d 不能被 group_size=%d 整除" % (name, dim, group))

    if not for_kernel:
        plan["quant"] = {
            "bits": int(q.bits),
            "group_size": group,
            "for_kernel": False,
        }
        return plan

    cfg = merged_config(config)
    for tag, block_key, dim_keys in (
        ("gate_up_act", "gate_up_act", ("hidden",)),
        ("down_sum", "down_sum", ("ffn",)),
    ):
        blk = int(cfg[block_key]["BLOCK_K"])
        split = int(cfg[block_key]["SPLIT_K"])
        if group % blk != 0:
            raise ValueError(
                "%s 的 BLOCK_K=%d 必须整除 group_size=%d：否则一个 k 块会跨组边界，"
                "整块共用一个 scale 就不成立了。" % (tag, blk, group)
            )
        for dim_key in dim_keys:
            dim = plan[dim_key]
            if dim % (split * blk) != 0:
                raise ValueError(
                    "%s 的 %s=%d 不能被 SPLIT_K*BLOCK_K=%d 整除：kernel 的 k 循环无掩码。"
                    % (tag, dim_key, dim, split * blk)
                )

    plan["quant"] = {
        "bits": int(q.bits),
        "group_size": group,
        "packed_gate_up_bytes": int(q.packed_gate_up.numel()),
        "scale_gate_up_bytes": int(q.scale_gate_up.numel()) * int(q.scale_gate_up.element_size()),
    }
    return plan


# ---------------------------------------------------------------------------
# oracle（纯 PyTorch fp32，**不需要 triton / GPU**）
# ---------------------------------------------------------------------------
def grouped_gate_up_act_quant_reference(
    x: Any, q: QuantFusedExperts, ids: Any, *, top_k: int, ffn: int
) -> Any:
    """量化路径的 gate/up+激活 oracle：**先反量化**，再用与 fp16 路径同一个参考实现。

    这就是 kernel 必须匹配的数值（kernel 只是不物化那个 fp16 中间张量）。
    """
    from lemoq.kernels.moe_grouped import grouped_gate_up_act_reference  # noqa: PLC0415

    return grouped_gate_up_act_reference(
        x, q.dequant_gate_up(dtype=None), ids, top_k=top_k, ffn=ffn
    )


def grouped_down_sum_quant_reference(
    act: Any, q: QuantFusedExperts, ids: Any, top_w: Any, *, top_k: int
) -> Any:
    """量化路径的 down+加权 oracle（同上：先反量化再走 fp16 路径的参考实现）。"""
    from lemoq.kernels.moe_grouped import grouped_down_sum_reference  # noqa: PLC0415

    return grouped_down_sum_reference(act, q.dequant_down(dtype=None), ids, top_w, top_k=top_k)


# ---------------------------------------------------------------------------
# GPU 路径
# ---------------------------------------------------------------------------
def grouped_moe_forward_quant(
    x: Any,
    q: QuantFusedExperts,
    ids: Any,
    top_w: Any,
    *,
    config: dict[str, Any] | None = None,
    backend: str = "auto",
) -> Any:
    """`(H,)` hidden state + `(K,)` expert id + `(K,)` 路由权重 → `(H,)` 输出。

    与 `moe_grouped.grouped_moe_forward` 同一契约、同一 grid，区别只在权重是
    packed int8 + scale（kernel 内解包，不物化 fp16）。
    """
    torch = _torch()

    if backend not in ("auto", "triton", "oracle"):
        raise ValueError("unknown backend %r" % (backend,))
    use_oracle = backend == "oracle"
    if backend == "auto" and not quantized_grouped_available():
        raise RuntimeError(
            "量化 grouped kernels 需要 triton；本机不可用。"
            "用 backend='oracle' 可以只跑 host 管线 + 参考实现。"
        )
    if not use_oracle:
        # ★ 同上（`moe_grouped.grouped_moe_forward`）：`_gate_up_act_kernel_q` /
        #   `_down_sum_kernel_q` 的 store 也是硬编码 `.to(tl.float16)`。量化路径的 x
        #   会被转到 `scale_gate_up.dtype`，所以这里守的是 scale 的 dtype。
        #   位置同样在 `_kernels()` **之前**，理由同上。
        if q.scale_gate_up.dtype != torch.float16:
            raise NotImplementedError(
                "量化 grouped kernel 的 store 目前硬编码 fp16，只支持 fp16 scale；"
                "得到 scale_gate_up.dtype=%s。" % str(q.scale_gate_up.dtype).replace("torch.", "")
            )
        _triton, k = _kernels()

    xf = x.reshape(-1).contiguous()
    if xf.dtype != q.scale_gate_up.dtype:
        xf = xf.to(q.scale_gate_up.dtype)
    idx = ids.reshape(-1).contiguous()
    if idx.dtype != torch.int64:
        idx = idx.to(torch.int64)
    wsum = top_w.reshape(-1).contiguous()
    if wsum.dtype != xf.dtype:
        wsum = wsum.to(xf.dtype)

    top_k = int(idx.numel())
    plan = quant_launch_plan(q, top_k=top_k, config=config, for_kernel=not use_oracle)
    hidden = int(plan["hidden"])
    ffn = int(plan["ffn"])
    if int(xf.numel()) != hidden:
        raise ValueError(
            "grouped_moe_forward_quant 只处理单 token：x 有 %d 个元素，hidden 是 %d"
            % (int(xf.numel()), hidden)
        )
    if int(wsum.numel()) != top_k:
        raise ValueError("top_w 的长度 %d 与槽位数 %d 不符" % (int(wsum.numel()), top_k))

    dev = xf.device
    dt = xf.dtype
    group = int(q.group_size)

    # ---- gate_up + activation ----
    ga = plan["gate_up_act"]
    act = torch.empty(plan["gate_up_act"]["act_shape"], dtype=dt, device=dev)
    if use_oracle:
        act = grouped_gate_up_act_quant_reference(xf, q, idx, top_k=top_k, ffn=ffn).to(dt)
    else:
        split = int(ga["SPLIT_K"])
        if split == 1:
            part_g = act
            part_u = act
        else:
            part_g = torch.empty(ga["partial_shape"], dtype=torch.float32, device=dev)
            part_u = torch.empty(ga["partial_shape"], dtype=torch.float32, device=dev)
        k._gate_up_act_kernel_q[ga["grid"]](
            xf,
            q.packed_gate_up,
            q.scale_gate_up,
            idx,
            act,
            part_g,
            part_u,
            F=ffn,
            H=hidden,
            GROUP=group,
            BLOCK_N=int(ga["BLOCK_N"]),
            BLOCK_K=int(ga["BLOCK_K"]),
            SPLIT_K=split,
            K_TOKENS=top_k,
            BITS=int(q.bits),
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
        out = (
            grouped_down_sum_quant_reference(act, q, idx, wsum, top_k=top_k).to(dt).reshape(hidden)
        )
    else:
        dsplit = int(ds["SPLIT_K"])
        part = (
            out
            if dsplit == 1
            else torch.empty(ds["partial_shape"], dtype=torch.float32, device=dev)
        )
        k._down_sum_kernel_q[ds["grid"]](
            act,
            q.packed_down,
            q.scale_down,
            idx,
            wsum,
            out,
            part,
            H=hidden,
            F=ffn,
            GROUP=group,
            K_TOKENS=top_k,
            BLOCK_N=int(ds["BLOCK_N"]),
            BLOCK_K=int(ds["BLOCK_K"]),
            SPLIT_K=dsplit,
            BITS=int(q.bits),
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


def static_moe_forward_grouped_quant(
    hidden_states: Any,
    router_weight: Any,
    q: QuantFusedExperts,
    *,
    top_k: int,
    norm_topk_prob: bool,
    act_fn: Any,
    config: dict[str, Any] | None = None,
    backend: str = "auto",
) -> tuple[Any, Any]:
    """`OlmoeSparseMoeBlock.forward` 的量化 grouped 版本，契约与 `static_moe_forward` 相同。

    路由**复用** `moe_static._route`：两边的 top-k 选择逐位相同，任何差异都只能来自
    专家计算，不会和路由差异混在一起。

    `backend="oracle"` 让**没有 GPU / triton** 的机器也能跑通整条 host 管线
    （CPU 测试用），kernel 本体仍只在真机上过 K1 门。
    """
    from lemoq.kernels.moe_grouped import _check_act_fn  # noqa: PLC0415
    from lemoq.kernels.moe_static import _route  # noqa: PLC0415

    _check_act_fn(act_fn)

    shape = tuple(hidden_states.shape)
    hidden = shape[-1]
    tokens = 1
    for s in shape[:-1]:
        tokens *= s
    x = hidden_states.reshape(tokens, hidden)
    logits, top_w, top_i = _route(x, router_weight, top_k=top_k, norm_topk_prob=norm_topk_prob)
    out = grouped_moe_forward_quant(
        x,
        q,
        top_i.reshape(-1),
        top_w.reshape(-1),
        config=config,
        backend=backend,
    )
    return out.reshape(*shape), logits


def warm_quant(
    q: QuantFusedExperts,
    *,
    hidden: int,
    top_k: int,
    dtype: Any = None,
    device: Any = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """在 capture **之前**把量化 kernel 编译好（与 `moe_grouped.warm` 同理）。"""
    torch = _torch()
    dev = device if device is not None else q.packed_gate_up.device
    dt = dtype or q.scale_gate_up.dtype
    x = torch.zeros(hidden, dtype=dt, device=dev)
    ids = torch.zeros(top_k, dtype=torch.int64, device=dev)
    wsum = torch.full((top_k,), 1.0 / max(top_k, 1), dtype=dt, device=dev)
    grouped_moe_forward_quant(x, q, ids, wsum, config=config)
    if getattr(dev, "type", None) == "cuda":
        torch.cuda.synchronize()
    return {
        "hidden": hidden,
        "top_k": top_k,
        "bits": int(q.bits),
        "group_size": int(q.group_size),
        "dtype": str(dt),
        "device": str(dev),
    }
