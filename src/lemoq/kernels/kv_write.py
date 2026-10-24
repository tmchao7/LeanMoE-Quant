"""kv_write — F4（KV 直写）的语义契约与设备无关参照。

为什么存在：`docs/FUSION_OPPORTUNITIES.md` §F4 指出 KV 写入是纯 launch 开销
（`index_elementwise_kernel ×32` + `CatArrayBatchedCopy ×32` ≈ 0.109 ms/step，
平均 1.7 µs 只为写 8 KB），把写融进 fused RoPE 或 attention kernel 可以整段拿掉。
但 F4 的风险清单写得很清楚：**写错位置不会崩，只会让 attention 读到旧值**——
所以先把语义契约钉死（离线、任意设备可测）：

    给定 pos 与新 k/v，直写后的 cache 内容必须与 HF `StaticCache.update()`
    （transformers 4.57.6：`StaticLayer.update` 即 `keys.index_copy_(2, cache_position, …)`
    并返回**整段** buffer）逐位相同，且不得触碰 pos 之后未填充的槽位。

`write_kv_slot` 是 CPU/任意设备的参照实现；F4 落地时 Triton kernel 或 fused-RoPE
直写替换其函数体，`tests/test_kv_direct_write.py` 原样作为 K 门在 GPU 上重跑。

transformers 版本锚：4.57.6。`StaticCache.layers[i].{keys,values}` 与
`lazy_initialization` 是 HF 的内部布局——上游改动时测试会响，这正是 F4 风险
清单（"最容易被上游破坏的一处"）要的绊线，不要用 getattr 软化它。
"""

from __future__ import annotations

from typing import Any


def write_kv_slot(
    cache: Any,
    layer_idx: int,
    pos: int,
    key_states: Any,
    value_states: Any,
) -> tuple[Any, Any]:
    """把 `(b, h, q_len, d)` 的新 KV 直写进 `cache.layers[layer_idx]` 的 `[pos:pos+q_len]`。

    语义 = HF `StaticLayer.update` 的 `index_copy_(2, cache_position, …)`：

    - 原地写**同一块** buffer（返回的对象就是 `layer.keys/layer.values` 本身——
      buffer 地址不变是 CUDA Graph replay 的前提）；
    - 只触碰 `[pos, pos+q_len)`，其余槽位（包括 pos 之后未填充区）不动；
    - 惰性初始化行为与 HF 一致（首次写入前 buffer 可能未分配）。

    F4 的 kernel 版必须通过 `tests/test_kv_direct_write.py` 的同一组断言。
    """
    layer = cache.layers[layer_idx]
    if not layer.is_initialized:
        layer.lazy_initialization(key_states)
    q_len = key_states.shape[2]
    layer.keys[:, :, pos : pos + q_len, :] = key_states
    layer.values[:, :, pos : pos + q_len, :] = value_states
    return layer.keys, layer.values
