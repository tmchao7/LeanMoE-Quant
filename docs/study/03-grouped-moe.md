# 03｜无 gather 的 grouped MoE GEMV

## 1. 问题与原理

单 token MoE 的 top-8 专家各有不同权重。HF 路径遍历全部 64 专家，`torch.where` 让输出长度依赖数据；静态 gather+bmm 虽可 capture，却先把权重/激活搬成大中间张量。grouped 路径把 expert id 留在 GPU：`id` 只算地址，**grid 不依赖 id 的值**。这同时去掉 gather 拷贝并保住 CUDA Graph 的静态 launch 形状。机制和证据见 [p2-02](../../experiments/EXP-p2-02-moe-grouped-gemm.md)、[MOE_CUDAGRAPH](../reference/MOE_CUDAGRAPH.md)。

它叫 grouped GEMV 更准确：`T=1`，每个 `(slot, expert)` 对算 `W[e]·x`，没有可复用的多行 A 矩阵。用 `tl.dot` 为 Tensor Core 填充 M 维会额外读权重；当前 `tl.sum(w*x, axis=K)` 对此带宽受限形状更合适。此取舍仅对本项目形状和硬件成立，不能推广到大 batch GEMM。

## 2. 张量、grid 与指针

[`moe_static.fuse_experts`](../../src/lemoq/kernels/moe_static.py) 把权重布局为：

```text
gate_up: (E, 2F, H) 连续；前 F 行是 gate，后 F 行是 up
down:    (E, H, F)  连续
x:       (H,)      ids: (K,)      top_w: (K,)
act:     (K, F)    out: (H,)
```

这里 `E=64`、`K=8`、`H=2048`；`F` 从实际 fused 张量推，不靠记忆。host 侧 [`launch_plan`](../../src/lemoq/kernels/moe_grouped.py) 纯 Python 算 grid：gate/up 为 `(K, ceil(F/BLOCK_N), SPLIT_K)`；down 为 `(ceil(H/BLOCK_N), SPLIT_K)`。若 `SPLIT_K>1`，还会有 partial 合并 kernel 和 fp32 缓冲区；所以“每层两个 kernel”只适用于两个 split 均为 1 的配置。

在 [`_gate_up_act_kernel`](../../src/lemoq/kernels/_grouped_triton.py) 中，一个 program 先读 `slot=program_id(0)`、`e=load(ids[slot])`；权重基址是 `w_ptr + e·(2F·H)`，gate 元素偏移为 `n·H+k`，up 为 `(F+n)·H+k`。`e` 不进入 Python 或 grid 计算。`n<F`、`k<k_end` 的 mask 保护边界；沿 K 用 fp32 累加并把 SiLU 与 gate×up 放在 epilogue，避免多次中间张量往返。

[`_down_sum_kernel`](../../src/lemoq/kernels/_grouped_triton.py) 对每个输出 `n`，静态遍历 K 个 slot：

```text
out[n] = Σ_slot top_w[slot] · Σ_f down[ids[slot],n,f] · act[slot,f]
```

top-K 加权和在 kernel 内完成，直接写 `(H,)`；不生成 `(K,H)` 临时结果。K 是 `constexpr`，循环次数固定。fp32 累加与原 gather 路径的 fp16 中间舍入不逐位相同，所以 kernel parity 与更高精度 oracle 比，而不是强求与 gather 逐位一致。

## 3. 量化专家是另一套权重对象

[`moe_grouped_quant.py`](../../src/lemoq/kernels/moe_grouped_quant.py) 先堆专家再做逐行 RTN。`QuantFusedExperts` 持有 `packed_gate_up`、`packed_down` 和各自的 fp16 scale；W8 code 是 int8，W4 的最后一维缩半。量化版 [`_gate_up_act_kernel_q` / `_down_sum_kernel_q`](../../src/lemoq/kernels/_grouped_triton.py) 在读 code 后立即乘 scale，不物化整块 fp16 权重。`BLOCK_K` 与 group 的对齐是硬约束：一个 tile 共享 scale，跨组而不正确选择 scale 会**静默算错**。

“先融合再量化”与“每个专家分别量化再融合”在这里可等价，因为 RTN 按行、按最后一维的组运算；堆叠不跨行。此结论由 [`test_moe_grouped_quant.py`](../../tests/test_moe_grouped_quant.py) 钉住。W4 kernel 已实现，但 W4A16 的模型 gate FAIL，不能将它的延迟作为性能成果。

## 4. 为什么 capture 安全，以及何时失效

固定 `K`、`F`、`H` 决定 grid 与临时 buffer，设备端 id 只改变权重地址。没有 `@triton.autotune`，第一次 JIT 必须在 capture 前完成。prefill 是多 token，不能直接走这个单 token GEMV。host 侧对不连续权重、超 int32 偏移、错误 dtype、shape 不匹配会显式拒绝；fp16 kernel store 当前硬编码 fp16，不能把 bf16/fp32 权重悄悄塞进去。实现契约见 [`grouped_moe_forward`](../../src/lemoq/kernels/moe_grouped.py)。

## 5. 证据怎么读

[p2-02](../../experiments/EXP-p2-02-moe-grouped-gemm.md) 的同轮 A/B，静态 gather 路径 **5.2637→4.0715 ms/token（1.293×）**，其字节账中的 gather 税约 **3.22 GB/token**。这个 A/B 是当轮、当实现的证据；P2j 融合 norm 后该格的绝对值刷新为 **3.6828 ms**，不能拿刷新值去除旧轮的 5.2637。最初默认 kernel 配置和两轮扫描曾给出误导，记录保留了失败经过；调参必须在同一协议下比较。

## 6. Mac 练习

1. 取假张量 `E=3,F=5,H=8,K=2,BLOCK_N=4,SPLIT_K=1`，手算 gate grid `(2,2,1)` 与 down grid `(2,1)`；再在纸上算 `e=2,n=3,k=4` 的 gate、up、down 元素一维偏移。
2. 运行 CPU host/oracle 测试，重点看非整除、索引与 dtype 拒绝：

```bash
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_moe_grouped.py tests/test_moe_grouped_quant.py
```

3. 读 [`launch_plan`](../../src/lemoq/kernels/moe_grouped.py)，解释若 gate `SPLIT_K=2` 为什么需要额外的 `part_g/part_u` 与 `act_combine`。不要在 Mac 上调用 Triton backend；使用 `backend="oracle"`。

## 7. 自测

1. expert id 为什么可以是运行时数据，而 `top_k` 不可随当前 token 改变 grid？
2. 为什么 down kernel 要把 routing 权重乘在 fp32 累加后？
3. “删除 3.22 GB gather 税”是否等于 HBM 实际流量少了 3.22 GB？

答案见 [09](09-answers.md#grouped-moe)。
