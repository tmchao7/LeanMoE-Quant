# 06｜CUDA Graph 怎样进入 OLMoE decode

## 1. 问题与原理

CUDA Graph 记录一组 GPU 操作和地址，再由 host 一次 replay。它减少逐 kernel 的 Python/CUDA launch 间隔；**不会把图里的数百个 kernel 自动融合成一个 kernel**。本项目的同轮 fp16 grouped 对照为 eager **9.8841 ms** → graph **4.0670 ms**（2.43×，ctx=1024），host share 67.3%→1.4%，见 [p4b-01 §2.8](../../experiments/EXP-p4b-01-truncation.md) 和 [RESULTS §2](../../RESULTS.md)。这个 A/B 属于早期同轮实现，不用后续融合 norm 的 graph 数字与旧 eager 交叉相除。

图的三个前提：**形状固定、地址稳定、capture 时没有不允许的 host 同步/数据依赖形状分配**。数值可以变：同一地址里的 token、position、KV 内容变了，replay 的结果也会变。

## 2. 源码：固定地址的输入

[`CudaGraphConfig`](../../benchmarks/cudagraph.py) 要求 batch=1、q_len=1；容量 `max_cache_len=ctx+max_new_tokens`。[`GraphBuffers.allocate`](../../benchmarks/cudagraph.py) 一次分配 token `(1,1)`、`cache_position`、`position_ids`、4D attention mask、`next_token`、`filled_len:(1,) int32`。`set_step()` 用 `fill_` **原地改值**，`addresses()` 用 `data_ptr()` 验证地址未变。

split-KV attention 的 `filled_len` 在图外按 position 写入，kernel 在设备端读，因而可在固定 launch grid 下处理不同有效长度。注意 `mask` 和 `filled_len` 的更新操作本身在图外；性能记录的 wall 口径是否包含这些工作，应按 [`measure_graph_gap`](../../benchmarks/cudagraph.py) 的实际计时窗口解释。

## 3. 生命周期：每一步在哪里执行

[`StaticCacheDecodeSession`](../../benchmarks/cudagraph.py)：

```text
构建 model/session
  → prefill(): eager 处理 prompt，初始化 StaticCache 与地址
  → capture(): 复位 → 专用 stream warmup/JIT → 再复位 → 捕获 1 个 q_len=1 forward
  → replay(): 原地更新输入/位置 → graph.replay() → 同步 → 图外 argmax/.item()
  → position+1，继续下一步
```

prefill **不在图里**。`capture()` 在 warmup 与捕获前复位，是为了避免之前的 eager 参考步把 position 推到 cache 尾部。JIT 首次编译、权重替换、quantization 都要在 capture 前完成。`replay(synchronize=False)` 用于计时路径，不推进 host token；正常 `replay()` 会同步并从 logits 算 greedy token。`replay_at_position()` 可以在不重新捕获的情况下改设备端有效长度，用于“换长度仍算对”的验证。

## 4. 为什么 HF MoE 捕获失败，grouped 能捕获

HF 原版每层遍历 64 专家并用 `torch.where(expert_mask)` 找当前专家收到的 token；输出个数依赖路由数据，触发 device→host count 和动态分配。基线 profile 中每步 **1024 次 DtoH = 64×16**，见 [MOE_CUDAGRAPH](../reference/MOE_CUDAGRAPH.md) 与 [p0-03](../../experiments/EXP-p0-03-baseline-cudagraph.md)。即使某次捕获侥幸成功，换 token 后专家分布变化还可能静默算错，因此必须测 replay 正确性。

grouped 路径固定 K=8 个 slot，grid 与 buffer 形状不随专家 id 变；kernel 在运行时读 id，只用于权重指针偏移。attention 长度类似：设备值可变，grid 固定。关键区别是**设备端循环/地址的数据依赖允许，host 端新张量形状/launch 拓扑的数据依赖不允许**。

## 5. 计时与正确性

[`measure_graph_gap`](../../benchmarks/cudagraph.py) 对每个 repeat 先 reset、warm，再 reset，随后用 `perf_counter` 得 wall、CUDA event 得 GPU 时间区间，另用 `torch.profiler` 的 self device time 求 kernel busy 与每步 kernel 数。`host_share≈(wall−busy)/wall` 是诊断量，跨工具测量的残差有偏差，别当“精确 Python 耗时”。图内仍可有 **539 kernel/步**（融合 norm 后的本地工件 `results/graph_profile.json`）；“一次 graph replay”只是 host 发起一次捕获图的重放。

正确性分两层：图 vs 同实现 eager 的 logits/轨迹；优化实现 vs 参考实现的 kernel K1 或 prompt-set gate。只测 `capture_success=True` 不能证明 replay 忠实，也不能证明 W8A16 对 fp16 的精度门已过。

## 6. Mac 练习

用 CPU `GraphBuffers` 体验“值变、地址不变”，该命令不会调用 CUDA：

```bash
PYTHONPATH=src .venv-mac/bin/python - <<'PY'
import torch
from benchmarks.cudagraph import GraphBuffers
b = GraphBuffers.allocate(torch.device('cpu'), batch=1, max_cache_len=8)
before = b.addresses()
b.set_step(token=7, position=3)
print('same addresses:', before == b.addresses())
print('token/position/filled:', b.token.item(), b.cache_position.item(), b.filled_len.item())
PY
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_cudagraph.py tests/test_runner_cells.py
```

预期为 `True` 与 `7 3 4`；若把 `b.token` **重新赋值**为新张量，地址契约会破坏。纸上追踪 `capture()` 中两次 `prefill()` 的目的，并指出真正包在 `torch.cuda.graph` 中的代码块。

## 7. 自测

1. 为什么“1.9 万个 eager kernel launch 收敛成一次图重放”不等于 GPU 只执行一个 kernel？
2. 改变 `filled_len` 与改变 `q_len`，哪个可以复用同一个图？为什么？
3. prefill、greedy argmax、权重量化分别在图内还是图外？

答案见 [09](09-answers.md#cuda-graph)。
