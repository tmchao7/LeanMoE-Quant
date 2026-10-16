# 07｜从 profiler 表到可检验的融合假设

## 1. 问题与原理：四种时间别混

| 量 | 代码入口 | 解释 |
| --- | --- | --- |
| wall p50 | [`measure_graph_gap`](../../benchmarks/cudagraph.py) 的 `perf_counter` | 每个计时块折算的单步主机可见时间；含图外推进/同步等工作 |
| CUDA event p50 | 同函数 `torch.cuda.Event` | stream 上的时间区间；与 profiler busy 不是同义词 |
| GPU busy | `torch.profiler` 的 `self_device_time_total` 逐 kernel 求和 | GPU kernel 自身忙时间；可按 kernel 名归因 |
| host share | `(wall−busy)/wall` 残差 | 定位 host/launch 的近似量；不是 Python 帧的直接计时 |

本地历史工件 `results/graph_profile.json` 为 `status=measured`：wall **1.9397 ms**、busy **1.8756 ms**、host share **3.3%**、**539 kernel/步**。这是另一轮定位工件，不能直接和 p4b/p4c 的 cell 数字相除。要看原始 top 表，用 [`tools/profile_graph.py`](../../tools/profile_graph.py) 的 `print_table`；工具调用 `run_cudagraph`，不写矩阵的 `result.json`。

## 2. 定位过程：从名字到功能块

直接按 self time 排序，最大的是量化专家 `_gate_up_act_kernel_q`（约 0.444 ms/步，16 次），其次 dense dequant（约 0.325 ms，64 次）、down_sum（约 0.259 ms，16 次）。这些是大桶。要找**可融合**热点，还需把属于同一功能的多个名字汇总。

RoPE 的 `rotate_half + cos/sin` 链分散在 `neg_kernel` 32 次、`CatArrayBatchedCopy` 32 次、`BinaryFunctor` mul 64 次及 add 的一部分约 32 次，合计约 **160 kernel / 0.216 ms / 11.5% busy**。这不是 profiler 自动给出的单行，而是用 HF 算子语义、每层调用次数和 kernel family 做的归因；add 的“RoPE 份额”尤其是估计。将它作为**优化假设**，随后用 fused RoPE 的 A/B 检验：busy 减约 0.194 ms，与估算同量级，见 [RoPE spec](../compose/spec/fused-rope.md)。

[`profile_graph_scopes.py`](../../tools/profile_graph_scopes.py) 用 eager 的 `record_function` 尝试模块归因，再与 graph kernel 表核对。**图 replay 不经过 Python `forward`**，不能直接把 Python 作用域贴到 graph 内核。该实验预注册的 K1（eager/graph top-8 一致性）只重叠 5/8，故模块级归因 **VOID**；kernel 级分类仍可用，见 [p4c-02 §4/§5](../../experiments/EXP-p4c-02-elementwise-tail.md)。这是一道重要面试题：仪器不通过自证时，必须缩小结论。

## 3. A/B：优化有没有兑现

[`tools/profile_graph.py::_run_env_ab`](../../tools/profile_graph.py) 在同一进程里按 `off,on,off,on` 交替。三件必须一起看：基线首尾 wall 漂移 ≤5%、跨臂 token 序列一致、wall 的同轮均值差。RoPE 的第一轮金丝雀 **5.923%**，按协议 **VOID**；第二轮漂移 **3.661%**、tokens 相同，wall **1.9655→1.8007 ms**，A/B **+8.39%**。第二轮 busy 两臂各自很稳定，但 3/4 run host share >5%，讲结果时应一并说明限制。routing 的对应工件为 `aten/fused`、漂移 **2.215%**、wall **1.9521→1.8935 ms（+3.0%）**。两者均为**候选**，不能说默认路径已经提速。

矩阵 cell 的协议更严格：正确性 gate PASS、`result.json`、记录、同轮 A/B、首尾金丝雀；graph cell 还需自己的 canary 和 `host_share≤5%` 污染检查。逻辑见 [`round_canary.py`](../../tools/round_canary.py) 与 [AGENTS §6](../../AGENTS.md)。候选定位工具的 `status=measured` **不是**矩阵 cell `status=done`，别混用两套发表口径。

## 4. ncu 被阻断后能和不能说什么

当前 5090 容器的 ncu 被宿主策略拒绝（`RmProfilingAdminOnly=1`、缺 `CAP_SYS_ADMIN`），连 `gpu__time_duration.sum` 都拿不到，见 [NCU_EVIDENCE §1](../reference/NCU_EVIDENCE.md)。[`ncu_evidence.py`](../../tools/ncu_evidence.py) 的替代方法用**解析必要字节 ÷ 隔离 cold 时间（扣事件地板）÷ 实测可达带宽**算 attainment，并在计时前冲掉 L2。它能回答“离带宽墙多远”，不能回答 warp stall、真实 achieved occupancy 或 memory sectors 的原因。解析 occupancy 上界见 `results/occupancy.json`，不是 ncu 实测 occupancy。

定位顺序：先看 wall/busy 与 kernel 数，确认 host 还是设备；再按功能块汇总 top kernel，估计优化头寸；最后写只改一个变量的 A/B、parity、canary。不要“看起来是热点”就写速度结论。

## 5. Mac 练习：读历史 JSON

```bash
.venv-mac/bin/python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('results/graph_profile.json').read_text())
print('wall/busy/host/kernels:', p['wall_p50_ms'], p['gpu_busy_ms'], p['host_share_pct'], p['kernels_per_call'])
for r in p['gap']['top_kernels'][:3]:
    print(r['name'][:40], r['self_ms_per_token'], r['occurrences_per_token'])
for name in ('fused_rope_ab', 'routing_ab'):
    d = json.loads(Path('results', name + '.json').read_text())
    print(name, d['status'], d['tokens_equal'], d.get('canary_drift_pct', d.get('canary_aten_drift_pct')))
PY
.venv-mac/bin/python -m pytest -m "not gpu" -q tests/test_profile_graph.py tests/test_round_canary.py
```

再在纸上造一轮 canary：首值 2.00 ms、尾值 2.12 ms，算漂移 `+6%`，判 VOID；不能把中间 treatment 的快值保留下来。对 RoPE 第 2 轮，直接用 JSON 的四个 `runs` 重算各臂均值和百分比，解释为什么不能只拿第一个 off 除以第一个 on。

再做一次**真正的 profiler API 演练**。Mac 只有 CPU，因此只学会 `record_function`、`key_averages()`、`count`、`self_cpu_time_total` 与 `cpu_time_total` 的关系；输出绝不当 5090 证据：

```bash
.venv-mac/bin/python - <<'PY'
import torch
from torch.profiler import ProfilerActivity, profile, record_function
x = torch.randn(128, 128)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function('study.matmul_relu'):
        y = torch.relu(x @ x)
for e in prof.key_averages():
    if e.key in ('study.matmul_relu', 'aten::mm', 'aten::relu'):
        print(e.key, 'count=', e.count,
              'self_cpu_us=', round(e.self_cpu_time_total, 1),
              'total_cpu_us=', round(e.cpu_time_total, 1))
PY
```

预期三行均 `count=1`；具体时间不稳定。`study.matmul_relu` 的 total 包含内部算子，self 不包含；**不能把父作用域 total 与子算子 total 相加**。GPU 路径则在 [`_profiler_rows`](../../benchmarks/cudagraph.py) 取 `self_device_time_total`，并除以步数转为每步毫秒。

## 6. 自测

1. `gpu_event_p50`、`gpu_busy_ms` 与 `wall_p50_ms` 为什么不能互换？
2. eager 模块归因 K1 失败后，哪些结论仍能保留？
3. 为什么 ncu 不可用时不能声称“寄存器 stall 是瓶颈”？
4. 一个候选 A/B `tokens_equal=True` 但金丝雀 +6%，状态是什么？

答案见 [09](09-answers.md#profiling)。
