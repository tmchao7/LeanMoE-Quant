# 01｜一个 token 怎么穿过项目

## 1. 问题与原理

本项目测的是 **OLMoE-1B-7B-0924、batch=1、单流 decode**，不是在线服务吞吐。模型有 16 层、每层 64 专家、每 token 选 8 个；`hidden=2048`。prefill 消化整个 prompt，之后每步只输入一个 token，但要读大量被选中的权重与不断增长的 KV cache。模型和硬件事实见 [MODELS](../reference/MODELS.md)、[RESULTS](../../RESULTS.md)；RTX 5090 的**实测**可达带宽为 1561.9 GB/s，不能拿规格带宽算屋顶。

```text
input token, cache_position, StaticCache
  ↓ 16 × decoder layer
  RMSNorm → q/k/v projection → q/k norm → RoPE → KV-cache update
  → attention core → o_proj → residual
  → RMSNorm → router(top-8) → 8 experts(gate/up/down) → residual
  ↓ final norm → lm_head → logits → greedy token（图外）
```

权重读量 `[W]` 与 context 无关，KV 与注意力成本随 context 增加；`[T]` 总字节因而不是 `[W]`。两种分母的定义见 [METRICS §3.2](../reference/METRICS.md)。这也是 W8A16 在 1024/4096 的收益不同的根本背景。

## 2. 源码路线：从 CLI 到一个 decode 步

| 站点 | 读什么 | 要找到的动作 |
| --- | --- | --- |
| [`configs/experiments.yaml`](../../configs/experiments.yaml) | `EXP-p4b-01`、`EXP-p4c-01` | 声明 cell 的 bit-width、execution、attention、ctx；不要自行改轴 |
| [`benchmarks/runner.py`](../../benchmarks/runner.py) | `main` → `run_cudagraph` / `run_baseline` | 解析配置、引用正确性门、替换模块、建 session、产 payload |
| [`moe_static.py`](../../src/lemoq/kernels/moe_static.py) | `replace_moe_blocks` → `_route` → `_decode_forward` | 将 HF 动态逐专家循环变成固定形状的 MoE 前向 |
| [`attn_decode.py`](../../src/lemoq/kernels/attn_decode.py) | `replace_attention_cores` → `_decode_forward_impl` | 只替换 decode attention core，保留投影、RoPE、cache、o_proj |
| [`rmsnorm.py`](../../src/lemoq/kernels/rmsnorm.py) | `replace_rms_norms` | 默认融合 norm；实例级替换，可恢复 |
| [`cudagraph.py`](../../benchmarks/cudagraph.py) | `StaticCacheDecodeSession` | prefill、固定缓冲区、capture、replay、计时 |
| [`schema.py`](../../benchmarks/schema.py) | `validate_result` | gate FAIL 时性能段不得引用 |

`runner.run_cudagraph` 在 capture **前**完成 MoE 替换、量化、attention/norm 替换与 Triton warmup。capture 后图重放不再经过这些 Python `forward` 分派；改变 env 开关不会改已捕获的图，须新建 session 再 capture。读这段代码时，用箭头写下每次替换的前置条件。

## 3. 三条路径的区别

| 路径 | 权重与算子 | 执行 | 状态 |
| --- | --- | --- | --- |
| fp16 + HF MoE | fp16 权重、逐专家 `torch.where` | eager | 起点；动态形状使原版 graph 不可捕获，见 [p0-03](../../experiments/EXP-p0-03-baseline-cudagraph.md) |
| fp16 + grouped MoE | fp16 融合专家权重；expert id 仅作设备指针偏移 | eager 或 cudagraph | 作为 E7 的同实现对照；见 [p2-02](../../experiments/EXP-p2-02-moe-grouped-gemm.md) |
| W8A16 + grouped MoE | int8 packed codes + fp16 scale，kernel 内解包/反量化 | cudagraph 为主要发表路径 | 过当前 gate，图内收益见 [p4b-01](../../experiments/EXP-p4b-01-truncation.md) §P2j |

“四个自研 Triton kernel 族”指已采纳的 grouped MoE、dense dequant、split-KV attention、RMSNorm。RoPE 与 routing 融合是**另两个候选**，环境变量默认关；它们的 A/B 工件独立于已录的矩阵 cell。见 [PORTFOLIO §2a](../apply/PORTFOLIO.md)。

## 4. 结果是怎么来的

矩阵 cell id 由 axes 生成。`benchmarks.runner` 先检查配置与 `GATE_CITATIONS`，正确性失败时 [`latency_section_for_gate`](../../benchmarks/runner.py) 把性能段置 `null`，测到的值只能作为 `gate_failed_evidence`。`result.json` 要有 `status=done`，记录要有结果节，才能说 cell 跑过；历史 precision 实验的 `planned` skeleton 不代表科学没做，先读记录。

性能主张还要看：**同轮 A/B**、首尾同格金丝雀、GPU 是否被共租进程污染、ctx/batch/dtype、记录采用的是哪版。`RESULTS.md` 的五条杠杆来自不同轮次，不能相乘成“总加速”。

## 5. 证据与 Mac 练习

从 [RESULTS §2](../../RESULTS.md) 找五条杠杆的出处，再去每份记录找采用节。执行下面的只读小检查，观察 cell 的身份与 gate：

```bash
.venv-mac/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('results/EXP-p4b-01-truncation__bit_width=w8a16_execution=cudagraph_ctx=1024/result.json')
d = json.loads(p.read_text())
print('id:', d['experiment_id'] if 'experiment_id' in d else d.get('exp_id'))
print('status:', d['status'])
print('config:', d['config'])
print('correctness keys:', sorted(d['correctness']))
PY
```

如果这个机器没有历史工件，只读同名实验记录，不新建 `result.json`。用纸画出 `run_cudagraph` 的“替换→prefill→warmup→capture→replay→payload”顺序，并在每个箭头旁标出是否会读 Python env。

## 6. 自测

1. 为什么 `ctx=4096` 会压低量化端到端收益，即使专家权重的字节数不变？
2. 为什么 RoPE 候选的 +8.4% 不能加在 `EXP-p4c-01` 的 v3 数字上？
3. 从 `--attn kernel` 到 `_attn_split_kernel`，至少经过哪三个函数/模块？

答案见 [09](09-answers.md#架构)。
