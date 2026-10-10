# docs/ — 文档地图

> 按读者分三层：**接手干活**看 `handoff/`，**查证口径与证据**看 `reference/`，
> **投递面试**看 `apply/`。仓库总入口是根目录 [README.md](../README.md)，
> 数字的唯一权威是 [RESULTS.md](../RESULTS.md) 与 [`experiments/`](../experiments/)。

## handoff/ — 交接（给下一个 agent）

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [HANDOFF_2026-10-10.md](handoff/HANDOFF_2026-10-10.md) | 当前状态的唯一权威摘要：分支、已实测数字、上机命令、纪律红线、坑位清单 | 新接手的 agent / 人 |

> 每次冲刺收尾后刷新（或新增带日期的一份并标注旧份过期）。过期交接一律删除——
> 历史在 git 里，不在文档树里。

## reference/ — 查证（口径 / 规格 / 证据 / 操作）

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [METRICS.md](reference/METRICS.md) | 指标定义与正确性阈值（KL/top-1 门的出处） | 复核任何数字的人 |
| [SPECS.md](reference/SPECS.md) | 机器规格、跨代特性矩阵、选卡理由 | 要确认硬件适用区间的人 |
| [MODELS.md](reference/MODELS.md) | 模型选型与已核实的 OLMoE 架构、字节预算 | 要确认模型事实的人 |
| [NCU_EVIDENCE.md](reference/NCU_EVIDENCE.md) | kernel 级证据：计数器为何被阻断、替代口径 | 深读性能归因的人 |
| [NCU_PLAN.md](reference/NCU_PLAN.md) | 换机才能做的 ncu 补测清单（当前 blocked） | 将来拿到 profiling 权限的人 |
| [MOE_CUDAGRAPH.md](reference/MOE_CUDAGRAPH.md) | 为什么 HF 的 MoE 进不了 CUDA Graph（预判+依据） | 理解 p0-03 blocked 结论的人 |
| [SERVER_RUNBOOK.md](reference/SERVER_RUNBOOK.md) | 服务器到手后的命令序列 | 上机执行的人 |

## apply/ — 投递材料

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [PORTFOLIO.md](apply/PORTFOLIO.md) | 简历 bullet（中英）、面试必问三题标准答法、岗位映射与如实缺口 | 投递与面试前的你 |

## 其它（不在本目录）

- [ROADMAP.md](ROADMAP.md) — **项目评估与扩展方向**（做得怎么样、下一步做什么；展望，不是状态）
- 根目录：[README](../README.md)（入口）、[AGENTS](../AGENTS.md)（协作规则+文档索引）、
  [PLAN](../PLAN.md)（冻结计划）、[RESULTS](../RESULTS.md)（一页纸结果）、
  [EXPERIMENTS](../EXPERIMENTS.md)（记录格式）、[ENVIRONMENT](../ENVIRONMENT.md)（环境）
- [`figures/`](figures/) — 三张图，由 `python -m tools.make_figures` 从工件生成
- [`compose/spec/`](compose/spec/) — 本仓库 compose-next 流程的 feature 文档
