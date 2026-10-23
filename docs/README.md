# docs/ — 文档地图

> 按用途阅读：**接手干活**看 `handoff/`，**查证口径与证据**看 `reference/`，
> **投递面试**看 `apply/`，**亲手吃透源码与实验**看 `study/`。仓库总入口是根目录 [README.md](../README.md)，
> 数字的唯一权威是 [RESULTS.md](../RESULTS.md) 与 [`experiments/`](../experiments/)。

## handoff/ — 交接（给下一个 agent）

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [HANDOFF_2026-10-23.md](handoff/HANDOFF_2026-10-23.md) | 当前 5090 验证、分支、工件及下一步 | 新接手的 agent / 人 |
| [ISSUES_2026-10-23.md](handoff/ISSUES_2026-10-23.md) | **关机后本地可解决的问题清单**（P0 采纳轮 / P1 协议 / P2 文档 / P3 需 GPU） | Mac 上接手的人 |
| [HANDOFF_2026-10-10.md](handoff/HANDOFF_2026-10-10.md) | 此前实验结论与历史交接 | 查此前状态的人 |

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
| [FUSION_RESEARCH.md](reference/FUSION_RESEARCH.md) | 联网调研及二次复核：融合/量化方法、避坑点、来源适用限制（公开报告值，非实测） | 做融合任务（F1–F6）前要外部依据的人 |
| [MOE_CUDAGRAPH.md](reference/MOE_CUDAGRAPH.md) | 为什么 HF 的 MoE 进不了 CUDA Graph（预判+依据） | 理解 p0-03 blocked 结论的人 |
| [SERVER_RUNBOOK.md](reference/SERVER_RUNBOOK.md) | 服务器到手后的命令序列 | 上机执行的人 |

## apply/ — 投递材料

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [PORTFOLIO.md](apply/PORTFOLIO.md) | 简历 bullet（中英）、面试必问三题标准答法、岗位映射与如实缺口 | 投递与面试前的你 |

## study/ — 两周源码与实验自学

| 文档 | 是什么 | 给谁 |
| --- | --- | --- |
| [README.md](study/README.md) | 14 天路线、掌握标准、Mac 练习边界 | 想自己讲清并修改本项目的你 |
| [01–05 专题](study/01-architecture.md) | decode 地图、量化、grouped MoE、dense/attention、norm 与候选融合 | 深入读源码的人；从 01 开始顺序阅读 |
| [CUDA Graph](study/06-cudagraph.md)、[profiling](study/07-profiling.md) | capture/replay 生命周期、图内归因与 A/B 判据 | 理解性能机制的人 |
| [简历追问](study/08-interview.md)、[答案](study/09-answers.md)、[真机练习](study/10-gpu-labs.md) | 证据索引、自测核对、将来可选的 5090 演练 | 面试准备与复盘 |

## 其它（不在本目录）

- [ROADMAP.md](ROADMAP.md) — **项目评估与扩展方向**（做得怎么样、下一步做什么；展望，不是状态）
- [FUSION_OPPORTUNITIES.md](FUSION_OPPORTUNITIES.md) — **融合优化的剩余空间：可执行任务规格**（F1–F6 按实测/估计分级、验收判据、八条注意事项红线、复算命令；给下一个 agent 干活用）
- 根目录：[README](../README.md)（入口）、[AGENTS](../AGENTS.md)（协作规则+文档索引）、
  [PLAN](../PLAN.md)（冻结计划）、[RESULTS](../RESULTS.md)（一页纸结果）、
  [EXPERIMENTS](../EXPERIMENTS.md)（记录格式）、[ENVIRONMENT](../ENVIRONMENT.md)（环境）
- [`figures/`](figures/) — 三张图，由 `python -m tools.make_figures` 从工件生成
- [`compose/spec/`](compose/spec/) — 本仓库 compose-next 流程的 feature 文档
- [`compose/spec/review-2026-10-22.md`](compose/spec/review-2026-10-22.md) — **review/debug 二次复核**（旧 P1–P9 状态纠正、新 R1–R12；后续修复状态见离线验证记录）
- [`compose/spec/offline-validation-2026-10-22.md`](compose/spec/offline-validation-2026-10-22.md) — **5090 前离线验证/修复**（CPU 反例、验收器与入口守卫修复、自然文本 GPU 检查脚本；待 GPU 回归项明确单列）
- [`compose/spec/quant-w4-calibration.md`](compose/spec/quant-w4-calibration.md) — **专家 W4 校准分支**（GPTQ/校准/退路实现、预量化加载、数据隔离、成本账；**GPU 已执行 → 负结果**，见 `experiments/EXP-p1-05-w4-expert-calibration.md`）
