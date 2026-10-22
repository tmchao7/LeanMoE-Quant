# 5090 前离线验证与修复 — 2026-10-22

承接 [review R1–R12](review-2026-10-22.md)。这是工具/契约修复记录，不是新实验：
矩阵轴、cell id、引用门、默认融合开关和历史结果不变；没有连接 GPU、安装依赖或重跑 done 格。
基于 `70f4610` 的工作区，保留进入本轮前已有的 RoPE/测试改动，未提交或推送。

## 已执行的验证

先新增针对性反例，再改实现。首次运行 `tests/test_offline_review.py` 是 **38 failed / 13 passed**：
NaN 放行、身份错配、空 token 被当成相等、异常指标崩溃、尾 canary/环境隔离缺口、布局契约、
空 routing launch 和错误设备资源兜底均有 CPU 或 mock 证据。
这些是参数化反例数，不是 38 个不同缺陷。另两项 mask 能力测试在修复前 **2 failed / 7 passed**。

新增独立数值反例确认：

- 两组专家的路由权重完全相等，换掉专家仍会改变加权输出；这证明旧“等权即等价”理由不成立，
  **尚未证明 sm_120 的实际 ATen/Triton 选择差异**。
- `fp16(fp16(acc)+residual)` 与 `fp16(acc+residual)` 有可重复的舍入反例；没有实施残差 epilogue。
- 刻意让新加入的 KV 有贡献后，陈旧 mask 与新 mask 的 attention 输出不同。

验证命令（Mac，已有 `.venv-mac`）：

```bash
.venv-mac/bin/ruff check .
.venv-mac/bin/black --check .
.venv-mac/bin/python -m pytest -m "not gpu" -q
.venv-mac/bin/python -m tools.check_docs
git diff --check
```

最终测试结果见本文件末尾的验收记录。Mac 不产出可报告的性能数字。

## 修复状态

| Review | 当前状态 | 实施内容与剩余边界 |
| --- | --- | --- |
| R1 | 工具修复，CPU 通过 | 每个 run（含尾 canary）host-share >5% 都使整轮 invalid；旧工件不改写 |
| R2 | 工具修复，CPU 通过 | 拒绝 NaN/Inf/bool/非法域；比较 exp_id、config、model、runtime 与可用硬件身份；阈值不能放宽至 >5%；精确 5% 保持合法 |
| R3 | 工具修复，CPU 通过 | token 必须为声明长度的非空合法序列，同时比较 seed；缺证据不能算相等 |
| R4 | 工具修复，CPU 通过 | 顺序 A/B × N + 独立尾 A；尾 A 不进 arm 汇总；保留均值，新增 median/MAD 和逐对差/胜率；不据此自动判断显著性 |
| R5 | 工具修复，CPU 通过 | 三个候选 env 显式固定，每 run 记录实际值，异常也恢复原环境；记录版本、commit、dirty 与含未跟踪代码的源码哈希 |
| R6 | 入口收窄，待 GPU 回归 | 混合 dtype/device、cos/sin 不同 stride、零维/空 head 不进入 Triton；保留 HF 回退；修正 position_ids 错误注释 |
| R7 | 部分修复，待 GPU | 拒绝不支持的 K/V 序列/head stride、batch/head_dim/device 错配；统一 oracle 长度检查；**CUDA scalar 的值上界仍依赖调用方，尚未改 kernel** |
| R8 | 数学反例成立，待 GPU 复现 | 删除等权专家可交换的错误说明；准备全等、K 边界平局、fp16 舍入三组实际 MoE 输出比较；没有自行修改 tie 策略 |
| R9 | 空输入修复，CPU 通过 | T=0 在 CUDA 初始化/JIT/launch 前返回空输出；非空路径仍要求 CUDA |
| R10 | 能力约束修复，待 GPU 回归 | skip-mask 必须绑定捕获时已确认的全部 OLMoE Triton decode forwards；SDPA A/B 提前拒绝；不满足能力时更新 buffer 前报错 |
| R11 | 证据与脚本准备，GPU 未运行 | RoPE oracle 相对误差同时检查 q/k；新增独立数值反例与自然文本检查脚本；没有宣称新候选过门 |
| R12 | 工具修复，CPU 通过 | sm_120 fallback 改为 128 KiB/SM，未知设备信息不足时报错；只有三族全部有估计才 done，部分结果保留并报告 coverage |

**R7 未修的部分**：禁止用 CUDA `.item()` 逐步同步去掩盖图约束；非法设备长度不能在普通进程中
盲试越界。后续先用已有 compute-sanitizer 或独立隔离进程确认，再决定设备侧拒绝/安全处理方式。
正常 session 已有容量检查，但公共 helper 的设备值契约仍不完整。

R12 历史工件输入为 `smem_per_sm=102400`，没有使用错误兜底；本轮没有重新判定历史三族上界。

## 新的诊断输出约定

[`tools/profile_graph.py`](../../../tools/profile_graph.py) A/B 输出 `protocol_version=2`：
`status=measured` 仅表示这轮定位采集满足当前工具检查；`adoption_verdict=not_evaluated` 明示
**不能单凭它采纳候选**。退化轨迹仍保留警告，正式独立正确性门不得省略。
默认两对依然只适合探索，不把小收益宣布为显著；数据低于噪声时应记录 inconclusive。

每轮最后一个 run 为 `role=tail_canary`，其正确性/host-share 也受检查，但不进两臂均值、MAD 和配对数。
其他候选开关固定为 0；若需要 combined 背景，应另行明确设计，不能从 shell 环境偷偷继承。
输出文件已存在时拒绝覆盖，使用新路径保存新轮次。

## 已准备的 5090 检查入口（尚未执行）

GPU 反例已能正常 collection：本轮选定的四份测试文件收集到 **23 项 GPU 测试**。
路由平局测试若失败，就是候选正确性失败，必须停止该候选的性能采纳；没有 xfail 掩盖。

```bash
# 在确认实例/路径之后执行；本轮未运行这些命令。
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 非空就记录并等待，不杀进程、不安装升级。
env LEMOQ_FUSED_ROUTE=0 LEMOQ_FUSED_ROPE=0 LEMOQ_SKIP_MASK_FILL=0 \
  /root/miniconda3/bin/python -m pytest -m gpu -q \
  tests/test_fusion_adversarial.py tests/test_fused_rope.py \
  tests/test_routing_fusion.py tests/test_attn_decode.py
```

新脚本 [`tools/check_fusion_candidate.py`](../../../tools/check_fusion_candidate.py) 默认 dry-run，
只读已有本地模型与 wikitext token 缓存，**不下载、不测性能、不更新 GATE_CITATIONS**。
明确范围为 w8a16/g128/fused × grouped × kernel × fused RMSNorm × batch=1。

其独立参考保留 HF attention/MoE，RMSNorm 与当前项目默认实现一致。先在同一模型权重变换前保存
fp16 HF+StaticCache 的 5 窗口 greedy 输入轨迹，再量化/替换当前路径。两臂 teacher forcing
使用完全相同的 HF 输入，另测自由 greedy，避免轨迹分叉后误算逐位置 KL。
模型只在 GPU 上驻留一份；不做双模型并驻，也不需要多卡。

```bash
# 新路径；不要用已有 results/*.json 或 result.json。
/root/miniconda3/bin/python -m tools.check_fusion_candidate \
  --model /root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master \
  --prompt-cache /root/autodl-tmp/lemoq-data/wikitext2_test_8k.pt \
  --candidate rope --ctx 1024 --steps 32 --windows 5 \
  --json /tmp/fusion-rope-natural-NEW.json --dry-run
```

GPU micro tests 通过、实例空闲且路径确认后，才把 `--dry-run` 换成 `--execute`。
`--candidate` 每次只选 `rope` / `routing` / `mask` 一项，输出各用新路径。
脚本按已有 `QUANT_THRESHOLDS` 比较两臂与独立 HF 的 KL/top-1，同时执行每臂原有
`graph_correctness_passed` 与跨臂自由轨迹比较；失败窗口和异常保留。
即使通过，也只是该配置的正确性诊断；kernel parity、更多 cache 状态回归及正式性能轮次仍需分别验收。

需要覆盖 result.json、重跑已完成 cell 或改变 cell 含义时，仍按 AGENTS 先确认并归档。

## 验收记录

- 修复前反例：38 failed / 13 passed；mask 新增拒绝/能力测试 2 failed / 7 passed。
- 修复后最终 CPU 套件：**590 passed / 31 deselected**（原基线 509 passed / 24 deselected）；
  新增 81 项 CPU 用例与 7 项 GPU 用例。已有极值量化 oracle 的 overflow warning 仍为 1 条。
- ruff、black（96 个 Python 文件）、check_docs（55 份 Markdown）和 `git diff --check` 均通过。
- 自然文本工具 dry-run 与真实 tiny OLMoE/StaticCache 的 CPU teacher-forcing/reset 集成已测试；GPU execute 未运行。
- 所有 GPU 性能、候选采纳和新的实际 CUDA parity：**未测**。
