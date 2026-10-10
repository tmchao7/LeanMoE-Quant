---
feature: final-sprint
status: designed
updated: 2026-10-10
branch: codex/offline-wrapup-p0p1
commits: # 开发中，交付时填
---

# Final Sprint — 上机实测 + 全量收尾 + 投递定稿

> **Workspace 覆盖记录**：不在 `.worktrees/` 建新 worktree，直接在主 checkout 的
> `codex/offline-wrapup-p0p1` 分支上工作。理由：该分支已领先 main 37 个 commit，
> `.venv-mac`、`results/`（机器本地工件）都绑定此
> checkout；HANDOFF §1 与 AGENTS §8 约定的协作流就是"Mac 在此分支开发、合并走 merge
> 不 rebase"。最终合回 main 也从本 checkout 执行。（2026-10-10 更新：`.workbuddy` 残余
> 已清除，服务器工件一律直接同步进 `results/`。）

## Report

（交付时填写）

## [S1] Problem

项目（OLMoE-1B-7B 单卡 decode 推理优化）科学与工程本体已完成，Mac 侧离线门全绿。
收尾缺口有三：

1. **一次服务器会话未跑**：G2 batch 曲线（两轮 VOID，冷态守卫已修复待验证）与
   G3 occupancy 上界（工具已就绪未上机），见 `docs/handoff/HANDOFF_2026-10-10.md` §4。
2. **文档占位符未填**：`docs/apply/PORTFOLIO.md` §2b 简历定稿留有 `[待测]` 行，
   `RESULTS.md` §6 / `docs/reference/METRICS.md` §6 缺口表待按实测终态更新。
3. **投递闭环未完成**：分支领先 main 37 commit 未合并；GitHub 首页是研究风格，
   面试官点开投递链接后需要 30 秒内看懂项目；投递用的项目文案（GitHub 链接 +
   关键词行 + 4 条 bullet）需定稿。

目标读者是**推理优化 / AI 编译器 / HPC / 算子优化**类 AI infra 实习岗的筛选人；
标准是工程与叙事达标，不追求 SOTA。

## [S2] Design

### A. 上机会话（约 40 分钟，用户开机并提供新 SSH 地址）

前置：用户重开实例后给新的 `ssh -p <PORT> root@<HOST>`（10-09 的地址可能失效）。

1. **共租检查**：`nvidia-smi --query-compute-apps=…` 非空 ⇒ 等待，绝不 kill。
2. **G2 batch 曲线（只试 1 轮）**：
   `python -m tools.batch_sweep --model <MODEL> --ctx 1024 --include-batch-1 --json results/batch_sweep.json`
   - 验收：金丝雀 drift ≤5% ⇒ 落盘有效，每 batch 有 n/stddev，曲线**单列一表**
     （不得与历史 batch=1 混表）；
   - drift >5% ⇒ **整轮 VOID，放弃 G2**（不重试第二轮），缺口表维持诚实条目。
3. **G3 occupancy 上界**：`python -m tools.occupancy_probe --json results/occupancy.json`
   - `status=done` ⇒ 三族 kernel blocks/SM 上界入库；
   - `status=planned` ⇒ 看 `attempts` 记录失败原因，按 planned 如实收尾。
4. **工件回传**：tar 回本 checkout 的 `results/`（`| tar xzf - -C .`，禁止 scp 多文件）。
5. 会话期间遵守共租/金丝雀/归档三条纪律（AGENTS §6、HANDOFF §5）。

### B. 文档回填（工件到手后，Mac 侧）

- `RESULTS.md` §6 缺口表：G2/G3 行按实测终态改写（成功 ⇒ 引工件；VOID/planned ⇒ 保持缺口）。
- `docs/reference/METRICS.md` §6：同步 G2/G3 状态。
- `docs/apply/PORTFOLIO.md` §2b：G2 成功 ⇒ 填 batch 曲线行；G3 成功 ⇒ 填 occupancy 行；
  未成 ⇒ 占位符删除或标"未测"，**绝不填预估数字**。
- `README.md` 状态行与 `docs/handoff/HANDOFF_2026-10-10.md`：更新为冲刺后的终态（HANDOFF 顶部
  加"已交付/过期"标注或生成终态摘要）。
- 每个数字必须指向工件或记录（硬规则 1）。

### C. GitHub 首页 polish（投递导向首屏）

README 顶部（现状行之后、"这个项目在问什么"之前）加一个 **30 秒结论区**：

- 一行定位：模型 / 单卡 / 课题一句话；
- 关键词行：`大模型推理优化 | 算子融合 | 量化(W8A16) | Triton | CUDA Graph | 高性能计算 | MoE`；
- 4 条核心 bullet，内容 = `docs/apply/PORTFOLIO.md` §2b 中文定稿（数字逐字一致，不引入新数字）；
- 指向 `RESULTS.md`（数字出处）与 `docs/apply/PORTFOLIO.md`（面试三题）的链接。
- 原研究内容全部保留在下方，不删改；`check_docs` 必须过（死链检查）。

### D. 验证、评审、合并、定稿

1. **离线四门**（每次提交前）：`ruff check . && black --check . && pytest -m "not gpu" -q && python -m tools.check_docs`，0 failed。
2. **评审**：验证全绿后派一个全新 subagent 审完整 diff，产出三个独立结论
   （spec 合规 / 正确性 / 代码库一致性）；critical 归零才进下一步。
3. **合并**：`main` 上 merge（**不 rebase、不 force-push**）`codex/offline-wrapup-p0p1`，
   push origin；若服务器还开着则在其上 `git pull --no-rebase`，否则留给下台机器。
   合并前确认工作区干净、四门全绿（AGENTS §8）。
4. **投递文案定稿**：产出与用户示例同构的最终文案——
   `【MoE模型单卡推理优化（模型：OLMoE-1B-7B、显卡：RTX 5090 × 1)  <GitHub URL>
   关键词：… 】+ 4 条 bullet（+ 可选的 G2/G3 两条，仅在实测成功时）`。
5. **Finalize**：spec 置 `delivered`、填 commits 区间与 Report，文档提交在分支上完成后
   再走 Finish。

### 提交纪律

沿用 AGENTS §8 前缀（`docs:` / `bench:` / `analysis:` 等），一提交一逻辑变更；
`results/` 不进 git；覆盖已有 `result.json` 前先归档（规则 9）。

## [S3] Out of Scope

- FP8 / MXFP4、E5 专家敏感度、E6 KV INT8、E8 跨带宽、ncu 计数器补测（HANDOFF §7 明确不做）；
- G2 的第二轮重试（本轮 VOID 即放弃）；
- GSM8K 口径复活（已按 floor 放弃）；
- 实验矩阵新增轴/格、重跑已记录格（需用户批准，本次冲刺不触碰）；
- `results/` 工件入库、模型权重、`__pycache__` 提交；
- 简历整体改写 / 投递信（另属用户的求职材料流程，不在本仓库冲刺内）。

## Tasks

- [x] T1: 上机前置检查 — acceptance: 新 SSH 地址可用；`nvidia-smi` 共租检查执行并记录（非空则等待） (covers: S2-A; needs: 用户开机并提供地址)
- [x] T2: G2 batch 曲线单轮重试 — acceptance: drift ≤5% ⇒ `results/batch_sweep.json` 有效落盘且每 batch 有 n/stddev；drift >5% ⇒ VOID 记录留存并停止 G2 (covers: S2-A; depends: T1)
- [x] T3: G3 occupancy 上界 — acceptance: `results/occupancy.json` 落盘，`status=done`（三族 blocks/SM）或 `planned`（含 attempts 失败原因） (covers: S2-A; depends: T1)
- [x] T4: 工件 tar 回 Mac — acceptance: T2/T3 的 JSON 在本 checkout 内可读，内容完整 (covers: S2-A; depends: T2, T3)
- [x] T5: 文档回填 — acceptance: RESULTS §6 / METRICS §6 / PORTFOLIO §2b / README 状态行 / HANDOFF 均按实测终态更新，每个数字带出处，`check_docs` 通过 (covers: S2-B; depends: T4)
- [x] T6: README 投递导向首屏 — acceptance: 顶部 30 秒结论区（定位 + 关键词行 + 4 条 bullet 与 PORTFOLIO §2b 逐字一致 + 出处链接），原内容保留，`check_docs` 通过 (covers: S2-C; depends: T5) — **2026-10-10 上午提前完成**（不依赖 G2/G3；T5 回填状态行时勿覆盖本区块）
- [ ] T7: 离线四门验证 — acceptance: `ruff` / `black` / `pytest -m "not gpu"` / `check_docs` 全绿，0 failed，命令与结果记入 Report (covers: S2-D; depends: T5, T6)
- [ ] T8: 独立评审 — acceptance: 一个全新 subagent 给出 spec 合规 / 正确性 / 一致性三个独立结论，critical 为 0；有则修复并复审 (covers: S2-D; depends: T7)
- [ ] T9: 合回 main 并 push — acceptance: merge（非 rebase）完成，`origin/main` 包含全部提交，push 成功，工作区干净 (covers: S2-D; depends: T8)
- [ ] T10: 投递文案定稿 + spec 交付 — acceptance: 产出与用户示例同构的最终文案（G2/G3 成功才加可选两条）；spec 置 `delivered` 并提交 (covers: S2-D; depends: T9)
