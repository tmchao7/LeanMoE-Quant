---
feature: docs-restructure
status: delivered
updated: 2026-10-10
branch: codex/offline-wrapup-p0p1
commits: 88cc07f..a2ec285
---

# Docs Restructure — 清理 workbuddy 残余 + 文档目录重组

> **Workspace**：主 checkout 的 `codex/offline-wrapup-p0p1` 分支（与 final-sprint 同一工作区，
> 覆盖理由见 `docs/compose/spec/final-sprint.md` 顶部记录）。
> **时序**：本 feature 在下午上机会话（final-sprint T1–T4）**之前或并行**执行，不依赖服务器。

## Report

**What was built** — 彻底清除了腾讯 WorkBuddy 残余：15 个唯一副本工件（MMLU `task_accuracy.json`、
batch VOID 证据、clone1 四格、PPL 归档等）先并入 `results/`，2 个同名冲突以 10-09 服务器版为规范
（旧值存 `*.pre-sync.json`），再删除两份工作日志与整个 `.workbuddy/` 并在 `.gitignore` 防复发。
删除 3 份过期/历史交接（共 1322 行），`docs/` 重组为 `handoff/`（当前唯一权威交接）、
`reference/`（7 份查证型文档）、`apply/`（PORTFOLIO 投递材料）三目录并新增 `docs/README.md`
文档地图；9 份 `git mv` + 57 个文件的 `docs/<name>.md` 路径 + 移动后变深的相对出链全量修正，
AGENTS §0/§9、README 目录树、HANDOFF、final-sprint spec 同步，三张图重生成（fig3 图内路径）。

**Verification** — `ruff check .` PASS；`black --check .` PASS（80 files；`tools/quant_probe.py`
因路径变长由 black 重排）；`pytest -m "not gpu" -q` **428 passed / 10 deselected / 0 failed**；
`python -m tools.check_docs` OK（36 md，内部链接全通）。独立评审总判 **APPROVE、无 critical**；
其 1 项 major（fig3 SVG 内嵌旧路径）与全部 minor（spec 计数 16→15/8→9、README 树漏
MOE_CUDAGRAPH、`src/lemoq/__init__.py` 松散引用）已修复并四门复验全绿。

**Journey log** —
1. zsh 对未加引号的 `$var` 不做分词，复制循环把整个列表当一个文件名；且循环内 `cd $SRC`
   把持久 shell 的 cwd 带偏，产生过一次假阳性复查。修法：绝对路径 + `while read`。
2. 实际缺失工件是 15 个（17 总数 − 2 同名），spec 初稿写 16；`git mv` 实际 9 份（含 handoff），
   初稿写 8——交付时勘误。事实性计数要在验收里写"逐文件 diff 清单"而不是心算。
3. `tools/make_figures.py` 的路径改了不代表图改了：fig3 SVG 内嵌 caption 路径，必须重跑
   `python -m tools.make_figures`（评审员抓到的唯一 major）。
4. 同名工件冲突（PPL_STATUS/machine_probe）不能猜：逐一 diff 后按"服务器回传版为规范、
   本地旧值 `.pre-sync.json` 归档"处理，两边都没丢。
5. "grep 旧路径 = 0"类验收必须限定**已跟踪文件**：`results/` 是 gitignored 历史工件，
   其内部旧字符串是证据本体，不改写。

## [S1] Problem

项目此前用腾讯 WorkBuddy 开发，留下两类问题：

1. **残余物**：`.workbuddy/` 目录仍留在仓库——其中 `server-sync-20261009/results/` 有
   **15 个文件在 `results/` 中不存在，是唯一本地副本**（含 `task_accuracy.json` 即
   MMLU G1 工件、`batch_sweep.void1.json`、clone1 四格、PPL 归档），直接删除会销毁实测
   证据；`memory/*.md` 两份工作日志已被 git 跟踪，且被当前 HANDOFF 引用。用户已不再使用
   WorkBuddy，残余物须清除。
2. **文档堆**：`docs/` 下堆了 4 份 HANDOFF（1400 行）——`HANDOFF.md`（已标过期）、
   `HANDOFF_ATTN_KERNEL.md`（一次性任务交接，任务已完成）、`HANDOFF_P2_FUSION.md`
   （已标历史存档）都是废文档；正经主题文档与交接文档混在一层，没有文档地图。

## [S2] Design

### A. 工件迁移（先于一切删除）

1. 把 `.workbuddy/server-sync-20261009/results/` 下 **缺失的 15 个文件**复制进
   `results/` 对应路径（含 `EXP-p4b-01-run/`、`ppl_merge_archive/` 子目录）。
2. 2 个同名文件（`PPL_STATUS.json`、`machine_probe.json`）逐一 diff：
   内容一致 ⇒ 保留 `results/` 中现有一份；不一致 ⇒ **以 server-sync（10-09 从服务器
   tar 回的版本）为规范**，本地旧值改名 `*.pre-sync.json` 归档保留，不静默丢弃。
3. 验收：17 个目标文件全部存在于 `results/`；两份同名文件处理留痕；JSON 可解析
   （`python -m json.tool` 抽查）。

### B. 清除 workbuddy 残余

1. `git rm` 两份工作日志（用户已确认直接删，git 历史可找回）。
2. `rm -rf .workbuddy/`（A 完成后才允许执行）。
3. `.gitignore` 追加 `.workbuddy/`，防复发。
4. 验收：`.workbuddy` 不存在；`git status` 无 `.workbuddy` 相关项。

### C. 删除 3 份历史交接

`git rm docs/HANDOFF.md docs/HANDOFF_ATTN_KERNEL.md docs/HANDOFF_P2_FUSION.md`。
保留 `docs/handoff/HANDOFF_2026-10-10.md`（当前唯一权威）与 `docs/reference/MOE_CUDAGRAPH.md`
（被 `experiments/EXP-p0-04`、`SERVER_RUNBOOK` 引用的技术预判文档）。
删除后 AGENTS §0/§9 中对 `HANDOFF_P2_FUSION` 的引用一并移除（见 E）。

### D. 目录重组（目标树）

**边界**：根级文档（README / AGENTS / PLAN / RESULTS / EXPERIMENTS / ENVIRONMENT）与
`experiments/`、`docs/figures/`、`docs/compose/` **不动**；只重组 `docs/` 下的散文件。

```text
docs/
├── README.md                      # 新增：文档地图（每份文档一行：是什么 / 给谁 / 链接）
├── handoff/
│   └── HANDOFF_2026-10-10.md      # 交接（随每次冲刺增长）
├── reference/                     # 查证型：规格 / 口径 / 证据 / 操作
│   ├── SPECS.md
│   ├── MODELS.md
│   ├── METRICS.md
│   ├── NCU_EVIDENCE.md
│   ├── NCU_PLAN.md
│   ├── MOE_CUDAGRAPH.md
│   └── SERVER_RUNBOOK.md
├── apply/
│   └── PORTFOLIO.md               # 投递材料（面试稿/文案的落点）
├── figures/                       # 不动
└── compose/spec/                  # 不动（compose-next 默认路径）
```

用 `git mv` 保留历史。`docs/README.md` 是新增的文档地图，同时被 AGENTS §9 与 README
目录树引用。

### E. 引用全量修复

- **入链**：`METRICS.md` 被 55 个文件引用（含 `experiments/*.md`、`.py`、`.sh`），
  `SPECS/MODELS/NCU_*/MOE_CUDAGRAPH/SERVER_RUNBOOK/PORTFOLIO/HANDOFF_2026-10-10`
  共约 60+ 处；全部按新路径更新（`docs/reference/METRICS.md` → `docs/reference/METRICS.md` 等）。
- **出链**：9 份被移动文档内部的相对链接随之变化（如 `../EXPERIMENTS.md` →
  `../../EXPERIMENTS.md`），逐份修正。
- **同步两处索引**：AGENTS §0（P2_FUSION 引用删除、路径更新）与 §9 表格
  （路径更新 + 新增 `docs/README.md` 行）；README 目录树的 `docs/` 段按新树改写。
- **同步现有 sprint spec**：`docs/compose/spec/final-sprint.md` 中
  `.workbuddy/server-sync-20261009/` 的两处引用改为 `results/`。
- `HANDOFF_2026-10-10.md` 中指向 `.workbuddy` 的三处（memory 链接、server-sync 说明、
  tar 目标路径）改为新位置/事实。
- 验收：**已跟踪文件**中 grep 旧路径与 `workbuddy` 为 0（本 spec 自述、HANDOFF/final-sprint
  的"残余已清除"记录性提述、`.gitignore` 防复发条目除外；`results/` 为 gitignored 历史工件，
  其内部旧字符串不改写）；`check_docs` 全绿。

### F. 验证与提交

1. 四门：`ruff check . && black --check . && pytest -m "not gpu" -q && python -m tools.check_docs`，0 failed（`tests/test_links.py` 会复跑 check_docs）。
2. 评审：一个全新 subagent 审完整 diff，三结论（spec 合规 / 正确性 / 一致性），critical 归零。
3. 提交粒度（AGENTS §8，一逻辑一提交，前缀 `docs:`/`chore:`）：
   ① 工件迁移+残余清除；② 删历史交接；③ 目录重组+引用修复+索引同步。
   `results/` 本身 gitignored，工件迁移不产生 git 变更（只有本地文件移动）。

## [S3] Out of Scope

- 根级六份文档的位置调整、`experiments/` 记录的结构改动（记录只允许改引用路径，不动数字与结论）；
- `docs/figures/`、`docs/compose/` 的重组；
- HANDOFF_2026-10-10 的内容刷新（属 final-sprint T5，本 feature 只改它的 `.workbuddy` 引用）；
- README 投递首屏（属 final-sprint T6）；
- 实验重跑、矩阵变更、`results/` 内文件的清理。

## Tasks

- [x] T1: 工件迁移 — acceptance: 15 个缺失文件进 `results/`；2 个同名文件按 A-2 策略处理并留痕；JSON 抽查可解析 (covers: S2-A)
- [x] T2: 清除 workbuddy 残余 — acceptance: `git rm` 两份日志、`rm -rf .workbuddy/`、`.gitignore` 追加；`git status` 无残留 (covers: S2-B; depends: T1)
- [x] T3: 删 3 份历史交接 — acceptance: `HANDOFF.md`/`HANDOFF_ATTN_KERNEL.md`/`HANDOFF_P2_FUSION.md` 不存在；保留文件未动 (covers: S2-C)
- [x] T4: 目录重组 — acceptance: 目标树与 D 完全一致（9 份 `git mv` + 新增 `docs/README.md` 文档地图） (covers: S2-D; depends: T3)
- [x] T5: 引用全量修复 — acceptance: **已跟踪文件** grep 旧路径/workbuddy 为 0（记录性提述与 spec 自述除外）；AGENTS §0/§9 与 README 目录树同步；final-sprint spec 两处引用更新 (covers: S2-E; depends: T4)
- [x] T6: 四门验证 — acceptance: ruff / black / pytest `-m "not gpu"` / check_docs 全绿 0 failed，命令与结果记入 Report (covers: S2-F; depends: T5)
- [x] T7: 独立评审 — acceptance: 全新 subagent 给出三独立结论，critical 为 0；有则修复复审 (covers: S2-F; depends: T6)
- [x] T8: 分批提交 — acceptance: 按 F-3 的三个提交落盘，`git log` 可读，工作区干净 (covers: S2-F; depends: T7)
