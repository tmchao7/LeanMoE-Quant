# `scripts/rounds/` — 每一轮上卡的驱动脚本

这些是**实际跑出记录里那些数字**的脚本，从租用服务器的仓库根目录搬进来（它们此前只存在于
机器本地）。收进来的理由很简单：`experiments/<EXP-ID>.md` 写了结论，但"这一轮到底跑了哪几格、
金丝雀放哪、占用快照怎么记"此前无从复核。

## 共同结构

每个脚本都是同一个形状，这是 AGENTS.md §6「共享卡协议」的可执行版本：

```bash
set -u                                   # 失败要响，不要跑一半
cd /root/LeanMoE-Quant || exit 1         # 机器本地路径（见 §7；不写进代码）
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master
snap () { ... nvidia-smi --query-compute-apps=... }   # 每格前后记同租户快照
# 首尾同格金丝雀 → 逐格跑 → 每格用一段内嵌 Python 自检 payload（gate / host_share / kernel 数）
```

三个必须有的自检，缺一条那一轮就不能用：

1. **首尾同格金丝雀**，漂移 >5% ⇒ 整轮 VOID；
2. **逐格占用快照**：`nvidia-smi --query-compute-apps` 记录同租户（**只记录，绝不 kill**）；
3. **payload 自检**：门是否 PASS、`host_share_pct`、`kernels_per_call`（实现改变的签名）。

## 索引

| 脚本 | 对应轮次 | 记录 |
| --- | --- | --- |
| `run_p201.sh` | E2 反量化融合的 6 格（w8a8/w4a16 × fused/separate + w8a16 两格对照） | [EXP-p2-01](../../experiments/EXP-p2-01-dequant-fusion.md) |
| `run_p2d.sh` | 投影 kernel 修复（按组广播 scale）后**重跑受影响的格**并归档 v1 | [EXP-p4c-01](../../experiments/EXP-p4c-01-in-graph-attention.md) §5.4 |
| `run_p3_e7.sh` | E7 第一轮（16 格：bit_width × execution × ctx，eager/cudagraph 同轮相邻） | [EXP-p4b-01](../../experiments/EXP-p4b-01-truncation.md) |
| `run_p3_e7_v2.sh` | E7 第二轮（第一轮被外人进程 31929 污染 → 干净卡重跑） | 同上 §2.5 |
| `run_p3_e7_v3.sh` | E7 第三轮（补齐 evidence 格：w4a16 ×4 + w8a8 eager ×2） | 同上 |
| `run_p401.sh` | E4 路由开销的唯一一格（fp16 / eager / hf / ctx=1024） | [EXP-p4-01](../../experiments/EXP-p4-01-moe-routing-overhead.md) |
| `run_p4c01.sh` | 图内 attention × 量化 8 格（v1） | [EXP-p4c-01](../../experiments/EXP-p4c-01-in-graph-attention.md) |
| `run_p4c01_v2.sh` | 同 8 格（v2：投影 kernel 修复后） | 同上 §5.4 |
| `run_p4c01_v3.sh` | 同 8 格（v3：融合 RMSNorm 成为默认实现后） | 同上 §5.7 |
| `run_p2j_refresh.sh` | P2j 全面刷新（31 格；可中断可续，覆盖前幂等归档） | 各记录的「★ P2j 刷新」小节 |
| `run_f1_refresh.sh` | F1（+可选 F2）采纳刷新（23 个 graph 格；首尾同格 graph 金丝雀；可中断可续） | `docs/FUSION_OPPORTUNITIES.md` §F1/§F2 |
| `run_ab_fresh_process.sh` | **独立进程 · 同轮相邻 A/B**（长轮次用；每臂新进程，首尾同臂金丝雀，per-pair 胜负 + MAD） | `docs/FUSION_OPPORTUNITIES.md` §F2 的合规轮 |

## 注意

- 脚本里的路径是**那台机器**的（`/root/...`）；换机器要改 `M` 与 `PY`，别改用代码逻辑；
- `results/` 不入 git，这些脚本产出的是本机工件；记录里的 markdown 才是持久结论；
- 一次性分析胶水（`compare_p2j.py` / `gen_p2j_tables.py` / `finalize_p2j.py` /
  `insert_p2j_sections.py` / `p2j_split.py`）**没有**收进来 —— 它们只服务那一轮的表，且已经用过。
