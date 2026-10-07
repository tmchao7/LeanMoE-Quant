# HANDOFF — Phase 2：反量化融合 kernel（`EXP-p2-01`）+ E7 收益截断（`EXP-p4b-01`）

> 这份文件是给**下一个会话**的。读完 `AGENTS.md`、`PLAN.md` §3/§5、`docs/METRICS.md` §1–2 之后
> 按 §7 的顺序执行。**不要重跑已经完成的探索**（下面 §9 有可直接引用的数字）。

## 1. 为什么是这两件事

`PLAN.md` 的技术判断是：**batch=1 的 MoE decode 是带宽问题，所以量化必须先过正确性门，
然后收益会被 host/launch 开销截断**。这两句话的验证都在本文件里：

- `EXP-p2-01`：**反量化必须融合进 GEMV**，否则 dequant 的读写把收益吃掉 —— 这是"能不能拿到收益"。
- `EXP-p4b-01`：`{tolerant config} × {eager, cudagraph} × ctx` 的 12 格，给出
  **本项目头号结论**：量化收益在 batch=1 上被 host 开销截断到什么程度、`cudagraph` 消掉多少。

> ⚠️ 目前**所有性能数字都是 fp16 的**（见 §9）。量化的性能数字至今为 0 个 —— 这就是本任务的价值。

## 2. ★ 先读这一节：门现在的样子（它决定了哪一格能报性能数字）

`AGENTS.md` 硬规则 2：**没过正确性门的配置不得报性能数字**。门在 `docs/METRICS.md` §1，
但**测量口径在 `EXP-p1-04` 被更正过**（旧口径用的是 1024 个 token id=1，是退化输入）：

| 配置（attn+experts，g128，ctx=1024，32 步 teacher forcing） | 5 窗口真实 prompt 集 mean KL | 门 | 能报性能吗 |
|---|---|---|---|
| fp16 | 参照 | — | ✅ |
| **`w8a16`** | **0.001273**（最差窗口 0.00429） | 0.01 | ✅ **唯一过门的量化配置** |
| `w8a8` | 0.032753 | 0.01 | ❌（真失败，不是 prompt 效应） |
| `w4a16` | 0.093323 | 0.05 | ❌ |

**因此：`p4b-01` 矩阵里 `{fp16, w8a8, w4a16}` 三档，现在只有 `fp16` 与 `w8a16` 能报性能。**
`w8a8` 那一档要么先修激活量化（见 §8 的可选项），要么跑但**标成 correctness-failed、
明确写"不得用于性能结论"**。这是纪律，不是可以商量的地方。

好消息：`w8a16` 本身就是一次真实的量化（**权重的 49.4%**），足以支撑 E7 故事。
`EXP-p1-04` 已经把它的 mean/sd/min/max 与 noop 自检都落了盘，可以直接引用。

## 3. `EXP-p2-01`（计划里超时 25 GPU 小时，有兜底）

矩阵里已有 4 格：`bit_width ∈ {w8a8, w4a16} × dequant ∈ {fused, separate}`。
**建议再加 2 格 `w8a16 × {fused, separate}`** —— 它是唯一过门的量化配置，
E7 结论要靠它。加格要在记录里写明（这是对计划矩阵的扩展，不是替换）。

要做的事：
1. **`separate` 臂**：`dequant` 成 fp16 权重 → 普通 `F.linear`（两次显存往返）。
2. **`fused` 臂**：Triton kernel 里直接读 int8/int4 + scale，GEMV 输出 fp16。
   目标：达到**实测可达带宽**的 70%（`results/EXP-p0-01-machine-probe/result.json`
   里有实测 **1561.9 GB/s**，170 SM）。**不要用 spec 带宽。**
3. 报告：kernel 级延迟（p50/p99）+ `dequant 占比`（`docs/METRICS.md` §2）+ ncu 证据。
4. **兜底（计划已授权）**：到 25 GPU 小时若没到 70%，就用现成融合 kernel
   （vLLM / torchao）当"已融合"臂，并把贡献转落到"字节预算分析 + E7 截断解释"。

> ★ **2026-10-07 执行完毕（`experiments/EXP-p2-01-dequant-fusion.md` §6）**：
> 自研 fused kernel 落地（4/8-bit、无 autotune、grid 为 host 常量 → P3 可 capture），
> 6 格全跑、K1 的 1e-3 门 PASS、**兜底未触发**。结论是 **`refuted`**：fused 在
> w8a16/w8a8 上比 separate 慢 5%（60/60 对、|Δ|≫MAD），在 w4a16 上快 3.6%（门 FAIL，
> evidence only）。原因是**那条 eager 路径 host/launch 受限**（host_share 60.7–80.4%、
> 每步 1.9–4.3 万 kernel、达成率 0.4–0.5%），不是带宽。第 3 项要求的"dequant 占比"
> 已用同进程 A/B 实测（`attribution.dequant_share_pct`），**ncu 证据没做**（本轮够用，
> 若要 kernel 级 dram 计数仍需 ncu）；"70% 带宽达成率"这个目标**未达成、也未适用** ——
> 在 M=1 的 host-bound 区间里它测不出来，见 §6.4。

## 4. `EXP-p4b-01`（计划里的头号结论）

> ★ **预注册已完成（2026-10-06 离线会话）**，权威清单见
> `experiments/EXP-p4b-01-truncation.md`：矩阵现为 **16 格**（加了 `w8a16` 轴 +
> `config.moe: grouped`），其 §3 是**依赖链 P0–P3**。关键事实：**量化臂被 P1 阻塞**
> （`replace_moe_blocks` 只吃 fp16 融合权重，hf 又不能 capture），所以**今天可跑的只有
> fp16 的 4 格**；量化 × 非 hf 的组合 runner 已实现第四条拒绝。
>
> ★ **2026-10-07 上卡进展**（详见该记录 §2.5 / §2.6 / §2.7）：
> - **P2 已完成**：ctx=4096 的门实测补上（w8a16 PASS；**w8a8 的 KL 过门但 top-1 不过 →
>   FAIL**；w4a16 FAIL）。`GATE_CITATIONS` 现在存实测 `kl`/`top1`，`gate` 由
>   `gate_from_metrics()` **推导**（合取：KL < 门 且 top-1 ≥ 0.95），有测试钉住。
> - **P1 已完成（8-bit）**：新的 `moe_grouped_quant` 在融合张量上做 per-group RTN，
>   decode 走量化 grouped kernel（kernel 内解包）。K1 门在 5090 上 **PASS**。
> - **fp16 4 格已测且有效**（金丝雀 0.10%）：eager 9.73/9.67 ms、cudagraph 4.07/7.31 ms。
> - **量化臂首测是负结果**（金丝雀 0.70%）：`speedup_eager` = 0.824×(@1024) / 0.802×(@4096)。
>   机制（§2.7）：量化 expert kernel 隔离测**快 17%**，但 attention 仍走 separate 臂，
>   两种反量化语义混在一格里 —— 所以这个 0.82× **不能**归因给量化本身。
> - **第 1 轮曾整轮作废**：共享卡上另一条工作线的 `pareto_offload.py` 占到 17.8 GB，
>   金丝雀漂移 37.1%、`cudagraph×1024` 被 OOM 阻塞。协议是**记录不杀**，等它退出后重跑。
> - **★ ① dense fused 臂已完成（2026-10-07）**：`EXP-p2-01` 6 格跑完（金丝雀 0.61% →
>   本轮有效），K1 的 1e-3 门 **PASS**。结果：**fused 比 separate 慢 5%**（w8a16/w8a8，
>   60/60 对，|Δ|≫MAD），4-bit 反而快 3.6%（但门 FAIL、只作 evidence）—— 因为那条路径是
>   host/launch 受限的（host_share 60.7–80.4%、1.9–4.3 万 kernel/步、达成率 0.4–0.5%）。
>   隔离测里 fused 每调用快 1.21×（33.5 vs 40.6 µs），但其中 ~16 µs 是 triton launcher 地板。
>   **E2 的 1.3–2× 在 eager 配置下否证**（不是 inconclusive）；§2.7 那句 "+2.02 ms = 物化流量"
>   已被修正为 host/算子数成本。详见 `experiments/EXP-p2-01-dequant-fusion.md` §6。
> - **★ ② P3 已完成（2026-10-07）→ E7 已判定**（`EXP-p4b-01-truncation` §2.8）：
>   `run_cudagraph(quant_cfg=…)` 接上（量化在 capture 之前），`build_payload_cudagraph`
>   的 `bit_width` 改为读 cfg、门改成 **引用门 ∧ 图回放门**；`blocked_result` 的同类硬编码
>   也修了（实测撞上：一次 capture OOM 的 payload 里 `bit_width` 写成了 fp16）。
>   ```text
>   speedup = p50(fp16)/p50(w8a16)  (>1 = 量化更快)
>   ctx=1024  separate : eager 0.7665 -> cudagraph 0.7674  (host 77.5% -> 0.6%)
>   ctx=1024  fused    : eager 0.7594 -> cudagraph 1.1071  (host 77.5% -> 1.2%)
>   ctx=4096  separate : eager 0.7933 -> cudagraph 0.8530
>   ctx=4096  fused    : eager 0.7890 -> cudagraph 1.0512
>   ```
>   **结论**：截断效应成立，但**只在 fused 臂上兑现** —— `w8a16 × cudagraph × 1024 =
>   3.6737 ms` 是项目第一个 >1× 的量化格（fp16 4.0670 ms）。量级 1.05–1.11×，不是 2×
>   （2× 那行是 W4A16：门 FAIL + 4-bit grouped 未实现）。**P2-01 的融合是前提**：
>   只做 P3 会得出"量化在 cudagraph 下也不行"的错结论。
> - **第 1 轮 P3 作废**：共享卡上外人进程 pid 31929（峰值 14 GB）→ capture OOM +
>   `fp16×cudagraph×4096` 3× 离群。★ eager 金丝雀对这种污染**不敏感**（只漂 1.16%）——
>   下一轮起给 cudagraph 臂加**自己的同格金丝雀** + 每格 `host_share>5%` 自检。
> - **★ 4-bit 也做完了（2026-10-07，`EXP-p4b-01` §2.9）**：P1 的 nibble 解包落地
>   （K1 门 8/4-bit 两条路径都 PASS），**16/16 格收口**（8 格 gate PASS + 8 格 gate FAIL 的
>   evidence）。三个结论（都只到 evidence 层）：
>   ```text
>   实现 > 位宽 : w4a16×cudagraph×1024  separate _(9.1203)_ ms / 1810 kernel
>                                      fused    _(3.3812)_ ms /  978 kernel   → 2.7×
>   边际字节贴地板: w8a16 fused 3.6737 → w4a16 fused 3.3812 ms
>                  Δ0.2925 ms vs Δ[W]=0.537 GB÷1561.9=0.3438 ms  → 拿回 85%
>   但权重只占这一步 23%（§2.8.7 的 kernel 归因）⇒ 天花板 ~1.3×（ctx=4096）
>   ```
>   **下一步最值钱的方向**不再是位宽：是那 55% 拿不掉的部分（`fmha` 33–65% +
>   elementwise/copy 12–23%）。
>   ★ **attention 那一半已经做完**（`EXP-p4c-01`，2026-10-07）：图内换 self-attn 实现
>   （HF sdpa → 本仓库 split-KV，16 个 core 全部替换）把 step 砍 1.36–2.44×，并**提高**量化
>   speedup：1.109 → **1.153**（ctx=1024）/ 1.053 → **1.131**（ctx=4096）；两个杠杆合起来
>   **1.57×（1024）/ 2.57×（4096）**。机制：被换掉的 attention 段与位宽**无关**（两边都省
>   ~4.1 ms），是加性常数 ⇒ 对总量更小的量化臂更有利。
>   ★ **投影 kernel 那一半也做完了**（kernel `7d1f92b`）：先在图里测 config
>   （新工具 `tools/bench_dequant_gemm_graph.py`，micro-graph 64 次调用）→ 发现
>   p2-01 §6.4 的隔离排名是 **launcher 口径**、生产默认已是最优，剩下的 5.4 µs/次是
>   **结构性**的（scale 被逐元素 gather，而 128 个 k 共用同一个值）。改成按组广播后
>   **8.14 → 5.15 µs/次（523 → 826 GB/s）**，模型里投影桶 0.5245 → **0.3254 ms**，
>   最好一格变成 **1.70×（1024）/ 2.77×（4096）**。
>   ✅ **改前的格子已重跑**（P2d 轮，2026-10-07）：E7 的 fused 臂 `speedup_cudagraph`
>   **1.1071 → 1.1703**（1024）/ **1.0512 → 1.0839**（4096）；separate 臂一字未动
>   （0.7674 / 0.8530）；`EXP-p2-01` 的 6 格 A/B 比 1.0442 / 1.0523 / 0.9681 —— 判定全不变。
>   ★ 关键对照：**同一个 kernel 改进在 eager 臂里完全看不见**（A/B 1.0093 → 1.0091，host 61–88%），
>   只在图内兑现 —— 这是 E2/E7 机制最干净的一次复现。v1 工件在 `results/<EXP>-run/v1_archive/`。
>   **现在的新靶子**：elementwise/copy 长尾（~0.90 ms / ~35%）。**已归类**（`EXP-p4c-02`，
>   用户批准的新 id）：它不是一片平的噪声，而是 **13 个具名家族 × 每层 1/2/4/5 次调用** ——
>   拷贝族 0.17 ms（149 次/步）、RMSNorm 的 mean+rsqrt 0.15–0.25、KV cache 写+cat 0.11、
>   残差 0.05、路由 topK 0.05；**没有单点 >20%**，所以没有"改一处赚 0.4 ms"的机会。
>   ★ 模块级归因**按自己的预注册 K1 作废**（eager/graph 前 8 kernel 只重合 5/8）——
>   记录里写明作废，没有拿 eager 的数字去解释图。
>   下一步走过的路（都记在 `EXP-p4c-02`）：
>   **(a) 拷贝族已指认**：`aten::_to_copy` ×150/步 = 0.185 ms，全部来自 HF `OlmoeRMSNorm`
>   （每个 norm 7 个 kernel：2 次 cast + pow + mean + rsqrt + 2 次 mul；65 个 norm/步）
>   ⇒ **拷贝族与 RMSNorm 是同一个修法**。
>   **(b) 换 torch 自带融合 `F.rms_norm` —— 已试、已否证**：慢 5%（2.4106 → 2.5358 ms，HF 赢 15/15），
>   虽然数值上就是同一个归一化（KL 7.2e-4、top-1 1.0 过 K1）。机理：它保留同一个 `MeanOps`
>   reduce，却新增一次 **fp32 行乘**（`BinaryFunctor<float>` ×65 = 0.189 ms），而 HF 是在 fp16 里乘。
>   **(c) 自研 Triton RMSNorm —— 做完了，赢了 23.6%**（P2h，`src/lemoq/kernels/rmsnorm.py`）：
>   一次 launch，**舍入点与 HF 逐点一致**（fp32 mean → 舍入 fp16 → 在 fp16 里乘 weight）。
>   图内三臂 A/B：HF 2.4143 ms / torch 融合 2.5474（慢 5%，KL 1.1e-3 差）/**自研 1.9537 ms**
>   = **1.2358×，15/15 轮全胜**；kernel/步 **994 → 539**（−455 = 7×65 完全对上），
>   `_rms_norm_kernel` 1.26 µs/次；K1 与 HF 公式 KL **5.261e-04** ✓。
>   ### P2i 进度（2026-10-07 收尾，服务器关闭时的状态）
>
>   - ✅ 接线完成：`--rmsnorm {fused,hf}`（默认 fused，写进 `config.rmsnorm` +
>     `config.rms_norms_replaced`，第六条加载前拒绝，测试钉住）。
>   - ✅ ctx=1024 的门**在 fused norm 下重测过**（被服务器关闭打断在 26/51 臂）：
>     **三条引用臂全部完整**（6 个窗口；工件里 `rmsnorm: fused`、`norms_replaced: 65`）。
>     逐窗口手工汇总：w8a16 **0.001361**（原 0.001273，+6.9%）、w8a8 0.033304（原 0.032753）、
>     w4a16 0.093560（原 0.093323）⇒ **判定一个都没变**（w8a16 仍是唯一过门者）。
>     ⚠️ 这三个数是从**抢救下来的窗口**手工算的 —— 探针自己的 `aggregate` 段没写出来
>     （作业被杀），所以它们**还不是引用值**。
>   - 抢救位置：`results/EXP-p1-04-prompt-set-gate/probe_prompt_set_rmsnorm_fused_ctx1024.json`
>     （Mac 上也有）；p4c-01 的改前工件归档在 `results/EXP-p4c-01-run/pre_p2i_hf_norm/`。
>   - ⬜ **明天**：(1) 给探针加 `--aggregate-only <artifact>`（复用它自己的聚合代码）
>     或重跑；(2) 归档 HF-norm 的两张门工件、把新工件放到规范名；(3) 逐字更新
>     `GATE_CITATIONS` + 被钉住的断言 + ctx=4096 的第二次探针；(4) `--force` 重跑 p4c-01 的
>     8 格（默认已是 fused）→ 记录里标 **v3**。
>   - ⚠️ 共享卡上出现了**外来作业**（另一项目的 `pareto_offload.py` / Qwen3-30B、早前的
>     `block_h2d_probe.py`）：**没动它**。明天跑**计时**轮之前必须先 `nvidia-smi
>     --query-compute-apps` 复查占用 —— 探针那轮是正确性测量（KL），不受影响；
>     8 格是计时测量，有共租就必须等或作废重跑。
>   ★ W4A16 的门仍是 FAIL（0.0933 > 0.05）：要报它的数字必须先做精度侧，那是独立实验。
>
> 下面这段是当时按 12 格写的规划，保留作背景：

12 格 = `bit_width × execution × ctx`。按 §2 的实际可用性：
- 必跑：`fp16`、`w8a16` × `{eager, cudagraph}` × `ctx ∈ {1024, 4096}` = **8 格**
- 追加（标 correctness-failed，只报延迟不报收益）：`w8a8` × 同上 = 4 格
- 每格都要报：per-token p50/p99、**host/launch 占比**（一等指标）、以及
  "相对 fp16 的收益"是否被 host 开销截断。

**想清楚要问的问题**：`w8a16` 把 `[W]` 从 13.42 GB 降到 6.63 GB（49.4%），
理论带宽收益接近 2×；实测收益是多少、被什么吃掉、`cudagraph` 消掉多少 —— 这就是 E7。

## 5. 环境与命令

```bash
# 服务器（免密）
ssh 5090b

# 必设（否则 13 GB 模型落到 30 GB 系统盘）
export HF_HOME=/root/autodl-tmp/hf
export HF_ENDPOINT=https://hf-mirror.com          # huggingface.co 在这台机器上不可达（000）
PY=/root/miniconda3/bin/python
M=/root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master

# ★ 每次 timing 前都查：这台机器上还有别人的进程（见过 pareto_offload.py 占着卡，
#   那次 ctx=4096 测出 10.38 ms / host-share 67.5%，清干净重测是 3.2356 ms）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# 不要在别人的项目进程上动手：/root/autodl-tmp/mf-work/、/root/MegaKernel-vLLM、/root/models/Qwen3-0.6B

# 离线门（Mac 或服务器都行）
uv run pytest -m "not gpu"
uv run ruff check . && uv run black --check .
uv run python -m tools.check_docs

# 现有工具（都在 tools/ 下，可直接复用）
#   quant_probe.py   量化精度探针：--arms prompt_set --prompt-source wikitext \
#                    --prompt-cache /root/autodl-tmp/lemoq-data/wikitext2_test_8k.pt \
#                    --prompt-windows 5 --include-ones-control
#   quant_ppl.py     PPL + 配对 ΔNLL（第二个指标）
#   quant_error_audit.py  纯权重域审计
```

## 6. ⚠️ 坑清单（全部是本仓库实测踩过的，不是推测）

1. **`_snapshot` 会把原模块停在 CPU 省显存** → **任何模型前向之前必须先 `_restore`**，
   否则第一次 forward 就 `mat2 is on cpu`。
2. **就地改权重会污染快照**：`w.data.copy_()` 改的是快照里**同一个对象**，恢复变成空操作，
   噪声臂会累积（实测被交错插入的 `noop` 臂抓到）。做法：新建模块挂上去。
3. **名字必须在量化之前收集**：`quantize_model` 把 `nn.Linear` 换成 `QuantLinear`
   （**不是** `nn.Linear` 子类），之后再按类型找就一个也找不到，而统计会静默变成 0。
4. **门的 prompt 口径**：不要在 `ids = full((1, ctx), 1)` 上测 KL 并当门用
   （那是退化输入，膨胀 21–33×）。用 `--prompt-windows 5` 或 `quant_ppl.py`。
5. **计时必须交错**：臂之间 round-robin，报 MAD；不要一条臂跑完再跑下一条
   （机器上可能有别的负载漂移）。
6. **`results/` 是 gitignored**：跑完必须 `rsync -az 5090b:/root/LeanMoE-Quant/results/ ./results/`
   到 Mac，否则数据随机器关停消失。
7. **Mac 上的任何数字不得进 `results/` 或结论**。
8. **依赖**：`configs/constraints.txt` **已于 2026-10-07 按服务器实测冻结**
   （`frozen: 2026-10-07 · RTX 5090 (AutoDL PyTorch 2.8.0/12.8 镜像)`；torch 由预置的 2.11.0
   改为实测的 2.8.0，numpy/safetensors/pyyaml 也钉到实测值）。
   ★ 保留的坑：`pip install datasets` 默认会装 `datasets 5.1.0` + `huggingface-hub 2.1.1`，
   而 `transformers 4.57.6` 要求 `hub<1.0` → **transformers 直接坏了**；正确的一对是
   `datasets==3.6.0` + `huggingface-hub==0.36.2`（现写在 constraints.txt 的注释里，成对装）。
   **装任何包之前先记录版本，装完逐项核对。**
9. **merge，不要 rebase，不要 force-push**；运行中的 agent 走 feature 分支。
10. **`uv` 在服务器上是有的**：`/root/miniconda3/bin/uv`（0.12.17）。新 shell 里 **先
    `export PATH=/root/miniconda3/bin:$PATH`** —— 不 export 时 `tests/test_scripts.py` 里
    依赖 `uv` 的那条会失败（**pre-existing，不是你的回归**）；export 后服务器整套
    `pytest -m "not gpu"` 是 0 failed。

## 7. 建议的执行顺序

1. 读 `AGENTS.md` / `PLAN.md` §3.1、§5 / `docs/METRICS.md` §1–2 / 本文件 §2；
   再读 `experiments/EXP-p1-04-prompt-set-gate.md`（门的现状）与 `EXP-p0-04-cudagraph-static-moe.md`
   （已有的 cudagraph 执行路径）。
2. **先快照当前状态**：`nvidia-smi`（有没有别人的进程）、`git log -1`、跑一遍离线测试。
3. `EXP-p2-01`：先写 `separate` 臂并拿到数字（它是 fused 的对照，必须同进程同条件），
   再写 fused kernel。**每加一格都在同一个进程里交错测。**

   > ★ **离线接线已完成**（2026-10-06，本仓库当时的离线会话）：格子 id、
   > `--bit-width/--dequant/--group-size`、三条拒绝（fused 未实现 / 无门实测 /
   > 量化+cudagraph 未接线）、`GATE_CITATIONS` 与 `EXP-p1-04` 的逐字对齐、
   > 量化 hook 与字节账 —— 全部实现并有离线测试（352 passed）。
   > **还差的清单见 `experiments/EXP-p2-01-dequant-fusion.md` §4**（kernel、
   > dequant_share、跑 6 格、记录）。矩阵的 bit_width 轴已加 `w8a16`
   > ——它是 p1-04 之后唯一过门的配置，其余两档只能跑出 verdict=FAIL 的格子。
4. `EXP-p4b-01`：用 §2 判定过的配置跑 8 格（+4 格标 failed）。
5. 每格落 `results/<EXP-ID>/result.json` + `experiments/<EXP-ID>.md` + 一个 commit。
6. 最后：两台机器的测试、`ruff`/`black`/`check_docs`、`rsync` 结果回 Mac、
   **把服务器留在与 Mac 相同的 commit 上**。

## 8. 可选项（如果 $w8a8$ 那一档必须救）

`w8a8` 现在是 Phase 1 唯一真失败的条目（`EXP-p1-04`）。想救它是独立的一个实验：
激活 scale 粒度（per-tensor / per-token / per-channel）× clamp 搜索。
它同时回答 `[T]`（激活带宽）那一半，值得做，但**不要和 p2-01/p4b-01 混在同一个实验里**。

## 9. 现在就能引用的数字（省得重新推导）

- 机器：RTX 5090 32 GB、170 SM、驱动 595.71.05、**实测带宽 1561.9 GB/s**
  （`results/EXP-p0-01-machine-probe/result.json`）。
- 模型：OLMoE-1B-7B-0924，权重 **13.42 GB fp16**（attention 0.54 GB = 4%，experts 12.88 GB = 96%）。
- 字节账（实测，非估算）：`w4a16` = fp16 的 **25%**、`w8a16` = **49%**、padding 膨胀 **0.0%**。
- 已实现并实测的 fp16 执行路径：CUDA Graph（`EXP-p0-04`）、grouped MoE GEMV（`EXP-p2-02`）、
  eager attention（`EXP-p2-04`）、自研 split-KV attention kernel（`EXP-p2-05`）。
  累计 **27.5×**（ctx=1024：82.64 ms → 3.0077 ms，达成率 50.2%；ctx=4096：25.5×）。
  **这些是 fp16 的数字。**
- 正确性（更正后的门，`EXP-p1-04`）：`w8a16` **过门**（mean 0.00127，5 窗口）；
  `w8a8` 0.0328 与 `w4a16` 0.0933 **不过门**。
- PPL（`EXP-p1-03`，wikitext-2 test 前 8192 token，配对 ΔNLL）：fp16 PPL **9.1994**；
  `w8a16` 与 fp16 无法区分（|t| < 1）；`w4a16` +2.6…+5.9%（超 <1% 门）。
- 量化精度取证工具：`tools/quant_probe.py`（KL，多窗口）、`tools/quant_ppl.py`（PPL）、
  `tools/quant_error_audit.py`（权重域）。三者都带 noop 自检。

## 10. 验收标准（做完自检）

- [x] `EXP-p2-01`：`fused` vs `separate` 在**同一进程、交错**下测得（每臂 60 样本、报 MAD
      与逐对胜负）；dequant 占比是**同进程 A/B 实测**（不是 ncu）；兜底未触发。
      ⚠ 例外：ncu 的 dram 计数没做；"达成率 70%"在 host-bound 的 eager 格里测不出来，
      已在 `EXP-p2-01` §6.4/§6.6 明确写出。
- [ ] `EXP-p4b-01`：≥8 格（fp16/w8a16 × eager/cudagraph × ctx 1024/4096），
      每格报 p50/p99 + host 占比；`w8a8` 那 4 格如跑则明确标 correctness-failed。
- [ ] 每格有 `result.json` + `experiments/<ID>.md`（假设写在结果之前）+ commit。
- [ ] 门槛一个都没改；没过门的配置没有性能结论。
- [ ] 两台机器测试绿（服务器那个 `uv` 的失败要标注 pre-existing）；
      `results/` 已 rsync 回 Mac；服务器停在同一个 commit。
- [ ] 结论里写明适用性：GPU、**实测**带宽、dtype、batch=1、ctx。
