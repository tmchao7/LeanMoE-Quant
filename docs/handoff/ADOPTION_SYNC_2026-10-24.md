# 采纳刷新后的数字同步映射（2026-10-24）

> 用途：F1（+F2）翻默认并跑完刷新轮之后，**照这张表**把 `RESULTS.md` 与各记录的数字换成
> **同一轮**的 payload 取值。表格里的"现用值"是 2026-10-23 的 canonical payload（刷新前），
> 刷新后只改"取值"列，**不跨轮混比**（硬规则 3）。
>
> 取数模板（Mac，`results/` 有副本）：
> ```bash
> .venv-mac/bin/python - <<'PY'
> import json
> def v(p, *keys):
>     d = json.load(open(p))
>     for k in keys: d = d[k]
>     return d
> P = "results/EXP-p4b-01-truncation__bit_width=%s_execution=cudagraph_ctx=%s/result.json"
> print(v(P % ("fp16", 1024), "latency", "per_token_p50_ms"))   # 3.5881
> PY
> ```

## 1. 会随刷新变化的（必须换）

| 位置 | 现用值 | 来源格 | 取值表达式 | 刷新后 |
| --- | --- | --- | --- | --- |
| `RESULTS` §2 行2 量化 @1024 | 1.1911× | `p4b fp16 cg 1024` / `p4b w8a16 cg 1024` | `p50(fp16)/p50(w8a16)` | 3.5881/3.0124 → ？ |
| §2 行2 @4096 | 1.0888× | 同 ctx=4096 | 同上 | 6.8269/6.2702 → ？ |
| §2 行3 量化×kernel @1024 | 1.3034× | `p4c fp16 kern 1024` / `p4c w8a16 kern 1024` | `p50(fp16 kern)/p50(w8a16 kern)` | 2.5079/1.9241 → ？ |
| §2 行3 @4096 | 1.2555× | 同 ctx=4096 | 同上 | 2.7559/2.1951 → ？ |
| §2 行3 最好格 @1024 / @4096 | 1.87× / 3.11× | `p4c fp16 sdpa` / `p4c w8a16 kern` | `p50(fp16 sdpa)/p50(w8a16 kern)` | 3.5885/1.9241；6.8243/2.1951 → ？ |
| §3 E7 步级收益 | 1.1911/1.0888 | 同 §2 行2 | 同上 | 同上 |
| §3 权重流占比 | 42% / 23%* | `p4b fp16 cg` | `floor_w_ms / p50` | 1.5095/3.5881=42.1%；1.5095/6.8269=22.1% → ？ |
| §3 结构性上限 | ~1.74× / ~1.30× | 同上 | `1/(1-占比)` | 1/(1-0.421)=1.73；1/(1-0.221)=1.28 → ？ |
| §3 专家 GEMV 占一步 | 42.9% | `p4b fp16 cg 1024` | `1.540 / p50`（1.540 是 NCU 隔离值，**不随刷新变**） | 1.540/3.5881 → ？ |
| §3 该 kernel 解释步级收益 | 90.4% | `EXP-p4b-01` §4.3 的推导 | `16 × Δ_expert / Δ_step`（按记录里的公式重算） | ？ |
| 记录 `EXP-p4c-01` §5.7 | 1.1955/1.3034、1.87/3.11 | 同 §2 行3 + sdpa 臂 | `p50(fp16 sdpa)/p50(w8a16 sdpa)` 等 | ？ |
| 记录 `EXP-p4b-01` §P2j + 采用列 | 3.5881/6.8269、3.0124/6.2702 | p4b 4 格 | 直接 p50 | ？ |
| 记录 `EXP-p2-02` 采用值 | 3.6828 | `p2-02 fp16 cg 1024` | 直接 p50 | ？ |
| 记录 `EXP-p2-04` 采用值 | 2.6258 / 3.0004 | `p2-04 eager cg` | 直接 p50 | ？ |
| 记录 `EXP-p2-05` 采用值 | 2.5134 / 2.7483 | `p2-05 kernel cg` | 直接 p50 | ？ |
| 记录 `EXP-p0-04` 采用值 | 5.2637（1024） | `p0-04 1024` | 直接 p50 | ？ |
| `README` 状态 bullet / `PORTFOLIO` 简历数字 | 1.19×、1.74×、1.24× 等 | 同上各处 | 同上 | ？ |

\* RESULTS 现写 23%，按 payload 公式得 22.1% —— 刷新时**一并按公式统一**（这不是刷新引起的，
是之前的口径取整/来源差异，改的时候在记录里注明）。

## 2. 不随刷新变化的（不要动）

| 量 | 值 | 出处 | 为什么不变 |
| --- | --- | --- | --- |
| 执行轴（eager→cudagraph） | 2.43× / 1.35× | `results/EXP-p2j-refresh/archive/` 的两个 payload | RESULTS 明确用**归档里那一轮**；跨轮不可混 |
| 融合 RMSNorm A/B | 1.2358× | `EXP-p4c-02` §5.6 | 同进程交错 A/B，不是 graph 格 |
| 投影 kernel 修复 | 8.14 → 5.15 µs | `EXP-p4c-01` §5.4 | 图内配置扫描，不是 step p50 |
| kernel 级量化收益 | 96.3 → 63.5 µs = 1.52× | `NCU_EVIDENCE` §3 | 隔离计时，刷新轮不重测 |
| 起点/字节下限 | 82.6436 ms、1.5982 ms、≈52× | `EXP-p0-02` payload | 起点格 host 主导；payload 自带地板 |

## 3. 哪些格会动（按开关作用域）

| 开关 | 生效格 | 说明 |
| --- | --- | --- |
| `route` | **所有** MoE 格（`moe_static` / `moe_grouped`） | 含 fp16 的非 kernel 格 |
| `rope` / `skip_mask_fill` | `attn=kernel` 的 6 格 | 非 kernel 格上 `mask-fill` 会**显式拒绝** |
| `packed_qkv`（F2） | 其中**量化**的 2 格（p4c-01 w8a16×kernel 1024/4096） | fp16 kernel 格是 `nn.Linear` ⇒ 不适用，**no-op**（`packed_qkv_effective=false`） |

⇒ 刷新轮之后：`p0-04/p2-02/p4b/p4c-sdpa` 只反映 `route`；`p2-05/p4c-kernel` 反映
`route`（fp16 kernel 格）或 `route+rope+mask+F2`（w8a16 kernel 2 格）。

## 4. 刷新后的验收顺序

1. 逐格读刷新 payload：`status=done`、`gate`、`host_share≤5%`、`config.fused_route=true`
   （kernel 格再查 `fused_rope`/`skip_mask_fill`；**F2 看 `config.packed_qkv_effective`**，
   不看 `config.packed_qkv` —— 后者是"请求"，采纳后 21/23 格会是 `true` 而实际没装）；
2. 整轮 `round_canary`（首尾同格 graph）非零 ⇒ **VOID，什么都不改**；
3. 通过后才改 `FUSION_DEFAULTS`（四个 True）→ 按 §1 表填数字 → 四门 → 提交；
4. 每份记录加"★ 2026-10-2x 刷新"小节，**注明该轮是 combined（F1+F2）**。

## 5. 翻默认会碰到的连锁（2026-10-24 已在本地理顺，照此核对）

| 连锁 | 现在的行为 |
| --- | --- |
| `packed_qkv` 有两个前置（`attn=kernel` **且** 量化） | 不适用 ⇒ **no-op**（不安装、不报错）；payload 记 `packed_qkv`（请求）+ `packed_qkv_effective`（生效）。`blocked`/`failed` 的 payload 生效恒为 `False` |
| 两个测试曾写死"默认全 False" | 已改成对着 `FUSION_DEFAULTS` 断言 + 新增 `packed_qkv_applicable` 用例。★ **2026-10-25 复核**：该结论当时**只对 packed_qkv 轴成立**——`test_fused_rope` / `test_skip_mask_fill` / `test_routing_fusion`(gpu) 三处还写死 `is False`；`fused_route` 无 CPU 回退（翻默认后 ~15+ 个 CPU 测试会 RuntimeError）；`skip_mask_fill` 对非 kernel 图格 replay 会 raise（A3 的无 env 金丝雀 p4b/sdpa 会整格 blocked）。**已全部修复**（`d02442e`：route 可用性回退 + replay 门控 `_replay_skip_mask()` + 三断言改对 `FUSION_DEFAULTS`）——离线彩排：临时四默认全翻 ⇒ `pytest -m "not gpu"` **674 passed / 33 deselected 全绿** ⇒ 现在"翻默认不需要改测试"才真正成立 |
| 翻默认的 commit 里要顺手改的"默认关"注释（今天仍是正确描述，勿提前改） | `fusion_switches.py` 模块 docstring 的表格、`packed_qkv.py` docstring、`rope.py` 开关 docstring、`moe_static._fused_route_enabled` docstring、`cudagraph.py` 的 `_skip_mask_fill_enabled` 附近、`attn_decode.py` packed 回退注释、`_routing_triton.py` 模块 docstring、`docs/FUSION_OPPORTUNITIES.md` F2 候选状态行——与 A3 的默认翻转放同一个 commit，不留"注释说默认关、代码默认开"的矛盾 |
| 显式请求的 A/B（`--ab-packed-qkv`） | 仍**拒绝** fp16 / 非 kernel 格：那是主动比较，静默 no-op 会量出假的 0% |
| `scripts/rounds/run_ab_fresh_process.sh` 在翻默认之后 | off 臂必须给 `ENV_OFF="LEMOQ_PACKED_QKV=0"`；不给 ⇒ fail-closed 拒绝（否则两臂同实现） |
| `scripts/rounds/run_f1_refresh.sh` 的开关写法 | 四个开关**显式写 0/1**（不再 `unset`），所以采纳前后行为一致；断点跳过按本格期望值判断 |

## 6. 刷新后顺手要订正的计数（与采纳无关，但同一次提交做完）

`RESULTS.md` §5 的测试计数、`ROADMAP` §1、`PORTFOLIO` L105 已按 2026-10-24 实测订正为
**672 passed / 33 deselected / 40 个测试文件**（2026-10-25 起 **674 passed**：翻默认安全性修复
新增 replay 门控与 route CPU 回退两条用例，`d02442e`）；MATURITY_REVIEW 轮起
**694 passed / 42 个测试文件**（+20 = R1 显著性口径的两组测试，`2ccce61`；F4 K 门轮起
**700 passed / 43 个测试文件**，+6 = KV 直写离线 K 门）；刷新后再按新的
`results/` 复核 `payload` 计数。★ 2026-10-25 对账：Mac 侧 `EXP-*__*/result.json` 实为 **59 份**
（48 done / 10 planned / 1 blocked——p0-03 by-design），此前的"101 份（84/13/4）"口径
无法在 Mac 树上复现（59 格 + 31 份 p2j 归档 + 顶层历史工件另计）；**操作性验收以
`DRY=1 F2=1` 清单 = 23 格为准**（23 格所需 payload 已核实全部在盘，见 ISSUES N5）。

## 7. 刷新后重排算子账本（S2 的输入）

用刷新后的 `EXP-p4c-01__bit_width=w8a16_attn=kernel_ctx=1024` 取 `bottleneck.top_kernels`，
按"HANDOFF_2026-10-24 §5"的预注册规则算 `gap` 与可捕获量，再把
`docs/FUSION_OPPORTUNITIES.md` §1 的账本整表换成新值（该文 §1.2 的字节数**不随刷新变**，
只有实测 ms 与达成率变）。**不要**用旧账本挑选 S2 的算子。

★ 2026-10-25：这一步已机械化 —— `python -m tools.operator_ledger <cell>/result.json`
（`tools/operator_ledger.py`）内置 §1.2 的固定字节表与捕获率，直接输出排名；
用它对**刷新前** payload 复算与 HANDOFF §5 预演逐条一致（F4 KV 直写 97.9 µs > F3 43.2 >
gate_up 28.1 > down_sum 25.0；lm_head 的 gap ≈ −5 µs 被规则自动过滤，见 ISSUES N7；
dequant_gemm 不进榜——F2 本轮已采纳其 config 改善）。刷新轮跑完后再对同一格跑一次即为正式账本。
