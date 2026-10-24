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
| `packed_qkv`（F2） | 其中**量化**的 2 格（p4c-01 w8a16×kernel 1024/4096） | fp16 kernel 格是 `nn.Linear`，install 拒绝 |

⇒ 刷新轮之后：`p0-04/p2-02/p4b/p4c-sdpa` 只反映 `route`；`p2-05/p4c-kernel` 反映
`route`（fp16 kernel 格）或 `route+rope+mask+F2`（w8a16 kernel 2 格）。

## 4. 刷新后的验收顺序

1. 逐格读刷新 payload：`status=done`、`gate`、`host_share≤5%`、`config.fused_route=true`
   （kernel 格再查 `fused_rope`/`skip_mask_fill`/`packed_qkv`）；
2. 整轮 `round_canary`（首尾同格 graph）非零 ⇒ **VOID，什么都不改**；
3. 通过后才改 `FUSION_DEFAULTS`（四个 True）→ 按 §1 表填数字 → 四门 → 提交；
4. 每份记录加"★ 2026-10-2x 刷新"小节，**注明该轮是 combined（F1+F2）**。
