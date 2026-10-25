# 开机前工作计划(Mac 本地,≈3 h,不碰 GPU、不翻默认、不覆盖任何 result.json)

## Block 0 — §8 自检 + 计数对账(≈15 min)
1. 四门:`.venv-mac/bin/python -m pytest -m "not gpu" -q`(期望 672/33)、`ruff check .`、`black --check .`、`python -m tools.check_docs`。
2. payload 计数对账:按口径分项数(格子 / p2j 归档 / base 目录 / 顶层 json),解释 59 vs 101,修正 A0 验收线的期望值写法。
3. DRY 自检清单:**先做 Block 1.1 的 DRY 修复,再跑** `DRY=1 F2=1 ... run_f1_refresh.sh`(期望 23 格、恰好 2 行 packed=1)——顺序反了会把 f1.log 再抹一次。

## Block 1 — review/debug 修复(全部 Mac 可验证,分三个 commit)

**1.1 `scripts/rounds/run_f1_refresh.sh`**(`scripts:` commit)
- mkdir/清空日志移到 DRY exit 之后(DRY 不再碰 results/);
- 清单生成加 try/except(坏 payload 报警并中止,不以残缺清单续跑);
- 失败格不再静默:统计进对账(清单数 = 刷新数 + 显式失败数),不一致 ⇒ 非零退出。

**1.2 `scripts/rounds/run_ab_fresh_process.sh`**(同 commit)
- 守卫 fail-open 堵住:DEFAULT_STATE 探针失败 ⇒ 拒绝(fail-closed);
- 校验 ENV_OFF 关的是 ENV_ON 同一开关;
- 分析端逐 run 校验 switches 与臂一致(堵 off 臂环境残留、旧 payload 错配两个静默假结果路径)。
- 离线验证:Mac 上 DRY=1 与守卫路径各跑一遍;与 N2 修复同款的 payload 形态离线用例(若能挂进现有测试结构)。

**1.3 翻默认安全性(A3 前置,`kernel:` commit,引用"2026-10-25 review")**
- 三处写死断言改为对 `FUSION_DEFAULTS` 断言(L4 同款):`test_fused_rope.py:39`、`test_skip_mask_fill.py:85`、`test_routing_fusion.py:124`;
- `moe_static._route` 加 rope 同款可用性回退:非 CUDA 张量/无 triton ⇒ 走 ATen 原路(GPU 行为不变),配 CPU 测试钉"回退输出与强制关逐位相同";`routing_softmax_topk` 本身的严格性不放松(显式调用仍 raise);
- `cudagraph.py` replay 两处调用点改为 `skip_mask=_skip_mask_fill_enabled() and self._captured_mask_independent`(不适用 ⇒ False,packed_qkv L1 同款 no-op 模式);更新 `test_replay_paths_wire_the_flag...` 源码契约;核对 `tools/profile_graph` 的 mask-fill 显式 A/B 拒绝仍在(主动比较不许静默 no-op);
- **离线彩排**:临时翻 `FUSION_DEFAULTS` 四 True → `pytest -m "not gpu"` 应全绿 → 还原。这就是 A3 第 3 步的本地预演,彩排结论写进记录。

**1.4 文档订正**(`docs:` commit)
- ADOPTION_SYNC §5"翻默认不需要改测试"改准确(仅 packed_qkv 轴成立);
- ISSUES 复盘补本轮发现:N5 计数口径、N6 三处写死断言 + route 无回退 + mask replay 炸(阻断级)、N7 lm_head 的 S2 预演值(127×0.5=63)与预注册 gap 规则矛盾;
- HANDOFF §8 自检的 DRY 顺序注记(先修后跑)。

## Block 2 — 离线算子瓶颈账本(≈45 min,`analysis:` 或随 1.4 提交)
1. 新工具 `tools/operator_ledger.py`:读 result.json → 输出每 kernel 的 实测 ms / 必要字节(§1.2 固定字节表,标注"模型形状定死、不随刷新变")/ 地板 / gap / 按预注册捕获率(0.9 位精确延迟、0.3 配置、0.5 需门)加权的可捕获量,gap<30 µs 过滤并排序。刷新轮之后一条命令重算(ADOPTION_SYNC §7 的机械化执行器)。
2. 用现有 p4c-01 1024/4096 两格跑出 **pre-refresh 排名**(明确标注):预期 F4 KV 直写 98 µs 居首、F3 残差 43、gate_up 28、down_sum 25;lm_head 按 gap 规则过滤(gap≈−5 µs,已在墙上);4096 侧 attention 达成率 94.3% 佐证 S2 以 1024 为主。
3. 结论只进报告与 ISSUES 复盘,**不改 S2 决定**——最终选择等刷新后用本工具的输出机械执行。

## Block 3 — 收尾
- 三个 commit 依次过四门后推 `opt/packed-qkv`;**不 merge main**(等 Block A 采纳轮)。
- 最终报告:自检结果、修复清单(file:line)、瓶颈排名表、A0 验收线口径修正、A1–A4 待命状态(等你开机给端口)。

## 红线
不动矩阵轴/格 id;不覆盖 result.json(1.1 正是保护这条);不放宽任何门;不在本轮翻默认(只做使其安全的修复 + 离线彩排);F2 证据留给 A1;不 commit results/。