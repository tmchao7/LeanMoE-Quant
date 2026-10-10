---
feature: routing-fusion
status: delivered
updated: 2026-10-10
branch: opt/elementwise-fusion
commits: 2591b9d..c0c3611
---

# Routing Fusion — softmax+topk 单 kernel 融合（候选，不采纳）

## Report

**What was built** — 路由融合候选全链路：`tools/profile_graph` 首轮定位出路由桶
（softmax+topk+sort ×16 层 = 0.110 ms/step ≈ busy 的 5.9%）；`_routing_triton.py`
单 kernel 融合 softmax(fp32)+K 轮 argmax top-K（E/K 全 constexpr、capture 安全、
不依赖 tl.sort）；`_route` 增加 `LEMOQ_FUSED_ROUTE` 开关（**默认 aten，已录路径零影响**）；
A/B 工具 `profile_graph --ab-fused-route`（同进程交替 + 金丝雀 + 跨臂 token 比对）。
**候选结论（未采纳）**：同进程 A/B 实测 fused 臂 wall 1.9521 → 1.8935 ms（**+3.0%**）、
gpu_busy 1.8743 → 1.7954 ms（**+4.2%**），金丝雀 aten 漂移 2.215% ≤5%、
32-token 序列跨 4 次 run 逐位相同。工件 `results/routing_ab.json`（status=measured）。

**Verification** — Mac 四门全绿（最终 **491 passed / 0 failed**）；服务器
`pytest -m gpu tests/test_routing_fusion.py` **10/10 passed**（fp32 ids 逐位、fp16
平局集合、env 端到端、非连续、非 2 幂 E、norm 双分支、默认仍 aten）；A/B 三验收
（金丝雀 2.215%、token 逐位同、fused 快 3.0%/4.2%）全部达成；跨臂贪心输出为
32×同一 token（合成 prompt 下退化轨迹——相等性结论成立，但轨迹多样性有限，已如实
记录）。

**Journey log** —
1. graph 格 payload 的 top_kernels 全 0、老工件是 hf 旧路径 ⇒ 先造 `profile_graph`
   再选靶，避免拍脑袋（首轮 profile 4 分钟 ¥0.4，直接给出5.9% 的路由桶）。
2. kernel 不用 `tl.sort`（丢 index、API 脆弱），改用 K 轮 argmax + one-hot 置 -inf：
   O(K·E) 全 constexpr 展开、平局取最低 lane、语义与 torch.topk 的"等值集合"对齐。
3. `tl.arange` 只接受 2 的幂 ⇒ E_POW/K_POW 双装垫 + mask，测试覆盖非 2 幂（E=12,K=3）。
4. 第二靶（mask 两次 fill_ / KV cat）在 `cudagraph.py` 注释里被原作者标为"后续单独的
   改动（一次一个变量）"——本轮不硬塞，留作下一候选（见 §4）。
5. 采纳路由融合 = 刷新 grouped 路径全部已录格（P2d 同款，估 ¥5+ 与一个记录轮），
   超出本轮预算 ⇒ 保持候选态，采纳与否留给用户决策。

## [S1] Problem

`tools/profile_graph` 首轮实测（w8a16×grouped×kernel-attn×fused-norm，图内最佳路径，
wall 1.94 ms / gpu_busy 1.876 ms / 539 kernels·step⁻¹）显示：路由链
`softmax_warp ×16 + gatherTopK ×16 + bitonicSortKV ×16` 合计 **0.110 ms/step ≈ 5.9%**，
是**最大的非权重、非注意力、可融合**桶（工件 `results/graph_profile.json`）。
现链路在 `moe_static._route`：`F.linear → F.softmax(fp32) → torch.topk(sorted)`，
全部是 ATen 小 kernel，在 capture 内逐层打 3 发。

## [S2] Design

- **新 kernel**：`src/lemoq/kernels/_routing_triton.py::routing_softmax_topk`——
  每 token 一个 triton program，fp32 softmax 后**串行 insertion 选 top-K**（E/K 为
  `tl.constexpr`，不依赖 `tl.sort`；OLMoE = E64/K8）。输出 `top_w(fp32→cast x.dtype)`、
  `top_i(int64)`；`norm_topk_prob` 为 constexpr 分支。
- **接线**：`moe_static._route` 增加 `LEMOQ_FUSED_ROUTE=1` 分支（**默认关闭 = aten 原路**，
  环境变量在 Python 相位读取 ⇒ capture 烘焙所选实现；已录格子、默认路径零改动）。
  `logits = F.linear` 保留 ATen（router gemvx 不在本 feature 范围）。
- **capture 安全**：prefill（eager）在 capture 之前必然执行 `_route` ⇒ JIT 已完成；
  形状全程静态（T、E、K 均 constexpr 或固定）。
- **数值/离散语义**：softmax 用与 ATen 同序的 fp32（max-sub + exp-sum）；top-K 选法在
  无平局输入上与 `torch.topk` 逐位一致；平局时集合相同（加权求和可交换，输出不变），
  以输出 parity 与 A/B token 序列为最终判据。
- **A/B 协议**（不写 result.json，工具输出独立 JSON）：同一进程内
  `aten→fused→aten→fused` 交替 4 次 `run_cudagraph`（各 3 repeats），收集
  wall_p50 / gpu_busy / `graph_output_tokens`；验收：
  ① 金丝雀 = 两次 aten 漂移 ≤5%；② 两臂 **token 序列逐位相同**；
  ③ fused 臂 wall 优于 aten 臂（MAD 口径，同进程）。
- **验收口径**：GPU K1 = 与 ATen 参考 `allclose(atol=rtol=1e-3)`（`PARITY_GATE`）+
  无平局样本 ids 精确相等。CPU 侧 oracle（numpy）与 torch 参考 bit 级对照可离线钉住。

## [S3] Out of Scope

- **不采纳**：`LEMOQ_FUSED_ROUTE` 不改默认值——采纳 = 触碰 grouped 路径已录格
  （p2-02/p4b/p4c 等），须走归档→重测→记录刷新（P2d/P2j 同款），另立项另算预算。
- router 矩阵乘（gemvx ×16，0.037 ms）融合进同 kernel（二期候选，本次不做）。
- KV 写入桶（index_copy+Cat，0.110 ms，5.9%）——第二候选，本次不做。
- 任何 result.json / 实验矩阵 / 记录的改动。

## Tasks

- [x] T1: kernel + oracle + 测试（Mac） — acceptance: `_routing_triton` 落盘；`_route` 带 env 分支（默认 aten）；CPU oracle 测试 + GPU K1 测试齐；四门绿 (covers: S2)
- [x] T2: GPU 测试 + A/B 实测（服务器） — acceptance: `pytest -m gpu` 相关用例过；A/B JSON 落盘且三验收（金丝雀≤5%、token 逐位同、fused 快于 aten 或如实记录）有结论 (covers: S2; depends: T1)
- [x] T3: 工件回传 + 分支交付 — acceptance: A/B JSON 回传本地；Report 填毕；分支推送；给用户采纳/不采纳的建议 (covers: S2; depends: T2)
