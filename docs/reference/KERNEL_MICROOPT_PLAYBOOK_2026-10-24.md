# Kernel 微优 playbook — 带宽/寄存器压力/launch 开销(2026-10-24 调研)

> **性质：联网调研记录**（公开报告值/业界实践，非本项目实测），服务于两条线：
> S2=F4 的 kernel 设计依据（已落地 `kv_direct.py`）与后续 down_sum/gate_up 的
> 极限优化（67.7%/78.8% 达成率的 config 类余量）。引用全部带出处。

## 1. KV 直写 kernel 的设计依据（vLLM `reshape_and_cache_flash` 形态）

- vLLM 有专门的编译 pass 把 rope 与 KV 写融成 `do_rope_and_kv_cache_update`
  （[rope_kvcache_fusion.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/passes/fusion/rope_kvcache_fusion.py)、
  [issue #24678](https://github.com/vllm-project/vllm/issues/24678)）——本仓 F4 是同一融合的手工版。
- CUDA 参考实现（[cache_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu)）要点：
  `slot_mapping` 哨兵（<0 早退）、显式 stride（block/page/head）、16B 向量拷贝、
  **NHD 布局 = 平铺无转置**（HND 旧布局的 strided scatter 就是官方文档化的反面教材）。
- 本仓采纳：grid=(2H,)、BLOCK=head_dim、num_warps=1（256 B/head 是 latency-bound，
  多 warp 只加同步）、int32 偏移、pos 从设备端 `cache_position` 读（图安全）、
  **默认 cache 策略**（写入的 KV 下一步被 attention 读，96 MB L2 应留住——勿 evict_first）。
- 判据：N 个 aten kernel → 1 个 Triton kernel 赢的条件是 `N × (节点间隙 + 往返) > kernel 效率损失`；
  图内每节点仍有 ~1–3 µs GPU 侧间隙（[ORNL slides](https://www.olcf.ornl.gov/wp-content/uploads/2021/10/013_CUDA_Graphs.pdf)、
  [PyGraph arXiv 2503.19779](https://arxiv.org/abs/2503.19779)）——32×index_copy ≈ 50 µs vs 一个 ~5 µs kernel,
  即使 kernel 平庸也是净赢。S2 账本 50.3 µs 可捕获与此一致。

## 2. 寄存器压力与 occupancy(下一步给四族 kernel 建档)

- **测量**：编译产物 `kernel.n_regs / n_spills`（cubin load 时驱动填充，
  [mamba #386](https://github.com/state-spaces/mamba/issues/386)）+ `metadata.num_warps/shared`；
  PTX/SASS 落盘 grep `st.local`。⚠️ **Triton #1756**：含 sin/cos 的 kernel `n_spills` 恒报 >0
  （[bug](https://github.com/triton-lang/triton/issues/1756)）——RoPE 类 kernel 的 spill 读数要 SASS 复核。
- **occupancy ↔ 带宽达成率**（Little 定律）：1.79 TB/s × ~400–600 ns DRAM 延迟 ⇒ 全卡需
  ~0.7–1.1 MB 在途请求；每个 SM 只靠驻留 warp 贡献在途量——**occupancy 不足的流式 kernel
  即使完美 coalescing 也只有 30–60% 达成率**。流式目标 ≥4–8 CTA/SM 或 ≥32 独立 load 流/SM。
  （[Hopper 微基准 arXiv 2401.12155](https://arxiv.org/abs/2401.12155)、
  [CUDA Best Practices: Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)）
- num_warps 加大 ⇒ 每线程元素更少、寄存器省，但 barrier 多（[triton-shared softmax 例](https://github.com/facebookincubator/triton-shared/blob/main/python/examples/test_softmax.py)）；
  BLOCK 加大 ⇒ launch 摊薄但压力升。

## 3. 带宽达成率 playbook(5090 实况)

- 5090：GDDR7 512-bit，**spec 1792 GB/s**，96 MB L2，170 SM
  （[TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)）；
  社区流式实测 ~1.55–1.7 TB/s（85–95%）——本仓 1560.6 GB/s 的分母口径与之相符；
  L2 命中可超 100%（社区报 123%，[论坛](https://forums.developer.nvidia.com/t/l2-bandwidth-profiling-shows-100-peak-123-4-on-rtx-5090/345906)）。
- **128-bit 向量访存**是拷贝类第一杠杆（[NVIDIA Pro Tip](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)）；
  拷贝类优先精确整除（无 mask）而不是部分 tile。
- **cache modifier**（[Triton load/store 文档](https://triton-lang.org/main/python-api/generated/triton.language.load.html)）：
  权重流（每步读一次不重用）可试 `evict_first` 保护 L2 里的 KV/激活——**config 类单变量实验**；
  KV 写**不要**流式化（下一步就重读）。
- 尾效应：维度整除 BLOCK 时无 tail；本仓 head_dim=128、hidden=2048 均整除。

## 4. launch/派发微开销

- `cudaLaunchKernel` ≈5 µs CPU 侧（[NVIDIA 论坛](https://forums.developer.nvidia.com/t/any-way-to-measure-the-latency-of-a-kernel-launch/221413)）；
  Triton **eager** 派发历史上有 Python 参数处理大头（#2637：220 µs launch vs 80 µs run；
  #3619 回归）——**CUDA Graph capture 恰好消掉这段**，但图内每节点 GPU 侧间隙仍在 ⇒
  节点数本身是成本，融合 32 节点为 1 有真实收益。
- FlashInfer 的 append 路径有图兼容坑（[issue #919](https://github.com/flashinfer-ai/flashinfer/issues/919)）；
  本仓静态 cache + 固定形状天然规避。

## 5. 小拷贝 kernel 坑清单

1. 8 KB 级一个 CTA 足够（latency-bound，~1–3 µs）；**不要**为"用满 170 SM"开 grid。
2. 多 CTA 只在 ≥64–256 KB 才有跨 SM 延迟隐藏收益。
3. 拷贝类优先精确整除、少 mask。
4. int32 位置算术（int64 指针算术在 latency-bound kernel 里双倍地址 ALU）。
5. num_warps=1–2（≤8 KB）；更大只加同步。
6. 无 `tl.dot`、无整 tile fp32 上转——LDG→STG；SASS 复核。
7. KV 写保持默认/`evict_last`；`evict_first` 只用于不重读的流。
8. 位精确：fp16→fp16 纯拷贝无舍入；K1 在 rope 后的 k 上比，不只比布局。

## 6. 给 down_sum(67.7%)/gate_up(78.8%) 的候选实验(全部 config 类,一轮一变量)

1. down_sum：`num_warps` {2,4,8} 扫描 + 输出维 BLOCK 加宽（每线程向量更宽）；
2. down_sum/gate_up：expert 权重 load 加 `evict_first`（一步内不重用，护住 L2 里的 KV/激活）；
3. gate_up：`num_stages` {2,3,4} 扫（更深 load 流水 ⇒ 更多在途字节），盯 n_regs/n_spills 过拐点；
4. gate_up：num_warps 8 vs 4（更多线程 ⇒ 更少寄存器/线程 ⇒ 流式达成率的 occupancy 上限）；
5. 每次改动后 SASS 复核 spill；KV 写 kernel 落地后**重测四族达成率**（L2 内容与节点数变了，
   达成率漂移会被误读成 kernel 回退）。
6. 建档：下次刷新轮顺手记录每族 `n_regs/n_spills/shared/num_warps`（SASS 复核 sin/cos 伪 spill）。
