# EXP-p3-02-mxfp4-block-scale

> MXFP4（OCP MX v1.0：E2M1 码 + 32 元素共享 E8M0 幂 2 尺度，4.25 bits/权重）作为
> **存储格式**的量化实验。结论先说：**精度门 FAIL，负结果关闭**——RTN-MXFP4 比当年
> 判死 int4-RTN 的同一道门**更差**，方向与量级精确命中预注册预判
> （[`docs/reference/QUANT_FORMAT_RECON_2026-10-25.md`](../docs/reference/QUANT_FORMAT_RECON_2026-10-25.md) §0.6：
> "E8M0 幂 2 组尺度比 int4 的 fp16 组尺度更粗"）。

- **ID**: `EXP-p3-02-mxfp4-block-scale`
- **Date**: 2026-10-25
- **Machine**: AutoDL RTX 5090 32GB（sm_120，实测带宽 1560.6 GB/s）；torch 2.8.0+cu128 / triton 3.4.0 / transformers 4.57.6
- **Status**: `done`（**负结果**；精度门 FAIL ⇒ 无延迟发布，两格 `separate/fused` 不跑——
  与 [EXP-p1-05](EXP-p1-05-w4-expert-calibration.md) 的处置先例一致：数值都不对的配置，
  性能测量没有意义）
- **原始数据**:
  - `results/EXP-p3-02-gate-mxfp4-ctx1024.json` — 精度门工件（`tools/quant_probe.py`，
    wikitext 真实文本 **5 窗口** × 33 位置，ctx=1024，rmsnorm=fused，阈值 W4 档
    KL<0.05 / top1≥0.95，`--arms gate_mxfp4`）

## Hypothesis（矩阵冻结原文）

"FP4 的 GEMM 优势被反量化与 host 开销吃掉，端到端收益远小于 4x"；
refutation_signal = "端到端也拿到接近 4x"。

★ 2026-10-25 侦察（QUANT_FORMAT_RECON）退役了"必须先切 tl.dot"的前置（向量路径可实现），
并**预注册了本窗口的实际结果**："RTN-MXFP4 plausibly fails the same gate that killed
int4-RTN; pivot to NVFP4-style fp scales if FAIL"。门结果走了第一个分支。

## 结果

### 精度门（裁决性，负结果）

| 量 | MXFP4-RTN（本实验） | 对照：int4-RTN（EXP-p1-05） | 阈值（W4 档） |
| --- | --- | --- | --- |
| KL mean（5 窗口） | **0.0739**（sd 0.0202，min 0.0438，max 0.1033） | 0.0447 | < 0.05 ⇒ **FAIL** |
| top-1 agreement mean | **0.8788**（min 0.8182） | 0.903 | ≥ 0.95 ⇒ **FAIL** |
| 逐窗通过数 | 0 / 5 | — | — |

**机制解读**（与预注册一致）：E8M0 是**幂 2 组尺度**，组内分辨率完全由块 amax 的指数决定，
比 int4-RTN 的 fp16 组尺度**更粗**；叠加 E2M1 幅值格点稀疏（{0,.5,1,1.5,2,3,4,6}）与
>6s 饱和，误差全面超过 int4-RTN。这不是实现 bug——K1（下）证明 kernel 与主机侧语义逐位一致，
门量的是**格式本身**的精度。

### 工程交付（K1，全部 PASS——负结果里的可用资产）

- 主机侧量化器 `src/lemoq/quant/mxfp4.py`（OCP 规范：E2M1 RTN + E8M0 + 饱和 ±6），
  10 条规范对拍测试；
- Triton kernel 变体 `_dequant_triton.py`（`mxfp4=True`：无符号 nibble 解包 + 算术 E2M1
  解码 + `exp2(byte−127)` 尺度，无 tl.dot，图安全）；
- `QuantLinear` mxfp4 分支（scale 保持 uint8，独立组装路径）+ `quant_probe` 门臂 `gate_mxfp4`；
- K1：GPU 上 kernel vs 主机参照 allclose@1e-3（m=1/m=4/K=50 pad/K=2048）+ A6 合取半边，
  `tests/test_mxfp4_gpu.py` 3/3（教训在案：乘积二次幂精确 ≠ 归约顺序逐位一致）。

## 结论与后续

1. **RTN-MXFP4 负结果关闭**：不发布延迟、不跑 p3-02 性能格；矩阵假设的 refutation_signal
   未触发（端到端 4× 无从谈起——门都没过），按 EXP-p1-05 先例停止扩张。
2. **NVFP4 式两级尺度（16 元素块 + E4M3 fp 尺度 + fp32 二级）是 4-bit 的最后一条路**——
   属性**新格式新实验**，需登记/预算后才做（不做默认放弃 4-bit，8-bit 线照旧）。
3. FP8 存储与 int8 同字节（侦察 §0.1），维持跳过。
4. 工程资产（量化器/kernel/接线/K 门）保留——若 NVFP4 实验立项，E4M3 尺度与 16 元素块
   只需改尺度路径，解包/GEMV 骨架复用。

## 适用范围

OLMoE-1B-7B、batch=1、sm_120、RTN 量化（无 GPTQ/AWQ 重校准）、5 窗口 wikitext 门。
不外推到其它模型/更大的 NVFP4 类尺度设计。
