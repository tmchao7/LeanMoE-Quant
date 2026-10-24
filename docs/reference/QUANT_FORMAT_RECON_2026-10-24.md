# 量化格式侦察 — FP8 / MXFP4 作为**存储格式**(向量路径,非 MMA)— 2026-10-24

> **性质：联网调研记录**（公开报告值，非本项目实测），回答一个此前被记录为"前置"的问题：
> `experiments.yaml` 注释说 FP8/MXFP4 必须"先切 `tl.dot` 路径"——**该前置可以退役**。
> 用途：决定 EXP-p3-01/02 的实施顺序与形态。规则红线不变（AGENTS §2 规则 9/10）。

## ★ 裁决（2026-10-24 当窗实测，工件 `results/EXP-p3-02-gate-mxfp4-ctx1024.json`）

RTN-MXFP4 精度门 **FAIL**：KL mean **0.0739**（阈值 0.05；5 窗口 min 0.0438 也不过线）、
top-1 **0.879**（阈值 0.95）——**比 int4-RTN（KL 0.0447 / top1 0.903）更差**，§0.6 的预注册
预判（幂 2 组尺度更粗）方向与量级双双命中。负结果记录在
[`experiments/EXP-p3-02-mxfp4-block-scale.md`](../../experiments/EXP-p3-02-mxfp4-block-scale.md)；
kernel/量化器/接线/K1 四件工程资产全部 PASS 保留。**NVFP4 式两级尺度是 4-bit 的最后一条路**
（新格式新实验，需登记/预算）。下文 §0 的"先做 p3-02"决策已按此执行并关闭。

## 0. 决策摘要

1. **EXP-p3-01（fp8 存储）不值得做**：fp8 也是 1 B/权重 + 尺度——**与现有 int8 字节数完全相同**，
   无存储收益（payload 假设里本就写了"same class as INT8"）；唯一价值是精度数据点。跳过，
   除非做 w8a8-fp8 的激活量化研究。
2. **EXP-p3-02（MXFP4）是量化主线的真正下一张门票**：4.25 bits/权重（0.5 B + 1 B/32 尺度）
   ≈ int8 字节的一半；**向量路径可实现，不需要 tl.dot/MMA**。
3. **"tl.dot 前置"退役的依据**：① Triton 在 M=1 下 `tl.dot` 要求 M≥16（min_dot_size 16×16×16），
   等于 16× 权重读——与字节预算模型直接矛盾；② sm_120 本就没有 tcgen05（FA4 与社区 FP4 kernel
   报告均确认）；③ **业界同构先例**：vLLM 在消费级 Blackwell（SM120/121）上跑 MXFP4 就是
   **反量化→bf16 + Marlin W4A16**（issue #30135/#37030，无 tcgen05）；llama.cpp 的
   `vec_dot_mxfp4_q8_1` 是纯向量 dequant-on-load GEMV（16 项 LUT + E8M0 尺度 + dp4a）；
   AWQ 论文明确写"dequantize INT4 weights to FP16 in registers … on CUDA cores"。
4. **顺序**：先 p3-02（信息量最大的实验——它的门结果裁决"4-bit 对 OLMoE 是否可行"）；
   **风险如实**：RTN-MXFP4 大概率过不了当初杀掉 int4-RTN 的同一道门（E8M0 幂 2 组尺度比
   int4 的 fp16 组尺度更粗；NVFP4 用 fp 尺度正为此因）。**若 FAIL，退路是 NVFP4 式两级尺度**
   （16 元素块 + E4M3 fp 尺度 + fp32 per-tensor），再 FAIL 才真正关闭 4-bit。
5. FP8 若做：E4M3（E5M2 在小模型一致更差，arXiv 2502.01070）；尺度沿用现有 per-group fp16
   布局（8-bit 下粒度影响极小，同论文附录）；kernel 改动 = 换 load dtype + 寄存器转 fp16。

## 1. 关键事实与出处（均经核实）

- **MX 格式**（OCP MX v1.0）：32 元素/块共享一个 **E8M0** 字节尺度（值 = 2^(X−127)，bias 127，
  无 inf/NaN）；MXFP4 元素 = E2M1，可表示幅值 {0, .5, 1, 1.5, 2, 3, 4, 6}；
  量化式 `shared_exp = floor(log2(amax)) − emax_elem`（e2m1 的 emax=2）。
  [OCP 规范](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)、
  [MX 论文 Algorithm 1](https://arxiv.org/html/2310.10537v2)、[尺度格式解读](https://fprox.substack.com/p/ocp-mx-scaling-formats)。
- **NVFP4 vs MXFP4**：NVFP4 = 16 元素块 + **E4M3** 块尺度 + fp32 二级尺度——为"单一幂 2 尺度
  丢精度"而生。[NVIDIA blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)。
- **Triton 3.4.0 支持面**（源码核实）：`float8e4nv` 在 `capability >= 89` 进入 supported_fp8_dtypes
  ⇒ sm_120 可 load + `.to(tl.float16)`（PTX cvt）；**没有** fp4/e8m0 dtype——fp4 走 uint8 nibble
  解包（与本仓 int4 路径同构）+ `exp2(byte−127)` 尺度；`tl.dot_scaled` 是 MMA 路径且
  SM120 上软件模拟，M=1 不可用。
  [core.py](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/core.py)、
  [compiler.py](https://github.com/triton-lang/triton/blob/v3.4.0/third_party/nvidia/backend/compiler.py)。
- **torch 2.8 主机侧**：`torch.float8_e4m3fn`（2.1 起）、`float8_e8m0fnu` / `float4_e2m1fn_x2`
  （2.7 起）都在——**量化在主机侧做，kernel 只 load+convert**。
  [PyTorch 2.7 blog](https://pytorch.org/blog/pytorch-2-7)。
- **精度标定**（公开报告值）：E4M3 一致优于 E5M2（1B 模型 MMLU 45.5 vs 44.5）；fp8 weight-only
  与 bf16 差 ~0.15 pt（8B 模型 row-wise 62.80 vs 62.95）；MXFP4 直接 cast（W+A）很差
  （LLaMA-7B PPL 27.2 vs 9.5）——weight-only RTN 是另一个温和得多的 regime，但没有
  MXFP4-vs-INT4 的对照发表（核实缺失）。8-bit 下尺度粒度影响可忽略（per-tensor 62.90 vs
  row-wise 62.80）。

## 2. 实施坑清单（实现 p3-02/p3-01 前逐条对照）

1. `cvt.rn.satfinite.e4m3x2.f32` 会把超范围值**静默夹到 ±448**——尺度算错不报错，必须主机侧
   单元测试对拍（[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)）。
2. Triton fp8 转换有历史 bug（#3899 sm_89 PTX、#8822 inline-asm 全零）——上机先跑一个
   load→convert→store 的最小 parity 用例再进 kernel。
3. **E8M0 的 bias/off-by-one** 是最容易错的地方：`floor(log2(amax))−2`，字节 = shared_exp+127；
   错 1 位 ⇒ 整块下溢为 0 或饱和到 ±6。对拍 torchao `mx_formats` 参考实现
   （[pytorch/ao](https://github.com/pytorch/ao)）。
4. 组对齐：32×fp4 = 16 B/块（K 连续 + BLOCK_K 为 32 倍数自然满足）；padding 用 e2m1 code 0
   （解码 = 0.0）——现有 padded-K 方案兼容。
5. 保持 fp32 累加器，**不要**为"配平"fp8 而换 fp16 累加（K1 分辨率口径的既有担忧仍在）。
6. E8M0 尺度在 kernel 里 = `exp2(scale.to(f32) − 127)`（llama.cpp `e8m0_to_fp32` 的等价式）。

## 3. 与本仓的衔接

- 矩阵不变：EXP-p3-01/02 沿用既有声明（EXP-p3-02 的 `mxfp4_dequant=fused` 轴现成）；
  需要改的只是"tl.dot 前置"注释 → 由本文档替代（引用处：`docs/ROADMAP.md` Tier B1）。
- 新增实现清单（下个量化窗口）：主机侧 mxfp4 量化器（对拍 torchao）+ `_dequant_triton` 的
  mxfp4 load/decode 变体 + K1 parity（vs fp64 oracle）+ EXP-p1-04 式精度门 → 过门才有延迟发布。
- 引用原则：本文全部外部数字是**公开报告值**，进记录时必须注明非本项目实测。

## 4. 来源索引

[vLLM FP8 文档](https://docs.vllm.ai/en/stable/features/quantization/llm_compressor/fp8) ·
[arXiv 2502.01070](https://arxiv.org/html/2502.01070v2)（FP8 精度系统研究）·
[arXiv 2310.10537](https://arxiv.org/html/2310.10537v2)（MX 论文）·
[OCP MX v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) ·
[NVIDIA NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference) ·
[vLLM SM120 mxfp4 回退](https://github.com/vllm-project/vllm/issues/30135)、[#37030](https://github.com/vllm-project/vllm/issues/37030) ·
[llama.cpp vecdotq.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh) ·
[Triton v3.4.0 core.py](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/language/core.py) ·
[torchao mx_formats](https://github.com/pytorch/ao) ·
[AWQ MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf) ·
[Marlin arXiv 2408.11743](https://arxiv.org/pdf/2408.11743) ·
[FireQ arXiv 2505.20839](https://arxiv.org/html/2505.20839v2) ·
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) ·
[Triton #3899](https://github.com/triton-lang/triton/issues/3899) / [#8822](https://github.com/triton-lang/triton/issues/8822)
