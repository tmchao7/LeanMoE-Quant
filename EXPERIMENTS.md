# EXPERIMENTS — 记录格式与指标定义

本项目每一次测量都写成同一个形状。**没有按这个格式写下来的结果，对本项目来说不存在。**

每个实验一个目录：`results/<EXP-ID>/`，内含 `env-<date>.md`、原始日志、以及填好的记录。
`results/` 进 `.gitignore`；结果稳定后再把记录摘录进提交。

命名：

```text
EXP-<phase>-<nn>-<slug>

EXP-p0-01-machine-probe
EXP-p0-02-baseline-fp16-alignment
EXP-p1-01-w8a8-functional
EXP-p2-02-dequant-fusion-vs-separate
EXP-p4-01-moe-routing-overhead
```

---

## 1. 必填字段

### Model

```text
model:              allenai/OLMoE-1B-7B  (primary MoE; note any deviation)
arch:               olmoe
dtype:              float16            # V100: 一律 fp16；bf16 不可用
layers:
hidden:
num experts:
top-k:
expert FFN dim:
total params:
activated params:
```

### Hardware

```text
GPU:                Tesla V100 (SXM2 / PCIe — 型号必填)
GPU memory:         32 GB per card
GPU count:          physical cards on the machine (15)
GPU count used:     1   # 本项目恒为 1；>1 必须显式说明并标注为扩展实验
measured achievable bandwidth:  <GB/s>   # 来自 tools/probe_machine.py，勿用规格值
```

### Runtime

```text
Python:
PyTorch:            (CUDA build)
CUDA (driver + torch build):
transformers:
triton:             (if used)
```

### Configuration

```text
bit-width:          W8A8 / W8A16 / W4A16 / fp16
group size:         32 / 64 / 128 / per-channel / per-tensor   (W4A16 必填)
dequant:            fused-into-gemm / separate-kernel / library
fusion:             none / rmsnorm+rope / swiglu / attention
kv cache dtype:     fp16 / int8
context length:
batch size:
prompt / max new tokens:
warmup / timed iterations:
```

### Results

```text
# 正确性（先填，且必须过门槛才能填性能）
KL divergence (vs fp16 reference):
top-1 agreement:
PPL (wikitext-2):            fp16 = _, this config = _, delta = _
task score:                  GSM8K / MMLU subset (if applicable)

# 性能
TTFT p50 / p99 (ms):
per-token p50 / p99 (ms)  @ batch = 1:
throughput (tok/s)        @ batch ∈ {1,2,4,8}:
bytes-per-token [W] = _   # 权重：attention _ + router _ + experts+scale _ + shared _ + lm_head _
bytes-per-token [T] = _   # = [W] + KV 读(ctx)；KV @ctx = (2 × 层数 × kv_heads × head_dim × ctx × dtype字节)
context length = _        # 达成率必须带 ctx（口径见 docs/METRICS.md §3.2）
floor_W (ms/step) = [W] ÷ measured bw = _
floor_T (ms/step) = [T] ÷ measured bw = _
quantization attainment = floor_W ÷ measured latency = _%   # 量化兑现了多少
bandwidth attainment    = floor_T ÷ measured latency = _%   # 端到端离墙多远

# 显存（必须分列，用 allocator 峰值）
weights:      _ GB
KV:           _ GB
activations:  _ GB
workspace:    _ GB
peak total:   _ GB

# 归因
dequant share of step time:  _%    # 融合 vs 不融合的关键指标
routing/packing share:       _%    # MoE 专用；见 E4
```

### Verdict

```text
verdict:    PASS / CASCADE / FAIL / planned
hypothesis: <跑之前写下的假设>
outcome:    supported / refuted / inconclusive
notes:      负结果同样记录；与其它实验的差异必须显式说明
```

---

## 2. 指标定义（完整口径见 [docs/METRICS.md](docs/METRICS.md)）

| 指标 | 定义 | 为什么用它 |
|---|---|---|
| **KL 散度** | 本配置与 fp16 参考在 logits 上的 KL（固定 prompt 集，逐位置平均） | 量化文献标准主指标；不受采样策略影响 |
| **top-1 一致率** | 与 fp16 参考 greedy 首 token 的一致比例 | 直观、可解释 |
| **PPL** | wikitext-2 上的困惑度，报绝对值与相对退化 | 与传统对照可比 |
| **带宽达成率** | `理论下限 / 实测延迟`，下限 = 每 token 字节 ÷ **实测可达带宽** | 物理口径，跨硬件可比 |
| **dequant 占比** | 反量化耗时 ÷ 单步总耗时 | 判断"融合是否必要"的直接证据 |
| **路由占比** | 路由/打包耗时 ÷ 单步总耗时（MoE） | 量化收益上限的来源 |

---

## 3. 记录纪律

1. **假设先写**：跑之前把假设与预期的反证信号填进 `Verdict.hypothesis`，避免事后解释。
2. **一次一个变量**：位宽、group size、融合策略不同时改。组合实验必须标 `combined`。
3. **跨阶段不可混比**：不同环境/不同卡/不同阶段的数字不能放在同一张表里比较。
4. **三态标注**（沿用 MegaKernel-vLLM 的做法）：`已实测` / `待标定`（测了但物理上说不通，
   必须先复采）/ `待跑`（工具就绪但没数据）。
5. **正确性门槛优先**：KL/top-1 不达标 → 只记录正确性失败，**不报性能数字**。
6. 每份记录末尾必须写清**适用区间**（哪个卡、哪个 batch、哪个精度档），以及**不适用**的情况。
