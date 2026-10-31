<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=soft&height=150&color=0:1E3A8A,55:2563EB,100:0D9488&text=LeanMoE-Quant&fontSize=42&fontColor=ffffff&desc=Byte-budget%20quantization%20%2B%20Triton%20kernels%20for%20MoE%20decode%20at%20batch%3D1&descSize=15&descAlignY=76" alt="LeanMoE-Quant" />
</div>

<p align="center"><b>English</b> · <a href="README.md">简体中文</a></p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-v7%20canonical%20adopted-12b886" />
  <img alt="tests" src="https://img.shields.io/badge/tests-733%20passed-brightgreen" />
  <img alt="gate" src="https://img.shields.io/badge/accuracy%20gate-KL%3C0.01%20%C2%B7%20top1%E2%89%A50.95-blue" />
  <img alt="gpu" src="https://img.shields.io/badge/GPU-RTX%205090%20%C2%B7%20sm__120-4c6ef5" />
  <img alt="model" src="https://img.shields.io/badge/model-OLMoE--1B--7B--0924%20%C2%B7%2064%20experts-845ef7" />
  <img alt="canonical" src="https://img.shields.io/badge/canonical-1.46%20%2F%201.61%20ms%20%C2%B7%20685%20%2F%20622%20tok%2Fs-20c997" />
  <img alt="best cell" src="https://img.shields.io/badge/best%20cell-2.44%C3%97%20%2F%204.25%C3%97-12b886" />
</p>

**Byte-budget-driven quantization + Triton kernel optimization for MoE decode at batch=1.**
On a single RTX 5090, single-request decode of OLMoE-1B-7B (64 experts / top-8) goes from
**82.6 ms/token** down to **1.46 / 1.61 ms** (ctx 1024/4096, 685 / 622 tok/s); the best cell is
**2.44× / 4.25×** over the native baseline — and the **measured structural ceiling** of this
route is established along the way. Every number on this page traces to an artifact; nothing
is published without passing its accuracy gate.

---

## 🏆 Highlights

**1 · Seven levers, each with its own same-round A/B, all adopted**
CUDA Graph / weight W8 / split-KV attention / fused RMSNorm / packed q·k·v / KV direct write /
lm_head W8 / **KV-INT8** — kernels per step **994 → 345** (figure below).

**2 · The quantization ceiling, measured**
The weight stream is only 40.9% / 23.5% of a decode step (backed out from refresh payloads) ⇒
even a perfect weight path caps at **~1.69× / ~1.31×**. The premise of Marlin-class 3–4×
kernels (4-bit passing the gate) does not hold for this model.

**3 · KV-INT8 read path (custom kernel)**
Per-token symmetric int8 with scale folding inside the attention kernel (zero unpacking on the
read path); near-lossless (gate KL 0.0006), **−109 µs/step (−6.5%)** @4096.

**4 · Negative results are first-class citizens**
All 4-bit variants (W4-RTN / GPTQ, MXFP4, mixed-bit fallback) were stopped by the accuracy
gate — evidence published, latency never; w8a8@4096's 5-window PASS was actively de-published
after a 10-window re-measurement showed it not robust; dequant fusion was refuted on the eager
path (the isolated kernel is 1.21× cheaper, end-to-end 5% slower).

**5 · Measurement discipline**
Same-cell canary first and last in every round (>5% drift VOIDs the round), independent-process
adjacent A/B with per-pair wins + MAD, host_share ≤5% per-cell self-check, every gate table
pinned to its artifacts by tests — 733 CPU tests + four CI gates.

---

## 📊 Metrics

### Decode step latency (ms/token, lower is better)

| Configuration | ctx=1024 | ctx=4096 | gate |
| --- | ---: | ---: | :---: |
| fp16 × sdpa (native baseline) | 3.5660 | 6.8329 | PASS |
| fp16 × split-KV kernel | 2.2028 | 2.3484 | PASS |
| w8a16 × sdpa | 2.9165 | 6.2298 | PASS |
| **w8a16 + lm_head W8 + KV-INT8 × kernel (canonical)** | **1.4602** | **1.6075** | PASS |

> Same-round cells only; best cell = fp16×sdpa / canonical = **2.44× / 4.25×**.

### Seven levers — same-round A/B step-time reduction

![Seven adopted levers](docs/figures/fig4_lever_ab.svg)

### To the bandwidth wall (82.6 ms start → 1.60 ms byte floor)

![To the wall](docs/figures/fig2_to_the_wall.svg)

### Per-axis speedups

![Per-axis speedups](docs/figures/fig1_per_axis_speedups.svg)

### Kernel bandwidth attainment (isolated, L2 flushed)

![Kernel attainment](docs/figures/fig3_kernel_attainment.svg)

### Accuracy (gate: KL < 0.01 and top-1 ≥ 0.95)

| Surface | KL | top-1 | Task-level |
| --- | ---: | ---: | --- |
| weight w8a16 | 0.0014 | 0.976 | PPL +0.0088% (wikitext-2 full) · MMLU −0.50 pt |
| lm_head W8 | 0.000061 | 1.000 | (same three chains) |
| KV-INT8 | 0.0006 / 0.0013 | 0.988 / 0.982 | forced-trajectory gate (165 positions/ctx) |

The one-page result (every number with its artifact path): **[RESULTS.md](RESULTS.md)**.

---

## Reproduce

```bash
# 1) Environment (all non-GPU tests run on Mac/CPU; latency measurement needs an NVIDIA GPU + triton)
uv venv -p 3.12 .venv && uv pip install -e ".[dev]"

# 2) Correctness tests (733, no GPU needed)
uv run pytest -m "not gpu" -q

# 3) Machine calibration: measured achievable bandwidth (the roofline denominator; spec values are banned)
uv run python -m tools.probe_machine --json results/machine_probe.json

# 4) Run one experiment cell (example: the canonical cell w8a16×kernel×cudagraph @1024)
uv run python -m benchmarks.runner \
    --exp EXP-p4c-01-in-graph-attention \
    --model <local OLMoE snapshot> \
    --ctx 1024 --bit-width w8a16 --attn kernel \
    --execution cudagraph --moe grouped --dequant fused --force

# 5) Regenerate all figures from artifacts (the 4 SVGs on this page are exactly this output,
#    each with a per-figure source footnote)
uv run python -m tools.make_figures
```

The experiment matrix (the only source of cells) is [configs/experiments.yaml](configs/experiments.yaml);
ops docs (`ENVIRONMENT.md` etc.) are kept maintainer-local and not in the repo; technical
references live in [docs/reference/](docs/reference/).

---

## Repo layout

```text
RESULTS.md                 # ★ the one-page result (every number cites its source)
experiments/<EXP-ID>.md    # ★ the authoritative record per experiment (pre-registered
                           #   hypothesis/criteria/results/applicability)
src/lemoq/kernels/         # 6 Triton kernel families: grouped MoE GEMV, fused dequant GEMM,
                           #   split-KV decode attention, fused RMSNorm, packed q/k/v, KV-INT8
src/lemoq/quant/           # quantization (W8/W4 per-group, gate citations, per-surface gate tables)
benchmarks/runner.py       # harness: five load-time refusals + payload schema + correctness gate
tools/                     # probe / operator ledger / figures / quant gate toolchain
tests/                     # 733 CPU tests (parity gate, schema guards, pinned gate tables)
docs/figures/              # 4 figures, regenerated from artifacts
docs/reference/            # METRICS / MODELS / SPECS / NCU_EVIDENCE (technical reference)
```

> Applicability: every number is a **batch=1 single-stream** measurement on one RTX 5090
> (sm_120), OLMoE-1B-7B-0924, ctx ∈ {1024, 4096}; conclusions do not transfer across
> hardware/models (each record's Applicability section states the boundary).
