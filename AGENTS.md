# AGENTS.md

Compact guide for agent sessions working in LeanMoE-Quant. Read this before editing.

## What this repo is

Research project on **byte-budget-driven quantization and kernel optimization for MoE decode
at batch=1**. Status: **Phase 0 (environment + baseline)** — no optimization is implemented, no
benchmark has been run. See `PLAN.md`.

- **Primary hardware: rented RTX 5090 32 GB (sm_120)**, single card. Single-card project only —
  never design something that needs more than 1 GPU.
- **Secondary (free): school V100 32 GB (sm_70)**, an optional second bandwidth data point. It
  lacks `cp.async`, FP8, FP4 and bf16 tensor cores, so it cannot host the modern kernel arms.
- **The Mac may run CPU `torch` for functional/unit tests** (`torch==2.14.0` CPU arm64 wheel),
  but **no number produced on the Mac may ever enter `results/` or any conclusion** — the Mac has
  no NVIDIA GPU, so nothing it measures is reportable here.
- Package name: `lemoq`; source under `src/lemoq/`.

## Project hard rules (an agent will get these wrong if it guesses)

1. **No invented results.** Unmeasured values are written `planned` / `target` /
   `to be evaluated` — never as achieved speedups. Negative and null results are reported too.
2. **Correctness gate before performance.** A performance number may only be published for a
   configuration whose KL divergence / top-1 agreement against the fp16 reference has passed
   the thresholds in `docs/METRICS.md`. A faster-but-different model is a failed experiment.
3. **One variable per experiment.** Never change bit-width, group size, and fusion strategy in
   the same run. A combined run is allowed only *after* the individual effects are known, and
   must be labelled `combined`.
4. **Never use spec bandwidth for the roofline.** Calibrate measured-achievable bandwidth on the
   actual card first (`tools/probe_machine.py`); every theoretical floor must cite that number.
5. **Report the applicability range of the card you measured on.** Every conclusion names the
   GPU, the **measured** bandwidth, dtype and batch. sm_120 (5090) has `cp.async` / TMA / FP8 /
   MXFP4 block-scale but **no `wgmma`**; sm_70 (V100) has none of those. Nothing transfers between
   these without re-measurement.
6. **VRAM is reported split** (weights / KV / activations / workspace), via allocator peaks, not
   `nvidia-smi` totals.
7. Every measurement is recorded in the `EXPERIMENTS.md` shape (copy
   `experiments/TEMPLATE.md`) under `results/<EXP-ID>/`. A result not written in that format
   does not exist for the purposes of this project.

## Environment

Phase 0 creates it. Rules are **per machine** — do not apply one box's constraints to another
(details in `ENVIRONMENT.md`, `docs/SPECS.md`, `docs/SERVER_RUNBOOK.md`):

**Rented GPU box (primary — this one is ours):**
- Python 3.12 via `uv`; `torch==2.14.0` with the CUDA index chosen from the **actual driver**
  (`nvidia-smi`), never from the image's marketing name. `configs/constraints.txt` freezes the
  set after the first verified run.
- Installing toolkits is allowed here. Prefer a **system image** (`Ubuntu-nvidia`) over a
  pre-baked PyTorch/vLLM image so we do not inherit someone else's pins.
- Pay-as-you-go billing only — never spot/preemptible (a long experiment dies with it).

**School V100 box (shared — hands off the machine layer):**
- `fp16` only; no bf16 tensor cores.
- **Never** install or upgrade the CUDA toolkit / driver / system Python — shared with other
  students. Release memory when done.

**Mac (docs + CPU logic tests):** may run CPU `torch`; see the hard limit in "What this repo is".

## Daily commands

```bash
uv venv -p 3.12 .venv                 # interpreter is pinned by .python-version
uv pip install -e ".[dev]"            # editable install of `lemoq`
uv run pytest -m "not gpu"            # correctness gates first (also runs on the Mac)
uv run ruff check . && uv run black . # lint / format (line-length 100)
uv run python -m tools.probe_machine --json results/machine_probe.json   # device facts to disk
```

**Never call bare `python3`.** On the Mac it resolves to Xcode's 3.9.6, which violates
`requires-python>=3.10`. Always go through `uv run` / `.venv/bin/python`.

## Models

Single source of truth: `configs/models.yaml`. Roles:

- **dev**: `Qwen/Qwen2.5-0.5B` / `1.5B` (dense) — kernel and quantization debugging;
- **primary (MoE)**: `allenai/OLMoE-1B-7B-0924` — the MoE main line. The `-0924` suffix is
  **required**; `allenai/OLMoE-1B-7B` does not exist (verified 2026-09). Use the *base* checkpoint,
  not `-Instruct`: the primary gates are KL/PPL on a base reference.
- **dense comparison**: `Qwen/Qwen3-4B` / `8B`;
- **stretch (48 GB box)**: `Qwen/Qwen3-30B-A3B` — cannot fit on 24 GB, and not on 32 GB by default.

Any run using a different checkpoint must say so explicitly and justify it.

## Remotes and sync

`origin` is GitHub (`tmchao7/LeanMoE-Quant`). This is **not** a fork of any upstream — there is
no upstream remote and no upstream sync story. If a remote is not yet configured, do not invent
one; the repo works locally until the user adds it.

Work happens from **three writers** — the rented GPU box (which may itself run an autonomous
agent), the school V100 box, and the Mac — so histories diverge:

- **merge, never rebase, never force-push**;
- an agent running on the GPU box works on a **feature branch** and pushes it; the Mac only
  `git pull --no-rebase`s;
- every agent iteration must leave: `results/<EXP-ID>/result.json`, a matching
  `experiments/<EXP-ID>.md` filled from `experiments/TEMPLATE.md`, and a commit;
- **an agent may not relax a correctness gate.** If KL/top-1 fails, the record says `FAIL`.

## Commit conventions

Small, scoped, phase-prefixed commits: `chore:`, `docs:`, `env:`, `baseline:`, `quant:`,
`kernel:`, `bench:`, `analysis:`. Reference the experiment id (`EXP-p1-01`) in the body when a
commit implements a measured change. One logical change per commit; do not mix docs and
behaviour. Never commit `results/`, model weights, `__pycache__`, or credentials.

## Reference docs

| File | For |
| --- | --- |
| `PLAN.md` | the authoritative plan: thesis, hardware, phases, experiment matrix, risks |
| `ENVIRONMENT.md` | machine access, environment bootstrap, model acquisition, first-run checklist |
| `docs/SERVER_RUNBOOK.md` | the exact command sequence to run the moment a GPU box exists |
| `EXPERIMENTS.md` | mandatory experiment record format and metric definitions |
| `docs/SPECS.md` | confirmed hardware specs, the cross-generation feature matrix, GPU selection |
| `docs/MODELS.md` | model candidates, verified OLMoE architecture, per-token byte budget |
| `docs/METRICS.md` | metric definitions: KL, PPL, `[W]`/`[T]` byte budgets, attainment, VRAM split |
| `configs/experiments.yaml` | the declarative experiment matrix the harness expands |
