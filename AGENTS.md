# AGENTS.md

Compact guide for agent sessions working in LeanMoE-Quant. Read this before editing.

## What this repo is

Research project on **byte-budget-driven quantization and kernel optimization for MoE decode
under edge constraints**. Status: **Phase 0 (environment + baseline)** — no optimization is
implemented, no benchmark has been run. See `PLAN.md`.

- **Main hardware: V100 32 GB (sm_70)**, a school-provided single-GPU server. Single-card
  project only — never design something that needs more than 1 GPU.
- The Mac checkout is for docs/repo work only. **It must not produce any performance number**
  (no NVIDIA GPU, no `torch`).
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
5. **Report the V100 applicability range.** No BF16, no `cp.async`/TMA. Any conclusion must say
   so explicitly; it does not transfer to Ampere+ without re-measurement.
6. **VRAM is reported split** (weights / KV / activations / workspace), via allocator peaks, not
   `nvidia-smi` totals.
7. Every measurement is recorded in the `EXPERIMENTS.md` shape (copy
   `experiments/TEMPLATE.md`) under `results/<EXP-ID>/`. A result not written in that format
   does not exist for the purposes of this project.

## Environment

There is no prepared environment yet; Phase 0 creates it. The target runtime is the Linux V100
server. Key constraints (details in `ENVIRONMENT.md` and `docs/SPECS.md`):

- Python 3.10–3.12; PyTorch with **CUDA 12.x**, sm_70 support required;
- **fp16 only** — do not request `bfloat16` (V100 has no bf16 tensor cores; it silently degrades);
- no `cp.async`, no TMA, no `wgmma`; shared memory ≤ ~96 KB/SM (confirm with a compile probe);
- never install or upgrade the CUDA toolkit / driver / system Python — those are machine-level,
  and this machine is shared with other students.

## Daily commands (once Phase 0 exists)

```bash
python -m pip install -e .            # editable install of `lemoq`
pytest                                # correctness gates first
ruff check . && black .               # lint / format (line-length 100)
python -m tools.probe_machine --json results/machine_probe.json   # device facts to disk
```

## Models

Single source of truth: `configs/models.yaml`. Roles:

- **dev**: `Qwen/Qwen2.5-0.5B` / `1.5B` (dense) — kernel and quantization debugging;
- **primary (MoE)**: `allenai/OLMoE-1B-7B` — the MoE main line;
- **dense comparison**: `Qwen/Qwen3-4B` / `8B`;
- **stretch (rented 48 GB only)**: `Qwen/Qwen3-30B-A3B` — cannot fit on the 32 GB V100.

Any run using a different checkpoint must say so explicitly and justify it.

## Remotes and sync

`origin` is GitHub (`tmchao7/LeanMoE-Quant`). This is **not** a fork of any upstream — there is
no upstream remote and no upstream sync story. If a remote is not yet configured, do not invent
one; the repo works locally until the user adds it.

Work happens from two machines (the V100 server and the Mac), so histories diverge:
**merge, never rebase, never force-push.**

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
| `EXPERIMENTS.md` | mandatory experiment record format and metric definitions |
| `docs/SPECS.md` | confirmed hardware specs and the sm_70 hard constraints |
| `docs/MODELS.md` | model candidates and why the main line is OLMoE-1B-7B |
| `docs/METRICS.md` | metric definitions: KL, PPL, bandwidth attainment, VRAM split |
