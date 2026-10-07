# AGENTS.md

Compact guide for agent sessions working in LeanMoE-Quant. Read this before editing.

## 0. Sixty-second state (2026-10-07)

Phases 0, 1 (accuracy gate), 2 (kernels) and 4b (E7) are **executed**. Two headline results are
settled — do not re-derive them and do not re-run their cells:

- **E2 (`EXP-p2-01`) — `refuted`.** Fusing the dequant into the GEMV is *not* the 1.3–2× the
  matrix predicted: in the eager cells the fused arm is **5% slower** (`host_share` 71.7%,
  ~19k kernels/step). Isolated the fused kernel is 1.21× cheaper per call — Python/launch eats it.
- **E7 (`EXP-p4b-01`) — judged.** The truncation is real and **only the fused arm cashes it in**:
  the same `w8a16` config goes `speedup_eager` **0.76 → speedup_cudagraph 1.11** (ctx=1024;
  0.79 → 1.05 at 4096) while `host_share` drops 77.5% → 1.2%. The ceiling is measured: the weight
  stream is only 42% (1024) / 23% (4096) of a graph step ⇒ even a perfect weight path ≤ ~1.74× / ~1.30×.

- **`EXP-p4c-01` — supported (the other big non-weight block was the ATTENTION IMPLEMENTATION).**
  Swapping HF sdpa for this repo's split-KV core *inside the graph* cuts the step 1.36–2.44× and
  **raises the quantized speedup**: 1.109 → **1.153** (1024) / 1.053 → **1.131** (4096); best cell
  `w8a16 × kernel` = **1.57× / 2.57×** vs fp16 × sdpa. Mechanism: the removed attention cost is
  bit-width-independent (both arms save ~4.1 ms @4096) — a constant that helps the smaller total.

- **E4 (`EXP-p4-01`) — supported.** Routing/packing is the largest bucket in the hf/gather baseline
  (4.478 ms = **44.2%** of GPU busy) and **quantization-invariant** (4.478 / 4.512 / 4.526 / 4.555 ms
  for fp16 → w4a16) — the cap the plan predicted. The grouped path already removed it; see P2b.

Read, in this order: **`configs/experiments.yaml`** (the only source of hypotheses and cells),
the **record** of the experiment you are about to touch (`experiments/<EXP-ID>.md`), then
**`docs/HANDOFF_P2_FUSION.md`** (the current handoff).

Two rules come before everything else: **do not redo what §3 lists as done**, and **ask the user**
before changing a matrix axis, a cell id, or overwriting a recorded `result.json` (§6).

## 1. What this repo is

Research project on **byte-budget-driven quantization and kernel optimization for MoE decode at
batch=1**. Delivered so far: the fp16 and quantized decode paths, the CUDA-Graph execution axis,
and three Triton kernel families (grouped MoE, dense fused dequant, split-KV attention) — the
inventory with its authority is in §3. The thesis, the frozen hypotheses and the phase plan live
in `PLAN.md`; **execution status lives here and in the records, not in `PLAN.md`.**

- **Primary hardware: rented RTX 5090 32 GB (sm_120)**, single card. Single-card project only —
  never design something that needs more than 1 GPU.
- **Secondary (free): school V100 32 GB (sm_70)**, an optional second bandwidth data point. It
  lacks `cp.async`, FP8, FP4 and bf16 tensor cores, so it cannot host the modern kernel arms.
- **The Mac may run CPU `torch` for functional/unit tests**, but **no number produced on the Mac
  may ever enter `results/` or any conclusion** — the Mac has no NVIDIA GPU.
- Package name: `lemoq`; source under `src/lemoq/`.

## 2. Project hard rules (an agent will get these wrong if it guesses)

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
8. **A cell that fails the correctness gate is evidence, never a performance result.** Its
   `latency` section is `null`; the measured values live in `_measurement.gate_failed_evidence`.
   `benchmarks/schema.py` enforces this (and `tests/test_schema.py` guards the guard) — do not
   weaken it, and do not hand-edit a payload to get numbers past it.
9. **Never add, rename or re-axis the experiment matrix on your own.** Cell ids are built from the
   declared axes; changing one invalidates recorded artifacts and the harness skeletons. If a new
   axis is genuinely needed (`attn`, `moe`, `batch` are the known candidates), **ask first**.
   The same applies to overwriting a `result.json` that already exists: archive the old one under
   `results/<EXP-ID>-run/v1_archive/`, re-run with `--force`, and say `v1`/`v2` in the record.

## 3. Where the project stands (authority: the records + artifacts)

| Area | State | Authority |
| --- | --- | --- |
| Gate-0 probe, fp16 baseline, static MoE + cudagraph | done | `experiments/EXP-p0-02-baseline-fp16.md`, `experiments/EXP-p0-04-cudagraph-static-moe.md` |
| cudagraph feasibility with the HF MoE path | `blocked` **by design** (hf + dynamic cache cannot capture) | `experiments/EXP-p0-03-baseline-cudagraph.md` |
| Accuracy gate — the only source of `GATE_CITATIONS` | done | `experiments/EXP-p1-04-prompt-set-gate.md` |
| Precision side: error attribution, mixed bits, W8/W4 sweep | done (the old `EXP-p1-01` protocol was superseded by `p1-04`) | `experiments/EXP-p1-02-quant-error-attribution.md`, `experiments/EXP-p1-03-quant-mixed-bits.md`, `experiments/EXP-p1-01-w8-w4-accuracy.md` |
| Attention path: eager, kernel, in-graph | done | `experiments/EXP-p2-03-decode-attention.md`, `experiments/EXP-p2-04-eager-attention.md`, `experiments/EXP-p2-05-attn-kernel.md` |
| Grouped MoE GEMV (fp16) | done | `experiments/EXP-p2-02-moe-grouped-gemm.md` |
| E2 dequant fusion, 6 cells, 8-bit **and** 4-bit | done — **`refuted`** | `experiments/EXP-p2-01-dequant-fusion.md` |
| P3 wiring + E7, 16/16 cells | done — **judged** | `experiments/EXP-p4b-01-truncation.md` |
| In-graph attention × quantization, 8 cells | done — **supported** (1.57× / 2.57× combined) | `experiments/EXP-p4c-01-in-graph-attention.md` |
| E4 routing overhead | done — **supported**: 44.2% of GPU busy, and **quantization-invariant** (4.478/4.512/4.526/4.555 ms for fp16/w8a16/w8a8/w4a16) | `experiments/EXP-p4-01-moe-routing-overhead.md` |
| E5 expert sensitivity, E6 KV INT8, E8 cross-bandwidth, FP8/MXFP4, dense curves, batch sweep | **not run** | `configs/experiments.yaml`, `PLAN.md` §7 |
| ncu evidence (a Phase 2 deliverable) | **not done** | `PLAN.md` §5 |

**Do not redo** (recorded): P0 / P1 gate / P2 / P3 wiring; the dense fused dequant kernel; the
grouped quant expert kernel (8- and 4-bit); the split-KV attention kernel; the E2, E7 and p4c-01
verdicts; p4b's 16, p2-01's 6 and p4c-01's 8 cells. A rerun needs a justification in the record
plus archiving (§2 rule 9).

**Known artifact-id wrinkle**: older precision experiments stored their artifacts under the *base*
id (e.g. `results/EXP-p1-04-prompt-set-gate/probe_prompt_set*.json`) while the matrix expands them
to `...__ctx=1024` skeletons that still read `planned`. A `planned` skeleton is **not** evidence
that the science was not done — read the record first.

## 4. Queue (ordered; hypotheses stay in `configs/experiments.yaml`)

| # | Task | Why now | Acceptance | Needs user? |
| --- | --- | --- | --- | --- |
| P0 | ✅ **done 2026-10-07** (see §5): README status, `PLAN.md` pointer, p4b `Status`, p1-02/03/04 artifact notes, `constraints.txt` freeze | those files contradicted the records | four gates green, both machines on one commit | no |
| P1 | ✅ **done 2026-10-07**: E4 → routing/pack = **44.2%** of GPU busy and constant within 1.7% across fp16→w4a16 | the plan's `minimal_credible` scope is now **6/6** | `experiments/EXP-p4-01-moe-routing-overhead.md` | no |
| P2 | ✅ **done 2026-10-07**: in-graph attention → speedup 1.11 → **1.15** (1024) / 1.05 → **1.13** (4096); best cell **1.57× / 2.57×** vs fp16 × sdpa | the user approved a new id (`EXP-p4c-01`), see §3 | `experiments/EXP-p4c-01-in-graph-attention.md` | no |
| P2b | **The next bottleneck is the elementwise/copy tail** (~35% of the step, ~330 tiny kernels); runner-up: the 64 `_dequant_gemm_kernel` projection launches (~5.4 µs of launch overhead each) | §5.4 of `EXP-p4c-01` measured it after the attention swap; it is the largest non-weight bucket left | a record with the tail classified by source + a fusion decision | **YES — no matrix entry exists**; ask before adding an `EXP-` id (or an axis) |
| P2c | Widen the routing classifier (torch `topk`/`sort`/`index_select`/`gather`/`scatter`) so the **grouped** path's routing cost can be quoted | `EXP-p4-01` §5.3: the classifier's 0.0% on the grouped path is a **blind spot**, not a measurement | new rules + tests; the number must be produced in its own round | **YES — it changes the meaning of the `routing_share_pct` field** |
| P3 | Breadth: E5 (`EXP-p4-02`), E6 (`EXP-p5-01`), E8 (`EXP-p8-01`, needs the V100), E1's group axis under the corrected prompt set, dense curves, batch sweep (E3 — today it has **no** experiment), FP8/MXFP4 (`EXP-p3-01/02`), repairing the W4A16 gate | each closes a declared hypothesis or lever; several need a matrix entry that does not exist | per matrix entry; E8 only after the same protocol passes on the V100 | partial: new matrix entries need approval |
| P4 | Publication: ncu evidence, E7 charts, final README/PLAN pass | closes Phase 2's last deliverable and Phase 7 | ncu metrics per `lemoq.kernels.NCU_METRICS`; charts regenerated from `result.json` | no |

## 5. Wrap-up (docs/config only — **done 2026-10-07**)

README status, `PLAN.md` pointer, p4b `Status`, the p1-02/03/04 artifact-id notes and the
`constraints.txt` freeze all landed; details in git log. Any *new* wrap-up item goes in §4.

## 6. Guardrails

### Ask the user first

- adding / renaming a matrix axis or a cell id, or changing an existing cell's meaning;
- overwriting an existing `result.json`, or re-running a cell that is already `done`;
- installing or upgrading anything on the server (the documented `export PATH=…` is not an install);
- committing `results/`, model weights, `__pycache__` or credentials.

### Before you claim anything

| Claim | Prerequisite |
| --- | --- |
| a performance number | cited gate PASS (`GATE_CITATIONS`, never relaxed) + canary + clean occupancy + `result.json` + record + commit |
| "this kernel is correct" | K1 parity (`PARITY_GATE` 1e-3, not relaxed) + `pytest -m gpu` actually run |
| "this cell ran" | `result.json` with `status=done` + a results section in the record — a `planned` skeleton is not a run |
| "quantization pays / fusion helps" | one-round adjacent or interleaved A/B with MAD and per-pair wins; a single arm run in isolation is not accepted |
| "I changed X safely" | the tests that pin X (see §9) + a statement of whether recorded cell ids move |

### Shared card protocol (this box is shared)

- check `nvidia-smi --query-compute-apps=…` before timing; a foreign process is **recorded, never
  killed** — wait for it to exit and re-run the round;
- **canary**: same cell first and last; >5% drift ⇒ the whole round is `VOID` (keep the log, publish nothing);
- **cudagraph cells need their own canary**: an `eager` canary is *not* sensitive to a co-tenant
  (measured: it drifted 1.16% while three graph cells in the same round were ruined — one OOM, one
  3× outlier). Add a same-cell graph canary plus a payload self-check: a graph cell with
  `host_share_pct > 5` is contaminated.

### Commit and verification gate

Run all four before every commit, on the machine you are editing:

```bash
ruff check . && black --check . && pytest -m "not gpu" -q && python -m tools.check_docs
```

The bar is **0 failed** (a few skips are normal). Then push, `git pull --no-rebase` on the other
machine, re-run there, and rsync `results/` back to the Mac.

### Never relax or bypass

- the thresholds in `docs/METRICS.md` §1, `PARITY_GATE` (1e-3), `benchmarks/schema.py`, or the
  existing `GATE_CITATIONS` rows — new rows only from a **new measurement with its own artifact**;
- the runner's five load-time refusals (all before any model load; pinned by
  `tests/test_runner_cells.py`): fused on a non-quantized cell; fused without triton/CUDA; a
  quantized cell with no gate citation; quantized × `--moe gather`; grouped quant with an odd
  `--group-size`. If a cell "cannot run", fix the cause — do not delete the refusal.

## 7. Machines and exact commands

**Mac (docs + CPU logic tests).** The working interpreter is the repo-local `.venv-mac`
(Python 3.12, torch 2.14.0 CPU); a `.venv` may also exist — do not rebuild either one.

```bash
.venv-mac/bin/python -m pytest -m "not gpu" -q     # ~all green
.venv-mac/bin/ruff check . && .venv-mac/bin/black --check .
```

Never call bare `python3` (Xcode 3.9.6, violates `requires-python>=3.10`). Nothing measured here
is reportable.

**Rented RTX 5090 (primary; ours).** `ssh -p 43086 root@connect.weste.seetacloud.com`
(alias `5090b`). Verified set: Python 3.12.3, `torch 2.8.0+cu128`, `triton 3.4.0`,
`transformers 4.57.6`, driver 595.71.05, sm_120 with 170 SMs, **measured** read bandwidth
1561.9 GB/s.

```bash
export PATH=/root/miniconda3/bin:$PATH          # python AND uv live here (uv 0.12.17)
/root/miniconda3/bin/python -m pytest -m "not gpu" -q
# always pass the local snapshot, or the runner tries huggingface.co and times out:
/root/miniconda3/bin/python -m benchmarks.runner --exp <EXP-ID> \
    --model /root/autodl-tmp/ms/models/allenai--OLMoE-1B-7B-0924/snapshots/master --force
# prompt cache: /root/autodl-tmp/lemoq-data/wikitext2_test_8k.pt
# HF_HOME=/root/autodl-tmp/hf   HF_ENDPOINT=https://hf-mirror.com
```

Installing packages is allowed here but never mid-experiment; record versions in the result.

**School V100 (shared — hands off the machine layer).** `fp16` only; no bf16 tensor cores; the
modern kernel arms cannot run there. **Never** install or upgrade the toolkit / driver / system
Python — it is shared with other students. Release memory when done.

## 8. Remotes, commits, verification

`origin` is GitHub (`tmchao7/LeanMoE-Quant`) — not a fork, no upstream. Three writers (the GPU box,
the V100 box, the Mac) diverge, so:

- **merge, never rebase, never force-push**;
- an agent on the GPU box may work on a feature branch and push it; the Mac only
  `git pull --no-rebase`s;
- every agent iteration leaves: `results/<EXP-ID>/result.json`, a matching
  `experiments/<EXP-ID>.md`, and a commit;
- `results/` is gitignored and machine-local: the durable record is the markdown, and `results/`
  must be rsynced back to the Mac before the round is called finished;
- commit prefixes: `chore:`, `docs:`, `env:`, `baseline:`, `quant:`, `kernel:`, `bench:`,
  `analysis:`; reference the experiment id; one logical change per commit, docs and behaviour in
  separate commits.

## 9. Reference docs

| File | For |
| --- | --- |
| `PLAN.md` | the frozen plan: thesis, hypotheses, phases, budget, risks (not a status page) |
| `configs/experiments.yaml` | the declarative matrix: the only source of hypotheses and cell ids |
| `experiments/<EXP-ID>.md` | the authoritative record of each experiment (status, results, applicability) |
| `docs/HANDOFF_P2_FUSION.md` | the current handoff (finished + next); `docs/HANDOFF*.md` are historical |
| `EXPERIMENTS.md` | mandatory record format and metric definitions |
| `docs/METRICS.md` | metric definitions and the correctness thresholds |
| `docs/MODELS.md` | model candidates and the verified OLMoE architecture (incl. the ctx≤4096 limit) |
| `docs/SPECS.md`, `docs/SERVER_RUNBOOK.md`, `ENVIRONMENT.md` | hardware facts, runbook, machine access |
| `benchmarks/runner.py` | the harness that produces cells; its refusals and payloads are the contract |
| `benchmarks/schema.py` | the machine-checked shape of `result.json` (hard rule 8 lives here) |
| `tools/check_docs.py` | dead-link and table-column checker (also run by `tests/test_links.py`) |
