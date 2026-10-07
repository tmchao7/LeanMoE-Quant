# AGENTS.md

Compact guide for agent sessions working in LeanMoE-Quant. Read this before editing.

## 0. Sixty-second state (2026-10-07)

Phases 0, 1 (accuracy gate), 2 (kernels) and 4b (E7) are **executed**. Two headline results are
settled — do not re-derive them and do not re-run their cells:

- **E2 (`EXP-p2-01`) — `refuted`.** Fusing the dequant into the GEMV is *not* the 1.3–2× the
  matrix predicted: in the eager cells the fused arm is **5% slower** (`host_share` 71.7%,
  ~19k kernels/step). Isolated the fused kernel is 1.21× cheaper per call — Python/launch eats it.
- **E7 (`EXP-p4b-01`) — judged.** The truncation is real and **only the fused arm cashes it in**:
  the same `w8a16` config goes `speedup_eager` **~0.78 → speedup_cudagraph 1.170** (ctx=1024;
  → 1.084 at 4096) while `host_share` drops 77.5% → 1.6%; the separate arm does not move
  (0.7674 / 0.8530) even after the kernel fix. Ceiling is measured: the weight stream is only
  42% (1024) / 23% (4096) of a graph step ⇒ even a perfect weight path ≤ ~1.74× / ~1.30×.

- **`EXP-p4c-01` — supported (the other big non-weight block was the ATTENTION IMPLEMENTATION).**
  Swapping HF sdpa for this repo's split-KV core *inside the graph* cuts the step 1.36–2.44×;
  with the projection-kernel fix below, the quantized speedup is **1.169 (sdpa) / 1.250 (kernel)**
  @1024 and 1.084 / 1.214 @4096, and the best cell `w8a16 × kernel` = **1.70× / 2.77×** vs
  fp16 × sdpa. Mechanism: the removed attention cost is bit-width-independent (@4096 both arms
  save ~4.1 ms) — a constant that helps the smaller (quantized) total relatively more.
- **The projection kernel was 3× off its byte floor — and it is fixed.** In-graph config tuning
  (new tool `tools/bench_dequant_gemm_graph.py`) showed the old ranking was launcher-bound and that
  the default config is optimal, so the 5.4 µs/call gap was structural: the scale was gathered
  per element while one value covers 128 of them. Loading it per group + broadcasting gives
  **8.14 → 5.15 µs/call (523 → 826 GB/s)**; the model's projection bucket fell 0.5245 → 0.3254 ms.
  ★ The pre-fix cells were then **re-measured** (P2d, 2026-10-07): E7's fused `speedup_cudagraph`
  1.1071 → **1.1703** (1024) / 1.0512 → **1.0839** (4096); in the **eager** arm the same fix is
  invisible (A/B ratio 1.0093 → 1.0091) — the improvement only cashes in inside the graph.
  v1 artifacts are archived under `results/<EXP>-run/v1_archive/` and labelled in the records.

- **The elementwise/copy tail is now classified (`EXP-p4c-02`).** The "~330 kernels / 0.90 ms" is
  **13 named families × 16–83 calls/step** (copies 0.17 ms; RMSNorm mean+rsqrt 0.15–0.25; KV-cache
  write+cat 0.11; residual add 0.05; routing topK 0.05) — no single kernel >20%, so there is no
  one-shot fix. ★ Its **module-level** attribution is **VOID by its own pre-registered K1**
  (eager/graph top-8 overlap 5/8) — the record says so instead of quoting it.

- **★ A custom fused RMSNorm wins 23.6% — the biggest single win so far.** The elementwise/copy tail's
  dominant chain was HF's `OlmoeRMSNorm` (7 kernels/norm × 65 norms/step, incl. 130 dtype casts).
  `src/lemoq/kernels/rmsnorm.py` does it in **one** launch with HF's rounding point (fp32 mean →
  round to fp16 → multiply by fp16 weight) and is **1.2358×** on the whole step (2.4143 → 1.9537 ms,
  wins 15/15), kernel/step **994 → 539**, K1 vs HF's formula **5.261e-04** (torch's `F.rms_norm`
  was both slower *and* numerically farther: 1.118e-03). ★ The gate re-measurement (P2i) is
  **half done**: under the fused norm the ctx=1024 readings are w8a16 0.001361 / w8a8 0.033304 /
  w4a16 0.093560 vs 0.001273 / 0.032753 / 0.093323 with HF's norm ⇒ **no verdict changes**; the
  probe's own aggregate section still has to be written and the citations re-pinned (§4 P2i).

- **A fix was tried and REFUTED (kept on purpose).** Replacing HF's `OlmoeRMSNorm` chain (7 kernels ×
  65 norms/step, incl. 130 dtype casts) with torch's fused `F.rms_norm` is **5% slower**
  (2.4106 → 2.5358 ms, HF wins 15/15) *although numerically identical* (KL 7.2e-4, top-1 1.0):
  it keeps the same `MeanOps` reduce and adds an fp32 row multiply (`BinaryFunctor<float>` ×65 =
  0.189 ms) where HF multiplies in fp16. ⇒ A custom Triton norm is the only remaining route (P2h).

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
| In-graph attention × quantization, 8 cells | done — **supported**; re-measured after the kernel fix (**1.70× / 2.77×** combined, §5.6) | `experiments/EXP-p4c-01-in-graph-attention.md` |
| E4 routing overhead | done — **supported**: 44.2% of GPU busy, and **quantization-invariant** (4.478/4.512/4.526/4.555 ms for fp16/w8a16/w8a8/w4a16) | `experiments/EXP-p4-01-moe-routing-overhead.md` |
| The elementwise/copy tail (0.90 ms, "~330 kernels") | done — classified: it is **~13 named families × 16–83 calls/step**, not a flat tail; module-level attribution VOID (its own K1 failed) | `experiments/EXP-p4c-02-elementwise-tail.md` |
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
| P2 | ✅ **done 2026-10-07** (re-measured after the kernel fix): in-graph attention + projection-kernel fix → speedup **1.169 / 1.250** (1024) and 1.084 / 1.214 (4096); best cell **1.70× / 2.77×** vs fp16 × sdpa | the user approved a new id (`EXP-p4c-01`), see §3 | `experiments/EXP-p4c-01-in-graph-attention.md` | no |
| P2b | ✅ **classified 2026-10-07** (`EXP-p4c-02`): the 0.90 ms tail is **13 named families × 16–83 calls/step** — copies 0.17 (149 calls), RMSNorm mean+rsqrt 0.15–0.25, KV-cache write+cat 0.11, residual add 0.05, routing topK 0.05; **no single kernel >20%** | it was the largest non-weight bucket left | `experiments/EXP-p4c-02-elementwise-tail.md` | no |
| P2f | ✅ **identified 2026-10-07** (`EXP-p4c-02` §5.5): the copy family is **`aten::_to_copy` ×150/step (0.185 ms)** — dtype casts from HF's RMSNorm (`to(fp32)` + `to(fp16)` per norm × 65 norms) plus the norm's `pow/mean/rsqrt/mul` chain ⇒ P2f and the norm question are **the same fix** | — | — | no |
| P2i | 🔄 **IN PROGRESS — resume here 2026-10-08** (user approved the full pass). Done: plumbing (`--rmsnorm {fused,hf}`, default fused, recorded in `config.rmsnorm` + `config.rms_norms_replaced`, 6th load-time refusal, pinned by tests); the ctx=1024 gate **probe was re-run under the fused norm** and stopped at 26/51 arms when the server was shut down. **The three citation arms are complete** (all 6 windows; `rmsnorm: fused`, `norms_replaced: 65`): preliminary reading over w0–w4 — w8a16 **0.001361** (was 0.001273, +6.9%), w8a8 0.033304 (was 0.032753), w4a16 0.093560 (was 0.093323) ⇒ **no verdict changes**, so adopting is safe numerically. ⚠️ Those three numbers were computed **by hand from the salvaged windows** — the tool's own `aggregate` section was never written (run killed), so they are **not** a citation yet. | next steps | (1) add `--aggregate-only <artifact>` to `tools/quant_probe.py` (reuse its own aggregation code path) **or** re-run the probe; the salvaged artifact is `results/EXP-p1-04-prompt-set-gate/probe_prompt_set_rmsnorm_fused_ctx1024.json` (also on the Mac). (2) Write the aggregate, archive the HF-norm artifacts (`probe_prompt_set.json`, `..._ctx4096.json`) as `..._v1_hf_norm.json`, copy the new ones to the canonical names. (3) Update `GATE_CITATIONS` kl/top-1 **verbatim** from the new artifact + the pinned assertion in `tests/test_runner_cells.py` (~line 507) + the ctx=4096 probe (second run). (4) Re-run `EXP-p4c-01`'s 8 cells with `--force` (default is already fused): the pre-P2i artifacts are archived at `results/EXP-p4c-01-run/pre_p2i_hf_norm/` (synced to the Mac) — cells stay the same ids, label them **v3** in the record. (5) Add the v3 table to `experiments/EXP-p4c-01-in-graph-attention.md` §5.7. | **YES — already given** |
| P2d | ✅ **done 2026-10-07** (9 + 2 cells, canary −0.006%, occupancy clean): E7 fused `speedup_cudagraph` **1.1071 → 1.1703** (1024) / **1.0512 → 1.0839** (4096); E2's eager ratios unchanged (1.0442 / 1.0523 / 0.9681) — the fix is graph-only | the kernel fix made the recorded absolute ms stale | v1 archived under `results/<EXP>-run/v1_archive/`, v1/v2 labelled in both records | no |
| P2e | Optional: fuse q/k/v into **one** launch (3→1) — the projection kernel is now 5.15 µs/call vs a 2.72 µs byte floor, so ~2.4 µs/call of launch/ramp is left | follows P2b's elementwise verdict; needs a shape decision (the three projections must share `in_features`) | isolated in-graph A/B (the P2d-style tool) before any cell run | no, if it stays inside `EXP-p4c-01`'s declared config; ask if it needs a new id |
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
