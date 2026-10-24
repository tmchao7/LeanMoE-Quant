# AGENTS.md

Compact guide for agent sessions working in LeanMoE-Quant. Read this before editing.

## 0. Sixty-second state (2026-10-08)

Phases 0, 1 (accuracy gate), 2 (kernels) and 4b (E7) are **executed**. Two headline results are
settled — do not re-derive them and do not re-run their cells:

- **E2 (`EXP-p2-01`) — `refuted`.** Fusing the dequant into the GEMV is *not* the 1.3–2× the
  matrix predicted: in the eager cells the fused arm is **5% slower** (`host_share` 71.7%,
  ~19k kernels/step). Isolated the fused kernel is 1.21× cheaper per call — Python/launch eats it.
- **E7 (`EXP-p4b-01`) — judged.** The truncation is real and **only the fused arm cashes it in**.
  ★ Refreshed 2026-10-08 (P2j, fused RMSNorm): the graph cells now read `speedup_cudagraph`
  **1.1979** (1024) / **1.0898** (4096) — ★ **2026-10-24 F4 adoption** (F1+F2+F4 defaults on,
  23/23 cells, canary 0.336%): adopted p50s fp16 3.5446/6.7805, w8a16 2.9589/6.2219 ms
  (2026-10-24 values 1.1974/1.0912 stay in that round's record). Canary drift 0.34%; the **eager**
  arm's 8 cells are **pre-P2i** — their refresh was VOID because a co-tenant's CPU load inflated
  `host_p50` (evidence: the *old* implementation re-run now shows
  `host_p50` 35.2 ms vs 1.5 ms archived, while `kernels_per_call` and `gpu_busy` match exactly).
  Host_share drops 77.5% → 1.6%, and the weight stream is only 42% (1024) / 23% (4096) of a graph
  step ⇒ even a perfect weight path ≤ ~1.74× / ~1.29×.
  ★★ **`w8a8@4096` is DE-PUBLISHED (R3 adjudication, 2026-10-24).** The P2i flip FAIL → PASS was
  3/165 positions (borderline); the pre-registered 10-window re-measurement
  (`results/EXP-p1-04-r3-10win-ctx4096.json`) shows KL mean **0.0262 > 0.01** (heavy tail: w7
  0.2247, sd 0.066) — the 5-window PASS is **not robust**. The citation row now cites the
  10-window artifact (derived verdict **FAIL**), so those cells run as evidence only: the
  published 8.8232 ms cudagraph is **withdrawn**. See `EXP-p1-04` v3 and `EXP-p4b-01` §R3.

- **`EXP-p4c-01` — supported (the other big non-weight block was the ATTENTION IMPLEMENTATION).**
  Swapping HF sdpa for this repo's split-KV core *inside the graph* cuts the step 1.36–2.44×.
  **Canonical numbers are now v5** (2026-10-24 F4 adoption, F1+F2+F4 on): quantized
  **1.1948 (sdpa) / 1.4528 (kernel)** @1024 and **1.0898 / 1.3801** @4096, best cell
  `w8a16 × kernel` = **2.36× / 3.85×** vs fp16 × sdpa (v4 was 1.1948/1.4392, 2.27×/3.73×).
  Mechanism: the removed cost (attention core, then the norm chain) is bit-width-independent —
  a constant that helps the smaller (quantized) total relatively more.
  Canary in the v3 round: −0.067%; one cell was contaminated (`host_share` 51%) and re-run
  clean (2.5079/2.5120 ms) — see §5.7.
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

- **★ The fused RMSNorm is now the DEFAULT decode path, and the gate was re-measured under it
  (P2i, 2026-10-08).** `--rmsnorm {fused,hf}` (default `fused`), the implementation is recorded in
  `config.rmsnorm`, and there is a 6th load-time refusal for machines without triton/CUDA.
  Gate under the fused norm (5 windows): w8a16 **0.001361/0.000842** @1024/@4096 (both still PASS),
  w4a16 FAIL both, **and `w8a8@4096` FLIPPED from FAIL to PASS** (top-1 0.9455 → 0.9636). ★ That flip
  is **3/165 positions** — the two measurements straddle the threshold, and the norm change also
  moves the fp16 reference's greedy tokens (w2/w4) ⇒ the honest reading is "**not resolvable at
  this budget**", not "it passes now". τ was not touched. ★★ **R3 (2026-10-24) then de-published
  it**: the 10-window re-measurement (KL mean 0.0262 > 0.01, heavy tail) showed the 5-window PASS
  is not robust, and the user approved withdrawing the published latency — the citation row now
  cites the 10-window artifact (verdict FAIL, evidence-only, w4a16 semantics); see `EXP-p1-04` v3.

- **A fix was tried and REFUTED (kept on purpose).** Replacing HF's `OlmoeRMSNorm` chain (7 kernels ×
  65 norms/step, incl. 130 dtype casts) with torch's fused `F.rms_norm` is **5% slower**
  (2.4106 → 2.5358 ms, HF wins 15/15) *although numerically identical* (KL 7.2e-4, top-1 1.0):
  it keeps the same `MeanOps` reduce and adds an fp32 row multiply (`BinaryFunctor<float>` ×65 =
  0.189 ms) where HF multiplies in fp16. ⇒ A custom Triton norm is the only remaining route (P2h).

- **E4 (`EXP-p4-01`) — supported.** Routing/packing is the largest bucket in the hf/gather baseline
  (4.478 ms = **44.2%** of GPU busy) and **quantization-invariant** (4.478 / 4.512 / 4.526 / 4.555 ms
  for fp16 → w4a16) — the cap the plan predicted. The grouped path already removed it; see P2b.

- **★ W4 expert quantization is CLOSED as a negative result (2026-10-23, `EXP-p1-05`).** On the 5090,
  W8-RTN control passes the gate (KL 0.001154 / top-1 0.981818 / PPL +0.0443%); W4-RTN fails
  (0.044720 / 0.903030 / +4.39%), W4-GPTQ fails harder (0.054835 / 0.915152 / +7.15%), and the
  pre-registered ≤25% whole-layer W8 fallback (layers 0/13/14/15, bytes −36.92%) still fails
  (0.026455 / 0.933333 / +4.19%). Per the hypothesis's own refutation clause the expansion **stops**:
  no ctx=4096, no test split, no 4-bit latency. Do **not** re-run it; the record is
  `experiments/EXP-p1-05-w4-expert-calibration.md` and the artifacts are on the GPU box +
  `lemoq-artifacts/EXP-p1-05-2026-10-23/` (outside git). 4-bit stays gated by rule 2.
- **★ MXFP4 joins it as a negative result (2026-10-24, `EXP-p3-02`)**: RTN-MXFP4 FAILs the
  gate worse than int4-RTN did (KL 0.0739 vs 0.0447, top1 0.879 vs 0.903; 5 wikitext windows,
  0/5 pass) — the pre-registered "E8M0 power-of-2 scales are coarser" prediction hit. The
  "tl.dot first" prerequisite is retired (vector path implemented; K1 PASS). The last 4-bit
  route is NVFP4-style two-level scales — a NEW experiment needing approval. Do not re-run.
  Record: `experiments/EXP-p3-02-mxfp4-block-scale.md`; recon:
  `docs/reference/QUANT_FORMAT_RECON_2026-10-24.md`.

Read, in this order: **`RESULTS.md`** (the one-page result with a citation per number),
**`configs/experiments.yaml`** (the only source of hypotheses and cells), the **record** of the
experiment you are about to touch (`experiments/<EXP-ID>.md`), then
**`docs/handoff/HANDOFF_2026-10-25.md`** (the current handoff: §0 context after the 2026-10-24
round (de-publish closed + F5 wired & gate-PASSED) → §2 the F5 adoption runbook (PPL/MMLU → A/B
→ flip default → LMHEAD=1 refresh → sync) → §8 self-check; earlier handoffs are retired —
history lives in git).

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
   the thresholds in `docs/reference/METRICS.md`. A faster-but-different model is a failed experiment.
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
10. **Positioning rule (user-set 2026-10-24): batch=1 single-stream — no serving-engine
    integration.** vLLM/SGLang (or their nano/mini variants) are never adopted as a runtime or
    dependency: their wins (continuous batching, PagedAttention paging, scheduler reuse) live at
    high concurrency, outside this problem domain (`docs/ROADMAP.md` Tier D; every payload stamps
    `SCOPE_NOT_APPLICABLE`). R7 (`docs/handoff/R7_RECON_2026-10-24.md` §5) is a **one-shot
    optional external measurement anchor only** — it must not grow into an integration. The known
    **extension direction** is the nano/mini variants of those engines — but adopting them
    **implies supporting batch>1, i.e. a change of project positioning**: that needs the user's
    approval first plus a matrix/scope re-open per rule 9. An agent never starts it on its own.

## 3. Where the project stands (authority: the records + artifacts)

| Area | State | Authority |
| --- | --- | --- |
| Gate-0 probe, fp16 baseline, static MoE + cudagraph | done — P2j refresh: probes re-run, but all their cells are **host-bound ⇒ refresh VOID**, pre-P2i values kept | `experiments/EXP-p0-02-baseline-fp16.md`, `experiments/EXP-p0-04-cudagraph-static-moe.md` |
| cudagraph feasibility with the HF MoE path | `blocked` **by design** (hf + dynamic cache cannot capture) | `experiments/EXP-p0-03-baseline-cudagraph.md` |
| Accuracy gate — the only source of `GATE_CITATIONS` | done — **v3** (2026-10-24, R3 10-window re-adjudication): `w8a8@4096` **de-published** (10-window KL mean 0.0262 > 0.01 ⇒ 5-window PASS not robust; row now cites the 10-window artifact, verdict FAIL); all other rows unchanged | `experiments/EXP-p1-04-prompt-set-gate.md` |
| Precision side: error attribution, mixed bits, W8/W4 sweep | done (the old `EXP-p1-01` protocol was superseded by `p1-04`) | `experiments/EXP-p1-02-quant-error-attribution.md`, `experiments/EXP-p1-03-quant-mixed-bits.md`, `experiments/EXP-p1-01-w8-w4-accuracy.md` |
| Attention path: eager, kernel, in-graph | done — ★ 2026-10-24 F4 refresh: eager 2.5642/2.9275（sdpa 格不动），kernel **2.1946/2.4326 ms**（F4 前 2.2406/2.4779） | `experiments/EXP-p2-03-decode-attention.md`, `experiments/EXP-p2-04-eager-attention.md`, `experiments/EXP-p2-05-attn-kernel.md` |
| Grouped MoE GEMV (fp16) | done — ★ 2026-10-24 adoption refresh: 4.0829 → 3.6828 → **3.5340 ms** | `experiments/EXP-p2-02-moe-grouped-gemm.md` |
| E2 dequant fusion, 6 cells, 8-bit **and** 4-bit | done — **`refuted`**; P2j refresh **VOID** for all 6 cells (eager/host-bound) ⇒ pre-P2i values kept, verdict unchanged | `experiments/EXP-p2-01-dequant-fusion.md` |
| P3 wiring + E7, 16/16 cells | done — **judged**; refreshed 2026-10-08 (P2j): graph cells under the fused norm, eager cells **pre-P2i** (refresh VOID — host contention); `w8a8@4096` gate now reads PASS (borderline) | `experiments/EXP-p4b-01-truncation.md` |
| In-graph attention × quantization, 8 cells | done — **supported**; **canonical = v5** (2026-10-24 F4 adoption): quantized **1.1948/1.4528** @1024, **1.0898/1.3801** @4096, best cell **2.36× / 3.85×** vs fp16 × sdpa（w8a16×kernel **1.5116/1.7626 ms**, kernels/步 **329**） | `experiments/EXP-p4c-01-in-graph-attention.md` |
| E4 routing overhead | done — **supported**: 44.2% of GPU busy, and **quantization-invariant** (4.478/4.512/4.526/4.555 ms for fp16/w8a16/w8a8/w4a16) | `experiments/EXP-p4-01-moe-routing-overhead.md` |
| The elementwise/copy tail (0.90 ms, "~330 kernels") | done — classified: it is **~13 named families × 16–83 calls/step**, not a flat tail; module-level attribution VOID (its own K1 failed) | `experiments/EXP-p4c-02-elementwise-tail.md` |
| W4 expert calibration (4-bit) | **done 2026-10-23 — negative result** (RTN / GPTQ / ≤25% whole-layer W8 fallback all fail the gate; no latency published) | `experiments/EXP-p1-05-w4-expert-calibration.md` |
| E5 expert sensitivity, E6 KV INT8, E8 cross-bandwidth, FP8/MXFP4, dense curves, quant × batch (E3) | **not run**（batch sweep 的 **fp16 曲线已测**：G2 done 2026-10-10，金丝雀 −3.448%，`results/batch_sweep.json`；未跑的是矩阵里的**量化 × batch**） | `configs/experiments.yaml`, `PLAN.md` §7, `tools/batch_sweep.py` |
| ncu evidence (a Phase 2 deliverable) | **partly delivered 2026-10-08** — the counter path is **blocked by the host** (`RmProfilingAdminOnly: 1` + no `CAP_SYS_ADMIN` in the container; even the counter-free `gpu__time_duration.sum` is refused). Substitute delivered: kernel/call-level bandwidth attainment from hand-computed necessary bytes ÷ isolated cold timing ÷ measured achievable bandwidth, with an L2 flush and an explicit event-floor subtraction. Headline: fp16 grouped GEMV **71.0%** of the wall, fused dequant-GEMM **94.9%**, and the expert GEMV alone accounts for **90%** of E7's step-level saving. Occupancy **analytic bounds** delivered 2026-10-10 (`tools/occupancy_probe.py`, `status=done`: blocks/SM ≤12/≤12/≤6). Missing: stall attribution (needs counters) | `docs/reference/NCU_EVIDENCE.md`, `results/ncu_evidence/isolated.json`, `results/occupancy.json`, `tools/ncu_evidence.py`, `tools/occupancy_probe.py` |

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
| P2 | ✅ **done 2026-10-07** (then superseded: **v3** numbers are in §0/§3): this row's 1.169/1.250 and 1.70×/2.77× are the **v2** reading (HF norm) | the user approved a new id (`EXP-p4c-01`), see §3 | `experiments/EXP-p4c-01-in-graph-attention.md` | no |
| P2b | ✅ **classified 2026-10-07** (`EXP-p4c-02`): the 0.90 ms tail is **13 named families × 16–83 calls/step** — copies 0.17 (149 calls), RMSNorm mean+rsqrt 0.15–0.25, KV-cache write+cat 0.11, residual add 0.05, routing topK 0.05; **no single kernel >20%** | it was the largest non-weight bucket left | `experiments/EXP-p4c-02-elementwise-tail.md` | no |
| P2f | ✅ **identified 2026-10-07** (`EXP-p4c-02` §5.5): the copy family is **`aten::_to_copy` ×150/step (0.185 ms)** — dtype casts from HF's RMSNorm (`to(fp32)` + `to(fp16)` per norm × 65 norms) plus the norm's `pow/mean/rsqrt/mul` chain ⇒ P2f and the norm question are **the same fix** | — | — | no |
| P2i | ✅ **done 2026-10-08**: plumbing + `--aggregate-only` (interrupted-run finalisation by the tool's own aggregation code) + gate re-measured **under the fused norm** for both ctx + `GATE_CITATIONS`/artifacts/tests updated + `EXP-p4c-01`'s 8 cells re-run as **v3** (canary −0.067%, one contaminated cell re-run clean) | P2h passed K1+D but the gate on record was HF-norm | `EXP-p1-04` v2 section + `EXP-p4c-01` §5.7 + `results/EXP-p4c-01-run/v3_*` | done |
| P2j | ✅ **done 2026-10-08** (31 cells re-run, canary +0.13%): **13 graph cells adopted** (host ≤5%) — e.g. p4b's cudagraph set (E7 now **1.1911/1.0888**; 1.1927 was the canary-first denominator), p2-02 4.08→3.68, p2-04 3.11/3.48→2.63/3.00, p2-05 3.01/3.24→2.51/2.75; **18 host-bound cells VOID** (co-tenant CPU load; proved by the `--rmsnorm hf` control: `kernels_per_call`/`gpu_busy` match the archive exactly while `host_p50` goes 1.5 → 35.2 ms) ⇒ their pre-P2i payloads were **restored as canonical** and the contended refresh copies are kept in `results/EXP-p2j-refresh/contended/`. `w8a8@4096`'s gate flip is documented as a re-adjudication (borderline) — **since withdrawn by the R3 de-publish (2026-10-24, see the v3 row above)** | the norm swap made every recorded ms stale | per-record "★ P2j 刷新" sections + `results/EXP-p2j-refresh/` | done |
| P2k | **Re-run the 18 host-bound cells on a quiet box** (p0-02 ×2, p0-04 ×2, p2-01 ×6, p4b eager ×8) | their refresh was VOID on the shared card; their conclusions don't depend on the norm (~0.4 ms in an 8–272 ms step) so this is about record hygiene, not science | the same protocol (archive → re-run → canary → per-record update); if the box is never quiet, keep the pre-P2i labels | no (just needs a quiet machine) |
| P2c | Widen the routing classifier (torch `topk`/`sort`/`index_select`/`gather`/`scatter`) so the **grouped** path's routing cost can be quoted | `EXP-p4-01` §5.3: the classifier's 0.0% on the grouped path is a **blind spot**, not a measurement | new rules + tests; the number must be produced in its own round | **YES — it changes the meaning of the `routing_share_pct` field** |
| P3 | Breadth: E5 (`EXP-p4-02`), E6 (`EXP-p5-01`), E8 (`EXP-p8-01`, needs the V100), E1's group axis under the corrected prompt set, dense curves, batch sweep (E3 — today it has **no** experiment), FP8/MXFP4 (`EXP-p3-01/02`); W4A16 is **closed as a negative result** (`EXP-p1-05`), so only a >25% mixed-bit fallback would be a new experiment | each closes a declared hypothesis or lever; several need a matrix entry that does not exist | per matrix entry; E8 only after the same protocol passes on the V100 | partial: new matrix entries need approval |
| P4 | ✅ **done 2026-10-08 (except the counters)**: `tools/ncu_evidence.py` (counter path written, blocked on this host) + `docs/reference/NCU_EVIDENCE.md` + `RESULTS.md` + 3 charts generated from artifacts by `tools/make_figures.py` + README/HANDOFF corrections | closes Phase 2's last deliverable and Phase 7 | charts regenerate from `result.json` via `python -m tools.make_figures`; every number in RESULTS.md cites an artifact or record | no |
| W4 | ✅ **done 2026-10-23** (`EXP-p1-05`, GPU diagnostic — **negative**): W8 control PASS; W4 RTN / W4 GPTQ / ≤25% whole-layer W8 fallback all FAIL ⇒ pre-registered refutation clause stops the expansion; no matrix registration, no `GATE_CITATIONS` change | 4-bit was the last unclosed bit-width in the quantization-ceiling story | record + artifacts rsynced to `lemoq-artifacts/EXP-p1-05-2026-10-23/` (outside git) | done |
| **S1** | ✅ **done 2026-10-24** (Block A): A1 F2 A/B re-take **VALID** (off 1.9384 → on 1.8406 ms, −5.05%, 3/3 pairs, canary −0.12%) → A2 refresh **23/23 OK** (canary 0.30%, host ≤3.4%) → A3 `FUSION_DEFAULTS` four True (`938de63`; no-env canaries: kernel 1.5590 Δ+0.03% `packed_qkv_effective=True`, sdpa 2.9542 Δ−0.14% effective=False) → A4 numbers synced per `ADOPTION_SYNC_2026-10-24.md` → merge `main`. Headline: w8a16×kernel **1.5586/1.8140 ms** (kernels/step 539→345), E7 **1.1974/1.0912** | — | all acceptance criteria met | done |
| **S2** | ✅ **done 2026-10-24**: F4 KV direct write adopted (pre-registered rule picked it at 50.3 µs capturable; GPU K-gate 9/9 bitwise; A/B **VALID** off 1.5576 → on 1.5118 ms, −2.94%, 5/5 pairs, canary 0.154%; refresh 23/23, canary 0.336%; `FUSION_DEFAULTS` five True; kernels/步 345→329). Ledger re-run: latency-class vein mined out — next leader is gate_up 28.5 µs (config class) | — | all acceptance criteria met (tokens_equal + K1 + canary) | done |
| **S3** | `lm_head` w8 (≈60 µs, N7 口径：地板 0.1319→0.066 ms) **or** E6/E8 breadth | the last piece of the quantization lever, or the cheapest Tier B closure | **new gate row from a new measurement** (new experiment) before any latency is published | **F5 半边已批准（2026-10-24）**：走 `LEMOQ_LMHEAD_W8` 开关 + 独立 `LMHEAD_GATE_CITATIONS`（KL/top-1 + PPL<1% + MMLU<2 点三链）+ A/B + 刷新；E6/E8 仍待批 |
| **R3** | ✅ **done 2026-10-24 — de-publish 裁决（是）**：`w8a8@4096` 门行重指向 10 窗复测工件（KL 0.026159 ⇒ 推导 FAIL，top-1 0.963636 与 5 窗同值），已发布的 8.8232 ms 撤销；该格照跑、latency 只进 `gate_failed_evidence`（w4a16 同款语义）；引用来源集中在 `GATE_CITATION_SOURCES` | 10 窗重测显示 5 窗 PASS 不稳健（w7 重尾 0.2247），用户批准撤销 | `benchmarks/runner.py` 门表 + `experiments/EXP-p1-04` v3 + `EXP-p4b-01` §R3 | done |

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
- **canary**: same cell first and last; >5% drift ⇒ the whole round is `VOID` (keep the log, publish nothing).
  **验收用 `python -m tools.round_canary head.json tail.json [--graph] [--mark] [--out v.json]`** —
  退出码非零 = VOID，fail-closed（指标缺失/读不出一律 VOID）；`--mark` 归档后置 `status=void`
  （report 不计入已完成、`is_done` 为 False ⇒ 下轮重跑）；
- **cudagraph cells need their own canary**: an `eager` canary is *not* sensitive to a co-tenant
  (measured: it drifted 1.16% while three graph cells in the same round were ruined — one OOM, one
  3× outlier). Add a same-cell graph canary plus a payload self-check: a graph cell with
  `host_share_pct > 5` is contaminated（`tools/round_canary --graph` 机械执行这条自检）。

### Commit and verification gate

Run all four before every commit, on the machine you are editing:

```bash
ruff check . && black --check . && pytest -m "not gpu" -q && python -m tools.check_docs
```

The bar is **0 failed** (a few skips are normal). Then push, `git pull --no-rebase` on the other
machine, re-run there, and rsync `results/` back to the Mac.

### Never relax or bypass

- the thresholds in `docs/reference/METRICS.md` §1, `PARITY_GATE` (1e-3), `benchmarks/schema.py`, or the
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

**Rented RTX 5090 (primary; ours).** `ssh -p 12037 root@connect.weste.seetacloud.com`
(alias `5090b`；端口随租用实例变化，以当前 `docs/handoff/HANDOFF_*.md` 为准). Verified set: Python 3.12.3, `torch 2.8.0+cu128`, `triton 3.4.0`,
`transformers 4.57.6`, driver 595.71.05, sm_120 with 170 SMs, **measured** read bandwidth
1560.3 GB/s（只读；同一工件的 copy 是 1529.5；旧实例曾 1561.9，`results/machine_probe.json`）.

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
| `RESULTS.md` | the one-page result: five levers, the measured ceiling, negative results, gaps |
| `PLAN.md` | the frozen plan: thesis, hypotheses, phases, budget, risks (not a status page) |
| `configs/experiments.yaml` | the declarative matrix: the only source of hypotheses and cell ids |
| `experiments/<EXP-ID>.md` | the authoritative record of each experiment (status, results, applicability) |
| `docs/reference/NCU_EVIDENCE.md` | kernel-level evidence; why ncu counters are blocked on this host and what replaced them |
| `docs/reference/NCU_PLAN.md` | **换机才能做**的 ncu 补测清单：要测哪些 kernel、抓哪些 counter、验收判据（当前 blocked） |
| `docs/handoff/HANDOFF_2026-10-25.md` | **当前交接（自足版，可从零接管）**：§0 2026-10-24 轮后上下文（de-publish 闭环 + F5 接线/门过）→ §2 F5 采纳执行手册（PPL/MMLU→A/B→翻默认→LMHEAD=1 刷新→回填）→ §3 中止/负结果路径 → §8 接手自检 |
| `docs/handoff/HANDOFF_2026-10-23.md` | 此前一轮：W4 负结果、F2 定位与合规 A/B、刷新轮作用域（历史上下文） |
| `docs/handoff/ISSUES_2026-10-23.md` | 遗留问题清单 + **2026-10-24 复盘**（N1–N4 四条新发现与处置、剩余项按 S1/S2/S3 归属） |
| `docs/handoff/ADOPTION_SYNC_2026-10-24.md` | 采纳后的数字同步映射 + 翻默认的连锁（测试/refusal/`packed_qkv_effective`） |
| `docs/handoff/ADOPTION_SYNC_F5_2026-10-25.md` | F5（lm_head w8）采纳轮的取数地图（2026-10-24 夜预写，10-25 采纳轮回填） |
| `docs/apply/PORTFOLIO.md` | 面向实习投递的自述：简历 bullet（中英）、面试必问三题的标准答法、岗位映射与如实缺口（**不引入任何新数字**） |
| `docs/figures/` | the three charts, regenerated from artifacts by `python -m tools.make_figures` |
| `docs/README.md` | 文档地图：`docs/` 下每份文档是什么、给谁看 |
| `docs/ROADMAP.md` | 项目评估与扩展方向（Tier A 投递前 / B 矩阵缺口 / C 技术栈升级 / D 明确不做）——展望，不是状态 |
| `docs/FUSION_OPPORTUNITIES.md` | **融合优化任务规格（F1–F6）**：已测未采纳的三项、q/k/v 三合一、残差 epilogue、KV 直写、lm_head 量化、w4a16（阻塞）；含验收判据与八条注意事项红线 |
| `EXPERIMENTS.md` | mandatory record format and metric definitions |
| `docs/reference/METRICS.md` | metric definitions and the correctness thresholds |
| `docs/reference/MODELS.md` | model candidates and the verified OLMoE architecture (incl. the ctx≤4096 limit) |
| `docs/reference/SPECS.md`, `docs/reference/SERVER_RUNBOOK.md`, `ENVIRONMENT.md` | hardware facts, runbook, machine access |
| `benchmarks/runner.py` | the harness that produces cells; its refusals and payloads are the contract |
| `benchmarks/schema.py` | the machine-checked shape of `result.json` (hard rule 8 lives here) |
| `tools/check_docs.py` | dead-link and table-column checker (also run by `tests/test_links.py`) |
