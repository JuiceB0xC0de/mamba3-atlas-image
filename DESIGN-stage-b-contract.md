# Stage B measurement contract — `capture_contrast.py` v2

**Status: CONTRACT, frozen before implementation.** Nothing here is authorized to run on
rented compute until Rick approves the spend explicitly. Written 2026-07-31, after Stage A closed.

**Standing boilerplate for every figure produced under this contract:**

> One released checkpoint per size and architecture; no seed replication.
> Family cross-section, not a fitted scaling law.

---

## 0. What Stage A established, and what it cannot

Stage B exists to answer what static weights structurally cannot. Carrying forward:

| Claim | Status after Stage A | Scope |
|---|---|---|
| D sign progression with depth | Established, all 8 checkpoints | Mamba-3-wide |
| L0 boundary discontinuity | Established, all 8 checkpoints | Mamba-3-wide |
| MIMO 5–7x larger static median \|D\| | Established | **Bundle-level association only** |
| ~half of `B_bias` drift is rank-differential | Established | MIMO-specific static *capacity* |
| Universal late-stack kink | **Rejected** (2 MIMO models prefer 0 breakpoints) | — |
| Rank redundancy increasing with scale | **Rejected** (drift-denominator artifact) | — |
| Meaningful `dt` means | **Rejected** (heavy-tailed; use medians/quantiles) | — |
| Crossover scaling trend | **Unsupported** (93% / 71.5% / 68% ordering stability) | Exploratory only |
| L0 as detokenizer | **Not established.** Boundary anomaly pending §6 | — |

**Attribution ceiling.** The released pairs are parameter-matched (+0.16 to +0.27%) but not
architecture-identical: MIMO runs an MLP exactly 256 narrower at every size, and `chunk_size`
16 vs 64. Therefore three distinct claim types, never conflated:

1. **SISO/MIMO comparison** — differences between two complete released *systems* (bundle-level).
2. **Within-MIMO rank intervention** — evidence about MIMO-specific mechanism.
3. **Architectural attribution** — requires matched training ablations with equal MLPs. **We do
   not have these and cannot manufacture them post hoc.** No normalization removes this confound.

---

## 1. Preconditions (gate — nothing downstream is valid until these pass)

**1.1 Source and runtime assertion.** Record and assert at capture time: checkpoint repo + commit
sha, `mamba_ssm` source revision, tilelang 0.1.8, apache-tvm-ffi 0.1.8.post2, torch/triton
versions, GPU arch. Abort on mismatch rather than warn.

**1.2 Split-order assertion.** Re-assert `[z, x, B, C, dd_dt, dd_A, trap, angles]` and every
derived size against `in_proj.weight.shape[0]`. Already verified for both arms; assert anyway.

**1.3 Prefill / decode / reference parity.** Short sequence, identical tokens, matched precision,
three paths: kernel prefill, single-step `step()` decode, and a plain PyTorch reference
recurrence. **No state-level interpretation is permitted until these agree.** Record tolerances
achieved, not assumed.

> ### SUPERSEDING CORRECTION — 2026-07-31 (Stage 1A)
>
> **What this section used to say, and why.** When this contract was frozen, three
> recurrence semantics could not be settled by reading the vendored source, so they were
> carried as `RefSwitches` in `analysis/reference_recurrence.py` — `rotate_q`,
> `include_diagonal`, `d_before_gate` — and §1.3 assigned parity the job of *adjudicating*
> them. The reasoning at the time: the switches provably changed the output, so a parity
> comparison could in principle select the correct combination.
>
> **Why that was wrong.** Selecting semantics by output agreement is fitting, not deriving.
> A top-k or even tensor-level agreement can be reached by a wrong recurrence that happens
> to compensate, and once a switch is chosen that way there is no independent evidence left
> to check it against. The switches also encouraged treating a parity failure as a search
> signal rather than a defect.
>
> **This language is SUPERSEDED, not deleted.** It is preserved above so the record shows
> why the switches existed and what replaced them. Stage 1A re-derived every rule directly
> from the pinned official source; `RefSwitches` no longer exists. The fixed semantics are
> recorded immediately below, and parity's role is correspondingly narrowed.

**Fixed recurrence semantics (source-derived, not parity-selected).** Line references are the
ones recorded in `analysis/reference_recurrence.py`:
`M = upstream/pypi_232_post1/mamba_ssm/modules/mamba3.py`,
`K = upstream/tilelang/mamba3/mamba3_mimo_fwd.py`,
`G = upstream/triton/mamba3/angle_dt.py` (byte-identical to the `pypi_base` copy).

| # | Fixed semantics | Source |
|---|---|---|
| 1 | **Prefill rotates BOTH sides.** Query/C is rotated in place at K L246-259 and key/B at K L266-275, with the same angle and the same sign. The two are rotated by their own cumulative phase. | K L246-259, K L266-275 |
| 2 | **Phase increment is `tanh(raw_angle) * pi * Delta`**, accumulated by an INCLUSIVE cumsum plus any carried phase, reduced `mod 2*pi` with the floor convention. Persistent `angle_dt_state` is what makes it cumulative across chunks and steps. | G L94, G L101, G L104-108, G L114-117; M L447 |
| 3 | **The current-token diagonal uses `gamma` alone**, re-added separately after the intrachunk mask deliberately excludes the same-step term. | K L326; mask at K L296-298 |
| 4 | **Strictly earlier contributions use the official trapezoidal scaling** `trap_scale = gamma + shifted`, where `shifted_t = Delta_{t+1} * (1 - lambda_{t+1})`, zero at the final position. `trap_scale` multiplies K only. | K L182-198, K L283-287 |
| 5 | **D operates on the RANK-EXPANDED value** (`D[h] * PsiV`, where `PsiV[r,p] = x[p] * mimo_x[h,r,p]`) and enters the diagonal term BEFORE the gate and BEFORE the `mimo_o` collapse. | K L328-335; PsiV at K L206-207 |
| 6 | **Gate and collapse order are source-derived.** Gate is `o*tanh(o)+o` with `o = z*mimo_z*0.5`, algebraically `silu(z*mimo_z)`; the collapse multiplies by `mimo_o` and sums over rank. | K L346-349, K L351-365 |

Also fixed by the same derivation: partial RoPE, `rotary_dim_divisor = int(2/rope_fraction)`, so only
`d_state // divisor` pairs rotate (M L99); the diagonal is built from the UNROTATED product because
`qk_dot` is computed at K L226, before either rotation.

**Parity's role is now narrower.**

- **G2 measures numerical agreement** between the FIXED candidate recurrence and the official
  prefill **mixer output**. It is a tolerance measurement, not a semantic search.
- **A G2 failure is STOP and requires debugging.** It does **not** authorize toggling a semantic
  switch — there are none — nor widening a declared tolerance. Debug source revision, precision,
  slicing, or the transcription itself.
- **Mixer-output parity does NOT establish kernel-state equivalence.** Agreement on the block's
  output tensor is compatible with a different internal state factorization.
- **Explicit `ssm_state` / `step()` parity remains separately required for any state claim.** The
  kernel accumulates chunkwise with `DA_CS_REV` scaling (K L388-394) and its final-state convention
  is unverified; the reference's own accumulator is not asserted to equal it.
- **G3 BOUNDARY still permits** prefill-derived quantities and behavioural probes, and still
  **prohibits** kernel-state claims, `h_t` claims, and state-trajectory claims.

**Two failure modes, with different consequences. Do not conflate them.**

| Failure | Diagnosis | Consequence |
|---|---|---|
| `step()` unavailable or unsupported in the environment (CuteDSL decode path does not build or run) | **Infrastructure limitation** | Continue with prefill observables. **Prohibit** explicit decode-state / `h_t` claims and all state-based Stage C work. Prefill-derived analysis remains valid and publishable. |
| `step()` runs but disagrees numerically with prefill or reference | **Genuine parity failure** | **Stop.** Debug source revision, precision, recurrence formulation, or cache semantics before any capture. |

The first is a boundary on scope. The second is a correctness stop. A broken CuTe path must not
kill useful prefill analysis, and must not be allowed to quietly license state claims either.

---

## 2. Known extraction hazards (each one already cost us something)

- **`ngroups = 1`: B and C are SHARED across all heads.** Raw sliced B/C expanded to 64 heads is
  64 identical copies. Per-head B/C from the `in_proj` hook is meaningless. Layer-level use is
  fine and is how the existing L14/L18-19 results should be read.
- **The `in_proj` hook is pre-norm and pre-bias.** `B_norm`/`C_norm` apply after it; `B_bias`/
  `C_bias` `(nheads, rank, d_state)` enter *inside* the kernel as `K_bias`/`Q_bias`. Post-bias
  per-head B/C must be **reconstructed**, not hooked.
- **Sigmoid is applied inside the kernel.** Prefill receives a raw `trap` logit. Capturing `trap`
  without applying `sigmoid` yields nonsense.
- **`dt` is heavy-tailed across heads.** Report medians and quantiles. Means hid a 2.3-token
  median behind a 145-token mean.
- **Head index is not shared across independently trained models.** Never pair SISO head *i*
  with MIMO head *i*.

---

## 3. Measurement inventory

All statistics accumulate **online, per head**, never as per-token tensor dumps. Small bounded
reservoirs are permitted for inspection. Per-head resolution is a **prerequisite for Stage C**,
not an enhancement: interventions must know which head to touch.

### 3.1 Recurrence quantities (per head, per layer, per class)

Derived, not raw. From `mamba3.py` L194-198 and `mamba3_mimo_fwd.py` L182-198:

```
Δ_t          = softplus(dd_dt + dt_bias)
A_t          = -heavy_tail_activation(dd_A),  clamped ≤ -A_floor
α_t          = exp(A_t · Δ_t)                      per-step retention
λ_t          = sigmoid(trap_t)
γ_t          = Δ_t · λ_t
shifted_γ_t  = Δ_{t+1} · (1 - λ_{t+1})             0 at final position
trap_scale_t = γ_t + shifted_γ_t                   the scalar actually applied to K
```

Capture: distributions (median, IQR, p10/p90) of `λ`, `α`, `Δ`, `trap_scale`; **local half-life**
`ln2 / (-A_t Δ_t)`; and **cumulative retention** — how many future positions until
`Σ A Δ` crosses `-ln 2`. The cumulative form is the faithful sequence quantity because the
parameters change every token. `β` is reconstructed only when discussing the recurrence
(`β_{t+1} = α_{t+1} Δ_{t+1}(1 - λ_{t+1})`), never captured as if the kernel applied it.

Also capture RoPE phase: rotation angle, circular concentration, cumulative phase, winding.

### 3.2 B/C utilization — three distinct objects, not one

1. **Shared dynamic rank utilization**, post-BCNorm. Layer-level by construction (`ngroups=1`).
2. **Reconstructed post-bias per-head effective B/C.** The only per-head B/C object that exists.
3. **Injection-weighted utilization**, after recurrence scaling and rotation.

Each measured with the **same rank-differential denominator as the corrected static analysis**:

```
‖X - mean_r(X)‖² / ‖X‖²
```

**This closes the loop on the retracted claim.** Static weights commit ~50% of `B_bias` drift to
rank-differential structure. If activations route through those channels at a far lower rate, the
retraction overshot; if they match, capacity and use agree. Neither answer is assumed.

### 3.3 Per-rank output contribution through `mimo_x`, `mimo_z`, `mimo_o`

B/C Grams alone cannot show whether rank channels affect the mixer *output*. Measure each rank
slot's contribution to the final mixer output, per head.

### 3.4 Pathway norms

Actual measured magnitudes, per head, per layer: retained state, previous-token injection,
current-token injection, `D ⊙ x` feedthrough, gated output, recurrent output, and the MLP path.
Mixer-vs-MLP residual contribution is **mediation characterization only**; it does not erase the
training confound (§0).

**Note:** larger static `|D|` does **not** establish larger activation-level skip contribution.
That inference requires §3.4 and is currently unmeasured.

### 3.5 Sparsity, labeled honestly

The existing metric is a **thresholded near-zero fraction relative to tensor scale**, not
sparsity in any principled sense. Keep it under that name, and supplement with quantiles and
concentration measures.

---

## 4. The L0 direct-logit test (§6 of the review, promoted)

The only valid test of the detokenizer reading. Static weight cosine **cannot** do this: the
`out_proj` row/unembedding dimensions do not even match (`(d_model, d_inner)` vs `(vocab,
d_model)`), and column alignment ignores which channels activate, their signs, gating, and norms.

Procedure, on identical tokens across layers:

1. Capture each layer's actual residual update `Δr_l`.
2. Push `(r + Δr_l)` through the final norm and the tied unembedding.
3. Measure direct-logit change, target-token logit change, entropy change, next-token loss.
4. Compare L0 against every later layer.

This can support **or kill** the detokenizer interpretation. Until it runs, L0 is a boundary
anomaly and nothing more.

---

## 5. Null policy — matched to the claim, not applied uniformly

A same-class permutation cannot wrap every metric; it is only the right null for class contrasts.

| Claim type | Correct null |
|---|---|
| Class contrasts | Same-class splits **and** label permutations |
| Corpus statistics | Document-block bootstrap; position and length matching |
| Rank utilization | Rank permutations, orthogonal rotations, matched random subspaces |
| Causal interventions (Stage C) | Same-norm random edits, dose-response controls |
| SISO/MIMO comparison | Distributional comparison labeled **bundle-level**. Never a permutation pretending the architectures are exchangeable |

Class contrasts additionally require **matched positions and document structure**, or
code-vs-prose differences may be token-frequency or sequence-boundary artifacts.

Block sizes must respect token autocorrelation.

---

## 6. Falsification gates

- No state interpretation if prefill/decode/reference parity fails (§1.3).
- No MIMO attribution if an effect disappears under MLP-path analysis or varies inconsistently
  across scale. Bundle-level wording is mandatory regardless.
- No class-mechanism claim unless it exceeds **both** matched same-class and permutation nulls.
- No rank-information claim from Gram separation alone; requires Stage C intervention.
- No stable-atlas claim if 150k and 500k disagree on layer rankings or interval widths.
- No detokenizer claim without §4.

---

## 7. Execution order — three phases, split by who pays

The implementation is free. The *complete* 187m smoke is not, because the parity gate needs CUDA.

### B0 — free, local, no GPU

Everything verifiable without the kernel stack:

- source / runtime / commit assertions (§1.1) and split-order assertion (§1.2)
- tensor slicing and hook placement
- post-BCNorm / post-bias B/C **reconstruction**, checked against hand-computed values
- accumulator correctness (online moments, quantiles, Grams, reservoirs) on synthetic input
- corpus and token contract: pinned tokenizer revision, no chat template, explicit BOS/EOS and
  document separators, identical token ids and packing across arms, corpus hash recorded
- any reference-path execution that runs on CPU

B0 must pass in full before anything is rented.

### B1 — minimal GPU parity gate (**first paid operation**)

Identical-token comparison across official prefill, `step()`, and the PyTorch reference at matched
precision. Likely hardware-sensitive; the CuTe decode path requires CUDA. Smallest checkpoint,
shortest sequence that exercises the paths. Outcome routes per the §1.3 failure table.

### B2 — paid capture (**only after B1 passes**)

1. **187m pair**, both arms, full pipeline.
2. **Frozen 1.5b layers**, from A2:
   - `mimo-1.5b` `[0, 1, 4, 12, 13, 14, 16, 17, 18, 23]`
   - `siso-1.5b` `[0, 1, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 23]`
   Keep the dense SISO L7–L15 block until **measured** runtime gives a reason to prune it.
3. **150k tokens per class first.** 500k only if intervals or layer rankings have not converged.

Corpora: `~/mamba-atlas/prompts/` (all four present). Diff against the other on-disk copies and
pin the chosen one by hash in the run record.

---

## 8. Deferred to Stage C

State trajectories `h_t`; retention/overwrite probes (key-value recall, distractors,
contradictions, deletion, replacement); mechanism interventions on λ, phase, B/C biases, rank
channels, D; rank causality by removal/rotation/patching; SISO/MIMO mediation analysis.

Stage C head selection must use **frozen Stage B criteria**, decided before looking at Stage C
results, not designed around layer averages after the fact.
