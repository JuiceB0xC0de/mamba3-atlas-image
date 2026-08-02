# Mamba-3 atlas W&B measurement and visualization contract

**Status: CONTRACT ONLY. Not yet implemented or executed against W&B.**

Written 2026-08-01 before the first rented GPU run. This contract governs Stage A
backfill, GPU Probe, Capture Final, Stage B analysis, Stage C interventions, and the
final atlas workspace/report.

The objective is not to maximize uploaded bytes. It is to make every scientific result
queryable, traceable to exact inputs, visibly gated, and recoverable after a disposable
RunPod instance disappears.

---

## 0. Non-negotiable boundaries

1. **RunPod storage is ephemeral.** A local path on the pod is staging, never durable
   storage.
2. **Hugging Face remains the canonical evacuation store** for raw captures, complete
   authority reports, manifests, static figures, and W&B recovery spools.
3. **W&B is the analytical surface:** runs, exact artifact lineage, queryable Tables,
   custom Vega charts, static-image previews, runtime telemetry, and the claim ledger.
4. **No scientific dependency may use `:latest`.** Code must consume an exact W&B
   artifact version and verify its digest. Human-facing aliases may exist for navigation,
   but never select an input to a run.
5. Calling `run.use_artifact()` declares a lineage edge; it does not by itself prove the
   local bytes used by the computation were identical. The local input SHA256 (or the
   frozen checkpoint commit and manifest) must match the artifact metadata before work.
6. A W&B chart is not evidence by itself. Its backing Table/artifact, grain, sample count,
   null, uncertainty, exclusions, scope tier, and provenance must remain available.
7. W&B failure and scientific failure are separate. Neither may overwrite the other.

Standing family boilerplate for cross-scale figures:

> One released checkpoint per size and architecture; no seed replication. Family
> cross-section, not a fitted scaling law.

Standing cross-arm boilerplate:

> SISO/MIMO results compare released parameter-matched architecture bundles. They do
> not isolate MIMO rank from the MLP-width and chunk-size differences.

---

## 1. What success means

The atlas has three operational KPIs. They measure evidence quality, not whether a
favoured hypothesis survives.

| KPI | Definition | Passing condition |
|---|---|---|
| `atlas/evidence_complete` | Required gates have terminal outcomes; capture coverage is complete; every published claim has a scope, null, interval, and status; raw evidence is evacuated | Boolean true before a run is called publishable |
| `atlas/reproducibility_complete` | Exact checkpoint commit, code commit, token-contract SHA256, contract version, input artifact versions/digests, and output digests are present and mutually consistent | Boolean true; missing provenance is STOP |
| `atlas/falsification_complete` | Every predeclared claim has a terminal status: supported, exploratory, retracted, rejected, unsupported, removed-by-gate, or not-run—with a recorded reason | 100% of declared claims accounted for |

Guardrails:

- `coverage/blocks_fraction`, `coverage/valid_tokens_fraction`, and
  `coverage/content_tokens_fraction` must each equal 1.0 for a completed capture.
- `archive/hf_verified` must be true before pod termination.
- Either `archive/wandb_verified` or `archive/wandb_spool_verified_on_hf` must be true
  before pod termination. A spool is recoverable, not dashboard-complete.
- `statistics/multiplicity_controlled` must be true for a writeup-grade class effect.
- `scope/bundle_confound_labeled` must be true for every SISO/MIMO comparison.
- A higher claim-survival rate is **not** a success metric. Killing a bad claim is a
  successful atlas outcome.

---

## 2. Run identity and grouping

One W&B run represents one executable scientific job, not an entire project lifetime.
All related jobs share an `atlas_build_id` and W&B group.

Required immutable config on every run:

| Field | Meaning |
|---|---|
| `atlas_build_id` | Stable identifier joining probe, capture, analysis, intervention, and report inputs |
| `contract_version` | Version/date of this contract and the Stage B contract |
| `stage` / `job_type` / `script` | Exact executable role |
| `code_git_commit` / `code_dirty` | Source provenance; dirty state is explicit, never hidden |
| `model_repo` / `model_tag` / `model_commit` | Exact checkpoint identity |
| `arm` / `size` / `n_layer` / `nheads` / `mimo_rank` | Realized architecture identity |
| `token_contract_sha256` / `token_budget` / `stream` | Exact corpus input and selected prefix |
| `selected_layers` / `head_selection_rule` | Predeclared observation/intervention surface |
| `runtime_pin_set` / `gpu_name` / `gpu_capability` | Execution environment |
| `attribution_ceiling` | `mamba3-wide`, `bundle-level`, `within-mimo`, or `state-prohibited` as applicable |
| `bundle_confounds` | MLP-width, parameter-count, and chunk-size differences |
| `declared_tolerances` / `null_policy_version` | Rules fixed before seeing GPU results |

Do not put result state in config. Terminal gate verdicts, evidence status, coverage,
archive receipts, and claim status belong in run summary and Tables.

Recommended job types:

- `checkpoint-register`
- `static-atlas`
- `gpu-probe`
- `activation-capture`
- `stage-b-analysis`
- `head-freeze`
- `causal-intervention`
- `mediation`
- `atlas-report`

Use explicit axes with `wandb.define_metric`; never let W&B's default step imply an
ordering that does not exist:

- depth views: `layer` or `depth_fraction`
- capture progress: `blocks_processed` and `valid_tokens_processed`
- interventions: `dose` within an `operation_id`
- probe execution: `gate_index`

---

## 3. Exact artifact lineage

The intended graph is:

```text
exact checkpoint reference(s) ----+
                                   +--> gpu-probe-report
token-contract -------------------+

exact checkpoint + token-contract + frozen-stage-a + gpu-probe-report
                                   --> stage-b-capture

stage-b-capture + null-policy      --> stage-b-analysis
stage-b-analysis                   --> frozen-head-selection

exact checkpoint + token-contract + gpu-probe-report + stage-b-capture
+ frozen-head-selection            --> stage-c-interventions / mediation

all terminal analysis artifacts    --> atlas-evidence-ledger --> report assets
```

Required artifact collections and types:

| Name pattern | Type | Contents |
|---|---|---|
| `checkpoint-{model_tag}` | `model-reference` | Exact HF repo/commit, config, realized shape manifest, stable HTTPS references where supported; weights are not duplicated |
| `token-contract` | `dataset-contract` | Schema-v3 NPZ, manifest, SHA256, frozen corpus metadata |
| `stage-a-static-atlas` | `analysis` | `static_atlas_all8.npz/json`, `a2_results.json`, checkpoint census, claim ledger |
| `gpu-probe-report` | `validation` | Complete G1-G7 report plus both full G2 authority sidecars and archive receipts |
| `stage-b-capture-{model}-{stream}-{budget}` | `capture` | Manifest, queryable summaries, HF URI + SHA256 for the full capture, never a silent 4 GiB duplicate |
| `stage-b-analysis` | `analysis` | Null results, corrected intervals/q-values, convergence results, evidence ledger |
| `frozen-head-selection` | `selection-contract` | Exact selected heads/layers and criterion provenance |
| `stage-c-interventions` | `causal-analysis` | Dose-response, controls, ablations, mediation outputs |
| `atlas-chart-data` | `visualization` | Chart Tables, preset-version manifest, static PNG/SVG, source artifact versions/digests |

Every downstream run must:

1. resolve an exact artifact version such as `:v7`, never `:latest`;
2. call `run.use_artifact()` for the exact input;
3. verify expected artifact digest and local SHA256/manifest identity;
4. record the resolved version and digest in config/summary;
5. call `run.log_artifact()` for each output and wait for commit before declaring W&B
   archival successful.

Registry linkage is optional polish. Exact reference artifacts are sufficient for lineage.
Do not make Registry availability a prerequisite for scientific execution.

---

## 4. Canonical Tables

Tables are wide enough to support alternative honest views, not minimal `x/y` chart feeds.
Each table carries `atlas_build_id`, run ID, model repo/commit, artifact version/digest,
token-contract SHA, stream/budget, sample count, evidence status, and relevant exclusions.

| Table key | One row per | Required scientific fields |
|---|---|---|
| `gate_verdicts` | gate | gate, verdict, reason, timing, tolerance, achieved error, evidence path, allowed/prohibited claims |
| `evidence_ledger` | declared claim | claim ID/text, scope, required gates, null, multiplicity rule, status, reason, supporting artifact/chart |
| `static_head_geometry` | model × layer × tensor × head | spread, rotation-from-init, scale, participation ratio/effective rank; head identity only within that checkpoint |
| `depth_sign_model` | model × layer/depth | D-positive count, nheads, observed fraction, fitted probability, interval, BIC model, crossover/break uncertainty, L0 boundary flag |
| `capture_role_summary` | block × layer × role × head | recurrence quantities, phase summaries, B/C geometry, finite/censored counts, semantic label/source category |
| `rank_geometry` | model × block/class × layer × head × object | Gram eigenvalues, participation ratio, total/common/differential norms and both unsquared/squared ratios, alignment validity |
| `class_contrasts` | contrast × model × layer × head × quantity | effect, interval, raw p, corrected q, same-class null, permutation null, block/sample counts |
| `convergence` | model × quantity × layer/class | 150k and 500k values/intervals, Spearman, concordance correlation, absolute and relative change, interval-overlap decision |
| `l0_direct_logit` | model × block/class × layer/control | loss/logit/entropy change, interval, null/control, valid-token count |
| `dose_response` | operation × model × layer × head/subspace × dose × series | real, same-norm control, net, interval, sample count, displacement norm, gate state |
| `mediation` | model × layer × pathway/intervention | effect and interval, mixer/MLP context, bundle-confound flag, admissible interpretation |
| `runtime_telemetry` | gate/phase/length | warmed and compile time, peak GPU/host memory, projected bytes, upload/verification time, coverage |

Custom-chart logged keys should stay at or below roughly 10,000 plotted rows. Larger
artifact Tables may retain more rows, but chart feeds are partitioned by checkpoint,
quantity, or purpose instead of silently sampled. No raw per-token activations go to W&B.

---

## 5. Custom chart contract

Custom panels use versioned Vega presets. The preset source must also live in the repo
when implementation begins; the W&B UI copy is not the only definition. Code refers to a
stable `vega_spec_name` plus a recorded preset version.

Every chart has a neutral descriptive title and a subtitle/caption containing units,
checkpoint/arm, corpus and token budget, sample count, uncertainty/null, exclusions,
scope tier, and evidence status. Interpretations belong beside the chart, not smuggled
into the title.

| Preset / figure | Analytical question | Backing data and form | Required guardrail |
|---|---|---|---|
| `atlas/gate-matrix-v1` | Which evidence is admissible? | gate/claim matrix scorecard | STOP, BOUNDARY, and archive failure remain distinct |
| `atlas/d-sign-depth-v1` | How does P(D>0) vary with depth? | observed binomial counts + fitted logistic and bootstrap band; L0 distinct | Use A2 results, **not** rank-spread CIs; family trend labelled exploratory |
| `atlas/within-checkpoint-head-heatmap-v1` | Where is a metric concentrated inside one checkpoint? | layer × head heatmap, faceted by tensor/metric | Never pair or align head indices across independently trained models |
| `atlas/rank-spectrum-v1` | Is dynamic rank geometry concentrated or distributed? | eigenspectrum/scree plus participation-ratio distribution | Capacity/utilization, not causal necessity |
| `atlas/static-vs-dynamic-rank-v1` | Does static differential commitment appear in activations? | paired distribution/scatter with intervals | Same definitions/denominators; no per-head cross-model pairing |
| `atlas/phase-role-v1` | How do phase increment, winding, and concentration vary by depth and token role? | small multiples/heatmaps for BOS, interior, final | Circular quantities retain pair resolution; G2 evidence status visible |
| `atlas/boundary-role-v1` | Is L0/other-layer behavior boundary-conditioned? | faceted dot-and-interval or distribution view | BOS/interior/final sample counts and censoring shown |
| `atlas/class-contrast-forest-v1` | Which class contrasts survive both nulls? | effect + interval forest plot | Corrected q-values, both nulls, document-block grain |
| `atlas/l0-direct-logit-v1` | Does L0 produce an unusual direct-logit effect? | layer/control effects with intervals | Boundary anomaly until the declared test passes |
| `atlas/convergence-v1` | Are 150k and 500k estimates stable in rank and magnitude? | identity scatter + interval/relative-change diagnostics | Spearman never stands alone; include CCC and magnitude change |
| `atlas/dose-response-v1` | Does a targeted edit beat a same-norm control over dose? | real/control/net lines with intervals, faceted by operation | Neutral title; interpretation/status separate; multiplicity recorded |
| `atlas/rank-necessity-v1` | Is performance sensitive to slots, subsets, or basis-robust subspaces? | cumulative ablation and principal-direction curves | Raw slot identity labelled basis-dependent; matched random subspaces |
| `atlas/mediation-v1` | Which measured pathway best explains a bundle difference? | pathway effects with intervals | Cannot promote bundle association to pure MIMO causation |
| `atlas/runtime-v1` | What did the build cost and where was time/memory spent? | phase/length bars and telemetry table | Operational diagnostic, never scientific evidence |
| `atlas/evidence-ledger-v1` | What survived, was killed, or was blocked? | filterable status table/scorecard | Retractions and negative results remain visible |

Static publication figures are generated from the same reviewed Tables, exported as PNG
and SVG, logged with `wandb.Image`, and uploaded to HF. Each image manifest records its
SHA256, source Table artifact/version/digest, filters, preset/version or renderer commit,
and caption. A pretty image without its backing evidence is incomplete.

Color and accessibility:

- neutral backgrounds and quiet grids;
- one-root palette for single metrics, at most two roots for focal/control or signed
  comparisons, and no default red/green scientific semantics;
- color is never the only distinction—use line style, marker fill, direct labels, or
  faceting;
- comparable panels use identical scales unless a focused scale is explicitly labelled.

---

## 6. GPU Probe instrumentation

The real probe run logs all G1-G7 outcomes, not just terminal success:

- immutable config from §2;
- `gate_verdicts` after each gate;
- summary fields for overall scientific verdict, archive status, G2 coverage, G3 state
  boundary, G4 compilation policy, G5 memory/artifact projection, G6 lower-bound compute,
  and G7 storage-precision decision;
- complete G2 authority sidecars as W&B validation artifacts and HF files;
- explicit `state_parity_validated: false` until cache/state tensors are actually compared;
- explicit allowed/prohibited claim lists.

The G2 elementwise mixer parity report is the authority. Top-k/logit agreement may be logged
as a supplemental diagnostic but cannot set a verdict.

Scientific STOP halts later science. W&B archival failure halts new GPU work, preserves the
scientific verdict already reached, and triggers the evacuation policy in §9.

---

## 7. Capture and analysis instrumentation

Capture Final logs progress scalars at bounded cadence, never every token:

- blocks/valid/content tokens processed and coverage fractions;
- peak GPU allocated/reserved and host RSS;
- compile/warm/accumulator/serialization/upload timings when measured;
- current layer/block only as operational progress, not a scientific x-axis.

At completion it logs the canonical Tables from §4, the full manifest, evidence status for
each quantity (`source-derived`, `fixture-tested`, `gpu-parity-validated`, `withheld`), and
an HF reference plus SHA256 for the full capture. The multi-gigabyte capture remains on HF;
W&B receives queryable summaries and lineage rather than a duplicate solely for volume.

Analysis logs every tested cell, not only survivors. Multiplicity correction and minimum-cell
rules are applied before `evidence_ledger` can mark a class effect writeup-grade. Convergence
uses both ordering and magnitude-sensitive diagnostics; Spearman alone cannot pass the gate.

Cross-arm results are always distributional bundle comparisons. No chart or table pairs
SISO head `i` with MIMO head `i`.

---

## 8. Stage C instrumentation

Every intervention logs:

- exact selected layer/head or basis-robust subspace and selection artifact version;
- operation, dose, displacement norm, affected rank-coupled paths, and random seed;
- real edit, same-norm control, and net effect on the same examples;
- uncertainty, sample count, null/control definition, multiplicity status;
- gate/evidence state and admissible claim scope.

Required visible comparisons include individual slots, subsets/cumulative rank, joint
rank-coupled edits, activation-SVD/principal directions, and matched random subspaces.
B and C must be tested jointly and separately before any use of `synergy`. Basis-dependent
slot results remain labelled as such.

The gate is per-rank before collapse in this architecture. Generic claims about a nonlinear
gate over the sum of ranks are superseded by the source-derived/reference-tested decomposition,
pending GPU parity.

---

## 9. Disposable-pod archival and recovery

Before model load:

1. validate W&B entity/project and `WANDB_API_KEY` presence without printing the key;
2. validate HF destination and `HF_TOKEN` presence without printing the key;
3. create the stable `atlas_build_id` and W&B run ID;
4. finish structural preflight before CUDA allocation.

During execution, reports/manifests are written atomically and uploaded to HF at existing
scientific gate boundaries. W&B logs live in online mode when reachable.

Before pod termination:

1. atomically publish local scientific outputs;
2. upload to HF and remotely verify existence, byte size, and recorded SHA256/receipt;
3. finish the W&B run and wait for artifact commits;
4. verify the remote W&B run ID and required artifact versions/digests through the API;
5. if W&B verification fails, preserve the complete syncable local W&B run directory plus
   manifests/Tables on HF and remotely verify that recovery bundle;
6. print a terminal receipt listing HF verification, W&B verification or recoverable-spool
   status, scientific verdict, prohibited claims, and exact unresolved work.

Pod termination is prohibited until HF archival is verified and either W&B is remotely
verified or a syncable W&B recovery spool is verified on HF. A later `wandb sync` must finish
before the dashboard/report is called complete.

---

## 10. Things deliberately not logged

- raw per-token activations or recurrent states;
- the full 4+ GiB capture duplicated into W&B without a scientific need;
- `wandb.watch()` on inference-only atlas jobs;
- uncorrected significant cells without the full tested family;
- head-index pairings across independent checkpoints;
- auto-generated causal titles or interpretation labels;
- superseded/retracted results hidden from the project;
- system metrics presented as scientific findings;
- secrets, absolute personal paths, or pod-local paths as if they were durable URIs.

---

## 11. Implementation sequence

This is one instrumentation effort, not a new scientific stage ladder:

1. Add one shared W&B helper/schema module and offline self-checks.
2. Instrument GPU Probe and Capture Final, including HF/W&B dual archival and exact lineage.
3. Instrument Stage B analysis/head freezing and the corrected A2/static backfill.
4. Instrument audited Stage C interventions/mediation.
5. Create/version the Vega presets, static figures, workspace, evidence-ledger report, and
   final runbook commands.

No real GPU run is authorized by this document alone. Code-complete, offline-tested,
W&B-live-tested, GPU-executed, HF-verified, and report-complete are separate statuses and
must never be collapsed into the word “done.”
