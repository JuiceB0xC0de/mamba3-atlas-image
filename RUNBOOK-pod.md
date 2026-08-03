# Pod runbook — Stage B and C

Everything below is already written and syntax-clean. Nothing here needs authoring
on the clock. Run top to bottom; stop where a gate says stop.

Env pin set (from the release-stack notes, verified 2026-07-30):
python 3.10 · torch 2.11.0+cu128 · triton 3.6.0 · **tilelang 0.1.8** ·
**apache-tvm-ffi 0.1.8.post2** · quack-kernels>=0.3.4

The tvm-ffi pin is load-bearing and undocumented: tilelang 0.1.8 leaves it
unbounded, pip resolves 0.1.12, and you get either an `AttributeError` on
`__dict__` or a hard `TypeAttr __ffi_repr__ is already registered` core dump.

---

## 0. Preflight — run this first, it replaces the steps below it

    source /opt/mamba3/analysis/pod_preflight.sh

Source it, do not execute it: it exports `HF_TOKEN` into your shell, and that
does not survive into a new one. It is idempotent. It checks the env pins,
fetches the matched token-contract pair, caches both checkpoint arms, and then
prints the probe and capture commands with everything wired.

Every check in it exists because it cost pod time on 2026-08-03.

### DO NOT rebuild the token contract

The instruction that used to live here —

    python analysis/token_contract.py --bos --out token_contract.npz

— is **stale and harmful**. `token_contract.py` is non-deterministic: at least
three artifacts exist from the same script and the same prompts.

| artifact_sha256 | where | used by |
|---|---|---|
| `8893c3aa...` | public repo `contracts/token_contract.npz` | the 187M captures |
| `42568fe5...` | private repo `mamba3/stage-b/token-contract/` | mimo-1.5b, and the 0b run |
| `08e679e3...` | described by the public `contracts/` manifest | nothing current |

The public `contracts/` folder pairs a manifest describing `08e679e3` with an
npz that hashes to `8893c3aa`. They do not describe each other and the probe
rejects the pair outright. **Use the private repo's pair**, which is internally
consistent — the preflight script does this for you.

Note that `08e679e3` appears as the old manifest's `artifact_sha256` and as the
new manifest's `content_sha256`, which indicates the field was renamed and a
container hash added. If so the token *content* is reproducible and only the
npz packaging varies. Not yet verified; do not lean on it.

Schema v3 also requires the sibling `.manifest.json`. Fetching only the `.npz`
fails with `no manifest at ...`.

### DO NOT `pip install hf`

It pulls `huggingface_hub` 1.x. `transformers` requires `<1.0`, and `mamba_ssm`
imports `transformers`, so the whole stack stops importing and the probe reports
`mamba_ssm: MISSING (ImportError)` with nothing pointing at the real cause.
Recover with `pip install --no-deps 'huggingface_hub==0.36.2'`.
Use `huggingface-cli` for downloads and uploads on the pinned stack.

---

## 1. Probe suite — first thing on the pod

The preflight prints this with the paths filled in. The bare form below is
**not runnable**; four arguments are mandatory and it will STOP at `config`
without them:

* `--hf-repo` — mandatory on any real CUDA run. Pod storage is ephemeral, so
  the probe refuses to start unless evidence can leave the machine.
* `--token-contract` — required by `--all`; G4–G6 are defined by the validated
  contract, not by an invented uniform sequence length.
* `--g2-revision-siso` / `--g2-revision-mimo` — `G2_MODELS` is hardcoded to
  `siso-187m,mimo-187m`, so g2 needs **both** arms cached and pinned even when
  capturing only one.
* `--layers` — g5 measures the hook surface on this list and the capture
  refuses if the two differ. Probe and capture must agree.

The probe resolves checkpoints with `local_files_only=True` deliberately, so it
will not download for you. Cache both arms first (the preflight does).

Seven gates, cheapest first, halts on STOP.

| verdict | meaning |
|---|---|
| all PASS | proceed |
| g3 BOUNDARY | `step()` unavailable. **Proceed.** Prefill work stays valid. Skip C-5 step backend and the state half of C-3. |
| any STOP | stop the pod. Debug before paying more. |

Read `g4.recompiled_on` before the capture: if shape changes retrigger the JIT,
keep `--pad-multiple` large so shapes stay fixed.
Read `g6.est_s_for_stream_b_168k` to decide whether the dense SISO L7-L15 set
survives.

---

## 2. Capture smoke, 187m pair

    python analysis/capture_stage_b.py --model mimo-187m --stream b \
        --token-contract token_contract.npz --out cap_187m_mimo_b.npz
    python analysis/capture_stage_b.py --model siso-187m --stream b \
        --token-contract token_contract.npz --out cap_187m_siso_b.npz

Both arms. If these complete with assertions green, the pipeline works.

---

## 3. The real capture, frozen 1.5b layers

    python analysis/capture_stage_b.py --model mimo-1.5b --stream b \
        --layers 0,1,4,12,13,14,16,17,18,23 --out cap_1p5b_mimo_b.npz
    python analysis/capture_stage_b.py --model siso-1.5b --stream b \
        --layers 0,1,4,7,8,9,10,11,12,13,14,15,23 --out cap_1p5b_siso_b.npz

Then stream A for the long-horizon quantities that prompts cannot support:

    python analysis/capture_stage_b.py --model mimo-1.5b --stream a \
        --layers 0,1,4,12,13,14,16,17,18,23 --out cap_1p5b_mimo_a.npz

---

## 4. L0 direct-logit test

    python analysis/l0_direct_logit.py --model mimo-1.5b --stream a \
        --out l0_mimo_1p5b.npz
    python analysis/l0_direct_logit.py --model siso-1.5b --stream a \
        --out l0_siso_1p5b.npz

This is what decides the detokenizer question. Either answer is publishable.

---

## 5. Analysis (CPU — can be done after the pod dies)

    python analysis/analyze_stage_b.py --capture cap_1p5b_mimo_b.npz \
        --parity-report gpu_probe_report.json \
        --l0-report l0_mimo_1p5b.manifest.json --out ledger_mimo.json

    python analysis/freeze_heads.py --capture cap_1p5b_mimo_b.npz \
        --ledger ledger_mimo.json --out frozen_heads_mimo.json

Claims that fail their gates are moved to `removed_by_gates` with a reason.

---

## 6. Stage C

    python analysis/rank_causality.py --model mimo-1.5b \
        --frozen-heads frozen_heads_mimo.json --out rank_causality.json

    python analysis/retention_probes.py --model mimo-1.5b \
        --parity-report gpu_probe_report.json --out retention.json

    python analysis/mediation.py --siso siso-1.5b --mimo mimo-1.5b \
        --out mediation.json

    # only if g3 PASSed; otherwise use --backend reference
    python analysis/state_trajectories.py --model mimo-1.5b \
        --backend reference --layer 0 --out traj_L0.json

---

## Standing rules

* Every cross-scale figure carries: *one released checkpoint per size and
  architecture; no seed replication; family cross-section, not a fitted scaling
  law.*
* Every SISO/MIMO statement is **bundle-level**. The arms differ in MIMO **and**
  MLP width (exactly 256 narrower at every size) **and** chunk_size (16 vs 64)
  **and** parameter count (+0.16 to +0.27%).
* Rank-differential numbers are **norm ratios**, not energy fractions. The
  weight-side reference is 0.53 (its energy equivalent would be 0.28).
* Never pair SISO head *i* with MIMO head *i*.
* `capture_contrast.py` is kept as history. Do not run it: it hooks `in_proj`
  and expands `ngroups=1` across heads (64 identical copies) and compares Grams
  with the diagonal included, which pins the metric near 1.0 regardless of data.
