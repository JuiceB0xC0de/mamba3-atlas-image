"""B2-4: turn a capture into the Stage B ledger, with the falsification gates enforced.

CPU only. Consumes capture_stage_b.npz (+ manifest) and emits every claim TAGGED
BY SCOPE, because the released arms are not an isolated MIMO ablation.

SCOPE TIERS (contract 0), applied to every claim automatically:
  mamba3-wide   holds in both arms across sizes
  bundle-level  a SISO/MIMO difference. The arms differ in MIMO *and* MLP width
                (exactly 256 narrower at every size) *and* chunk_size (16 vs 64)
                *and* parameter count (+0.16 to +0.27%). Never call this
                "MIMO causes X".
  mimo-specific within-MIMO structure that SISO cannot represent at all
                (rank-differential quantities)
  unsupported   failed its null, or its interval includes zero

GATES (contract 6). A claim that trips one is REMOVED, not footnoted:
  * no state interpretation if the parity gate did not pass
  * no class-mechanism claim unless it beats BOTH the same-class split and the
    label permutation
  * no class claim at all if length-match coverage is too low
  * no rank-information claim from Gram separation alone (needs Stage C)
  * no detokenizer claim without the L0 direct-logit test
  * no stable-atlas claim if 150k and 500k disagree on layer rankings

Usage:
  python analysis/analyze_stage_b.py --capture capture_187m_b.npz \\
      --out stage_b_ledger.json
  python analysis/analyze_stage_b.py --capture cap_500k.npz --compare cap_150k.npz
"""

import argparse
import hashlib
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from nulls import (  # noqa: E402
    block_bootstrap, ci, permutation_null, same_class_split_null, stratified_contrast,
)

MIN_COVERAGE = 0.35   # below this, the classes are not length-comparable
ALPHA = 0.05


def bh_fdr(p_values):
    """Benjamini-Hochberg q-values for one predeclared tested family."""
    p = np.asarray(p_values, dtype=float)
    q = np.ones_like(p)
    finite = np.isfinite(p)
    pv = p[finite]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adjusted = ranked * pv.size / np.arange(1, pv.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = restored
    return q


def file_sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    d = np.load(path, allow_pickle=False)
    man = {}
    try:
        with open(path.replace(".npz", ".manifest.json")) as fh:
            man = json.load(fh)
    except FileNotFoundError:
        # Capture Final stores the authoritative manifest inside the atomic NPZ;
        # the W&B/HF wrapper may not leave a same-directory sidecar behind.
        if "manifest_json" in d.files:
            raw = d["manifest_json"]
            man = json.loads(str(raw.item() if raw.shape == () else raw[0]))
    return d, man


def class_contrast(d, quantity, a, b, lengths=None, n_perm=2000):
    """One class contrast, with BOTH required nulls and length matching."""
    rows = d[f"blocks|{quantity}"]
    labels = d["block_semantic_label"]
    if not ((labels == a).any() and (labels == b).any()):
        return None

    obs, _, p_perm = permutation_null(rows, labels, a, b, n_perm=n_perm)
    split_a = same_class_split_null(rows, labels, a, n_split=n_perm)
    split_b = same_class_split_null(rows, labels, b, n_split=n_perm)
    # Require the effect to beat sampling variation within BOTH classes.
    floor = np.maximum(
        np.quantile(np.abs(split_a), 0.95, axis=0),
        np.quantile(np.abs(split_b), 0.95, axis=0),
    )
    beats_floor = np.abs(obs) > floor
    boot = (block_bootstrap(rows[labels == a], n_boot=1000)
            - block_bootstrap(rows[labels == b], n_boot=1000))
    effect_lo, effect_hi = ci(boot)

    result = {
        "quantity": quantity, "classes": [a, b],
        "n_a": int((labels == a).sum()), "n_b": int((labels == b).sum()),
        "max_abs_effect": float(np.abs(obs).max()),
        "cells_raw_p_lt_alpha": int((p_perm < ALPHA).sum()),
        "cells_beating_same_class_floor": int(beats_floor.sum()),
        "cells_passing_BOTH": int(((p_perm < ALPHA) & beats_floor).sum()),
        "n_cells": int(obs.size),
    }

    if lengths is not None:
        obs_s, _, p_s, cov = stratified_contrast(rows, labels, lengths, a, b,
                                                 n_perm=n_perm)
        result |= {
            "length_matched": True,
            "coverage": float(cov),
            "cells_raw_p_lt_alpha_matched": int((np.nan_to_num(p_s, nan=1.0) < ALPHA).sum()),
            "cells_passing_BOTH_matched": int((
                (np.nan_to_num(p_s, nan=1.0) < ALPHA) & beats_floor).sum()),
            "max_abs_effect_matched": float(np.nanmax(np.abs(obs_s))),
        }
        if cov < MIN_COVERAGE:
            result["verdict"] = "unsupported"
            result["reason"] = (
                f"length-match coverage {cov:.0%} < {MIN_COVERAGE:.0%}: the classes "
                "are not length-comparable, so any difference is confounded"
            )
            return result

        tested_p = np.nan_to_num(p_s, nan=1.0)
        tested_effect = obs_s
    else:
        tested_p = np.nan_to_num(p_perm, nan=1.0)
        tested_effect = obs

    # Kept private until the complete contrast x quantity x layer x head family
    # receives one BH correction in main().
    result["_cell_data"] = {
        "p": tested_p,
        "effect": tested_effect,
        "effect_lo": effect_lo,
        "effect_hi": effect_hi,
        "same_class_floor": floor,
        "beats_floor": beats_floor,
    }
    result["verdict"] = "pending multiplicity correction"
    result["scope"] = "within-arm; says nothing about MIMO vs SISO"
    return result


def rank_utilization(d, man):
    """Activation-side rank-differential use, against the weight-side ~0.53.

    Closes the loop on the retracted Stage A claim. Weight side commits about
    half its B_bias DRIFT to rank-differential structure (a NORM RATIO, 0.53;
    the energy fraction would be 0.28). If activations route through those
    channels at a far lower rate, the retraction overshot.
    """
    out = {}
    for key in d.files:
        if (not key.startswith("blocks|")
                or key.split("|", 1)[1] not in {
                    "B_diff_shared", "B_diff_post",
                    "C_diff_shared", "C_diff_post"}):
            continue
        q = key.split("|")[1]
        rows = d[key]
        boot = block_bootstrap(rows, n_boot=1000)
        lo, hi = ci(boot)
        out[q] = {
            "mean_norm_ratio": float(np.nanmean(rows)),
            "ci_lo": float(np.nanmean(lo)), "ci_hi": float(np.nanmean(hi)),
            "weight_side_reference": 0.53,
            "units": "NORM RATIO (unsquared). Do not compare to an energy fraction.",
            "scope": "mimo-specific" if man.get("is_mimo") else "n/a (SISO rank 1)",
        }
    return out


def convergence(d_small, d_large, quantity="lambda"):
    """Do 150k and 500k agree on LAYER RANKINGS? A gate, not a nicety."""
    from scipy import stats

    a = np.nanmean(d_small[f"blocks|{quantity}"], axis=(0, 2))
    b = np.nanmean(d_large[f"blocks|{quantity}"], axis=(0, 2))
    n = min(len(a), len(b))
    rho = stats.spearmanr(a[:n], b[:n]).statistic
    return {
        "quantity": quantity, "layer_rank_spearman": float(rho),
        "converged": bool(rho > 0.9),
        "gate": "no stable-atlas claim if layer rankings disagree",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--compare", default=None, help="a smaller capture, for the convergence gate")
    ap.add_argument("--parity-report", default=None, help="gpu_probe_report.json")
    ap.add_argument("--l0-report", default=None, help="l0_direct_logit.manifest.json")
    ap.add_argument("--pairs", default="compliance/pos:compliance/neg,code/pos:code/neg")
    ap.add_argument("--out", default="stage_b_ledger.json")
    ap.add_argument("--wandb-project", default="mamba3-mimo-atlas")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
                    default="online")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    capture_sha = file_sha256(args.capture)
    run = None
    wandb = None
    if args.wandb_mode != "disabled":
        import wandb as _wandb
        wandb = _wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="stage-b-analysis",
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config={
                "capture_sha256": capture_sha,
                "capture_name": args.capture.rsplit("/", 1)[-1],
                "alpha": ALPHA,
                "multiplicity": "BH-FDR over all contrast x quantity x layer x head cells",
                "minimum_length_match_coverage": MIN_COVERAGE,
                "state_claims_default": "blocked",
            },
        )
        print(f"W&B LIVE: {run.url}", flush=True)
    d, man = load(args.capture)
    ledger = {
        "capture": args.capture,
        "capture_sha256": capture_sha,
        "manifest": man,
        "claims": {},
        "gates": {},
        "multiplicity": {
            "method": "Benjamini-Hochberg FDR",
            "alpha": ALPHA,
            "family": "all requested contrast x quantity x layer x head cells",
        },
    }

    # ---- gate: parity ----
    if args.parity_report:
        rep = json.load(open(args.parity_report))
        v = rep.get("verdicts", {})
        ledger["gates"]["parity"] = v
        ledger["gates"]["state_claims_allowed"] = (
            v.get("g2") == "PASS" and v.get("g3") == "PASS"
        )
        if v.get("g3") == "BOUNDARY":
            ledger["gates"]["note"] = (
                "step() unavailable: prefill analysis valid, decode-state/h_t "
                "claims and state-based Stage C (C-3, C-5) PROHIBITED"
            )
    else:
        ledger["gates"]["state_claims_allowed"] = False
        ledger["gates"]["note"] = "no parity report supplied; state claims blocked"

    # ---- gate: detokenizer ----
    ledger["gates"]["detokenizer_claim_allowed"] = bool(args.l0_report)
    if args.l0_report:
        ledger["claims"]["l0_direct_logit"] = json.load(open(args.l0_report)).get("verdict")

    # ---- class contrasts ----
    lengths = d["block_n_used"].astype(float) if "block_n_used" in d.files else None
    for spec in args.pairs.split(","):
        a, b = spec.split(":")
        for q in ("lambda", "alpha", "Delta", "trap_scale",
                  "B_diff_post", "C_diff_post",
                  "B_participation_post", "C_participation_post"):
            if f"blocks|{q}" not in d.files:
                continue
            r = class_contrast(d, q, a, b, lengths)
            if r:
                ledger["claims"][f"contrast/{a}_vs_{b}/{q}"] = r
                if run is not None:
                    run.log({"analysis/contrasts_computed": len(ledger["claims"])})

    # One correction across the complete predeclared family. Every tested cell,
    # including failures, remains in the ledger.
    refs, pvals = [], []
    layer_ids = d["layers"].astype(int) if "layers" in d.files else None
    for claim_id, claim in ledger["claims"].items():
        if not isinstance(claim, dict) or "_cell_data" not in claim:
            continue
        cell = claim["_cell_data"]
        for li, hi in np.ndindex(cell["p"].shape):
            refs.append((claim_id, li, hi))
            pvals.append(float(cell["p"][li, hi]))
    qvals = bh_fdr(pvals)
    for (claim_id, li, hi), q in zip(refs, qvals):
        claim = ledger["claims"][claim_id]
        cell = claim["_cell_data"]
        claim.setdefault("cells", []).append({
            "layer": int(layer_ids[li]) if layer_ids is not None else int(li),
            "head": int(hi),
            "effect": float(cell["effect"][li, hi]),
            "effect_ci_lo": float(cell["effect_lo"][li, hi]),
            "effect_ci_hi": float(cell["effect_hi"][li, hi]),
            "permutation_p": float(cell["p"][li, hi]),
            "fdr_q": float(q),
            "same_class_floor": float(cell["same_class_floor"][li, hi]),
            "beats_both_same_class_floors": bool(cell["beats_floor"][li, hi]),
            "supported": bool(q < ALPHA and cell["beats_floor"][li, hi]),
        })
    for claim in ledger["claims"].values():
        if not isinstance(claim, dict) or "_cell_data" not in claim:
            continue
        claim.pop("_cell_data")
        n_supported = sum(c["supported"] for c in claim["cells"])
        claim["cells_fdr_and_same_class_supported"] = int(n_supported)
        claim["verdict"] = "supported" if n_supported else "unsupported"
        if not n_supported:
            claim["reason"] = "no cell passes global BH-FDR and both same-class floors"

    # ---- rank utilization vs the weight-side reference ----
    if any(d[k].shape[-1] > 1 for k in d.files if k.startswith("blockrank|")):
        ledger["claims"]["rank_utilization"] = {
            "status": "descriptive_only",
            "verdict": "not causal",
            "measurements": rank_utilization(d, man),
            "reason": "rank geometry is not causal utilization without a valid intervention/null",
        }
    else:
        ledger["claims"]["rank_utilization"] = {
            "status": "not_applicable",
            "verdict": "unsupported",
            "reason": "SISO has rank 1 and no rank-differential subspace",
        }

    # ---- convergence gate ----
    if args.compare:
        d2, _ = load(args.compare)
        ledger["gates"]["convergence"] = convergence(d2, d)

    # Negative and blocked results remain first-class evidence; none are deleted.
    ledger["removed_by_gates"] = {}

    if args.out != "-":
        with open(args.out, "w") as fh:
            json.dump(ledger, fh, indent=2, default=str)

    claim_rows = []
    for claim_id, claim in ledger["claims"].items():
        if isinstance(claim, dict):
            claim_rows.append([
                claim_id, claim.get("verdict"), claim.get("scope", "n/a"),
                claim.get("cells_fdr_and_same_class_supported", 0),
                claim.get("reason", ""),
            ])
    if run is not None:
        run.log({"evidence_ledger": wandb.Table(
            columns=["claim_id", "verdict", "scope", "supported_cells", "reason"],
            data=claim_rows,
        )})
        if args.out != "-":
            artifact = wandb.Artifact("stage-b-siso-ledger", type="analysis",
                                      metadata={"capture_sha256": capture_sha})
            artifact.add_file(args.out)
            run.log_artifact(artifact)
        run.summary["analysis/claims_total"] = len(ledger["claims"])
        run.summary["analysis/claims_supported"] = sum(
            isinstance(v, dict) and v.get("verdict") == "supported"
            for v in ledger["claims"].values())
        run.summary["analysis/state_claims_allowed"] = ledger["gates"]["state_claims_allowed"]
        run.finish()

    print("ANALYSIS_SUMMARY_JSON=" + json.dumps({
        "capture_sha256": capture_sha,
        "claims": [dict(zip(
            ["claim_id", "verdict", "scope", "supported_cells", "reason"], row
        )) for row in claim_rows],
        "claims_total": len(ledger["claims"]),
        "claims_supported": sum(
            isinstance(v, dict) and v.get("verdict") == "supported"
            for v in ledger["claims"].values()),
        "state_claims_allowed": ledger["gates"]["state_claims_allowed"],
    }, default=str), flush=True)

    print(f"wrote {args.out}")
    print(f"  claims recorded : {len(ledger['claims'])} (negative results retained)")
    print(f"  state claims allowed: {ledger['gates']['state_claims_allowed']}")
    print("\nSTANDING CAVEAT for every cross-scale figure:")
    print("  one released checkpoint per size and architecture; no seed replication;")
    print("  family cross-section, not a fitted scaling law.")


if __name__ == "__main__":
    main()
