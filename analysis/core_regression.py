"""B0-6: prove mamba3_core reproduces the FROZEN Stage A artifact.

Stage A ran with its own implementations inside static_atlas.py. Stage B uses
mamba3_core. If those two ever disagree, every cross-stage comparison is void
and the ~0.5 rank-differential result cannot be checked against activations.

static_atlas.py is deliberately NOT refactored -- it already ran, and rewriting
a finished artifact to match new code is how you launder a discrepancy. This
script recomputes from the raw checkpoints with core and compares against
static_atlas_all8.npz, which is READ-ONLY here. Nothing regenerates or modifies
a Stage A output. Any disagreement is a stop, not a warning.

Stage 2D migration: checkpoint resolution and shape derivation now come from
mamba3_core (resolve_checkpoint, InProjSpec.from_state_dict). The compared
quantities, their tolerance (2e-5), and the Stage A artifact are UNCHANGED. No
new metrics; no reinterpretation of Stage A.

One deliberate tightening: the drift share is now computed by core's
`differential_share_of_drift` rather than an inline copy of the same expression.
It is the same quantity -- ||X - mean_r(X)|| / ||X - init|| -- but computing it
through the canonical function means this regression actually exercises core,
which is the point of the file.

COMPARED FIELDS (four, per layer, per checkpoint):
  dt                    softplus(dt_bias)          vs  <tag>|dt
  prior_halflife        ln2 / softplus(dt_bias)    vs  <tag>|prior_halflife
  B_bias/diff_energy    differential NORM RATIO    vs  <tag>|B_bias/diff_energy
  drift_share           ||diff|| / ||W - init||    vs  de*sc/dr from the artifact

  The third is a norm ratio despite the artifact's field name; see the units
  correction recorded with the Stage A findings.

Overall PASS requires ALL EIGHT released checkpoints and EVERY expected
comparison. Missing models, incomplete artifacts, skipped fields, shape
disagreement, incomplete provenance, or partial coverage exit nonzero.

Resolution is LOCAL-ONLY; this never downloads.

Run:  python analysis/core_regression.py
"""

import argparse
import json
import math
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import (  # noqa: E402
    CheckpointResolveError, InProjSpec, ShapeContractError,
    differential_norm_ratio, differential_share_of_drift, recurrence_quantities,
    resolve_checkpoint,
)

NPZ = "static_atlas_all8.npz"
TOL = 2e-5
PASS, STOP = "PASS", "STOP"

TAGS = (
    "siso-187m", "siso-443m", "siso-893m", "siso-1.5b",
    "mimo-187m", "mimo-444m", "mimo-894m", "mimo-1.5b",
)

# field -> artifact key suffix. Every one of these must be present and compared.
COMPARED_FIELDS = ("dt", "prior_halflife", "B_bias/diff_energy", "drift_share")
DERIVED_KEYS = ("B_bias/diff_energy", "B_bias/scale", "B_bias/drift")


def resolve_arm(spec, ssm, where=""):
    """Realized arm is authoritative; ssm_cfg.is_mimo is a CROSS-CHECK only."""
    if ssm is not None and "is_mimo" in ssm:
        cfg_arm = bool(ssm["is_mimo"])
        if cfg_arm != spec.is_mimo:
            raise ShapeContractError(
                f"arm disagreement{(' at ' + where) if where else ''}: config "
                f"ssm_cfg.is_mimo={cfg_arm} but realized mimo_rank="
                f"{spec.mimo_rank} implies is_mimo={spec.is_mimo}")
    return spec.is_mimo


def recompute(sd, cfg, n_layer, tag):
    """Core's values for every layer. Spec is derived AT EVERY LAYER."""
    out = {k: [] for k in ("dt", "prior_halflife", "B_bias/diff_energy",
                           "drift_share")}
    arms = set()
    for li in range(n_layer):
        spec = InProjSpec.from_state_dict(sd, cfg, layer=li)
        arms.add(resolve_arm(spec, cfg.get("ssm_cfg"), f"{tag} L{li}"))

        mx = f"backbone.layers.{li}.mixer."
        dt_bias = sd[mx + "dt_bias"].float()
        B_bias = sd[mx + "B_bias"].float()

        if tuple(dt_bias.shape) != (spec.nheads,):
            raise ShapeContractError(
                f"{tag} L{li} dt_bias: expected {(spec.nheads,)}, "
                f"got {tuple(dt_bias.shape)}")
        exp = (spec.nheads, spec.mimo_rank, spec.d_state)
        if tuple(B_bias.shape) != exp:
            raise ShapeContractError(
                f"{tag} L{li} B_bias: expected {exp}, got {tuple(B_bias.shape)}")

        # zero-input prior through the full recurrence path, not a shortcut:
        # heavy_tail_activation(0) == 1 => A == -1 exactly
        nh = spec.nheads
        q = recurrence_quantities(torch.zeros(1, nh), torch.zeros(1, nh),
                                  torch.zeros(1, nh), dt_bias)
        out["dt"].append(q["Delta"][0].numpy())
        out["prior_halflife"].append(q["local_halflife"][0].numpy())
        out["B_bias/diff_energy"].append(
            differential_norm_ratio(B_bias, rank_dim=-2).numpy())
        out["drift_share"].append(
            differential_share_of_drift(B_bias, init_val=1.0, rank_dim=-2).numpy())

    if len(arms) != 1:
        raise ShapeContractError(f"{tag}: layers disagree on arm: {arms}")
    return {k: np.stack(v) for k, v in out.items()}, arms.pop()


def artifact_values(d, full, n_layer):
    """Pull the frozen Stage A values. Missing keys are reported, never skipped."""
    missing, vals = [], {}
    for key in ("dt", "prior_halflife"):
        k = f"{full}|{key}"
        if k not in d:
            missing.append(k)
        else:
            vals[key] = d[k]

    for k in (f"{full}|{x}" for x in DERIVED_KEYS):
        if k not in d:
            missing.append(k)
    if not missing:
        de = d[f"{full}|B_bias/diff_energy"]
        sc = d[f"{full}|B_bias/scale"]
        dr = d[f"{full}|B_bias/drift"]
        vals["B_bias/diff_energy"] = de
        # Stage A derived the drift share as diff_energy * scale / drift, all
        # unsquared norm ratios. Reproduced here exactly; NOT recomputed.
        vals["drift_share"] = de * sc / dr
    return vals, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--models", default=",".join(TAGS))
    ap.add_argument("--revision", default=None,
                    help="exact checkpoint revision; resolution stays local-only")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--out", default="core_regression_report.json")
    a = ap.parse_args()

    requested = a.models.split(",")
    report = {
        "artifact": a.npz, "tolerance": a.tol,
        "compared_fields": list(COMPARED_FIELDS),
        "expected_checkpoints": list(TAGS),
        "requested_checkpoints": requested,
        "checkpoints": {}, "failures": [], "verdict": PASS,
    }

    try:
        d = np.load(a.npz)
    except Exception as e:  # noqa: BLE001
        print(f"[{STOP}] cannot read frozen artifact {a.npz}: {e}")
        report["verdict"] = STOP
        report["failures"].append(f"artifact unreadable: {e}")
        with open(a.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        return 1

    if set(requested) != set(TAGS):
        report["failures"].append(
            f"coverage: requested {sorted(requested)} != all eight {sorted(TAGS)}")
        report["verdict"] = STOP

    print(f"{'checkpoint':14s} {'arm':5s} {'layers':>6s} "
          + " ".join(f"{f:>20s}" for f in COMPARED_FIELDS))

    n_comparisons = 0
    for tag in requested:
        full = f"mamba3-{tag}"
        entry = {"expected_fields": list(COMPARED_FIELDS)}
        try:
            ck = resolve_checkpoint(tag, revision=a.revision, local_only=True)
            prov = ck.provenance()
            if not prov.get("path") or not prov.get("weights_file"):
                raise CheckpointResolveError(f"incomplete provenance {prov}")
            if prov.get("resolved_commit") is None and not ck.from_local_dir:
                raise CheckpointResolveError(
                    "resolved_commit is None for a hub-resolved snapshot")
            entry["provenance"] = prov

            cfg = ck.load_config()
            n_layer = int(cfg["n_layer"])
            sd = ck.load_state_dict()
            core_vals, is_mimo = recompute(sd, cfg, n_layer, tag)
            del sd
        except (CheckpointResolveError, ShapeContractError, KeyError) as e:
            msg = f"{tag}: {type(e).__name__}: {e}"
            print(f"{tag:14s} {STOP}  {msg}")
            entry["verdict"] = STOP
            entry["error"] = str(e)
            report["checkpoints"][tag] = entry
            report["failures"].append(msg)
            report["verdict"] = STOP
            continue

        art, missing = artifact_values(d, full, n_layer)
        entry |= {"arm": "MIMO" if is_mimo else "SISO", "n_layer": n_layer,
                  "missing_fields": missing, "comparisons": {}}
        if missing:
            msg = f"{tag}: artifact missing {missing}"
            report["failures"].append(msg)
            report["verdict"] = STOP
            entry["verdict"] = STOP

        cells = []
        for f in COMPARED_FIELDS:
            if f not in art:
                entry["comparisons"][f] = {"status": "MISSING"}
                cells.append(f"{'MISSING':>20s}")
                continue
            a_arr, b_arr = core_vals[f], art[f]
            if a_arr.shape != b_arr.shape:
                msg = (f"{tag} {f}: shape {a_arr.shape} vs artifact {b_arr.shape}")
                entry["comparisons"][f] = {"status": "SHAPE", "core": list(a_arr.shape),
                                           "artifact": list(b_arr.shape)}
                report["failures"].append(msg)
                report["verdict"] = STOP
                entry["verdict"] = STOP
                cells.append(f"{'SHAPE':>20s}")
                continue
            err = float(np.abs(a_arr - b_arr).max())
            ok = math.isfinite(err) and err < a.tol
            entry["comparisons"][f] = {"status": "ok" if ok else "FAIL",
                                       "max_abs_err": err, "tolerance": a.tol,
                                       "shape": list(a_arr.shape)}
            n_comparisons += 1
            if not ok:
                report["failures"].append(
                    f"{tag} {f}: max abs err {err:.3e} >= {a.tol}")
                report["verdict"] = STOP
                entry["verdict"] = STOP
            cells.append(f"{err:20.2e}")

        entry.setdefault("verdict", PASS)
        report["checkpoints"][tag] = entry
        print(f"{tag:14s} {entry['arm']:5s} {n_layer:6d} " + " ".join(cells))

    expected_comparisons = len(TAGS) * len(COMPARED_FIELDS)
    report["coverage"] = {
        "checkpoints_examined": len(report["checkpoints"]),
        "checkpoints_expected": len(TAGS),
        "comparisons_performed": n_comparisons,
        "comparisons_expected": expected_comparisons,
        "complete": (n_comparisons == expected_comparisons
                     and len(report["checkpoints"]) == len(TAGS)),
    }
    if not report["coverage"]["complete"]:
        report["failures"].append(
            f"coverage incomplete: {n_comparisons}/{expected_comparisons} "
            f"comparisons over {len(report['checkpoints'])}/{len(TAGS)} checkpoints")
        report["verdict"] = STOP

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\ncoverage: {n_comparisons}/{expected_comparisons} comparisons over "
          f"{len(report['checkpoints'])}/{len(TAGS)} checkpoints")
    if report["failures"]:
        print("\nFAILURES:")
        for f in report["failures"]:
            print("  " + f)
    print(f"\nverdict: {report['verdict']}")
    if report["verdict"] == PASS:
        print("core reproduces the frozen Stage A artifact within tolerance")
    print(f"wrote {a.out}")
    return 0 if report["verdict"] == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
