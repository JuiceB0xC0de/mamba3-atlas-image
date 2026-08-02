"""B2-5: freeze the Stage C target heads using PRE-DECLARED criteria.

The criteria live in this file and are written BEFORE Stage C results exist.
That is the entire point: selecting heads after seeing intervention outcomes
would make Stage C circular, and "we picked the heads that worked" is not a
finding.

Head index carries no shared meaning across independently trained models
(permutation freedom), so a frozen set is per-checkpoint. Never transplant a
head index from one arm to the other.

CRITERIA, in priority order. Each contributes a fixed quota so no single
criterion dominates the set:

  1. rank_extreme      heads with the highest and lowest activation-side
                       rank-differential use. Tests whether MIMO rank matters
                       causally, which is the question Gram separation cannot
                       answer.
  2. lambda_extreme    heads at the extremes of the trapezoid gate. lambda is
                       the model's own blend between current and next token.
  3. retention_extreme heads with the shortest and longest local half-life,
                       i.e. the memory-span extremes.
  4. feedthrough       heads where the D*x skip path is loudest, since MIMO's
                       one robust static signature is skip amplitude.
  5. class_selective   heads that survived BOTH class nulls, if any did.
  6. controls          heads nearest the median on every axis. Without these
                       every intervention lacks a comparison and dose-response
                       has no baseline.

Usage:
  python analysis/freeze_heads.py --capture capture_1p5b_b.npz \\
      --ledger stage_b_ledger.json --per-criterion 2 --out frozen_heads.json
"""

import argparse
import json
import sys

import numpy as np

CRITERIA_VERSION = "1.0-predeclared-2026-07-31"


def layer_head_grid(d, quantity):
    """Mean over blocks -> (n_layers, n_heads)."""
    key = f"blocks|{quantity}"
    if key not in d.files:
        return None
    return np.nanmean(d[key], axis=0)


def pick_extremes(grid, k, layers):
    """k highest and k lowest cells, returned as (layer, head) pairs."""
    if grid is None:
        return []
    flat = grid.ravel()
    order = np.argsort(np.nan_to_num(flat, nan=np.nanmedian(flat)))
    picks = list(order[:k]) + list(order[-k:])
    return [(int(layers[i // grid.shape[1]]), int(i % grid.shape[1])) for i in picks]


def pick_median(grids, k, layers):
    """Cells closest to the median on every supplied axis, jointly."""
    grids = [g for g in grids if g is not None]
    if not grids:
        return []
    z = np.zeros_like(grids[0])
    for g in grids:
        med = np.nanmedian(g)
        iqr = np.nanpercentile(g, 75) - np.nanpercentile(g, 25)
        z += np.abs((g - med) / (iqr if iqr else 1.0))
    order = np.argsort(np.nan_to_num(z.ravel(), nan=np.inf))
    return [(int(layers[i // z.shape[1]]), int(i % z.shape[1])) for i in order[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--ledger", default=None)
    ap.add_argument("--per-criterion", type=int, default=2)
    ap.add_argument("--out", default="frozen_heads.json")
    args = ap.parse_args()

    d = np.load(args.capture, allow_pickle=False)
    layers = d["layers"] if "layers" in d.files else np.arange(
        d["blocks|lambda"].shape[1])
    k = args.per_criterion

    grids = {
        "rank": layer_head_grid(d, "bc_diff_posbias"),
        "lambda": layer_head_grid(d, "lambda"),
        "retention": layer_head_grid(d, "local_halflife"),
        "feedthrough": layer_head_grid(d, "feedthrough_norm"),
    }

    selected = {
        "rank_extreme": pick_extremes(grids["rank"], k, layers),
        "lambda_extreme": pick_extremes(grids["lambda"], k, layers),
        "retention_extreme": pick_extremes(grids["retention"], k, layers),
        "feedthrough": pick_extremes(grids["feedthrough"], k, layers)[-k:],
        "controls": pick_median(list(grids.values()), k * 2, layers),
    }

    # class-selective heads, only if the ledger says any survived BOTH nulls
    selected["class_selective"] = []
    if args.ledger:
        led = json.load(open(args.ledger))
        survivors = [
            key for key, v in led.get("claims", {}).items()
            if isinstance(v, dict) and v.get("verdict") == "class-effect present"
        ]
        selected["class_selective_note"] = (
            f"{len(survivors)} class claims survived both nulls: {survivors}"
            if survivors else
            "no class claim survived both nulls; criterion contributes nothing"
        )

    flat, roles = [], {}
    for role, cells in selected.items():
        if not isinstance(cells, list):
            continue
        for lh in cells:
            t = tuple(lh)
            if t not in roles:
                roles[t] = role
                flat.append(t)

    out = {
        "criteria_version": CRITERIA_VERSION,
        "declared_before_stage_c": True,
        "capture": args.capture,
        "per_criterion": k,
        "selected_by_role": {r: [list(c) for c in v] if isinstance(v, list) else v
                             for r, v in selected.items()},
        "frozen_heads": [{"layer": l, "head": h, "role": roles[(l, h)]}
                         for (l, h) in flat],
        "n_frozen": len(flat),
        "warning": (
            "Head index is NOT shared across checkpoints. This set is valid only "
            "for the capture it was derived from. Do not transplant to the other arm."
        ),
    }

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"wrote {args.out}: {len(flat)} heads frozen under {CRITERIA_VERSION}")
    for role, cells in selected.items():
        if isinstance(cells, list):
            print(f"  {role:18s} {[tuple(c) for c in cells]}")
        else:
            print(f"  {role:18s} {cells}")


if __name__ == "__main__":
    main()
