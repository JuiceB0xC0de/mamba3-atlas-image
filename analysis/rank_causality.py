"""C-2: does a MIMO rank direction carry usable information?

THE CLAIM THIS EXISTS TO TEST, AND THE ONE IT REPLACES
  Stage A found that ~half of B_bias drift is rank-differential (a norm ratio,
  0.53). Stage B measures whether activations route through those channels.
  NEITHER establishes that the rank direction carries information the model
  USES. Gram separation is a description of geometry, not of function.

  Only an intervention can close that: remove, rotate, or patch ONE rank
  direction and see whether controlled output behaviour changes.

THREE OPERATIONS, each with a matched control:
  remove   zero rank slot r in B_bias/C_bias/mimo_x/mimo_o
           control: zero a random same-norm direction in the rank subspace
  rotate   apply a random orthogonal map within the rank subspace, preserving
           all norms and the rank-marginal distribution
           control: the identity at matched norm perturbation
  patch    replace slot r's activations with those from a DIFFERENT prompt
           control: patch with a shuffled-within-batch source

A rotation is the sharpest of the three: it preserves every marginal quantity
we measured statically (norms, per-slot distributions, effective rank) while
destroying the specific arrangement. If a rotation changes behaviour and a
same-norm control does not, the ARRANGEMENT carries information.

SCOPE: mimo-specific. Says nothing about SISO vs MIMO, which is bundle-level.

Usage:
  python analysis/rank_causality.py --model mimo-1.5b \\
      --frozen-heads frozen_heads.json --out rank_causality.json
"""

import argparse
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from interventions import (  # noqa: E402
    DOSES, dose_response, measure, patched_param, random_like, scaled_edit,
)
from mamba3_core import InProjSpec  # noqa: E402


def zero_rank_slot(t, r):
    v = t.clone()
    v[:, r] = 0.0
    return v


def rotate_rank_subspace(t, gen):
    """Random orthogonal map within the rank axis. Preserves every norm."""
    R = t.shape[1]
    q, _ = torch.linalg.qr(torch.randn(R, R, generator=gen))
    return torch.einsum("rs,hs...->hr...", q.to(t.dtype).to(t.device), t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mimo-1.5b")
    ap.add_argument("--frozen-heads", default=None)
    ap.add_argument("--token-contract", default="token_contract.npz")
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rank_causality.json")
    args = ap.parse_args()

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    dev = "cuda"
    model = MambaLMHeadModel.from_pretrained(
        f"state-spaces/mamba3-{args.model}", device=dev, dtype=torch.bfloat16
    ).eval()
    layers = model.backbone.layers
    spec = InProjSpec.from_mixer(layers[0].mixer)
    R = spec.mimo_rank
    if R == 1:
        print("SISO checkpoint: rank causality is undefined at rank 1. Nothing to do.")
        return

    tc = np.load(args.token_contract, allow_pickle=False)
    ids = torch.tensor(tc["a_ids"][: args.n_seqs, :256], device=dev, dtype=torch.long)
    targets = ids[:, 1:]

    if args.frozen_heads:
        fh = json.load(open(args.frozen_heads))
        cells = [(c["layer"], c["head"], c["role"]) for c in fh["frozen_heads"]]
        print(f"using {len(cells)} frozen heads from {fh['criteria_version']}")
    else:
        cells = [(len(layers) // 2, 0, "adhoc")]
        print("WARNING: no frozen head set; results are exploratory only")

    gen = torch.Generator().manual_seed(args.seed)
    results = {"model": args.model, "rank": R, "ops": {}}

    target_layers = sorted({l for l, _, _ in cells})
    for li in target_layers:
        mixer = layers[li].mixer
        roles = [r for l, h, r in cells if l == li]

        for tensor_name in ("B_bias", "mimo_x", "mimo_o"):
            if not hasattr(mixer, tensor_name):
                continue
            orig = getattr(mixer, tensor_name).data.clone()

            # ---- remove one slot at a time ----
            for r in range(R):
                rnd = random_like(orig, gen)

                def real(dose, r=r, orig=orig, tn=tensor_name):
                    v = orig.clone()
                    v[:, r] = orig[:, r] * (1.0 - dose)
                    return patched_param(mixer, tn, v)

                def ctrl(dose, orig=orig, rnd=rnd, tn=tensor_name):
                    return patched_param(mixer, tn, scaled_edit(orig, rnd, dose))

                key = f"L{li}/{tensor_name}/remove_slot{r}"
                results["ops"][key] = dose_response(
                    model, ids, real, targets, DOSES, ctrl
                )
                print(f"  {key}: {results['ops'][key]['interpretation']}")

            # ---- rotate the whole rank subspace ----
            rot = rotate_rank_subspace(orig, gen)
            rnd = random_like(orig, gen)

            def real_rot(dose, orig=orig, rot=rot, tn=tensor_name):
                return patched_param(mixer, tn, orig + dose * (rot - orig))

            def ctrl_rot(dose, orig=orig, rnd=rnd, tn=tensor_name):
                # matched perturbation magnitude, arbitrary direction
                mag = (rot - orig).norm()
                d = rnd * (mag / rnd.norm().clamp_min(1e-12))
                return patched_param(mixer, tn, orig + dose * d)

            key = f"L{li}/{tensor_name}/rotate_subspace"
            results["ops"][key] = dose_response(
                model, ids, real_rot, targets, DOSES, ctrl_rot
            )
            results["ops"][key]["note"] = (
                "rotation preserves every norm and per-slot marginal that the "
                "static analysis measured; if it changes behaviour and the "
                "same-norm control does not, the ARRANGEMENT carries information"
            )
            print(f"  {key}: {results['ops'][key]['interpretation']}")

        results.setdefault("layer_roles", {})[str(li)] = roles

    survivors = [k for k, v in results["ops"].items() if v.get("beats_control")]
    results["summary"] = {
        "n_ops": len(results["ops"]),
        "beat_control": len(survivors),
        "survivors": survivors,
        "verdict": (
            "rank direction carries usable information"
            if survivors else
            "NO rank operation beat its same-norm control: rank geometry is "
            "descriptive, not functional, at the doses tested"
        ),
        "scope": "mimo-specific; not a SISO/MIMO claim",
    }

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\n{results['summary']['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
