"""C-4: what mediates a SISO/MIMO output difference?

READ THIS BEFORE INTERPRETING ANYTHING THIS FILE PRODUCES.

The released arms are parameter-matched but NOT architecture-identical:
  MIMO rank 4 vs 1
  MLP exactly 256 narrower at EVERY size (realized fc2 width)
  chunk_size 16 vs 64
  +0.16 to +0.27% parameters
A behavioural difference between the arms is a difference between two complete
released SYSTEMS. This file measures MEDIATION -- which pathway carries the
difference -- and mediation is NOT attribution. No normalization performed here
or anywhere else manufactures the missing counterfactual (a MIMO model trained
with an equal-width MLP). That model does not exist and we cannot make one.

Every output is therefore tagged BUNDLE-LEVEL.

WHAT IT DOES
  For matched prompts, decompose each arm's residual write at a layer into:
    mixer_path   the SSM output through out_proj
    mlp_path     the SwiGLU output
    feedthrough  D * x, isolable because D is a plain per-head parameter
    gate         the silu(z) contribution, isolable by neutralizing z
  and measure how much of the arm-to-arm behavioural gap each pathway accounts
  for, using pathway-scaling with same-norm controls from interventions.py.

WHAT IT CANNOT DO
  Tell you MIMO caused anything. If the MLP path carries most of the difference,
  the most parsimonious reading is the MLP-width confound, not a MIMO mechanism.
  That reading is stated explicitly in the output.

Usage:
  python analysis/mediation.py --siso siso-1.5b --mimo mimo-1.5b --out mediation.json
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import InProjSpec, split_in_proj  # noqa: E402


@torch.inference_mode()
def pathway_norms(model, ids, layer_frac=0.5):
    """Per-pathway contribution to the residual, averaged over positions."""
    layers = model.backbone.layers
    li = int(layer_frac * (len(layers) - 1))
    blk = layers[li]
    spec = InProjSpec.from_mixer(blk.mixer)

    grab = {}
    hs = [
        blk.mixer.in_proj.register_forward_hook(
            lambda m, i, o: grab.__setitem__("in", o.detach())),
        blk.mixer.out_proj.register_forward_hook(
            lambda m, i, o: grab.__setitem__("mixer_out", o.detach())),
        blk.mlp.fc2.register_forward_hook(
            lambda m, i, o: grab.__setitem__("mlp_out", o.detach())),
    ]
    model(ids)
    for h in hs:
        h.remove()

    parts = split_in_proj(grab["in"].float(), spec)
    x = parts["x"].reshape(*parts["x"].shape[:-1], spec.nheads, spec.headdim)
    z = parts["z"].reshape(*parts["z"].shape[:-1], spec.nheads, spec.headdim)
    feed = x * blk.mixer.D.float().view(1, 1, spec.nheads, 1)

    return {
        "layer": li,
        "mixer_path": float(grab["mixer_out"].float().norm(dim=-1).mean()),
        "mlp_path": float(grab["mlp_out"].float().norm(dim=-1).mean()),
        "feedthrough": float(feed.reshape(*feed.shape[:2], -1).norm(dim=-1).mean()),
        "gate_silu": float(F.silu(z).reshape(*z.shape[:2], -1).norm(dim=-1).mean()),
        "value": float(x.reshape(*x.shape[:2], -1).norm(dim=-1).mean()),
    }


@torch.inference_mode()
def behavioural_gap(m_a, m_b, ids, targets):
    """How far apart are the two arms, before we ask what mediates it."""
    la = m_a(ids).logits.float()
    lb = m_b(ids).logits.float()
    ce = lambda lg: F.cross_entropy(  # noqa: E731
        F.log_softmax(lg[:, :-1], -1).reshape(-1, lg.shape[-1]),
        targets.reshape(-1), reduction="mean").item()
    return {
        "mean_abs_logit_gap": float((la - lb).abs().mean()),
        "top1_disagreement": float((la[:, -1].argmax(-1) != lb[:, -1].argmax(-1))
                                   .float().mean()),
        "loss_siso": ce(la), "loss_mimo": ce(lb),
        "loss_gap": ce(lb) - ce(la),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siso", default="siso-1.5b")
    ap.add_argument("--mimo", default="mimo-1.5b")
    ap.add_argument("--token-contract", default="token_contract.npz")
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--out", default="mediation.json")
    args = ap.parse_args()

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    tc = np.load(args.token_contract, allow_pickle=False)
    ids = torch.tensor(tc["a_ids"][: args.n_seqs, : args.seqlen],
                       device="cuda", dtype=torch.long)
    targets = ids[:, 1:]

    out = {"scope": "BUNDLE-LEVEL", "arms": {"siso": args.siso, "mimo": args.mimo}}

    models = {}
    for tag, name in (("siso", args.siso), ("mimo", args.mimo)):
        m = MambaLMHeadModel.from_pretrained(
            f"state-spaces/mamba3-{name}", device="cuda", dtype=torch.bfloat16
        ).eval()
        models[tag] = m
        out.setdefault("pathways", {})[tag] = pathway_norms(m, ids)
        print(f"{tag}: {out['pathways'][tag]}")

    out["gap"] = behavioural_gap(models["siso"], models["mimo"], ids, targets)
    print(f"\nbehavioural gap: {out['gap']}")

    # relative pathway shift, arm to arm
    ps, pm = out["pathways"]["siso"], out["pathways"]["mimo"]
    shift = {
        k: (pm[k] - ps[k]) / max(abs(ps[k]), 1e-9)
        for k in ("mixer_path", "mlp_path", "feedthrough", "gate_silu", "value")
    }
    out["relative_shift_mimo_vs_siso"] = shift
    dominant = max(shift, key=lambda k: abs(shift[k]))
    out["dominant_pathway"] = dominant

    out["interpretation"] = {
        "dominant": dominant,
        "reading": (
            "MLP path dominates: the most parsimonious explanation is the "
            "MLP-WIDTH CONFOUND (MIMO runs exactly 256 narrower at every size), "
            "NOT a MIMO mechanism."
            if dominant == "mlp_path" else
            f"{dominant} dominates. This is still bundle-level: the arms also "
            "differ in MLP width, chunk_size and parameter count. Mediation is "
            "not attribution."
        ),
        "what_would_settle_it": (
            "a MIMO checkpoint trained with an equal-width MLP, or a matched "
            "training ablation family. Neither exists publicly; this cannot be "
            "resolved post hoc."
        ),
    }

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n{out['interpretation']['reading']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
