"""C-5: SSM state trajectories. h_t norms, effective rank, reset events.

HARD GATE: this file requires state access and does nothing useful without it.
The SSM state lives inside the kernel and is NOT observable from a forward hook.
Two ways to get it, in preference order:

  1. step() decode path, if gpu_probe G3 returned PASS. Gives the model's own
     state, exactly.
  2. the CPU reference recurrence (reference_recurrence.py), which computes h_t
     explicitly. Slower and float32, but it reproduced the official kernel's
     top-4 predictions, so it is a legitimate instrument for state SHAPE
     questions even when step() is unavailable.

If G3 returned BOUNDARY, option 1 is prohibited and option 2 must be labeled as
reference-derived, not measured from the deployed kernel. The distinction goes
in every figure caption.

WHAT THIS MEASURES
  h_norm            state magnitude over position. Growth means accumulation,
                    decay means forgetting, plateau means steady state.
  state_eff_rank    participation ratio of the state's singular values. A
                    fixed-size state that uses few directions is compressing
                    hard; one that uses many is spreading thin.
  reset_events      positions where ||h_t|| drops sharply relative to its
                    running level. In an SSM these are the closest thing to a
                    segment boundary, and whether they align with document or
                    sentence boundaries is a real question.
  per_rank_energy   for MIMO, how state energy distributes across the rank
                    channels. The activation-side companion to the static ~0.53
                    rank-differential norm ratio.

Usage:
  python analysis/state_trajectories.py --model mimo-1.5b --backend reference
  python analysis/state_trajectories.py --model mimo-1.5b --backend step \\
      --parity-report gpu_probe_report.json
"""

import argparse
import glob
import json
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import InProjSpec, rms_norm  # noqa: E402
from reference_recurrence import RefSwitches, reference_block_forward  # noqa: E402

CACHE = "/Users/chiggy/.cache/huggingface/hub/models--state-spaces--mamba3-{}/snapshots/*"


def eff_rank(m):
    """Participation ratio of singular values, in [1, min(shape)]."""
    s = torch.linalg.svdvals(m.float())
    return float(s.sum() ** 2 / (s**2).sum().clamp_min(1e-24))


def reset_events(norms, window=16, drop=0.5):
    """Positions where the state norm falls below `drop` of its running median."""
    n = norms.numpy() if torch.is_tensor(norms) else np.asarray(norms)
    out = []
    for t in range(window, len(n)):
        ref = np.median(n[t - window:t])
        if ref > 0 and n[t] < drop * ref:
            out.append(int(t))
    return out


def trajectories_via_reference(model_name, ids, layer):
    """Compute h_t explicitly with the CPU reference. Works without step()."""
    snap = glob.glob(CACHE.format(model_name))[0]
    cfg = json.load(open(f"{snap}/config.json"))
    sd = torch.load(f"{snap}/pytorch_model.bin", map_location="cpu", weights_only=True)
    ssm = cfg["ssm_cfg"]
    d_model = cfg["d_model"]

    H = sd["backbone.layers.0.mixer.dt_bias"].shape[0]
    R = sd["backbone.layers.0.mixer.B_bias"].shape[1]
    N = sd["backbone.layers.0.mixer.B_bias"].shape[-1]
    P = ssm["headdim"]
    spec = InProjSpec(
        d_inner=int(ssm["expand"]) * d_model, d_state=N, ngroups=ssm["ngroups"],
        mimo_rank=R, nheads=H, headdim=P,
        n_rope_angles=N // int(2 / float(ssm["rope_fraction"])),
    )

    # run the stack up to `layer`, then capture that block's state trace
    x = sd["backbone.embedding.weight"].float()[ids]
    traj = None
    for li in range(cfg["n_layer"]):
        p, mx = f"backbone.layers.{li}.", f"backbone.layers.{li}.mixer."
        params = {k.split("mixer.")[-1]: sd[k].float() for k in sd if k.startswith(mx)}
        h = rms_norm(x, sd[p + "norm.weight"].float())
        r = reference_block_forward(h, params, spec, bool(ssm.get("is_mimo")),
                                    float(ssm.get("A_floor", 1e-4)), RefSwitches())
        if li == layer:
            traj = r
        x = x + r["out"]
        h2 = rms_norm(x, sd[p + "norm2.weight"].float())
        hh = h2 @ sd[p + "mlp.fc1.weight"].float().T
        a, b = hh.chunk(2, dim=-1)
        x = x + (torch.nn.functional.silu(b) * a) @ sd[p + "mlp.fc2.weight"].float().T
        if li == layer:
            break
    del sd
    return traj, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mimo-1.5b")
    ap.add_argument("--backend", choices=["reference", "step"], default="reference")
    ap.add_argument("--parity-report", default=None)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--token-contract", default="token_contract.npz")
    ap.add_argument("--seqlen", type=int, default=256)
    ap.add_argument("--out", default="state_trajectories.json")
    args = ap.parse_args()

    provenance = "reference-derived (CPU float32), NOT measured from the deployed kernel"
    if args.backend == "step":
        if not args.parity_report:
            print("--backend step requires --parity-report; refusing to guess")
            return
        v = json.load(open(args.parity_report)).get("verdicts", {})
        if v.get("g3") != "PASS":
            print(f"g3={v.get('g3')}: step() state access is PROHIBITED by the "
                  "contract. Re-run with --backend reference.")
            return
        provenance = "measured via step() decode path"

    tc = np.load(args.token_contract, allow_pickle=False)
    if "a_ids" in tc.files:
        ids = torch.tensor(tc["a_ids"][0, : args.seqlen], dtype=torch.long)
    else:
        # stream-B fallback: concatenate prompts to reach the requested length.
        # Trajectories need a LONG sequence; single prompts (median 15 tokens)
        # cannot show growth, plateau, or resets. Flagged in the output.
        print("no stream A in the contract; concatenating stream-B prompts. "
              "Retention/plateau structure from concatenated prompts is NOT "
              "comparable to natural text.")
        flat, offs = tc["b_ids"], tc["b_offsets"]
        ids = torch.tensor(flat[: args.seqlen], dtype=torch.long)
        provenance += " [stream-B concatenation, not natural text]"

    if args.backend == "reference":
        traj, spec = trajectories_via_reference(args.model, ids, args.layer)
    else:
        raise NotImplementedError(
            "step()-backed trajectory capture is written against the CuteDSL "
            "cache API; wire it only after g3 PASSes so the shape contract is known"
        )

    norms = traj["state_norms"]            # (T, H)
    per_head_mean = norms.mean(0)
    resets = reset_events(norms.mean(-1))

    # state effective rank at a few checkpoints along the sequence
    ranks = {}
    st = traj["state_final"]               # (H, P, N) at the final position
    ranks["final_mean_eff_rank"] = float(np.mean([eff_rank(st[h]) for h in range(st.shape[0])]))

    out = {
        "model": args.model, "layer": args.layer, "provenance": provenance,
        "seqlen": int(args.seqlen),
        "h_norm_mean_over_heads": [float(v) for v in norms.mean(-1)],
        "h_norm_per_head_mean": [float(v) for v in per_head_mean],
        "growth_ratio_end_over_start": float(
            norms.mean(-1)[-1] / max(float(norms.mean(-1)[8]), 1e-9)),
        "reset_positions": resets,
        "n_resets": len(resets),
        **ranks,
        "caveat": ("one sequence, one layer. State SHAPE questions only; this is "
                   "not a corpus statistic."),
    }

    print(f"provenance: {provenance}")
    print(f"  h_norm start->end: {out['h_norm_mean_over_heads'][8]:.3f} -> "
          f"{out['h_norm_mean_over_heads'][-1]:.3f} "
          f"(x{out['growth_ratio_end_over_start']:.2f})")
    print(f"  state eff_rank at final position: {ranks['final_mean_eff_rank']:.2f} "
          f"(max possible {min(traj['state_final'].shape[1:])})")
    print(f"  reset events: {len(resets)} at {resets[:12]}")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
