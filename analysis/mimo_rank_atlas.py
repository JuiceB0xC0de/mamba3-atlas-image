"""Per-head MIMO rank-utilization atlas for state-spaces/mamba3-mimo-1.5b.

Static weight analysis, no corpus needed. Logs per-head values (24 layers x 3
tensors x 64 heads) to W&B plus a local .npz, so every layer-level summary
stays recomputable without the GPU.

Metrics, per head, on mimo_{x,z,o} of shape (nheads, mimo_rank, headdim):
  spread    mean pairwise angle (deg) between the mimo_rank channel vectors.
            0 = collapsed onto one direction, ~90 = mutually orthogonal.
  rot_init  mean angle (deg) from the all-ones init direction.
  scale     tensor norm relative to its init norm (mimo_z inits at 1.0,
            mimo_x and mimo_o at 1/mimo_rank -- see mamba3.py L131-133).
  eff_rank  participation ratio of the singular values, in [1, mimo_rank].

The init is CONSTANT ones, i.e. exactly rank 1. So 1.0 is the floor, not 0,
and a gaussian control is the wrong reference for "how much did this move".
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

MODEL_ID = "state-spaces/mamba3-mimo-1.5b"
ENTITY = "ricks-holmberg-juiceb0xc0de"
PROJECT = "mamba3-mimo-atlas"
TENSORS = {"mimo_x": 0.25, "mimo_z": 1.0, "mimo_o": 0.25}
RANK = 4
N_BOOT = 2000
N_CONTROL_SEEDS = 20


def _deg(t):
    return t.clamp(-1, 1).arccos() * 180 / math.pi


def per_head(W, init_val, headdim, dev):
    """W: (nheads, rank, headdim) float. Returns 4 tensors of shape (nheads,)."""
    off = ~torch.eye(RANK, dtype=torch.bool, device=dev)
    u = F.normalize(torch.ones(headdim, device=dev), dim=0)

    V = F.normalize(W, dim=-1)
    spread = _deg((V @ V.transpose(-1, -2))[:, off]).mean(-1)
    rot = _deg(V @ u).mean(-1)
    scale = W.norm(dim=(-2, -1)) / (init_val * math.sqrt(RANK * headdim))
    s = torch.linalg.svdvals(W)
    eff = s.sum(-1) ** 2 / (s**2).sum(-1)
    return spread, rot, scale, eff


def bootstrap_ci(vals, gen, n_boot=N_BOOT):
    n = vals.shape[0]
    idx = torch.randint(0, n, (n_boot, n), generator=gen, device=vals.device)
    bs = vals[idx].mean(-1)
    return bs.quantile(0.025).item(), bs.quantile(0.975).item()


def random_control(nheads, headdim, dev):
    """Spread for gaussian weights, over seeds. Expect ~90 deg in high dim."""
    off = ~torch.eye(RANK, dtype=torch.bool, device=dev)
    out = []
    for s in range(N_CONTROL_SEEDS):
        g = torch.Generator(device=dev).manual_seed(s)
        R = torch.randn(nheads, RANK, headdim, generator=g, device=dev)
        V = F.normalize(R, dim=-1)
        out.append(_deg((V @ V.transpose(-1, -2))[:, off]).mean().item())
    t = torch.tensor(out)
    return t.mean().item(), t.std().item()


def main():
    dev = "cuda"
    model = MambaLMHeadModel.from_pretrained(
        MODEL_ID, device=dev, dtype=torch.bfloat16
    ).eval()
    layers = model.backbone.layers
    nheads, _, headdim = layers[0].mixer.mimo_x.shape

    ctrl_mean, ctrl_std = random_control(nheads, headdim, dev)

    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type="weight-analysis",
        config={
            "model_id": MODEL_ID,
            "n_layer": len(layers),
            "nheads": nheads,
            "headdim": headdim,
            "mimo_rank": RANK,
            "torch": torch.__version__,
            "tilelang": "0.1.8",
            "apache_tvm_ffi": "0.1.8.post2",
            "control_spread_deg": ctrl_mean,
            "control_spread_std": ctrl_std,
        },
    )

    gen = torch.Generator(device=dev).manual_seed(0)
    names = list(TENSORS)
    arrs = {
        k: np.zeros((len(layers), len(names), nheads), dtype=np.float32)
        for k in ("spread", "rot_init", "scale", "eff_rank")
    }

    table = wandb.Table(
        columns=["layer", "tensor", "head", "spread", "rot_init", "scale", "eff_rank"]
    )
    # dt and D are per (layer, head), not per mimo tensor -- own table.
    mixer_table = wandb.Table(columns=["layer", "head", "dt", "D"])
    mixer_arrs = {
        k: np.zeros((len(layers), nheads), dtype=np.float32) for k in ("dt", "D")
    }
    summary = wandb.Table(
        columns=[
            "layer", "tensor", "spread", "spread_lo", "spread_hi",
            "rot_init", "scale", "eff_rank", "norm_rank",
        ]
    )

    for li, blk in enumerate(layers):
        # Mamba-3 has no static A: it is projected from the input alongside dt
        # and the trapezoid scalar, so dt/D are the only static timescale knobs.
        dt = F.softplus(blk.mixer.dt_bias.detach().float())
        D = blk.mixer.D.detach().float()
        mixer_arrs["dt"][li] = dt.cpu().numpy()
        mixer_arrs["D"][li] = D.cpu().numpy()
        for h in range(nheads):
            mixer_table.add_data(li, h, dt[h].item(), D[h].item())
        if li == 0:
            print("\nlayer     dt mean     min     max    |D| mean   D>0")
        print(
            f"{li:5d}  {dt.mean():9.3f} {dt.min():7.3f} {dt.max():7.3f}"
            f"  {D.abs().mean():9.3f}  {(D > 0).sum().item():3d}/64"
        )
        run.log(
            {
                "mixer/dt_mean": dt.mean().item(),
                "mixer/dt_max": dt.max().item(),
                "mixer/D_abs_mean": D.abs().mean().item(),
                "mixer/D_frac_pos": (D > 0).float().mean().item(),
                "layer": li,
            }
        )

        for ti, name in enumerate(names):
            W = getattr(blk.mixer, name).detach().float()
            spread, rot, scale, eff = per_head(W, TENSORS[name], headdim, dev)

            for key, v in zip(
                ("spread", "rot_init", "scale", "eff_rank"), (spread, rot, scale, eff)
            ):
                arrs[key][li, ti] = v.cpu().numpy()

            for h in range(nheads):
                table.add_data(
                    li, name, h,
                    spread[h].item(), rot[h].item(),
                    scale[h].item(), eff[h].item(),
                )

            lo, hi = bootstrap_ci(spread, gen)
            summary.add_data(
                li, name,
                spread.mean().item(), lo, hi,
                rot.mean().item(), scale.mean().item(),
                eff.mean().item(), (eff.mean().item() - 1) / (RANK - 1),
            )
            run.log(
                {
                    f"{name}/spread": spread.mean().item(),
                    f"{name}/rot_init": rot.mean().item(),
                    f"{name}/scale": scale.mean().item(),
                    f"{name}/eff_rank": eff.mean().item(),
                    f"{name}/norm_rank": (eff.mean().item() - 1) / (RANK - 1),
                    "layer": li,
                }
            )

    run.log({"per_head": table, "per_layer": summary, "per_head_mixer": mixer_table})

    out = "mimo_rank_atlas.npz"
    np.savez_compressed(
        out,
        tensors=np.array(names),
        control_spread=np.array([ctrl_mean, ctrl_std], dtype=np.float32),
        dt=mixer_arrs["dt"],
        D=mixer_arrs["D"],
        **arrs,
    )
    art = wandb.Artifact("mimo-rank-per-head", type="analysis")
    art.add_file(out)
    run.log_artifact(art)
    print(f"control spread {ctrl_mean:.2f} +/- {ctrl_std:.2f} deg")
    print(f"wrote {out}")
    run.finish()


if __name__ == "__main__":
    main()
