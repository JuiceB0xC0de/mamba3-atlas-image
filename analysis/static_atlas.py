"""Stage A static atlas across the released Mamba-3 family (SISO + MIMO, 4 sizes).

Weights only. Reads pytorch_model.bin directly with torch.load(weights_only=True),
so this needs neither mamba_ssm, tilelang, nor a GPU. Runs on CPU.

Tests implemented (see the Stage A battery):
  A1  depth geometry of the skip path and timescale prior: D and softplus(dt_bias),
      per head per layer. NOTE D is the direct/feedthrough term, NOT recurrent
      memory -- describe results as skip-path regime, never as memory inversion.
  A3  mimo_x / mimo_z / mimo_o channel geometry (MIMO arm only).
  A4  B_bias / C_bias geometry. THE load-bearing test: ngroups=1 means B and C are
      shared across all heads, so these biases carry the entire per-head state
      geometry, and they exist in BOTH arms (rank 1 for SISO, rank 4 for MIMO).
      This is the only per-head structure directly comparable across arms.
  A5  MLP confound census. The released pairs are parameter-matched, not
      architecture-identical: MIMO buys its mimo_* tensors with a narrower MLP.
      Any SISO/MIMO difference has the MLP as a live alternative explanation.

Init controls, not gaussian:
  mimo_x, mimo_o  ones / mimo_rank      (mamba3.py L131-133)
  mimo_z          ones
  B_bias, C_bias  ones                  (mamba3.py L121-122)
All are CONSTANT, i.e. exactly rank 1, so eff_rank 1.0 is the floor and the
all-ones direction is the reference for rotation. A gaussian baseline is the
wrong reference for displacement and inverts the story.

COMPARISON RULES (violating these invalidates the cross-arm claim):
  * NEVER pair SISO head 7 with MIMO head 7. Independently trained models have
    permutation freedom over heads; head index carries no shared meaning. Compare
    head DISTRIBUTIONS (bootstrap, distributional distance), never index to index.
  * Pair models by size, and pair layers by DEPTH FRACTION, not layer index
    (187m has 12 layers, the rest have 24).
  * SISO/MIMO pairs are parameter-matched, not architecture-identical: MIMO runs a
    narrower MLP (d_intermediate lower by 256-272 at every size) and chunk_size 16
    vs 64. Any cross-arm difference has the MLP as a live alternative explanation.

Usage:
  python static_atlas.py                      # all eight
  python static_atlas.py --only mimo-1.5b     # substring filter
  python static_atlas.py --wandb              # also log to W&B
"""

import argparse
import glob
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

SISO_SIZES = ("187m", "443m", "893m", "1.5b")
MIMO_SIZES = ("187m", "444m", "894m", "1.5b")
REPOS = [f"state-spaces/mamba3-siso-{s}" for s in SISO_SIZES] + [
    f"state-spaces/mamba3-mimo-{s}" for s in MIMO_SIZES
]

ENTITY = "ricks-holmberg-juiceb0xc0de"
PROJECT = "mamba3-family-atlas"
N_BOOT = 2000

# Provenance for the retention-prior transform. The zero-input prior below is
# only valid for this definition of the A activation; record it in the artifact.
SOURCE_REV = (
    "upstream/pypi_232_post1/mamba_ssm/modules/mamba3.py "
    "(heavy_tail_activation: f(x)=1+x for x>=0, 1/(1-x) for x<0; "
    "A=-f(dd_A), clamp max=-A_floor, A_floor=1e-4)"
)

# tensor -> per-element init value
MIMO_INIT = {"mimo_x": None, "mimo_z": 1.0, "mimo_o": None}  # None => 1/mimo_rank
BIAS_INIT = {"B_bias": 1.0, "C_bias": 1.0}


def deg(t):
    return t.clamp(-1, 1).arccos() * 180 / math.pi


def channel_geometry(W, init_val):
    """W: (nheads, rank, dim) float32. Returns per-head dict of (nheads,) tensors.

    spread    mean pairwise angle between the rank channel vectors, deg.
              0 = collapsed to one direction, ~90 = mutually orthogonal.
              Undefined (nan) when rank == 1.
    rot_init  mean angle from the all-ones direction, deg.
    scale     Frobenius norm relative to the constant init's norm.
    eff_rank  participation ratio of singular values, in [1, rank].
    drift     norm of (W - init), relative to the init norm.
    """
    nheads, rank, dim = W.shape
    out = {}

    V = F.normalize(W, dim=-1)
    if rank > 1:
        off = ~torch.eye(rank, dtype=torch.bool)
        out["spread"] = deg((V @ V.transpose(-1, -2))[:, off]).mean(-1)
    else:
        out["spread"] = torch.full((nheads,), float("nan"))

    u = F.normalize(torch.ones(dim), dim=0)
    out["rot_init"] = deg(V @ u).mean(-1)

    init_norm = init_val * math.sqrt(rank * dim)
    out["scale"] = W.norm(dim=(-2, -1)) / init_norm

    s = torch.linalg.svdvals(W)
    out["eff_rank"] = s.sum(-1) ** 2 / (s**2).sum(-1)

    d = W - init_val
    out["drift"] = d.norm(dim=(-2, -1)) / init_norm
    return out


def rank_decompose(W, init_val):
    """Split a rank-R bias into its SISO-comparable common part and MIMO-only part.

    Comparing SISO eff_rank 1 against MIMO eff_rank <=4 is structurally
    predetermined and proves nothing. Instead decompose:

        mean_h    = mean_r W[h, r]        the rank-COMMON component. SISO has
                                          exactly this and nothing else, so it
                                          is the fair cross-arm comparison.
        diff[h,r] = W[h, r] - mean_h      the rank-DIFFERENTIAL component. Sums
                                          to zero over r by construction, so its
                                          effective rank is bounded by R-1.
                                          This is what MIMO adds beyond SISO.

    Returns per-head (nheads,) tensors. At rank 1 the differential is identically
    zero and its shape metrics are nan.
    """
    nheads, rank, dim = W.shape
    out = {}

    mean = W.mean(dim=1)  # (nheads, dim)
    diff = W - mean.unsqueeze(1)

    u = F.normalize(torch.ones(dim), dim=0)
    out["mean_rot_init"] = deg(F.normalize(mean, dim=-1) @ u)
    out["mean_scale"] = mean.norm(dim=-1) / (init_val * math.sqrt(dim))
    out["mean_drift"] = (mean - init_val).norm(dim=-1) / (init_val * math.sqrt(dim))

    wn = W.norm(dim=(-2, -1)).clamp_min(1e-12)
    out["diff_energy"] = diff.norm(dim=(-2, -1)) / wn

    if rank > 1:
        s = torch.linalg.svdvals(diff)
        out["diff_eff_rank"] = s.sum(-1) ** 2 / (s**2).sum(-1).clamp_min(1e-24)
        V = F.normalize(diff, dim=-1)
        off = ~torch.eye(rank, dtype=torch.bool)
        out["diff_spread"] = deg((V @ V.transpose(-1, -2))[:, off]).mean(-1)
    else:
        nan = torch.full((nheads,), float("nan"))
        out["diff_eff_rank"] = nan
        out["diff_spread"] = nan.clone()
    return out


def differential_alignment(B, C):
    """Cosine between the B and C rank-DIFFERENTIAL components, per head.

    Tests whether MIMO's extra rank structure is coordinated across B and C, the
    differential counterpart of the synergy measured on the full biases.
    """
    nheads, rank, _ = B.shape
    if rank == 1:
        return torch.full((nheads,), float("nan"))
    db = (B - B.mean(dim=1, keepdim=True)).reshape(nheads, -1)
    dc = (C - C.mean(dim=1, keepdim=True)).reshape(nheads, -1)
    return F.cosine_similarity(db, dc, dim=-1)


def bias_synergy(B, C, init_val):
    """Paper reports B/C biases behave synergistically. Measure it two ways.

    raw_cos    cosine between the per-head bias vectors as-is. Both start at
               ones, so this is high by construction and mostly uninformative.
    drift_cos  cosine between (B - init) and (C - init). This is the real test:
               did training push the two biases in a COORDINATED direction.
    """
    nheads = B.shape[0]
    b, c = B.reshape(nheads, -1), C.reshape(nheads, -1)
    raw = F.cosine_similarity(b, c, dim=-1)
    drift = F.cosine_similarity(b - init_val, c - init_val, dim=-1)
    return raw, drift


def bootstrap_ci(vals, gen, n_boot=N_BOOT):
    v = vals[~torch.isnan(vals)]
    if v.numel() == 0:
        return float("nan"), float("nan")
    idx = torch.randint(0, v.shape[0], (n_boot, v.shape[0]), generator=gen)
    bs = v[idx].mean(-1)
    return bs.quantile(0.025).item(), bs.quantile(0.975).item()


def load_checkpoint(repo):
    cfg_path = hf_hub_download(repo, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    bin_path = hf_hub_download(repo, "pytorch_model.bin")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    return cfg, sd


def analyze(repo, gen, run=None):
    cfg, sd = load_checkpoint(repo)
    ssm = cfg.get("ssm_cfg", {}) or {}
    n_layer = cfg["n_layer"]
    d_model = cfg["d_model"]
    rank = ssm.get("mimo_rank") or 1
    headdim = ssm.get("headdim")
    is_mimo = ssm.get("mimo_rank") is not None

    nheads = sd["backbone.layers.0.mixer.dt_bias"].shape[0]
    d_state = sd["backbone.layers.0.mixer.B_bias"].shape[-1]

    # exact parameter count. Equal FILE SIZES only support approximate matching;
    # this is the actual sum. lm_head is tied to the embedding, so count uniques.
    seen, n_params = set(), 0
    for k, v in sd.items():
        ptr = v.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        n_params += v.numel()

    print(f"\n=== {repo}")
    print(
        f"  n_layer={n_layer} d_model={d_model} nheads={nheads} headdim={headdim} "
        f"d_state={d_state} rank={rank} mimo={is_mimo} params={n_params/1e6:.3f}M"
    )

    rows = []
    arrays = {}

    for li in range(n_layer):
        p = f"backbone.layers.{li}."
        mx = p + "mixer."
        row = {
            "repo": repo, "layer": li, "is_mimo": is_mimo,
            "n_params": n_params, "n_layer": n_layer, "d_model": d_model,
            # boundary layers keep their identity; only interior layers get
            # interpolated onto normalized depth (see A2).
            "is_first": li == 0,
            "is_final": li == n_layer - 1,
            "depth_frac": li / max(n_layer - 1, 1),
            "retention_transform": "heavy_tail_activation(0)=1 => A(0)=-1",
            "source_rev": SOURCE_REV,
        }

        # --- A1: skip path + ZERO-INPUT RETENTION PRIOR ---
        # dt is heavy-tailed across heads, so the mean is a misleading summary
        # (siso-187m L0: mean half-life 145 tokens, MEDIAN 2.3). Mean retained
        # only for continuity with the earlier 1.5b atlas; foreground the median
        # and percentiles.
        dt = F.softplus(sd[mx + "dt_bias"].float())
        D = sd[mx + "D"].float()

        row["dt_mean"] = dt.mean().item()
        row["dt_median"] = dt.median().item()
        for q in (10, 25, 75, 90):
            row[f"dt_p{q}"] = torch.quantile(dt, q / 100).item()
        row["dt_lo"], row["dt_hi"] = bootstrap_ci(dt, gen)

        # Zero-input retention prior, in tokens. NOT the model's half-life:
        # both A_t and Delta_t are data-dependent. This is the t=0 default,
        # valid because heavy_tail_activation(0) == 1 => A(0) == -1 exactly,
        # leaving the A_floor clamp inactive. Monotone in dt_bias, so it adds
        # interpretable units but NO independent changepoint evidence.
        hl = math.log(2) / dt
        row["prior_halflife_median"] = hl.median().item()
        row["prior_halflife_p25"] = torch.quantile(hl, 0.25).item()
        row["prior_halflife_p75"] = torch.quantile(hl, 0.75).item()
        row["prior_halflife_iqr"] = row["prior_halflife_p75"] - row["prior_halflife_p25"]

        row["D_absmean"] = D.abs().mean().item()
        row["D_absmedian"] = D.abs().median().item()
        row["D_frac_pos"] = (D > 0).float().mean().item()
        # positive-head COUNT, for binomial modelling in A2 (not the rounded frac)
        row["D_n_pos"] = int((D > 0).sum().item())
        row["D_n_heads"] = int(D.numel())
        row["D_absmean_lo"], row["D_absmean_hi"] = bootstrap_ci(D.abs(), gen)
        arrays.setdefault("dt", []).append(dt.numpy())
        arrays.setdefault("D", []).append(D.numpy())
        arrays.setdefault("prior_halflife", []).append(hl.numpy())

        # --- A4: B_bias / C_bias, the cross-arm comparable per-head geometry ---
        for name, init in BIAS_INIT.items():
            W = sd[mx + name].float()
            g = channel_geometry(W, init)
            for k, v in g.items():
                a = v.numpy()
                # spread is nan by definition at rank 1 (SISO); keep the column
                row[f"{name}/{k}"] = float("nan") if np.isnan(a).all() else float(np.nanmean(a))
                arrays.setdefault(f"{name}/{k}", []).append(a)
            lo, hi = bootstrap_ci(g["rot_init"], gen)
            row[f"{name}/rot_init_lo"], row[f"{name}/rot_init_hi"] = lo, hi

        # A4b: rank-common vs rank-differential decomposition. The common part is
        # what SISO also has, so it is the fair cross-arm comparison; the
        # differential part is MIMO-only.
        for name in BIAS_INIT:
            dec = rank_decompose(sd[mx + name].float(), BIAS_INIT[name])
            for k, v in dec.items():
                a = v.numpy()
                row[f"{name}/{k}"] = (
                    float("nan") if np.isnan(a).all() else float(np.nanmean(a))
                )
                arrays.setdefault(f"{name}/{k}", []).append(a)
            lo, hi = bootstrap_ci(dec["mean_rot_init"], gen)
            row[f"{name}/mean_rot_init_lo"] = lo
            row[f"{name}/mean_rot_init_hi"] = hi

        dalign = differential_alignment(
            sd[mx + "B_bias"].float(), sd[mx + "C_bias"].float()
        )
        da = dalign.numpy()
        row["BC_diff_align"] = float("nan") if np.isnan(da).all() else float(np.nanmean(da))
        row["BC_diff_align_lo"], row["BC_diff_align_hi"] = bootstrap_ci(dalign, gen)
        arrays.setdefault("BC_diff_align", []).append(da)

        raw, drift = bias_synergy(
            sd[mx + "B_bias"].float(), sd[mx + "C_bias"].float(), 1.0
        )
        row["BC_raw_cos"] = raw.mean().item()
        row["BC_drift_cos"] = drift.mean().item()
        row["BC_drift_cos_lo"], row["BC_drift_cos_hi"] = bootstrap_ci(drift, gen)
        arrays.setdefault("BC_drift_cos", []).append(drift.numpy())

        # --- A3: MIMO channel geometry (MIMO arm only) ---
        if is_mimo:
            for name, init in MIMO_INIT.items():
                init_val = init if init is not None else 1.0 / rank
                W = sd[mx + name].float()
                g = channel_geometry(W, init_val)
                for k, v in g.items():
                    row[f"{name}/{k}"] = float(np.nanmean(v.numpy()))
                    arrays.setdefault(f"{name}/{k}", []).append(v.numpy())
                lo, hi = bootstrap_ci(g["spread"], gen)
                row[f"{name}/spread_lo"], row[f"{name}/spread_hi"] = lo, hi

        # --- A5: MLP confound + mixer-vs-MLP weight budget ---
        fc1 = sd[p + "mlp.fc1.weight"].float()
        fc2 = sd[p + "mlp.fc2.weight"].float()
        outp = sd[mx + "out_proj.weight"].float()
        inp = sd[mx + "in_proj.weight"].float()
        row["mlp_hidden"] = fc2.shape[1]
        row["mlp_fc1_norm"] = fc1.norm().item()
        row["mlp_fc2_norm"] = fc2.norm().item()
        row["mixer_out_norm"] = outp.norm().item()
        row["mixer_in_norm"] = inp.norm().item()
        # crude write-strength proxy: how loud is each path into the residual
        row["mixer_vs_mlp"] = outp.norm().item() / max(fc2.norm().item(), 1e-9)

        rows.append(row)
        if run is not None:
            run.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

    print(f"{'L':>3s} {'dt_med':>8s} {'hl_med':>8s} {'hl_IQR':>9s} "
          f"{'|D|med':>8s} {'D+/n':>8s} {'Bb_dNRG':>8s} {'BCdrift':>8s} "
          f"{'BCdiff':>8s}")
    for r in rows:
        print(
            f"{r['layer']:3d} {r['dt_median']:8.4f} "
            f"{r['prior_halflife_median']:8.2f} {r['prior_halflife_iqr']:9.2f} "
            f"{r['D_absmedian']:8.3f} "
            f"{str(r['D_n_pos']) + '/' + str(r['D_n_heads']):>8s} "
            f"{r['B_bias/diff_energy']:8.4f} {r['BC_drift_cos']:8.3f} "
            f"{r['BC_diff_align']:8.3f}"
        )

    del sd
    return rows, {k: np.stack(v) for k, v in arrays.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated substring filters on repo name")
    ap.add_argument("--skip", default=None,
                    help="comma-separated substrings to exclude")
    ap.add_argument("--out", default="static_atlas.npz")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    repos = REPOS
    if args.only:
        pats = [p.strip() for p in args.only.split(",") if p.strip()]
        repos = [r for r in repos if any(p in r for p in pats)]
    if args.skip:
        pats = [p.strip() for p in args.skip.split(",") if p.strip()]
        repos = [r for r in repos if not any(p in r for p in pats)]
    gen = torch.Generator().manual_seed(0)

    run = None
    if args.wandb:
        import wandb

        run = wandb.init(
            entity=ENTITY, project=PROJECT, job_type="static-atlas",
            config={"repos": repos, "n_boot": N_BOOT},
        )

    all_rows, all_arrays = [], {}
    for repo in repos:
        rows, arrays = analyze(repo, gen, run)
        all_rows.extend(rows)
        tag = repo.split("/")[-1]
        for k, v in arrays.items():
            all_arrays[f"{tag}|{k}"] = v

    np.savez_compressed(args.out, **all_arrays)
    with open(args.out.replace(".npz", ".json"), "w") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"\nwrote {args.out} and {args.out.replace('.npz', '.json')}")

    # corrected V0 census: exact parameter counts, replacing the zeros left by
    # the safetensors-header probe (these releases ship pytorch_model.bin).
    census = {}
    for r in all_rows:
        census[r["repo"]] = {
            "n_params": r["n_params"],
            "n_layer": r["n_layer"],
            "d_model": r["d_model"],
            "mlp_hidden": r["mlp_hidden"],
            "is_mimo": r["is_mimo"],
        }
    with open("checkpoint_census.json", "w") as fh:
        json.dump(census, fh, indent=2)
    print("wrote checkpoint_census.json (exact parameter counts)")

    print("\n=== parameter match, exact ===")
    for ss, ms in zip(SISO_SIZES, MIMO_SIZES):
        a = census.get(f"state-spaces/mamba3-siso-{ss}")
        b = census.get(f"state-spaces/mamba3-mimo-{ms}")
        if not (a and b):
            continue
        d = b["n_params"] - a["n_params"]
        print(
            f"  {ss:>5s}  siso {a['n_params']:>12,}  mimo {b['n_params']:>12,}  "
            f"delta {d:>+9,} ({100 * d / a['n_params']:+.3f}%)  "
            f"mlp {a['mlp_hidden']} -> {b['mlp_hidden']}"
        )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
