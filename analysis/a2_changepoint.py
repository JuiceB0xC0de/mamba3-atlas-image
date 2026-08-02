"""A2: depth changepoint + head-permutation-safe cross-arm comparison.

Consumes static_atlas_all8.{npz,json}. Pure analysis, no model loading.

STANDING CAVEAT for every figure produced here:
    One released checkpoint per size and architecture; no seed replication.
    Family cross-section, not a fitted scaling law.

Design decisions, all deliberate:

  * BOUNDARIES ARE NOT INTERIOR. L0 and the terminal layer keep their identity
    and are excluded from the segmented fit. They are reported separately as
    boundary effects. Interpolating them onto normalized depth would blur the
    exact thing we are trying to measure.
  * D SIGN uses positive-head COUNTS with a binomial likelihood, never the
    rounded fraction. k_l of n_l heads positive at layer l.
  * SIGN AND AMPLITUDE ARE SEPARATE QUESTIONS. The sign transition is fit as
    segmented logistic; log|D| is fit separately as segmented linear. A model
    can flip sign without changing magnitude, or vice versa.
  * MODEL SELECTION: 0 vs 1 breakpoint always; 2 breakpoints only offered to
    24-layer models, and only accepted when BIC wins by a clear margin.
  * BREAKPOINT UNCERTAINTY: bootstrap over HEADS within layer. This is
    WITHIN-CHECKPOINT uncertainty only. It says nothing about seed or training
    variation, of which we have exactly one draw.
  * CROSS-ARM COMPARISON NEVER PAIRS HEAD INDICES. Independently trained models
    have permutation freedom over heads, so head i in SISO and head i in MIMO
    are unrelated. Compare distributions with Wasserstein distance and bootstrap
    the distance itself. Within a released pair n_layer matches (12/12 or
    24/24), so layer index is directly comparable and no depth interpolation is
    needed for the cross-arm test.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import optimize, stats

BIC_MARGIN = 6.0  # "clear" win, on the usual strong-evidence scale
N_BOOT = 1000


# --------------------------------------------------------------------------
# segmented fits
# --------------------------------------------------------------------------
def _hinge(x, taus):
    cols = [np.ones_like(x), x]
    for t in taus:
        cols.append(np.maximum(0.0, x - t))
    return np.stack(cols, axis=1)


def _binom_nll(beta, X, k, n):
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))


def fit_logistic(x, k, n, taus):
    X = _hinge(x, taus)
    beta0 = np.zeros(X.shape[1])
    res = optimize.minimize(_binom_nll, beta0, args=(X, k, n), method="BFGS")
    return res.fun, res.x


def _lin_rss(x, y, taus):
    X = _hinge(x, taus)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), beta


def bic_from_nll(nll, n_obs, n_par):
    return 2 * nll + n_par * np.log(n_obs)


def bic_from_rss(rss, n_obs, n_par):
    # gaussian MLE profile
    sigma2 = max(rss / n_obs, 1e-300)
    nll = 0.5 * n_obs * (np.log(2 * np.pi * sigma2) + 1)
    return 2 * nll + (n_par + 1) * np.log(n_obs)


def grid_taus(x, n_break):
    """Candidate breakpoints on the interior grid, keeping segments non-degenerate."""
    lo, hi = x.min(), x.max()
    cand = np.linspace(lo, hi, 25)[3:-3]
    if n_break == 1:
        return [(t,) for t in cand]
    out = []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            if b - a > 0.12:
                out.append((a, b))
    return out


def select_segmented(x, k, n, kind, allow_two):
    """Return dict with best model, its breakpoints, and the BIC table."""
    n_obs = len(x)
    fits = {}

    if kind == "logistic":
        # BIC sample size is the number of Bernoulli head observations, not the
        # number of layers, under the stated head-exchangeability model.
        n_bern = float(np.sum(n))
        nll0, _ = fit_logistic(x, k, n, ())
        fits[0] = (bic_from_nll(nll0, n_bern, 2), ())
        best1 = min(
            ((fit_logistic(x, k, n, t)[0], t) for t in grid_taus(x, 1)),
            key=lambda z: z[0],
        )
        fits[1] = (bic_from_nll(best1[0], n_bern, 4), best1[1])
        if allow_two:
            cands = grid_taus(x, 2)
            if cands:
                best2 = min(
                    ((fit_logistic(x, k, n, t)[0], t) for t in cands),
                    key=lambda z: z[0],
                )
                fits[2] = (bic_from_nll(best2[0], n_bern, 6), best2[1])
    else:
        y = k  # for linear kind, k carries y
        rss0, _ = _lin_rss(x, y, ())
        fits[0] = (bic_from_rss(rss0, n_obs, 2), ())
        best1 = min(
            ((_lin_rss(x, y, t)[0], t) for t in grid_taus(x, 1)), key=lambda z: z[0]
        )
        fits[1] = (bic_from_rss(best1[0], n_obs, 4), best1[1])
        if allow_two:
            cands = grid_taus(x, 2)
            if cands:
                best2 = min(
                    ((_lin_rss(x, y, t)[0], t) for t in cands), key=lambda z: z[0]
                )
                fits[2] = (bic_from_rss(best2[0], n_obs, 6), best2[1])

    best = min(fits, key=lambda m: fits[m][0])
    # Require a CLEAR BIC win before accepting extra breakpoints, comparing
    # against the BEST simpler model at each step (not the simplest one), and
    # walking all the way down so 2 -> 1 -> 0 demotion is possible.
    while best > 0:
        simpler = min((m for m in fits if m < best), key=lambda m: fits[m][0])
        if fits[simpler][0] - fits[best][0] < BIC_MARGIN:
            best = simpler
        else:
            break
    out = {
        "n_break": best,
        "taus": [float(t) for t in fits[best][1]],
        "bic": {str(m): float(v[0]) for m, v in fits.items()},
    }
    out["bic_delta_vs_0"] = float(fits[0][0] - fits[best][0])
    if kind == "logistic":
        _, beta = fit_logistic(x, k, n, tuple(fits[best][1]))
        out["beta"] = [float(b) for b in beta]
    return out


def fitted_crossover(beta, taus, lo=0.0, hi=1.0):
    """Depth where the fitted P(D>0) curve crosses 0.5, i.e. eta = 0.

    A zero-breakpoint model still has a meaningful crossover; it just has no
    kink. Evaluating the fitted curve is not the same as fitting a parameter,
    so a dense grid here does not manufacture precision.
    """
    xs = np.linspace(lo, hi, 2001)
    eta = _hinge(xs, taus) @ np.asarray(beta)
    sign = np.sign(eta)
    idx = np.where(np.diff(sign) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    # linear interpolation between the bracketing grid points
    x0, x1 = xs[i], xs[i + 1]
    e0, e1 = eta[i], eta[i + 1]
    return float(x0 - e0 * (x1 - x0) / (e1 - e0))


def bootstrap_breakpoint(x, D_per_layer, n_break, rng, n_boot=N_BOOT, n_grid=13):
    """Resample HEADS within each layer and profile the breakpoint location.

    Deliberately does NOT redo model selection per replicate. The number of
    breakpoints is a point estimate made once on the real data; re-selecting it
    inside the bootstrap multiplies the work by the size of the model space for
    no added information about WHERE the breakpoint sits. Here we hold the
    one-breakpoint form fixed and profile tau on a coarse grid, which is the
    quantity the interval is actually about.
    """
    n = np.array([len(lay) for lay in D_per_layer], dtype=float)
    grid = np.linspace(x.min(), x.max(), n_grid + 6)[3:-3]
    if n_break == 0:
        cands = [()]  # still refit: the crossover exists without a kink
    elif n_break == 1:
        cands = [(t,) for t in grid]
    else:
        cands = [
            (a, b) for i, a in enumerate(grid) for b in grid[i + 1:] if b - a > 0.12
        ]

    taus = np.empty((n_boot, n_break))
    cross = np.full(n_boot, np.nan)

    for bi in range(n_boot):
        k = np.array(
            [
                float((lay[rng.integers(0, len(lay), len(lay))] > 0).sum())
                for lay in D_per_layer
            ]
        )
        best, best_t, best_beta = np.inf, cands[0], None
        for t in cands:
            nll, beta = fit_logistic(x, k, n, t)
            if nll < best:
                best, best_t, best_beta = nll, t, beta
        if n_break:
            taus[bi] = best_t
        c = fitted_crossover(best_beta, best_t)
        if c is not None:
            cross[bi] = c

    out = {
        "n_boot": n_boot,
        "crossover": {
            "median": float(np.nanmedian(cross)),
            "lo": float(np.nanquantile(cross, 0.025)),
            "hi": float(np.nanquantile(cross, 0.975)),
            "frac_defined": float(np.mean(~np.isnan(cross))),
        },
        "crossover_draws": cross,  # kept in-memory for paired/ordering tests
        "note": "within-checkpoint head resampling only; one seed, one checkpoint",
    }
    if n_break:
        # report EVERY breakpoint, jointly resampled, not just the first
        out["per_break"] = [
            {
                "median": float(np.median(taus[:, j])),
                "lo": float(np.quantile(taus[:, j], 0.025)),
                "hi": float(np.quantile(taus[:, j], 0.975)),
            }
            for j in range(n_break)
        ]
    return out


# --------------------------------------------------------------------------
# cross-arm distribution distance (no head-index pairing)
# --------------------------------------------------------------------------
def wasserstein_ci(a, b, rng, n_boot=N_BOOT):
    d = stats.wasserstein_distance(a, b)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = stats.wasserstein_distance(
            a[rng.integers(0, len(a), len(a))], b[rng.integers(0, len(b), len(b))]
        )
    return float(d), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="static_atlas_all8.npz")
    ap.add_argument("--json", default="static_atlas_all8.json")
    ap.add_argument("--out", default="a2_results.json")
    ap.add_argument("--boot", type=int, default=200,
                    help="breakpoint bootstrap replicates: 200 pilot, 1000 final")
    ap.add_argument("--wboot", type=int, default=200,
                    help="Wasserstein bootstrap replicates")
    args = ap.parse_args()

    d = np.load(args.npz)
    rows = json.load(open(args.json))
    rng = np.random.default_rng(0)

    by_repo = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)

    results = {"caveat": (
        "One released checkpoint per size and architecture; no seed replication. "
        "Family cross-section, not a fitted scaling law."
    ), "per_checkpoint": {}, "cross_arm": {}, "stage_b_layers": {}}

    cross_draws, interior_span = {}, {}

    print("=" * 78)
    print("PER-CHECKPOINT: sign transition and amplitude, interior layers only")
    print("=" * 78)

    for repo, rs in by_repo.items():
        tag = repo.split("/")[-1]
        rs = sorted(rs, key=lambda r: r["layer"])
        n_layer = rs[0]["n_layer"]
        Dall = d[f"{tag}|D"]  # (n_layer, nheads)

        interior = [r for r in rs if not r["is_first"] and not r["is_final"]]
        idx = [r["layer"] for r in interior]
        # normalized depth WITHIN the interior only
        x = np.array([(i - idx[0]) / (idx[-1] - idx[0]) for i in idx])
        k = np.array([r["D_n_pos"] for r in interior], dtype=float)
        n = np.array([r["D_n_heads"] for r in interior], dtype=float)
        allow_two = n_layer >= 24

        sign_fit = select_segmented(x, k, n, "logistic", allow_two)
        boot = bootstrap_breakpoint(
            x, [Dall[i] for i in idx], sign_fit["n_break"], rng, n_boot=args.boot
        )
        # keep the raw draws in memory for paired + ordering tests, out of JSON
        cross_draws[tag] = boot.pop("crossover_draws")
        interior_span[tag] = (idx[0], idx[-1])

        logabs = np.array([np.log(np.abs(Dall[i]) + 1e-12).mean() for i in idx])
        amp_fit = select_segmented(x, logabs, None, "linear", allow_two)

        first, final = rs[0], rs[-1]
        interior_pos = float(np.mean([r["D_n_pos"] / r["D_n_heads"] for r in interior[:3]]))

        def to_layer(t):
            return idx[0] + t * (idx[-1] - idx[0])

        cross_x = fitted_crossover(sign_fit["beta"], sign_fit["taus"])
        cross_layer = to_layer(cross_x) if cross_x is not None else None

        entry = {
            "n_layer": n_layer,
            "is_mimo": rs[0]["is_mimo"],
            "sign": sign_fit,
            "sign_breakpoint_boot": boot,
            "crossover_depth": cross_x,
            "crossover_layer": cross_layer,
            "breakpoint_layers": [to_layer(t) for t in sign_fit["taus"]],
            "amplitude_log_absD": amp_fit,
            "boundary": {
                "L0_frac_pos": first["D_n_pos"] / first["D_n_heads"],
                "L0_halflife_median": first["prior_halflife_median"],
                "early_interior_frac_pos": interior_pos,
                "final_frac_pos": final["D_n_pos"] / final["D_n_heads"],
                "final_halflife_median": final["prior_halflife_median"],
            },
        }
        results["per_checkpoint"][tag] = entry

        print(f"\n{tag}  n_layer={n_layer}")
        bicd = "  ".join(f"k={m}:{v:.1f}" for m, v in sorted(sign_fit["bic"].items()))
        print(f"  sign      : {sign_fit['n_break']} breakpoint(s)   BIC[{bicd}]"
              f"  (delta vs k=0: {sign_fit['bic_delta_vs_0']:+.1f})")
        if sign_fit["n_break"] == 0:
            print("              no kink; smooth logistic in depth")
        for j, t in enumerate(sign_fit["taus"]):
            # report uncertainty as LAYER BINS. 10-22 interior points cannot
            # support sub-layer precision, so integer bins are the honest unit.
            b = boot["per_break"][j] if boot and "per_break" in boot else None
            if b:
                lo_l, hi_l = to_layer(b["lo"]), to_layer(b["hi"])
                print(f"              break {j + 1}: depth {t:.2f} ~L{to_layer(t):.0f}"
                      f"   boot layer bin L{int(np.floor(lo_l))}-L{int(np.ceil(hi_l))}")
            else:
                print(f"              break {j + 1}: depth {t:.2f} ~L{to_layer(t):.0f}")
        if cross_layer is not None:
            cb = boot["crossover"] if boot else None
            if cb:
                lo_l, hi_l = to_layer(cb["lo"]), to_layer(cb["hi"])
                entry_cross_bin = (int(np.floor(lo_l)), int(np.ceil(hi_l)))
                print(f"  crossover : P(D>0)=0.5 at depth {cross_x:.2f} (~L{cross_layer:.0f})"
                      f"   boot depth [{cb['lo']:.3f}, {cb['hi']:.3f}]"
                      f"  layer bin L{entry_cross_bin[0]}-L{entry_cross_bin[1]}")
                entry["crossover_boot"] = {k: v for k, v in cb.items()}
                entry["crossover_layer_bin"] = entry_cross_bin
            else:
                print(f"  crossover : P(D>0)=0.5 at depth {cross_x:.2f} (~L{cross_layer:.0f})")
        else:
            print("  crossover : none within the interior range")
        print(f"  amplitude : {amp_fit['n_break']} breakpoint(s) in log|D|, taus={[round(t,2) for t in amp_fit['taus']]}")
        print(f"  boundary  : L0 {entry['boundary']['L0_frac_pos']:.2f} pos vs "
              f"early-interior {interior_pos:.2f}; "
              f"L0 half-life {entry['boundary']['L0_halflife_median']:.2f} tok, "
              f"final {entry['boundary']['final_halflife_median']:.2f} tok")

    print("\n" + "=" * 78)
    print("CROSS-ARM: SISO vs MIMO head DISTRIBUTIONS (Wasserstein, no index pairing)")
    print("=" * 78)

    pairs = [("187m", "187m"), ("443m", "444m"), ("893m", "894m"), ("1.5b", "1.5b")]
    for ss, ms in pairs:
        st, mt = f"mamba3-siso-{ss}", f"mamba3-mimo-{ms}"
        if f"{st}|D" not in d or f"{mt}|D" not in d:
            continue
        Ds, Dm = d[f"{st}|D"], d[f"{mt}|D"]
        assert Ds.shape[0] == Dm.shape[0], "paired arms must share n_layer"

        per_layer = []
        for li in range(Ds.shape[0]):
            a = np.log(np.abs(Ds[li]) + 1e-12)
            b = np.log(np.abs(Dm[li]) + 1e-12)
            dist, lo, hi = wasserstein_ci(a, b, rng, n_boot=args.wboot)
            per_layer.append(
                {"layer": li, "w_log_absD": dist, "lo": lo, "hi": hi,
                 "median_shift": float(np.median(b) - np.median(a))}
            )
        results["cross_arm"][ss] = per_layer
        worst = max(per_layer, key=lambda r: r["w_log_absD"])
        mean_shift = float(np.mean([r["median_shift"] for r in per_layer]))
        results.setdefault("cross_arm_worst_layer", {})[ss] = worst["layer"]
        # WORDING: this is the geometric mean ACROSS LAYERS of the layerwise
        # MEDIAN amplitude ratio. Not a per-head geometric mean.
        print(
            f"\n{ss}: geometric mean across layers of layerwise median |D| ratio "
            f"(MIMO/SISO) = {np.exp(mean_shift):.2f}x  (log shift {mean_shift:+.3f})"
        )
        print(
            f"      largest divergence at L{worst['layer']}: W={worst['w_log_absD']:.3f} "
            f"[{worst['lo']:.3f}, {worst['hi']:.3f}]"
        )

    # ---- crossover: matched arm difference + ordering stability ----
    # Replicates are independent draws per checkpoint, so pairing replicate i
    # across two models is a valid construction of the difference distribution
    # under independence. This is WITHIN-checkpoint head uncertainty only; it
    # cannot speak to seed or training variation, of which we have one draw.
    print("\n" + "=" * 78)
    print("CROSSOVER: matched arm difference and family ordering stability")
    print("=" * 78)

    results["crossover_paired"] = {}
    for ss, ms in pairs:
        st, mt = f"mamba3-siso-{ss}", f"mamba3-mimo-{ms}"
        if st not in cross_draws or mt not in cross_draws:
            continue
        diff = cross_draws[mt] - cross_draws[st]  # MIMO minus SISO, in depth
        d_med = float(np.nanmedian(diff))
        lo, hi = float(np.nanquantile(diff, 0.025)), float(np.nanquantile(diff, 0.975))
        excl = "yes" if (lo > 0 or hi < 0) else "NO"
        results["crossover_paired"][ss] = {
            "median_depth_diff": d_med, "lo": lo, "hi": hi, "excludes_zero": excl == "yes"
        }
        print(
            f"  {ss:>5s}  MIMO-SISO crossover depth diff = {d_med:+.3f} "
            f"[{lo:+.3f}, {hi:+.3f}]   excludes 0: {excl}"
        )

    ORDER = ("187m", "443m", "893m", "1.5b")
    MORDER = ("187m", "444m", "894m", "1.5b")
    print()
    for arm, sizes in (("siso", ORDER), ("mimo", MORDER)):
        tags = [f"mamba3-{arm}-{s}" for s in sizes]
        if not all(t in cross_draws for t in tags):
            continue
        M = np.vstack([cross_draws[t] for t in tags])  # (4, n_boot)
        ok = np.all(np.diff(M, axis=0) < 0, axis=0)
        frac = float(np.nanmean(ok))
        results.setdefault("crossover_ordering", {})[arm] = frac
        print(f"  {arm}: strictly decreasing across all four sizes in "
              f"{frac:.1%} of replicates")

    both = None
    if all(f"mamba3-siso-{s}" in cross_draws for s in ORDER) and all(
        f"mamba3-mimo-{s}" in cross_draws for s in MORDER
    ):
        Ms = np.vstack([cross_draws[f"mamba3-siso-{s}"] for s in ORDER])
        Mm = np.vstack([cross_draws[f"mamba3-mimo-{s}"] for s in MORDER])
        both = float(
            np.nanmean(
                np.all(np.diff(Ms, axis=0) < 0, axis=0)
                & np.all(np.diff(Mm, axis=0) < 0, axis=0)
            )
        )
        results["crossover_ordering"]["both_arms"] = both
        print(f"  both arms simultaneously: {both:.1%} of replicates")

    # ---- frozen Stage B layer set, chosen by rules declared here ----
    print("\n" + "=" * 78)
    print("FROZEN STAGE B TARGET LAYERS")
    print("=" * 78)
    # Selection rules, declared here rather than chosen after looking:
    #   L0 and final                 boundary layers
    #   crossover                    where P(D>0)=0.5, exists even with no kink
    #   before / at / after          each accepted sign breakpoint
    #   flat early control           interior layer nearest depth 0.15
    #   largest cross-arm divergence layer of max Wasserstein on log|D|
    # then deduplicate and clamp to range.
    size_of = {"187m": "187m", "443m": "443m", "893m": "893m", "1.5b": "1.5b",
               "444m": "443m", "894m": "893m"}
    for tag, e in results["per_checkpoint"].items():
        n_layer = e["n_layer"]
        lo_i, hi_i = 1, n_layer - 2
        picks = {}

        def add(name, layer):
            if layer is None:
                return
            li = int(round(layer))
            picks.setdefault(name, max(0, min(n_layer - 1, li)))

        add("first", 0)
        add("final", n_layer - 1)
        add("crossover", e["crossover_layer"])
        # if the crossover interval spans more than one layer, capture its
        # shoulders too, same as we do for accepted breakpoints
        cb = e.get("crossover_layer_bin")
        if cb and (cb[1] - cb[0]) >= 1 and e["crossover_layer"] is not None:
            add("crossover_before", e["crossover_layer"] - 1)
            add("crossover_after", e["crossover_layer"] + 1)
        for j, bl in enumerate(e["breakpoint_layers"]):
            add(f"break{j + 1}_before", bl - 1)
            add(f"break{j + 1}_at", bl)
            add(f"break{j + 1}_after", bl + 1)
        add("flat_early_control", lo_i + 0.15 * (hi_i - lo_i))

        size = size_of[tag.rsplit("-", 1)[-1]]
        if size in results.get("cross_arm_worst_layer", {}):
            add("max_cross_arm_divergence", results["cross_arm_worst_layer"][size])

        uniq = sorted(set(picks.values()))
        results["stage_b_layers"][tag] = {"roles": picks, "layers": uniq}
        print(f"  {tag:20s} {uniq}")
        print(f"  {'':20s} {picks}")

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
