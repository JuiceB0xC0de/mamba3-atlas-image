"""B2-3: the null harness. Runs on CPU, after capture, at zero GPU cost.

This file only works because capture stored per-BLOCK statistics (decision D1).
Pooling per class would have made every null another GPU run.

NULLS ARE MATCHED TO CLAIMS, NOT APPLIED UNIFORMLY (contract 5). A same-class
label permutation is the right null for a class contrast and the WRONG null for
almost everything else.

  claim type            null
  --------------------  ------------------------------------------------------
  class contrast        same-class split AND label permutation
  corpus statistic      document-block bootstrap; position/length matching
  rank utilization      rank permutation, orthogonal rotation, matched subspace
  SISO/MIMO comparison  distributional, labeled BUNDLE-LEVEL. Never a
                        permutation, which would pretend the two architectures
                        are exchangeable. They are not: different MLP width,
                        different chunk_size, +0.2% params.

LENGTH MATCHING IS NOT OPTIONAL for stream B. Measured on the pinned tokenizer:
authentic median 16 tokens vs corporate 11 (+45%); code_probes 5 vs
neutral_stems 9 (-80%). On sequences that short the state is dominated by
early-position transients, so an unmatched class difference is substantially a
length difference wearing a class label. `stratified_contrast` handles this.

Usage as a library; `python analysis/nulls.py --self-test` checks it on
synthetic data with a known answer.
"""

import argparse
import sys

import numpy as np


# --------------------------------------------------------------------------
# class contrasts
# --------------------------------------------------------------------------


def observed_contrast(rows, labels, a, b):
    """Mean difference between two label groups. rows: (n_blocks, L, H)."""
    ma = rows[labels == a].mean(0)
    mb = rows[labels == b].mean(0)
    return ma - mb


def permutation_null(rows, labels, a, b, n_perm=2000, seed=0):
    """Shuffle the labels. The correct null for 'is there a class effect'.

    Returns (observed, null_distribution, two_sided_p) with shapes
    (L,H), (n_perm,L,H), (L,H).
    """
    rng = np.random.default_rng(seed)
    mask = (labels == a) | (labels == b)
    sub, lab = rows[mask], labels[mask]
    obs = observed_contrast(sub, lab, a, b)

    null = np.empty((n_perm, *obs.shape), dtype=np.float32)
    for i in range(n_perm):
        null[i] = observed_contrast(sub, rng.permutation(lab), a, b)

    p = (np.abs(null) >= np.abs(obs)).mean(0)
    return obs, null, p


def same_class_split_null(rows, labels, cls, n_split=2000, seed=0):
    """Split ONE class in half at random and contrast the halves.

    This is the floor: how much apparent separation arises from sampling alone.
    Without it, a dip at some layer is uninterpretable. This is the same
    discipline that saved the weight-side analysis, where a gaussian control had
    the story exactly backwards.
    """
    rng = np.random.default_rng(seed)
    sub = rows[labels == cls]
    n = len(sub)
    out = np.empty((n_split, *sub.shape[1:]), dtype=np.float32)
    for i in range(n_split):
        idx = rng.permutation(n)
        h = n // 2
        out[i] = sub[idx[:h]].mean(0) - sub[idx[h:2 * h]].mean(0)
    return out


def stratified_contrast(rows, labels, lengths, a, b, n_bins=5, n_perm=2000, seed=0):
    """Length-matched class contrast: stratify on token count, then pool.

    Removes the length confound rather than hoping it is small. Bins are
    quantiles of the pooled length distribution, so each bin holds comparable
    counts from both classes.
    """
    rng = np.random.default_rng(seed)
    mask = (labels == a) | (labels == b)
    sub, lab, ln = rows[mask], labels[mask], lengths[mask]

    edges = np.quantile(ln, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1
    binid = np.digitize(ln, edges[1:-1])

    def pooled(labs, want_cov=False):
        parts, weights, matched = [], [], 0
        for k in range(n_bins):
            m = binid == k
            if m.sum() < 4:
                continue
            ga, gb = sub[m & (labs == a)], sub[m & (labs == b)]
            if len(ga) == 0 or len(gb) == 0:
                continue
            parts.append(ga.mean(0) - gb.mean(0))
            weights.append(min(len(ga), len(gb)))
            matched += min(len(ga), len(gb)) * 2
        cov = matched / max(len(labs), 1)
        if not parts:
            out = np.full(sub.shape[1:], np.nan, dtype=np.float32)
            return (out, cov) if want_cov else out
        w = np.asarray(weights, dtype=np.float64)
        out = np.tensordot(w / w.sum(), np.stack(parts), axes=(0, 0)).astype(np.float32)
        return (out, cov) if want_cov else out

    obs, coverage = pooled(lab, want_cov=True)
    null = np.stack([pooled(rng.permutation(lab)) for _ in range(n_perm)])
    p = (np.abs(null) >= np.abs(obs)).mean(0)
    # coverage is the fraction of blocks that had a length-matched counterpart.
    # LOW COVERAGE MEANS THE CLASSES ARE NOT LENGTH-COMPARABLE and no amount of
    # stratification fixes it -- the contrast is confounded by construction.
    return obs, null, p, coverage


# --------------------------------------------------------------------------
# corpus statistics
# --------------------------------------------------------------------------


def block_bootstrap(rows, n_boot=2000, seed=0, block_weights=None):
    """Resample BLOCKS with replacement. The unit is the block, not the token.

    Token-level resampling would ignore autocorrelation inside a document or
    prompt and produce intervals that are far too tight.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    out = np.empty((n_boot, *rows.shape[1:]), dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if block_weights is None:
            out[i] = rows[idx].mean(0)
        else:
            w = block_weights[idx]
            out[i] = (rows[idx] * w[:, None, None]).sum(0) / w.sum()
    return out


def ci(dist, q=0.95):
    lo = (1 - q) / 2
    return np.quantile(dist, lo, axis=0), np.quantile(dist, 1 - lo, axis=0)


# --------------------------------------------------------------------------
# rank utilization
# --------------------------------------------------------------------------


def rank_permutation_null(x, rank_dim=-2, n_perm=500, seed=0):
    """Permute the rank slots. Tests whether rank STRUCTURE matters or only its
    marginal distribution. x: (..., rank, d)."""
    rng = np.random.default_rng(seed)
    R = x.shape[rank_dim]
    out = []
    for _ in range(n_perm):
        out.append(np.take(x, rng.permutation(R), axis=rank_dim))
    return np.stack(out)


def random_rotation_null(x, n_perm=500, seed=0):
    """Replace the rank subspace with a random orthogonal one of equal norm.

    The matched control for 'is this arrangement special', as opposed to a
    gaussian control, which is the wrong reference for constant-initialized
    tensors and inverted the story once already.
    """
    rng = np.random.default_rng(seed)
    R = x.shape[-2]
    out = np.empty((n_perm, *x.shape), dtype=np.float32)
    for i in range(n_perm):
        q, _ = np.linalg.qr(rng.normal(size=(R, R)))
        out[i] = np.einsum("rs,...sd->...rd", q, x)
    return out


# --------------------------------------------------------------------------
# SISO vs MIMO: distributional only
# --------------------------------------------------------------------------


def bundle_level_distance(a_heads, b_heads, n_boot=1000, seed=0):
    """Wasserstein distance between two arms' head distributions, with a CI.

    NEVER pairs head index i with head index i: independently trained models
    have permutation freedom over heads. And this is labeled BUNDLE-LEVEL: the
    released arms differ in MLP width and chunk_size as well as MIMO, so any
    difference is between two complete systems, not an isolated mechanism.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)
    d = stats.wasserstein_distance(a_heads, b_heads)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = stats.wasserstein_distance(
            rng.choice(a_heads, len(a_heads), replace=True),
            rng.choice(b_heads, len(b_heads), replace=True),
        )
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"distance": float(d), "lo": float(lo), "hi": float(hi),
            "scope": "bundle-level: MIMO + narrower MLP + chunk_size 16 vs 64"}


# --------------------------------------------------------------------------


def _self_test():
    rng = np.random.default_rng(0)
    L, H, n = 3, 8, 400

    # a real effect in exactly one (layer, head) cell
    rows = rng.normal(size=(n, L, H)).astype(np.float32)
    labels = np.array(["a"] * (n // 2) + ["b"] * (n // 2))
    rows[labels == "a", 1, 5] += 1.2

    obs, null, p = permutation_null(rows, labels, "a", "b", n_perm=500)
    off = np.delete(p.ravel(), 1 * H + 5)
    # NOTE: do not assert that zero off-cells reach p=0. With 24 cells and 500
    # permutations, ~5% of runs produce one by chance. That is the multiplicity
    # problem, not a bug, and pretending otherwise would hide it. Assert the
    # planted cell is the strongest and the bulk of off-cells are unremarkable.
    assert p[1, 5] < 0.01, p[1, 5]
    assert p[1, 5] <= off.min(), (p[1, 5], off.min())
    assert np.median(off) > 0.1, np.median(off)
    print(f"  permutation finds the planted cell (p={p[1,5]:.4f}); "
          f"off-cell median p={np.median(off):.2f}  OK")

    sc = same_class_split_null(rows, labels, "a", n_split=300)
    assert abs(float(sc.mean())) < 0.1
    print(f"  same-class split centred on zero ({float(sc.mean()):+.4f})          OK")

    # length confound with OVERLAPPING distributions, as in the real corpora
    # (authentic median 16 vs corporate 11). Class carries no true effect; the
    # apparent difference is entirely length-driven.
    lengths = np.where(labels == "a", rng.poisson(16, n), rng.poisson(11, n)) + 1
    rows2 = rng.normal(size=(n, L, H)).astype(np.float32)
    rows2 += 0.10 * lengths[:, None, None]
    _, _, p_raw = permutation_null(rows2, labels, "a", "b", n_perm=400)
    _, _, p_str, cov = stratified_contrast(rows2, labels, lengths, "a", "b",
                                           n_bins=5, n_perm=400)
    print(f"  length confound: raw p={p_raw.mean():.3f} -> stratified "
          f"p={np.nanmean(p_str):.3f} (coverage {cov:.0%})  OK")
    assert np.nanmean(p_str) > p_raw.mean() + 0.1, (p_raw.mean(), np.nanmean(p_str))

    # disjoint lengths CANNOT be matched; the harness must say so via coverage
    disjoint = np.where(labels == "a", 30, 5)
    _, _, _, cov_bad = stratified_contrast(rows2, labels, disjoint, "a", "b",
                                           n_bins=5, n_perm=50)
    assert cov_bad < 0.2, cov_bad
    print(f"  disjoint lengths flagged as unmatchable (coverage {cov_bad:.0%})  OK")

    bs = block_bootstrap(rows, n_boot=300)
    lo, hi = ci(bs)
    assert (lo <= rows.mean(0)).all() and (rows.mean(0) <= hi).all()
    print("  block bootstrap CI covers the point estimate                  OK")

    x = rng.normal(size=(6, 4, 16)).astype(np.float32)
    assert rank_permutation_null(x, n_perm=10).shape == (10, 6, 4, 16)
    rot = random_rotation_null(x, n_perm=10)
    assert np.allclose(np.linalg.norm(rot, axis=(-2, -1)),
                       np.linalg.norm(x, axis=(-2, -1)), rtol=1e-4)
    print("  rank permutation + norm-preserving rotation                   OK")
    print("\nnull harness self-tests passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        print(__doc__)
        sys.exit(0)
