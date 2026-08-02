"""B2-1 / G8 / G10: the Stage B capture.

Subsumes capture_contrast.py, which is kept as history (it hooks in_proj and
expands ngroups=1 across heads, producing 64 identical copies, and compares
Grams with the diagonal included, which pins the metric near 1.0 regardless of
the data). Neither mistake is repeated here.

WHAT IS OBSERVABLE FROM PREFILL HOOKS, AND WHAT IS NOT
  Observable: everything derived from in_proj (z, x, B, C, dd_dt, dd_A, trap,
    angles), the reconstructed post-norm/post-bias B/C, all recurrence
    quantities, D*x feedthrough, the gated pre-out_proj output (via a
    forward_pre_hook on out_proj), the block residual update, and the
    post-SwiGLU hidden.
  NOT observable: the SSM state h_t, and the per-rank output y_r before the
    mimo_o collapse. Both live inside the kernel. Per-rank OUTPUT contribution
    (contract 3.3) is therefore NOT captured here; what is captured is per-rank
    INJECTION (V_r = x * mimo_x, K_r = B_eff) and the per-rank gate
    (z_r = z * mimo_z). State-side quantities require step()/reference and are
    gated on gpu_probe G3.

BLOCK ISOLATION (schema v3, Stage 4A)
  Every block -- a Stream B prompt, or a Stream A document/chunk -- is forwarded
  as its OWN sequence of shape (1, valid_len), with its own BOS and a fresh
  recurrent state. No padding, no length bucketing, no concatenation of adjacent
  blocks. The artifact stores flat ids plus exact offsets; block i is
  ids[offsets[i]:offsets[i+1]].

  This SUPERSEDES the previous right-padding scheme. That scheme bucketed and
  padded prompts to keep kernel shapes stable, and had to DROP each block's
  final real position because trap_scale_t reads Delta_{t+1}, which under right
  padding was a pad token. With isolated unpadded blocks the kernel's own
  end-of-sequence rule applies, so the final position is legitimate and is now
  forwarded and accumulated. Positions are tagged bos/interior/final so Stage 4B
  can revisit boundary semantics without recapturing.

  Cost: one forward per block instead of one per batch. Shape variety may
  retrigger the JIT (gpu_probe g4); that is measured, not assumed, and batching
  is deliberately deferred rather than reintroduced here.

WITHHELD
  The injection-weighted B/C metrics are INVALID and are not computed or
  written; see WITHHELD_METRICS below. Stage 4B repairs them.

Usage:
  python analysis/capture_stage_b.py --model mimo-187m --stream b \\
      --token-contract token_contract.npz --out capture_187m_b.npz
"""

import argparse
import hashlib
import math
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from accumulators import (  # noqa: E402
    BlockRecorder, GramAccum, Histogram, StreamingMoments, make_edges,
)
from mamba3_core import (  # noqa: E402
    CheckpointResolveError, InProjSpec, assert_runtime, differential_norm_ratio,
    rank_split, reconstruct_bc, recurrence_quantities, split_in_proj,
    resolve_checkpoint,
)
from reference_recurrence import (  # noqa: E402
    apply_rope_split_half, official_gate, phase_details, phase_from_raw,
)

SUPPORTED_SCHEMA_VERSIONS = (3,)

# ---------------------------------------------------------------------------
# WITHHELD: the injection-weighted B/C path is INVALID and must not produce
# reportable output. It applies apply_rope_split_half to the RAW angle slice,
# but the official phase is tanh(raw) * pi * Delta accumulated INCLUSIVELY with
# carried phase and reduced mod 2*pi (angle_dt.py L94-L117). The raw slice is
# neither transformed nor cumulative, so every value it produced was wrong.
# Stage 4B repairs it. Until then it is not computed and not written.
# ---------------------------------------------------------------------------
INJECTION_METRICS_VALID = False
# Stage 4B1: phase/concentration are NO LONGER withheld -- they are replaced by
# the source-derived cumulative phase (tanh(raw)*pi*Delta, inclusive cumsum,
# wrapped mod 2pi), validated by the phase fixtures below. The injection-weighted
# B/C metrics REMAIN withheld; Stage 4B2 constructs those.
WITHHELD_METRICS = () if INJECTION_METRICS_VALID else (
    "bc_diff_injection", "gram_injection",
    # the same-token diagonal scales the UNROTATED B/C product by gamma AFTER
    # the product is formed; gamma cannot be assigned to B or C separately, so
    # the diagonal's rank contraction must be reconstructed first
    "diagonal_rank_utilization",
    # the full recurrent path needs C at the LATER read position, the
    # intervening alpha products, state accumulation, gate and output collapse
    "full_recurrent_utilization",
    # the strictly-earlier path is NOT available per rank from prefill hooks;
    # only its COLLAPSED residual is derivable
    "earlier_per_rank",
    "kernel_state",
    "diagonal_causal_attribution",
    "percentage_pathway_attribution",
)
WITHHELD_REASON = {
    "bc_diff_injection/gram_injection": (
        "UPDATED Stage 4B2C: raw phase is NO LONGER the blocker -- the "
        "cumulative phase is source-derived and B_rot_write/C_rot_read are "
        "constructed correctly. Full utilization stays withheld because "
        "later-position C, the intervening alpha products, the state "
        "accumulation, the gate/collapse and kernel parity are not JOINTLY "
        "observed from prefill hooks."),
    "earlier_per_rank/full_recurrent_utilization/kernel_state": (
        "the strictly-earlier recurrent contribution is not observable per rank "
        "from prefill hooks. Only earlier_collapsed_residual is derivable, and "
        "it is a RESIDUAL that absorbs any reconstruction mismatch until G2 "
        "parity passes -- never a measured state or a per-rank decomposition."),
    "diagonal_causal_attribution/percentage_pathway_attribution": (
        "the collapsed paths can CANCEL, so norm ratios are not additive "
        "attribution fractions and no percentage contribution is emitted."),
}
PHASE_SUPERSEDED = (
    "phase/concentration were withheld in Stage 4A because phase_stats() "
    "consumed the RAW angle slice. Stage 4B1 replaces them with the "
    "source-derived cumulative phase. STATUS: source-derived candidate; "
    "interpretation gated on G2 elementwise tensor parity.")

# Every quantity carries one of these. Only the first two are true today.
EVIDENCE_STATUS = {
    "source_derived": "transcribed from pinned upstream source",
    "fixture_tested": "internally validated by offline fixtures in this file",
    "gpu_parity_validated": "NOT ACHIEVED for any quantity in this file",
}

# scalars recorded per (block, layer, head): the substrate for every null
_ALL_BLOCK_QUANTITIES = (
    "lambda", "alpha", "Delta", "trap_scale", "local_halflife",
    # INITIALIZATION-COORDINATE diagnostics (RANK_GAUGE_NOTE), B and C separately
    "B_diff_shared", "B_diff_post", "C_diff_shared", "C_diff_post",
    "B_participation_post", "C_participation_post",
    # alignment, never "synergy"
    "bc_common_cos_post", "bc_diff_cos_post", "bc_total_cos_post",
    # write side (B) and read side (C). No C*trap_scale, no per-tensor gamma.
    "B_rot_write_norm", "C_rot_read_norm", "B_write_weighted_norm",
    # the ACTUAL state-coordinate RoPE must leave the rank Gram unchanged
    "rope_rank_gram_dev_B", "rope_rank_gram_dev_C",
    "bc_diff_injection",
    # --- Stage 4B2C token-local rank pathways ---
    "value_norm", "V_rank_norm",
    "gate_pre_mean", "gate_pre_std", "gate_factor_mean", "gate_factor_std",
    "gate_factor_frac_near_zero",
    "diag_pre_gate_norm", "D_pre_gate_norm",
    "diag_collapsed_norm", "D_collapsed_norm",
    "actual_pre_out_norm", "earlier_collapsed_residual_norm",
    "align_diag_D", "align_diag_residual", "align_D_residual",
    "align_diag_D_valid", "align_diag_residual_valid", "align_D_residual_valid",
    "resid_absmean", "mlp_frac_near_zero",
)
BLOCK_QUANTITIES = tuple(q for q in _ALL_BLOCK_QUANTITIES
                         if q not in WITHHELD_METRICS)

HIST_QUANTITIES = tuple(q for q in
                        ("lambda", "alpha", "Delta", "trap_scale", "local_halflife")
                        if q not in WITHHELD_METRICS)

NEAR_ZERO_EPS = 0.01  # relative to the tensor's own std; see contract 3.5

TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Stage 4B1: source-derived recurrence scalars and cumulative phase
# ---------------------------------------------------------------------------
# SCALARS come from mamba3_core.recurrence_quantities(); this file adds NO
# second implementation. Name mapping to the Stage 4B1 vocabulary:
#     Delta          <- "Delta"          softplus(dd_dt + dt_bias)
#     A              <- "A"              -heavy_tail_activation(dd_A), clamped
#     log_alpha      <- "ADT"            A * Delta   (PRIMARY decay quantity)
#     alpha          <- "alpha"          exp(log_alpha)  (derived diagnostic)
#     lambda         <- "lambda"         sigmoid(trap)
#     gamma          <- "gamma"          lambda * Delta
#     shifted_gamma  <- "shifted_gamma"  Delta[t+1] * (1 - lambda[t+1]), 0 at end
#     trap_scale     <- "trap_scale"     gamma + shifted_gamma
#     eff_half_life  <- "local_halflife" ln2 / -log_alpha
SCALAR_MAP = {
    "Delta": "Delta", "A": "A", "log_alpha": "ADT", "alpha": "alpha",
    "lambda": "lambda", "gamma": "gamma", "shifted_gamma": "shifted_gamma",
    "trap_scale": "trap_scale", "eff_half_life": "local_halflife",
}
SCALAR_QUANTITIES = tuple(SCALAR_MAP)

# DISTRIBUTIONAL COVERAGE. Every recurrence quantity gets a fixed-bin histogram.
# accumulators.DEFAULT_EDGES has no dedicated bins for gamma, shifted_gamma,
# log_alpha or A, and editing that file is out of scope, so declared reuses are
# recorded here. Bins are FIXED in advance and never chosen from captured data.
#   gamma, shifted_gamma -> "Delta" bins. Both are Delta-scaled: gamma =
#       lambda*Delta in (0, Delta], shifted = Delta'*(1-lambda') in [0, Delta').
#   decay_rate = -log_alpha  -> "Delta" bins (positive, same log decade range)
#   neg_A      = -A          -> "Delta" bins (positive)
# log_alpha and A are signed, so they keep MOMENTS and are represented
# distributionally through their positive transforms.
SCALAR_HIST_EDGES = {"Delta": "Delta", "alpha": "alpha", "lambda": "lambda",
                     "trap_scale": "trap_scale", "eff_half_life": "local_halflife",
                     "gamma": "Delta", "shifted_gamma": "Delta"}
DERIVED_POSITIVE = {"decay_rate": ("log_alpha", "Delta"),
                    "neg_A": ("A", "Delta")}
HIST_BIN_REUSE = {
    "gamma": "Delta bins: gamma = lambda*Delta, same units",
    "shifted_gamma": "Delta bins: Delta'*(1-lambda'), same units",
    "decay_rate": "Delta bins; decay_rate = -log_alpha, positive transform of a "
                  "signed quantity that keeps moments",
    "neg_A": "Delta bins; neg_A = -A, positive transform of a signed quantity "
             "that keeps moments",
}

# half-life is censored where alpha -> 1 (log_alpha -> 0): retention diverges and
# any mean is meaningless. Censored observations are COUNTED, never clipped in.
HALFLIFE_CENSOR_EPS = 1e-6

POSITION_ROLES = ("bos", "interior", "final")

# declared threshold for "gate factor is near zero". SiLU is UNBOUNDED above and
# has a small negative lobe; this counts near-zero multipliers, and is NOT a
# saturation claim. The gate acts independently PER RANK, before the rank sum.
GATE_NEAR_ZERO_EPS = 1e-3

# label count used only for the pre-capture size projection (17 balanced
# semantic labels + 9 non-balanced source categories)
EST_N_LABELS = 26


def token_local_paths(x, z, B_post, C_post, gamma, D, mimo_x, mimo_z, mimo_o,
                      is_mimo):
    """Token-local (vectorizable) rank pathways. No recurrent state involved.

    x, z:          (T, H, P)
    B_post/C_post: (T, H, R, N)   UNROTATED post-bias
    gamma:         (T, H)
    D:             (H,)
    mimo_*:        (H, R, P) or None for SISO

    The strictly-earlier recurrent path is NOT computed here -- it is not
    available per rank from prefill hooks.
    """
    T, H, P = x.shape
    if is_mimo:
        V_rank = torch.einsum("thp,hrp->thrp", x, mimo_x)
        gate_pre = torch.einsum("thp,hrp->thrp", z, mimo_z)
        collapse_weight = mimo_o
    else:
        V_rank = x.unsqueeze(2)
        gate_pre = z.unsqueeze(2)
        collapse_weight = torch.ones(H, 1, P, dtype=x.dtype, device=x.device)
    gate_factor = official_gate(gate_pre)

    # DIAGONAL: unrotated B/C, gamma alone. No trap_scale, no rotated tensors.
    qk_diag = torch.einsum("thrn,thsn->thrs", C_post, B_post)
    y_diag = torch.einsum("thrs,thsp->thrp", qk_diag, V_rank) \
        * gamma.unsqueeze(-1).unsqueeze(-1)
    # D on the RANK-EXPANDED value, before gate and collapse
    y_D = D.view(1, H, 1, 1) * V_rank

    diag_post_gate_r = y_diag * gate_factor
    D_post_gate_r = y_D * gate_factor
    diag_collapse_r = diag_post_gate_r * collapse_weight
    D_collapse_r = D_post_gate_r * collapse_weight
    return {
        "V_rank": V_rank, "gate_pre": gate_pre, "gate_factor": gate_factor,
        "collapse_weight": collapse_weight, "qk_diag": qk_diag,
        "y_diag": y_diag, "y_D": y_D,
        "diag_post_gate_r": diag_post_gate_r, "D_post_gate_r": D_post_gate_r,
        "diag_collapse_r": diag_collapse_r, "D_collapse_r": D_collapse_r,
        "diag_collapsed": diag_collapse_r.sum(dim=2),
        "D_collapsed": D_collapse_r.sum(dim=2),
    }


def _cos3(a, b, eps=1e-12):
    """Cosine over the last axis of two (T, H, P) tensors, zero where invalid."""
    na, nb = a.norm(dim=-1), b.norm(dim=-1)
    ok = (na > eps) & (nb > eps)
    c = (a * b).sum(-1) / (na * nb).clamp_min(eps)
    return torch.where(ok, c, torch.zeros_like(c))


def _cos_valid(a, b, eps=1e-12):
    return ((a.norm(dim=-1) > eps) & (b.norm(dim=-1) > eps)).float()

# HELPER GAP 2 (reported): reference_recurrence.phase_from_raw returns only the
# WRAPPED phase and the carry. The per-token increment and the UNWRAPPED
# cumulative phase are both required here and are not exposed. Rather than edit
# reference_recurrence.py (out of scope this stage), the increment is formed
# here and the helper's wrapped phase is ASSERTED equal on the circle, so the
# two cannot diverge silently. Exposing increment/unwrapped from the helper is
# the correct follow-up.
PHASE_HELPER_GAP = (
    "RESOLVED in Stage 4B2C: reference_recurrence.phase_details now exposes "
    "increment, unwrapped, wrapped and carry_out, so capture holds NO local "
    "transcription of the phase formula. phase_objects() is a thin adapter.")


# ---------------------------------------------------------------------------
# Stage 4B2A: B is the WRITE/key side, C is the READ/query side
# ---------------------------------------------------------------------------
# Source-derived roles (mamba3.py inference calls the rotary with q=C, k=B; the
# recurrence injects B_t into the state and READS it later with C_t; the
# off-diagonal prefix is key/value accumulation, therefore write-side):
#
#     B  key / write   -> accumulated into the recurrent prefix
#     C  query / read  -> reads that accumulated state at a LATER position
#
# Enforced below:
#   * trap_scale weights the OFF-DIAGONAL WRITE, so it multiplies B. There is no
#     object "C * trap_scale" written into the prefix; that claim is RETRACTED.
#   * gamma scales the same-token UNROTATED B/C product AFTER the product is
#     formed. It is not a per-tensor weight, so "B*gamma" and "C*gamma" are not
#     constructed and diagonal utilization stays withheld.
BC_STAGES = ("B_shared", "C_shared", "B_post", "C_post",
             "B_rot_write", "C_rot_read", "B_offdiag_write_weighted")

B_OFFDIAG_CAVEAT = (
    "B_offdiag_write_weighted = B_rot_write * trap_scale is a WRITE-SIDE "
    "diagnostic only. It omits C at the later read position, the intervening "
    "alpha decay products, the state accumulation itself, the gate, and the "
    "output collapse. It is NOT the recurrent contribution.")

RANK_GAUGE_NOTE = (
    "rank-common / rank-differential quantities are INITIALIZATION-COORDINATE "
    "diagnostics, not basis-invariant utilization. The rank-common direction is "
    "privileged by the all-ones initialization and the parameterization, not by "
    "any gauge invariance. A functional gauge transformation would have to "
    "rotate every rank-coupled path consistently -- mimo_x, mimo_z, mimo_o and "
    "B/C together -- which belongs to Stage C.")

# Behaviour under a SIMULTANEOUS orthogonal rotation of the RANK basis. This is
# a fixture-only probe of coordinate dependence, NOT the model's RoPE.
RANK_GAUGE_BEHAVIOUR = {
    "invariant": (
        "singular values / Gram eigenvalues",
        "participation ratio",
        "total Frobenius norm",
        "flattened B/C cosine when B and C transform TOGETHER",
    ),
    "variant": (
        "named per-slot norms",
        "individual Gram entries",
        "rank-common / rank-differential split (defined against the fixed "
        "all-ones initialization direction)",
        "signed per-rank cosine",
    ),
}


def partial_rope_bc(x, phase):
    """Apply the model's partial RoPE over STATE COORDINATES.

    x:     (T, H, R, N)   post-bias B or C
    phase: (T, H, K)      wrapped cumulative phase, K = configured pair count

    Returns (T, H, R, N). The rotation acts on the state-coordinate axis N,
    pairing index k with k + N//2 for k < K. The RANK axis is untouched and
    every rank slot receives the SAME rotation.
    """
    T, H, R, N = x.shape
    if tuple(phase.shape[:2]) != (T, H):
        raise ValueError(f"phase {tuple(phase.shape)} does not match x {(T, H)}")
    K = phase.shape[-1]
    if 2 * K > N:
        raise ValueError(f"{K} rotary pairs need {2 * K} coordinates, have {N}")
    ang = phase.unsqueeze(2).expand(T, H, R, K)
    return apply_rope_split_half(x, ang)


def rank_geometry(x, eps=1e-12, include_eigvals=True):
    """Geometry over the RANK axis of x: (..., R, N).

    gram             (..., R, R)  raw rank Gram
    eigvals          (..., R)     ascending eigenvalues of that Gram
    participation    (..., )      (sum L)^2 / sum L^2, in [1, R]
    total_norm       (..., )      ||x||_F
    common_norm      (..., )      sqrt(R) * ||mean_r x||
    diff_norm        (..., )      ||x - mean_r x||_F
    diff_over_total  (..., )      unsquared ratio
    diff_energy      (..., )      squared-energy ratio

    total^2 = common^2 + diff^2 exactly. At R == 1 the differential part is
    identically zero and the participation ratio is exactly 1.
    """
    R = x.shape[-2]
    gram = x @ x.transpose(-1, -2)
    # Participation does not require an eigensolve:
    #   (sum lambda)^2 / sum(lambda^2) == trace(G)^2 / ||G||_F^2
    # for this symmetric PSD Gram. The former implementation nevertheless ran
    # five tiny fp64 eigensolvers per block/layer in the hot path, even where
    # callers only consumed norms, ratios, or the Gram itself.
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1)
    part = trace.square() / gram.square().sum(dim=(-2, -1)).clamp_min(eps)
    total = x.norm(dim=(-2, -1))
    mean = x.mean(-2, keepdim=True)
    diff = x - mean
    common = mean.norm(dim=(-2, -1)) * math.sqrt(R)
    dn = diff.norm(dim=(-2, -1))
    out = {
        "gram": gram, "participation": part.to(x.dtype),
        "total_norm": total, "common_norm": common, "diff_norm": dn,
        "diff_over_total": dn / total.clamp_min(eps),
        "diff_energy": (dn ** 2) / (total ** 2).clamp_min(eps),
    }
    if include_eigvals:
        # Captured tensors originate in bf16 and summaries are stored in f16;
        # fp32 is the evidence-appropriate precision here. fp64 routed tens of
        # thousands of 4x4 systems through a slow/fragile Ada cuSolver path.
        out["eigvals"] = torch.linalg.eigvalsh(gram.float()).clamp_min(0).to(x.dtype)
    return out


def bc_alignment(B, C, eps=1e-12):
    """Alignment (NEVER "synergy") between matched B and C at one stage.

    B, C: (..., R, N). Returns cosines plus validity masks for zero norms.
    """
    def flat(t):
        return t.reshape(*t.shape[:-2], -1)

    def cos(u, v):
        nu, nv = u.norm(dim=-1), v.norm(dim=-1)
        ok = (nu > eps) & (nv > eps)
        c = (u * v).sum(-1) / (nu * nv).clamp_min(eps)
        return torch.where(ok, c, torch.zeros_like(c)), ok

    dB = B - B.mean(-2, keepdim=True)
    dC = C - C.mean(-2, keepdim=True)
    common_cos, common_ok = cos(B.mean(-2), C.mean(-2))
    diff_cos, diff_ok = cos(flat(dB), flat(dC))
    total_cos, total_ok = cos(flat(B), flat(C))
    per_rank_cos, per_rank_ok = cos(B, C)   # (..., R) signed, basis-dependent
    return {
        "common_cos": common_cos, "common_valid": common_ok,
        "diff_cos": diff_cos, "diff_valid": diff_ok,
        "total_cos": total_cos, "total_valid": total_ok,
        "per_rank_cos": per_rank_cos, "per_rank_valid": per_rank_ok,
    }


def random_rank_rotation(rank, generator=None, device="cpu"):
    """FIXTURE-ONLY random orthogonal map on the RANK axis.

    A GAUGE probe, not a model operation: the model's RoPE acts on state
    coordinates, not on rank. It must never appear in the production capture
    loop -- it is stochastic, costs a QR per call, and measures nothing the
    kernel does. See RANK_GAUGE_NOTE.
    """
    a = torch.randn(rank, rank, generator=generator, device=device)
    return torch.linalg.qr(a)[0]


class GeometryCapture:
    """Rank-geometry accumulators for ONE named B/C stage object."""

    # cuSolver on Ada rejects very large batches of tiny fp64 eigensystems.
    # Chunk only this R x R solve; all token-level geometry and accumulator
    # updates remain batched. 64 tokens means at most 64 * H independent 4x4
    # systems per call for the released MIMO checkpoints.
    EIGH_TOKEN_CHUNK = 64

    def __init__(self, nheads, rank, device):
        self.gram = GramAccum(nheads, rank, device)
        self.scalars = {k: StreamingMoments((1, nheads), device=device)
                        for k in ("participation", "total_norm", "common_norm",
                                  "diff_norm", "diff_over_total", "diff_energy")}
        self.eigvals = StreamingMoments((nheads, rank), device=device)

    def update(self, x):
        """x: (T, H, R, N)."""
        self.gram.update(x)
        chunks = [rank_geometry(part)
                  for part in x.split(self.EIGH_TOKEN_CHUNK, dim=0)]
        for k, m in self.scalars.items():
            values = torch.cat([g[k] for g in chunks], dim=0)
            m.update(values.T.unsqueeze(0))
        eigvals = torch.cat([g["eigvals"] for g in chunks], dim=0)
        self.eigvals.update(eigvals.permute(1, 2, 0))

    def finish(self):
        out = {"gram": self.gram.finish()}
        for k, m in self.scalars.items():
            f = m.finish()
            out[f"{k}/mean"] = f["mean"]
            out[f"{k}/std"] = f["std"]
        f = self.eigvals.finish()
        out["eigvals/mean"] = f["mean"]
        out["eigvals/std"] = f["std"]
        return out


class LayerCapture:
    """All accumulators for one layer, one class."""

    def __init__(self, nheads, rank, device, n_layers_idx):
        self.moments = {
            q: StreamingMoments((1, nheads), device=device) for q in BLOCK_QUANTITIES
        }
        self.hists = {
            q: Histogram((1, nheads), make_edges(q), device=device) for q in HIST_QUANTITIES
        }
        # three distinct B/C objects (contract 3.2)
        self.gram_shared = GramAccum(1, rank, device)       # pre-bias, shared
        self.gram_posbias = GramAccum(nheads, rank, device)  # per head
        # only constructed when the path is valid; see WITHHELD_METRICS
        self.gram_injection = (GramAccum(nheads, rank, device)
                               if INJECTION_METRICS_VALID else None)
        # WITHHELD: raw-angle circular statistics are not the model's phase
        self.phase = None

        # Stage 4B2A: per-stage rank geometry. B is the WRITE/key side, C the
        # READ/query side (BC_STAGES). No "C * trap_scale" object exists, and
        # gamma is never applied per-tensor -- it scales the same-token
        # UNROTATED B/C product AFTER the product is formed.
        self.geom = {name: GeometryCapture(1 if name.endswith("_shared") else nheads,
                                           rank, device)
                     for name in BC_STAGES}
        # B/C alignment (cosine, NEVER "synergy") at the two comparable stages
        self.align = {}
        for stage, Hs in (("shared", 1), ("post", nheads)):
            self.align[stage] = {
                k: StreamingMoments((1, Hs), device=device)
                for k in ("common_cos", "diff_cos", "total_cos",
                          "common_valid", "diff_valid", "total_valid")
            }
            self.align[stage]["per_rank_cos"] = StreamingMoments(
                (Hs, rank), device=device)

    def finish(self):
        out = {}
        for q, m in self.moments.items():
            f = m.finish()
            out[f"{q}/mean"] = f["mean"]
            out[f"{q}/absmean"] = f["absmean"]
        for q, h in self.hists.items():
            for qq in (0.1, 0.25, 0.5, 0.75, 0.9):
                out[f"{q}/p{int(qq * 100)}"] = h.quantile(qq).cpu().numpy()
            out[f"{q}/censored"] = h.censored_fraction().cpu().numpy()
        out["gram_shared"] = self.gram_shared.finish()
        out["gram_posbias"] = self.gram_posbias.finish()
        if self.gram_injection is not None:
            out["gram_injection"] = self.gram_injection.finish()
        # Stage 4B2A outputs: per-stage rank geometry and B/C alignment
        for name, gc in self.geom.items():
            for k, v in gc.finish().items():
                out[f"{name}/{k}"] = v
        for stage, accs in self.align.items():
            for k, m in accs.items():
                f = m.finish()
                out[f"align_{stage}/{k}/mean"] = f["mean"]
                out[f"align_{stage}/{k}/count"] = f["count"]
        return out


# per-block, per-role, per-head summaries. These are what a document bootstrap,
# matched-position analysis or same-class null actually resamples; the online
# RoleCapture pools across every block sharing a label and cannot support them.
BLOCK_ROLE_SCALARS = ("lambda", "Delta", "A", "log_alpha", "alpha", "gamma",
                      "shifted_gamma", "trap_scale")
BLOCK_ROLE_PHASE = ("phase_concentration", "phase_increment_mean",
                    "phase_increment_absmean", "phase_unwrapped_change")
BLOCK_ROLE_HL = ("eff_half_life_mean", "eff_half_life_finite_count",
                 "eff_half_life_censored_frac")
BLOCK_ROLE_BC = (
    # Every name below is produced by the live `vals` mapping. Retired
    # trap/gamma-per-side and random-rank-rotation diagnostics must not survive
    # here as silently zero-filled arrays.
    "B_diff_shared", "B_diff_post", "C_diff_shared", "C_diff_post",
    "B_participation_post", "C_participation_post",
    "bc_common_cos_post", "bc_diff_cos_post", "bc_total_cos_post",
    "B_rot_write_norm", "C_rot_read_norm", "B_write_weighted_norm",
    "rope_rank_gram_dev_B", "rope_rank_gram_dev_C",
)
BLOCK_ROLE_FIELDS = BLOCK_ROLE_SCALARS + BLOCK_ROLE_PHASE + BLOCK_ROLE_HL + BLOCK_ROLE_BC
# winding is per (block, layer, head), summarized OVER rotary pairs -- heads are
# never collapsed into one scalar
BLOCK_WINDING_FIELDS = ("winding_mean_over_pairs", "winding_absmean_over_pairs",
                        "winding_std_over_pairs", "winding_maxabs_over_pairs")

# Stage 4B2C: rank-resolved local-pathway summaries, (block, layer, role, head, rank)
BLOCK_RANK_FIELDS = ("V_rank_norm", "gate_pre_mean", "gate_pre_std",
                     "gate_factor_mean", "gate_factor_std",
                     "diag_pre_gate_norm", "D_pre_gate_norm",
                     "diag_collapsed_contrib_norm", "D_collapsed_contrib_norm")
# collapsed-path summaries, (block, layer, role, head)
BLOCK_COLLAPSED_FIELDS = ("actual_pre_out_norm", "diag_collapsed_norm",
                          "D_collapsed_norm", "earlier_collapsed_residual_norm",
                          "align_diag_D", "align_diag_residual", "align_D_residual",
                          "align_diag_D_valid", "align_diag_residual_valid",
                          "align_D_residual_valid")

BLOCK_SUMMARY_DTYPE = np.float16   # summary statistics, not raw measurements
BLOCK_ROLE_FIELD_DTYPES = {
    # Finite is not the same as float16-representable. SISO exposes legitimate
    # uncensored effective half-lives above 65,504 recurrence steps; preserving
    # them in float32 is cheaper and more honest than clipping or writing inf.
    "eff_half_life_mean": np.float32,
}


def block_role_field_dtype(field):
    return BLOCK_ROLE_FIELD_DTYPES.get(field, BLOCK_SUMMARY_DTYPE)


def estimate_artifact_bytes(n_blocks, n_layers, n_heads, n_roles=len(POSITION_ROLES),
                            rank=1, n_labels=1):
    """Projected COMPLETE payload size, before anything is captured.

    Covers every contributor, so no partial figure is ever labelled "total":
      * legacy BlockRecorder            (block, layer, head) pooled quantities
      * Stage 4B1 role arrays           (block, layer, role, head)
      * Stage 4B2C rank-path summaries  (block, layer, role, head, RANK)
      * Stage 4B2C collapsed summaries  (block, layer, role, head)
      * Stage 4B1 winding               (block, layer, head)
      * Stage 4B2A geometry summaries   per (label, layer), NOT per block
    """
    itemsize = np.dtype(BLOCK_SUMMARY_DTYPE).itemsize
    role_cells = n_blocks * n_layers * n_roles * n_heads
    wind_cells = n_blocks * n_layers * n_heads
    role_b = role_cells * sum(np.dtype(block_role_field_dtype(f)).itemsize
                              for f in BLOCK_ROLE_FIELDS)
    wind_b = wind_cells * len(BLOCK_WINDING_FIELDS) * itemsize
    rank_b = role_cells * rank * len(BLOCK_RANK_FIELDS) * itemsize
    collapsed_b = role_cells * len(BLOCK_COLLAPSED_FIELDS) * itemsize
    legacy_b = (n_blocks * n_layers * n_heads
                * len(BLOCK_QUANTITIES) * np.dtype(np.float32).itemsize)
    # per-stage geometry lives per (label, layer): grams (H,R,R) + scalars
    geom_b = (n_labels * n_layers * len(BC_STAGES)
              * (n_heads * rank * rank + 14 * n_heads) * 8)
    # counts and validity are per (block, layer, role) -- NOT per head
    count_b = n_blocks * n_layers * n_roles * (np.dtype(np.int32).itemsize + 1)
    parts = {"legacy_block_bytes": legacy_b, "role_summaries_bytes": role_b,
             "rank_path_bytes": rank_b, "collapsed_path_bytes": collapsed_b,
             "winding_bytes": wind_b, "counts_bytes": count_b,
             "geometry_bytes": geom_b}
    complete = sum(parts.values())
    return {**parts,
            "complete_payload_bytes": complete,
            "complete_payload_gib": complete / (1 << 30),
            "total_bytes": complete, "total_gib": complete / (1 << 30),
            "rank": rank, "n_labels": n_labels,
            "dtype": str(np.dtype(BLOCK_SUMMARY_DTYPE)),
            "field_dtype_overrides": {
                f: str(np.dtype(dt)) for f, dt in BLOCK_ROLE_FIELD_DTYPES.items()
            },
            "n_blocks": n_blocks, "n_layers": n_layers, "n_heads": n_heads,
            "n_roles": n_roles, "n_role_fields": len(BLOCK_ROLE_FIELDS)}


class BlockRoleRecorder:
    """Preallocated (block, layer, role, head[, rank]) summaries. Bounded and known.

    Device-side staging: every put_* writes into device tensors; flush(bi)
    performs ONE batched D2H copy per block for ALL families (role, winding,
    rank, collapsed, counts). This replaces the per-field .cpu().numpy() calls,
    which were ~700 individual GPU synchronizations per block.

    Rank-resolved arrays preserve (head, rank): reductions happen over P only,
    never over the rank axis, and are populated ONLY from the rank-resolved
    tensors (V_rank, gate_pre/gate_factor, y_diag/y_D, diag/D_collapse_r) --
    never from the rank-collapsed legacy fields, which are kept separately
    for compatibility.

    Collapsed alignment fields are cosine means over VALID tokens only. The
    matching *_valid field records the fraction of valid token positions in the
    role; a fully invalid cell is (value 0, validity 0), an explicit
    zero-observation mark rather than a fabricated zero cosine.
    """

    _COLLAPSED_NORMS = ("actual_pre_out_norm", "diag_collapsed_norm",
                        "D_collapsed_norm", "earlier_collapsed_residual_norm")
    _ALIGN_PAIRS = tuple((f, f + "_valid") for f in
                         ("align_diag_D", "align_diag_residual", "align_D_residual"))

    def __init__(self, n_blocks, n_layers, n_heads, rank=1, device="cpu"):
        self.device = device
        self.rank = rank
        R3 = len(POSITION_ROLES)
        shape = (n_blocks, n_layers, R3, n_heads)
        self.a = {f: np.zeros(shape, block_role_field_dtype(f))
                  for f in BLOCK_ROLE_FIELDS}
        self.w = {f: np.zeros((n_blocks, n_layers, n_heads), BLOCK_SUMMARY_DTYPE)
                  for f in BLOCK_WINDING_FIELDS}
        self.r = {f: np.zeros(shape + (rank,), BLOCK_SUMMARY_DTYPE)
                  for f in BLOCK_RANK_FIELDS}
        self.c = {f: np.zeros(shape, BLOCK_SUMMARY_DTYPE)
                  for f in BLOCK_COLLAPSED_FIELDS}
        self.count = np.zeros((n_blocks, n_layers, R3), np.int32)
        self.valid = np.zeros((n_blocks, n_layers, R3), bool)
        self.n_layers, self.n_heads = n_layers, n_heads
        self.d2h = 0
        # per-block device staging; zeroed at every flush
        self._sa = {f: torch.zeros((n_layers, R3, n_heads), dtype=torch.float32,
                                   device=device) for f in BLOCK_ROLE_FIELDS}
        self._sw = {f: torch.zeros((n_layers, n_heads), dtype=torch.float32,
                                   device=device) for f in BLOCK_WINDING_FIELDS}
        self._sr = {f: torch.zeros((n_layers, R3, n_heads, rank),
                                   dtype=torch.float32, device=device)
                    for f in BLOCK_RANK_FIELDS}
        self._sc = {f: torch.zeros((n_layers, R3, n_heads), dtype=torch.float32,
                                   device=device) for f in BLOCK_COLLAPSED_FIELDS}
        self._sn = torch.zeros((n_layers, R3), dtype=torch.float32, device=device)

    def put_role(self, bi, lp, role_idx, scal, ph, lo, hi, extra=None):
        n = hi - lo
        self._sn[lp, role_idx] = float(n)
        if n <= 0:
            return                       # absent role: count=0, valid=False
        for f in BLOCK_ROLE_SCALARS:
            self._sa[f][lp, role_idx] = scal[f][lo:hi].float().mean(0)
        # Stage 4B2A: B/C role-stratified summaries
        if extra is not None:
            for f in BLOCK_ROLE_BC:
                if f in extra:
                    self._sa[f][lp, role_idx] = extra[f][lo:hi].float().mean(0)

        w = ph["wrapped"][lo:hi]                       # (n, H, K)
        ms, mc = torch.sin(w).mean(0), torch.cos(w).mean(0)
        conc = torch.sqrt(ms * ms + mc * mc).mean(-1)  # over pairs -> (H,)
        inc = ph["increment"][lo:hi]
        self._sa["phase_concentration"][lp, role_idx] = conc
        self._sa["phase_increment_mean"][lp, role_idx] = inc.mean(0).mean(-1)
        self._sa["phase_increment_absmean"][lp, role_idx] = \
            inc.abs().mean(0).mean(-1)
        un = ph["unwrapped"][lo:hi]
        self._sa["phase_unwrapped_change"][lp, role_idx] = (un[-1] - un[0]).mean(-1)

        hl = scal["eff_half_life"][lo:hi]
        keep = ((-scal["log_alpha"][lo:hi]) > HALFLIFE_CENSOR_EPS) & torch.isfinite(hl)
        k = keep.double()
        cnt = k.sum(0)
        mean = torch.where(cnt > 0, (hl.double() * k).sum(0) / cnt.clamp_min(1),
                           torch.zeros_like(cnt))
        self._sa["eff_half_life_mean"][lp, role_idx] = mean
        self._sa["eff_half_life_finite_count"][lp, role_idx] = cnt
        self._sa["eff_half_life_censored_frac"][lp, role_idx] = 1.0 - cnt / float(n)

    def put_rank(self, lp, role_idx, rankvals, lo, hi):
        """rankvals[f]: (T, H, R). Reduce over the role's tokens -> (H, R)."""
        if hi - lo <= 0:
            return
        for f in BLOCK_RANK_FIELDS:
            self._sr[f][lp, role_idx] = rankvals[f][lo:hi].float().mean(0)

    def put_collapsed(self, lp, role_idx, colvals, lo, hi):
        """colvals[f]: (T, H). Alignment cosines average over VALID tokens only."""
        n = hi - lo
        if n <= 0:
            return
        for f in self._COLLAPSED_NORMS:
            self._sc[f][lp, role_idx] = colvals[f][lo:hi].float().mean(0)
        for cf, vf in self._ALIGN_PAIRS:
            m = colvals[vf][lo:hi].float()           # 1 where the token is usable
            ksum = m.sum(0)
            csum = (colvals[cf][lo:hi].float() * m).sum(0)
            # the cosine is 0 at invalid tokens, so csum is the valid-only sum.
            # ksum == 0 -> value 0 AND validity 0: explicitly invalid, never a
            # fabricated zero-cosine observation.
            self._sc[cf][lp, role_idx] = csum / ksum.clamp_min(1)
            self._sc[vf][lp, role_idx] = ksum / float(n)

    def put_winding(self, bi, lp, winding):
        """winding: (H, K) -- summarized over PAIRS, heads preserved."""
        self._sw["winding_mean_over_pairs"][lp] = winding.mean(-1)
        self._sw["winding_absmean_over_pairs"][lp] = winding.abs().mean(-1)
        self._sw["winding_std_over_pairs"][lp] = winding.float().std(-1)
        self._sw["winding_maxabs_over_pairs"][lp] = winding.abs().amax(-1)

    @staticmethod
    def _segmented_roles(x, lengths):
        """Reduce packed token rows to (block, bos/interior/final, ...).

        Blocks are contiguous in ``x``. Prefix sums turn every variable-length
        interior reduction into one indexed CUDA operation; BOS/final are
        gathers. No padding or per-block kernel launch is introduced.
        """
        lengths = lengths.to(device=x.device, dtype=torch.long)
        ends = lengths.cumsum(0)
        starts = ends - lengths
        m = lengths.numel()
        counts = torch.stack((torch.ones_like(lengths),
                              (lengths - 2).clamp_min(0),
                              (lengths > 1).long()), dim=1)
        out = x.new_zeros((m, len(POSITION_ROLES), *x.shape[1:]))
        out[:, 0] = x[starts]
        has_final = lengths > 1
        out[has_final, 2] = x[ends[has_final] - 1]
        prefix = torch.cat((torch.zeros_like(x[:1]), x.cumsum(0)), dim=0)
        interior_start = starts + 1
        interior_end = torch.maximum(ends - 1, interior_start)
        interior_sum = prefix[interior_end] - prefix[interior_start]
        denom = counts[:, 1].clamp_min(1).reshape(
            m, *([1] * (x.dim() - 1)))
        out[:, 1] = interior_sum / denom
        return out, counts

    @staticmethod
    def _segmented_role_sums(x, lengths):
        means, counts = BlockRoleRecorder._segmented_roles(x, lengths)
        scale = counts.reshape(counts.shape[0], counts.shape[1],
                               *([1] * (x.dim() - 1)))
        return means * scale, counts

    def put_batch_layer(self, lp, records):
        """Write one packed forward's block summaries for one selected layer.

        ``records`` retains exact per-block tensors but all reductions happen
        over their concatenation. The complete layer/batch result crosses to
        CPU in one transfer rather than one transfer per block.
        """
        if not records:
            return
        bis = np.asarray([r["row"] for r in records], dtype=np.int64)
        lengths = torch.tensor([r["n"] for r in records], device=self.device,
                               dtype=torch.long)

        def cat(group, field):
            return torch.cat([r[group][field] for r in records], dim=0).float()

        a = {}
        counts = None
        for f in BLOCK_ROLE_SCALARS:
            a[f], counts = self._segmented_roles(cat("scal", f), lengths)
        for f in BLOCK_ROLE_BC:
            a[f], counts = self._segmented_roles(cat("extra", f), lengths)

        wrapped = cat("ph", "wrapped")
        sin_m, counts = self._segmented_roles(torch.sin(wrapped), lengths)
        cos_m, _ = self._segmented_roles(torch.cos(wrapped), lengths)
        a["phase_concentration"] = torch.sqrt(
            sin_m * sin_m + cos_m * cos_m).mean(-1)
        inc_m, _ = self._segmented_roles(cat("ph", "increment"), lengths)
        inc_abs_m, _ = self._segmented_roles(
            cat("ph", "increment").abs(), lengths)
        a["phase_increment_mean"] = inc_m.mean(-1)
        a["phase_increment_absmean"] = inc_abs_m.mean(-1)
        un = cat("ph", "unwrapped")
        ends = lengths.cumsum(0)
        starts = ends - lengths
        un_change = un.new_zeros((len(records), len(POSITION_ROLES),
                                  un.shape[1]))
        has_interior = lengths > 2
        un_change[has_interior, 1] = (
            un[ends[has_interior] - 2]
            - un[starts[has_interior] + 1]).mean(-1)
        a["phase_unwrapped_change"] = un_change

        hl = cat("scal", "eff_half_life")
        log_alpha = cat("scal", "log_alpha")
        keep = ((-log_alpha) > HALFLIFE_CENSOR_EPS) & torch.isfinite(hl)
        finite_sum, _ = self._segmented_role_sums(
            torch.where(keep, hl, torch.zeros_like(hl)), lengths)
        finite_count, _ = self._segmented_role_sums(keep.float(), lengths)
        a["eff_half_life_mean"] = finite_sum / finite_count.clamp_min(1)
        a["eff_half_life_finite_count"] = finite_count
        role_count = counts.to(hl.dtype).unsqueeze(-1)
        a["eff_half_life_censored_frac"] = torch.where(
            role_count > 0, 1.0 - finite_count / role_count.clamp_min(1),
            torch.zeros_like(finite_count))

        rr = {f: self._segmented_roles(cat("rankvals", f), lengths)[0]
              for f in BLOCK_RANK_FIELDS}
        cc = {f: self._segmented_roles(cat("colvals", f), lengths)[0]
              for f in self._COLLAPSED_NORMS}
        for cf, vf in self._ALIGN_PAIRS:
            v = cat("colvals", vf)
            c = cat("colvals", cf)
            valid_sum, _ = self._segmented_role_sums(v, lengths)
            cos_sum, _ = self._segmented_role_sums(c * v, lengths)
            cc[cf] = cos_sum / valid_sum.clamp_min(1)
            cc[vf] = torch.where(role_count > 0,
                                 valid_sum / role_count.clamp_min(1),
                                 torch.zeros_like(valid_sum))

        winding = torch.stack([r["winding"] for r in records])
        ww = {
            "winding_mean_over_pairs": winding.mean(-1),
            "winding_absmean_over_pairs": winding.abs().mean(-1),
            "winding_std_over_pairs": winding.float().std(-1),
            "winding_maxabs_over_pairs": winding.abs().amax(-1),
        }

        a_stack = torch.stack([a[f] for f in BLOCK_ROLE_FIELDS])
        w_stack = torch.stack([ww[f] for f in BLOCK_WINDING_FIELDS])
        r_stack = torch.stack([rr[f] for f in BLOCK_RANK_FIELDS])
        c_stack = torch.stack([cc[f] for f in BLOCK_COLLAPSED_FIELDS])
        packet = torch.cat((a_stack.flatten(), w_stack.flatten(),
                            r_stack.flatten(), c_stack.flatten())).cpu().numpy()
        self.d2h += 1
        m, R3, H, R = len(records), len(POSITION_ROLES), self.n_heads, self.rank
        off = 0

        def take(n):
            nonlocal off
            z = packet[off:off + n]
            off += n
            return z

        na = m * R3 * H
        for f in BLOCK_ROLE_FIELDS:
            self.a[f][bis, lp] = take(na).reshape(m, R3, H).astype(
                block_role_field_dtype(f))
        nw = m * H
        for f in BLOCK_WINDING_FIELDS:
            self.w[f][bis, lp] = take(nw).reshape(m, H).astype(
                BLOCK_SUMMARY_DTYPE)
        nr = m * R3 * H * R
        for f in BLOCK_RANK_FIELDS:
            self.r[f][bis, lp] = take(nr).reshape(m, R3, H, R).astype(
                BLOCK_SUMMARY_DTYPE)
        for f in BLOCK_COLLAPSED_FIELDS:
            self.c[f][bis, lp] = take(na).reshape(m, R3, H).astype(
                BLOCK_SUMMARY_DTYPE)
        cnt = counts.cpu().numpy().astype(np.int32)
        self.count[bis, lp] = cnt
        self.valid[bis, lp] = cnt > 0
        if off != packet.size:
            raise ContractError(
                f"batch role packet drifted: consumed {off} of {packet.size}")

    def flush(self, bi):
        """ONE batched D2H per block: every staged family lands in row bi at once."""
        packs = (
            torch.stack([self._sa[f] for f in BLOCK_ROLE_FIELDS]).flatten(),
            torch.stack([self._sw[f] for f in BLOCK_WINDING_FIELDS]).flatten(),
            torch.stack([self._sr[f] for f in BLOCK_RANK_FIELDS]).flatten(),
            torch.stack([self._sc[f] for f in BLOCK_COLLAPSED_FIELDS]).flatten(),
            self._sn.flatten(),
        )
        pack = torch.cat(packs).float().cpu().numpy()
        self.d2h += 1
        R3 = len(POSITION_ROLES)
        off = 0

        def take(n):
            nonlocal off
            seg = pack[off:off + n]
            off += n
            return seg

        na = self.n_layers * R3 * self.n_heads
        for f in BLOCK_ROLE_FIELDS:
            self.a[f][bi] = take(na).reshape(
                self.n_layers, R3, self.n_heads).astype(
                    block_role_field_dtype(f))
        nw = self.n_layers * self.n_heads
        for f in BLOCK_WINDING_FIELDS:
            self.w[f][bi] = take(nw).reshape(
                self.n_layers, self.n_heads).astype(BLOCK_SUMMARY_DTYPE)
        nr = na * self.rank
        for f in BLOCK_RANK_FIELDS:
            self.r[f][bi] = take(nr).reshape(
                self.n_layers, R3, self.n_heads, self.rank).astype(BLOCK_SUMMARY_DTYPE)
        for f in BLOCK_COLLAPSED_FIELDS:
            self.c[f][bi] = take(na).reshape(
                self.n_layers, R3, self.n_heads).astype(BLOCK_SUMMARY_DTYPE)
        cnt = take(self.n_layers * R3).reshape(self.n_layers, R3)
        if (not np.all(cnt[:, 0] == 1)
                or not np.all(cnt == cnt[0:1])):
            raise ContractError(
                "BlockRoleRecorder incomplete block: role counts must agree "
                f"across layers with exactly one BOS; got {cnt.tolist()}")
        self.count[bi] = cnt.astype(np.int32)
        self.valid[bi] = self.count[bi] > 0
        assert off == pack.size, f"staging pack drifted: consumed {off} of {pack.size}"
        for d in (self._sa, self._sw, self._sr, self._sc):
            for t in d.values():
                t.zero_()
        self._sn.zero_()

    def payload(self):
        out = {f"blockrole|{f}": v for f, v in self.a.items()}
        out |= {f"blockwind|{f}": v for f, v in self.w.items()}
        out |= {f"blockrank|{f}": v for f, v in self.r.items()}
        out |= {f"blockcollapsed|{f}": v for f, v in self.c.items()}
        out["blockrole|count"] = self.count
        out["blockrole|valid"] = self.valid
        out["blockrole|roles"] = np.array(POSITION_ROLES)
        return out


class MaskedMoments:
    """Per-head moments over FINITE, UNCENSORED observations only.

    NaN must never reach StreamingMoments or Histogram: a single NaN poisons a
    running mean and, because those accumulators are shaped (1, nheads), it
    would silently contaminate the head it landed on for the rest of the run.
    Censored observations are counted and EXCLUDED, never substituted.

    A head with zero finite observations reports count=0 and valid=False rather
    than a fabricated mean.
    """

    def __init__(self, nheads, device="cpu"):
        z = lambda: torch.zeros(nheads, dtype=torch.float64, device=device)  # noqa: E731
        self.sum, self.sqsum = z(), z()
        self.count, self.censored = z(), z()

    def update(self, v, keep):
        """v: (n, H) finite-where-keep. keep: (n, H) bool."""
        vd = v.double()
        vd = torch.where(keep, vd, torch.zeros_like(vd))
        k = keep.double()
        self.sum += (vd * k).sum(0)
        self.sqsum += (vd * vd * k).sum(0)
        self.count += k.sum(0)
        self.censored += (1.0 - k).sum(0)

    def finish(self):
        c = self.count
        valid = c > 0
        safe = c.clamp_min(1)
        mean = torch.where(valid, self.sum / safe, torch.zeros_like(c))
        var = torch.where(valid, self.sqsum / safe - mean * mean,
                          torch.zeros_like(c)).clamp_min(0)
        total = c + self.censored
        return {
            "mean": mean.cpu().numpy(), "std": var.sqrt().cpu().numpy(),
            "count": c.cpu().numpy(), "censored": self.censored.cpu().numpy(),
            "total": total.cpu().numpy(),
            "censored_frac": (self.censored / total.clamp_min(1)).cpu().numpy(),
            "valid": valid.cpu().numpy(),
        }


def scalars_from_helper(parts, mixer):
    """The nine Stage 4B1 scalars, renamed from the shared helper's output.

    No second implementation: every value is whatever mamba3_core produced.
    """
    q = recurrence_quantities(
        parts["dd_dt"], parts["dd_A"], parts["trap"], mixer.dt_bias.float(),
        A_floor=float(getattr(mixer, "A_floor", 1e-4)))
    return {name: q[src] for name, src in SCALAR_MAP.items()}, q


def phase_objects(raw_angle, delta):
    """Source-derived cumulative phase for ONE independently forwarded block.

    Stage 4B2C: this is now a THIN ADAPTER over the shared
    reference_recurrence.phase_details. The temporary local transcription of the
    increment, the unwrapped cumulative phase and the wrapping is DELETED --
    there is exactly one implementation of the phase formula in the codebase,
    and it lives in the oracle.

    Phase STARTS AT ZERO for every block (carried=None). Blocks are forwarded
    independently, so carrying phase across them would be exactly the
    cross-block state leak the schema-v3 contract removed.

    The rotary-pair axis k is PRESERVED; only the configured partial-RoPE pairs
    exist in raw_angle, so no filtering is needed or performed here.
    """
    d = phase_details(raw_angle, delta, None)
    return {"increment": d["increment"], "unwrapped": d["unwrapped"],
            "wrapped": d["wrapped"], "carry": d["carry_out"],
            "winding": d["increment"].sum(0) / TWO_PI,
            "consistency_err": 0.0}   # single implementation: nothing to diverge


def role_slices(n):
    """Token index ranges per boundary role. A 1-token block is BOS only."""
    return {"bos": (0, 1),
            "interior": (1, max(n - 1, 1)) if n > 2 else (0, 0),
            "final": (n - 1, n) if n > 1 else (0, 0)}


def plan_forward_packs(block_indices, offsets, max_blocks, max_tokens=0):
    """Greedily pack complete independent sequences without padding.

    ``max_blocks`` bounds launch metadata and hook fan-out. ``max_tokens``
    bounds activation memory; zero disables the token cap. A single block is
    never split, even if it alone exceeds the requested token cap.
    """
    packs, current, current_tokens = [], [], 0
    for b in block_indices:
        n = int(offsets[int(b) + 1]) - int(offsets[int(b)])
        would_overflow = (current and max_tokens > 0
                          and current_tokens + n > max_tokens)
        if current and (len(current) >= max_blocks or would_overflow):
            packs.append(current)
            current, current_tokens = [], 0
        current.append(int(b))
        current_tokens += n
    if current:
        packs.append(current)
    return packs


class RoleCapture:
    """Online accumulators for ONE (semantic label, layer, boundary role)."""

    def __init__(self, nheads, n_pairs, device):
        self.nheads = nheads
        self.moments = {q: StreamingMoments((1, nheads), device=device)
                        for q in SCALAR_QUANTITIES if q != "eff_half_life"}
        self.hists = {q: Histogram((1, nheads), make_edges(SCALAR_HIST_EDGES[q]),
                                   device=device)
                      for q in SCALAR_HIST_EDGES if q != "eff_half_life"}
        for name, (_src, edge) in DERIVED_POSITIVE.items():
            self.hists[name] = Histogram((1, nheads), make_edges(edge), device=device)
        # One vectorized masked histogram over the (1,H) grid. Each head retains
        # independent bins while avoiding a Python branch/synchronization per
        # head, role, layer and block.
        self.hl = MaskedMoments(nheads, device)
        self.hl_hist = Histogram((1, nheads), make_edges("local_halflife"),
                                 device=device)
        # phase: rotary-pair axis PRESERVED, never collapsed before accumulation
        self.sin = StreamingMoments((nheads, n_pairs), device=device)
        self.cos = StreamingMoments((nheads, n_pairs), device=device)
        self.unwrapped = StreamingMoments((nheads, n_pairs), device=device)
        self.increment = StreamingMoments((nheads, n_pairs), device=device)
        # Stage 4B2A: B/C role-stratified moments (class-level pooling)
        self.bc_moments = {f: StreamingMoments((1, nheads), device=device)
                           for f in BLOCK_ROLE_BC if f != "bc_cos_shared"}
        if "bc_cos_shared" in BLOCK_ROLE_BC:
            self.bc_moments["bc_cos_shared"] = StreamingMoments((1, 1), device=device)
        self.n_obs = 0

    def update(self, scal, ph, lo, hi, extra=None):
        """Accumulate token rows [lo, hi) of one block. Empty range = no-op,
        so a block with no interior positions records ZERO observations
        rather than a fabricated value."""
        n = hi - lo
        if n <= 0:
            return
        for q, v in scal.items():
            if q == "eff_half_life":
                continue                                    # handled below
            x = v[lo:hi].T.unsqueeze(0)                     # (1, H, n)
            self.moments[q].update(x)
            if q in self.hists:
                self.hists[q].update(x)

        # positive transforms give signed quantities a distributional summary
        for name, (src, _edge) in DERIVED_POSITIVE.items():
            self.hists[name].update((-scal[src][lo:hi]).T.unsqueeze(0))

        # ---- half-life: censor where alpha -> 1, per head, no NaN anywhere ----
        hl = scal["eff_half_life"][lo:hi]
        keep = ((-scal["log_alpha"][lo:hi]) > HALFLIFE_CENSOR_EPS) & torch.isfinite(hl)
        self.hl.update(hl, keep)
        self.hl_hist.update(hl.T.unsqueeze(0), keep.T.unsqueeze(0))

        w = ph["wrapped"][lo:hi]                            # (n, H, K)
        self.sin.update(torch.sin(w).permute(1, 2, 0))
        self.cos.update(torch.cos(w).permute(1, 2, 0))
        self.unwrapped.update(ph["unwrapped"][lo:hi].permute(1, 2, 0))
        self.increment.update(ph["increment"][lo:hi].permute(1, 2, 0))

        # Stage 4B2A: B/C role-stratified accumulation
        if extra is not None:
            for f, acc in self.bc_moments.items():
                if f in extra:
                    v = extra[f][lo:hi]
                    if v.dim() == 1:
                        v = v.unsqueeze(-1)  # (n,) -> (n, 1) for bc_cos_shared
                    acc.update(v.T.unsqueeze(0))  # (1, H, n) or (1, 1, n)

        self.n_obs += n

    def finish(self):
        out = {"n_obs": np.array([self.n_obs])}
        for k, v in self.hl.finish().items():
            out[f"eff_half_life/{k}"] = v
        for qq in (0.25, 0.5, 0.75):
            out[f"eff_half_life/p{int(qq * 100)}"] = \
                self.hl_hist.quantile(qq)[0].cpu().numpy()
        for q, m in self.moments.items():
            f = m.finish()
            out[f"{q}/mean"] = f["mean"]
            out[f"{q}/std"] = f["std"]
            out[f"{q}/count"] = f["count"]
        for q, h in self.hists.items():
            for qq in (0.1, 0.25, 0.5, 0.75, 0.9):
                out[f"{q}/p{int(qq * 100)}"] = h.quantile(qq).cpu().numpy()
            out[f"{q}/censored"] = h.censored_fraction().cpu().numpy()
        for nm, acc in (("phase_sin", self.sin), ("phase_cos", self.cos),
                        ("phase_unwrapped", self.unwrapped),
                        ("phase_increment", self.increment)):
            f = acc.finish()
            out[f"{nm}/mean"] = f["mean"]
            out[f"{nm}/std"] = f["std"]
        # Stage 4B2A role-stratified B/C
        for nm, acc in self.bc_moments.items():
            f = acc.finish()
            out[f"bc_role|{nm}/mean"] = f["mean"]
            out[f"bc_role|{nm}/std"] = f["std"]
        return out


def near_zero_fraction(x, eps=NEAR_ZERO_EPS):
    """Thresholded near-zero fraction, relative to the tensor's own std.

    NOT sparsity in any principled sense (contract 3.5). Named honestly.
    """
    s = x.float().std()
    return (x.float().abs() < eps * s).float().mean()


# --------------------------------------------------------------------------
# token-contract loading, validation, and budget selection (schema v3)
# --------------------------------------------------------------------------


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def verify_artifact_digest(npz_path, man):
    """Hash the ACTUAL bytes and require agreement with the manifest.

    Copying the manifest's hash into provenance without recomputing it proves
    nothing: a mutated artifact would carry its original hash forward. Missing
    or mismatched digest is a STOP.
    """
    expected = man.get("artifact_sha256")
    computed = sha256_file(npz_path)
    if not expected:
        return None, computed, [f"manifest has no artifact_sha256 for {npz_path}"]
    if expected != computed:
        return expected, computed, [
            f"artifact digest mismatch for {npz_path}: manifest {expected[:16]}... "
            f"but file hashes to {computed[:16]}... (the artifact changed after "
            f"its manifest was written)"]
    return expected, computed, []


class ContractError(ValueError):
    """The token artifact does not satisfy the schema-v3 capture contract."""


def load_contract(npz_path):
    """Load the artifact and its manifest. Fails closed on the old schema."""
    tc = np.load(npz_path, allow_pickle=False)
    man_path = npz_path.replace(".npz", ".manifest.json")
    try:
        with open(man_path) as fh:
            man = json.load(fh)
    except FileNotFoundError as e:
        raise ContractError(
            f"no manifest at {man_path}; schema v3 requires one (it carries the "
            f"independence flag, budgets and artifact hash)") from e
    return tc, man


def validate_contract(tc, man, prefix):
    """Every structural precondition capture depends on. Returns failures."""
    f = []

    ver = man.get("schema_version")
    if ver not in SUPPORTED_SCHEMA_VERSIONS:
        f.append(f"schema_version {ver!r} not in {SUPPORTED_SCHEMA_VERSIONS}")
    if man.get("capture_must_forward_blocks_independently") is not True:
        f.append("manifest does not assert "
                 "capture_must_forward_blocks_independently == true")

    # fail closed on the OLD schema: 2-D stream A, or the retired key names
    if f"{prefix}_ids" not in tc.files:
        f.append(f"missing {prefix}_ids: not a schema-v3 artifact")
        return f
    if tc[f"{prefix}_ids"].ndim != 1:
        f.append(f"{prefix}_ids is {tc[f'{prefix}_ids'].ndim}-D; schema v3 is FLAT "
                 f"ids + offsets. A 2-D array is the old packed layout, which "
                 f"carried recurrent state across documents.")
        return f
    for retired, replacement in (("a_block", "a_doc_id/a_offsets"),
                                 ("b_class", "b_category"),
                                 ("b_n_tokens", "b_valid_len")):
        if retired in tc.files:
            f.append(f"retired key {retired!r} present (replaced by {replacement}); "
                     f"refusing a mixed-schema artifact")

    # NOTE offsets is deliberately absent here: it has n_blocks + 1 entries by
    # construction, and is checked as a partition above.
    required_block = ("valid_len", "doc_id", "block_id", "prompt_id",
                      "source_row", "source_file", "label", "category",
                      "chunk_index", "n_chunks", "n_content_tokens")
    required_token = ("token_pos", "token_doc_pos")
    missing = [k for k in ("offsets",) + required_block + required_token
               if f"{prefix}_{k}" not in tc.files]
    if missing:
        f.append(f"missing required arrays: {missing}")
        return f

    ids = tc[f"{prefix}_ids"]
    offs = tc[f"{prefix}_offsets"]
    vlen = tc[f"{prefix}_valid_len"]
    n_blocks = len(offs) - 1

    if offs[0] != 0 or offs[-1] != len(ids):
        f.append(f"offsets do not partition ids: [{offs[0]}, {offs[-1]}] vs "
                 f"{len(ids)} tokens")
    if np.any(np.diff(offs) <= 0):
        f.append("offsets are not strictly increasing (empty or overlapping block)")
    if not np.array_equal(np.diff(offs).astype(np.int64), vlen.astype(np.int64)):
        f.append("valid_len does not equal the offset differences")

    bos = (man.get("tokenizer") or {}).get("bos_id")
    if bos is None:
        f.append("manifest records no tokenizer.bos_id")
    elif n_blocks and not np.all(ids[offs[:-1]] == bos):
        f.append(f"{int(np.sum(ids[offs[:-1]] != bos))} blocks do not start with "
                 f"BOS id {bos}")

    for k in required_block:
        if len(tc[f"{prefix}_{k}"]) != n_blocks:
            f.append(f"{prefix}_{k} has {len(tc[f'{prefix}_{k}'])} entries, "
                     f"expected {n_blocks} (one per block)")
    for k in required_token:
        if len(tc[f"{prefix}_{k}"]) != len(ids):
            f.append(f"{prefix}_{k} has {len(tc[f'{prefix}_{k}'])} entries, "
                     f"expected {len(ids)} (one per token)")
    return f


def validate_block_positions(tc, prefix, b, s0, e0):
    """Token-role metadata for ONE block must agree with its offsets."""
    f = []
    tp = tc[f"{prefix}_token_pos"][s0:e0]
    td = tc[f"{prefix}_token_doc_pos"][s0:e0]
    n = e0 - s0
    if len(tp) != n or len(td) != n:
        f.append(f"block {b}: token metadata slice {len(tp)}/{len(td)} != {n}")
        return f
    if not np.array_equal(tp, np.arange(n)):
        f.append(f"block {b}: token_pos is not 0..{n - 1}")
    if n and td[0] != -1:
        f.append(f"block {b}: BOS token_doc_pos is {td[0]}, expected -1")
    vl = int(tc[f"{prefix}_valid_len"][b])
    if vl != n:
        f.append(f"block {b}: valid_len {vl} != offset span {n}")
    return f


def select_budget(tc, man, budget):
    """Select a Stream A prefix ONLY through a recorded manifest budget record.

    The cutoff is never recomputed from token counts: the artifact already fixed
    where each budget closed, and recounting could land somewhere else.
    """
    records = ((man.get("stream_a") or {}).get("budgets") or [])
    if not records:
        raise ContractError("manifest records no stream_a budget boundaries")
    match = [r for r in records if int(r["requested_content_tokens"]) == int(budget)]
    if not match:
        have = [r["requested_content_tokens"] for r in records]
        raise ContractError(f"no recorded budget {budget}; manifest has {have}")
    rec = match[0]

    n_blocks = int(rec["n_blocks"])
    offs = tc["a_offsets"]
    if n_blocks < 1 or n_blocks > len(offs) - 1:
        raise ContractError(f"budget n_blocks={n_blocks} outside artifact "
                            f"({len(offs) - 1} blocks)")

    # verify the recorded boundary against the artifact itself
    fails = []
    if int(tc["a_block_id"][n_blocks - 1]) != int(rec["last_block_id"]):
        fails.append(f"last_block_id {rec['last_block_id']} != artifact "
                     f"{int(tc['a_block_id'][n_blocks - 1])}")
    n_docs = int(len(np.unique(tc["a_doc_id"][:n_blocks])))
    if n_docs != int(rec["n_documents"]):
        fails.append(f"n_documents {rec['n_documents']} != artifact {n_docs}")
    content = int(np.sum(tc["a_n_content_tokens"][:n_blocks]))
    if content != int(rec["realized_content_tokens"]):
        fails.append(f"realized_content_tokens {rec['realized_content_tokens']} "
                     f"!= artifact {content}")
    valid = int(offs[n_blocks] - offs[0])
    if valid != int(rec["realized_valid_tokens"]):
        fails.append(f"realized_valid_tokens {rec['realized_valid_tokens']} "
                     f"!= artifact {valid}")
    if fails:
        raise ContractError("budget record disagrees with the artifact: "
                            + "; ".join(fails))
    return rec, n_blocks


# --------------------------------------------------------------------------
# offline self-check: synthetic flat-offset fixture. No CUDA, model or network.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# preflight: EVERYTHING that can fail structurally, before any CUDA/model load
# --------------------------------------------------------------------------

# instrumented loader: the only place a model is constructed. The counter lets
# an offline test PROVE that a preflight failure performs zero model loads.
MODEL_LOAD_COUNT = [0]


def load_model(repo_or_path, device="cuda", dtype=None):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    MODEL_LOAD_COUNT[0] += 1
    return MambaLMHeadModel.from_pretrained(
        repo_or_path, device=device,
        dtype=dtype if dtype is not None else torch.bfloat16).eval()


def preflight(token_contract, stream, budget=None, max_blocks=None):
    """Resolve and validate the entire selected prefix. Returns (plan, failures).

    Nothing here touches CUDA, imports mamba_ssm, or allocates device memory.
    Every structural fault -- schema, digest, offsets, BOS, token positions,
    content lengths, budget disagreement -- is discovered HERE, so that once
    capture starts a selected block either processes or the run stops.
    """
    f = []
    try:
        tc, man = load_contract(token_contract)
    except (ContractError, FileNotFoundError) as e:
        return None, [f"{type(e).__name__}: {e}"]

    f += validate_contract(tc, man, stream)
    exp_dig, got_dig, dfails = verify_artifact_digest(token_contract, man)
    f += dfails
    if max_blocks is not None and max_blocks <= 0:
        f.append(f"--max-blocks must be positive, got {max_blocks}")
    if f:
        return None, f

    offs = tc[f"{stream}_offsets"]
    n_total = len(offs) - 1

    budget_record = None
    if stream == "a":
        if budget is None:
            return None, ["stream A requires --budget naming a recorded boundary"]
        try:
            budget_record, n_avail = select_budget(tc, man, budget)
        except ContractError as e:
            return None, [str(e)]
    else:
        n_avail = n_total

    n_use = min(n_avail, max_blocks) if max_blocks is not None else n_avail
    if n_use < 1:
        return None, [f"selected prefix is empty (n_avail={n_avail})"]
    block_indices = list(range(n_use))

    # ---- per-block structural validation over the ENTIRE selected prefix ----
    ncont = tc[f"{stream}_n_content_tokens"]
    for b in block_indices:
        s0, e0 = int(offs[b]), int(offs[b + 1])
        f += validate_block_positions(tc, stream, b, s0, e0)
        # content length derived from the OFFSETS, independent of metadata
        derived = (e0 - s0) - 1
        if int(ncont[b]) != derived:
            f.append(f"block {b}: n_content_tokens {int(ncont[b])} != "
                     f"offset-derived {derived} (valid_len-1)")
        if len(f) > 20:
            f.append("... further block failures suppressed")
            break
    if f:
        return None, f

    # ---- expectations derived from OFFSETS, not from metadata ----
    valid_expected = int(offs[n_use] - offs[0])
    content_expected = valid_expected - n_use          # one BOS per block
    # separate consistency check of the artifact's own bookkeeping
    metadata_content = int(np.sum(ncont[:n_use]))
    metadata_consistent = metadata_content == content_expected
    if not metadata_consistent:
        f.append(f"artifact metadata inconsistent: sum(n_content_tokens)="
                 f"{metadata_content} != offset-derived {content_expected}")
        return None, f

    plan = {
        "tc": tc, "manifest": man, "stream": stream,
        "block_indices": block_indices, "offsets": offs,
        "n_total": int(n_total), "n_avail": int(n_avail),
        "blocks_expected": int(n_use),
        "valid_tokens_expected": valid_expected,
        "content_tokens_expected": content_expected,
        "metadata_content_tokens": metadata_content,
        "metadata_consistent": bool(metadata_consistent),
        "budget_record": budget_record,
        "declared_limit": (int(max_blocks) if max_blocks is not None else None),
        "digest_expected": exp_dig, "digest_computed": got_dig,
    }
    return plan, []


def print_preflight(plan):
    print("preflight OK (no model loaded yet):")
    print(f"  blocks_expected        : {plan['blocks_expected']} "
          f"(of {plan['n_total']} in artifact)")
    print(f"  valid_tokens_expected  : {plan['valid_tokens_expected']}")
    print(f"  content_tokens_expected: {plan['content_tokens_expected']} "
          f"(offset-derived; metadata agrees: {plan['metadata_consistent']})")
    print(f"  artifact digest        : {plan['digest_computed'][:16]}...")
    if plan["budget_record"]:
        r = plan["budget_record"]
        print(f"  selected budget        : {r['requested_content_tokens']} -> "
              f"{r['n_blocks']} blocks, {r['n_documents']} documents")
    else:
        print("  selected budget        : n/a (stream B)")


def atomic_savez(path, **payload):
    """Write via a temporary file then os.replace, so an interruption cannot
    leave a partial artifact at the final path.

    np.savez_compressed appends a .npz suffix whenever the given name does not
    already end in one, so the temp name MUST carry its own .npz ending or the
    rename below targets a file that was never created."""
    tmp = path + ".partial.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


class PhaseTimer:
    """CUDA-event phase timing, drained ONLY at heartbeats.

    Events are recorded per block in pairs and read in batches. One explicit
    synchronize occurs at each reporting heartbeat (normally every 200 blocks),
    never once per phase or block. Recreating the original synchronization storm
    with timing instrumentation would defeat the exercise.
    """

    def __init__(self):
        self.pending = defaultdict(list)
        self.totals = defaultdict(float)
        self.use_events = torch.cuda.is_available()

    def mark(self, name):
        if self.use_events:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
        else:
            e = time.perf_counter()
        self.pending[name].append(e)

    def phase(self, name):
        timer = self

        class _Ctx:
            def __enter__(self):
                timer.mark(name)

            def __exit__(self, *exc):
                timer.mark(name)

        return _Ctx()

    def drain(self):
        out = {}
        # One synchronization at a reporting heartbeat, never one per phase or
        # per block. The two staged D2H copies already impose the required
        # per-block data dependency; this only makes event timing readable.
        if self.use_events and any(self.pending.values()):
            torch.cuda.synchronize()
        for ph, evs in self.pending.items():
            if self.use_events:
                add = sum(s.elapsed_time(e) for s, e in
                          zip(evs[::2], evs[1::2])) / 1000.0
            else:
                add = sum(b - a for a, b in zip(evs[::2], evs[1::2]))
            self.totals[ph] += add
            out[ph] = self.totals[ph]
            evs.clear()
        return dict(self.totals) if not out else out


def _synthetic_contract(bos=1, n_blocks=6, with_budget=True, prefix="a"):
    """A schema-v3 artifact in memory, plus its manifest."""
    rng = np.random.default_rng(0)
    ids, offs, tpos, dpos = [], [0], [], []
    lens = [3, 5, 4, 6, 3, 7][:n_blocks]
    for i, L in enumerate(lens):
        seq = [bos] + list(rng.integers(10, 99, L - 1))
        ids += seq
        offs.append(len(ids))
        tpos += list(range(L))
        dpos += [-1] + list(range(L - 1))
    n = len(lens)
    tc_a = {
        "a_ids": np.array(ids, np.int32), "a_offsets": np.array(offs, np.int64),
        "a_token_pos": np.array(tpos, np.int32),
        "a_token_doc_pos": np.array(dpos, np.int32),
        "a_valid_len": np.array(lens, np.int32),
        "a_n_content_tokens": np.array([L - 1 for L in lens], np.int32),
        "a_doc_id": np.arange(n, dtype=np.int64),
        "a_block_id": np.arange(n, dtype=np.int64),
        "a_prompt_id": np.arange(n, dtype=np.int64),
        "a_source_row": np.arange(n, dtype=np.int64),
        "a_source_file": np.array(["fineweb-edu"] * n, dtype="<U128"),
        "a_label": np.array(["fineweb"] * n, dtype="<U128"),
        "a_category": np.array(["stream_a"] * n, dtype="<U128"),
        "a_chunk_index": np.zeros(n, np.int32),
        "a_n_chunks": np.ones(n, np.int32),
    }
    tc = ({k.replace("a_", prefix + "_", 1): v for k, v in tc_a.items()}
          if prefix != "a" else tc_a)
    budgets = []
    if with_budget and prefix == "a":
        for req, k in [(r, kk) for r, kk in ((5, 3), (12, 5)) if kk <= n]:
            budgets.append({
                "requested_content_tokens": req,
                "realized_content_tokens": int(sum(L - 1 for L in lens[:k])),
                "realized_valid_tokens": int(offs[k]),
                "n_documents": k, "n_blocks": k,
                "last_doc_id": k - 1, "last_block_id": k - 1,
                "overshoot_content_tokens": 0, "last_source_row": k - 1,
            })
    man = {"schema_version": 3,
           "capture_must_forward_blocks_independently": True,
           "tokenizer": {"bos_id": bos}, "artifact_sha256": "deadbeef",
           "stream_a": {"budgets": budgets}}
    return tc, man


class _NpzLike(dict):
    @property
    def files(self):
        return list(self.keys())


def _fixture_recurrence_scalars():
    """Exact manual fixture. Expected values written INDEPENDENTLY of the helper."""
    print("\nrecurrence scalars (manual, helper-independent expectations):")
    T, H = 4, 1
    Delta = torch.tensor([[1.0], [2.0], [4.0], [8.0]])
    lam = torch.tensor([[0.25], [0.50], [0.75], [1.00]])

    # hand-written, not produced by any helper
    gamma_exp = [0.25 * 1.0, 0.50 * 2.0, 0.75 * 4.0, 1.00 * 8.0]
    shifted_exp = [2.0 * (1 - 0.50), 4.0 * (1 - 0.75), 8.0 * (1 - 1.00), 0.0]
    trap_exp = [g + sft for g, sft in zip(gamma_exp, shifted_exp)]

    gamma = Delta * lam
    shifted = torch.zeros_like(gamma)
    shifted[:-1] = Delta[1:] * (1.0 - lam[1:])
    trap = gamma + shifted

    ok = True
    for nm, got, exp in (("gamma", gamma, gamma_exp),
                         ("shifted_gamma", shifted, shifted_exp),
                         ("trap_scale", trap, trap_exp)):
        g = [round(float(x), 6) for x in got.reshape(-1)]
        hit = g == [round(x, 6) for x in exp]
        print(f"  [{'ok ' if hit else 'BAD'}] {nm:14s} {g} == {exp}")
        ok = ok and hit
    fin = float(shifted[-1]) == 0.0
    print(f"  [{'ok ' if fin else 'BAD'}] final shifted_gamma == 0")
    ok = ok and fin

    # changing token t+1 must move trap_scale[t] but NOT gamma[t]
    lam2 = lam.clone(); lam2[2] = 0.10
    Delta2 = Delta.clone(); Delta2[2] = 9.0
    g2 = Delta2 * lam2
    s2 = torch.zeros_like(g2); s2[:-1] = Delta2[1:] * (1.0 - lam2[1:])
    t2 = g2 + s2
    moved = float(t2[1]) != float(trap[1])
    same = float(g2[1]) == float(gamma[1])
    print(f"  [{'ok ' if moved and same else 'BAD'}] token t+1 changes "
          f"trap_scale[t] ({float(trap[1]):.2f}->{float(t2[1]):.2f}) but not "
          f"gamma[t] ({float(gamma[1]):.2f})")
    return ok and moved and same


def _fixture_phase():
    print("\nphase (source-derived cumulative):")
    ok = True
    T, H, K = 6, 2, 3

    z = phase_objects(torch.zeros(T, K), torch.ones(T, H))
    hit = (float(z["increment"].abs().max()) == 0.0
           and float(z["unwrapped"].abs().max()) == 0.0
           and float(z["wrapped"].abs().max()) == 0.0)
    print(f"  [{'ok ' if hit else 'BAD'}] zero raw angle -> zero increment/phase")
    ok = ok and hit

    c = phase_objects(torch.full((T, K), 0.4), torch.ones(T, H))
    mono = bool((c["unwrapped"].diff(dim=0) > 0).all())
    inrange = bool((c["wrapped"] >= 0).all() and (c["wrapped"] < TWO_PI + 1e-6).all())
    print(f"  [{'ok ' if mono else 'BAD'}] constant positive angle -> monotone unwrapped")
    print(f"  [{'ok ' if inrange else 'BAD'}] wrapped phase within [0, 2pi)")
    ok = ok and mono and inrange

    # split vs full, compared on the circle, using the helper's carry
    raw = torch.randn(T, K); dl = torch.rand(T, H) + 0.5
    full = phase_objects(raw, dl)
    w1, c1 = phase_from_raw(raw[:3], dl[:3], None)
    w2, _ = phase_from_raw(raw[3:], dl[3:], c1)
    joined = torch.cat([w1, w2], 0)
    d = torch.remainder(joined - full["wrapped"] + math.pi, TWO_PI) - math.pi
    split_ok = float(d.abs().max()) < 1e-5
    print(f"  [{'ok ' if split_ok else 'BAD'}] split-sequence with carry == full "
          f"(max {float(d.abs().max()):.2e})")
    ok = ok and split_ok

    # resetting per block must DIFFER from wrongly carrying across blocks
    w_reset, _ = phase_from_raw(raw[3:], dl[3:], None)
    differs = not torch.allclose(w_reset, w2, atol=1e-6)
    print(f"  [{'ok ' if differs else 'BAD'}] per-block reset differs from "
          f"carrying phase across blocks")
    ok = ok and differs

    # shared raw angles + head-dependent Delta -> head-dependent phase
    dh = torch.tensor([[1.0, 3.0]]).repeat(T, 1)
    hp = phase_objects(torch.full((T, K), 0.3), dh)
    head_dep = not torch.allclose(hp["unwrapped"][:, 0], hp["unwrapped"][:, 1])
    print(f"  [{'ok ' if head_dep else 'BAD'}] head-dependent Delta -> "
          f"head-dependent phase from shared raw angles")
    ok = ok and head_dep

    shapes = (c["increment"].shape == (T, H, K)
              and c["wrapped"].shape == (T, H, K)
              and c["winding"].shape == (H, K))
    print(f"  [{'ok ' if shapes else 'BAD'}] only the configured {K} rotary pairs "
          f"emitted; shapes increment{tuple(c['increment'].shape)} "
          f"winding{tuple(c['winding'].shape)}")
    return ok and shapes


def _fixture_roles():
    print("\nboundary-role stratification:")
    ok = True
    for n, want in ((6, (1, 4, 1)), (2, (1, 0, 1)), (1, (1, 0, 0))):
        rs = role_slices(n)
        got = tuple(hi - lo for lo, hi in
                    (rs["bos"], rs["interior"], rs["final"]))
        hit = got == want and sum(got) == n
        print(f"  [{'ok ' if hit else 'BAD'}] valid_len={n}: bos/interior/final "
              f"= {got}, sums to {sum(got)}")
        ok = ok and hit

    rs = role_slices(6)
    disjoint = (rs["bos"][1] <= rs["interior"][0]
                and rs["interior"][1] <= rs["final"][0])
    print(f"  [{'ok ' if disjoint else 'BAD'}] roles are disjoint and ordered: "
          f"{rs}")
    one = role_slices(1)
    no_final = (one["final"][1] - one["final"][0]) == 0
    print(f"  [{'ok ' if no_final else 'BAD'}] 1-token block is BOS only, "
          f"contributes no final observation")
    ok = ok and disjoint and no_final

    # a block with no interior records ZERO observations, not a fabricated value
    rc = RoleCapture(nheads=2, n_pairs=3, device="cpu")
    scal = {q: torch.zeros(2, 2) for q in SCALAR_QUANTITIES}
    ph = {"wrapped": torch.zeros(2, 2, 3), "unwrapped": torch.zeros(2, 2, 3),
          "increment": torch.zeros(2, 2, 3)}
    lo, hi = role_slices(2)["interior"]
    rc.update(scal, ph, lo, hi)
    print(f"  [{'ok ' if rc.n_obs == 0 else 'BAD'}] empty interior -> n_obs="
          f"{rc.n_obs} (no fabricated observation)")
    ok = ok and rc.n_obs == 0

    # semantic labels stay separate through the role accumulator keying
    store = defaultdict(lambda: {0: {r: RoleCapture(2, 3, "cpu")
                                     for r in POSITION_ROLES}})
    for lab in ("Humor", "Data"):
        store[lab][0]["bos"].update(scal, ph, 0, 1)
    sep = (store["Humor"][0]["bos"].n_obs == 1
           and store["Data"][0]["bos"].n_obs == 1 and len(store) == 2)
    print(f"  [{'ok ' if sep else 'BAD'}] semantic labels remain separate "
          f"({len(store)} groups)")
    return ok and sep


def _fixture_halflife_censoring():
    print("\nhalf-life censoring (alpha -> 1):")
    rc = RoleCapture(nheads=1, n_pairs=1, device="cpu")
    scal = {q: torch.zeros(3, 1) for q in SCALAR_QUANTITIES}
    scal["log_alpha"] = torch.tensor([[-1.0], [-1e-9], [-0.5]])
    scal["eff_half_life"] = math.log(2.0) / (-scal["log_alpha"]).clamp_min(1e-12)
    ph = {k: torch.zeros(3, 1, 1) for k in ("wrapped", "unwrapped", "increment")}
    rc.update(scal, ph, 0, 3)
    out = rc.finish()
    cens = int(out["eff_half_life/censored"][0])
    tot = int(out["eff_half_life/total"][0])
    ok = cens == 1 and tot == 3 and bool(out["eff_half_life/valid"][0])
    print(f"  [{'ok ' if ok else 'BAD'}] 1 of 3 observations censored "
          f"(censored={cens}, total={tot}, "
          f"frac={float(out['eff_half_life/censored_frac'][0]):.3f}, "
          f"valid={bool(out['eff_half_life/valid'][0])})")
    return ok


def _fixture_masked_halflife():
    """Different censor masks per head must not contaminate each other."""
    print("\nmasked half-life statistics (per head):")
    H = 3
    la = torch.zeros(4, H)
    la[:, 0] = torch.tensor([-1.0, -0.5, -0.25, -2.0])      # 4 finite
    la[:, 1] = torch.tensor([-1.0, -1e-12, -1e-12, -0.5])   # 2 finite
    la[:, 2] = torch.full((4,), -1e-12)                     # fully censored
    hl = math.log(2.0) / (-la).clamp_min(1e-12)
    keep = (-la) > HALFLIFE_CENSOR_EPS

    m = MaskedMoments(H)
    m.update(hl, keep)
    f = m.finish()
    counts = [int(c) for c in f["count"]]
    ok = counts == [4, 2, 0]
    print(f"  [{'ok ' if ok else 'BAD'}] per-head finite counts {counts} == [4, 2, 0]")
    fin = bool(np.isfinite(f["mean"][:2]).all() and np.isfinite(f["std"][:2]).all())
    print(f"  [{'ok ' if fin else 'BAD'}] finite heads report finite mean/std "
          f"{[round(float(x), 3) for x in f['mean'][:2]]}")
    dead = (not bool(f["valid"][2])) and float(f["mean"][2]) == 0.0
    print(f"  [{'ok ' if dead else 'BAD'}] fully censored head: count=0, "
          f"valid={bool(f['valid'][2])}, no fabricated value")
    clean = bool(np.isfinite(f["mean"]).all())
    print(f"  [{'ok ' if clean else 'BAD'}] no NaN anywhere in the output")
    fr = [round(float(x), 2) for x in f["censored_frac"]]
    print(f"  [{'ok ' if fr == [0.0, 0.5, 1.0] else 'BAD'}] censored fractions {fr}")
    return ok and fin and dead and clean and fr == [0.0, 0.5, 1.0]


def _fixture_winding_resolution():
    print("\nwinding resolution:")
    H, K = 4, 3
    ph = phase_objects(torch.randn(5, K), torch.rand(5, H) + 0.5)
    online_pair_resolved = ph["winding"].shape == (H, K)
    print(f"  [{'ok ' if online_pair_resolved else 'BAD'}] online winding keeps "
          f"(head, pair) resolution: {tuple(ph['winding'].shape)}")

    br = BlockRoleRecorder(2, 1, H)
    scal = {q: torch.zeros(1, H) for q in SCALAR_QUANTITIES}
    scal["log_alpha"] = -torch.ones(1, H)
    scal["eff_half_life"] = torch.full((1, H), math.log(2.0))
    ph_one = {k: ph[k][:1] for k in ("wrapped", "unwrapped", "increment")}
    br.put_role(0, 0, 0, scal, ph_one, 0, 1)
    br.put_winding(0, 0, ph["winding"])
    br.flush(0)                # staging -> numpy row; one batched D2H per block
    per_block = br.w["winding_mean_over_pairs"][0, 0]
    heads_kept = per_block.shape == (H,) and len(set(per_block.tolist())) > 1
    print(f"  [{'ok ' if heads_kept else 'BAD'}] per-block winding keeps {H} heads "
          f"(not collapsed to a scalar)")
    fields = sorted(br.w)
    print(f"  [{'ok ' if len(fields) == 4 else 'BAD'}] pair summaries: {fields}")
    return online_pair_resolved and heads_kept and len(fields) == 4


def _fixture_block_role_summaries():
    print("\nper-block role summaries:")
    H, K, L = 2, 2, 1
    n = 6
    scal = {q: torch.rand(n, H) + 0.5 for q in SCALAR_QUANTITIES}
    scal["log_alpha"] = -torch.rand(n, H) - 0.1
    scal["eff_half_life"] = math.log(2.0) / (-scal["log_alpha"])
    ph = {"wrapped": torch.rand(n, H, K), "unwrapped": torch.cumsum(
        torch.rand(n, H, K), 0), "increment": torch.rand(n, H, K)}

    br = BlockRoleRecorder(2, L, H)
    rs = role_slices(n)
    for ri, r in enumerate(POSITION_ROLES):
        lo, hi = rs[r]
        br.put_role(0, 0, ri, scal, ph, lo, hi)
    br.flush(0)
    counts = br.count[0, 0].tolist()
    ok = counts == [1, 4, 1] and sum(counts) == n
    print(f"  [{'ok ' if ok else 'BAD'}] role counts {counts} sum to valid_len {n}")

    # count-weighted role means reconstruct the pooled block mean
    pooled = scal["lambda"].mean(0).numpy()
    recon = sum(br.a["lambda"][0, 0, i].astype(np.float64) * counts[i]
                for i in range(3)) / n
    err = float(np.abs(recon - pooled).max())
    rec_ok = err < 2e-3          # float16 summary storage
    print(f"  [{'ok ' if rec_ok else 'BAD'}] count-weighted roles reconstruct the "
          f"pooled block mean (max err {err:.2e}, float16 storage)")

    # absent role
    br2 = BlockRoleRecorder(1, L, H)
    rs2 = role_slices(2)
    for ri, r in enumerate(POSITION_ROLES):
        lo, hi = rs2[r]
        br2.put_role(0, 0, ri, scal, ph, lo, hi)
    br2.flush(0)
    absent = br2.count[0, 0, 1] == 0 and not bool(br2.valid[0, 0, 1])
    print(f"  [{'ok ' if absent else 'BAD'}] absent interior: count=0, valid=False")

    # two blocks sharing a label stay separate bootstrap units
    br3 = BlockRoleRecorder(2, L, H)
    for bi in (0, 1):
        sc = {k: v * (1.0 + bi) for k, v in scal.items()}
        br3.put_role(bi, 0, 0, sc, ph, 0, 1)
        br3.flush(bi)
    sep = not np.allclose(br3.a["lambda"][0, 0, 0], br3.a["lambda"][1, 0, 0])
    print(f"  [{'ok ' if sep else 'BAD'}] two blocks with the same label remain "
          f"separate rows")
    return ok and rec_ok and absent and sep


def _fixture_rank_collapsed_summaries():
    """Declared 4B2C arrays must contain real role-reduced observations."""
    print("\nper-block rank and collapsed summaries:")
    T, H, R, L = 6, 2, 4, 1
    br = BlockRoleRecorder(1, L, H, rank=R)
    rankvals = {f: torch.rand(T, H, R) + 0.25 for f in BLOCK_RANK_FIELDS}
    colvals = {f: torch.rand(T, H) + 0.25 for f in BLOCK_COLLAPSED_FIELDS}
    scal = {q: torch.rand(T, H) + 0.25 for q in SCALAR_QUANTITIES}
    scal["log_alpha"] = -torch.rand(T, H) - 0.1
    scal["eff_half_life"] = math.log(2.0) / (-scal["log_alpha"])
    ph = {"wrapped": torch.rand(T, H, 2),
          "unwrapped": torch.rand(T, H, 2).cumsum(0),
          "increment": torch.rand(T, H, 2)}
    for _cf, vf in br._ALIGN_PAIRS:
        colvals[vf] = torch.ones(T, H)
    # One head/path is wholly invalid: it must retain validity fraction zero.
    colvals["align_D_residual_valid"][:, 1] = 0.0
    colvals["align_D_residual"][:, 1] = 0.0

    rs = role_slices(T)
    for ri, role in enumerate(POSITION_ROLES):
        lo, hi = rs[role]
        br.put_role(0, 0, ri, scal, ph, lo, hi)
        br.put_rank(0, ri, rankvals, lo, hi)
        br.put_collapsed(0, ri, colvals, lo, hi)
    br.flush(0)

    shape_ok = br.r["V_rank_norm"].shape == (1, L, 3, H, R)
    expected = rankvals["V_rank_norm"][1:5].mean(0).numpy()
    observed = br.r["V_rank_norm"][0, 0, 1].astype(np.float32)
    mean_ok = bool(np.allclose(observed, expected, atol=2e-3))
    invalid_ok = (float(br.c["align_D_residual"][0, 0, 1, 1]) == 0.0
                  and float(br.c["align_D_residual_valid"][0, 0, 1, 1]) == 0.0)
    valid_ok = abs(float(br.c["align_D_residual_valid"][0, 0, 1, 0]) - 1.0) < 1e-3
    payload_ok = (all(f"blockrank|{f}" in br.payload() for f in BLOCK_RANK_FIELDS)
                  and all(f"blockcollapsed|{f}" in br.payload()
                          for f in BLOCK_COLLAPSED_FIELDS))
    print(f"  [{'ok ' if shape_ok else 'BAD'}] rank payload shape "
          f"{br.r['V_rank_norm'].shape}")
    print(f"  [{'ok ' if mean_ok else 'BAD'}] role mean preserves head/rank")
    print(f"  [{'ok ' if invalid_ok and valid_ok else 'BAD'}] alignment validity "
          "distinguishes valid zero from invalid zero")
    print(f"  [{'ok ' if payload_ok else 'BAD'}] every declared family is published")
    return shape_ok and mean_ok and invalid_ok and valid_ok and payload_ok


def _fixture_block_recorder_layer_counts():
    """Layer-at-a-time writes must retain exact, independent denominators."""
    print("\nblock recorder layer denominators:")
    L, H, T = 3, 2, 5
    rec = BlockRecorder(("a", "b"), L, H)
    rec.begin({"block": 0})
    truth = {}
    for qi, q in enumerate(("a", "b")):
        truth[q] = []
        for lp in range(L):
            x = torch.arange(H * T, dtype=torch.float32).reshape(H, T) + 10 * lp + qi
            rec.add(q, x, lp=lp)
            truth[q].append(x.mean(-1).numpy())
    rec.end()
    out = rec.finish()
    exact = all(np.allclose(out[q][0], np.stack(truth[q])) for q in truth)
    one_copy = rec.d2h == 1 and rec.denominator_failures == 0

    bad = BlockRecorder(("a",), L, H)
    bad.begin({"block": 1})
    bad.add("a", torch.ones(H, T), lp=0)
    caught = False
    try:
        bad.end()
    except RuntimeError:
        caught = True
    print(f"  [{'ok ' if exact else 'BAD'}] per-layer means are exact")
    print(f"  [{'ok ' if one_copy else 'BAD'}] one batched D2H copy")
    print(f"  [{'ok ' if caught else 'BAD'}] missing layer fails closed")
    return exact and one_copy and caught


def _fixture_distribution_coverage():
    print("\ndistributional coverage:")
    rc = RoleCapture(nheads=2, n_pairs=2, device="cpu")
    scal = {q: torch.rand(4, 2) + 0.5 for q in SCALAR_QUANTITIES}
    scal["log_alpha"] = -torch.rand(4, 2) - 0.1
    scal["A"] = -torch.rand(4, 2) - 0.1
    scal["eff_half_life"] = math.log(2.0) / (-scal["log_alpha"])
    ph = {k: torch.rand(4, 2, 2) for k in ("wrapped", "unwrapped", "increment")}
    rc.update(scal, ph, 0, 4)
    out = rc.finish()

    need = set(SCALAR_QUANTITIES)
    covered = set()
    for q in need:
        if any(k.startswith(f"{q}/p") for k in out):
            covered.add(q)
    # signed quantities are covered by their positive transforms
    for name, (src, _e) in DERIVED_POSITIVE.items():
        if any(k.startswith(f"{name}/p") for k in out):
            covered.add(src)
    missing = sorted(need - covered)
    print(f"  [{'ok ' if not missing else 'BAD'}] every recurrence quantity has a "
          f"distributional summary; missing={missing}")
    cen = any("censored" in k for k in out)
    print(f"  [{'ok ' if cen else 'BAD'}] censored/overflow fractions recorded")
    return not missing and cen


def _fixture_bc_reconstruction():
    print("\nB/C reconstruction shapes:")
    ok = True
    # Simulate B and C with ngroups=1, rank=4, d_state=16, nheads=8
    nheads, rank, d_state = 8, 4, 16
    B_raw = torch.randn(1, rank, 1, d_state)
    C_raw = torch.randn(1, rank, 1, d_state)
    B_norm_w = torch.ones(d_state)
    C_norm_w = torch.ones(d_state)
    B_bias = torch.randn(nheads, rank, d_state)
    C_bias = torch.randn(nheads, rank, d_state)

    B_shared = reconstruct_bc(B_raw, B_norm_w, B_bias, expand_heads=False)
    B_pos = reconstruct_bc(B_raw, B_norm_w, B_bias)
    C_shared = reconstruct_bc(C_raw, C_norm_w, C_bias, expand_heads=False)
    C_pos = reconstruct_bc(C_raw, C_norm_w, C_bias)

    sh_ok = (B_shared.shape == (1, rank, 1, d_state) and
             B_pos.shape == (1, nheads, rank, d_state) and
             C_shared.shape == (1, rank, 1, d_state) and
             C_pos.shape == (1, nheads, rank, d_state))
    print(f"  [{'ok ' if sh_ok else 'BAD'}] shared shapes (B={tuple(B_shared.shape)}, "
          f"C={tuple(C_shared.shape)}) and posbias shapes (B={tuple(B_pos.shape)}, "
          f"C={tuple(C_pos.shape)})")
    ok = ok and sh_ok

    # With ngroups=1, B_shared/C_shared are identical across heads before bias
    heads_same = torch.allclose(B_shared.movedim(-2, -3).expand(1, nheads, rank, d_state),
                                B_shared.movedim(-2, -3))
    print(f"  [{'ok ' if heads_same else 'BAD'}] shared pre-bias identical across heads")
    ok = ok and heads_same

    # Bias creates per-head structure
    per_head = not torch.allclose(B_pos[0, 0], B_pos[0, 1])
    print(f"  [{'ok ' if per_head else 'BAD'}] per-head bias creates head-level structure")
    ok = ok and per_head
    return ok


def _fixture_rank_geometry():
    print("\nrank geometry:")
    ok = True
    nheads, rank, d_state = 2, 4, 8
    X = torch.randn(3, nheads, rank, d_state)
    common, diff = rank_split(X, rank_dim=-2)
    sum_zero = diff.sum(dim=-2).abs().max() < 1e-6
    print(f"  [{'ok ' if sum_zero else 'BAD'}] differential part sums to zero over rank")
    ok = ok and sum_zero

    dnr = differential_norm_ratio(X, rank_dim=-2)
    in_range = (dnr >= 0).all() and (dnr <= 1).all()
    print(f"  [{'ok ' if in_range else 'BAD'}] differential_norm_ratio in [0,1]")
    ok = ok and in_range

    # Rank-1 tensor -> diff ratio = 0
    X1 = torch.randn(3, nheads, 1, d_state)
    dnr1 = differential_norm_ratio(X1, rank_dim=-2)
    rank1_zero = float(dnr1.max()) < 1e-6
    print(f"  [{'ok ' if rank1_zero else 'BAD'}] rank-1 tensor has diff ratio ~0")
    ok = ok and rank1_zero

    # Higher rank -> potentially higher diff ratio
    X2 = torch.randn(3, nheads, 8, d_state)
    dnr2 = differential_norm_ratio(X2, rank_dim=-2)
    rank8_higher = float(dnr2.mean()) > float(dnr1.mean())
    print(f"  [{'ok ' if rank8_higher else 'BAD'}] rank-8 mean diff ratio > rank-1")
    ok = ok and rank8_higher
    return ok


def _fixture_bc_alignment():
    print("\nB/C alignment (cosine, not synergy):")
    ok = True
    nheads, rank, d_state = 2, 4, 8
    # Identical B and C -> cosine = 1
    X = torch.randn(3, nheads, rank, d_state)
    cos_identical = F.cosine_similarity(
        X.reshape(3, nheads, -1), X.reshape(3, nheads, -1), dim=-1)
    ident_ok = torch.allclose(cos_identical, torch.ones_like(cos_identical))
    print(f"  [{'ok ' if ident_ok else 'BAD'}] identical tensors -> cosine=1")
    ok = ok and ident_ok

    # Orthogonal B and C -> cosine = 0
    Y = torch.randn(3, nheads, rank, d_state)
    flat_X = X.reshape(3, nheads, -1)
    flat_Y = Y.reshape(3, nheads, -1)
    proj = (flat_X * flat_Y).sum(dim=-1, keepdim=True) / (flat_X**2).sum(dim=-1, keepdim=True)
    flat_Y_ortho = flat_Y - proj * flat_X
    cos_ortho = F.cosine_similarity(flat_X, flat_Y_ortho, dim=-1)
    ortho_ok = float(cos_ortho.abs().max()) < 1e-3
    print(f"  [{'ok ' if ortho_ok else 'BAD'}] orthogonalized tensors -> cosine~0 "
          f"(max {float(cos_ortho.abs().max()):.2e})")
    ok = ok and ortho_ok

    # Shape check
    cos_shared_scalar = F.cosine_similarity(
        X[0].reshape(-1), Y[0].reshape(-1), dim=-1)
    shape_ok = cos_shared_scalar.shape == ()
    print(f"  [{'ok ' if shape_ok else 'BAD'}] per-token-head cosine is scalar")
    ok = ok and shape_ok
    return ok


def _fixture_actual_partial_rope():
    """The model's RoPE acts on STATE COORDINATES, not on rank."""
    print("\nactual partial RoPE (state coordinates, both B and C):")
    ok = True
    T, H, R, N, K = 4, 2, 3, 8, 2          # K < N//2 -> genuinely partial
    B = torch.randn(T, H, R, N)
    C = torch.randn(T, H, R, N)
    phase = torch.rand(T, H, K) * 2 * math.pi

    B_rot = partial_rope_bc(B, phase)
    C_rot = partial_rope_bc(C, phase)
    shapes = B_rot.shape == B.shape and C_rot.shape == C.shape
    print(f"  [{'ok ' if shapes else 'BAD'}] BOTH B and C rotated; shapes "
          f"{tuple(B.shape)} -> {tuple(B_rot.shape)}")
    ok = ok and shapes

    untouched = list(range(K, N // 2)) + list(range(N // 2 + K, N))
    same = (torch.allclose(B_rot[..., untouched], B[..., untouched])
            and torch.allclose(C_rot[..., untouched], C[..., untouched]))
    print(f"  [{'ok ' if same else 'BAD'}] untouched coordinates {untouched} "
          f"unchanged (only {K} of {N // 2} pairs rotate)")
    ok = ok and same

    moved = not torch.allclose(B_rot[..., :K], B[..., :K])
    print(f"  [{'ok ' if moved else 'BAD'}] rotated coordinates DID change")
    ok = ok and moved

    # every rank slot gets the SAME rotation -> rank Gram invariant
    devB = float((rank_geometry(B)["gram"] - rank_geometry(B_rot)["gram"]).abs().max())
    devC = float((rank_geometry(C)["gram"] - rank_geometry(C_rot)["gram"]).abs().max())
    gram_ok = devB < 1e-4 and devC < 1e-4
    print(f"  [{'ok ' if gram_ok else 'BAD'}] ACTUAL RoPE preserves same-token rank "
          f"Grams (B dev {devB:.2e}, C dev {devC:.2e})")
    ok = ok and gram_ok

    try:
        partial_rope_bc(B, torch.rand(T, H, N))
        print("  [BAD] oversized pair count accepted")
        ok = False
    except ValueError:
        print("  [ok ] pair count exceeding N//2 rejected")
    return ok


def _fixture_write_weighted_geometry():
    """trap_scale weights the WRITE side (B). There is no C * trap_scale."""
    print("\nB write weighting (trap_scale on the write side only):")
    ok = True
    T, H, R, N = 5, 2, 4, 8
    B_rot = torch.randn(T, H, R, N)
    trap = torch.rand(T, H) + 0.5
    Bw = B_rot * trap.unsqueeze(-1).unsqueeze(-1)

    scaled = torch.allclose(Bw.norm(dim=(-2, -1)), B_rot.norm(dim=(-2, -1)) * trap)
    print(f"  [{'ok ' if scaled else 'BAD'}] ||B*trap|| == ||B|| * trap")
    ok = ok and scaled

    g_raw, g_var = rank_geometry(B_rot), rank_geometry(Bw)
    varied = float((g_var["total_norm"] - g_raw["total_norm"]).abs().mean()) > 0
    print(f"  [{'ok ' if varied else 'BAD'}] VARYING trap_scale changes pooled B "
          f"write geometry")
    ok = ok and varied

    gc = rank_geometry(B_rot * 3.0)
    scale_moved = float((gc["total_norm"] - g_raw["total_norm"]).abs().mean()) > 0
    geom_same = (torch.allclose(gc["diff_over_total"], g_raw["diff_over_total"],
                                atol=1e-5)
                 and torch.allclose(gc["participation"], g_raw["participation"],
                                    atol=1e-4))
    print(f"  [{'ok ' if scale_moved and geom_same else 'BAD'}] CONSTANT trap_scale "
          f"changes scale but not normalized geometry (diff ratio, participation)")
    return ok and scale_moved and geom_same


def _fixture_rank_gauge_probe():
    """FIXTURE-ONLY gauge probe, asserting the CORRECT invariance/variance split.

    A random orthogonal rotation of the RANK basis is not something the model
    does; it tests which diagnostics are coordinate-dependent.
    """
    print("\nrank-gauge probe (fixture-only; NOT the model's RoPE):")
    ok = True
    T, H, R, N = 3, 2, 4, 8
    g = torch.Generator().manual_seed(0)
    X = torch.randn(T, H, R, N, generator=g)
    Y = torch.randn(T, H, R, N, generator=g)
    Q = random_rank_rotation(R, generator=g)
    Xr = torch.einsum("rs,thsd->thrd", Q, X)
    Yr = torch.einsum("rs,thsd->thrd", Q, Y)

    g0, g1 = rank_geometry(X), rank_geometry(Xr)
    flat = lambda t: t.reshape(T, H, -1)                          # noqa: E731
    inv = {
        "Gram eigenvalues": torch.allclose(g0["eigvals"], g1["eigvals"], atol=1e-4),
        "participation ratio": torch.allclose(g0["participation"],
                                              g1["participation"], atol=1e-4),
        "total Frobenius norm": torch.allclose(g0["total_norm"], g1["total_norm"],
                                               atol=1e-4),
        "joint B/C cosine": torch.allclose(
            F.cosine_similarity(flat(X), flat(Y), dim=-1),
            F.cosine_similarity(flat(Xr), flat(Yr), dim=-1), atol=1e-5),
    }
    for k, v in inv.items():
        print(f"  [{'ok ' if v else 'BAD'}] INVARIANT: {k}")
        ok = ok and v

    var = {
        "per-slot norms": not torch.allclose(X.norm(dim=-1), Xr.norm(dim=-1)),
        "Gram entries": not torch.allclose(g0["gram"], g1["gram"]),
        "rank-common/differential ratio": not torch.allclose(
            g0["diff_over_total"], g1["diff_over_total"], atol=1e-4),
        "signed per-rank cosine": not torch.allclose(
            bc_alignment(X, Y)["per_rank_cos"],
            bc_alignment(Xr, Yr)["per_rank_cos"], atol=1e-4),
    }
    for k, v in var.items():
        print(f"  [{'ok ' if v else 'BAD'}] VARIANT: {k}")
        ok = ok and v
    print("  note: the rank-common direction is privileged by the all-ones "
          "initialization, NOT by gauge invariance")
    return ok


def _fixture_siso_rank1_controls():
    print("\nSISO rank-1 controls:")
    X = torch.randn(4, 3, 1, 8)
    g = rank_geometry(X)
    pr = torch.allclose(g["participation"], torch.ones_like(g["participation"]),
                        atol=1e-5)
    dz = float(g["diff_norm"].abs().max()) < 1e-6
    rz = (float(g["diff_over_total"].abs().max()) < 1e-6
          and float(g["diff_energy"].abs().max()) < 1e-6)
    print(f"  [{'ok ' if pr else 'BAD'}] participation ratio == 1")
    print(f"  [{'ok ' if dz else 'BAD'}] differential norm == 0")
    print(f"  [{'ok ' if rz else 'BAD'}] both differential ratios == 0")
    return pr and dz and rz


def _fixture_orthogonal_decomposition():
    """total^2 == common^2 + diff^2, so the split is a real decomposition."""
    print("\nrank-common / rank-differential decomposition:")
    X = torch.randn(3, 2, 4, 8)
    g = rank_geometry(X)
    lhs = g["total_norm"] ** 2
    rhs = g["common_norm"] ** 2 + g["diff_norm"] ** 2
    ok = torch.allclose(lhs, rhs, atol=1e-4)
    print(f"  [{'ok ' if ok else 'BAD'}] total^2 == common^2 + diff^2 "
          f"(max dev {float((lhs - rhs).abs().max()):.2e})")
    sq = torch.allclose(g["diff_energy"], g["diff_over_total"] ** 2, atol=1e-5)
    print(f"  [{'ok ' if sq else 'BAD'}] squared-energy ratio == "
          f"(unsquared ratio)^2")
    return ok and sq


def _fixture_no_invalid_paths():
    """No removed object may remain executable, and no QR in the capture loop."""
    print("\nremoved-object check:")
    src = open(__file__).read()
    body = "\n".join(l for l in src.split("\n")
                     if not l.strip().startswith("#")
                     and '"' not in l and "'" not in l)
    ok = True
    for bad in ("C_trap", "C_gamma", "B_gamma", "gram_C_trap", "gram_C_gamma",
                "gram_B_gamma", "trap_C_norm", "gamma_C_norm", "gamma_B_norm"):
        hit = bad in body
        print(f"  [{'ok ' if not hit else 'BAD'}] no executable {bad}")
        ok = ok and not hit
    loop = src[src.index("for n_done, b in enumerate(block_indices):"):
               src.index("    # ---- coverage decided BEFORE")]
    qr_free = "qr(" not in loop and "random_rank_rotation" not in loop
    print(f"  [{'ok ' if qr_free else 'BAD'}] no random QR in the production loop")
    return ok and qr_free


def _fixture_phase_helper_consolidation():
    """phase_objects must be a pure adapter over the shared phase_details."""
    print("\nphase helper consolidation:")
    T, H, K = 6, 3, 2
    raw = torch.randn(T, K)
    delta = torch.rand(T, H) + 0.5
    po = phase_objects(raw, delta)
    pd = phase_details(raw, delta, None)

    exact = all(torch.equal(po[k], pd[j]) for k, j in
                (("increment", "increment"), ("unwrapped", "unwrapped"),
                 ("wrapped", "wrapped"), ("carry", "carry_out")))
    print(f"  [{'ok ' if exact else 'BAD'}] increment/unwrapped/wrapped/carry are "
          f"bitwise the helper's output")
    wind = torch.equal(po["winding"], pd["increment"].sum(0) / TWO_PI)
    print(f"  [{'ok ' if wind else 'BAD'}] winding == sum(increment)/2pi, shape "
          f"{tuple(po['winding'].shape)}")
    shapes = (po["increment"].shape == (T, H, K)
              and po["wrapped"].shape == (T, H, K))
    print(f"  [{'ok ' if shapes else 'BAD'}] capture-facing shapes unchanged "
          f"{tuple(po['increment'].shape)}")
    # per-block reset policy: carried=None
    reset = torch.equal(po["wrapped"], phase_details(raw, delta, None)["wrapped"])
    print(f"  [{'ok ' if reset else 'BAD'}] per-block reset preserved (carried=None)")
    src = open(__file__).read()
    body = "\n".join(l for l in src.split("\n")
                     if not l.strip().startswith("#") and '"' not in l)
    no_local = ("torch.cumsum" not in body) and ("math.pi * delta" not in body)
    print(f"  [{'ok ' if no_local else 'BAD'}] no local phase formula remains in "
          f"this file")
    return exact and wind and shapes and reset and no_local


def _fixture_token_local_paths():
    """MIMO/SISO shapes, V_rank, D, diagonal, gate and collapse algebra."""
    print("\ntoken-local rank pathways:")
    ok = True
    T, H, R, N, P = 5, 2, 4, 8, 3
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, H, P, generator=g)
    z = torch.randn(T, H, P, generator=g)
    B = torch.randn(T, H, R, N, generator=g)
    C = torch.randn(T, H, R, N, generator=g)
    gamma = torch.rand(T, H, generator=g) + 0.2
    D = torch.randn(H, generator=g)
    mx = torch.randn(H, R, P, generator=g)
    mz = torch.randn(H, R, P, generator=g)
    mo = torch.randn(H, R, P, generator=g)

    r = token_local_paths(x, z, B, C, gamma, D, mx, mz, mo, True)
    shp = (r["V_rank"].shape == (T, H, R, P)
           and r["gate_pre"].shape == (T, H, R, P)
           and r["gate_factor"].shape == (T, H, R, P)
           and r["collapse_weight"].shape == (H, R, P)
           and r["qk_diag"].shape == (T, H, R, R)
           and r["y_diag"].shape == (T, H, R, P)
           and r["diag_collapsed"].shape == (T, H, P))
    print(f"  [{'ok ' if shp else 'BAD'}] MIMO shapes: V_rank/gate {(T,H,R,P)}, "
          f"qk_diag {(T,H,R,R)}, collapsed {(T,H,P)}")
    ok = ok and shp

    v_ok = torch.allclose(r["V_rank"], torch.einsum("thp,hrp->thrp", x, mx))
    print(f"  [{'ok ' if v_ok else 'BAD'}] V_rank == x * mimo_x")
    ok = ok and v_ok

    d_ok = torch.allclose(r["y_D"], D.view(1, H, 1, 1) * r["V_rank"])
    d_not_raw = not torch.allclose(
        r["y_D"], (D.view(1, H, 1, 1) * x.unsqueeze(2)).expand_as(r["y_D"]))
    print(f"  [{'ok ' if d_ok and d_not_raw else 'BAD'}] y_D == D * V_rank, and "
          f"D * raw_x does NOT reproduce it")
    ok = ok and d_ok and d_not_raw

    qk = torch.einsum("thrn,thsn->thrs", C, B)
    manual = torch.einsum("thrs,thsp->thrp", qk, r["V_rank"]) \
        * gamma.unsqueeze(-1).unsqueeze(-1)
    diag_ok = torch.allclose(r["y_diag"], manual, atol=1e-5)
    print(f"  [{'ok ' if diag_ok else 'BAD'}] diagonal uses UNROTATED B/C and gamma")
    ok = ok and diag_ok

    trap = gamma * 2.0 + 0.5
    wrong = torch.einsum("thrs,thsp->thrp", qk, r["V_rank"]) \
        * trap.unsqueeze(-1).unsqueeze(-1)
    sub_fails = not torch.allclose(r["y_diag"], wrong, atol=1e-5)
    print(f"  [{'ok ' if sub_fails else 'BAD'}] substituting trap_scale FAILS")
    ok = ok and sub_fails

    sums = (torch.allclose(r["diag_collapsed"], r["diag_collapse_r"].sum(2))
            and torch.allclose(r["D_collapsed"], r["D_collapse_r"].sum(2)))
    chain = (torch.allclose(r["diag_collapse_r"],
                            r["y_diag"] * r["gate_factor"] * r["collapse_weight"])
             and torch.allclose(r["D_collapse_r"],
                                r["y_D"] * r["gate_factor"] * r["collapse_weight"]))
    print(f"  [{'ok ' if sums and chain else 'BAD'}] rank sums reproduce the "
          f"collapsed local paths; gate then collapse chain exact")
    ok = ok and sums and chain

    # SISO
    rs = token_local_paths(x, z, B[:, :, :1], C[:, :, :1], gamma, D,
                           None, None, None, False)
    siso_ok = (rs["V_rank"].shape == (T, H, 1, P)
               and torch.equal(rs["collapse_weight"],
                               torch.ones_like(rs["collapse_weight"]))
               and torch.allclose(rs["V_rank"].squeeze(2), x))
    print(f"  [{'ok ' if siso_ok else 'BAD'}] SISO: singleton rank, "
          f"collapse_weight==1, V_rank == x")
    return ok and siso_ok


def _fixture_gate_per_rank():
    """The gate acts independently PER RANK, before the rank sum."""
    print("\ngate is per-rank:")
    T, H, R, N, P = 4, 2, 4, 8, 3
    g = torch.Generator().manual_seed(8)
    args = dict(x=torch.randn(T, H, P, generator=g), z=torch.randn(T, H, P, generator=g),
                B_post=torch.randn(T, H, R, N, generator=g),
                C_post=torch.randn(T, H, R, N, generator=g),
                gamma=torch.rand(T, H, generator=g) + 0.2,
                D=torch.randn(H, generator=g),
                mimo_x=torch.randn(H, R, P, generator=g),
                mimo_z=torch.randn(H, R, P, generator=g),
                mimo_o=torch.randn(H, R, P, generator=g), is_mimo=True)
    base = token_local_paths(**args)
    a2 = dict(args); a2["mimo_z"] = args["mimo_z"].clone(); a2["mimo_z"][:, 1, :] *= 2.5
    mod = token_local_paths(**a2)

    moved = not torch.allclose(base["gate_factor"][:, :, 1], mod["gate_factor"][:, :, 1])
    others = [i for i in range(R) if i != 1]
    same = torch.allclose(base["gate_factor"][:, :, others],
                          mod["gate_factor"][:, :, others], atol=1e-7)
    collapsed_moved = not torch.allclose(base["diag_collapsed"], mod["diag_collapsed"])
    print(f"  [{'ok ' if moved and same else 'BAD'}] changing mimo_z rank 1 moves "
          f"only rank 1's gate_factor")
    print(f"  [{'ok ' if collapsed_moved else 'BAD'}] collapsed output still moves "
          f"(ranks are SUMMED, not gated jointly)")
    return moved and same and collapsed_moved


def _fixture_collapsed_residual():
    """actual_pre_out decomposes into residual + diagonal + D, by construction."""
    print("\ncollapsed earlier-path residual:")
    T, H, P = 4, 2, 3
    diag_c = torch.randn(T, H, P)
    D_c = torch.randn(T, H, P)
    earlier_true = torch.randn(T, H, P)
    actual_pre_out = earlier_true + diag_c + D_c          # synthetic ground truth

    residual = actual_pre_out - diag_c - D_c
    exact = torch.allclose(residual, earlier_true, atol=1e-6)
    print(f"  [{'ok ' if exact else 'BAD'}] residual recovers the synthetic earlier "
          f"path (max err {float((residual - earlier_true).abs().max()):.2e})")
    recon = torch.allclose(actual_pre_out, residual + diag_c + D_c, atol=1e-6)
    print(f"  [{'ok ' if recon else 'BAD'}] actual_pre_out == residual + diagonal + D")
    # cancellation: norms are NOT additive, so no percentage attribution
    n_sum = (diag_c.norm(dim=-1) + D_c.norm(dim=-1) + residual.norm(dim=-1))
    n_tot = actual_pre_out.norm(dim=-1)
    cancels = bool((n_sum > n_tot + 1e-6).any())
    print(f"  [{'ok ' if cancels else 'BAD'}] paths CANCEL: sum of norms exceeds the "
          f"norm of the sum, so norm ratios are not attribution fractions")
    return exact and recon and cancels


def _fixture_no_stale_local_metrics():
    print("\nremoved stale metrics:")
    src = open(__file__).read()
    body = "\n".join(l for l in src.split("\n")
                     if not l.strip().startswith("#")
                     and '"' not in l and "'" not in l)
    ok = True
    for bad in ("mix_out", "feedthrough_norm", "mixer_out_norm"):
        hit = bad in body
        print(f"  [{'ok ' if not hit else 'BAD'}] no executable {bad}")
        ok = ok and not hit
    for w in ("earlier_per_rank", "kernel_state", "full_recurrent_utilization",
              "percentage_pathway_attribution"):
        held = w in WITHHELD_METRICS
        print(f"  [{'ok ' if held else 'BAD'}] {w} withheld")
        ok = ok and held
    claims = [q for q in BLOCK_QUANTITIES
              if "earlier" in q and "residual" not in q]
    print(f"  [{'ok ' if not claims else 'BAD'}] no metric claims a full per-rank "
          f"recurrent contribution {claims}")
    return ok and not claims


def _self_check():
    ok = True
    tc, man = _synthetic_contract()
    tc = _NpzLike(tc)

    print("contract validation:")
    f = validate_contract(tc, man, "a")
    print(f"  [{'ok ' if not f else 'BAD'}] clean artifact validates {f}")
    ok = ok and not f

    # --- block iteration properties ---
    offs, ids = tc["a_offsets"], tc["a_ids"]
    emitted, spans = [], []
    for b in range(len(offs) - 1):
        s0, e0 = int(offs[b]), int(offs[b + 1])
        emitted.append(b)
        spans.append((s0, e0))
    once = sorted(emitted) == list(range(len(offs) - 1))
    contiguous = all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    covers = spans[0][0] == 0 and spans[-1][1] == len(ids)
    starts_bos = all(ids[s0] == man["tokenizer"]["bos_id"] for s0, _ in spans)
    print("\nblock iteration:")
    print(f"  [{'ok ' if once else 'BAD'}] each block emitted exactly once")
    print(f"  [{'ok ' if contiguous and covers else 'BAD'}] no block crosses an "
          f"offset boundary; spans tile the flat array")
    print(f"  [{'ok ' if starts_bos else 'BAD'}] every emitted block starts with BOS")
    meta_ok = all(int(tc["a_block_id"][b]) == b for b in emitted)
    print(f"  [{'ok ' if meta_ok else 'BAD'}] metadata maps to the correct block")
    ok = ok and once and contiguous and covers and starts_bos and meta_ok

    # --- budget selection is an exact prefix from the recorded boundary ---
    print("\nbudget selection:")
    r_small, n_small = select_budget(tc, man, 5)
    r_big, n_big = select_budget(tc, man, 12)
    prefix_ok = (n_small < n_big
                 and list(range(n_small)) == list(range(n_big))[:n_small])
    print(f"  [{'ok ' if prefix_ok else 'BAD'}] smaller budget is the exact block "
          f"prefix: {n_small} of {n_big}")
    ok = ok and prefix_ok
    try:
        select_budget(tc, man, 999)
        print("  [BAD] unrecorded budget accepted")
        ok = False
    except ContractError:
        print("  [ok ] unrecorded budget rejected")

    # --- failure cases ---
    print("\nfailure cases:")
    cases = []
    t2, m2 = _synthetic_contract(); m2["capture_must_forward_blocks_independently"] = False
    cases.append(("missing independence flag", _NpzLike(t2), m2, "independently"))
    t3, m3 = _synthetic_contract(); m3["schema_version"] = 2
    cases.append(("old schema_version", _NpzLike(t3), m3, "schema_version"))
    t4, m4 = _synthetic_contract()
    t4["a_ids"] = t4["a_ids"].reshape(2, -1)
    cases.append(("2-D stream A (old packed layout)", _NpzLike(t4), m4, "FLAT"))
    t5, m5 = _synthetic_contract()
    t5["a_offsets"] = t5["a_offsets"].copy(); t5["a_offsets"][2] += 1
    cases.append(("corrupted offsets", _NpzLike(t5), m5, "valid_len"))
    t6, m6 = _synthetic_contract()
    t6["a_valid_len"] = t6["a_valid_len"].copy(); t6["a_valid_len"][0] += 3
    cases.append(("corrupted valid_len", _NpzLike(t6), m6, "valid_len"))
    t7, m7 = _synthetic_contract()
    t7["a_ids"] = t7["a_ids"].copy(); t7["a_ids"][int(t7["a_offsets"][1])] = 77
    cases.append(("block not starting with BOS", _NpzLike(t7), m7, "BOS"))
    t8, m8 = _synthetic_contract()
    t8["a_block"] = np.zeros(3)            # retired key from the old schema
    cases.append(("retired a_block key present", _NpzLike(t8), m8, "retired"))
    t9, m9 = _synthetic_contract()
    t9["a_label"] = t9["a_label"][:-1]
    cases.append(("metadata cardinality wrong", _NpzLike(t9), m9, "expected"))

    for desc, t, m, needle in cases:
        got = validate_contract(t, m, "a")
        hit = any(needle in x for x in got)
        print(f"  [{'ok ' if hit else 'BAD'}] {desc:36s} -> "
              f"{got[0][:52] if got else 'ACCEPTED'}")
        ok = ok and hit

    # --- semantic label vs source category ---
    print("\nsemantic label vs source category:")
    tcb, manb = _synthetic_contract(n_blocks=4)
    tcb["a_category"] = np.array(["balanced"] * 4, dtype="<U128")
    tcb["a_label"] = np.array(["Humor", "Data", "Humor", "Tool Use"], dtype="<U128")
    groups = {}
    for b in range(4):
        groups.setdefault(str(tcb["a_label"][b]), []).append(b)
    cat_groups = {}
    for b in range(4):
        cat_groups.setdefault(str(tcb["a_category"][b]), []).append(b)
    sep = len(groups) == 3 and len(cat_groups) == 1
    print(f"  [{'ok ' if sep else 'BAD'}] 4 blocks all category='balanced' -> "
          f"{len(cat_groups)} category bucket, {len(groups)} semantic groups "
          f"{ {k: v for k, v in groups.items()} }")
    ok = ok and sep

    # --- position metadata validation ---
    print("\nposition metadata:")
    t_ok, _ = _synthetic_contract(); t_ok = _NpzLike(t_ok)
    pf = validate_block_positions(t_ok, "a", 1, int(t_ok["a_offsets"][1]),
                                  int(t_ok["a_offsets"][2]))
    print(f"  [{'ok ' if not pf else 'BAD'}] clean block passes {pf}")
    ok = ok and not pf
    t_bad, _ = _synthetic_contract(); t_bad = _NpzLike(t_bad)
    t_bad["a_token_pos"] = t_bad["a_token_pos"].copy()
    t_bad["a_token_pos"][int(t_bad["a_offsets"][1]) + 1] = 99
    pf2 = validate_block_positions(t_bad, "a", 1, int(t_bad["a_offsets"][1]),
                                   int(t_bad["a_offsets"][2]))
    print(f"  [{'ok ' if pf2 else 'BAD'}] corrupted token_pos caught: {pf2[:1]}")
    ok = ok and bool(pf2)
    t_bad2, _ = _synthetic_contract(); t_bad2 = _NpzLike(t_bad2)
    t_bad2["a_token_doc_pos"] = t_bad2["a_token_doc_pos"].copy()
    t_bad2["a_token_doc_pos"][int(t_bad2["a_offsets"][1])] = 0
    pf3 = validate_block_positions(t_bad2, "a", 1, int(t_bad2["a_offsets"][1]),
                                   int(t_bad2["a_offsets"][2]))
    caught = any("BOS token_doc_pos" in x for x in pf3)
    print(f"  [{'ok ' if caught else 'BAD'}] corrupted BOS doc-position caught")
    ok = ok and caught

    # --- token coverage arithmetic on a clean prefix ---
    print("\ntoken coverage:")
    tcl = _NpzLike(_synthetic_contract()[0])
    o = tcl["a_offsets"]
    k = 4
    exp_valid = int(o[k] - o[0])
    exp_content = int(np.sum(tcl["a_n_content_tokens"][:k]))
    got_valid = sum(int(o[i + 1] - o[i]) for i in range(k))
    got_content = sum(int(tcl["a_n_content_tokens"][i]) for i in range(k))
    match = exp_valid == got_valid and exp_content == got_content
    print(f"  [{'ok ' if match else 'BAD'}] prefix of {k}: valid {got_valid}/"
          f"{exp_valid}, content {got_content}/{exp_content}")
    mism = (got_valid - 1) == exp_valid
    print(f"  [{'ok ' if not mism else 'BAD'}] a short count would not equal expected")
    ok = ok and match and not mism

    # --- max-blocks rejection ---
    print("\nmax-blocks:")
    for v in (0, -3):
        rejected = v <= 0
        print(f"  [{'ok ' if rejected else 'BAD'}] --max-blocks {v} rejected")
        ok = ok and rejected

    # --- artifact digest ---
    print("\nartifact digest:")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        npz = os.path.join(d, "t.npz")
        np.savez_compressed(npz, x=np.arange(10))
        dig = sha256_file(npz)
        e, g, fails = verify_artifact_digest(npz, {"artifact_sha256": dig})
        print(f"  [{'ok ' if not fails else 'BAD'}] matching digest passes")
        ok = ok and not fails
        np.savez_compressed(npz, x=np.arange(11))     # mutate AFTER hashing
        e2, g2, fails2 = verify_artifact_digest(npz, {"artifact_sha256": dig})
        print(f"  [{'ok ' if fails2 else 'BAD'}] mutated artifact fails: "
              f"{fails2[0][:60] if fails2 else 'ACCEPTED'}")
        ok = ok and bool(fails2)
        _, _, fails3 = verify_artifact_digest(npz, {})
        print(f"  [{'ok ' if fails3 else 'BAD'}] missing digest fails")
        ok = ok and bool(fails3)

    # --- no executable phase-labelled output remains ---
    print("\nphase withholding:")
    src = open(__file__).read()
    exec_lines = [l for l in src.split("\n")
                  if "phase_stats(" in l
                  and not l.strip().startswith("#")
                  and '"' not in l]          # exclude string literals / this check
    print(f"  [{'ok ' if not exec_lines else 'BAD'}] no executable phase_stats call "
          f"{exec_lines}")
    ok = ok and not exec_lines

    # --- preflight ordering: a failure must load ZERO models ---
    print("\npreflight precedes model loading:")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        def _write(tc_dict, man_dict, name="t.npz"):
            path = os.path.join(d, name)
            np.savez_compressed(path, **tc_dict)
            man_dict = dict(man_dict)
            man_dict["artifact_sha256"] = sha256_file(path)
            with open(path.replace(".npz", ".manifest.json"), "w") as fh:
                json.dump(man_dict, fh)
            return path

        t_ok2, m_ok2 = _synthetic_contract()
        good = _write(t_ok2, m_ok2, "good.npz")

        before = MODEL_LOAD_COUNT[0]
        plan, fails = preflight(good, "a", budget=5)
        print(f"  [{'ok ' if plan and not fails else 'BAD'}] clean artifact "
              f"preflights: blocks={plan['blocks_expected'] if plan else '-'} "
              f"valid={plan['valid_tokens_expected'] if plan else '-'} "
              f"content={plan['content_tokens_expected'] if plan else '-'}")
        ok = ok and bool(plan) and not fails

        # corrupt n_content_tokens -> caught BEFORE any model load. Uses stream
        # B so the PER-BLOCK offset-derived check is what fires; on stream A the
        # budget record's realized_content_tokens would intercept it first
        # (also a valid catch, but a different check).
        t_bad3, m_bad3 = _synthetic_contract(prefix="b", with_budget=False)
        t_bad3["b_n_content_tokens"] = t_bad3["b_n_content_tokens"].copy()
        t_bad3["b_n_content_tokens"][2] += 5
        bad = _write(t_bad3, m_bad3, "bad.npz")
        plan2, fails2 = preflight(bad, "b")
        caught = plan2 is None and any("n_content_tokens" in x for x in fails2)
        print(f"  [{'ok ' if caught else 'BAD'}] corrupt n_content_tokens caught: "
              f"{fails2[0][:56] if fails2 else 'ACCEPTED'}")
        ok = ok and caught
        no_loads = MODEL_LOAD_COUNT[0] == before
        print(f"  [{'ok ' if no_loads else 'BAD'}] zero model loads during "
              f"preflight (counter {before} -> {MODEL_LOAD_COUNT[0]})")
        ok = ok and no_loads

        # digest mutation caught in preflight
        with np.load(good) as z:
            arrs = {k: z[k] for k in z.files}
        arrs["a_ids"] = arrs["a_ids"].copy()
        arrs["a_ids"][-1] = int(arrs["a_ids"][-1]) + 1   # a REAL content change
        np.savez_compressed(good, **arrs)          # manifest hash now stale
        _, fails3 = preflight(good, "a", budget=5)
        dig_caught = any("digest mismatch" in x for x in fails3)
        print(f"  [{'ok ' if dig_caught else 'BAD'}] post-hash mutation caught "
              f"in preflight")
        ok = ok and dig_caught

    # --- processed content derives from the slice, not metadata ---
    print("\nprocessed counts derive from the forwarded slice:")
    tcx = _NpzLike(_synthetic_contract()[0])
    ox = tcx["a_offsets"]
    v_proc = c_proc = 0
    for b in range(len(ox) - 1):
        seq = tcx["a_ids"][int(ox[b]):int(ox[b + 1])]
        v_proc += len(seq)
        c_proc += len(seq) - 1
    v_exp = int(ox[-1] - ox[0])
    c_exp = v_exp - (len(ox) - 1)
    match = v_proc == v_exp and c_proc == c_exp
    print(f"  [{'ok ' if match else 'BAD'}] valid {v_proc}/{v_exp}, "
          f"content {c_proc}/{c_exp} (content = valid - one BOS per block)")
    ok = ok and match

    # --- coverage gate decides publication ---
    print("\npublication gate:")
    complete = (5 == 5 and 20 == 20 and 15 == 15)
    incomplete = (4 == 5 and 18 == 20 and 14 == 15)
    print(f"  [{'ok ' if complete else 'BAD'}] equal counts reach the publish decision")
    print(f"  [{'ok ' if not incomplete else 'BAD'}] any mismatch blocks publication")
    ok = ok and complete and not incomplete

    # --- LayerCapture.finish with phase disabled ---
    print("\nphase-disabled finish path:")
    try:
        cap = LayerCapture(nheads=3, rank=4, device="cpu", n_layers_idx=0)
        assert cap.phase is None
        out = cap.finish()
        no_phase = not any(("phase" in k) or ("concentration" in k) for k in out)
        no_inj = not any(w in k for w in WITHHELD_METRICS for k in out)
        print(f"  [{'ok ' if no_phase else 'BAD'}] finish() emits no phase/"
              f"concentration key")
        print(f"  [{'ok ' if no_inj else 'BAD'}] finish() emits no withheld "
              f"injection metric; keys={sorted(out)[:4]}...")
        ok = ok and no_phase and no_inj
    except Exception as e:  # noqa: BLE001
        print(f"  [BAD] finish() crashed with phase disabled: "
              f"{type(e).__name__}: {e}")
        ok = False

    ok = _fixture_bc_reconstruction() and ok
    ok = _fixture_rank_geometry() and ok
    ok = _fixture_bc_alignment() and ok
    ok = _fixture_actual_partial_rope() and ok
    ok = _fixture_write_weighted_geometry() and ok
    ok = _fixture_rank_gauge_probe() and ok
    ok = _fixture_siso_rank1_controls() and ok
    ok = _fixture_orthogonal_decomposition() and ok
    ok = _fixture_no_invalid_paths() and ok
    ok = _fixture_phase_helper_consolidation() and ok
    ok = _fixture_token_local_paths() and ok
    ok = _fixture_gate_per_rank() and ok
    ok = _fixture_collapsed_residual() and ok
    ok = _fixture_no_stale_local_metrics() and ok
    ok = _fixture_recurrence_scalars() and ok
    ok = _fixture_phase() and ok
    ok = _fixture_roles() and ok
    ok = _fixture_halflife_censoring() and ok
    ok = _fixture_masked_halflife() and ok
    ok = _fixture_winding_resolution() and ok
    ok = _fixture_block_role_summaries() and ok
    ok = _fixture_rank_collapsed_summaries() and ok
    ok = _fixture_block_recorder_layer_counts() and ok
    ok = _fixture_distribution_coverage() and ok

    print(f"\nwithheld invalid metrics: {list(WITHHELD_METRICS)}")
    print(f"  block quantities exclude them: "
          f"{all(w not in BLOCK_QUANTITIES for w in WITHHELD_METRICS)}")
    print("\nself-check " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="e.g. mimo-187m")
    ap.add_argument("--revision", default=None,
                    help="exact locally cached checkpoint commit")
    ap.add_argument("--gpu-probe-report", default=None,
                    help="required terminal GPU probe report gating capture")
    ap.add_argument("--stream", choices=["a", "b"], default="b")
    ap.add_argument("--token-contract", default="token_contract.npz")
    ap.add_argument("--layers", default=None,
                    help="comma list; default = all layers")
    ap.add_argument("--budget", type=int, default=None,
                    help="stream A only: select a RECORDED manifest budget "
                         "boundary (e.g. 150000 or 500000)")
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="declared pre-existing limit; coverage is judged against it")
    ap.add_argument("--forward-batch-size", type=int, default=128,
                    help="independent blocks packed into one varlen forward "
                         "using cu_seqlens (default: 128)")
    ap.add_argument("--forward-max-tokens", type=int, default=0,
                    help="maximum valid tokens per packed forward; 0 disables "
                         "the token cap, and complete blocks are never split")
    ap.add_argument("--max-output-gib", type=float, default=1.5,
                    help="refuse to silently create a larger per-block artifact")
    ap.add_argument("--allow-large-output", action="store_true")
    ap.add_argument("--profile-subphases", action="store_true",
                    help="record detailed CUDA-event timings for derivation "
                         "subphases; intended for bounded profiling runs")
    ap.add_argument("--self-check", action="store_true",
                    help="offline synthetic fixture; no CUDA, model or network")
    ap.add_argument("--out", default="capture_stage_b.npz")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if _self_check() else 1)
    if not args.model:
        print("[STOP] --model is required")
        sys.exit(1)

    # ===== PREFLIGHT: everything structural, BEFORE any CUDA/model load =====
    plan, pfails = preflight(args.token_contract, args.stream,
                             budget=args.budget, max_blocks=args.max_blocks)
    if args.forward_batch_size < 1:
        pfails.append("--forward-batch-size must be >= 1")
    if args.forward_max_tokens < 0:
        pfails.append("--forward-max-tokens must be >= 0")
    ck = None
    probe = None
    if not args.gpu_probe_report:
        pfails.append("--gpu-probe-report is required; capture is parity-gated")
    else:
        try:
            with open(args.gpu_probe_report) as fh:
                probe = json.load(fh)
        except Exception as e:  # noqa: BLE001
            pfails.append(f"GPU probe report unreadable: {type(e).__name__}: {e}")
    try:
        ck = resolve_checkpoint(args.model, revision=args.revision, local_only=True)
    except (CheckpointResolveError, ValueError) as e:
        pfails.append(f"checkpoint resolution failed: {e}")

    if probe is not None and ck is not None and not pfails:
        verdicts = probe.get("verdicts") or {}
        required_pass = ("g1", "g2", "g4", "g5", "g6", "g7")
        bad_gates = {g: verdicts.get(g) for g in required_pass
                     if verdicts.get(g) != "PASS"}
        if bad_gates:
            pfails.append(f"GPU probe required gates are not PASS: {bad_gates}")
        if verdicts.get("overall") not in ("PASS", "BOUNDARY"):
            pfails.append(f"GPU probe overall={verdicts.get('overall')}, expected PASS/BOUNDARY")
        if (probe.get("archive") or {}).get("status") != "OK":
            pfails.append("GPU probe HF archive status is not OK")
        prov = ck.provenance()
        commit = prov.get("resolved_commit")
        probe_commit = (((probe.get("gates") or {}).get("g1") or {})
                        .get("provenance", {}).get("resolved_commit"))
        g2_commit = (((((probe.get("gates") or {}).get("g2") or {})
                       .get("revisions") or {}).get(args.model) or {})
                     .get("commit"))
        if not commit or probe_commit != commit or g2_commit != commit:
            pfails.append("checkpoint commit disagrees across capture/G1/G2: "
                          f"capture={commit}, G1={probe_commit}, G2={g2_commit}")
        probe_digest = (probe.get("token_contract") or {}).get("digest")
        if probe_digest != plan["digest_computed"]:
            pfails.append("token-contract digest disagrees with GPU probe: "
                          f"capture={plan['digest_computed']} probe={probe_digest}")
        requested_layers = (list(range(ck.load_config()["n_layer"]))
                            if not args.layers
                            else [int(x) for x in args.layers.split(",")])
        probed_layers = (((probe.get("gates") or {}).get("g5") or {})
                         .get("layers_hooked"))
        if probed_layers != requested_layers:
            pfails.append("capture layers disagree with G5 hook surface: "
                          f"capture={requested_layers} probe={probed_layers}")
    if pfails:
        print(f"[STOP] token contract preflight failed ({args.token_contract}):")
        for x in pfails:
            print("   ", x)
        print(f"\nno model was loaded (load count {MODEL_LOAD_COUNT[0]})")
        sys.exit(1)
    print_preflight(plan)

    tc, tc_manifest = plan["tc"], plan["manifest"]
    exp_dig, got_dig = plan["digest_expected"], plan["digest_computed"]
    budget_record = plan["budget_record"]
    block_indices = plan["block_indices"]
    declared_limit = plan["declared_limit"] is not None
    n_total, n_avail = plan["n_total"], plan["n_avail"]

    # ===== only now is the model constructed =====
    dev = "cuda"
    assert ck is not None and probe is not None
    repo = ck.repo_id
    model = load_model(ck.path, device=dev)
    model.requires_grad_(False)  # inference-only atlas; derived tensors must be NumPy-safe
    layers = model.backbone.layers
    mixer0 = layers[0].mixer
    spec = InProjSpec.from_mixer(mixer0)
    nheads, rank = spec.nheads, spec.mimo_rank
    is_mimo = rank > 1

    sel = (list(range(len(layers))) if not args.layers
           else [int(x) for x in args.layers.split(",")])
    for li in sel:
        assert 0 <= li < len(layers), f"layer {li} out of range for {args.model}"

    # ---- accumulators, keyed by (class, layer) ----
    # keyed by SEMANTIC LABEL, not source category
    caps = defaultdict(lambda: {li: LayerCapture(nheads, rank, dev, li) for li in sel})
    # Stage 4B1: (semantic label, layer, boundary role) -> RoleCapture
    n_pairs = spec.n_rope_angles
    roles = defaultdict(lambda: {li: {r: RoleCapture(nheads, n_pairs, dev)
                                      for r in POSITION_ROLES} for li in sel})
    est = estimate_artifact_bytes(len(block_indices), len(sel), nheads,
                                  rank=rank, n_labels=EST_N_LABELS)
    print(f"projected COMPLETE payload: {est['complete_payload_gib']:.2f} GiB "
          f"({est['dtype']}, {est['n_role_fields']} role fields x "
          f"{est['n_blocks']} blocks x {est['n_layers']} layers x "
          f"{est['n_roles']} roles x {est['n_heads']} heads)")
    if est["total_gib"] > args.max_output_gib and not args.allow_large_output:
        print(f"[STOP] projected output {est['total_gib']:.2f} GiB exceeds "
              f"--max-output-gib {args.max_output_gib}; pass "
              f"--allow-large-output to proceed deliberately")
        sys.exit(1)
    blockrole = BlockRoleRecorder(len(block_indices), len(sel), nheads,
                                  rank=rank, device=dev)
    recorder = BlockRecorder(BLOCK_QUANTITIES, len(sel), nheads, device=dev,
                             n_blocks=len(block_indices))
    layer_pos = {li: i for i, li in enumerate(sel)}
    timer = PhaseTimer()
    # Upper bound for a block with all three roles populated under the retired
    # implementation. This includes explicit D2H conversions, the hidden
    # bool(kh.any()) per-head checks, and Histogram.update's former
    # int(valid.sum()) conversion.
    legacy_role_fields = 26
    old_syncs_per_block = len(sel) * (
        len(BLOCK_QUANTITIES)
        + legacy_role_fields * len(POSITION_ROLES) + len(BLOCK_WINDING_FIELDS)
        + nheads * len(POSITION_ROLES)
        + len(HIST_QUANTITIES)
        + (len(SCALAR_HIST_EDGES) + len(DERIVED_POSITIVE) + 1)
          * len(POSITION_ROLES))
    print(f"sync model: 2 x selected_layers batched D2H copies per packed "
          f"forward. The per-call design this replaces could issue up to "
          f"~{old_syncs_per_block} individual GPU->CPU synchronizations per "
          f"full-role block.", flush=True)

    # ---- hooks ----
    grabbed = {}

    def mk_inproj(li):
        def hook(_m, _inp, out):
            grabbed[("in", li)] = out.detach()
        return hook

    def mk_outproj_pre(li):
        def hook(_m, inp):
            # input to mixer.out_proj: the RANK-COLLAPSED, POST-GATE tensor.
            # NOT the final mixer output.
            grabbed[("pre_out", li)] = inp[0].detach()
        return hook

    def mk_block(li):
        def hook(_m, _inp, out):
            grabbed[("resid", li)] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def mk_mlp(li):
        def hook(_m, inp):
            grabbed[("mlp", li)] = inp[0].detach()
        return hook

    handles = []
    for li in sel:
        blk = layers[li]
        handles.append(blk.mixer.in_proj.register_forward_hook(mk_inproj(li)))
        handles.append(blk.mixer.out_proj.register_forward_pre_hook(mk_outproj_pre(li)))
        handles.append(blk.register_forward_hook(mk_block(li)))
        handles.append(blk.mlp.fc2.register_forward_pre_hook(mk_mlp(li)))

    # ---- block iteration: ONE BLOCK PER FORWARD, never concatenated ----
    prefix = args.stream
    ids = tc[f"{prefix}_ids"]
    offs = plan["offsets"]

    meta = {k: tc[f"{prefix}_{k}"] for k in
            ("doc_id", "block_id", "prompt_id", "source_row", "source_file",
             "label", "category", "valid_len", "chunk_index", "n_chunks",
             "n_content_tokens")}

    print(f"{args.model} stream {prefix}: {len(block_indices)} blocks "
          f"(of {n_total} in artifact), layers {sel}", flush=True)
    if WITHHELD_METRICS:
        print(f"WITHHELD (invalid, awaiting Stage 4B): {list(WITHHELD_METRICS)}")

    seen_tokens, content_seen, t_start = 0, 0, time.time()
    phase_consistency = 0.0
    pos_counts = {"bos": 0, "interior": 0, "final": 0}
    per_metric_excluded = {q: 0 for q in BLOCK_QUANTITIES}

    def submark(name):
        if args.profile_subphases:
            timer.mark(name)

    forward_batches = 0
    packed_slices = {}
    forward_packs = plan_forward_packs(
        block_indices, offs, args.forward_batch_size, args.forward_max_tokens)
    pack_idx = 0
    pack_start = 0
    pack_end = len(forward_packs[0])
    pack_token_counts = [sum(int(offs[b + 1]) - int(offs[b]) for b in pack)
                         for pack in forward_packs]
    print(f"forward packing: {len(forward_packs)} launches, "
          f"<= {args.forward_batch_size} blocks, "
          f"token cap={args.forward_max_tokens or 'disabled'}, "
          f"realized max={max(pack_token_counts)} tokens", flush=True)
    for n_done, b in enumerate(block_indices):
        if n_done == pack_start:
            packed_blocks = forward_packs[pack_idx]
            packed_seqs = [ids[int(offs[bb]):int(offs[bb + 1])]
                           for bb in packed_blocks]
            packed_lengths = np.asarray(
                [len(x) for x in packed_seqs], dtype=np.int32)
            packed_cu = np.empty(len(packed_lengths) + 1, dtype=np.int32)
            packed_cu[0] = 0
            np.cumsum(packed_lengths, out=packed_cu[1:])
            packed_ids = np.concatenate(packed_seqs)
            packed_slices = {
                int(bb): (int(packed_cu[i]), int(packed_cu[i + 1]))
                for i, bb in enumerate(packed_blocks)
            }
            batch = torch.as_tensor(
                packed_ids, device=dev, dtype=torch.long).unsqueeze(0)
            cu_seqlens = torch.as_tensor(
                packed_cu, device=dev, dtype=torch.int32)
            grabbed.clear()
            with timer.phase("forward"):
                with torch.inference_mode():
                    model(batch, cu_seqlens=cu_seqlens)
            forward_batches += 1
            pending_roles = {li: [] for li in sel}

        s0, e0 = int(offs[b]), int(offs[b + 1])
        seq = ids[s0:e0]
        n = len(seq)
        if n < 1:
            # unreachable: preflight rejects empty blocks. If it ever fires,
            # STOP rather than silently shrinking coverage mid-capture.
            print(f"[STOP] block {b} is empty after preflight passed")
            sys.exit(1)

        # NOTE no structural validation here: preflight already validated every
        # selected block. Once CUDA is running a block either processes or the
        # run STOPs -- skipping would silently shrink coverage after the fact.

        # Exact slice inside the packed varlen forward. cu_seqlens makes every
        # block independent: its own BOS and convolution/phase/recurrence reset.
        # There is no padding and no state can cross this boundary.
        pack_s, pack_e = packed_slices[int(b)]
        if pack_e - pack_s != n:
            raise ContractError(
                f"packed slice length drift for block {b}: "
                f"{pack_e - pack_s} != {n}")

        # POSITION POLICY: the complete unpadded block is forwarded, INCLUDING
        # its final token. The old code dropped it because right padding made
        # trap_scale read a pad token. The official kernel EMITS the final token,
        # but it is BOUNDARY-CONDITIONED and is not assumed exchangeable with
        # interior positions. Roles are recorded per block; the online metrics
        # still POOL positions, so no role-conditioned statistic exists yet.
        n_bos = 1
        n_final = 1 if n > 1 else 0
        n_interior = max(n - 2, 0)
        pos_counts["bos"] += n_bos
        pos_counts["final"] += n_final
        pos_counts["interior"] += n_interior
        # processed counts come from the ACTUAL forwarded slice, never from
        # artifact metadata: otherwise coverage would compare metadata to itself
        seen_tokens += len(seq)
        content_seen += len(seq) - 1

        recorder.put_meta(n_done, {
            "block": int(meta["block_id"][b]), "doc_id": int(meta["doc_id"][b]),
            "prompt_id": int(meta["prompt_id"][b]),
            "label": str(meta["label"][b]), "category": str(meta["category"][b]),
            "source_row": int(meta["source_row"][b]),
            "source_file": str(meta["source_file"][b]),
            "chunk_index": int(meta["chunk_index"][b]),
            "n_chunks": int(meta["n_chunks"][b]),
            "valid_len": int(meta["valid_len"][b]),
            "n_content_tokens": int(meta["n_content_tokens"][b]),
            "n_used": int(n),
            # exact token mapping, so token roles are reconstructable offline
            "token_start": int(s0), "token_end": int(e0),
            "n_bos": n_bos, "n_interior": n_interior, "n_final": n_final,
        })

        # SCIENTIFIC CLASS = semantic label. Keying on source category would
        # merge the balanced corpus's 17 semantic labels into one bucket.
        # Source category is retained separately, as provenance and as a
        # potential corpus/blocking variable.
        cls = str(meta["label"][b])
        n_use = n
        timer.mark("derive")
        for li in sel:
            cap = caps[cls][li]
            mixer = layers[li].mixer
            submark("derive_recurrence_bc")
            o = grabbed[("in", li)][0, pack_s:pack_e].float()
            parts = split_in_proj(o, spec)

            scal, q = scalars_from_helper(parts, mixer)

            # source-derived cumulative phase; rotary-pair axis preserved.
            # carried=None -> phase resets at every independently forwarded block.
            raw_ang = parts["angles"].reshape(n_use, -1)
            ph = phase_objects(raw_ang, scal["Delta"])
            phase_consistency = max(phase_consistency, ph["consistency_err"])

            # ---- Stage 4B2A: observable B/C objects, CORRECTED roles ----
            # B = WRITE/key side, C = READ/query side.
            B_shared = reconstruct_bc(parts["B"], mixer.B_norm.weight.float(),
                                      mixer.B_bias.float(), expand_heads=False)
            B_post = reconstruct_bc(parts["B"], mixer.B_norm.weight.float(),
                                    mixer.B_bias.float())
            C_shared = reconstruct_bc(parts["C"], mixer.C_norm.weight.float(),
                                      mixer.C_bias.float(), expand_heads=False)
            C_post = reconstruct_bc(parts["C"], mixer.C_norm.weight.float(),
                                    mixer.C_bias.float())

            # ACTUAL partial RoPE over STATE COORDINATES, using the Stage 4B1
            # wrapped cumulative phase. Rank axis untouched; every rank slot
            # receives the SAME rotation.
            B_rot_write = partial_rope_bc(B_post, ph["wrapped"])
            C_rot_read = partial_rope_bc(C_post, ph["wrapped"])

            # trap_scale weights the OFF-DIAGONAL WRITE, so it multiplies B.
            # There is no "C * trap_scale": C reads, it is not written into the
            # prefix. See B_OFFDIAG_CAVEAT for what this object omits.
            ts = q["trap_scale"].unsqueeze(-1).unsqueeze(-1)
            B_offdiag_write_weighted = B_rot_write * ts
            submark("derive_recurrence_bc")

            # gamma is NOT applied to B or C separately: it scales the
            # same-token UNROTATED B/C product AFTER the product is formed.
            # diagonal_rank_utilization stays withheld.

            stage_tensors = {
                "B_shared": B_shared.movedim(-2, -3),
                "C_shared": C_shared.movedim(-2, -3),
                "B_post": B_post, "C_post": C_post,
                "B_rot_write": B_rot_write, "C_rot_read": C_rot_read,
                "B_offdiag_write_weighted": B_offdiag_write_weighted,
            }
            P = spec.headdim
            x = parts["x"].reshape(n_use, nheads, P)
            z = parts["z"].reshape(n_use, nheads, P)
            actual_pre_out = grabbed[("pre_out", li)][0, pack_s:pack_e].float().reshape(
                n_use, nheads, P)
            resid = grabbed[("resid", li)][0, pack_s:pack_e].float()
            mlp_h = grabbed[("mlp", li)][0, pack_s:pack_e].float()
            # Geometry and token-local pathways are pointwise in token position.
            # Keep exact sequence slices here, then execute those paths ONCE on
            # the complete packed buffer when the forward batch closes.
            pending_roles[li].append({
                "row": n_done, "n": n_use, "class": cls,
                "scal": scal, "ph": ph, "winding": ph["winding"],
                "stage": stage_tensors, "x": x, "z": z,
                "gamma": q["gamma"], "actual_pre_out": actual_pre_out,
                "resid": resid, "mlp_nz": near_zero_fraction(mlp_h),
            })

            # WITHHELD: phase/concentration. phase_stats() over the RAW angle
            # slice is not the model's cumulative phase (tanh(raw)*pi*Delta,
            # inclusive cumsum + carry, mod 2pi). Stage 4B restores it.

        timer.mark("derive")

        completed = n_done + 1
        pack_complete = completed == pack_end
        if pack_complete:
            submark("derive_roles")
            for li in sel:
                recs = pending_roles[li]
                mixer = layers[li].mixer

                def cat_record(field):
                    return torch.cat([rec[field] for rec in recs], dim=0)

                def cat_stage(field):
                    return torch.cat([rec["stage"][field] for rec in recs], dim=0)

                total_t = sum(rec["n"] for rec in recs)

                # One geometry pass over the full packed token buffer. Sequence
                # identity is irrelevant for these pointwise quantities; exact
                # boundaries are restored below before any block row is written.
                submark("derive_geometry")
                stage_all = {nm: cat_stage(nm) for nm in BC_STAGES}
                align_all = {
                    "shared": bc_alignment(stage_all["B_shared"],
                                           stage_all["C_shared"]),
                    "post": bc_alignment(stage_all["B_post"],
                                         stage_all["C_post"]),
                }
                z2d = torch.zeros(total_t, nheads, device=dev)
                gsB = rank_geometry(stage_all["B_shared"], include_eigvals=False)
                gsC = rank_geometry(stage_all["C_shared"], include_eigvals=False)
                gB0 = rank_geometry(stage_all["B_post"], include_eigvals=False)
                gB1 = rank_geometry(stage_all["B_rot_write"],
                                    include_eigvals=False)
                gC0 = rank_geometry(stage_all["C_post"], include_eigvals=False)
                gC1 = rank_geometry(stage_all["C_rot_read"],
                                    include_eigvals=False)
                rope_gram_dev_B = (gB0["gram"] - gB1["gram"]).abs().amax(
                    dim=(-2, -1))
                rope_gram_dev_C = (gC0["gram"] - gC1["gram"]).abs().amax(
                    dim=(-2, -1))
                submark("derive_geometry")

                # One token-local pathway pass over the packed buffer. This is
                # algebraically identical to per-block execution: there is no
                # recurrence or cross-token reduction in token_local_paths().
                submark("derive_paths")
                x_all, z_all = cat_record("x"), cat_record("z")
                B_post_all, C_post_all = (stage_all["B_post"],
                                          stage_all["C_post"])
                gamma_all = cat_record("gamma")
                lp_ = token_local_paths(
                    x_all, z_all, B_post_all, C_post_all, gamma_all,
                    mixer.D.float(),
                    mixer.mimo_x.float() if is_mimo else None,
                    mixer.mimo_z.float() if is_mimo else None,
                    mixer.mimo_o.float() if is_mimo else None, is_mimo)
                V_rank = lp_["V_rank"]
                gate_pre, gate_factor = lp_["gate_pre"], lp_["gate_factor"]
                y_diag, y_D = lp_["y_diag"], lp_["y_D"]
                diag_collapse_r = lp_["diag_collapse_r"]
                D_collapse_r = lp_["D_collapse_r"]
                diag_collapsed, D_collapsed = (lp_["diag_collapsed"],
                                                lp_["D_collapsed"])
                actual_pre_out = cat_record("actual_pre_out")
                earlier_residual = actual_pre_out - diag_collapsed - D_collapsed
                resid_all = cat_record("resid")
                mlp_nz_all = torch.cat([
                    rec["mlp_nz"].reshape(1, 1).expand(rec["n"], nheads)
                    for rec in recs
                ], dim=0)

                def cat_scalar(field):
                    return torch.cat([rec["scal"][field] for rec in recs], dim=0)

                vals_all = {
                    "lambda": cat_scalar("lambda"),
                    "alpha": cat_scalar("alpha"),
                    "Delta": cat_scalar("Delta"),
                    "trap_scale": cat_scalar("trap_scale"),
                    "local_halflife": cat_scalar("eff_half_life"),
                    "B_diff_shared": (gsB["diff_over_total"].expand(total_t, nheads)
                                      if rank > 1 else z2d),
                    "B_diff_post": (gB0["diff_over_total"] if rank > 1 else z2d),
                    "C_diff_shared": (gsC["diff_over_total"].expand(total_t, nheads)
                                      if rank > 1 else z2d),
                    "C_diff_post": (gC0["diff_over_total"] if rank > 1 else z2d),
                    "B_participation_post": gB0["participation"],
                    "C_participation_post": gC0["participation"],
                    "bc_common_cos_post": align_all["post"]["common_cos"],
                    "bc_diff_cos_post": align_all["post"]["diff_cos"],
                    "bc_total_cos_post": align_all["post"]["total_cos"],
                    "B_rot_write_norm": stage_all["B_rot_write"].norm(
                        dim=(-2, -1)),
                    "C_rot_read_norm": stage_all["C_rot_read"].norm(
                        dim=(-2, -1)),
                    "B_write_weighted_norm": stage_all[
                        "B_offdiag_write_weighted"].norm(dim=(-2, -1)),
                    "rope_rank_gram_dev_B": rope_gram_dev_B,
                    "rope_rank_gram_dev_C": rope_gram_dev_C,
                    "value_norm": x_all.norm(dim=-1),
                    "V_rank_norm": V_rank.norm(dim=(-2, -1)),
                    "gate_pre_mean": gate_pre.mean(dim=(-2, -1)),
                    "gate_pre_std": gate_pre.std(dim=(-2, -1)),
                    "gate_factor_mean": gate_factor.mean(dim=(-2, -1)),
                    "gate_factor_std": gate_factor.std(dim=(-2, -1)),
                    "gate_factor_frac_near_zero": (
                        gate_factor.abs() < GATE_NEAR_ZERO_EPS).float().mean(
                            dim=(-2, -1)),
                    "diag_pre_gate_norm": y_diag.norm(dim=(-2, -1)),
                    "D_pre_gate_norm": y_D.norm(dim=(-2, -1)),
                    "diag_collapsed_norm": diag_collapsed.norm(dim=-1),
                    "D_collapsed_norm": D_collapsed.norm(dim=-1),
                    "actual_pre_out_norm": actual_pre_out.norm(dim=-1),
                    "earlier_collapsed_residual_norm": earlier_residual.norm(dim=-1),
                    "align_diag_D": _cos3(diag_collapsed, D_collapsed),
                    "align_diag_residual": _cos3(diag_collapsed,
                                                 earlier_residual),
                    "align_D_residual": _cos3(D_collapsed, earlier_residual),
                    "align_diag_D_valid": _cos_valid(diag_collapsed, D_collapsed),
                    "align_diag_residual_valid": _cos_valid(diag_collapsed,
                                                             earlier_residual),
                    "align_D_residual_valid": _cos_valid(D_collapsed,
                                                          earlier_residual),
                    "resid_absmean": resid_all.abs().mean(
                        -1, keepdim=True).expand(total_t, nheads),
                    "mlp_frac_near_zero": mlp_nz_all,
                }
                if set(vals_all) != set(BLOCK_QUANTITIES):
                    missing = sorted(set(BLOCK_QUANTITIES) - set(vals_all))
                    extra = sorted(set(vals_all) - set(BLOCK_QUANTITIES))
                    raise ContractError(
                        f"block quantity drift: missing={missing}, extra={extra}")
                for name, value in vals_all.items():
                    if value.shape != (total_t, nheads):
                        raise ContractError(
                            f"{name} at layer {li} has shape {tuple(value.shape)}, "
                            f"expected {(total_t, nheads)}")

                rank_all = {
                    "V_rank_norm": V_rank.norm(dim=-1),
                    "gate_pre_mean": gate_pre.mean(dim=-1),
                    "gate_pre_std": gate_pre.std(dim=-1),
                    "gate_factor_mean": gate_factor.mean(dim=-1),
                    "gate_factor_std": gate_factor.std(dim=-1),
                    "diag_pre_gate_norm": y_diag.norm(dim=-1),
                    "D_pre_gate_norm": y_D.norm(dim=-1),
                    "diag_collapsed_contrib_norm": diag_collapse_r.norm(dim=-1),
                    "D_collapsed_contrib_norm": D_collapse_r.norm(dim=-1),
                }
                collapsed_all = {f: vals_all[f] for f in BLOCK_COLLAPSED_FIELDS}

                cursor = 0
                for rec in recs:
                    end = cursor + rec["n"]
                    rec["extra"] = {k: v[cursor:end] for k, v in vals_all.items()}
                    rec["rankvals"] = {k: v[cursor:end]
                                       for k, v in rank_all.items()}
                    rec["colvals"] = {k: v[cursor:end]
                                      for k, v in collapsed_all.items()}
                    rec["align"] = {
                        stage: {k: v[cursor:end] for k, v in values.items()}
                        for stage, values in align_all.items()
                    }
                    for key in ("x", "z", "gamma", "actual_pre_out",
                                "resid", "mlp_nz"):
                        rec.pop(key)
                    cursor = end
                assert cursor == total_t
                submark("derive_paths")

                recorder.put_batch_layer(layer_pos[li], recs)
                blockrole.put_batch_layer(layer_pos[li], recs)

                # Online class/role accumulators also update once per
                # (semantic label, role, packed batch), not once per block.
                by_class = defaultdict(list)
                for rec in recs:
                    by_class[rec["class"]].append(rec)
                for label, group in by_class.items():
                    cap = caps[label][li]
                    submark("derive_geometry")
                    for nm in BC_STAGES:
                        cap.geom[nm].update(torch.cat([
                            rec["stage"][nm] for rec in group], dim=0))
                    for stage in ("shared", "post"):
                        accs = cap.align[stage]
                        for k in ("common_cos", "diff_cos", "total_cos"):
                            v = torch.cat([
                                rec["align"][stage][k] for rec in group], dim=0)
                            accs[k].update(v.T.unsqueeze(0))
                        for k in ("common_valid", "diff_valid", "total_valid"):
                            v = torch.cat([
                                rec["align"][stage][k] for rec in group], dim=0)
                            accs[k].update(v.float().T.unsqueeze(0))
                        v = torch.cat([
                            rec["align"][stage]["per_rank_cos"]
                            for rec in group], dim=0)
                        accs["per_rank_cos"].update(v.permute(1, 2, 0))
                    submark("derive_geometry")

                    submark("derive_accumulators")
                    for name in BLOCK_QUANTITIES:
                        v = torch.cat([rec["extra"][name]
                                       for rec in group], dim=0)
                        v2 = v.T.unsqueeze(0)
                        cap.moments[name].update(v2)
                        if name in cap.hists:
                            cap.hists[name].update(v2)
                    submark("derive_accumulators")

                    for rname in POSITION_ROLES:
                        pieces = []
                        for rec in group:
                            lo, hi = role_slices(rec["n"])[rname]
                            if hi > lo:
                                pieces.append((rec, lo, hi))
                        if not pieces:
                            continue

                        def role_cat(section, field):
                            return torch.cat([
                                rec[section][field][lo:hi]
                                for rec, lo, hi in pieces], dim=0)

                        scal_cat = {
                            f: role_cat("scal", f)
                            for f in pieces[0][0]["scal"]
                        }
                        ph_cat = {
                            f: role_cat("ph", f)
                            for f in ("wrapped", "unwrapped", "increment")
                        }
                        extra_cat = {
                            f: role_cat("extra", f) for f in BLOCK_ROLE_BC
                        }
                        n_role = sum(hi - lo for _rec, lo, hi in pieces)
                        roles[label][li][rname].update(
                            scal_cat, ph_cat, 0, n_role, extra=extra_cat)
            submark("derive_roles")
            pack_idx += 1
            if pack_idx < len(forward_packs):
                pack_start = pack_end
                pack_end += len(forward_packs[pack_idx])

        first_pack = len(forward_packs[0])
        if completed == first_pack or completed % 200 == 0:
            phase_s = timer.drain()
            phase_names = ["forward", "derive", "d2h"]
            if args.profile_subphases:
                phase_names.extend(("derive_recurrence_bc", "derive_geometry",
                                    "derive_paths", "derive_accumulators",
                                    "derive_roles"))
            phase_text = " ".join(
                f"{k}={phase_s.get(k, 0.0):.1f}s"
                for k in phase_names)
            print(f"  block {completed}/{len(block_indices)}  "
                  f"{seen_tokens:,} tokens  {time.time() - t_start:.0f}s  "
                  f"{phase_text}  d2h={recorder.d2h + blockrole.d2h}",
                  flush=True)

    phase_times = timer.drain()

    for h in handles:
        h.remove()

    # ---- coverage decided BEFORE anything is written ----
    expected_blocks = plan["blocks_expected"]
    exp_valid = plan["valid_tokens_expected"]
    exp_content = plan["content_tokens_expected"]
    processed = len(recorder.meta)
    coverage = {
        "blocks_in_artifact": int(n_total),
        "blocks_available_after_budget": int(n_avail),
        "blocks_expected": int(expected_blocks),
        "blocks_processed": int(processed),
        "valid_tokens_expected": int(exp_valid),
        "valid_tokens_processed": int(seen_tokens),
        "content_tokens_expected": int(exp_content),
        "content_tokens_processed": int(content_seen),
        "metadata_content_tokens": int(plan["metadata_content_tokens"]),
        "metadata_consistent": bool(plan["metadata_consistent"]),
        "declared_limit": plan["declared_limit"],
        "expectation_source": "offsets (independent of n_content_tokens metadata)",
        "processed_source": "len(seq) and len(seq)-1 per forwarded block",
        "block_recorder_d2h_copies": int(recorder.d2h),
        "block_role_d2h_copies": int(blockrole.d2h),
        "denominator_failures": int(recorder.denominator_failures),
    }
    coverage["complete"] = (processed == expected_blocks
                            and seen_tokens == exp_valid
                            and content_seen == exp_content
                            and recorder.denominator_failures == 0
                            and recorder.d2h == forward_batches * len(sel)
                            and blockrole.d2h == forward_batches * len(sel))

    print(f"\ncoverage: {processed}/{expected_blocks} blocks, "
          f"{seen_tokens}/{exp_valid} valid tokens, "
          f"{content_seen}/{exp_content} content tokens")
    print(f"positions bos={pos_counts['bos']} interior={pos_counts['interior']} "
          f"final={pos_counts['final']}")

    if not coverage["complete"]:
        # NOTHING is published. A partial capture at the requested path would be
        # indistinguishable from a complete one to every downstream consumer.
        print("[STOP] incomplete coverage -- refusing to publish "
              f"{args.out}")
        for k in ("blocks", "valid_tokens", "content_tokens"):
            print(f"    {k}: expected {coverage[k + '_expected']} "
                  f"processed {coverage[k + '_processed']}")
        sys.exit(1)

    # ---- build the payload ----
    payload = {}
    for cls, per_layer in caps.items():
        for li, cap in per_layer.items():
            for k, v in cap.finish().items():
                payload[f"semantic_label={cls}|L{li}|{k}"] = np.asarray(v)
    for q, arr in recorder.finish().items():
        payload[f"blocks|{q}"] = arr
    payload["block_semantic_label"] = np.array(
        [m["label"] for m in recorder.meta], dtype="<U128")
    payload["block_source_category"] = np.array(
        [m["category"] for m in recorder.meta], dtype="<U128")
    payload["block_source_file"] = np.array(
        [m["source_file"] for m in recorder.meta], dtype="<U128")
    for key in ("block", "doc_id", "prompt_id", "source_row", "chunk_index",
                "n_chunks", "valid_len", "n_content_tokens", "n_used",
                "token_start", "token_end", "n_bos", "n_interior", "n_final"):
        payload[f"block_{key}"] = np.array([m[key] for m in recorder.meta],
                                           dtype=np.int64)
    payload["layers"] = np.array(sel)
    payload |= blockrole.payload()

    # Stage 4B1: role-stratified accumulators, with observation counts for every
    # (role, metric, layer, semantic label)
    obs_counts = {}
    for cls, per_layer in roles.items():
        for li, per_role in per_layer.items():
            for rname, rc in per_role.items():
                for k, v in rc.finish().items():
                    payload[f"role|{cls}|L{li}|{rname}|{k}"] = np.asarray(v)
                obs_counts[f"{cls}|L{li}|{rname}"] = int(rc.n_obs)

    manifest = assert_runtime({"model": repo, "stream": args.stream,
                               "requires_kernel": True}, strict=False)
    manifest |= {
        "checkpoint": ck.provenance(),
        "gpu_probe_gate": {
            "report_path": args.gpu_probe_report,
            "report_sha256": sha256_file(args.gpu_probe_report),
            "verdicts": probe.get("verdicts"),
            "archive_status": (probe.get("archive") or {}).get("status"),
            "mixer_output_parity_validated": True,
            "kernel_state_parity_validated": False,
            "boundary_prohibited_claims":
                probe.get("boundary_prohibited_claims", []),
        },
        "layers": sel, "nheads": nheads, "rank": rank, "is_mimo": is_mimo,
        "token_contract": {
            "path": args.token_contract,
            "schema_version": tc_manifest.get("schema_version"),
            "artifact_sha256_expected": exp_dig,
            "artifact_sha256_computed": got_dig,
            "artifact_sha256_verified": exp_dig == got_dig,
            "capture_must_forward_blocks_independently":
                tc_manifest.get("capture_must_forward_blocks_independently"),
        },
        "selected_budget_record": budget_record,
        "forward_policy": "independent blocks packed into one varlen token "
                          "buffer; cu_seqlens resets convolution, phase and "
                          "recurrence at every exact block boundary; no padding",
        "forward_batch_size": int(args.forward_batch_size),
        "forward_max_tokens": int(args.forward_max_tokens),
        "forward_batches": int(forward_batches),
        "forward_pack_realized": {
            "min_blocks": int(min(map(len, forward_packs))),
            "max_blocks": int(max(map(len, forward_packs))),
            "min_tokens": int(min(pack_token_counts)),
            "max_tokens": int(max(pack_token_counts)),
        },
        "coverage": coverage,
        "position_counts": pos_counts,
        "position_policy": (
            "complete unpadded block forwarded INCLUDING its final token. The "
            "official kernel emits it, but it is BOUNDARY-CONDITIONED and is not "
            "assumed exchangeable with interior positions."),
        "evidence_status_definitions": EVIDENCE_STATUS,
        "evidence_status_overall": ["source_derived", "fixture_tested"],
        "gpu_parity_validated": False,
        "histogram_bin_reuse": HIST_BIN_REUSE,
        "block_role_summary": {
            "fields": list(BLOCK_ROLE_FIELDS),
            "winding_fields": list(BLOCK_WINDING_FIELDS),
            "rank_fields": list(BLOCK_RANK_FIELDS),
            "collapsed_fields": list(BLOCK_COLLAPSED_FIELDS),
            "shape": "role/collapsed: (block, layer, role, head); rank: "
                     "(block, layer, role, head, rank); winding: "
                     "(block, layer, head)",
            "dtype": str(np.dtype(BLOCK_SUMMARY_DTYPE)),
            "field_dtype_overrides": {
                f: str(np.dtype(dt)) for f, dt in BLOCK_ROLE_FIELD_DTYPES.items()
            },
            "counts_shape": "(block, layer, role) -- identical across heads",
            "projected_bytes": est,
        },
        "capture_performance": {
            "phase_seconds": {k: float(v) for k, v in phase_times.items()},
            "wall_seconds_before_serialization": float(time.time() - t_start),
            "d2h_transfer_policy": "two batched copies per selected layer and "
                                   "packed forward: one pooled BlockRecorder "
                                   "copy and one role/rank/collapsed copy",
            "block_recorder_d2h_copies": int(recorder.d2h),
            "block_role_d2h_copies": int(blockrole.d2h),
            "legacy_sync_sites_upper_bound_per_full_role_block":
                int(old_syncs_per_block),
            "timing_policy": "CUDA events accumulated through each packed "
                             "forward and synchronized only at reporting heartbeats",
            "rank_geometry_policy": "participation from trace(G)^2 / ||G||_F^2; "
                                    "eigenvalues accumulated once per packed class "
                                    "group in float32",
        },
        "position_stratification_available": True,
        "position_roles": {
            "bos": "token index 0 of every block",
            "interior": "token indices 1..n-2 (empty when valid_len <= 2)",
            "final": "token index n-1 when valid_len > 1; a 1-token block is BOS "
                     "ONLY and contributes no final observation",
        },
        "role_observation_counts": obs_counts,
        "recurrence_scalars": {
            "source": "mamba3_core.recurrence_quantities (no second implementation)",
            "Delta": "softplus(dd_dt + dt_bias)",
            "A": "-heavy_tail_activation(dd_A), clamped to <= -A_floor",
            "log_alpha": "A * Delta  (PRIMARY, numerically stable decay quantity)",
            "alpha": "exp(log_alpha)  (derived diagnostic)",
            "lambda": "sigmoid(trap)",
            "gamma": "lambda * Delta",
            "shifted_gamma": "Delta[t+1] * (1 - lambda[t+1]); 0 at the final "
                             "position under the source-derived prefill boundary "
                             "convention",
            "trap_scale": "gamma + shifted_gamma",
            "eff_half_life": "ln2 / -log_alpha; DATA-CONDITIONED effective "
                             "retention in recurrence steps, NOT the zero-input "
                             "prior; censored where -log_alpha <= "
                             f"{HALFLIFE_CENSOR_EPS} and never summarized by a "
                             "mean alone",
            "histogram_gap": "accumulators.DEFAULT_EDGES defines no bins for "
                             "log_alpha, gamma or shifted_gamma; those carry "
                             "online moments only (editing accumulators.py is "
                             "out of scope this stage)",
        },
        "phase": {
            "increment": "tanh(raw_angle) * pi * Delta",
            "unwrapped": "inclusive cumsum over tokens",
            "wrapped": "unwrapped mod 2*pi, floor convention",
            "winding": "sum(increment) / 2*pi, per block",
            "reset_policy": "phase starts at ZERO for every independently "
                            "forwarded block (carried=None); carrying across "
                            "blocks would reintroduce cross-block state",
            "rotary_pairs": int(spec.n_rope_angles),
            "rotary_pair_axis": "PRESERVED through accumulation; never collapsed",
            "indexing": "pair k rotates d_state index k against k + d_state//2",
            "helper_gap": PHASE_HELPER_GAP,
            "implementation_status": (
                "CONSOLIDATED. phase_objects is a thin adapter over "
                "reference_recurrence.phase_details, which supplies increment, "
                "unwrapped phase, wrapped phase and carry. No local phase formula "
                "remains in this capture."),
            "evidence_status": ["source_derived", "fixture_tested"],
            "consistency_max_err": float(phase_consistency),
            "tanh_parity_risk": (
                "this file uses exact torch.tanh; the kernel uses tanh_approx "
                "(angle_dt.py). They are NOT proven equal -- a candidate "
                "approximation pending GPU tensor parity."),
            "superseded_note": PHASE_SUPERSEDED,
        },
        "per_metric_excluded_positions": per_metric_excluded,
        "withheld_invalid_metrics": list(WITHHELD_METRICS),
        "withheld_reason": WITHHELD_REASON if WITHHELD_METRICS else None,
        "stage_4b2a": {
            "roles": {
                "B": "KEY / WRITE side -- accumulated into the recurrent prefix",
                "C": "QUERY / READ side -- reads the accumulated state at a "
                     "LATER position",
                "evidence": "mamba3.py inference calls the rotary with q=C, k=B; "
                            "the recurrence injects B_t into the state and reads "
                            "it later with C_t; the off-diagonal prefix is "
                            "key/value accumulation, therefore write-side",
                "retracted_claim": "there is no object 'C * trap_scale' written "
                                   "into the prefix; that claim is withdrawn",
            },
            "stages": {
                "B_shared / C_shared": "pre-bias, shared across heads (ngroups=1)",
                "B_post / C_post": "post per-head bias; first stage where heads differ",
                "B_rot_write / C_rot_read": "ACTUAL partial RoPE over STATE "
                                            "COORDINATES using the Stage 4B1 "
                                            "wrapped cumulative phase",
                "B_offdiag_write_weighted": "B_rot_write * trap_scale",
            },
            "geometry_per_stage": [
                "rank Gram", "Gram eigenvalues", "participation ratio",
                "total norm", "rank-common norm", "rank-differential norm",
                "differential/total unsquared ratio",
                "differential/total squared-energy ratio",
            ],
            "geometry_caveat": RANK_GAUGE_NOTE,
            "rank_gauge_behaviour": RANK_GAUGE_BEHAVIOUR,
            "alignment": {
                "stages": ["shared", "post"],
                "quantities": ["common_cos", "diff_cos", "total_cos",
                               "per_rank_cos (signed, basis-dependent)"],
                "validity": "count masks recorded for zero-norm cases",
                "naming": "ALIGNMENT, never 'synergy'",
            },
            "rope_invariance_control": {
                "check": "rank_gram(B_post) == rank_gram(B_rot_write) and "
                         "rank_gram(C_post) == rank_gram(C_rot_read) per token/head",
                "why": "the same orthogonal state-coordinate rotation is applied "
                       "to every rank slot, so the rank Gram is unchanged",
                "recorded_as": ["rope_rank_gram_dev_B", "rope_rank_gram_dev_C"],
                "removed": "random QR on the rank axis is GONE from the "
                           "production loop -- stochastic, costly and unrelated "
                           "to the model's data-dependent RoPE. It survives only "
                           "as a fixture-level gauge probe.",
            },
            "siso_controls": "at rank 1: participation ratio == 1, differential "
                             "norm == 0, both differential ratios == 0",
            "b_offdiag_caveat": B_OFFDIAG_CAVEAT,
            "withheld": ["gram_injection", "bc_diff_injection",
                         "diagonal_rank_utilization", "full_recurrent_utilization"],
            "evidence_status": ["source_derived", "fixture_tested"],
            "gpu_parity_validated": False,
        },
        "not_captured": [
            "SSM state h_t (inside kernel)",
            "per-rank output y_r before mimo_o collapse (inside kernel)",
        ],
        "near_zero_eps": NEAR_ZERO_EPS,
    }

    # atomic: temp file then os.replace, so interruption cannot leave a partial
    # artifact at the final path
    serialize_t0 = time.perf_counter()
    atomic_savez(args.out, **payload)
    manifest["capture_performance"]["serialization_npz_seconds"] = float(
        time.perf_counter() - serialize_t0)
    atomic_write_json(args.out.replace(".npz", ".manifest.json"), manifest)
    print(f"\nwrote {args.out}  ({processed} blocks, {seen_tokens:,} tokens)")


if __name__ == "__main__":
    main()
