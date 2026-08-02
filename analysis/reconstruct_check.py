"""B0-4: prove the B/C reconstruction is correct, on CPU, against independent math.

`reconstruct_bc()` is load-bearing for every per-head Stage B claim, because with
ngroups == 1 the per-head structure exists ONLY inside the kernel. If this
function is wrong, every per-head number in Stage B is wrong and nothing
downstream would notice.

Stage 2C migration: checkpoint resolution and shape derivation now come from
mamba3_core (resolve_checkpoint, InProjSpec.from_state_dict). THE
RECONSTRUCTION MATHEMATICS IS UNCHANGED -- identical checks, identical seeds
(0, 1, 2), identical tolerance (1e-5). Only how weights and shapes are obtained
changed, plus reporting and exit status.

PIPELINE STAGES whose invariants this file preserves:
  raw projected B/C     the in_proj slice, shape (T, rank, ngroups, d_state)
  RMS-normalized B/C    rms_norm over d_state; still (T, rank, ngroups, d_state),
                        i.e. NO head axis -- with ngroups == 1 this tensor is
                        SHARED by every head
  per-head post-bias    (T, nheads, rank, d_state) after the additive per-head
                        bias applied inside the kernel; this is the FIRST stage
                        at which heads differ at all

CHECKS, in increasing strength:
  1. hand-computed tiny case, arithmetic written out in the docstring
  2. independent RMSNorm formulation (different op order) on real norm weights
  3. the ngroups==1 sharing identity: per-head differences MUST equal bias
     differences exactly, since the normalized part is common
  4. order sensitivity: norm-then-bias != bias-then-norm, demonstrating that the
     transcribed pipeline order is not cosmetic

Exit status is 0 only if every check passes on every requested model. A missing
tensor, shape disagreement, arm disagreement, incomplete provenance, or failed
reconstruction check exits nonzero.

Resolution is LOCAL-ONLY; this never downloads.

Run:  python analysis/reconstruct_check.py
      python analysis/reconstruct_check.py --models mimo-187m --layer 3
"""

import argparse
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import (  # noqa: E402
    CheckpointResolveError, InProjSpec, ShapeContractError, reconstruct_bc,
    resolve_checkpoint, rms_norm,
)

TOL = 1e-5
STOP, OK = "STOP", "OK"
DEFAULT_MODELS = "mimo-187m,mimo-1.5b,siso-1.5b"


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


def load_layer(model, layer, revision=None):
    """Resolve, load config + weights through ONE ResolvedCheckpoint, derive spec."""
    ck = resolve_checkpoint(model, revision=revision, local_only=True)

    prov = ck.provenance()
    if not prov.get("path") or not prov.get("weights_file"):
        raise CheckpointResolveError(f"{model}: incomplete provenance {prov}")
    if prov.get("resolved_commit") is None and not ck.from_local_dir:
        raise CheckpointResolveError(
            f"{model}: resolved_commit is None for a hub-resolved snapshot")

    cfg = ck.load_config()
    if layer < 0 or layer >= cfg["n_layer"]:
        raise ShapeContractError(
            f"{model}: layer {layer} out of range (n_layer={cfg['n_layer']})")

    sd = ck.load_state_dict()
    spec = InProjSpec.from_state_dict(sd, cfg, layer=layer)
    arm_is_mimo = resolve_arm(spec, cfg.get("ssm_cfg"), f"{model} L{layer}")

    mx = f"backbone.layers.{layer}.mixer."
    need = ("B_bias", "C_bias", "B_norm.weight", "C_norm.weight", "in_proj.weight")
    missing = [n for n in need if (mx + n) not in sd]
    if missing:
        raise ShapeContractError(f"{model} L{layer}: missing tensors {missing}")

    p = {
        "B_bias": sd[mx + "B_bias"].float(),
        "C_bias": sd[mx + "C_bias"].float(),
        "B_norm": sd[mx + "B_norm.weight"].float(),
        "C_norm": sd[mx + "C_norm.weight"].float(),
    }
    # realized shapes must agree with the derived spec
    exp = (spec.nheads, spec.mimo_rank, spec.d_state)
    for nm in ("B_bias", "C_bias"):
        if tuple(p[nm].shape) != exp:
            raise ShapeContractError(
                f"{model} L{layer} {nm}: expected {exp}, got {tuple(p[nm].shape)}")
    for nm in ("B_norm", "C_norm"):
        if tuple(p[nm].shape) != (spec.d_state,):
            raise ShapeContractError(
                f"{model} L{layer} {nm}: expected {(spec.d_state,)}, "
                f"got {tuple(p[nm].shape)}")
    del sd
    return ck, cfg, spec, arm_is_mimo, p


def stage_shapes(p, spec, T=4):
    """Record the exact shape at each reconstruction stage. No new mathematics.

    Also states the ngroups sharing fact numerically: the pre-bias tensor has no
    head axis at all, so every head necessarily sees the same values until the
    per-head bias is added.
    """
    torch.manual_seed(7)
    raw = torch.randn(T, spec.mimo_rank, spec.ngroups, spec.d_state)
    normed = reconstruct_bc(raw, p["B_norm"], p["B_bias"], expand_heads=False)
    post = reconstruct_bc(raw, p["B_norm"], p["B_bias"])
    return {
        "raw_projected": list(raw.shape),
        "rms_normalized_pre_bias": list(normed.shape),
        "per_head_post_bias": list(post.shape),
        "pre_bias_has_head_axis": normed.dim() == 4 and normed.shape[1] == spec.nheads,
        "heads_differ_only_after_bias": True,  # asserted numerically by check 3
        "in_proj_slice_widths": {k: spec.sizes[k] for k in ("B", "C")},
    }


# --------------------------------------------------------------------------
# checks -- mathematics unchanged from the pre-migration version
# --------------------------------------------------------------------------


def check_hand_computed():
    """Tiny case with the arithmetic written out.

    x = [1, 2, 3, 4], weight = ones, eps = 0
      mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
      rms       = sqrt(7.5)            = 2.7386127875258306
      normed    = x / 2.7386127875258306
                = [0.36514837, 0.73029674, 1.09544512, 1.46059349]
    then add a per-head bias of 10 -> every entry shifts by exactly 10.
    """
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).reshape(1, 1, 1, 4)  # (T=1, r=1, g=1, n=4)
    w = torch.ones(4)
    expect_norm = torch.tensor([0.36514837, 0.73029674, 1.09544512, 1.46059349])

    got = rms_norm(x, w, eps=0.0).reshape(-1)
    err = (got - expect_norm).abs().max().item()
    if err >= 1e-6:
        return False, {"hand_rmsnorm_err": err}

    bias = torch.full((2, 1, 4), 10.0)  # 2 heads
    eff = reconstruct_bc(x, w, bias, eps=0.0)
    if tuple(eff.shape) != (1, 2, 1, 4):
        return False, {"hand_shape": list(eff.shape)}
    err2 = (eff[0, 0, 0] - (expect_norm + 10.0)).abs().max().item()
    if err2 >= 1e-6:
        return False, {"hand_bias_err": err2}
    return True, {"hand_rmsnorm_err": err, "hand_bias_err": err2}


def check_independent_formulation(p, rank, d_state, nheads):
    """Same result from a differently-ordered computation on real norm weights."""
    torch.manual_seed(0)
    raw = torch.randn(9, rank, 1, d_state)

    got = reconstruct_bc(raw, p["B_norm"], p["B_bias"])

    # independent path: explicit sum-of-squares, explicit sqrt, explicit divide,
    # then expand and add. Deliberately not the fused rsqrt used in core.
    ss = (raw.double() ** 2).sum(dim=-1, keepdim=True) / d_state
    normed = raw.double() / torch.sqrt(ss + 1e-5) * p["B_norm"].double()
    ref = normed.movedim(-2, -3).expand(9, nheads, rank, d_state) + p["B_bias"].double()

    err = (got.double() - ref).abs().max().item()
    return err < TOL, {"independent_formulation_err": err}


def check_sharing_identity(p, rank, d_state, nheads):
    """With ngroups==1 the normalized part is common to all heads.

    Therefore eff[h1] - eff[h2] == bias[h1] - bias[h2] EXACTLY, for any input.
    This is the numerical statement of why hooking in_proj and expanding heads
    measures nothing: the shared part cancels.
    """
    torch.manual_seed(1)
    raw = torch.randn(13, rank, 1, d_state) * 5.0
    eff = reconstruct_bc(raw, p["B_norm"], p["B_bias"])

    worst = 0.0
    for h1, h2 in ((0, 1), (0, nheads - 1), (nheads // 2, nheads - 1)):
        d_eff = eff[:, h1] - eff[:, h2]
        d_bias = (p["B_bias"][h1] - p["B_bias"][h2]).expand_as(d_eff)
        worst = max(worst, (d_eff - d_bias).abs().max().item())
    return worst < TOL, {"sharing_identity_err": worst}


def check_order_sensitivity(p, rank, d_state, nheads):
    """norm-then-bias must differ from bias-then-norm, or the order is untested."""
    torch.manual_seed(2)
    raw = torch.randn(5, rank, 1, d_state)

    correct = reconstruct_bc(raw, p["B_norm"], p["B_bias"])

    shared = raw.movedim(-2, -3).expand(5, nheads, rank, d_state)
    wrong = rms_norm(shared + p["B_bias"], p["B_norm"])

    gap = (correct - wrong).abs().max().item()
    rel = gap / correct.abs().max().item()
    return rel > 0.01, {"order_sensitivity_rel_gap": rel}


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=DEFAULT_MODELS)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--revision", default=None,
                    help="exact checkpoint revision; resolution stays local-only")
    ap.add_argument("--out", default="reconstruct_check_report.json")
    a = ap.parse_args()

    report = {"tol": TOL, "layer": a.layer, "models": {}, "verdict": OK}

    print("hand-computed:")
    ok, det = check_hand_computed()
    report["hand_computed"] = det | {"ok": ok}
    print(f"  1. hand-computed RMSNorm + additive bias   "
          f"{'OK' if ok else 'FAILED'}  {det}")
    if not ok:
        report["verdict"] = STOP

    for name in a.models.split(","):
        entry = {}
        try:
            ck, cfg, spec, is_mimo, p = load_layer(name, a.layer, a.revision)
        except (CheckpointResolveError, ShapeContractError) as e:
            print(f"\n{name}: {STOP} {type(e).__name__}: {e}")
            report["models"][name] = {"verdict": STOP, "error": str(e)}
            report["verdict"] = STOP
            continue

        entry["provenance"] = ck.provenance()
        entry |= {
            "layer": a.layer,
            "arm": "MIMO" if is_mimo else "SISO",
            "mimo_rank": spec.mimo_rank,
            "ngroups": spec.ngroups,
            "nheads": spec.nheads,
            "d_state": spec.d_state,
            "headdim": spec.headdim,
            "shapes": stage_shapes(p, spec),
        }

        print(f"\n{name}  L{a.layer}  arm={entry['arm']} "
              f"nheads={spec.nheads} rank={spec.mimo_rank} "
              f"ngroups={spec.ngroups} d_state={spec.d_state}")
        print(f"  stages: raw {entry['shapes']['raw_projected']} -> "
              f"normed {entry['shapes']['rms_normalized_pre_bias']} (no head axis) "
              f"-> post-bias {entry['shapes']['per_head_post_bias']}")

        results = {}
        for label, fn in (
            ("2. independent RMSNorm formulation", check_independent_formulation),
            ("3. ngroups=1 sharing identity", check_sharing_identity),
            ("4. order matters (norm->bias vs bias->norm)", check_order_sensitivity),
        ):
            passed, det = fn(p, spec.mimo_rank, spec.d_state, spec.nheads)
            results |= det
            k, v = next(iter(det.items()))
            print(f"  {label:44s} {k}={v:.3e}  {'OK' if passed else 'FAILED'}")
            if not passed:
                report["verdict"] = STOP
                entry["verdict"] = STOP
        entry["checks"] = results
        entry.setdefault("verdict", OK)
        report["models"][name] = entry

    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    print(f"\nverdict: {report['verdict']}")
    if report["verdict"] == OK:
        print("reconstruction verified against independent math on real weights")
    print(f"wrote {a.out}")
    return 0 if report["verdict"] == OK else 1


if __name__ == "__main__":
    sys.exit(main())
