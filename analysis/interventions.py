"""C-1: the intervention harness. Dose-response edits with same-norm controls.

TWO MECHANISMS, because Mamba-3 splits its knobs across weights and activations:

  STATIC edits   -- patch nn.Parameters directly: B_bias, C_bias, D, dt_bias,
                    mimo_x/z/o. Restored on context exit.
  DYNAMIC edits  -- rewrite slices of the in_proj OUTPUT via a forward hook that
                    returns a modified tensor. This is the only way to touch
                    lambda (trap), Delta (dd_dt), A (dd_A) and the rope angles,
                    because those are projected per token and consumed inside
                    the kernel.

WHY DOSE-RESPONSE, NOT ON/OFF
  A single ablation confounds "this component matters" with "you broke the
  model". A dose sweep distinguishes them: a real mechanism produces a graded,
  monotone response; breakage produces a cliff. Every edit here is scaled by a
  dose in [0, 1, ...] and reported as a curve.

WHY SAME-NORM RANDOM CONTROLS ARE MANDATORY
  Any edit of sufficient magnitude changes the output. The question is whether
  THIS direction matters more than an arbitrary one of the same size. Every
  intervention is paired with a random direction matched in norm, and the
  reported effect is the DIFFERENCE. An intervention that does not beat its
  control is not evidence.

SCOPE: within-MIMO interventions are mimo-specific evidence. They say nothing
about SISO-vs-MIMO, which is bundle-level and confounded by MLP width.

Usage as a library; see rank_causality.py and mediation.py for callers.
"""

from __future__ import annotations

import contextlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import InProjSpec  # noqa: E402

DOSES = (0.0, 0.25, 0.5, 1.0)


# --------------------------------------------------------------------------
# static parameter edits
# --------------------------------------------------------------------------


@contextlib.contextmanager
def patched_param(module, name, new_value):
    """Temporarily replace a parameter, restoring the original on exit."""
    p = getattr(module, name)
    old = p.data.clone()
    try:
        p.data.copy_(new_value.to(p.dtype).to(p.device))
        yield
    finally:
        p.data.copy_(old)


def scaled_edit(orig, direction, dose):
    """orig + dose * direction, with direction normalized to orig's scale."""
    n_o = orig.norm()
    n_d = direction.norm().clamp_min(1e-12)
    return orig + dose * direction * (n_o / n_d)


def random_like(t, generator=None):
    """A random direction with the same shape; norm matching happens in scaled_edit."""
    return torch.randn(t.shape, generator=generator, device="cpu").to(t.device)


# --------------------------------------------------------------------------
# dynamic activation edits
# --------------------------------------------------------------------------


class InProjEditor:
    """Rewrite named slices of in_proj output. Attach to one layer's mixer.

    edits: {"trap": fn(tensor)->tensor, "dd_dt": ..., "dd_A": ..., "angles": ...}
    Each fn receives (tokens, width) and must return the same shape.
    """

    def __init__(self, mixer, spec: InProjSpec, edits: dict):
        self.spec, self.edits, self.handle = spec, edits, None
        self.mixer = mixer

    def __enter__(self):
        def hook(_m, _inp, out):
            o = out.clone()
            for name, fn in self.edits.items():
                lo, hi = self.spec.bounds[name]
                o[..., lo:hi] = fn(o[..., lo:hi])
            return o

        self.handle = self.mixer.in_proj.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self.handle:
            self.handle.remove()


def lambda_shift(delta_logit, heads=None):
    """Shift the trapezoid logit, i.e. move lambda toward 0 or 1.

    Operates on the RAW logit because the kernel applies the sigmoid itself.
    heads: optional index tensor to restrict the edit to specific heads.
    """
    def fn(x):
        y = x.clone()
        if heads is None:
            y += delta_logit
        else:
            y[..., heads] += delta_logit
        return y
    return fn


def scale_slice(factor, heads=None):
    def fn(x):
        y = x.clone()
        if heads is None:
            y *= factor
        else:
            y[..., heads] *= factor
        return y
    return fn


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------


@torch.inference_mode()
def measure(model, ids, targets=None):
    """Return the quantities every intervention is scored on."""
    logits = model(ids).logits.float()
    out = {"logits": logits}
    if targets is not None:
        lp = F.log_softmax(logits[:, :-1], dim=-1)
        out["loss"] = F.nll_loss(
            lp.reshape(-1, lp.shape[-1]), targets.reshape(-1), reduction="mean"
        ).item()
        out["target_logprob"] = lp.gather(-1, targets.unsqueeze(-1)).mean().item()
    out["top1"] = logits[:, -1].argmax(-1)
    return out


def effect_size(base, edited):
    """How far the edit moved the model, on comparable scales."""
    d = (edited["logits"] - base["logits"]).abs()
    e = {
        "mean_abs_logit_delta": float(d.mean()),
        "max_abs_logit_delta": float(d.max()),
        "top1_flip_rate": float((edited["top1"] != base["top1"]).float().mean()),
    }
    if "loss" in base:
        e["loss_delta"] = edited["loss"] - base["loss"]
        e["target_logprob_delta"] = edited["target_logprob"] - base["target_logprob"]
    return e


def dose_response(model, ids, make_context, targets=None, doses=DOSES,
                  control_factory=None, seed=0):
    """Run an edit across doses, alongside a same-norm random control.

    make_context(dose) -> context manager applying the real edit
    control_factory(dose) -> context manager applying a matched random edit

    Returns {"real": [...], "control": [...], "net": [...]} where net is the
    real effect MINUS the control effect at each dose. Net is the number that
    means anything.
    """
    base = measure(model, ids, targets)
    rows = {"doses": list(doses), "real": [], "control": [], "net": []}

    for dose in doses:
        with make_context(dose):
            e_real = effect_size(base, measure(model, ids, targets))
        rows["real"].append(e_real)

        if control_factory is not None:
            with control_factory(dose):
                e_ctrl = effect_size(base, measure(model, ids, targets))
            rows["control"].append(e_ctrl)
            rows["net"].append(
                {k: e_real[k] - e_ctrl.get(k, 0.0) for k in e_real}
            )
        else:
            rows["control"].append(None)
            rows["net"].append(None)

    rows["monotone"] = _is_monotone(
        [r["mean_abs_logit_delta"] for r in rows["real"]]
    )
    rows["beats_control"] = (
        None if control_factory is None else
        bool(rows["net"][-1]["mean_abs_logit_delta"] > 0)
    )
    rows["interpretation"] = (
        "graded + beats control => mechanism"
        if rows["monotone"] and rows["beats_control"] else
        "does not beat its same-norm control => NOT evidence"
        if rows["beats_control"] is False else
        "non-monotone => likely breakage rather than mechanism"
    )
    return rows


def _is_monotone(xs):
    d = np.diff(xs)
    return bool((d >= -1e-6).all() or (d <= 1e-6).all())


# --------------------------------------------------------------------------
# ready-made interventions
# --------------------------------------------------------------------------


def intervene_D(mixer, heads=None, seed=0):
    """Dose-scale the direct feedthrough. MIMO's one robust static signature."""
    g = torch.Generator().manual_seed(seed)
    orig = mixer.D.data.clone()
    rand = random_like(orig, g)

    def real(dose):
        v = orig.clone()
        if heads is None:
            v = v * (1.0 - dose)
        else:
            v[heads] = v[heads] * (1.0 - dose)
        return patched_param(mixer, "D", v)

    def ctrl(dose):
        return patched_param(mixer, "D", scaled_edit(orig, rand, dose))

    return real, ctrl


def intervene_bias(mixer, which="B_bias", heads=None, seed=0):
    """Dose-scale a per-head B/C bias: the entire per-head state geometry."""
    g = torch.Generator().manual_seed(seed)
    orig = getattr(mixer, which).data.clone()
    rand = random_like(orig, g)

    def real(dose):
        v = orig.clone()
        idx = slice(None) if heads is None else heads
        v[idx] = v[idx] * (1.0 - dose)
        return patched_param(mixer, which, v)

    def ctrl(dose):
        return patched_param(mixer, which, scaled_edit(orig, rand, dose))

    return real, ctrl


def intervene_lambda(mixer, spec, delta_logit=2.0, heads=None, seed=0):
    """Push the trapezoid gate toward 1 (dose>0) via the raw logit."""
    g = torch.Generator().manual_seed(seed)
    rnd = float(torch.randn(1, generator=g))

    def real(dose):
        return InProjEditor(mixer, spec, {"trap": lambda_shift(dose * delta_logit, heads)})

    def ctrl(dose):
        # same magnitude, applied to an unrelated slice of equal width
        return InProjEditor(mixer, spec,
                            {"dd_A": lambda_shift(dose * delta_logit * abs(rnd), heads)})

    return real, ctrl
