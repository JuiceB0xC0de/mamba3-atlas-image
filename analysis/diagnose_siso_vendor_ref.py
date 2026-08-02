"""Temporary diagnostic: compare one SISO mixer case to the vendor reference.

This is not a gate and writes no artifact.  It feeds the exact tensors produced
by the live mixer preprocessing into the upstream PyTorch SISO reference, then
compares both that result and our oracle with the official kernel output.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from full_model_smoke import extract_layer_params, parity_metrics  # noqa: E402
from mamba3_core import InProjSpec, heavy_tail_activation, resolve_checkpoint  # noqa: E402
from reference_recurrence import reference_block_forward  # noqa: E402


def load_vendor_reference():
    path = ROOT / "upstream/tests/triton/test_mamba3_siso.py"
    spec = importlib.util.spec_from_file_location("vendor_siso_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vendor reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mamba3_siso_fwd_ref


def one(dtype: torch.dtype) -> dict:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    ck = resolve_checkpoint("siso-187m", local_only=True)
    cfg = ck.load_config()
    model = MambaLMHeadModel.from_pretrained(
        ck.path, device="cuda", dtype=dtype
    ).eval()
    mixer = model.backbone.layers[0].mixer
    spec = InProjSpec.from_mixer(mixer)
    grabbed: dict[str, torch.Tensor] = {}
    pre = mixer.register_forward_pre_hook(
        lambda _m, inp: grabbed.__setitem__("u", inp[0].detach())
    )
    post = mixer.register_forward_hook(
        lambda _m, _inp, out: grabbed.__setitem__("out", out.detach())
    )
    ids = torch.tensor([[128000, 791]], device="cuda")
    try:
        with torch.inference_mode():
            model(ids)
    finally:
        pre.remove()
        post.remove()

    u = grabbed["u"]
    off = grabbed["out"]
    with torch.inference_mode():
        proj = mixer.in_proj(u)
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            proj,
            [
                spec.d_inner,
                spec.d_inner,
                spec.d_state * spec.ngroups * spec.mimo_rank,
                spec.d_state * spec.ngroups * spec.mimo_rank,
                spec.nheads,
                spec.nheads,
                spec.nheads,
                spec.n_rope_angles,
            ],
            dim=-1,
        )
        z = rearrange(z, "b l (h p) -> b l h p", p=spec.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=spec.headdim)
        B = rearrange(
            B, "b l (r g n) -> b l r g n", r=spec.mimo_rank, g=spec.ngroups
        )
        C = rearrange(
            C, "b l (r g n) -> b l r g n", r=spec.mimo_rank, g=spec.ngroups
        )
        trap_hl = rearrange(trap, "b l h -> b h l")
        A = -heavy_tail_activation(dd_A.float())
        A = A.clamp(max=-float(cfg["ssm_cfg"].get("A_floor", 1e-4)))
        dt = F.softplus(dd_dt + mixer.dt_bias)
        adt = rearrange(A * dt, "b l h -> b h l")
        dt_hl = rearrange(dt, "b l h -> b h l")
        angles_blhk = angles.unsqueeze(-2).expand(
            -1, -1, spec.nheads, -1
        ).float()
        Bn = mixer.B_norm(B).squeeze(2)
        Cn = mixer.C_norm(C).squeeze(2)

        vendor_ref = load_vendor_reference()
        y_vendor, _ = vendor_ref(
            Q=Cn,
            K=Bn,
            V=x,
            ADT=adt,
            DT=dt_hl,
            Trap=trap_hl,
            Q_bias=mixer.C_bias.squeeze(1),
            K_bias=mixer.B_bias.squeeze(1),
            Angles=angles_blhk,
            D=mixer.D,
            Z=z,
            chunk_size=mixer.chunk_size,
            dtype=torch.float32,
        )
        vendor_out = mixer.out_proj(
            y_vendor.reshape(1, ids.shape[1], spec.d_inner).to(x.dtype)
        )

    params = extract_layer_params(model, 0)
    ours = reference_block_forward(
        u[0].float().cpu(),
        params,
        spec,
        False,
        float(cfg["ssm_cfg"].get("A_floor", 1e-4)),
    )["out"]
    result = {
        "dtype": str(dtype),
        "vendor_ref_vs_kernel": parity_metrics(
            vendor_out[0].float().cpu(), off[0].float().cpu(), 1e-2
        ),
        "our_oracle_vs_kernel": parity_metrics(
            ours, off[0].float().cpu(), 1e-2
        ),
        "vendor_vs_ours": parity_metrics(
            vendor_out[0].float().cpu(), ours, 1e-2
        ),
    }
    del model
    torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    print(json.dumps([one(torch.bfloat16), one(torch.float32)], indent=2))
