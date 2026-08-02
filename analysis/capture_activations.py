"""Streaming activation capture for state-spaces/mamba3-mimo-1.5b.

Answers the question the weight analysis cannot: the weights show MIMO rank
CAPACITY, this shows whether that capacity is USED at inference.

Design: accumulate statistics online rather than dumping tensors. The residual
stream alone is ~96 KB/token across 24 layers, so a 20 GB dump buys only ~200k
tokens -- enough for descriptive stats, useless for SAE training. So we stream
as many tokens as we want through running accumulators, and separately keep a
small bounded raw sample for inspection.

Per layer we accumulate:
  resid_*     residual stream moments + a near-zero fraction (activation sparsity)
  mlp_*       post-SwiGLU hidden sparsity (hooked as the input to mlp.fc2)
  B/C rank    Gram matrix across the mimo_rank axis of the B and C state
              projections, per head, averaged over tokens. Effective rank of
              THAT is utilization, to be compared against the weight-side
              spread from mimo_rank_atlas.py.

NOTE: IN_PROJ_ORDER below must match the torch.split in mamba3.py forward().
Sizes are derived from config; only the ORDER is asserted here. Verify against
the source before trusting any B/C number.
"""

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

MODEL_ID = "state-spaces/mamba3-mimo-1.5b"
TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
ENTITY = "ricks-holmberg-juiceb0xc0de"
PROJECT = "mamba3-mimo-atlas"

# Order of the concatenated blocks in in_proj's output. CONFIRM against the
# torch.split call in mamba3.py forward() before trusting B/C results.
IN_PROJ_ORDER = ("z", "x", "B", "C", "dt", "A", "trapezoid", "rope")

# A value counts as "off" when |a| is below this multiple of the tensor's own
# std. Relative, so it does not assume a scale.
SPARSITY_EPS = 0.01


class LayerAccum:
    """Running stats for one layer. Everything is O(1) in token count."""

    def __init__(self, nheads, rank, d_state, dev):
        self.n = 0
        self.resid_sum = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_sqsum = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_absum = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_zero = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_count = 0
        self.mlp_zero = torch.zeros((), device=dev, dtype=torch.float64)
        self.mlp_count = 0
        # Gram across the rank axis, per head, accumulated over tokens.
        self.B_gram = torch.zeros((nheads, rank, rank), device=dev, dtype=torch.float64)
        self.C_gram = torch.zeros((nheads, rank, rank), device=dev, dtype=torch.float64)

    def add_resid(self, h):
        f = h.float()
        self.resid_sum += f.sum()
        self.resid_sqsum += (f * f).sum()
        self.resid_absum += f.abs().sum()
        self.resid_zero += (f.abs() < SPARSITY_EPS * f.std()).sum()
        self.resid_count += f.numel()
        self.n += f.shape[0] * f.shape[1]

    def add_mlp(self, h):
        f = h.float()
        self.mlp_zero += (f.abs() < SPARSITY_EPS * f.std()).sum()
        self.mlp_count += f.numel()

    def add_state(self, B, C):
        """B, C: (tokens, nheads, rank, d_state). Normalize then accumulate Gram."""
        for src, dst in ((B, self.B_gram), (C, self.C_gram)):
            v = F.normalize(src.float(), dim=-1)
            dst += torch.einsum("thrd,thsd->hrs", v, v).double()

    def finish(self, rank):
        out = {}
        mean = (self.resid_sum / self.resid_count).item()
        sq = (self.resid_sqsum / self.resid_count).item()
        out["resid_mean"] = mean
        out["resid_std"] = math.sqrt(max(sq - mean * mean, 0.0))
        out["resid_absmean"] = (self.resid_absum / self.resid_count).item()
        out["resid_frac_off"] = (self.resid_zero / self.resid_count).item()
        out["mlp_frac_off"] = (self.mlp_zero / max(self.mlp_count, 1)).item()

        off = ~torch.eye(rank, dtype=torch.bool, device=self.B_gram.device)
        for tag, G in (("B", self.B_gram), ("C", self.C_gram)):
            G = G / max(self.n, 1)
            d = G.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            corr = G / d.unsqueeze(-1) / d.unsqueeze(-2)
            out[f"{tag}_act_cos"] = corr[:, off].abs().mean().item()
            s = torch.linalg.svdvals(G.float())
            eff = s.sum(-1) ** 2 / (s**2).sum(-1)
            out[f"{tag}_act_eff_rank"] = eff.mean().item()
            out[f"{tag}_act_norm_rank"] = ((eff.mean() - 1) / (rank - 1)).item()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sample-seqs", type=int, default=4, help="raw seqs to keep")
    ap.add_argument("--out", default="mamba3_activation_stats.npz")
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    model = MambaLMHeadModel.from_pretrained(
        MODEL_ID, device=dev, dtype=torch.bfloat16
    ).eval()
    layers = model.backbone.layers
    mixer0 = layers[0].mixer
    nheads, rank, headdim = mixer0.mimo_x.shape
    d_inner = mixer0.out_proj.weight.shape[1]
    d_state = mixer0.B_norm.weight.shape[0]
    n_bc = d_state * mixer0.num_bc_heads * rank

    sizes = {
        "z": d_inner, "x": d_inner, "B": n_bc, "C": n_bc,
        "dt": nheads, "A": nheads, "trapezoid": nheads,
        "rope": mixer0.num_rope_angles,
    }
    total = sum(sizes[k] for k in IN_PROJ_ORDER)
    expected = mixer0.in_proj.weight.shape[0]
    assert total == expected, f"slice sizes {total} != in_proj {expected}"

    bounds, at = {}, 0
    for k in IN_PROJ_ORDER:
        bounds[k] = (at, at + sizes[k])
        at += sizes[k]

    acc = [LayerAccum(nheads, rank, d_state, dev) for _ in layers]
    handles = []

    def mk_inproj(i):
        def hook(_m, _inp, out):
            o = out.reshape(-1, out.shape[-1])
            lo, hi = bounds["B"]
            B = o[:, lo:hi].reshape(-1, rank, mixer0.num_bc_heads, d_state)
            lo, hi = bounds["C"]
            C = o[:, lo:hi].reshape(-1, rank, mixer0.num_bc_heads, d_state)
            # (t, r, g, n) -> (t, h, r, n); ngroups=1 broadcasts across heads
            B = B.permute(0, 2, 1, 3).expand(-1, nheads, -1, -1)
            C = C.permute(0, 2, 1, 3).expand(-1, nheads, -1, -1)
            acc[i].add_state(B, C)
        return hook

    def mk_resid(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            acc[i].add_resid(h)
        return hook

    def mk_mlp(i):
        def hook(_m, inp):
            acc[i].add_mlp(inp[0])
        return hook

    for i, blk in enumerate(layers):
        handles.append(blk.mixer.in_proj.register_forward_hook(mk_inproj(i)))
        handles.append(blk.register_forward_hook(mk_resid(i)))
        handles.append(blk.mlp.fc2.register_forward_pre_hook(mk_mlp(i)))

    run = wandb.init(
        entity=ENTITY, project=PROJECT, job_type="activation-capture",
        config={
            "model_id": MODEL_ID, "tokens": args.tokens, "seqlen": args.seqlen,
            "batch": args.batch, "sparsity_eps": SPARSITY_EPS,
            "in_proj_order": list(IN_PROJ_ORDER), "corpus": "fineweb-edu sample-10BT",
        },
    )

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT",
        split="train", streaming=True,
    )

    buf, seen, samples = [], 0, []
    for rec in ds:
        buf.extend(tok(rec["text"], add_special_tokens=False).input_ids)
        while len(buf) >= args.seqlen * args.batch and seen < args.tokens:
            chunk = buf[: args.seqlen * args.batch]
            buf = buf[args.seqlen * args.batch :]
            ids = torch.tensor(chunk, device=dev).view(args.batch, args.seqlen)
            with torch.inference_mode():
                model(ids)
            seen += ids.numel()
            if len(samples) < args.sample_seqs:
                samples.append(ids[0].cpu().numpy())
            if seen % (args.seqlen * args.batch * 25) == 0:
                print(f"{seen:,} / {args.tokens:,} tokens", flush=True)
        if seen >= args.tokens:
            break

    for h in handles:
        h.remove()

    keys = None
    rows = []
    for i, a in enumerate(acc):
        st = a.finish(rank)
        keys = keys or sorted(st)
        rows.append([st[k] for k in keys])
        run.log({f"act/{k}": st[k] for k in keys} | {"layer": i})

    arr = np.array(rows, dtype=np.float32)
    table = wandb.Table(columns=["layer"] + keys)
    for i, r in enumerate(rows):
        table.add_data(i, *r)
    run.log({"per_layer_activation": table})

    np.savez_compressed(
        args.out, metrics=np.array(keys), values=arr,
        sample_token_ids=np.array(samples), tokens_seen=np.array([seen]),
    )
    art = wandb.Artifact("activation-stats", type="analysis")
    art.add_file(args.out)
    run.log_artifact(art)
    print(f"\n{seen:,} tokens, wrote {args.out}")
    run.finish()


if __name__ == "__main__":
    main()
