"""Class-conditioned activation capture for state-spaces/mamba3-mimo-1.5b.

Companion to capture_activations.py. That script shows whether MIMO rank
capacity is used on generic web text; this one asks whether USAGE SHIFTS with
what is being said. We replay the Sub-Zero contrast corpora (corporate vs
authentic, code vs neutral) and accumulate the SAME statistics per class, then
log the per-layer deltas between them.

If a rank slot separates prompt classes anywhere, the claim becomes: class
information lives in the MIMO rank dimension of the SSM state projection. That
is the activation-side counterpart of the weight-side rank atlas.

Per class we accumulate exactly what capture_activations.py accumulates
(resid moments, mlp sparsity, per-head B/C rank Grams). Per contrast PAIR we
additionally log:
  B/C_gram_cos  cosine similarity between the two classes' averaged Grams
                (1.0 = identical usage; <1.0 = class-conditional usage)
  B/C_eff_rank_delta  difference in activation effective rank
  resid/mlp_frac_off_delta  sparsity shift between classes

NOTE: same caveat as capture_activations.py -- IN_PROJ_ORDER must match the
torch.split order in mamba3.py forward().

Corpora are the standard atlas prompts files ({\"text\": ...} per line),
expected in --corpus-dir under these names:
  authentic.jsonl  corporate.jsonl  code_probes.jsonl  neutral_stems.jsonl
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

MODEL_ID = "state-spaces/mamba3-mimo-1.5b"
TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
ENTITY = "ricks-holmberg-juiceb0xc0de"
PROJECT = "mamba3-mimo-atlas"

IN_PROJ_ORDER = ("z", "x", "B", "C", "dt", "A", "trapezoid", "rope")
SPARSITY_EPS = 0.01

# name -> (positive file, negative file), both relative to --corpus-dir
DEFAULT_PAIRS = (
    "compliance=corporate.jsonl:authentic.jsonl",
    "code=code_probes.jsonl:neutral_stems.jsonl",
)


class LayerAccum:
    """Running stats for one layer, one class. O(1) in token count."""

    def __init__(self, nheads, rank, dev):
        self.n = 0
        self.resid_absum = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_zero = torch.zeros((), device=dev, dtype=torch.float64)
        self.resid_count = 0
        self.mlp_zero = torch.zeros((), device=dev, dtype=torch.float64)
        self.mlp_count = 0
        self.B_gram = torch.zeros((nheads, rank, rank), device=dev, dtype=torch.float64)
        self.C_gram = torch.zeros((nheads, rank, rank), device=dev, dtype=torch.float64)

    def add_resid(self, h):
        f = h.float()
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

    def eff_rank(self, G, rank):
        G = G.float() / max(self.n, 1)
        s = torch.linalg.svdvals(G)
        return (s.sum(-1) ** 2 / (s**2).sum(-1)).mean().item()

    def finish(self, rank):
        out = {}
        out["resid_absmean"] = (self.resid_absum / self.resid_count).item()
        out["resid_frac_off"] = (self.resid_zero / self.resid_count).item()
        out["mlp_frac_off"] = (self.mlp_zero / max(self.mlp_count, 1)).item()
        out["B_act_eff_rank"] = self.eff_rank(self.B_gram, rank)
        out["C_act_eff_rank"] = self.eff_rank(self.C_gram, rank)
        off = ~torch.eye(rank, dtype=torch.bool, device=self.B_gram.device)
        for tag, G in (("B", self.B_gram), ("C", self.C_gram)):
            G = G / max(self.n, 1)
            d = G.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            corr = G / d.unsqueeze(-1) / d.unsqueeze(-2)
            out[f"{tag}_act_cos"] = corr[:, off].abs().mean().item()
        return out


def grams_cos(a, b, n_a, n_b):
    ga = (a / max(n_a, 1)).float().flatten()
    gb = (b / max(n_b, 1)).float().flatten()
    return F.cosine_similarity(ga, gb, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--tokens-per-class", type=int, default=500_000)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS),
                    help="name=pos.jsonl:neg.jsonl entries")
    ap.add_argument("--out", default="mamba3_contrast_stats.npz")
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

    # one accumulator set per (pair, side)
    acc = {}

    handles = []
    current = {"pair": None, "side": None}

    def target():
        return acc[(current["pair"], current["side"])]

    def mk_inproj(i):
        def hook(_m, _inp, out):
            o = out.reshape(-1, out.shape[-1])
            lo, hi = bounds["B"]
            B = o[:, lo:hi].reshape(-1, rank, mixer0.num_bc_heads, d_state)
            lo, hi = bounds["C"]
            C = o[:, lo:hi].reshape(-1, rank, mixer0.num_bc_heads, d_state)
            B = B.permute(0, 2, 1, 3).expand(-1, nheads, -1, -1)
            C = C.permute(0, 2, 1, 3).expand(-1, nheads, -1, -1)
            target()[i].add_state(B, C)
        return hook

    def mk_resid(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            target()[i].add_resid(h)
        return hook

    def mk_mlp(i):
        def hook(_m, inp):
            target()[i].add_mlp(inp[0])
        return hook

    pairs = []
    for spec in args.pairs:
        name, files = spec.split("=")
        pos, neg = files.split(":")
        pairs.append((name, pos, neg))
        for side in ("pos", "neg"):
            acc[(name, side)] = [
                LayerAccum(nheads, rank, dev) for _ in layers
            ]

    for i, blk in enumerate(layers):
        handles.append(blk.mixer.in_proj.register_forward_hook(mk_inproj(i)))
        handles.append(blk.register_forward_hook(mk_resid(i)))
        handles.append(blk.mlp.fc2.register_forward_pre_hook(mk_mlp(i)))

    run = wandb.init(
        entity=ENTITY, project=PROJECT, job_type="contrast-capture",
        config={
            "model_id": MODEL_ID, "tokens_per_class": args.tokens_per_class,
            "seqlen": args.seqlen, "batch": args.batch,
            "sparsity_eps": SPARSITY_EPS,
            "in_proj_order": list(IN_PROJ_ORDER),
            "pairs": [f"{p}={po}:{ne}" for p, po, ne in pairs],
            "corpus_dir": args.corpus_dir,
        },
    )

    def stream_class(pair, side, fname):
        current["pair"], current["side"] = pair, side
        buf, seen = [], 0
        with open(f"{args.corpus_dir}/{fname}") as fh:
            texts = [json.loads(l)["text"] for l in fh if l.strip()]
        print(f"{pair}/{side}: {len(texts)} docs from {fname}", flush=True)
        docs, di = texts, 0
        while seen < args.tokens_per_class:
            buf.extend(tok(docs[di], add_special_tokens=False).input_ids)
            di = (di + 1) % len(docs)
            while len(buf) >= args.seqlen * args.batch and seen < args.tokens_per_class:
                chunk = buf[: args.seqlen * args.batch]
                buf = buf[args.seqlen * args.batch :]
                ids = torch.tensor(chunk, device=dev).view(args.batch, args.seqlen)
                with torch.inference_mode():
                    model(ids)
                seen += ids.numel()
            if seen % (args.seqlen * args.batch * 25) == 0 and seen:
                print(f"  {pair}/{side} {seen:,} tokens", flush=True)
        print(f"  {pair}/{side} done: {seen:,} tokens", flush=True)

    for pair, pos, neg in pairs:
        stream_class(pair, "pos", pos)
        stream_class(pair, "neg", neg)

    for h in handles:
        h.remove()

    # per-class tables + per-pair separation rows
    for (pair, side), accs in acc.items():
        keys = None
        for i, a in enumerate(accs):
            st = a.finish(rank)
            keys = keys or sorted(st)
            run.log({f"{pair}/{side}/{k}": st[k] for k in keys} | {"layer": i})

    sep_rows = []
    for pair, pos, neg in pairs:
        pos_acc = acc[(pair, "pos")]
        neg_acc = acc[(pair, "neg")]
        for i, (pa, na) in enumerate(zip(pos_acc, neg_acc)):
            row = {
                "pair": pair, "layer": i,
                "B_gram_cos": grams_cos(pa.B_gram, na.B_gram, pa.n, na.n),
                "C_gram_cos": grams_cos(pa.C_gram, na.C_gram, pa.n, na.n),
                "B_eff_rank_delta": pa.eff_rank(pa.B_gram, rank) - na.eff_rank(na.B_gram, rank),
                "C_eff_rank_delta": pa.eff_rank(pa.C_gram, rank) - na.eff_rank(na.C_gram, rank),
            }
            pa_st = pa.finish(rank)
            na_st = na.finish(rank)
            row["resid_frac_off_delta"] = pa_st["resid_frac_off"] - na_st["resid_frac_off"]
            row["mlp_frac_off_delta"] = pa_st["mlp_frac_off"] - na_st["mlp_frac_off"]
            sep_rows.append(row)
            run.log({f"sep/{pair}/{k}": v for k, v in row.items()
                     if k not in ("pair", "layer")} | {"layer": i})

    table = wandb.Table(columns=["pair", "layer", "B_gram_cos", "C_gram_cos",
                                 "B_eff_rank_delta", "C_eff_rank_delta",
                                 "resid_frac_off_delta", "mlp_frac_off_delta"])
    for r in sep_rows:
        table.add_data(r["pair"], r["layer"], r["B_gram_cos"], r["C_gram_cos"],
                       r["B_eff_rank_delta"], r["C_eff_rank_delta"],
                       r["resid_frac_off_delta"], r["mlp_frac_off_delta"])
    run.log({"contrast_separation": table})

    # headline: most class-separable layer per pair/metric, for the summary
    for pair, _, _ in pairs:
        rows = [r for r in sep_rows if r["pair"] == pair]
        for k in ("B_gram_cos", "C_gram_cos"):
            best = min(rows, key=lambda r: r[k])
            run.summary[f"{pair}/{k}"] = best[k]
            run.summary[f"{pair}/{k}_layer"] = best["layer"]

    np.savez_compressed(
        args.out,
        sep=np.array([[r["pair"], r["layer"], r["B_gram_cos"], r["C_gram_cos"],
                       r["B_eff_rank_delta"], r["C_eff_rank_delta"],
                       r["resid_frac_off_delta"], r["mlp_frac_off_delta"]]
                      for r in sep_rows], dtype=object),
    )
    art = wandb.Artifact("contrast-stats", type="analysis")
    art.add_file(args.out)
    run.log_artifact(art)
    print(f"\nwrote {args.out}")
    run.finish()


if __name__ == "__main__":
    main()
