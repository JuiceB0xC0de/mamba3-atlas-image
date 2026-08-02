"""B2-2 / G9: the L0 direct-logit test. Can support OR kill the detokenizer read.

Stage A found, in all eight checkpoints and both arms, that L0's skip-path sign
statistics are discontinuous from L1-L3, and that L0 has the shortest zero-input
retention prior in the stack. That is a BOUNDARY ANOMALY. It is not evidence of
detokenization, and no amount of weight-space analysis can make it so.

Why the tempting weights-only test is invalid: out_proj is (d_model, d_inner)
while the unembedding is (vocab, d_model), so the obvious row cosine is not even
dimensionally defined. Using out_proj COLUMNS fixes the dimensions but not the
inference -- raw alignment ignores which channels actually fire, their signs,
gating, and the norms. It is not an invariant measure of a layer's functional
contribution to logits.

THE VALID TEST (contract 4):
  For each layer l, take the residual stream r_l entering the block and the
  update dr_l the block adds. Push both r_l and (r_l + dr_l) through the final
  norm and the tied unembedding. The DIFFERENCE in logits is that layer's direct
  contribution to the output distribution at that position.

  Measured per layer:
    direct_logit_delta   mean |change| in logits attributable to the layer
    target_logit_delta   change in the logit of the ACTUAL next token
    entropy_delta        change in predictive entropy (negative = sharpening)
    loss_delta           change in next-token cross-entropy (negative = helps)
    top1_flip_rate       how often the layer changes the argmax

WHAT WOULD SUPPORT THE DETOKENIZER READ
  L0 contributing a LARGE, token-local sharpening: big direct_logit_delta and a
  strongly negative entropy_delta, concentrated on the current/next token rather
  than accumulating over context.

WHAT WOULD KILL IT
  L0 contributing near-nothing to logits (its role being purely to set up state
  for later layers), or contributing indistinguishably from L1-L3.

Both outcomes are publishable. This script does not assume either.

Usage:
  python analysis/l0_direct_logit.py --model mimo-1.5b --stream a \\
      --token-contract token_contract.npz --out l0_direct_logit_1p5b.npz
"""

import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from mamba3_core import assert_runtime  # noqa: E402


def entropy(logits):
    lp = F.log_softmax(logits.float(), dim=-1)
    return -(lp.exp() * lp).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stream", choices=["a", "b"], default="a")
    ap.add_argument("--token-contract", default="token_contract.npz")
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="l0_direct_logit.npz")
    args = ap.parse_args()

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    dev = "cuda"
    repo = f"state-spaces/mamba3-{args.model}"
    model = MambaLMHeadModel.from_pretrained(repo, device=dev, dtype=torch.bfloat16).eval()
    layers = model.backbone.layers
    n_layer = len(layers)
    norm_f = model.backbone.norm_f
    head = model.lm_head

    tc = np.load(args.token_contract, allow_pickle=False)
    if args.stream == "a":
        seqs = tc["a_ids"][: args.max_seqs]
    else:
        ids, offs = tc["b_ids"], tc["b_offsets"]
        n = min(args.max_seqs, len(offs) - 1)
        L = int(max(offs[b + 1] - offs[b] for b in range(n)))
        seqs = np.stack([
            np.pad(ids[int(offs[b]):int(offs[b + 1])], (0, L - int(offs[b + 1] - offs[b])))
            for b in range(n)
        ])
    print(f"{repo}: {seqs.shape[0]} sequences of {seqs.shape[1]}, {n_layer} layers")

    # capture the residual entering and leaving every block
    pre, post = {}, {}
    handles = []
    for li, blk in enumerate(layers):
        handles.append(blk.register_forward_pre_hook(
            lambda m, inp, li=li: pre.__setitem__(li, inp[0].detach())))
        handles.append(blk.register_forward_hook(
            lambda m, i, o, li=li: post.__setitem__(
                li, (o[0] if isinstance(o, tuple) else o).detach())))

    acc = {k: torch.zeros(n_layer, dtype=torch.float64)
           for k in ("direct", "target", "entropy", "loss", "flip")}
    n_pos_total = 0
    t0 = time.time()

    for bi in range(0, seqs.shape[0], args.batch):
        batch = torch.tensor(seqs[bi:bi + args.batch], device=dev, dtype=torch.long)
        with torch.inference_mode():
            model(batch)

            # next-token targets; drop the final position which has no target
            tgt = batch[:, 1:]
            for li in range(n_layer):
                r_in = pre[li][:, :-1].float()
                r_out = post[li][:, :-1].float()

                lg_before = head(norm_f(r_in.to(head.weight.dtype))).float()
                lg_after = head(norm_f(r_out.to(head.weight.dtype))).float()
                delta = lg_after - lg_before

                acc["direct"][li] += delta.abs().mean().double().cpu() * tgt.numel()

                t = tgt.unsqueeze(-1)
                acc["target"][li] += delta.gather(-1, t).mean().double().cpu() * tgt.numel()

                acc["entropy"][li] += (
                    entropy(lg_after) - entropy(lg_before)
                ).mean().double().cpu() * tgt.numel()

                ce_a = F.cross_entropy(lg_after.reshape(-1, lg_after.shape[-1]),
                                       tgt.reshape(-1), reduction="mean")
                ce_b = F.cross_entropy(lg_before.reshape(-1, lg_before.shape[-1]),
                                       tgt.reshape(-1), reduction="mean")
                acc["loss"][li] += (ce_a - ce_b).double().cpu() * tgt.numel()

                acc["flip"][li] += (
                    lg_after.argmax(-1) != lg_before.argmax(-1)
                ).float().mean().double().cpu() * tgt.numel()

            n_pos_total += tgt.numel()
        if (bi // args.batch) % 4 == 0:
            print(f"  {bi}/{seqs.shape[0]} seqs  {time.time() - t0:.0f}s", flush=True)

    for h in handles:
        h.remove()
    res = {k: (v / max(n_pos_total, 1)).numpy() for k, v in acc.items()}

    print(f"\n{'L':>3s} {'direct':>10s} {'target':>10s} {'entropy':>10s} "
          f"{'loss':>10s} {'flip%':>8s}")
    for li in range(n_layer):
        print(f"{li:3d} {res['direct'][li]:10.4f} {res['target'][li]:10.4f} "
              f"{res['entropy'][li]:10.4f} {res['loss'][li]:10.4f} "
              f"{100 * res['flip'][li]:8.2f}")

    # L0 against the L1-L3 reference band, which is what Stage A found it
    # discontinuous from
    band = slice(1, 4)
    verdict = {}
    for k, v in res.items():
        ref = float(np.mean(v[band]))
        verdict[k] = {"L0": float(v[0]), "L1_L3_mean": ref,
                      "ratio": float(v[0] / ref) if ref else float("nan")}
    print("\nL0 vs the L1-L3 band:")
    for k, d in verdict.items():
        print(f"  {k:8s} L0={d['L0']:.4f}  L1-3={d['L1_L3_mean']:.4f}  "
              f"ratio={d['ratio']:.2f}x")

    print("\nreading guide:")
    print("  detokenizer SUPPORTED if L0 direct is large AND entropy_delta is")
    print("  strongly negative (sharpening) relative to the L1-L3 band.")
    print("  detokenizer KILLED if L0 direct is near zero or indistinguishable")
    print("  from L1-L3: L0 would then be setting up state, not writing logits.")

    np.savez_compressed(args.out, **res, n_positions=np.array([n_pos_total]))
    manifest = assert_runtime({"model": repo, "requires_kernel": True}, strict=False)
    manifest |= {"n_layer": n_layer, "n_positions": int(n_pos_total),
                 "stream": args.stream, "verdict": verdict}
    with open(args.out.replace(".npz", ".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
