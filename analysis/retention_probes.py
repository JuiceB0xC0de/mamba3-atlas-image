"""C-3: retention and overwrite probes. What does the state actually hold?

GATE: the BEHAVIOURAL half of this file runs on prefill alone and is valid
regardless of gpu_probe G3. The MECHANISTIC half (correlating recall with state
quantities) requires state access and is SKIPPED if G3 returned BOUNDARY.
Run with --behavioural-only to force the safe subset.

WHY THIS AND NOT PERPLEXITY
  An SSM has a fixed-size state. The interesting question is not "is it a good
  language model" but "what survives, how long, and what evicts it". These
  probes construct that directly:

    distance      recall a key-value pair after N filler tokens. Sweeping N
                  measures the retention curve, which is the state's actual
                  memory span rather than a per-token half-life extrapolation.
    distractor    insert unrelated pairs between binding and query. Measures
                  interference capacity, i.e. how many bindings coexist.
    contradiction bind X=1, then X=2, then query X. Tests whether the state
                  OVERWRITES or superposes. A superposing state answers 1
                  sometimes; an overwriting state always answers 2.
    reorder       same pairs, different order. If recall depends on order
                  beyond recency, the state is not a simple decaying store.
    deletion      bind then explicitly negate. Tests active erasure.
    replacement   bind X=1, then "actually X=2". Softer than contradiction.

The zero-input retention prior from Stage A (ln2 / softplus(dt_bias), median
2.3 tokens at L0 on siso-187m) is a PRIOR at zero input. These probes measure
the realized span on real tokens, which is the number that matters.

Usage:
  python analysis/retention_probes.py --model mimo-1.5b --out retention.json
  python analysis/retention_probes.py --model mimo-1.5b --behavioural-only
"""

import argparse
import json
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])

KEYS = ["alpha", "bravo", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
VALUES = ["red", "blue", "green", "amber", "violet", "silver", "orange", "teal"]
FILLER = ("The weather is mild today. Nothing important is happening. "
          "The room is quiet and the light is steady. ")


def make_probe(kind, rng, distance=0, n_distract=0):
    """Return (prompt, expected_answer, meta). Prompts are deliberately plain."""
    k, v = rng.choice(KEYS), rng.choice(VALUES)
    other_k = rng.choice([x for x in KEYS if x != k])
    v2 = rng.choice([x for x in VALUES if x != v])

    filler = FILLER * max(distance // 12, 0)
    distract = "".join(
        f"The {rng.choice([x for x in KEYS if x != k])} box is "
        f"{rng.choice([x for x in VALUES if x != v])}. "
        for _ in range(n_distract)
    )

    if kind == "distance":
        p = f"The {k} box is {v}. {filler}The {k} box is"
        return p, v, {"distance": distance}
    if kind == "distractor":
        p = f"The {k} box is {v}. {distract}The {k} box is"
        return p, v, {"n_distract": n_distract}
    if kind == "contradiction":
        p = f"The {k} box is {v}. The {k} box is {v2}. {filler}The {k} box is"
        return p, v2, {"expect": "overwrite", "alternative": v}
    if kind == "reorder":
        p = f"The {other_k} box is {v2}. The {k} box is {v}. {filler}The {k} box is"
        return p, v, {}
    if kind == "deletion":
        p = (f"The {k} box is {v}. The {k} box is no longer {v}. "
             f"{filler}The {k} box is")
        return p, v, {"note": "answer is ambiguous by design; we score entropy"}
    if kind == "replacement":
        p = f"The {k} box is {v}. Actually, the {k} box is {v2}. {filler}The {k} box is"
        return p, v2, {"alternative": v}
    raise ValueError(kind)


@torch.inference_mode()
def score(model, tok, prompt, answer, alternative=None):
    ids = torch.tensor(
        [tok.bos_token_id] + tok(prompt, add_special_tokens=False).input_ids,
        device="cuda"
    ).unsqueeze(0)
    logits = model(ids).logits[0, -1].float()
    lp = F.log_softmax(logits, dim=-1)

    a_ids = tok(" " + answer, add_special_tokens=False).input_ids
    out = {
        "answer_logprob": float(lp[a_ids[0]]),
        "top1": tok.decode([int(logits.argmax())]),
        "correct": int(logits.argmax()) == a_ids[0],
        "entropy": float(-(lp.exp() * lp).sum()),
    }
    if alternative:
        b_ids = tok(" " + alternative, add_special_tokens=False).input_ids
        out["alt_logprob"] = float(lp[b_ids[0]])
        out["pref_margin"] = out["answer_logprob"] - out["alt_logprob"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mimo-1.5b")
    ap.add_argument("--n-per-cell", type=int, default=24)
    ap.add_argument("--distances", default="0,12,24,48,96,192")
    ap.add_argument("--distractors", default="0,1,2,4,8")
    ap.add_argument("--behavioural-only", action="store_true")
    ap.add_argument("--parity-report", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="retention_probes.json")
    args = ap.parse_args()

    state_ok = False
    if args.parity_report and not args.behavioural_only:
        v = json.load(open(args.parity_report)).get("verdicts", {})
        state_ok = v.get("g3") == "PASS"
        if not state_ok:
            print(f"g3={v.get('g3')}: running BEHAVIOURAL probes only; "
                  "state-correlate analysis is prohibited")

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from transformers import AutoTokenizer

    model = MambaLMHeadModel.from_pretrained(
        f"state-spaces/mamba3-{args.model}", device="cuda", dtype=torch.bfloat16
    ).eval()
    tok = AutoTokenizer.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B",
        revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
    )

    rng = random.Random(args.seed)
    results = {"model": args.model, "state_analysis": state_ok, "cells": {}}

    for dist in [int(x) for x in args.distances.split(",")]:
        rows = [score(model, tok, *make_probe("distance", rng, distance=dist)[:2])
                for _ in range(args.n_per_cell)]
        results["cells"][f"distance/{dist}"] = summarize(rows)
        print(f"  distance {dist:4d}: acc={results['cells'][f'distance/{dist}']['accuracy']:.2f}")

    for nd in [int(x) for x in args.distractors.split(",")]:
        rows = [score(model, tok, *make_probe("distractor", rng, n_distract=nd)[:2])
                for _ in range(args.n_per_cell)]
        results["cells"][f"distractor/{nd}"] = summarize(rows)
        print(f"  distractors {nd:3d}: acc={results['cells'][f'distractor/{nd}']['accuracy']:.2f}")

    for kind in ("contradiction", "reorder", "deletion", "replacement"):
        rows = []
        for _ in range(args.n_per_cell):
            p, a, meta = make_probe(kind, rng, distance=12)
            rows.append(score(model, tok, p, a, meta.get("alternative")))
        results["cells"][kind] = summarize(rows)
        print(f"  {kind:14s}: acc={results['cells'][kind]['accuracy']:.2f}")

    # the retention curve is the headline: where does recall fall to chance
    accs = [(d, results["cells"][f"distance/{d}"]["accuracy"])
            for d in [int(x) for x in args.distances.split(",")]]
    half = next((d for d, a in accs if a < 0.5 * accs[0][1]), None)
    results["summary"] = {
        "retention_curve": accs,
        "distance_at_half_initial_accuracy": half,
        "note": ("this is the REALIZED span on real tokens. The Stage A "
                 "zero-input prior (ln2/softplus(dt_bias)) is a prior at zero "
                 "input and is not expected to match."),
    }
    if not state_ok:
        results["prohibited"] = ["state-correlate analysis (gpu_probe g3 not PASS)"]

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nretention half-point at distance {half}; wrote {args.out}")


def summarize(rows):
    return {
        "n": len(rows),
        "accuracy": float(np.mean([r["correct"] for r in rows])),
        "answer_logprob": float(np.mean([r["answer_logprob"] for r in rows])),
        "entropy": float(np.mean([r["entropy"] for r in rows])),
        "pref_margin": (float(np.mean([r["pref_margin"] for r in rows]))
                        if "pref_margin" in rows[0] else None),
    }


if __name__ == "__main__":
    main()
