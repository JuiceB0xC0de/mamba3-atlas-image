"""Talk to state-spaces/mamba3-mimo-1.5b.

Base completion model (100B tokens FineWeb-Edu, no instruct tuning), so give it
the START of something rather than a question. 2048 ctx.

Fast path uses the cached decode in mamba_ssm.utils.generation, which calls
Mamba3.step() and needs the CuteDSL kernel. If that is unavailable we fall back
to recomputing the full forward each token: correct, just O(n^2) and slow.
Fallback is fine for a few dozen tokens.

  python talk.py                          interactive
  python talk.py --prompt "The reason"    one shot
"""

import argparse

import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

MODEL_ID = "state-spaces/mamba3-mimo-1.5b"
TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"
MAX_CTX = 2048


def sample(logits, temp, top_p):
    if temp <= 0:
        return logits.argmax(-1, keepdim=True)
    probs = (logits.float() / temp).softmax(-1)
    sp, si = probs.sort(-1, descending=True)
    keep = (sp.cumsum(-1) - sp) < top_p
    sp = torch.where(keep, sp, torch.zeros_like(sp))
    sp /= sp.sum(-1, keepdim=True)
    return si.gather(-1, sp.multinomial(1))


@torch.inference_mode()
def slow_generate(model, ids, n_new, temp, top_p, eos):
    """No cache: full forward per token. Correct but quadratic."""
    for _ in range(n_new):
        nxt = sample(model(ids[:, -MAX_CTX:]).logits[:, -1], temp, top_p)
        ids = torch.cat([ids, nxt], dim=-1)
        if eos is not None and nxt.item() == eos:
            break
    return ids


@torch.inference_mode()
def generate(model, ids, n_new, temp, top_p, eos, force_slow=False):
    if not force_slow:
        try:
            return model.generate(
                input_ids=ids,
                max_length=ids.shape[1] + n_new,
                temperature=temp if temp > 0 else 1.0,
                top_p=top_p,
                top_k=0,
                cg=False,
                enable_timing=False,
            ).sequences
        except Exception as e:
            print(f"[cached decode unavailable: {type(e).__name__}: {e}]")
            print("[falling back to full-forward generation]")
    return slow_generate(model, ids, n_new, temp, top_p, eos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt")
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--slow", action="store_true", help="skip the cached path")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    model = MambaLMHeadModel.from_pretrained(
        MODEL_ID, device="cuda", dtype=torch.bfloat16
    ).eval()
    eos = tok.eos_token_id

    def run(text):
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        out = generate(model, ids, args.max_new, args.temp, args.top_p, eos, args.slow)
        print("\n" + tok.decode(out[0], skip_special_tokens=True) + "\n")

    if args.prompt:
        run(args.prompt)
        return

    print("base completion model -- give it the start of something. ctrl-c to quit.\n")
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if text.strip():
            run(text)


if __name__ == "__main__":
    main()
