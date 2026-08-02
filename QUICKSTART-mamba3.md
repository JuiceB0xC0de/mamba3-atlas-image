# Mamba-3 MIMO 1.5B — Quickstart

How to go from a bare GPU pod to a working forward pass on
`state-spaces/mamba3-mimo-1.5b`. Verified 2026-07-30 on an RTX 6000 Ada.

The load-bearing part is the **pin set**. The dependency resolver does the wrong
thing if you let it run free — it took six attempts to find this. Follow the
order; the `--no-deps` flags are not optional.

---

## Box requirements

| | |
|---|---|
| GPU | Ada (sm_89) or Ampere (sm_86), 24 GB+. 3 GB weights, tiny ctx — 24 GB is plenty. |
| Driver | CUDA **12.8** (`nvidia-smi`). If the image ships a cu13 torch, `torch.cuda.is_available()` returns False against a 12.8 driver. |
| Python | 3.10 |
| Avoid | Blackwell for now — HEAD commit `e9594ce` fixes silent forward-pass corruption in the Mamba-3 SISO kernel on Blackwell. Fine on Ada/Ampere. |

## Verified working stack

| package | version | note |
|---|---|---|
| torch | 2.11.0+cu128 | must be cu128 to match the 12.8 driver |
| triton | 3.6.0 | mamba needs `triton>=3.5`, so torch 2.7 (triton 3.3) is too old |
| tilelang | **0.1.8** | exact pin from mamba `setup.py` |
| apache-tvm-ffi | **0.1.8.post2** | the undocumented one — see gotcha #1 |
| quack-kernels | >=0.3.4 | |
| mamba-ssm | editable from source | image copies are pre-Mamba-3 |

---

## Steps

```bash
# 1. Source tree (the image's mamba_ssm is almost certainly pre-Mamba-3)
git clone https://github.com/state-spaces/mamba.git /workspace/mamba

# 2. Remove the image's prebuilt copy so the editable install wins
pip uninstall -y mamba-ssm

# 3. Editable install from source. --no-build-isolation reuses the installed
#    torch instead of pulling its own into a throwaway build env.
pip install -e /workspace/mamba --no-build-isolation

# 4. Correct torch: cu128 build brings triton 3.6 (>=3.5 required) and matches
#    the 12.8 driver. Do this even if the image already had a torch.
pip install -U torch --index-url https://download.pytorch.org/whl/cu128

# 5. THE PIN. tilelang 0.1.8 leaves apache-tvm-ffi unbounded, so a free resolve
#    grabs 0.1.12 and the process core-dumps on import. --no-deps to keep torch.
pip install tilelang==0.1.8 apache-tvm-ffi==0.1.8.post2 --no-deps

# 6. Remaining kernel dep
pip install quack-kernels

# 7. Kill stale torchvision/torchaudio (usually pinned to torch 2.2 in ML images).
#    transformers imports torchvision and crashes on torchvision::nms otherwise.
pip uninstall -y torchvision torchaudio

# 8. Only if you'll stream a corpus
pip install datasets

# 9. Auth (checkpoint repo may be private/gated)
hf auth login
```

Confirm the import chain before loading weights:

```bash
python -c "import torch, triton; print(torch.__version__, torch.cuda.is_available(), triton.__version__)"
# -> 2.11.0+cu128 True 3.6.0
python -c "from mamba_ssm import Mamba3; print(Mamba3)"
# -> <class 'mamba_ssm.modules.mamba3.Mamba3'>
```

## Load + forward

```python
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Ungated stand-in for the gated meta-llama tokenizer. Same 128256 vocab.
tok = AutoTokenizer.from_pretrained(
    "NousResearch/Meta-Llama-3.1-8B",
    revision="1f47e50cdbe801ad8a5174156ec3a0655108fb9f",
)
model = MambaLMHeadModel.from_pretrained(       # NOT AutoModelForCausalLM
    "state-spaces/mamba3-mimo-1.5b", device="cuda", dtype=torch.bfloat16
).eval()

ids = tok("Mamba-3 is", return_tensors="pt").input_ids.cuda()
with torch.inference_mode():
    logits = model(ids).logits                  # first call JIT-compiles ~12s
print(logits.shape)                             # [1, seq, 128256]
```

Baseline: `params 1.497B`, no NaN, ~3.55 GB VRAM, top-5 after "Mamba-3 is" =
` a` / ` an` / ` the` / ` one` / ` designed`.

---

## Gotchas (each cost real time)

1. **apache-tvm-ffi must be 0.1.8.post2.** A free resolve installs 0.1.12 and you
   get `AttributeError: attribute '__dict__' of 'type' objects is not writable`
   or a hard `TypeAttr __ffi_repr__ is already registered for type index 130`
   core dump. tilelang is a **hard import-time dep** — `mamba_ssm/__init__.py`
   imports `Mamba3`, which imports tilelang. No lazy fallback.
2. **torch 2.7 is too old** (ships triton 3.3, mamba wants >=3.5). Use 2.11 cu128.
3. **torchvision::nms crash** from stale torchvision in the image. Uninstall it.
4. **Gated tokenizer.** The card names `meta-llama/Llama-3.1-8B`; use the
   NousResearch pin above to skip the license gate.
5. **No conv.** Prebuilt `causal-conv1d` in an ML image does nothing here —
   Mamba-3 has no `d_conv`. Same for `flash_attn`: zero attention layers.
6. **Use `MambaLMHeadModel.from_pretrained`**, not `AutoModelForCausalLM`. This is
   a raw `mamba_ssm` checkpoint, not a transformers model.
7. **Generation needs CuteDSL.** `Mamba3.step()` (token decode) needs the CuteDSL
   kernel, untested here. Forward/prefill works. For quick generation, full-forward
   per token in a **fixed-size buffer** — growing the sequence length retriggers a
   ~12s TileLang recompile every token.
8. **Snapshot the env.** `pip freeze > requirements-lock.txt` on a working box so
   you never re-derive this. The TileLang JIT cache is sm-specific and only saves
   the ~12s first-forward compile.
