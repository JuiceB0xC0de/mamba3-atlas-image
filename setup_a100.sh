#!/usr/bin/env bash
set -euo pipefail

# Full env rebuild, /root paths (no /workspace on this pod).
# Order is load-bearing:
#   - tilelang gets --no-deps or it drags apache-tvm-ffi to 0.1.12 and core dumps
#   - quack-kernels and mamba_ssm declare a BARE `torch` dep; installing either
#     normally evicts 2.11.0+cu128 and pulls 2.13.0+cu130 -> cuda.is_available()=False
#   - torch is therefore reinstalled LAST, with --no-deps
#   - mamba comes from git main (only main's create_block maps "Mamba3"), with the
#     T.dynamic patch re-applied (main regressed it -> 67 recompiles, the 15h path)

VENV=/root/mamba3-venv
PY=$VENV/bin/python
PIP="$PY -m pip --disable-pip-version-check"
SRC=/root/mamba-src

echo "### venv (ubuntu python3.10 ships no ensurepip, so bootstrap pip by hand)"
rm -rf "$VENV"
python3.10 -m venv "$VENV" --without-pip
curl -sS -o /root/get-pip.py https://bootstrap.pypa.io/get-pip.py
$PY /root/get-pip.py > /dev/null
$PIP --version

$PIP install -U setuptools wheel ninja

echo "### torch 2.11.0+cu128"
$PIP install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "### the load-bearing pin"
$PIP install tilelang==0.1.8 apache-tvm-ffi==0.1.8.post2 --no-deps
$PIP install cloudpickle==3.1.2 ml_dtypes==0.5.4 psutil==7.2.2 \
    z3-solver==4.15.4.0 torch_c_dlpack_ext==0.1.5 tqdm==4.70.0

echo "### numeric + model stack"
$PIP install numpy==2.2.6 scipy==1.15.3 einops==0.8.2 \
    transformers==4.57.6 tokenizers==0.22.2 safetensors==0.8.0 \
    huggingface_hub==0.36.2 hf-xet==1.5.2 datasets==5.0.1 \
    pandas==2.3.3 pyarrow==25.0.0 wandb==0.28.1

echo "### cute / quack (quack --no-deps: bare torch dep)"
$PIP install nvidia-cutlass-dsl==4.6.0
$PIP install quack-kernels==0.6.1 --no-deps

echo "### mamba from git main + restore T.dynamic (PR #937, regressed on main)"
rm -rf "$SRC"
git clone --depth 1 https://github.com/state-spaces/mamba.git "$SRC" -q
cd "$SRC" && git log --oneline -1

$PY - <<'PYEOF'
import pathlib, sys
f = pathlib.Path("/root/mamba-src/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py")
src = f.read_text()
old = """    kernel = mamba_mimo_fwd(B,
                            S,
                            H,
                            G,
"""
new = """    kernel = mamba_mimo_fwd(T.dynamic("B"),
                            T.dynamic("S"),
                            T.dynamic("H"),
                            T.dynamic("G"),
"""
if src.count('T.dynamic("B")'):
    print("already dynamic")
elif old in src:
    f.write_text(src.replace(old, new, 1))
    print("T.dynamic restored")
else:
    sys.exit("FAILED: call site did not match")
PYEOF

$PIP install "$SRC" --no-deps --no-build-isolation --no-cache-dir

echo "### restore the pinned torch LAST"
$PIP install --force-reinstall --no-deps \
    torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "### verify"
$PY - <<'PYEOF'
import pathlib
import torch, tilelang, tvm_ffi
print("torch      ", torch.__version__)
print("cuda avail ", torch.cuda.is_available())
print("device     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("capability ", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "-")
print("tilelang   ", tilelang.__version__)

import mamba_ssm
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.modules.mamba3 import Mamba3

root = pathlib.Path(mamba_ssm.__file__).parent
has_m3 = '"Mamba3": Mamba3' in (root / "models/mixer_seq_simple.py").read_text()
n_dyn = (root / "ops/tilelang/mamba3/mamba3_mimo_fwd.py").read_text().count("T.dynamic")
print("create_block Mamba3:", has_m3)
print("T.dynamic  ", n_dyn, "occurrences")

assert torch.__version__.startswith("2.11.0+cu128"), "WRONG TORCH: " + torch.__version__
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
assert has_m3, "create_block CANNOT BUILD Mamba3"
assert n_dyn == 4, "T.dynamic MISSING -- recompile storm returns"
print("ENV GOOD")
PYEOF

echo "### done"
