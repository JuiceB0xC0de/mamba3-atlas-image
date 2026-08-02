# Mamba-3 atlas runtime.
#
# Every pin and every --no-deps below is load-bearing. Notes:
#
#   * ubuntu 22.04 ships python3.10, which is the version this stack is verified on.
#   * python3.10-dev is REQUIRED AT RUNTIME, not just to build. Triton JIT-compiles a
#     small C shim (cuda_utils.c) on first kernel launch and needs Python.h. Without it
#     every run dies with "fatal error: Python.h: No such file or directory".
#   * apache-tvm-ffi is unbounded in tilelang 0.1.8's metadata. A free resolve takes
#     0.1.12 and you get either AttributeError on __dict__ or a hard
#     "TypeAttr __ffi_repr__ is already registered" core dump. Hence --no-deps + pin.
#   * quack-kernels and mamba_ssm declare a BARE `torch` dependency. Installing either
#     normally evicts 2.11.0+cu128 and pulls 2.13.0+cu130, which cannot run on a
#     CUDA 12.x driver. Both go in with --no-deps, and torch is restored LAST.
#   * mamba comes from git main, NOT PyPI:
#       - PyPI 2.3.2.post1 has T.dynamic but its create_block cannot build Mamba3
#         ("Invalid ssm_layer: Mamba3, only support Mamba1 and Mamba2").
#       - main's create_block maps Mamba3, but main REGRESSED T.dynamic back to static
#         (B,S,H,G), which recompiles the TileLang kernel once per unique sequence
#         length. That is the difference between a 20-minute run and a 15-hour one.
#     So: main, plus re-apply the T.dynamic patch (upstream PR #937).
#   * MAMBA_KEEP_CUDA_BUILD defaults FALSE on main, which skips the ~30 min
#     selective_scan compile. Mamba-3 never calls it; main guards the import.
#   * causal-conv1d and flash-attn do NOTHING here: Mamba-3 has no d_conv and zero
#     attention layers. Do not add them.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VENV=/opt/mamba3-venv \
    PATH=/opt/mamba3-venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3.10-venv \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# ubuntu's python3.10 ships no ensurepip, so bootstrap pip by hand
RUN python3.10 -m venv $VENV --without-pip \
    && curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py \
    && $VENV/bin/python /tmp/get-pip.py \
    && rm /tmp/get-pip.py \
    && $VENV/bin/pip install -U setuptools wheel ninja

RUN pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# the load-bearing pin
RUN pip install tilelang==0.1.8 apache-tvm-ffi==0.1.8.post2 --no-deps \
    && pip install \
        cloudpickle==3.1.2 ml_dtypes==0.5.4 psutil==7.2.2 \
        z3-solver==4.15.4.0 torch_c_dlpack_ext==0.1.5 tqdm==4.70.0

RUN pip install \
        numpy==2.2.6 scipy==1.15.3 einops==0.8.2 \
        transformers==4.57.6 tokenizers==0.22.2 safetensors==0.8.0 \
        huggingface_hub==0.36.2 hf-xet==1.5.2 datasets==5.0.1 \
        pandas==2.3.3 pyarrow==25.0.0 wandb==0.28.1

RUN pip install nvidia-cutlass-dsl==4.6.0 \
    && pip install quack-kernels==0.6.1 --no-deps

# mamba: git main for create_block, patched to restore T.dynamic
ARG MAMBA_REF=main
RUN git clone --depth 1 --branch ${MAMBA_REF} \
        https://github.com/state-spaces/mamba.git /opt/mamba-src \
    && cd /opt/mamba-src && git rev-parse HEAD > /opt/mamba-src/COMMIT

RUN python - <<'PYEOF'
import pathlib, sys
f = pathlib.Path("/opt/mamba-src/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py")
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
    sys.exit("FAILED: mamba3_mimo_fwd call site did not match -- upstream moved it")
PYEOF

RUN MAMBA_KEEP_CUDA_BUILD=FALSE \
    pip install /opt/mamba-src --no-deps --no-build-isolation --no-cache-dir

# torch LAST: nothing above may be allowed to have replaced it
RUN pip install --force-reinstall --no-deps \
        torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Build-time verification. A broken image must fail HERE, not on a rented GPU.
# cuda.is_available() is deliberately NOT asserted: the builder has no device.
RUN python - <<'PYEOF'
import pathlib
import torch, tilelang, tvm_ffi
import mamba_ssm
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.modules.mamba3 import Mamba3

root = pathlib.Path(mamba_ssm.__file__).parent
has_m3 = '"Mamba3": Mamba3' in (root / "models/mixer_seq_simple.py").read_text()
n_dyn = (root / "ops/tilelang/mamba3/mamba3_mimo_fwd.py").read_text().count("T.dynamic")

print("torch     ", torch.__version__)
print("tilelang  ", tilelang.__version__)
print("mamba_ssm ", mamba_ssm.__version__)
print("create_block Mamba3:", has_m3)
print("T.dynamic ", n_dyn)

assert torch.__version__.startswith("2.11.0+cu128"), "WRONG TORCH: " + torch.__version__
assert tilelang.__version__ == "0.1.8", "WRONG TILELANG"
assert has_m3, "create_block CANNOT BUILD Mamba3 -- this is PyPI, not git main"
assert n_dyn == 4, "T.dynamic MISSING -- the recompile storm will return"
print("IMAGE GOOD")
PYEOF

WORKDIR /workspace
CMD ["/bin/bash"]
