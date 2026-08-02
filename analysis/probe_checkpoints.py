"""Stage 0 / V0 configuration census of the released Mamba-3 family.

No weight downloads. Pulls config.json per repo and reads only the safetensors
HEADER over an HTTP range request, so we learn every tensor name, shape and
dtype for a few hundred KB instead of ~12 GB.

Purposes:
  1. Confirm SISO exposes dt_bias and D with the same per-head shapes the MIMO
     analysis relies on, and record exactly which tensors SISO lacks.
  2. Test the MLP-width confound: the released SISO/MIMO pairs are reported to
     be PARAMETER-matched rather than architecture-identical, with MIMO using
     narrower MLPs to pay for its extra mimo_* tensors. If true, a SISO/MIMO
     difference cannot be attributed to MIMO alone. Measured here from
     d_intermediate and the realised MLP tensor shapes, not from the paper.
  3. Pin provenance: repo commit sha per checkpoint, for run records.

Everything printed here is read off the checkpoints themselves.
"""

import json
import struct
import sys
from collections import OrderedDict

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

SISO_SIZES = ("187m", "443m", "893m", "1.5b")
MIMO_SIZES = ("187m", "444m", "894m", "1.5b")  # repo names are offset by 1m

# per-layer tensors the static atlas depends on
WANT = (
    "dt_bias", "D",
    "mimo_x", "mimo_z", "mimo_o",
    "B_bias", "C_bias",
    "B_norm.weight", "C_norm.weight",
    "in_proj.weight", "out_proj.weight",
)

# MLP tensors, for the confound census
MLP_KEYS = ("mlp.fc1.weight", "mlp.fc2.weight")

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1}


def repos():
    for s in SISO_SIZES:
        yield f"state-spaces/mamba3-siso-{s}"
    for s in MIMO_SIZES:
        yield f"state-spaces/mamba3-mimo-{s}"


def safetensors_header(repo, fname):
    """Read the safetensors JSON header via a range request."""
    url = hf_hub_url(repo, fname)
    head = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=60)
    head.raise_for_status()
    (n,) = struct.unpack("<Q", head.content[:8])
    body = requests.get(url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=120)
    body.raise_for_status()
    return json.loads(body.content)


def numel(shape):
    out = 1
    for d in shape:
        out *= d
    return out


def probe(repo):
    api = HfApi()
    info = api.model_info(repo, files_metadata=False)
    files = [s.rfilename for s in info.siblings]

    with open(hf_hub_download(repo, "config.json")) as fh:
        cfg = json.load(fh)

    st = [f for f in files if f.endswith(".safetensors") and "index" not in f]
    if not st:
        return cfg, {}, info.sha, files

    hdr = safetensors_header(repo, st[0])
    hdr.pop("__metadata__", None)
    return cfg, hdr, info.sha, files


def main():
    rows = []
    for repo in repos():
        print(f"\n=== {repo}")
        try:
            cfg, hdr, sha, files = probe(repo)
        except Exception as e:  # noqa: BLE001 - exploratory probe
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue

        ssm = cfg.get("ssm_cfg", {}) or {}
        total_params = sum(numel(v["shape"]) for v in hdr.values())
        dtypes = sorted({v["dtype"] for v in hdr.values()})

        l0 = OrderedDict(
            (k.split("layers.0.", 1)[-1], v)
            for k, v in hdr.items()
            if k.startswith("backbone.layers.0.")
        )

        mlp = {k: l0[k]["shape"] for k in MLP_KEYS if k in l0}
        d_model = cfg.get("d_model")
        mlp_hidden = mlp.get("mlp.fc2.weight", [None, None])[1]

        print(f"  sha={sha}")
        print(
            f"  n_layer={cfg.get('n_layer')} d_model={d_model} "
            f"params={total_params / 1e6:.1f}M dtype={','.join(dtypes)}"
        )
        print(
            f"  d_state={ssm.get('d_state')} headdim={ssm.get('headdim')} "
            f"ngroups={ssm.get('ngroups')} mimo_rank={ssm.get('mimo_rank')} "
            f"expand={ssm.get('expand')} chunk={ssm.get('chunk_size')}"
        )
        print(
            f"  rope_fraction={ssm.get('rope_fraction')} "
            f"A_floor={ssm.get('A_floor')} d_intermediate={cfg.get('d_intermediate')}"
        )
        print(
            f"  MLP: hidden={mlp_hidden} ratio_to_d_model="
            f"{(mlp_hidden / d_model):.3f}" if mlp_hidden and d_model else "  MLP: ABSENT"
        )

        for want in WANT:
            hits = [(k, v["shape"]) for k, v in l0.items() if k.endswith(want)]
            if hits:
                for k, s in hits:
                    print(f"    {want:16s} {k:30s} {s}")
            else:
                print(f"    {want:16s} ABSENT")

        rows.append(
            {
                "repo": repo,
                "params_M": total_params / 1e6,
                "n_layer": cfg.get("n_layer"),
                "d_model": d_model,
                "mlp_hidden": mlp_hidden,
                "mimo_rank": ssm.get("mimo_rank"),
                "ngroups": ssm.get("ngroups"),
                "headdim": ssm.get("headdim"),
                "d_state": ssm.get("d_state"),
            }
        )

    # the confound table: pair SISO against MIMO at each nominal size
    print("\n\n=== parameter-match / MLP-width confound ===")
    print(f"{'pair':>10s} {'params(M)':>22s} {'n_layer':>14s} {'d_model':>14s} {'mlp_hidden':>16s}")
    by_name = {r["repo"]: r for r in rows}
    for ss, ms in zip(SISO_SIZES, MIMO_SIZES):
        a = by_name.get(f"state-spaces/mamba3-siso-{ss}")
        b = by_name.get(f"state-spaces/mamba3-mimo-{ms}")
        if not (a and b):
            continue
        print(
            f"{ss:>10s} "
            f"{a['params_M']:>10.1f} /{b['params_M']:>10.1f} "
            f"{a['n_layer']:>6} /{b['n_layer']:>6} "
            f"{a['d_model']:>6} /{b['d_model']:>6} "
            f"{str(a['mlp_hidden']):>7s} /{str(b['mlp_hidden']):>7s}"
        )

    with open("checkpoint_census.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\nwrote checkpoint_census.json")


if __name__ == "__main__":
    sys.exit(main())
