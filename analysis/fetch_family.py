"""Fetch all eight released Mamba-3 checkpoints into the HF cache.

~12.1 GB total (bf16, pytorch_model.bin). Idempotent: hf_hub_download skips
anything already cached, so re-running is cheap and safe.

Weights only, no model construction, so this needs neither mamba_ssm nor a GPU.
"""

import sys

from huggingface_hub import hf_hub_download

SISO_SIZES = ("187m", "443m", "893m", "1.5b")
MIMO_SIZES = ("187m", "444m", "894m", "1.5b")

REPOS = [f"state-spaces/mamba3-siso-{s}" for s in SISO_SIZES] + [
    f"state-spaces/mamba3-mimo-{s}" for s in MIMO_SIZES
]


def main():
    paths = {}
    for repo in REPOS:
        print(f"--> {repo}", flush=True)
        cfg = hf_hub_download(repo, "config.json")
        bin_path = hf_hub_download(repo, "pytorch_model.bin")
        paths[repo] = (cfg, bin_path)
        print(f"    {bin_path}", flush=True)

    print("\nall eight cached:")
    for repo, (_, p) in paths.items():
        print(f"  {repo:36s} {p}")


if __name__ == "__main__":
    sys.exit(main())
