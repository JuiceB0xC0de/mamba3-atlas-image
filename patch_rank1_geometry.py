"""Short-circuit rank_geometry at R == 1 (the SISO arm).

Not an approximation. At rank 1:
  * the Gram is 1x1, so its single eigenvalue IS its single element
  * mean_r x == x, so the differential part is exactly zero
  * common_norm == total_norm
  * participation == 1 exactly
The existing docstring already states all four. This makes the code do what the
docstring says instead of routing tens of thousands of 1x1 systems through
torch.linalg.eigvalsh on every layer of every pack.

MIMO (R=4) is untouched.
"""
import pathlib
import sys

TARGET = pathlib.Path("/root/mamba3-project/analysis/capture_stage_b.py")

ANCHOR = """    R = x.shape[-2]
    gram = x @ x.transpose(-1, -2)
"""

INSERT = """    R = x.shape[-2]
    gram = x @ x.transpose(-1, -2)
    if R == 1:
        # Closed form, NOT an approximation. A 1x1 Gram has exactly one
        # eigenvalue -- its only element. mean_r x == x, so diff is exactly 0
        # and common == total. participation == 1 by definition.
        # SISO runs at rank 1; this skips a batched eigensolve over tens of
        # thousands of 1x1 systems per layer per pack.
        total = x.norm(dim=(-2, -1))
        zero = torch.zeros_like(total)
        out = {
            "gram": gram,
            "participation": torch.ones_like(total).to(x.dtype),
            "total_norm": total,
            "common_norm": total,
            "diff_norm": zero,
            "diff_over_total": zero,
            "diff_energy": zero,
        }
        if include_eigvals:
            out["eigvals"] = gram[..., 0, :].clamp_min(0).to(x.dtype)
        return out
"""


def main() -> None:
    src = TARGET.read_text()
    if "Closed form, NOT an approximation" in src:
        print("already patched")
        return
    if src.count(ANCHOR) != 1:
        sys.exit(f"FAILED: anchor found {src.count(ANCHOR)} times, expected 1")
    TARGET.write_text(src.replace(ANCHOR, INSERT, 1))
    print("rank-1 short circuit applied")


if __name__ == "__main__":
    main()
