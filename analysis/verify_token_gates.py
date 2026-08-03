"""Reconstruction check for the experiment-0b per-token gate emission.

Both artifacts come out of the SAME capture run, from the same tensors, so this
check is decisive: re-reduce the per-token gates using the capture's own role
definition and they must reproduce the block-role summaries. A mismatch means
the emission is wrong. There is no confounding story available.

This exists because two checks earlier in this project reported success while
testing nothing (a None == None comparison, and a manifest lookup against a key
that did not exist). The property that matters is that this check CAN FAIL.

Role definition, replicated exactly from BlockRoleRecorder._segmented_roles:
    bos       token 0                       count 1
    interior  tokens 1 .. n-2               count max(n-2, 0)
    final     token n-1 when n > 1          count 1
When a role is absent the recorder leaves 0 rather than NaN, and so does this.

Tolerance is DECLARED, not tuned. blockrole fields are stored float16 while the
emission is float32, so exact agreement is impossible; float16 has ~4.9e-4
relative resolution and these are means over up to hundreds of tokens.

usage:
    python verify_token_gates.py --capture capture_x.npz --token-gates gates_x.npz
"""

import argparse
import sys

import numpy as np

# Declared before any data is loaded. Do not tune this to make a run pass.
#
# The tolerance is expressed in float16 UNITS IN THE LAST PLACE, not as a
# relative fraction, because the thing being compared is a float64
# reconstruction against a value that was stored in float16. One ULP is the
# smallest difference float16 can represent at a given magnitude; agreement
# closer than that is not merely good, it is the best the format allows.
#
# A plain rtol/atol pair is the wrong instrument here and was the first
# version's mistake: relative error is meaningless as the reference approaches
# zero, so near-zero cells report enormous relative error while being off by a
# fraction of a representable step. Experiment 0a already established that for
# this exact data (fp16 subnormal analysis, 2026-08-03) and the check should
# have been written with it in mind.
#
# Budget: 1 ULP of float16 storage rounding, plus headroom for the capture
# accumulating its mean in float32 on device while this rebuild accumulates in
# float64. The float32 term is ~sqrt(n)*eps32*|x| ~ 1e-6 relative for the
# longest interior runs, three orders below one float16 ULP (~9.8e-4 relative),
# so it is absorbed rather than modelled.
ULP_TOL = 2.0

ROLES = ("bos", "interior", "final")


def fp16_ulp(x):
    """Spacing of float16 at |x|. At x == 0 this is the smallest subnormal."""
    return np.spacing(np.abs(np.asarray(x, dtype=np.float64))
                      .astype(np.float16)).astype(np.float64)


def rebuild_roles(tok, lens, n_blocks, rows):
    """Per-token (T, L, H) -> (block, role, L, H) using the capture's own rule."""
    L, H = tok.shape[1], tok.shape[2]
    out = np.zeros((n_blocks, len(ROLES), L, H), dtype=np.float64)
    counts = np.zeros((n_blocks, len(ROLES)), dtype=np.int64)
    start = 0
    for row, n in zip(rows, lens):
        seg = tok[start:start + n].astype(np.float64)     # (n, L, H)
        out[row, 0] = seg[0]                              # bos
        counts[row, 0] = 1
        if n > 1:
            out[row, 2] = seg[n - 1]                      # final
            counts[row, 2] = 1
        n_int = max(n - 2, 0)
        if n_int > 0:
            out[row, 1] = seg[1:n - 1].mean(axis=0)       # interior
        counts[row, 1] = n_int
        start += n
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True,
                    help="primary Stage B capture npz")
    ap.add_argument("--token-gates", required=True,
                    help="artifact written by --emit-token-gates")
    ap.add_argument("--fields", default=None,
                    help="comma list; default is every emitted field")
    args = ap.parse_args()

    cap = np.load(args.capture, allow_pickle=False)
    tg = np.load(args.token_gates, allow_pickle=False)

    rows = tg["token_block_row"]
    lens = tg["token_block_len"]
    emitted = [str(x) for x in tg["token_fields"]]
    fields = ([f.strip() for f in args.fields.split(",")]
              if args.fields else emitted)

    print(f"capture      {args.capture}")
    print(f"token gates  {args.token_gates}")
    print(f"emitted fields {emitted}")
    print(f"checking       {fields}")
    print(f"declared tolerance  {ULP_TOL:g} float16 ULP "
          f"(the smallest difference the storage format can represent)")

    probe = tg[f"token|{fields[0]}"]
    T, L, H = probe.shape
    n_blocks = int(cap["blockrole|Delta"].shape[0])
    print(f"\ntokens={T:,}  layers={L}  heads={H}  blocks={n_blocks}")

    # --- structural checks, before any numeric comparison -------------------
    problems = []
    if int(lens.sum()) != T:
        problems.append(f"token_block_len sums to {int(lens.sum())} but the "
                        f"token axis is {T}")
    if rows.size != lens.size:
        problems.append(f"{rows.size} block rows vs {lens.size} lengths")
    if np.unique(rows).size != rows.size:
        problems.append("duplicate block rows in the emission")
    cap_valid = cap["block_valid_len"]
    mism = [(int(r), int(n), int(cap_valid[r]))
            for r, n in zip(rows, lens) if int(cap_valid[r]) != int(n)]
    if mism:
        problems.append(f"{len(mism)} blocks disagree with block_valid_len, "
                        f"first three: {mism[:3]}")
    if problems:
        print("\nSTRUCTURAL FAILURE")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("structural checks passed "
          "(token axis, block rows, lengths vs block_valid_len)")

    # --- the reconstruction -------------------------------------------------
    failed = []
    print(f"\n{'field':>10} {'role':>10} {'max abs err':>12} {'max ULP':>9} "
          f"{'p99.9 ULP':>10} {'over budget':>12} {'verdict':>8}")
    for f in fields:
        key = f"token|{f}"
        if key not in tg:
            print(f"  {f}: NOT EMITTED, skipping")
            continue
        capkey = f"blockrole|{f}"
        if capkey not in cap:
            print(f"  {f}: no {capkey} in the capture, skipping")
            continue
        rebuilt, counts = rebuild_roles(tg[key], lens, n_blocks, rows)
        # capture layout is (block, layer, role, head); rebuilt is
        # (block, role, layer, head)
        ref = cap[capkey].astype(np.float64).transpose(0, 2, 1, 3)
        for ri, rname in enumerate(ROLES):
            a, b = rebuilt[:, ri], ref[:, ri]
            if rname == "interior":
                keep = counts[:, 1] > 0        # absent interior is 0 on both sides
                a, b = a[keep], b[keep]
            aerr = np.abs(a - b)
            ulp = np.maximum(fp16_ulp(b), np.finfo(np.float16).smallest_subnormal)
            err_ulp = aerr / ulp
            over = err_ulp > ULP_TOL
            close = not over.any()
            print(f"{f:>10} {rname:>10} {aerr.max():>12.3e} {err_ulp.max():>9.2f} "
                  f"{np.percentile(err_ulp, 99.9):>10.2f} {int(over.sum()):>12} "
                  f"{'PASS' if close else 'FAIL':>8}")
            if not close:
                # report the cell that ACTUALLY fails the budget, not the one
                # with the largest absolute error -- those are different cells
                # whenever the reference spans magnitudes, and reporting the
                # wrong one is how a real defect gets argued away
                bad = np.unravel_index(np.argmax(err_ulp), a.shape)
                failed.append(
                    f"{f}/{rname}: worst cell {tuple(int(x) for x in bad)} "
                    f"rebuilt={a[bad]:.8g} capture={b[bad]:.8g} "
                    f"diff={aerr[bad]:.3e} = {err_ulp[bad]:.2f} float16 ULP "
                    f"(budget {ULP_TOL})  [{int(over.sum())} of {over.size} "
                    f"cells over budget]")

    print()
    if failed:
        print("RECONSTRUCTION FAILED -- the emission does not reproduce the "
              "capture's own summaries")
        for x in failed:
            print(f"  - {x}")
        sys.exit(1)
    print("RECONSTRUCTION PASSED -- per-token gates re-reduce to the block-role "
          "summaries within the declared tolerance")
    print("The emitted token axis is therefore the same data the capture "
          "already agreed on, with the ordering retained.")


if __name__ == "__main__":
    main()
