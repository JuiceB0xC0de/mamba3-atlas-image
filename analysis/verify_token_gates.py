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

# Declared before any data is loaded. Do not tune these to make a run pass.
RTOL = 2.0e-3     # ~4x float16 relative resolution, allowing for mean accumulation
ATOL = 1.0e-6     # guards genuine zeros (absent roles, exact-zero gates)

ROLES = ("bos", "interior", "final")


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
    print(f"declared tolerance  rtol={RTOL:g}  atol={ATOL:g}")

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
    print(f"\n{'field':>10} {'role':>10} {'max abs err':>13} {'max rel err':>13} "
          f"{'verdict':>9}")
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
            rerr = aerr / np.maximum(np.abs(b), 1e-12)
            close = np.allclose(a, b, rtol=RTOL, atol=ATOL)
            print(f"{f:>10} {rname:>10} {aerr.max():>13.3e} {rerr.max():>13.3e} "
                  f"{'PASS' if close else 'FAIL':>9}")
            if not close:
                bad = np.unravel_index(np.argmax(aerr), a.shape)
                failed.append(f"{f}/{rname}: worst cell {bad} "
                              f"rebuilt={a[bad]:.6g} capture={b[bad]:.6g}")

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
