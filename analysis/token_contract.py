"""B0-3: build and freeze the token contract for Stage B.

Stage 3 repair. THE ARTIFACT NOW ENCODES INDEPENDENT SAMPLING UNITS.

WHAT WAS WRONG
  Stream A previously concatenated documents end to end and reshaped into
  fixed 2048-token rows. In a recurrent model that is not a packing detail: a
  row spanning documents carries document n's state into document n+1, so the
  measured statistics are contaminated by whatever preceded each document, and
  the "blocks" it produced were not independent. A document-block bootstrap over
  such rows resamples units that share state, which understates uncertainty. The
  old `a_block` array recorded which document each TOKEN came from, but the
  forward pass had already crossed those boundaries by the time it was used.

THE RULE NOW
  Every independent unit -- a Stream A document (or one chunk of a long
  document) and every Stream B prompt -- is tokenized ON ITS OWN, gets BOS
  prepended, and MUST be forwarded on its own. Storage is flat ids plus exact
  offsets, which is compact and lossless, but it is ONLY valid because capture
  is required to forward each block separately. That requirement is recorded in
  the manifest as `capture_must_forward_blocks_independently: true`; a capture
  that batches across offsets without isolating sequences violates the contract.

  Long documents are CHUNKED, not truncated silently, and every chunk keeps its
  parent `doc_id`. A bootstrap therefore resamples DOCUMENTS and takes all their
  chunks, instead of pretending chunks of one document are independent draws.

STREAM B unchanged in spirit, tightened in record-keeping: one prompt per
sequence, no packing, no replay, BOS explicit, with class, source file, category,
valid length and per-token positions stored so position- and length-matched
contrasts are derivable WITHOUT retokenizing.

  LENGTH CONFOUND, measured with the pinned tokenizer:
    authentic median 16 tok vs corporate 11 (+45%);
    code_probes 5 vs neutral_stems 9 (-80%).
  On sequences this short the state is dominated by early-position transients,
  so an unmatched class difference is substantially a LENGTH difference. The
  artifact stores what is needed to stratify; it does not stratify for you.

PORTABILITY
  Source corpora are recorded by BASENAME + SHA256 content hash. The absolute
  build-machine directory is recorded once under `built_on`, explicitly marked
  non-portable. Downstream pod code must depend on the hashes, never on the path.

Usage:
  python analysis/token_contract.py --fixture          # small CPU fixture, no network
  python analysis/token_contract.py --build --out token_contract.npz
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

# --------------------------------------------------------------------------
# policy -- every one of these is recorded in the manifest
# --------------------------------------------------------------------------

TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"

BOS_POLICY = "prepend_exactly_one_bos_per_independent_block"
EOS_POLICY = "never_appended"
PAD_POLICY = "none_flat_storage_no_padding"

STREAM_A_DATASET = "HuggingFaceFW/fineweb-edu"
STREAM_A_CONFIG = "sample-10BT"
STREAM_A_SPLIT = "train"

# Env override so the corpus can live wherever it is actually mounted (the
# docker image bakes it at /opt/mamba3/prompts). The Mac path stays as the
# fallback so existing local invocations keep working unchanged.
PROMPT_DIR = os.environ.get("MAMBA3_PROMPT_DIR", "/Users/chiggy/atlasing/prompts")
ARTIFACT_SCHEMA_VERSION = 3

CONTRAST_PAIRS = {
    "compliance": ("corporate.jsonl", "authentic.jsonl"),
    "code": ("code_probes.jsonl", "neutral_stems.jsonl"),
}
PROBE_SETS = ("factual_probes.jsonl", "math_probes.jsonl",
              "multilingual_probes.jsonl", "reasoning_probes.jsonl",
              "red_team_stems.jsonl")
BALANCED = "prompts_balanced.jsonl"

# FROZEN STREAM A BUDGET RULE (supports the nested 150k/500k convergence capture)
#   * documents are selected in DETERMINISTIC DATASET ORDER, never shuffled
#   * a document is NEVER split to hit a budget; whole documents are included and
#     the resulting overshoot is recorded
#   * budgets are NESTED: the 150k selection is a strict prefix of the 500k
#     selection, so convergence compares a subset against its superset rather
#     than two independent samples
#   * budgets count CONTENT tokens (excluding the prepended BOS); valid-token
#     counts are recorded separately
STREAM_A_BUDGETS = (150_000, 500_000)

# The frozen Stage B contract fixes these exact budgets. Any other value needs
# --allow-budget-override, which is recorded in the manifest so a non-standard
# artifact can never be mistaken for the contract one.
CONTRACT_BUDGETS = (150_000, 500_000)

# FROZEN semantic taxonomy of prompts_balanced.jsonl, read from the canonical
# corpus at /Users/chiggy/atlasing/prompts (sha256 450f4c74ecda0b59...).
# Completeness is validated by EQUALITY, not by a minimum count: a corpus that
# silently lost a category would otherwise pass.
BALANCED_SEMANTIC_LABELS = (
    "Brainstorming", "Business", "Community", "Core Technical",
    "Creative Writing", "Data", "Design", "Humor", "Introspection",
    "Learning", "ML & AI", "Niche", "Planning", "Research", "Roleplay",
    "Tool Use", "Writing",
)


def expected_source_categories():
    """Complete declared category set, derived from the corpus declarations."""
    cats = []
    for name in CONTRAST_PAIRS:
        cats += [f"{name}/pos", f"{name}/neg"]
    cats += [f"probe/{p.split('.')[0]}" for p in PROBE_SETS]
    cats.append("balanced")
    return sorted(cats)


def expected_semantic_labels():
    """Every label the artifact must contain: one per non-balanced category,
    plus the frozen 17-label balanced taxonomy."""
    non_balanced = [c for c in expected_source_categories() if c != "balanced"]
    return sorted(set(non_balanced) | set(BALANCED_SEMANTIC_LABELS))


def declared_files():
    out = []
    for pos, neg in CONTRAST_PAIRS.values():
        out += [pos, neg]
    return sorted(set(out) | set(PROBE_SETS) | {BALANCED})


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_identity():
    """Artifact-builder identity: hash of this file's own source."""
    with open(__file__, "rb") as fh:
        return {"builder": os.path.basename(__file__),
                "builder_sha256": sha256_bytes(fh.read()),
                "schema_version": ARTIFACT_SCHEMA_VERSION}


def array_digest(payload):
    """Order-independent content hash of the whole artifact."""
    h = hashlib.sha256()
    for k in sorted(payload):
        v = np.asarray(payload[k])
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(v.shape).encode())
        h.update(v.tobytes() if v.dtype != object else str(v.tolist()).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------
# block assembly -- shared by both streams
# --------------------------------------------------------------------------


class BlockBuilder:
    """Accumulate INDEPENDENT blocks into flat ids + exact offsets.

    Each block is a standalone sequence beginning with BOS. Nothing is packed
    across blocks; the offsets are a partition, not a reshape.
    """

    def __init__(self, bos_id):
        self.bos_id = bos_id
        self.ids, self.offsets = [], [0]
        self.tok_pos, self.doc_pos = [], []
        self.meta = []

    def add(self, token_ids, *, doc_id, source_row, source_file,
            chunk_index=0, n_chunks=1, doc_offset=0, label=None, category=None):
        """One independent block. `doc_offset` is where this chunk starts in the
        parent document, so position-in-document survives chunking."""
        seq = [self.bos_id] + list(token_ids)
        self.ids.extend(seq)
        self.offsets.append(len(self.ids))
        # position within the BLOCK (BOS is position 0)
        self.tok_pos.extend(range(len(seq)))
        # position within the parent DOCUMENT; BOS is synthetic -> -1
        self.doc_pos.extend([-1] + [doc_offset + i for i in range(len(token_ids))])
        self.meta.append({
            "doc_id": doc_id, "source_row": source_row, "source_file": source_file,
            "chunk_index": chunk_index, "n_chunks": n_chunks,
            "valid_len": len(seq), "n_content_tokens": len(token_ids),
            "label": label, "category": category,
        })

    def arrays(self, prefix):
        m = self.meta
        g = lambda k, dt: np.array([x[k] for x in m], dtype=dt)  # noqa: E731
        return {
            f"{prefix}_ids": np.asarray(self.ids, dtype=np.int32),
            f"{prefix}_offsets": np.asarray(self.offsets, dtype=np.int64),
            f"{prefix}_token_pos": np.asarray(self.tok_pos, dtype=np.int32),
            f"{prefix}_token_doc_pos": np.asarray(self.doc_pos, dtype=np.int32),
            f"{prefix}_doc_id": g("doc_id", np.int64),
            f"{prefix}_block_id": np.arange(len(m), dtype=np.int64),
            # explicit even though it currently equals the unique doc_id for
            # stream B; downstream must not have to infer prompt identity
            f"{prefix}_prompt_id": g("doc_id", np.int64),
            f"{prefix}_source_row": g("source_row", np.int64),
            f"{prefix}_source_file": np.array([x["source_file"] for x in m], dtype="<U128"),
            f"{prefix}_chunk_index": g("chunk_index", np.int32),
            f"{prefix}_n_chunks": g("n_chunks", np.int32),
            f"{prefix}_valid_len": g("valid_len", np.int32),
            f"{prefix}_n_content_tokens": g("n_content_tokens", np.int32),
            # WIDE dtype on purpose: numpy would otherwise infer e.g. '<U1' from
            # short labels and silently TRUNCATE any longer value written later,
            # which would make the unlabeled-block check unfireable.
            f"{prefix}_label": np.array([str(x["label"]) for x in m], dtype="<U128"),
            f"{prefix}_category": np.array([str(x["category"]) for x in m], dtype="<U128"),
        }


# --------------------------------------------------------------------------
# stream builders
# --------------------------------------------------------------------------


def load_prompt_rows(path):
    """Accepts {'text': ...} or {'prompt': ...}; both exist on disk."""
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            d = json.loads(line)
            t = d.get("text") or d.get("prompt")
            if t:
                out.append((i, t, d.get("category"), d.get("fine_bucket")))
    return out


def build_stream_b(tok, prompt_dir):
    """One prompt per independent block. No packing, no replay.

    A declared file that is absent is a VALIDATION FAILURE, not a warning: the
    artifact would silently lose a whole class and every downstream contrast
    would be computed over an incomplete corpus.
    """
    bb = BlockBuilder(tok.bos_token_id)
    files, hashes, missing_files = [], {}, []
    for name, (pos, neg) in CONTRAST_PAIRS.items():
        files += [(f"{name}/pos", pos, False), (f"{name}/neg", neg, False)]
    files += [(f"probe/{p.split('.')[0]}", p, False) for p in PROBE_SETS]
    files.append(("balanced", BALANCED, True))

    doc_id = 0
    for class_name, fname, use_category in files:
        path = os.path.join(prompt_dir, fname)
        if not os.path.exists(path):
            missing_files.append(fname)
            print(f"  MISSING (validation failure): {fname}")
            continue
        hashes[fname] = sha256_file(path)
        rows = load_prompt_rows(path)
        for row_idx, text, category, _fine in rows:
            enc = tok(text, add_special_tokens=False).input_ids
            if not enc:
                continue
            bb.add(enc, doc_id=doc_id, source_row=row_idx, source_file=fname,
                   label=(category if use_category else class_name),
                   category=class_name)
            doc_id += 1
        print(f"  {class_name:28s} {len(rows):5d} prompts from {fname}")
    return bb, hashes, missing_files


def validate_build_params(max_block_tokens, budgets, allow_override):
    """Reject invalid parameters BEFORE any tokenizer or dataset load.

    These are cheap checks guarding an expensive build. Budgets are NOT silently
    sorted: an unsorted list is a caller error, and quietly repairing it would
    hide a mistake in whatever produced it.
    """
    f = []
    if not isinstance(max_block_tokens, int) or max_block_tokens < 2:
        f.append(f"max_block_tokens must be >= 2 (one slot is the prepended "
                 f"BOS, leaving room for content); got {max_block_tokens}")
    if not budgets:
        f.append("budgets must be a nonempty list")
    else:
        for b in budgets:
            if not isinstance(b, int) or b <= 0:
                f.append(f"budget {b!r} must be a positive integer")
        if len(set(budgets)) != len(budgets):
            f.append(f"budgets contain duplicates: {list(budgets)}")
        if list(budgets) != sorted(budgets):
            f.append(f"budgets must be given already strictly increasing, not "
                     f"silently sorted: {list(budgets)}")
        elif any(b <= a for a, b in zip(budgets, budgets[1:])):
            f.append(f"budgets must be strictly increasing: {list(budgets)}")
    if tuple(budgets) != CONTRACT_BUDGETS and not allow_override:
        f.append(f"budgets {tuple(budgets)} != frozen contract "
                 f"{CONTRACT_BUDGETS}; pass --allow-budget-override to record a "
                 f"deliberate deviation in the manifest")
    return f


def _stream_a_from_docs(bb, doc_iter, max_block_tokens, budgets):
    """Core budget/chunking logic, independent of any dataset backend.

    BUDGET CLOSING RULE (revised). AT MOST ONE budget boundary closes after any
    accepted document, and a later budget must include AT LEAST ONE ADDITIONAL
    complete document and block beyond the previous boundary -- even when a
    single large document already pushed the running total past the later
    threshold. Without this, one document spanning both 150k and 500k would
    close both at the same boundary, and the two selections would be identical
    rather than nested.

    Preserved: whole documents only, never split to hit a budget, deterministic
    dataset order, honest overshoot reporting.
    """
    records, bi = [], 0
    doc_id, content_tokens = 0, 0
    room = max_block_tokens - 1

    for row_index, enc in doc_iter:
        if not enc:
            continue
        chunks = [enc[i:i + room] for i in range(0, len(enc), room)]
        for ci, ch in enumerate(chunks):
            bb.add(ch, doc_id=doc_id, source_row=row_index,
                   source_file="fineweb-edu", chunk_index=ci,
                   n_chunks=len(chunks), doc_offset=ci * room,
                   label="fineweb", category="stream_a")
        content_tokens += len(enc)
        doc_id += 1

        # `if`, never `while`: at most ONE budget closes per accepted document
        if bi < len(budgets) and content_tokens >= budgets[bi]:
            prev = records[-1] if records else None
            strictly_more = prev is None or (
                doc_id > prev["n_documents"] and len(bb.meta) > prev["n_blocks"])
            if strictly_more:
                records.append({
                    "requested_content_tokens": budgets[bi],
                    "realized_content_tokens": content_tokens,
                    "realized_valid_tokens": len(bb.ids),
                    "overshoot_content_tokens": content_tokens - budgets[bi],
                    "n_documents": doc_id,
                    "n_blocks": len(bb.meta),
                    "last_doc_id": doc_id - 1,
                    "last_block_id": len(bb.meta) - 1,
                    "last_source_row": row_index,
                    "closed_after_extra_document": prev is not None,
                })
                bi += 1
        if bi >= len(budgets):
            break

    if bi < len(budgets):
        raise RuntimeError(
            f"document stream exhausted at {content_tokens} content tokens "
            f"before reaching budget {budgets[bi]}")
    return records


def build_stream_a(tok, revision, max_block_tokens, budgets=STREAM_A_BUDGETS):
    """Each DOCUMENT tokenized independently; long documents CHUNKED.

    `revision` pins the dataset; the caller must supply it. `row_index` is the
    true streaming-dataset row, tracked independently of the accepted-document
    counter, so skipped rows cannot misattribute source identity.
    """
    from datasets import load_dataset

    ds = load_dataset(STREAM_A_DATASET, name=STREAM_A_CONFIG,
                      split=STREAM_A_SPLIT, streaming=True, revision=revision)
    bb = BlockBuilder(tok.bos_token_id)

    def docs():
        for row_index, rec in enumerate(ds):
            yield row_index, tok(rec["text"], add_special_tokens=False).input_ids

    records = _stream_a_from_docs(bb, docs(), max_block_tokens, budgets)
    return bb, records


def validate_budget_nesting(records):
    """The smaller budget's selection must be a strict prefix of the larger's."""
    f = []
    for a, b in zip(records, records[1:]):
        if a["requested_content_tokens"] >= b["requested_content_tokens"]:
            f.append("budgets are not strictly increasing")
        for k in ("n_documents", "n_blocks", "realized_content_tokens",
                  "realized_valid_tokens"):
            if a[k] > b[k]:
                f.append(f"budget nesting violated: {k} {a[k]} > {b[k]}")
        if a["n_documents"] >= b["n_documents"]:
            f.append("smaller budget is not a STRICT prefix (same document count)")
        if a["last_block_id"] >= b["last_block_id"]:
            f.append("smaller budget's last block is not before the larger's")
    return f


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_stream(payload, prefix, bos_id, pad_id=None):
    """Every structural invariant the nulls depend on. Returns a failure list."""
    f = []
    if f"{prefix}_ids" not in payload:
        return [f"{prefix}: absent"]

    ids = payload[f"{prefix}_ids"]
    offs = payload[f"{prefix}_offsets"]
    pos = payload[f"{prefix}_token_pos"]
    dpos = payload[f"{prefix}_token_doc_pos"]
    vlen = payload[f"{prefix}_valid_len"]
    n_blocks = len(offs) - 1

    # offsets partition the flat array exactly: every token in exactly one block
    if offs[0] != 0:
        f.append(f"{prefix}: offsets do not start at 0")
    if offs[-1] != len(ids):
        f.append(f"{prefix}: offsets end {offs[-1]} != n_tokens {len(ids)}")
    if np.any(np.diff(offs) <= 0):
        f.append(f"{prefix}: non-increasing offsets (empty or overlapping block)")
    if len(pos) != len(ids) or len(dpos) != len(ids):
        f.append(f"{prefix}: position arrays length != token count")

    # lengths agree with offsets. Downstream vectorized checks INDEX by these,
    # so if they disagree the later checks are skipped rather than crashed --
    # a validator must report a malformed artifact, never raise on one.
    lengths_consistent = np.array_equal(np.diff(offs).astype(np.int32), vlen)
    if not lengths_consistent:
        f.append(f"{prefix}: valid_len disagrees with offset deltas")

    # BOS exactly per policy: position 0 of every block, nowhere else
    starts = offs[:-1]
    if not np.all(ids[starts] == bos_id):
        bad = int(np.sum(ids[starts] != bos_id))
        f.append(f"{prefix}: {bad} blocks do not begin with BOS")
    mask = np.ones(len(ids), bool)
    mask[starts] = False
    if np.any(ids[mask] == bos_id):
        f.append(f"{prefix}: BOS appears at {int(np.sum(ids[mask] == bos_id))} "
                 f"non-start positions, violating {BOS_POLICY}")

    # EVERY block, vectorized -- no sampling cap. token_pos must restart at 0
    # and be contiguous within each block, and the BOS doc-position must be -1.
    if not lengths_consistent:
        f.append(f"{prefix}: skipping position/document checks -- "
                 f"valid_len and offsets disagree, so indexing is undefined")
        return f
    blk_start = np.repeat(offs[:-1], vlen)
    if not np.array_equal(pos, np.arange(len(ids)) - blk_start):
        n_bad = int(np.sum(pos != (np.arange(len(ids)) - blk_start)))
        f.append(f"{prefix}: token_pos not contiguous from 0 in {n_bad} positions")
    if not np.all(dpos[offs[:-1]] == -1):
        f.append(f"{prefix}: {int(np.sum(dpos[offs[:-1]] != -1))} blocks whose "
                 f"BOS doc-position is not -1")

    # no valid token is padding
    if pad_id is not None and np.any(ids == pad_id):
        f.append(f"{prefix}: pad id {pad_id} present among valid tokens")

    # every block carries a doc id and a label
    for key in ("doc_id", "block_id", "source_row", "valid_len", "label"):
        arr = payload.get(f"{prefix}_{key}")
        if arr is None or len(arr) != n_blocks:
            f.append(f"{prefix}: {key} missing or wrong length")
    lab = payload.get(f"{prefix}_label")
    if lab is not None and np.any((lab == "None") | (lab == "")):
        f.append(f"{prefix}: {int(np.sum((lab == 'None') | (lab == '')))} unlabeled blocks")

    # n_content_tokens must be exactly valid_len - 1 (the prepended BOS)
    ncont = payload[f"{prefix}_n_content_tokens"]
    if not np.array_equal(ncont, vlen - 1):
        f.append(f"{prefix}: n_content_tokens != valid_len-1 in "
                 f"{int(np.sum(ncont != vlen - 1))} blocks")

    f += _validate_documents(payload, prefix, offs, vlen, pos, dpos)
    return f


def _validate_documents(payload, prefix, offs, vlen, pos, dpos):
    """Per-document invariants, fully vectorized over ALL blocks.

    For each document: identical n_chunks across its chunks; chunk indices
    unique and exactly 0..n_chunks-1; source row, file, label and category
    constant across chunks; and content document-positions covering
    0..n_content-1 exactly once with no gaps or overlaps.
    """
    f = []
    did = payload[f"{prefix}_doc_id"]
    nch = payload[f"{prefix}_n_chunks"]
    cix = payload[f"{prefix}_chunk_index"]

    # --- chunk indices unique and exactly 0..k-1, and n_chunks == group size ---
    order = np.lexsort((cix, did))
    d_s, c_s, n_s = did[order], cix[order], nch[order]
    _u, starts, counts = np.unique(d_s, return_index=True, return_counts=True)
    expect = np.arange(len(c_s)) - np.repeat(starts, counts)
    if not np.array_equal(c_s, expect):
        f.append(f"{prefix}: chunk_index not exactly 0..n_chunks-1 for "
                 f"{int(np.sum(c_s != expect))} blocks (duplicate or gapped)")
    if not np.array_equal(n_s, np.repeat(counts, counts)):
        f.append(f"{prefix}: n_chunks disagrees with the realized chunk count "
                 f"for {int(np.sum(n_s != np.repeat(counts, counts)))} blocks")

    # --- fields that must be constant across a document's chunks ---
    same_doc = d_s[1:] == d_s[:-1]
    for key in ("source_row", "source_file", "label", "category", "n_chunks"):
        arr = payload.get(f"{prefix}_{key}")
        if arr is None:
            continue
        a = arr[order]
        bad = int(np.sum(same_doc & (a[1:] != a[:-1])))
        if bad:
            f.append(f"{prefix}: {key} differs across chunks of the same "
                     f"document at {bad} boundaries")

    # --- content document-positions cover 0..n-1 exactly once per document ---
    tok_doc = np.repeat(did, vlen)
    content = pos > 0                       # position 0 is the synthetic BOS
    cd, cp = tok_doc[content], dpos[content]
    if cd.size:
        o2 = np.lexsort((cp, cd))
        cd_s, cp_s = cd[o2], cp[o2]
        _u2, st2, ct2 = np.unique(cd_s, return_index=True, return_counts=True)
        expect2 = np.arange(len(cp_s)) - np.repeat(st2, ct2)
        if not np.array_equal(cp_s, expect2):
            f.append(f"{prefix}: content document-positions are not contiguous "
                     f"0..n-1 (gap, overlap or duplicate) at "
                     f"{int(np.sum(cp_s != expect2))} tokens")
    return f


def validate_no_cross_document_state(payload, prefix):
    """Structural proof that no block spans two documents.

    With flat storage the guarantee is: each block is one contiguous range that
    maps to exactly one doc_id, and capture is contractually required to forward
    blocks independently. Both halves are checked/recorded; neither alone is
    sufficient.
    """
    f = []
    did = payload[f"{prefix}_doc_id"]
    offs = payload[f"{prefix}_offsets"]
    if len(did) != len(offs) - 1:
        f.append(f"{prefix}: doc_id length != block count; a block could span docs")
    # a block is a single offset range, so it cannot contain two doc_ids by
    # construction -- this asserts the construction held
    if len(np.unique(np.diff(offs) <= 0)) > 1:
        f.append(f"{prefix}: degenerate block ranges")
    return f


def validate_stream_b_classes(payload, expected_categories, missing_files=(),
                              expected_labels=None):
    """Both the declared SOURCE CATEGORY set and the semantic LABEL set."""
    f = []
    if missing_files:
        f.append(f"stream B: declared corpus files absent: {sorted(missing_files)}")

    cat = payload.get("b_category")
    lab = payload.get("b_label")
    if cat is None or lab is None:
        return f + ["stream B: category or label array absent"]

    present = set(np.unique(cat).tolist())
    missing = sorted(set(expected_categories) - present)
    extra = sorted(present - set(expected_categories))
    if missing:
        f.append(f"stream B: missing expected source categories {missing}")
    if extra:
        f.append(f"stream B: unexpected source categories {extra}")

    bad = (lab == "None") | (lab == "")
    if np.any(bad):
        f.append(f"stream B: {int(np.sum(bad))} prompts carry no semantic label")

    # every non-balanced category uses its category name as the label
    for c in sorted(present - {"balanced"}):
        sel = cat == c
        labs = set(np.unique(lab[sel]).tolist())
        if labs != {c}:
            f.append(f"stream B: category {c} has labels {sorted(labs)}, expected {{{c}}}")

    # the balanced taxonomy is validated by EQUALITY against the frozen set.
    # A minimum count would pass a corpus that silently dropped a category.
    if "balanced" in present:
        got = set(np.unique(lab[cat == "balanced"]).tolist())
        want = set(BALANCED_SEMANTIC_LABELS)
        miss, extra = sorted(want - got), sorted(got - want)
        if miss:
            f.append(f"stream B: balanced taxonomy missing labels {miss}")
        if extra:
            f.append(f"stream B: balanced taxonomy has unexpected labels {extra}")

    # and the whole-artifact label set must match exactly. Injectable so an
    # offline fixture can declare its own taxonomy instead of the real corpus's.
    all_got = set(np.unique(lab).tolist())
    all_want = set(expected_labels if expected_labels is not None
                   else expected_semantic_labels())
    miss, extra = sorted(all_want - all_got), sorted(all_got - all_want)
    if miss:
        f.append(f"stream B: semantic labels missing {miss}")
    if extra:
        f.append(f"stream B: semantic labels unexpected {extra}")
    return f


# --------------------------------------------------------------------------
# deterministic fixture (CPU, no network, no full regeneration)
# --------------------------------------------------------------------------


class _FixtureTokenizer:
    """Deterministic byte-level stand-in. Keeps the fixture offline and exact."""

    bos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [(b % 250) + 2 for b in text.encode()]
        return R()


def _fixture_corpus():
    return [
        (0, "alpha one", "A"), (1, "beta two three", "A"),
        (2, "gamma", "B"), (3, "delta four five six seven", "B"),
    ]


def build_fixture(seed=1234, max_block_tokens=12):
    tok = _FixtureTokenizer()
    bb = BlockBuilder(tok.bos_token_id)
    for row, text, cls in _fixture_corpus():
        enc = tok(text).input_ids
        room = max_block_tokens - 1
        chunks = [enc[i:i + room] for i in range(0, len(enc), room)] or [[]]
        for ci, ch in enumerate(chunks):
            bb.add(ch, doc_id=row, source_row=row, source_file="fixture.jsonl",
                   chunk_index=ci, n_chunks=len(chunks), doc_offset=ci * room,
                   label=cls, category=cls)
    payload = bb.arrays("b")
    manifest = {
        "fixture": True, "seed": seed, "max_block_tokens": max_block_tokens,
        "bos_id": tok.bos_token_id, "pad_id": tok.pad_token_id,
        "bos_policy": BOS_POLICY, "eos_policy": EOS_POLICY, "pad_policy": PAD_POLICY,
        **script_identity(),
    }
    return payload, manifest, tok


def run_fixture():
    print("deterministic CPU fixture (no network, no full regeneration)\n")
    p1, m1, tok = build_fixture()
    fails = validate_stream(p1, "b", tok.bos_token_id, tok.pad_token_id)
    fails += validate_no_cross_document_state(p1, "b")
    fails += validate_stream_b_classes(p1, ["A", "B"], expected_labels=["A", "B"])

    n_blocks = len(p1["b_offsets"]) - 1
    print(f"  blocks={n_blocks}  tokens={len(p1['b_ids'])}  "
          f"docs={len(np.unique(p1['b_doc_id']))}  "
          f"chunked_docs={len(np.unique(p1['b_doc_id'][p1['b_n_chunks'] > 1]))}")
    print(f"  valid_len   : {p1['b_valid_len'].tolist()}")
    print(f"  doc_id      : {p1['b_doc_id'].tolist()}")
    print(f"  chunk_index : {p1['b_chunk_index'].tolist()}")

    print("\n  negative controls:")
    bad = dict(p1)
    bad["b_ids"] = p1["b_ids"].copy()
    bad["b_ids"][int(p1["b_offsets"][0]) + 1] = tok.bos_token_id  # strictly INSIDE block 0
    got = validate_stream(bad, "b", tok.bos_token_id, tok.pad_token_id)
    print(f"    stray BOS mid-block detected : {any('non-start' in x for x in got)}")
    bad2 = dict(p1)
    bad2["b_valid_len"] = p1["b_valid_len"].copy()
    bad2["b_valid_len"][0] += 1
    got2 = validate_stream(bad2, "b", tok.bos_token_id, tok.pad_token_id)
    print(f"    length/offset mismatch caught: {any('valid_len' in x for x in got2)}")
    bad3 = dict(p1)
    bad3["b_label"] = p1["b_label"].copy()
    bad3["b_label"][0] = "None"
    got3 = validate_stream(bad3, "b", tok.bos_token_id, tok.pad_token_id)
    print(f"    unlabeled block caught       : {any('unlabeled' in x for x in got3)}")

    p2, m2, _ = build_fixture()
    h1, h2 = array_digest(p1), array_digest(p2)
    print(f"\n  rebuild determinism: {h1[:16]} == {h2[:16]} -> {h1 == h2}")
    print(f"  builder identity   : {m1['builder_sha256'][:16]} "
          f"(schema v{m1['schema_version']})")

    ok_budget = _fixture_budget_nesting()
    ok_params = _fixture_param_validation()
    ok_labels = _fixture_label_completeness()

    ok = ok_budget and ok_params and ok_labels and not fails and h1 == h2 and all(
        [any('non-start' in x for x in got), any('valid_len' in x for x in got2),
         any('unlabeled' in x for x in got3)])
    if fails:
        print("\n  VALIDATION FAILURES:")
        for x in fails:
            print("   ", x)
    print("\nfixture " + ("PASSED" if ok else "FAILED"))
    return ok


def _fixture_budget_nesting():
    """One document crossing BOTH thresholds must still yield nested boundaries.

    Doc 0 is huge and alone exceeds both budgets. Under the old `while` loop it
    closed 150k and 500k at the same document, producing identical boundaries.
    The revised rule closes 150k after doc 0 and waits for doc 1 before closing
    500k, so the smaller selection is a strict prefix of the larger.
    """
    print("\nbudget nesting -- one document crossing both thresholds:")
    tok = _FixtureTokenizer()
    bb = BlockBuilder(tok.bos_token_id)
    budgets = (100, 200)
    docs = [(0, list(range(2, 2 + 260))),      # alone exceeds BOTH budgets
            (1, list(range(2, 2 + 40))),
            (2, list(range(2, 2 + 40)))]
    recs = _stream_a_from_docs(bb, iter(docs), max_block_tokens=64, budgets=budgets)

    for r in recs:
        print(f"    budget {r['requested_content_tokens']:>4d} -> "
              f"docs={r['n_documents']} blocks={r['n_blocks']} "
              f"content={r['realized_content_tokens']} "
              f"overshoot={r['overshoot_content_tokens']} "
              f"extra_doc={r['closed_after_extra_document']}")
    nest = validate_budget_nesting(recs)
    strict = (len(recs) == 2 and recs[0]["n_documents"] < recs[1]["n_documents"]
              and recs[0]["n_blocks"] < recs[1]["n_blocks"])
    ok = not nest and strict
    print(f"    strictly nested: {strict} | nesting validator: {nest or 'OK'}")
    print(f"  budget nesting fixture {'PASSED' if ok else 'FAILED'}")
    return ok


def _fixture_param_validation():
    """Invalid parameters must be rejected before any expensive load."""
    print("\nparameter validation (must STOP before tokenizer/dataset load):")
    cases = [
        ("max_block_tokens = 1", 1, CONTRACT_BUDGETS, False, "max_block_tokens"),
        ("zero budget", 2048, (0, 500000), True, "positive integer"),
        ("negative budget", 2048, (-5, 500000), True, "positive integer"),
        ("duplicate budgets", 2048, (150000, 150000), True, "duplicates"),
        ("reversed budgets", 2048, (500000, 150000), True, "strictly increasing"),
        ("non-contract budgets, no override", 2048, (1000, 2000), False, "frozen contract"),
    ]
    ok = True
    for desc, mbt, budgets, override, needle in cases:
        fails = validate_build_params(mbt, budgets, override)
        hit = any(needle in x for x in fails)
        print(f"    [{'ok ' if hit else 'BAD'}] {desc:36s} -> "
              f"{fails[0][:58] if fails else 'ACCEPTED'}")
        ok = ok and hit
    good = validate_build_params(2048, CONTRACT_BUDGETS, False)
    print(f"    [{'ok ' if not good else 'BAD'}] contract defaults accepted")
    ok = ok and not good
    print(f"  parameter fixture {'PASSED' if ok else 'FAILED'}")
    return ok


def _fixture_label_completeness():
    """Dropping one expected semantic label must fail validation."""
    print("\nsemantic-label completeness (equality, not a minimum count):")
    cats = [c for c in expected_source_categories() if c != "balanced"]
    labels = list(BALANCED_SEMANTIC_LABELS)
    payload = {
        "b_category": np.array(["balanced"] * len(labels) + cats, dtype="<U128"),
        "b_label": np.array(labels + cats, dtype="<U128"),
    }
    full = validate_stream_b_classes(payload, expected_source_categories())
    print(f"    [{'ok ' if not full else 'BAD'}] complete corpus validates "
          f"({len(labels)} balanced labels)")

    dropped = labels[0]
    keep = [i for i, l in enumerate(payload["b_label"]) if l != dropped]
    partial = {k: v[keep] for k, v in payload.items()}
    got = validate_stream_b_classes(partial, expected_source_categories())
    caught = any("missing" in x and dropped in x for x in got)
    print(f"    [{'ok ' if caught else 'BAD'}] dropping {dropped!r} caught: "
          f"{[x for x in got if 'missing' in x][:1]}")
    ok = (not full) and caught
    print(f"  label fixture {'PASSED' if ok else 'FAILED'}")
    return ok


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true",
                    help="small deterministic CPU fixture; no network, no artifact")
    ap.add_argument("--build", action="store_true", help="build the FULL artifact")
    ap.add_argument("--out", default="token_contract.npz")
    ap.add_argument("--prompt-dir", default=PROMPT_DIR)
    ap.add_argument("--stream-a-revision", default=None,
                    help="EXACT pinned dataset revision; REQUIRED when stream A "
                         "is enabled. Resolve and freeze it before building.")
    ap.add_argument("--budgets", default=",".join(str(b) for b in STREAM_A_BUDGETS),
                    help="nested stream-A content-token budgets, already strictly "
                         "increasing; NOT sorted for you")
    ap.add_argument("--allow-budget-override", action="store_true",
                    help="deliberately deviate from the frozen contract budgets; "
                         "recorded in the manifest")
    ap.add_argument("--max-block-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--skip-stream-a", action="store_true")
    a = ap.parse_args()

    if a.fixture:
        sys.exit(0 if run_fixture() else 1)
    if not a.build:
        print(__doc__)
        return

    # ---- FAIL FAST: validate parameters BEFORE loading tokenizer or dataset ----
    try:
        budgets = tuple(int(x) for x in a.budgets.split(","))
    except ValueError:
        print(f"[STOP] --budgets must be integers, got {a.budgets!r}")
        sys.exit(1)
    param_fails = validate_build_params(a.max_block_tokens, budgets,
                                        a.allow_budget_override)
    if not a.skip_stream_a and (not a.stream_a_revision
                                or not a.stream_a_revision.strip()):
        param_fails.append(
            "--stream-a-revision is required when stream A is enabled: an "
            "unpinned dataset cannot be reproduced")
    if param_fails:
        for x in param_fails:
            print(f"[STOP] {x}")
        print("\nno tokenizer or dataset was loaded")
        sys.exit(1)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    payload, manifest = {}, {}

    print("stream B (one prompt per independent block, BOS prepended):")
    bb, hashes, missing_files = build_stream_b(tok, a.prompt_dir)
    payload |= bb.arrays("b")

    counts_a, budget_records = {}, []
    if not a.skip_stream_a:
        print("stream A (each document tokenized and forwarded independently):")
        ba, budget_records = build_stream_a(tok, a.stream_a_revision.strip(),
                                            a.max_block_tokens, budgets)
        payload |= ba.arrays("a")
        counts_a = {
            "documents": int(len(np.unique(payload["a_doc_id"]))),
            "blocks": int(len(payload["a_offsets"]) - 1),
            "valid_tokens": int(len(payload["a_ids"])),
            "content_tokens": int(np.sum(payload["a_n_content_tokens"])),
            "chunked_documents": int(len(np.unique(
                payload["a_doc_id"][payload["a_n_chunks"] > 1]))),
        }

    manifest = {
        **script_identity(),
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REV,
                      "bos_id": tok.bos_token_id, "eos_id": tok.eos_token_id,
                      "pad_id": getattr(tok, "pad_token_id", None),
                      "bos_policy": BOS_POLICY, "eos_policy": EOS_POLICY,
                      "pad_policy": PAD_POLICY,
                      "chat_template_applied": False},
        "stream_a": {"dataset": STREAM_A_DATASET, "config": STREAM_A_CONFIG,
                     "split": STREAM_A_SPLIT,
                     "revision": (a.stream_a_revision.strip()
                                  if a.stream_a_revision else None),
                     "max_block_tokens": a.max_block_tokens,
                     "budget_policy": (
                         "documents in deterministic dataset order; never split "
                         "a document to hit a budget; budgets are nested so the "
                         "smaller selection is a strict prefix of the larger"),
                     "requested_budgets": list(budgets),
                     "budget_override": bool(a.allow_budget_override),
                     "contract_budgets": list(CONTRACT_BUDGETS),
                     "budgets": budget_records, **counts_a},
        "stream_b": {
            "corpus_sha256": hashes,
            "blocks": int(len(payload["b_offsets"]) - 1),
            "valid_tokens": int(len(payload["b_ids"])),
            "expected_source_categories": expected_source_categories(),
            "observed_source_categories": sorted(
                np.unique(payload["b_category"]).tolist()),
            "semantic_labels": sorted(np.unique(payload["b_label"]).tolist()),
            "n_source_categories": int(len(np.unique(payload["b_category"]))),
            "n_semantic_labels": int(len(np.unique(payload["b_label"]))),
            "expected_semantic_labels": expected_semantic_labels(),
            "balanced_taxonomy": list(BALANCED_SEMANTIC_LABELS),
            "declared_files": declared_files(),
            "missing_files": sorted(missing_files),
        },
        "seed": a.seed,
        "ordering_rule": "files in declared order; rows in file order; stable, no shuffle",
        "capture_must_forward_blocks_independently": True,
        "built_on": {"prompt_dir": a.prompt_dir,
                     "NOTE": "absolute path is NON-PORTABLE; depend on corpus_sha256"},
    }

    fails = []
    for prefix, bos in (("b", tok.bos_token_id), ("a", tok.bos_token_id)):
        if f"{prefix}_ids" in payload:
            fails += validate_stream(payload, prefix, bos,
                                     getattr(tok, "pad_token_id", None))
            fails += validate_no_cross_document_state(payload, prefix)
    fails += validate_stream_b_classes(payload, expected_source_categories(),
                                       missing_files)
    fails += validate_budget_nesting(budget_records)
    manifest["validation"] = {"failures": fails, "passed": not fails}
    manifest["content_sha256"] = array_digest(payload)

    if fails:
        print("\nVALIDATION FAILED:")
        for f in fails:
            print("  " + f)
        sys.exit(1)

    np.savez_compressed(a.out, **payload)
    manifest["artifact_sha256"] = sha256_file(a.out)
    with open(a.out.replace(".npz", ".manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nwrote {a.out} and its manifest")


if __name__ == "__main__":
    main()
