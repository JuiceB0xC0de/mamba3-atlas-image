"""G1-G7: the GPU probe suite. Run this FIRST on a fresh pod, before any capture.

One command, seven gates, ordered cheapest-and-most-blocking first so a failure
stops the meter early. Every gate emits a verdict into a single JSON report,
and after EVERY completed gate that report is re-published locally (atomic) and
re-uploaded to Hugging Face, so evidence survives a later STOP followed by pod
termination.

    python analysis/gpu_probe.py --all --model mimo-187m --revision <COMMIT> \\
        --token-contract token_contract.npz \\
        --hf-repo <HF_DATASET_REPO> --hf-path-prefix mamba3/stage-b/gpu-probe/<RUN_ID>

CRITICAL STORAGE INVARIANT
  RunPod has no persistent storage in this workflow. Every pod path is
  disposable staging: losing or terminating the pod loses every local artifact.
  Every report this file produces must be uploaded to Hugging Face and REMOTELY
  VERIFIED before the pod is terminated. The report carries
  upload_required_before_pod_termination=true and local_storage_persistent=false.
  An archival failure HALTS all later gates immediately (no further GPU spend
  while evidence cannot leave the pod), preserves the scientific verdicts
  separately, and exits nonzero with an explicit do-not-terminate instruction.
  No pod path is persistent.

GATES
  g1 env         versions, GPU, memory/disk, EXACT checkpoint resolution + load
  g2 parity      elementwise mixer parity, delegated to full_model_smoke.py
  g3 step        prefill vs step()/decode boundary, elementwise final logits
  g4 jit         variable-length/JIT policy against the REAL capture contract
  g5 resources   capture-shaped memory/disk gate with the real hook surface
  g6 throughput  contract-derived wall-time estimate for the frozen capture
  g7 precision   derived-quantity float16 STORAGE precision (advisory; never
                 kernel recurrence parity evidence)

VERDICTS
  PASS      proceed. Never unconditional: every PASS is computed from measured
            values against tolerances declared in this file before any GPU run.
  BOUNDARY  scope limitation, not an error. Consequences are enumerated in the
            report: what remains allowed and what is prohibited.
  STOP      correctness or resource failure. Later scientific gates halt; the
            report is still published and archived.

RECURRENCE SEMANTICS ARE FIXED (source-derived, contract 1.3 superseding
correction). This file tests those fixed semantics; it does not choose among
alternatives, and there are no semantic switches left to search:
  * BOTH sides are rotated: C/query at K L246-259 and B/key at K L266-275, each
    by its own cumulative phase.
  * Phase increment is tanh(raw_angle) * pi * Delta, accumulated by an
    INCLUSIVE cumsum plus any carried phase, reduced mod 2*pi (G L94-117).
  * Same-token diagonal uses gamma alone; strictly earlier contributions use
    trap_scale = gamma + shifted_gamma applied to K only.
G2 is therefore a TOLERANCE measurement against full_model_smoke.py, the single
parity authority. A G2 failure authorizes debugging (source revision, precision,
slicing, transcription); it never authorizes switch search or tolerance
widening. Logit top-1/top-5 agreement NEVER determines any parity verdict; it
may appear only as a clearly labelled supplemental diagnostic.

CHECKPOINT RESOLUTION is delegated to mamba3_core.resolve_checkpoint,
LOCAL-ONLY. The kernel model, configuration and reference weights all come from
the same ResolvedCheckpoint: the model loads from ck.path, never a mutable hub
id. ck.provenance() (including resolved_commit) goes into the report; missing
or incomplete provenance is STOP.

Offline self-check (no CUDA, no network, mocked archival):

    python analysis/gpu_probe.py --self-check
"""

import argparse
import hashlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import full_model_smoke as fms  # noqa: E402  single parity authority
from capture_stage_b import (  # noqa: E402
    BLOCK_SUMMARY_DTYPE, EST_N_LABELS, estimate_artifact_bytes, preflight,
)
from mamba3_core import (  # noqa: E402
    CheckpointResolveError, InProjSpec, ShapeContractError, assert_runtime,
    recurrence_quantities, resolve_checkpoint, split_in_proj,
)
from reference_recurrence import phase_details  # noqa: E402

PASS, STOP, BOUNDARY = "PASS", "STOP", "BOUNDARY"

TOKENIZER_ID = "NousResearch/Meta-Llama-3.1-8B"
TOKENIZER_REV = "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"

# the required G2 matrix (deliverable D). Lengths come from full_model_smoke's
# own case_lengths(chunk_size) per model; they are never restated here.
G2_MODELS = "siso-187m,mimo-187m"
G2_LAYERS = "0,5"

# G3/G4-drift/G7 reuse the authority's declared tolerances by IMPORT, so there
# is exactly one place they are defined and calibrated (fms self-check).
TOL = {"cos": fms.DEFAULT_COS_TOL, "rel": fms.DEFAULT_REL_TOL,
       "norm_max": fms.DEFAULT_NORM_MAX_TOL,
       "rel_floor_frac": fms.DEFAULT_REL_FLOOR_FRAC}

# G4 timing classifier thresholds, declared before any GPU run
RECOMPILE_FACTOR = 5.0      # first_s > FACTOR * warmed median -> suspected JIT
RECOMPILE_ABS_S = 1.0       # ... and above this absolute floor
COMPILE_OVERHEAD_MAX_FRAC = 0.5   # projected compile time above this fraction
                                  # of warmed capture time -> BOUNDARY

# G5 disk gate: temporary staging must hold this multiple of the projected
# artifact (artifact + report + upload slack). Staging only, never storage.
DISK_SAFETY_FACTOR = 2.0

# claims prohibited under a G3 BOUNDARY (contract 1.3 failure table)
G3_BOUNDARY_PROHIBITED = (
    "kernel-state parity claims",
    "explicit decode-state / h_t claims",
    "deployed state-trajectory claims",
    "state-based Stage C work (C-3 state half, C-5 step backend)",
)
G3_BOUNDARY_ALLOWED = (
    "prefill-derived Stage B metrics",
    "behavioural probes",
)

STORAGE_FLAGS = {
    "upload_required_before_pod_termination": True,
    "local_storage_persistent": False,
    "note": ("No pod path is persistent. Local output is disposable staging "
             "that exists only long enough to be uploaded and remotely "
             "verified on Hugging Face."),
}


# --------------------------------------------------------------------------
# JSON: orjson through one helper, stdlib fallback so archival can never die
# on a formatting dependency missing from the pod pin set
# --------------------------------------------------------------------------


def _json_bytes(obj) -> bytes:
    try:
        import orjson

        return orjson.dumps(obj, default=str,
                            option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    except ImportError:
        import json

        return json.dumps(obj, indent=2, default=str).encode()


def _json_loads(b):
    try:
        import orjson

        return orjson.loads(b)
    except ImportError:
        import json

        return json.loads(b)


def publish_report(report: dict, out_path: str) -> dict:
    """Atomic local publication: temp file + os.replace, then SHA256 + size.

    An interruption can never leave a partial file at the final path.
    """
    payload = _json_bytes(report)
    tmp = out_path + ".partial"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, out_path)
    return {"local_path": out_path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


# --------------------------------------------------------------------------
# Hugging Face archival: upload after every gate, then VERIFY REMOTELY.
# An upload API success alone is not verification.
# --------------------------------------------------------------------------


class HFArchiver:
    """Uploads files to the declared repo/prefix and verifies existence + byte
    size remotely. The live report overwrites one fixed remote path each time;
    sidecars (e.g. the full G2 authority reports) go next to it under their own
    basenames. HF_TOKEN comes from the environment and is never printed or
    embedded in any output."""

    def __init__(self, repo: str, repo_type: str, path_prefix: str,
                 basename: str):
        self.repo = repo
        self.repo_type = repo_type
        self.prefix = (path_prefix or "").strip("/")
        self.report_basename = basename

    def remote_for(self, basename: str) -> str:
        return f"{self.prefix}/{basename}" if self.prefix else basename

    def upload_and_verify(self, pub: dict, remote_basename: str | None = None) -> dict:
        rname = self.remote_for(remote_basename or self.report_basename)
        res = {"attempted": True, "uploaded": False, "verified": False,
               "remote_repo": f"{self.repo_type}:{self.repo}",
               "remote_path": rname,
               "remote_bytes": None, "local_bytes": pub["bytes"],
               "error": None, "mocked": False}
        token = os.environ.get("HF_TOKEN")
        if not token:
            res["error"] = "HF_TOKEN is not set in the environment"
            return res
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=token)
            api.upload_file(path_or_fileobj=pub["local_path"],
                            path_in_repo=rname,
                            repo_id=self.repo, repo_type=self.repo_type,
                            commit_message=f"gpu_probe: {rname}")
            res["uploaded"] = True
            infos = api.get_paths_info(repo_id=self.repo, paths=[rname],
                                       repo_type=self.repo_type)
            match = [i for i in infos if getattr(i, "path", None) == rname]
            if not match:
                res["error"] = "remote path not found after upload"
                return res
            res["remote_bytes"] = int(getattr(match[0], "size", -1))
            if res["remote_bytes"] != pub["bytes"]:
                res["error"] = (f"remote byte size {res['remote_bytes']} != "
                                f"local {pub['bytes']}")
                return res
            res["verified"] = True
        except Exception as e:  # noqa: BLE001 - archival must degrade to a recorded failure
            res["error"] = f"{type(e).__name__}: {e}"
        return res


class MockArchiver:
    """Self-check stand-in. Records every attempt. verify_ok may be a single
    bool or a SEQUENCE of bools consumed per attempt (last value repeats), so
    sequence-dependent failures, e.g. a finalized upload that fails after a
    successful initial one, are testable offline."""

    def __init__(self, verify_ok=True):
        self.seq = (list(verify_ok)
                    if isinstance(verify_ok, (list, tuple)) else [verify_ok])
        self.attempts = []

    def upload_and_verify(self, pub: dict, remote_basename: str | None = None) -> dict:
        ok = self.seq[min(len(self.attempts), len(self.seq) - 1)]
        rname = "mock/" + (remote_basename
                           or os.path.basename(pub["local_path"]))
        res = {"attempted": True, "uploaded": True, "verified": ok,
               "remote_repo": "mock:none", "remote_path": rname,
               "remote_bytes": pub["bytes"] if ok else -1,
               "local_bytes": pub["bytes"],
               "error": None if ok else "mock remote verification failure",
               "mocked": True}
        self.attempts.append(res)
        return res


def _file_pub(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    return {"local_path": path,
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _attempt_upload(ctx: dict, gate: str, pub: dict,
                    remote_basename: str | None = None) -> dict:
    """One upload + remote verification, recorded in the receipt log.

    A verification failure flips archive_failed, which the main loop treats as
    an immediate halt: no further GPU spend while evidence cannot leave the
    disposable pod. Scientific verdicts are never altered by this.
    """
    archiver = ctx["archiver"]
    if archiver is None:
        entry = {"gate": gate, **pub, "attempted": False, "verified": False,
                 "skipped": ctx["archive_skip_reason"]}
        ctx["archive_log"].append(entry)
        return entry
    res = archiver.upload_and_verify(pub, remote_basename)
    entry = {"gate": gate, **pub, **res}
    ctx["archive_log"].append(entry)
    if not res["verified"]:
        ctx["archive_failed"] = True
        ctx["report"]["archive"]["status"] = "FAILED"
        print("DO NOT TERMINATE POD -- RETRY HUGGING FACE ARCHIVAL.")
        print(f"    archival failure at gate {gate}: {res['error']}")
        print(f"    local report preserved at {pub['local_path']}")
    return entry


def archive_sidecar(ctx: dict, gate: str, path: str) -> dict:
    """Upload a complete evidence file (e.g. a G2 authority report) verbatim,
    next to the live report. The sidecar is never stripped or rewritten."""
    return _attempt_upload(ctx, gate, _file_pub(path), os.path.basename(path))


def archive_now(ctx: dict, gate: str, terminal: bool = False) -> None:
    """Publish the current report atomically, upload it, verify remotely.

    The serialized snapshot carries status UPLOAD_IN_PROGRESS (its own upload
    cannot be inside itself) plus every PRIOR upload receipt. On a terminal
    call, a verified upload is followed by one re-publication + re-upload so
    the remote copy carries the concluded status, the overall verdict and the
    prior receipts. An archival failure leaves the FAILED status on disk.
    """
    report, args = ctx["report"], ctx["args"]
    report["archive"]["log"] = list(ctx["archive_log"])
    if report["archive"]["status"] != "FAILED":
        report["archive"]["status"] = ("UPLOAD_IN_PROGRESS"
                                       if ctx["archiver"] else "SKIPPED")
    pub = publish_report(report, args.out)
    _attempt_upload(ctx, gate, pub)
    if ctx["archiver"] is None:
        return
    if report["archive"]["status"] != "FAILED":
        report["archive"]["status"] = "OK"
    report["archive"]["log"] = list(ctx["archive_log"])
    pub2 = publish_report(report, args.out)     # concluded status on disk
    if terminal and report["archive"]["status"] == "OK":
        fin = _attempt_upload(ctx, f"{gate}(finalized)", pub2)
        if not fin["verified"]:
            # _attempt_upload already set status FAILED and flagged the halt.
            # Republish ONCE so the surviving local report truthfully says
            # FAILED and carries the failed finalized receipt. No third
            # upload is attempted automatically.
            report["archive"]["log"] = list(ctx["archive_log"])
            publish_report(report, args.out)


# --------------------------------------------------------------------------
# report plumbing
# --------------------------------------------------------------------------


def new_ctx(args, archiver, archive_skip_reason=""):
    report = {
        "stage": "gpu_probe",
        "storage": dict(STORAGE_FLAGS),
        **{k: v for k, v in STORAGE_FLAGS.items() if k != "note"},
        "invocation": {"model": args.model, "revision": args.revision,
                       "gates": args.gates, "all": args.all,
                       "token_contract": args.token_contract,
                       "stream": args.stream, "layers": args.layers,
                       "accept_isolated_compile_cost":
                           args.accept_isolated_compile_cost,
                       "out": args.out, "hf_repo": args.hf_repo,
                       "hf_repo_type": args.hf_repo_type,
                       "hf_path_prefix": args.hf_path_prefix},
        "tolerances": {"authority": "full_model_smoke.py (imported constants)",
                       **TOL,
                       "recompile_factor": RECOMPILE_FACTOR,
                       "recompile_abs_s": RECOMPILE_ABS_S,
                       "compile_overhead_max_frac": COMPILE_OVERHEAD_MAX_FRAC,
                       "disk_safety_factor": DISK_SAFETY_FACTOR},
        "gates": {}, "verdicts": {},
        "boundary_prohibited_claims": [],
        "archive": {"status": "PENDING", "log": []},
    }
    return {"args": args, "report": report, "archiver": archiver,
            "archive_log": [], "archive_failed": False,
            "archive_skip_reason": archive_skip_reason,
            "ck": None, "model": None, "contract_plan": None}


def record(ctx, gate, verdict, **data):
    ctx["report"]["gates"][gate] = data
    ctx["report"]["verdicts"][gate] = verdict
    print(f"\n[{gate}] {verdict}")
    for k, v in data.items():
        if isinstance(v, (dict, list)) and len(str(v)) > 200:
            print(f"    {k}: <{type(v).__name__}, {len(v)} entries>")
        else:
            print(f"    {k}: {v}")
    for claim in data.get("prohibited", ()):
        if claim not in ctx["report"]["boundary_prohibited_claims"]:
            ctx["report"]["boundary_prohibited_claims"].append(claim)
    return verdict


def host_memory() -> dict:
    out = {"host_ram_total_bytes": None, "host_ram_available_bytes": None}
    try:
        out["host_ram_total_bytes"] = (os.sysconf("SC_PAGE_SIZE")
                                       * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    out["host_ram_available_bytes"] = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------
# G1: environment and exact-load gate
# --------------------------------------------------------------------------


def g1_env(ctx):
    args = ctx["args"]
    info = {"python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    for mod in ("mamba_ssm", "tilelang", "tvm_ffi", "triton", "quack"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            info[mod] = f"MISSING ({type(e).__name__})"

    # exact checkpoint resolution, LOCAL-ONLY, before anything expensive
    try:
        ck = resolve_checkpoint(args.model, revision=args.revision,
                                local_only=True)
    except CheckpointResolveError as e:
        return record(ctx, "g1", STOP, **info,
                      error=f"checkpoint resolution failed: {e}")
    prov = ck.provenance()
    info["provenance"] = prov
    if (not prov.get("path") or not prov.get("weights_file")
            or (prov.get("resolved_commit") is None
                and not ck.from_local_dir)):
        return record(ctx, "g1", STOP, **info,
                      error="incomplete provenance: the exact checkpoint "
                            "cannot be identified in the manifest")
    ctx["ck"] = ck

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    du = shutil.disk_usage(out_dir)
    info.update(host_memory())
    info["tmp_output_dir"] = out_dir
    info["tmp_disk_free_bytes"] = du.free
    info["storage"] = dict(STORAGE_FLAGS)

    if not info["cuda_available"]:
        return record(ctx, "g1", STOP, **info, error="no CUDA device")
    info["device"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    info["device_capability"] = f"{cap[0]}.{cap[1]}"
    free_b, total_b = torch.cuda.mem_get_info()
    info["gpu_mem_free_bytes"], info["gpu_mem_total_bytes"] = free_b, total_b
    if info["mamba_ssm"].startswith("MISSING"):
        return record(ctx, "g1", STOP, **info, error="mamba_ssm not importable")

    # load from the RESOLVED SNAPSHOT PATH, never a mutable hub id
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    t0 = time.time()
    model = MambaLMHeadModel.from_pretrained(
        ck.path, device="cuda", dtype=torch.bfloat16).eval()
    info["load_s"] = round(time.time() - t0, 2)
    ctx["model"] = model

    # spec agreement: live mixer vs checkpoint tensors. Tensor-derived arm is
    # authoritative; config is a consistency check only (fms.resolve_arm).
    cfg = ck.load_config()
    sd = ck.load_state_dict()
    try:
        a = InProjSpec.from_mixer(model.backbone.layers[0].mixer)
        b = InProjSpec.from_state_dict(sd, cfg, layer=0)
        fms.resolve_arm(a, cfg.get("ssm_cfg"), "g1 live-mixer")
        fms.resolve_arm(b, cfg.get("ssm_cfg"), "g1 state-dict")
        diffs = [f for f in ("d_inner", "d_state", "ngroups", "mimo_rank",
                             "nheads", "headdim", "n_rope_angles", "total")
                 if getattr(a, f) != getattr(b, f)]
    except ShapeContractError as e:
        del sd
        return record(ctx, "g1", STOP, **info, error=f"spec contract: {e}")
    del sd
    if diffs:
        return record(ctx, "g1", STOP, **info,
                      error=f"live-mixer vs state-dict spec disagreement: {diffs}")
    info["arm"] = a.arm
    info["spec"] = {f: getattr(a, f) for f in
                    ("d_inner", "d_state", "ngroups", "mimo_rank", "nheads",
                     "headdim", "n_rope_angles", "total")}

    ids = torch.zeros(1, args.seqlen, dtype=torch.long, device="cuda")
    t0 = time.time()
    with torch.inference_mode():
        model(ids)
    torch.cuda.synchronize()
    info["first_forward_jit_s"] = round(time.time() - t0, 2)
    t0 = time.time()
    with torch.inference_mode():
        model(ids)
    torch.cuda.synchronize()
    info["warmed_forward_s"] = round(time.time() - t0, 4)
    return record(ctx, "g1", PASS, **info)


# --------------------------------------------------------------------------
# G2: authoritative elementwise mixer parity, DELEGATED to full_model_smoke
# --------------------------------------------------------------------------

# a case is tensor evidence only if it carries the authority's full elementwise
# metric set, including the boundary/interior position errors. Anything less
# (e.g. top-k or logit-agreement only) is rejected. Presence is required;
# pos_interior_* may legitimately be null for lengths <= 2.
G2_TENSOR_KEYS = ("max_abs_err", "mean_abs_err", "rel_p50", "rel_p99",
                  "rel_max", "vendor_rel_p95", "vendor_rel_n",
                  "cosine", "norm_max_err", "pos_first", "pos_final",
                  "pos_interior_max", "pos_interior_mean")


def adapt_g2(rep: dict, expected_cases: int | None = None):
    """Validate an ingested full_model_smoke report. Returns (verdict, reasons).

    This adapter performs NO tensor math and holds NO tolerances: those belong
    to the authority. It only checks that the evidence is complete, structured
    and elementwise, and that the authority itself passed:
      * coverage.complete is true and expected == executed == len(cases)
      * every case identity (model, layer, seqlen) is complete and unique
      * every case carries the full elementwise metric set
      * no spec/arm mismatches, no plan errors, overall PASS
    """
    reasons = []
    cov = rep.get("coverage") or {}
    exp, execd = cov.get("expected_cases"), cov.get("executed_cases")
    cases = rep.get("cases") or []
    if not exp:
        reasons.append("missing or zero expected cases")
    elif expected_cases is not None and exp != expected_cases:
        reasons.append(f"expected-case mismatch: authority planned {exp}, "
                       f"probe computed {expected_cases}")
    if not execd:
        reasons.append("zero executed cases")
    elif exp and execd != exp:
        reasons.append(f"coverage incomplete: executed {execd} of {exp}")
    if cov.get("complete") is not True:
        reasons.append("coverage.complete is not true")
    if exp and len(cases) != exp:
        reasons.append(f"case rows missing: {len(cases)} rows for "
                       f"{exp} expected cases")
    if cov.get("plan_errors"):
        reasons.append(f"plan errors: {cov['plan_errors']}")
    if rep.get("spec_mismatches"):
        reasons.append(f"shape/spec/arm mismatch: {rep['spec_mismatches']}")

    if not cases:
        reasons.append("no cases present")
    seen = set()
    for c in cases:
        ident = (c.get("model"), c.get("layer"), c.get("seqlen"))
        tag = f"{ident[0]} L{ident[1]} len={ident[2]}"
        if any(v is None for v in ident):
            reasons.append(f"case identity incomplete: {ident}")
        elif ident in seen:
            reasons.append(f"duplicate case identity: {tag}")
        seen.add(ident)
        if c.get("error"):
            reasons.append(f"case {tag} raised: {c['error']}")
        elif not all(k in c for k in G2_TENSOR_KEYS):
            missing = [k for k in G2_TENSOR_KEYS if k not in c]
            reasons.append(f"case {tag} is not elementwise tensor evidence "
                           f"(missing {missing}); top-k/logit-only evidence "
                           f"is rejected")
        elif c.get("verdict") != PASS:
            reasons.append(f"case {tag} tensor tolerance failure "
                           f"(verdict {c.get('verdict')})")
    overall = (rep.get("verdicts") or {}).get("overall")
    if overall != PASS:
        reasons.append(f"authority overall verdict: {overall}")
    return (PASS if not reasons else STOP), reasons


def _g2_revisions(ctx) -> dict:
    """Pin an EXACT commit per G2 repository. SISO and MIMO are separate
    repositories with different commits; one --revision can never serve both.

    Precedence per model: explicit --g2-revision-{siso,mimo}, then --revision
    when the model IS the G1 model, then the locally resolved snapshot commit.
    """
    args = ctx["args"]
    explicit = {"siso-187m": args.g2_revision_siso,
                "mimo-187m": args.g2_revision_mimo}
    out = {}
    for name in G2_MODELS.split(","):
        if explicit.get(name):
            out[name] = (explicit[name], "explicit --g2-revision argument")
        elif name == args.model and args.revision:
            out[name] = (args.revision, "--revision (this is the G1 model)")
        else:
            ck = resolve_checkpoint(name, local_only=True)
            rc = ck.provenance().get("resolved_commit")
            if rc is None and not ck.from_local_dir:
                raise CheckpointResolveError(
                    f"{name}: no resolvable commit to pin the G2 subprocess to")
            out[name] = (rc, "locally resolved snapshot commit")
    return out


def g2_parity(ctx):
    args = ctx["args"]
    try:
        revisions = _g2_revisions(ctx)
    except CheckpointResolveError as e:
        return record(ctx, "g2", STOP,
                      error=f"per-model revision pinning failed: {e}")

    authority = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "full_model_smoke.py")
    verdict, reasons = PASS, []
    runs, all_idents, combined_expected = {}, [], 0

    # full_model_smoke accepts ONE revision per invocation, so each repository
    # gets its own run, pinned to its own commit, with its own sidecar report.
    for name, (rev, rev_source) in revisions.items():
        plan_ns = argparse.Namespace(models=name, layers=G2_LAYERS,
                                     revision=rev, lengths=None)
        plan, plan_errors, expected = fms.build_plan(plan_ns)
        if plan_errors or expected == 0:
            return record(ctx, "g2", STOP,
                          error=f"G2 coverage plan invalid for {name}",
                          revision=rev, plan_errors=plan_errors,
                          expected_cases=expected)
        combined_expected += expected

        auth_out = f"{args.out}.g2_{name.replace('/', '_')}.json"
        cmd = [sys.executable, authority, "--models", name,
               "--layers", G2_LAYERS, "--out", auth_out]
        if rev:
            cmd += ["--revision", rev]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.isfile(auth_out):
            return record(ctx, "g2", STOP,
                          error=f"authority produced no report for {name}",
                          command=" ".join(cmd), returncode=proc.returncode,
                          stderr_tail=proc.stderr[-2000:])
        with open(auth_out, "rb") as fh:
            rep = _json_loads(fh.read())

        v, rs = adapt_g2(rep, expected_cases=expected)
        if v != PASS:
            verdict = STOP
        reasons += [f"{name}: {r}" for r in rs]

        # provenance: the authority must have used exactly the pinned commit
        prov = (rep.get("provenance") or {}).get(name) or {}
        if not prov:
            verdict = STOP
            reasons.append(f"{name}: authority recorded no provenance")
        elif rev is not None and prov.get("resolved_commit") != rev:
            verdict = STOP
            reasons.append(f"{name}: provenance mismatch: pinned {rev}, "
                           f"authority resolved {prov.get('resolved_commit')}")
        if name == args.model and ctx["ck"] is not None:
            own = ctx["ck"].provenance()
            for f in ("path", "resolved_commit"):
                if prov.get(f) != own.get(f):
                    verdict = STOP
                    reasons.append(f"{name}: provenance mismatch on {f}: "
                                   f"authority {prov.get(f)} vs probe {own.get(f)}")

        all_idents += [(c.get("model"), c.get("layer"), c.get("seqlen"))
                       for c in rep.get("cases", [])]

        # the COMPLETE authority report (with per_position_max_abs) is archived
        # verbatim as a sidecar; the embedded copy below is slim by design and
        # is never the only remotely preserved evidence.
        sidecar = archive_sidecar(ctx, "g2", auth_out)
        slim = dict(rep)
        slim["cases"] = [{k: v for k, v in c.items()
                          if k != "per_position_max_abs"}
                         for c in rep.get("cases", [])]
        runs[name] = {"revision": rev, "revision_source": rev_source,
                      "command": " ".join(cmd),
                      "authority_report_path": auth_out,
                      "sidecar_archive": sidecar,
                      "expected_cases": expected,
                      "authority_report_slim": slim}

        # a failed sidecar upload means evidence cannot leave the pod: stop
        # HERE, before the next repository's subprocess spends any more GPU.
        # This is an ARCHIVAL stop, not a parity failure, and is recorded as
        # such: the required matrix is simply incomplete.
        if ctx["archive_failed"]:
            return record(ctx, "g2", STOP,
                          parity_failure=False,
                          evidence_matrix_complete=False,
                          stop_reason="archival failure after authority sidecar",
                          error=("required G2 matrix incomplete: the authority "
                                 "sidecar could not be archived, so no further "
                                 "authority subprocess was launched"),
                          revisions={n: {"commit": r[0], "source": r[1]}
                                     for n, r in revisions.items()},
                          reasons=reasons, runs=runs)

    # combined required matrix across BOTH repositories
    if len(all_idents) != len(set(all_idents)):
        verdict = STOP
        reasons.append("duplicate case identities across the combined matrix")
    if len(all_idents) != combined_expected:
        verdict = STOP
        reasons.append(f"combined matrix incomplete: {len(all_idents)} cases "
                       f"for {combined_expected} expected")
    want_models = set(G2_MODELS.split(","))
    want_layers = {int(x) for x in G2_LAYERS.split(",")}
    got_models = {i[0] for i in all_idents}
    got_layers = {i[1] for i in all_idents}
    if got_models != want_models or not want_layers.issubset(got_layers):
        verdict = STOP
        reasons.append(f"combined matrix does not cover models {want_models} "
                       f"x layers {want_layers}: got {got_models} x {got_layers}")

    return record(ctx, "g2", verdict,
                  authority="full_model_smoke.py CLI, one run per repository, "
                            "JSON ingested",
                  revisions={n: {"commit": r[0], "source": r[1]}
                             for n, r in revisions.items()},
                  combined_expected_cases=combined_expected,
                  reasons=reasons,
                  runs=runs,
                  note=("a failed real run authorizes debugging, never switch "
                        "search or tolerance widening"))


# --------------------------------------------------------------------------
# G3: prefill vs step()/decode boundary
# --------------------------------------------------------------------------


def g3_verdict(step_available: bool, metrics_verdict: str | None) -> str:
    """Unavailable/unconstructible step() is BOUNDARY; a constructed step path
    that disagrees is STOP, not BOUNDARY."""
    if not step_available:
        return BOUNDARY
    return PASS if metrics_verdict == PASS else STOP


def _g3_ids(ctx):
    """Identical token ids for both paths. Prefer a real contract block; fall
    back to the pinned tokenizer on a fixed prompt."""
    plan = ctx["contract_plan"]
    if plan is not None:
        stream = plan["stream"]
        offs = plan["offsets"]
        lens = np.diff(offs)[:plan["blocks_expected"]]
        b = int(np.argsort(lens)[len(lens) // 2])       # median-length block
        ids = plan["tc"][f"{stream}_ids"][int(offs[b]):int(offs[b + 1])]
        return torch.tensor(np.asarray(ids, dtype=np.int64)), f"contract block {b}"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REV)
    prompt = ctx["args"].prompt
    return (torch.tensor([tok.bos_token_id]
                         + tok(prompt, add_special_tokens=False).input_ids),
            f"pinned tokenizer on prompt {prompt!r}")


def g3_step(ctx):
    model = ctx["model"]
    ids, ids_source = _g3_ids(ctx)
    full = ids.to("cuda").unsqueeze(0)
    T = int(ids.shape[0])

    base = {"ids_source": ids_source, "n_tokens": T,
            "checkpoint": ctx["ck"].provenance(),
            "state_parity_validated": False,
            "state_parity_note": ("only final-step OUTPUT logits are compared; "
                                  "no cache/state tensors are compared, so no "
                                  "kernel-state parity is claimed")}

    try:
        from mamba_ssm.utils.generation import InferenceParams

        with torch.inference_mode():
            prefill = model(full).logits[0, -1].float().cpu()
            ip = InferenceParams(max_seqlen=T + 1, max_batch_size=1)
            model(full[:, :-1], inference_params=ip)
            ip.seqlen_offset = T - 1
            stepped = model(full[:, -1:],
                            inference_params=ip).logits[0, -1].float().cpu()
    except Exception as e:  # noqa: BLE001 - unconstructible step path is a BOUNDARY
        return record(ctx, "g3", g3_verdict(False, None), **base,
                      step_available=False,
                      error=f"step path could not be constructed or run: "
                            f"{type(e).__name__}: {e}",
                      allowed=list(G3_BOUNDARY_ALLOWED),
                      prohibited=list(G3_BOUNDARY_PROHIBITED))

    # complete final-step logits, elementwise, via the authority's metric code
    m = fms.parity_metrics(stepped.view(1, -1), prefill.view(1, -1),
                           TOL["rel_floor_frac"])
    m.pop("per_position_max_abs", None)
    mv = fms.verdict_for(m, TOL["cos"], TOL["rel"], TOL["norm_max"])
    verdict = g3_verdict(True, mv)
    supplemental = {"top1_agree": bool(int(prefill.argmax())
                                       == int(stepped.argmax())),
                    "note": "SUPPLEMENTAL ONLY; never verdict-bearing"}
    data = {**base, "step_available": True, "metrics": m,
            "metrics_verdict": mv, "supplemental": supplemental}
    if verdict == STOP:
        data["error"] = ("constructed step path disagrees with prefill: this "
                         "is a genuine parity failure, not a boundary")
    return record(ctx, "g3", verdict, **data)


# --------------------------------------------------------------------------
# G4: variable-length / JIT policy against the REAL capture contract
# --------------------------------------------------------------------------


def classify_recompiles(first_by_len: dict, warm_by_len: dict,
                        revisit_by_len: dict,
                        factor: float = RECOMPILE_FACTOR,
                        abs_s: float = RECOMPILE_ABS_S):
    """Timing classifier. Every length is compared against ITS OWN warmed
    time: a legitimately slow long forward is not a recompile. Returns
    per-length bars, suspected first-call recompiles, revisited lengths that
    recompiled again, and the per-length compile-cost excess."""
    bars = {L: max(factor * warm_by_len[L], abs_s) for L in warm_by_len}
    suspected = {L: t for L, t in first_by_len.items()
                 if L in bars and t > bars[L]}
    revisit_recompiled = {L: t for L, t in revisit_by_len.items()
                          if L in bars and t > bars[L]}
    excess = {L: max(first_by_len[L] - warm_by_len[L], 0.0)
              for L in first_by_len if L in warm_by_len}
    return bars, suspected, revisit_recompiled, excess


def _contract_lengths(plan):
    offs = plan["offsets"]
    lens = np.diff(offs)[:plan["blocks_expected"]].astype(np.int64)
    reps = sorted({int(v) for v in
                   np.quantile(lens, [0.0, 0.25, 0.5, 0.75, 1.0],
                               method="nearest")})
    by_len = {}
    for b, L in enumerate(lens):
        by_len.setdefault(int(L), b)
    return lens, reps, by_len


def _block_tensor(plan, b):
    stream, offs = plan["stream"], plan["offsets"]
    ids = plan["tc"][f"{stream}_ids"][int(offs[b]):int(offs[b + 1])]
    return torch.tensor(np.asarray(ids, dtype=np.int64), device="cuda").unsqueeze(0)


def g4_jit(ctx):
    plan = ctx["contract_plan"]
    if plan is None:
        return record(ctx, "g4", STOP,
                      error="--token-contract is required for G4: the capture "
                            "policy is defined by the contract's real block "
                            "lengths, not an invented uniform sequence length")
    model = ctx["model"]
    lens, reps, by_len = _contract_lengths(plan)

    policy = ("accepted capture policy: one isolated unpadded forward per "
              "block, shape (1, valid_len), own BOS, fresh recurrent state, "
              "no concatenation, no right-padding")

    def timed(t):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model(t)
        torch.cuda.synchronize()
        return time.time() - t0, out

    # First-call timings come first and are NOT polluted: the drift reference
    # is captured FROM the median-length block's own first timed call, so no
    # shape is pre-warmed before its first_s measurement.
    drift_rep = reps[len(reps) // 2]
    drift_t = _block_tensor(plan, by_len[drift_rep])
    ref_logits = None

    first_s, warm_s, revisit_s = {}, {}, {}
    for L in reps:
        t = drift_t if L == drift_rep else _block_tensor(plan, by_len[L])
        first_s[L], out = timed(t)
        if L == drift_rep:
            ref_logits = out.logits[0, -1].float().cpu()
        w = [timed(t)[0] for _ in range(3)]
        warm_s[L] = sorted(w)[1]
    for L in reps:                                  # return after cycling shapes
        revisit_s[L], _ = timed(_block_tensor(plan, by_len[L]))

    bars, suspected, revisit_recompiled, excess = classify_recompiles(
        first_s, warm_s, revisit_s)

    # elementwise drift for the same logical prefix after shape exercise
    _, out1 = timed(drift_t)
    dm = fms.parity_metrics(out1.logits[0, -1].float().cpu().view(1, -1),
                            ref_logits.view(1, -1), TOL["rel_floor_frac"])
    dm.pop("per_position_max_abs", None)
    drift_verdict = fms.verdict_for(dm, TOL["cos"], TOL["rel"], TOL["norm_max"])

    n_unique = int(len(np.unique(lens)))
    total_tokens = int(lens.sum())

    # projected compile overhead from PER-LENGTH measured excesses: each unique
    # contract length is assigned the excess of its NEAREST measured
    # representative length (assumption recorded below).
    rep_arr = np.array(sorted(excess))
    exc_arr = np.array([excess[int(L)] for L in rep_arr])
    uniq = np.unique(lens)
    nearest_exc = exc_arr[np.abs(rep_arr[None, :] - uniq[:, None]).argmin(1)]
    if revisit_recompiled:
        per_block_exc = exc_arr[np.abs(rep_arr[None, :] - lens[:, None]).argmin(1)]
        projected_compile_s = float(per_block_exc.sum())
        compile_model = ("revisited lengths recompile: compile cost scales "
                         "with BLOCK COUNT, not unique lengths")
    else:
        projected_compile_s = float(nearest_exc.sum())
        compile_model = "compiled shapes are retained: one compile per unique length"

    warmed_tps = drift_rep / max(warm_s[drift_rep], 1e-9)
    warmed_capture_s = total_tokens / max(warmed_tps, 1e-9)

    data = {
        "capture_policy_under_test": policy,
        "representative_lengths": reps,
        "n_blocks": int(len(lens)), "n_unique_lengths": n_unique,
        "total_valid_tokens": total_tokens,
        "first_call_s": {str(k): round(v, 3) for k, v in first_s.items()},
        "warmed_s": {str(k): round(v, 4) for k, v in warm_s.items()},
        "revisit_s": {str(k): round(v, 4) for k, v in revisit_s.items()},
        "recompile_bar_s_by_length": {str(k): round(v, 3)
                                      for k, v in bars.items()},
        "suspected_recompiles": {str(k): round(v, 3)
                                 for k, v in suspected.items()},
        "revisit_recompiled": {str(k): round(v, 3)
                               for k, v in revisit_recompiled.items()},
        "compile_excess_s_by_length": {str(k): round(v, 3)
                                       for k, v in excess.items()},
        "compile_model": compile_model,
        "projected_compile_overhead_s": round(projected_compile_s, 1),
        "projected_warmed_capture_s": round(warmed_capture_s, 1),
        "assumptions": [
            "per-unique-length compile excess is mapped from the NEAREST "
            "measured representative length",
            "earlier gates (g1-g3) may already have compiled overlapping "
            "shapes, which can only UNDERSTATE first-call costs",
        ],
        "drift_metrics": dm, "drift_verdict": drift_verdict,
        "padding_note": ("padding is NOT recommended automatically: it changes "
                         "final-position/trapezoid semantics and would require "
                         "its own parity gate before any use"),
    }

    if drift_verdict != PASS:
        data["error"] = ("same logical prefix produced different outputs after "
                         "exercising other shapes: outputs disagree")
        return record(ctx, "g4", STOP, **data)
    if projected_compile_s > COMPILE_OVERHEAD_MAX_FRAC * warmed_capture_s:
        if args.accept_isolated_compile_cost:
            data["policy_decision"] = (
                "ACCEPTED: pay the measured one-time shape-compilation cost "
                "and retain the validated isolated, unpadded, one-block-per-"
                "forward capture policy")
            data["prohibited"] = [
                "running the capture under an untested padding policy"]
            return record(ctx, "g4", PASS, **data)
        data["required_policy_decision"] = (
            "projected JIT compile overhead exceeds the declared fraction "
            f"({COMPILE_OVERHEAD_MAX_FRAC}) of warmed capture time under the "
            "isolated-unpadded policy. Decide explicitly: (a) accept the "
            "measured wall-time cost and proceed with the accepted policy, or "
            "(b) design a padded capture, which changes final/trapezoid "
            "semantics and must pass its own parity gate first. No default is "
            "chosen here.")
        data["prohibited"] = [
            "running the capture under an untested padding policy"]
        return record(ctx, "g4", BOUNDARY, **data)
    return record(ctx, "g4", PASS, **data)


# --------------------------------------------------------------------------
# G5: realistic resource gate, mirroring the actual capture
# --------------------------------------------------------------------------


def _selected_layers(ctx):
    model, args = ctx["model"], ctx["args"]
    n = len(model.backbone.layers)
    if args.layers:
        sel = [int(x) for x in args.layers.split(",")]
        bad = [li for li in sel if li < 0 or li >= n]
        if bad:
            raise ValueError(f"--layers {bad} out of range (n_layer={n})")
        return sel
    return list(range(n))


def g5_resources(ctx):
    plan = ctx["contract_plan"]
    if plan is None:
        return record(ctx, "g5", STOP,
                      error="--token-contract is required for G5: the resource "
                            "gate must mirror the real capture shapes")
    model, args = ctx["model"], ctx["args"]
    try:
        sel = _selected_layers(ctx)
    except ValueError as e:
        return record(ctx, "g5", STOP, error=str(e))
    spec = InProjSpec.from_mixer(model.backbone.layers[0].mixer)
    lens, reps, by_len = _contract_lengths(plan)
    t = _block_tensor(plan, by_len[reps[-1]])       # longest selected block

    ram0 = host_memory()

    def fwd():
        with torch.inference_mode():
            model(t)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    fwd()
    bare_s = time.time() - t0
    bare_alloc = torch.cuda.max_memory_allocated()
    bare_res = torch.cuda.max_memory_reserved()

    # the REAL hook surface: capture_stage_b installs, per selected layer,
    # in_proj forward, out_proj forward_pre, block forward, mlp.fc2 forward_pre,
    # and holds the captured tensors while a block is processed. Mirror that.
    grabbed = {}
    handles = []
    for li in sel:
        blk = model.backbone.layers[li]
        handles += [
            blk.mixer.in_proj.register_forward_hook(
                lambda m, i, o, li=li: grabbed.__setitem__(("inproj", li),
                                                           o.detach())),
            blk.mixer.out_proj.register_forward_pre_hook(
                lambda m, i, li=li: grabbed.__setitem__(("outproj", li),
                                                        i[0].detach())),
            blk.register_forward_hook(
                lambda m, i, o, li=li: grabbed.__setitem__(
                    ("block", li), (o[0] if isinstance(o, tuple) else o).detach())),
            blk.mlp.fc2.register_forward_pre_hook(
                lambda m, i, li=li: grabbed.__setitem__(("mlp", li),
                                                        i[0].detach())),
        ]
    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        fwd()
        hooked_s = time.time() - t0
        hooked_alloc = torch.cuda.max_memory_allocated()
        hooked_res = torch.cuda.max_memory_reserved()
    finally:
        for h in handles:
            h.remove()
    hooks_fired = len(grabbed)
    grabbed.clear()
    torch.cuda.empty_cache()
    ram1 = host_memory()

    est = estimate_artifact_bytes(
        n_blocks=plan["blocks_expected"], n_layers=len(sel),
        n_heads=spec.nheads, rank=spec.mimo_rank, n_labels=EST_N_LABELS)
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    du = shutil.disk_usage(out_dir)
    need = DISK_SAFETY_FACTOR * est["complete_payload_bytes"]

    data = {
        "batch_size": 1,
        "layers_hooked": sel, "n_layers_hooked": len(sel),
        "hook_surface": "in_proj fwd + out_proj pre + block fwd + mlp.fc2 pre "
                        "per layer (capture_stage_b surface)",
        "hooks_fired": hooks_fired,
        "probe_block_len": int(t.shape[1]),
        "gpu_peak_alloc_bare_bytes": bare_alloc,
        "gpu_peak_reserved_bare_bytes": bare_res,
        "gpu_peak_alloc_hooked_bytes": hooked_alloc,
        "gpu_peak_reserved_hooked_bytes": hooked_res,
        "hook_time_overhead_pct": round(100 * (hooked_s / max(bare_s, 1e-9) - 1), 1),
        "host_ram_before": ram0, "host_ram_after": ram1,
        "projected_artifact": est,
        "tmp_disk_free_bytes": du.free,
        "disk_needed_with_safety_bytes": int(need),
        "disk_safety_factor": DISK_SAFETY_FACTOR,
        "staging_note": ("the projected artifact (a few GiB is expected and is "
                         "NOT a failure) is temporary staging that must fit "
                         "only long enough to be uploaded to Hugging Face; no "
                         "pod path is persistent"),
    }
    expected_fired = 4 * len(sel)
    if hooks_fired != expected_fired:
        data["error"] = (f"hook surface incomplete: {hooks_fired} captures, "
                         f"expected {expected_fired}")
        return record(ctx, "g5", STOP, **data)
    if du.free < need:
        data["error"] = (f"temporary disk cannot safely stage the artifact: "
                         f"free {du.free} < required {int(need)}")
        return record(ctx, "g5", STOP, **data)
    return record(ctx, "g5", PASS, **data)


# --------------------------------------------------------------------------
# G6: realistic throughput estimate from the token contract
# --------------------------------------------------------------------------


def g6_throughput(ctx):
    plan = ctx["contract_plan"]
    if plan is None:
        return record(ctx, "g6", STOP,
                      error="--token-contract is required for G6: the estimate "
                            "must come from the contract's real offsets, not a "
                            "hardcoded token count")
    model = ctx["model"]
    lens, reps, by_len = _contract_lengths(plan)

    g4 = ctx["report"]["gates"].get("g4") or {}
    measured_warm = {int(k): float(v) for k, v in (g4.get("warmed_s") or {}).items()}
    if not measured_warm:
        for L in reps:
            t = _block_tensor(plan, by_len[L])
            with torch.inference_mode():
                model(t)
            torch.cuda.synchronize()
            w = []
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    model(t)
                torch.cuda.synchronize()
                w.append(time.time() - t0)
            measured_warm[L] = sorted(w)[1]

    # per-block warmed time via the nearest measured representative length
    rep_arr = np.array(sorted(measured_warm))
    per_block = rep_arr[np.abs(rep_arr[None, :] - lens[:, None]).argmin(1)]
    warmed_total_s = float(sum(measured_warm[int(L)] for L in per_block))

    assumptions = [
        "per-block warmed time is taken from the NEAREST measured "
        "representative length, not measured per block",
        "estimate covers the currently selected layer/hook surface and "
        "stream only",
    ]

    # measured hook overhead from G5's capture-surface forward, applied as a
    # multiplicative factor on the warmed bare-forward total
    g5 = ctx["report"]["gates"].get("g5") or {}
    oh_pct = g5.get("hook_time_overhead_pct")
    if oh_pct is None:
        hook_factor = 1.0
        assumptions.append("G5 did not run: measured hook overhead is "
                           "EXCLUDED, tightening the lower bound downward")
    else:
        hook_factor = 1.0 + max(float(oh_pct), 0.0) / 100.0
    hooked_total_s = warmed_total_s * hook_factor

    compile_s = g4.get("projected_compile_overhead_s")
    if compile_s is None:
        compile_s = 0.0
        assumptions.append("G4 did not run: JIT compile overhead is EXCLUDED")

    data = {
        "label": ("LOWER-BOUND COMPUTE ESTIMATE. This is not complete capture "
                  "wall time: the excluded costs below are real and unmeasured"),
        "excluded_costs": [
            "accumulator / derived-quantity computation per block",
            "artifact serialization and compression",
            "Hugging Face upload and remote verification time",
            "per-block Python and hook bookkeeping beyond the single "
            "measured block",
        ],
        "measured": {
            "warmed_s_by_length": {str(k): round(v, 4)
                                   for k, v in measured_warm.items()},
            "hook_time_overhead_pct": oh_pct,
        },
        "n_isolated_forwards": int(len(lens)),
        "valid_tokens": int(plan["valid_tokens_expected"]),
        "content_tokens": int(plan["content_tokens_expected"]),
        "length_distribution": {
            "min": int(lens.min()), "p25": int(np.quantile(lens, .25)),
            "p50": int(np.quantile(lens, .50)), "p75": int(np.quantile(lens, .75)),
            "max": int(lens.max()), "mean": round(float(lens.mean()), 1)},
        "warmed_bare_forward_s": round(warmed_total_s, 1),
        "warmed_hooked_forward_s": round(hooked_total_s, 1),
        "compile_overhead_s": round(float(compile_s), 1),
        "lower_bound_compute_s": round(hooked_total_s + float(compile_s), 1),
        "assumptions": assumptions,
    }
    return record(ctx, "g6", PASS, **data)


# --------------------------------------------------------------------------
# G7: derived-quantity precision check (distinct from G2)
# --------------------------------------------------------------------------

G7_QUANTITIES = ("Delta", "A", "alpha", "lambda", "gamma", "shifted_gamma",
                 "trap_scale", "phase")

# G7 tolerances, declared BEFORE any GPU run and SPECIFIC to the storage
# question. These are deliberately NOT the mixer-output parity tolerances:
# those were calibrated for a bf16 kernel accumulating through ~5 rounding
# stages against an fp32 oracle, while G7 measures exactly ONE rounding stage,
# fp32 -> np.float16 storage (capture_stage_b.BLOCK_SUMMARY_DTYPE). float16
# has 10 mantissa bits, eps = 2^-10 ~= 9.8e-4, so a single storage rounding
# bounds relative error at ~eps for normal values; the bars below sit ~5x
# above that to absorb values near the subnormal boundary (~6.1e-5), and the
# real hazards, overflow past 65504 and subnormal flush-to-zero, are counted
# explicitly per quantity.
F16_EPS = 2.0 ** -10
G7_TOLS = {"rel_p99": 5 * F16_EPS, "norm_max": 5 * F16_EPS, "cos": 0.9999,
           "max_abs_input": 60000.0, "flush_frac_max": 0.0}


def g7_precision(ctx):
    """Precision guidance for the capture artifacts. This does NOT test kernel
    recurrence parity, and its verdict is advisory about STORAGE only.

    The kernel runs bf16, so the hooked in_proj activations are irreducibly
    bf16 at the source, and the shared helpers (recurrence_quantities,
    phase_details) deliberately upcast to fp32 internally: derivation is fp32
    by construction. Capture Final then stores per-block summaries as
    np.float16 (capture_stage_b.BLOCK_SUMMARY_DTYPE), so the verdict-bearing
    comparison is each fp32-derived quantity against its float16 round-trip,
    under the G7_TOLS declared above. A bf16 round-trip is reported as a
    separate, never verdict-bearing diagnostic. May PASS only when every
    quantity is finite, un-flushed, and within the declared tolerances;
    otherwise BOUNDARY (advisory) or STOP (non-finite).
    """
    model = ctx["model"]
    args = ctx["args"]
    li = _selected_layers(ctx)[0] if not args.layers else int(args.layers.split(",")[0])
    mixer = model.backbone.layers[li].mixer
    spec = InProjSpec.from_mixer(mixer)

    plan = ctx["contract_plan"]
    if plan is not None:
        lens, reps, by_len = _contract_lengths(plan)
        ids = _block_tensor(plan, by_len[reps[len(reps) // 2]])
        src = "contract median-length block"
    else:
        ids = torch.zeros(1, args.seqlen, dtype=torch.long, device="cuda")
        src = f"synthetic zeros len={args.seqlen} (no contract supplied)"

    grab = {}
    h = mixer.in_proj.register_forward_hook(
        lambda m, i, o: grab.__setitem__("o", o.detach()))
    try:
        with torch.inference_mode():
            model(ids)
    finally:
        h.remove()
    if "o" not in grab:
        return record(ctx, "g7", STOP, error=f"in_proj hook fired nothing at L{li}")

    parts = split_in_proj(grab["o"][0], spec)
    q = recurrence_quantities(parts["dd_dt"], parts["dd_A"], parts["trap"],
                              mixer.dt_bias.detach().float())
    hi = {k: q[k].float().cpu() for k in G7_QUANTITIES if k != "phase"}
    ph = phase_details(parts["angles"].float().cpu(),
                       hi["Delta"], None)
    hi["phase"] = ph["wrapped"].reshape(ph["wrapped"].shape[0], -1)

    per_q, reasons = {}, []
    for name in G7_QUANTITIES:
        a = hi[name]
        flat = a.reshape(a.shape[0], -1)
        # verdict-bearing arm: the ACTUAL storage dtype of Capture Final
        lo16 = torch.from_numpy(
            flat.detach().numpy().astype(BLOCK_SUMMARY_DTYPE).astype(np.float32))
        m = fms.parity_metrics(lo16, flat, TOL["rel_floor_frac"])
        m.pop("per_position_max_abs", None)
        m["finite_frac_fp32"] = float(torch.isfinite(flat).float().mean())
        m["finite_frac_f16"] = float(torch.isfinite(lo16).float().mean())
        m["max_abs_input"] = float(flat.abs().max())
        m["flush_to_zero_frac"] = float(((lo16 == 0) & (flat != 0)).float().mean())
        # G7 has its own predeclared storage-rounding contract.  Do not route
        # it through G2's vendor-kernel relative metric.
        g7_finite = all(math.isfinite(m[k]) for k in
                        ("max_abs_err", "cosine", "rel_p99", "norm_max_err"))
        v = (PASS if g7_finite
             and m["cosine"] >= G7_TOLS["cos"]
             and m["rel_p99"] <= G7_TOLS["rel_p99"]
             and m["norm_max_err"] <= G7_TOLS["norm_max"]
             else STOP)
        # bf16 round-trip: separate diagnostic, NEVER verdict-bearing
        b = fms.parity_metrics(flat.bfloat16().float(), flat,
                               TOL["rel_floor_frac"])
        m["bf16_diagnostic"] = {k: b[k] for k in
                                ("max_abs_err", "rel_p99", "cosine",
                                 "norm_max_err")}
        per_q[name] = {**m, "within_declared_tolerance": v == PASS}
        if m["finite_frac_fp32"] < 1.0 or m["finite_frac_f16"] < 1.0:
            reasons.append(f"{name}: non-finite values present")
        elif m["max_abs_input"] > G7_TOLS["max_abs_input"]:
            reasons.append(f"{name}: magnitude {m['max_abs_input']:.3g} risks "
                           f"float16 overflow (limit 65504)")
        elif m["flush_to_zero_frac"] > G7_TOLS["flush_frac_max"]:
            reasons.append(f"{name}: {m['flush_to_zero_frac']:.2e} of values "
                           f"flush to zero in float16")
        elif v != PASS:
            reasons.append(f"{name}: float16 storage exceeds the declared "
                           f"G7 tolerance")

    nonfinite = [r for r in reasons if "non-finite" in r]
    data = {
        "role": ("STORAGE-precision guidance only. NOT recurrence parity "
                 "evidence: nothing here tests the kernel recurrence"),
        "layer": li, "input_source": src,
        "storage_dtype_under_test": str(np.dtype(BLOCK_SUMMARY_DTYPE)),
        "declared_tolerances": G7_TOLS,
        "intended_path": ("bf16 kernel activations, fp32 derivation via the "
                          "shared helpers (which upcast internally), float16 "
                          "block-summary storage in Capture Final"),
        "compared": ("fp32-derived quantities vs their float16 round-trip "
                     "(verdict-bearing); bf16 round-trip is a separate "
                     "diagnostic"),
        "per_quantity": per_q, "reasons": reasons,
    }
    if nonfinite:
        data["error"] = "; ".join(nonfinite)
        return record(ctx, "g7", STOP, **data)
    if reasons:
        data["guidance"] = ("float16 block-summary storage measurably moves "
                            "the flagged quantities; keep them fp32 or accept "
                            "the recorded error explicitly")
        data["prohibited"] = [
            "treating float16-stored values of the flagged quantities as "
            "exact fp32 measurements"]
        return record(ctx, "g7", BOUNDARY, **data)
    data["guidance"] = ("float16 block-summary storage adequate within the "
                        "declared G7 tolerances")
    return record(ctx, "g7", PASS, **data)


# --------------------------------------------------------------------------
# main control flow
# --------------------------------------------------------------------------

GATES = {"g1": g1_env, "g2": g2_parity, "g3": g3_step, "g4": g4_jit,
         "g5": g5_resources, "g6": g6_throughput, "g7": g7_precision}


def run_probe(args, archiver, archive_skip_reason="") -> int:
    ctx = new_ctx(args, archiver, archive_skip_reason)
    report = ctx["report"]
    report["manifest"] = assert_runtime(
        {"model": args.model, "requires_kernel": True}, strict=False)

    names = list(GATES) if args.all or not args.gates else args.gates.split(",")
    unknown = [n for n in names if n not in GATES]
    config_errors = []
    if unknown:
        config_errors.append(f"unknown gates {unknown}")
    if torch.cuda.is_available() and archiver is None:
        config_errors.append(
            "--hf-repo is MANDATORY for a real CUDA run: local output is "
            "ephemeral and evidence must be archived to Hugging Face")
    if args.all and not args.token_contract:
        config_errors.append(
            "--all requires --token-contract: G4-G6 are defined by the "
            "validated contract, not an invented uniform sequence length")

    if args.token_contract and not config_errors:
        plan, fails = preflight(args.token_contract, args.stream)
        if fails:
            config_errors.append(f"token contract invalid: {fails}")
        else:
            ctx["contract_plan"] = plan
            report["token_contract"] = {
                "path": args.token_contract, "stream": args.stream,
                "blocks": plan["blocks_expected"],
                "valid_tokens": plan["valid_tokens_expected"],
                "content_tokens": plan["content_tokens_expected"],
                "digest": plan["digest_computed"]}

    if config_errors:
        # single-upload terminal case: the overall verdict is set BEFORE the
        # upload, so the remote copy never claims a pending/unattempted state
        record(ctx, "config", STOP, errors=config_errors)
        report["verdicts"]["overall"] = STOP
        archive_now(ctx, "config", terminal=True)
        return 1

    halted_by = None
    for name in names:
        if halted_by:
            report["verdicts"][name] = f"NOT_RUN (halted by {halted_by})"
            continue
        try:
            v = GATES[name](ctx)
        except Exception as e:  # noqa: BLE001 - every exception still reaches archival
            v = record(ctx, name, STOP,
                       error=f"uncaught {type(e).__name__}: {e}",
                       traceback=traceback.format_exc()[-4000:])
        archive_now(ctx, name)
        # archival failure halts IMMEDIATELY: no further GPU spend while
        # evidence cannot leave the disposable pod. Scientific verdicts stay
        # recorded separately and untouched.
        if ctx["archive_failed"]:
            print(f"\narchival failed after {name} -- halting all later "
                  f"gates; no more GPU spend until evidence can leave the pod")
            halted_by = "archive failure"
        elif v == STOP:
            print(f"\n{name} returned STOP -- halting later scientific gates")
            halted_by = f"{name} STOP"

    ran = [v for k, v in report["verdicts"].items() if not v.startswith("NOT_RUN")]
    overall = (STOP if STOP in ran
               else BOUNDARY if BOUNDARY in ran else PASS)
    report["verdicts"]["overall"] = overall
    report["requested_gates"] = names
    report["executed_gates"] = [k for k in report["gates"] if k != "config"]
    if report["boundary_prohibited_claims"]:
        print("\nBOUNDARY: the following claims are PROHIBITED:")
        for c in report["boundary_prohibited_claims"]:
            print(f"    - {c}")
    archive_now(ctx, "final", terminal=True)

    print(f"\nverdicts: {report['verdicts']}")
    print(f"archive_status: {report['archive']['status']}")
    print(f"wrote {args.out}")
    return 1 if (overall == STOP or ctx["archive_failed"]) else 0


# --------------------------------------------------------------------------
# offline self-checks: no CUDA, no network, mocked archival
# --------------------------------------------------------------------------


def _mk_args(tmp, **kw):
    d = dict(model="mimo-187m", gates=None, all=False, revision=None,
             g2_revision_siso=None, g2_revision_mimo=None,
             token_contract=None, stream="b", layers=None,
             accept_isolated_compile_cost=False,
             prompt="Mamba-3 is", seqlen=256,
             hf_repo=None, hf_repo_type="dataset", hf_path_prefix="",
             out=os.path.join(tmp, "report.json"))
    d.update(kw)
    return argparse.Namespace(**d)


def _g2_pass_fixture():
    case = {"model": "mimo-187m", "layer": 0, "seqlen": 7, "verdict": PASS,
            "max_abs_err": 1e-3, "mean_abs_err": 1e-4, "rel_p50": 1e-3,
            "rel_p99": 1e-2, "rel_max": 2e-2,
            "vendor_rel_p95": 8e-3, "vendor_rel_n": 500,
            "cosine": 0.99999,
            "norm_max_err": 0.01, "pos_first": 1e-3, "pos_final": 1e-3,
            "pos_interior_max": 1e-3, "pos_interior_mean": 5e-4}
    return {"coverage": {"expected_cases": 2, "executed_cases": 2,
                         "plan_errors": [], "complete": True},
            "cases": [dict(case), dict(case, layer=5)],
            "verdicts": {"overall": PASS},
            "provenance": {"mimo-187m": {"path": "/snap", "resolved_commit": "abc"}}}


def self_check() -> bool:
    results = []

    def check(n, desc, ok, detail=""):
        results.append(ok)
        print(f"  [{'ok ' if ok else 'BAD'}] {n:2d}. {desc}"
              + (f" -- {detail}" if detail and not ok else ""))

    tmp = tempfile.mkdtemp(prefix="gpu_probe_selfcheck_")
    print(f"self-check (offline, mocked archival) in {tmp}\n")

    # 1. no stale implementation remains in this file
    with open(os.path.abspath(__file__)) as fh:
        src = fh.read()
    stale = ["CA" + "CHE =", "gl" + "ob.glob", "import gl" + "ob",
             ".cache/hugging" + "face", "Ref" + "Switches",
             "run_mo" + "del("]
    hits = [t for t in stale if t in src]
    check(1, "no stale CACHE/glob/switch/oracle-call tokens remain",
          not hits, str(hits))

    # 2. local-only checkpoint resolution failure is clean
    try:
        resolve_checkpoint("definitely-not-a-cached-model", local_only=True)
        check(2, "local-only resolution failure is clean", False, "did not raise")
    except CheckpointResolveError as e:
        check(2, "local-only resolution failure is clean",
              "local_only=True" in str(e), str(e)[:120])

    # 3-5. the G2 adapter
    topk_only = _g2_pass_fixture()
    for c in topk_only["cases"]:
        for k in G2_TENSOR_KEYS:
            c.pop(k, None)
        c["top5_gpu"] = ["a"]
        c["top1_agree"] = True
    v, reasons = adapt_g2(topk_only, expected_cases=2)
    check(3, "G2 adapter rejects a top-k-only fixture",
          v == STOP and any("top-k" in r for r in reasons))

    empty = {"coverage": {"expected_cases": 0, "executed_cases": 0},
             "cases": [], "verdicts": {"overall": PASS}}
    v, reasons = adapt_g2(empty)
    check(4, "G2 adapter rejects missing/zero case coverage", v == STOP)
    partial = _g2_pass_fixture()
    partial["coverage"]["executed_cases"] = 1
    partial["coverage"]["complete"] = False
    check(4, "G2 adapter rejects incomplete coverage",
          adapt_g2(partial, expected_cases=2)[0] == STOP)
    missing_row = _g2_pass_fixture()
    missing_row["cases"] = missing_row["cases"][:1]     # coverage claims 2/2
    v, reasons = adapt_g2(missing_row, expected_cases=2)
    check(4, "G2 adapter rejects a missing case row",
          v == STOP and any("rows missing" in r for r in reasons))
    dup_row = _g2_pass_fixture()
    dup_row["cases"][1] = dict(dup_row["cases"][0])     # same identity twice
    v, reasons = adapt_g2(dup_row, expected_cases=2)
    check(4, "G2 adapter rejects a duplicate case row",
          v == STOP and any("duplicate" in r.lower() for r in reasons))
    incomplete_flag = _g2_pass_fixture()
    incomplete_flag["coverage"].pop("complete")
    check(4, "G2 adapter requires coverage.complete == true",
          adapt_g2(incomplete_flag, expected_cases=2)[0] == STOP)

    v, reasons = adapt_g2(_g2_pass_fixture(), expected_cases=2)
    check(5, "G2 adapter accepts a complete structured PASS fixture",
          v == PASS, str(reasons))

    # 6. G3 semantics
    check(6, "G3: unavailable step() is BOUNDARY, executed mismatch is STOP",
          g3_verdict(False, None) == BOUNDARY
          and g3_verdict(True, STOP) == STOP
          and g3_verdict(True, PASS) == PASS)

    # 7. G4 timing classifier: per-length bars. The 2048 case is naturally
    # slow (warm 1.5s, first 1.6s) and must NOT be classified as a recompile,
    # which a global warm-median threshold would have done.
    bars, susp, revisit, excess = classify_recompiles(
        {32: 2.6, 64: 2.4, 2048: 1.6},
        {32: 0.05, 64: 0.06, 2048: 1.5},
        {32: 0.05, 64: 2.3, 2048: 1.55})
    warm_ok = classify_recompiles(
        {32: 0.06, 64: 0.07}, {32: 0.05, 64: 0.06}, {32: 0.05, 64: 0.06})
    check(7, "G4 classifier compares each length against its own warm time",
          set(susp) == {32, 64} and 2048 not in susp
          and set(revisit) == {64} and 2048 not in revisit
          and abs(excess[2048] - 0.1) < 1e-9
          and not warm_ok[1] and not warm_ok[2])

    # 8. PASS, STOP and BOUNDARY reports all attempt mocked archival
    arch = MockArchiver()
    for i, verdict in enumerate((PASS, STOP, BOUNDARY)):
        a = _mk_args(tmp, out=os.path.join(tmp, f"r{i}.json"))
        c = new_ctx(a, arch)
        record(c, "gX", verdict, note="fixture")
        archive_now(c, "gX")
    check(8, "PASS/STOP/BOUNDARY reports all attempt mocked archival",
          len(arch.attempts) == 3 and all(x["attempted"] for x in arch.attempts))

    # 9. remote verification failure -> archive_status FAILED, nonzero exit
    a9 = _mk_args(tmp, gates="g1", out=os.path.join(tmp, "r9.json"))
    code9 = run_probe(a9, MockArchiver(verify_ok=False))
    with open(a9.out, "rb") as fh:
        rep9 = _json_loads(fh.read())
    check(9, "verification failure -> archive_status FAILED + nonzero exit",
          code9 != 0 and rep9["archive"]["status"] == "FAILED")

    # 10. no-CUDA execution writes a clean STOP report (and still archives)
    a10 = _mk_args(tmp, gates="g1", out=os.path.join(tmp, "r10.json"))
    arch10 = MockArchiver()
    code10 = run_probe(a10, arch10)
    with open(a10.out, "rb") as fh:
        rep10 = _json_loads(fh.read())
    check(10, "no-CUDA execution writes a clean STOP report",
          (not torch.cuda.is_available()) and code10 != 0
          and rep10["verdicts"].get("g1") == STOP
          and rep10["verdicts"].get("overall") == STOP
          and len(arch10.attempts) >= 1,
          f"exit={code10} verdicts={rep10.get('verdicts')}")

    # 11. atomic report publication
    p = os.path.join(tmp, "atomic.json")
    pub = publish_report({"x": 1, "storage": dict(STORAGE_FLAGS)}, p)
    with open(p, "rb") as fh:
        raw = fh.read()
    check(11, "atomic publication: no partial file, recorded sha matches",
          os.path.isfile(p) and not os.path.exists(p + ".partial")
          and hashlib.sha256(raw).hexdigest() == pub["sha256"]
          and len(raw) == pub["bytes"] and _json_loads(raw)["x"] == 1)

    # 12. no output labels any pod path persistent
    ok12 = True
    for repx in (rep9, rep10):
        s = repx.get("storage") or {}
        ok12 = (ok12 and s.get("local_storage_persistent") is False
                and s.get("upload_required_before_pod_termination") is True
                and repx.get("local_storage_persistent") is False
                and repx.get("upload_required_before_pod_termination") is True)
    check(12, "reports declare pod storage ephemeral, never persistent", ok12)

    # 13. an archival failure halts all later gates immediately, while the
    # scientific verdicts already earned stay recorded separately
    saved = dict(GATES)
    GATES.clear()
    GATES.update({
        "t1": lambda c: record(c, "t1", PASS, note="fixture"),
        "t2": lambda c: record(c, "t2", PASS, note="fixture"),
    })
    try:
        a13 = _mk_args(tmp, gates="t1,t2", out=os.path.join(tmp, "r13.json"))
        code13 = run_probe(a13, MockArchiver(verify_ok=False))
        with open(a13.out, "rb") as fh:
            rep13 = _json_loads(fh.read())
    finally:
        GATES.clear()
        GATES.update(saved)
    check(13, "archive failure halts later gates, verdicts kept separate",
          code13 != 0
          and rep13["verdicts"].get("t1") == PASS
          and str(rep13["verdicts"].get("t2", "")).startswith("NOT_RUN")
          and "archive failure" in str(rep13["verdicts"].get("t2"))
          and rep13["archive"]["status"] == "FAILED",
          f"exit={code13} verdicts={rep13.get('verdicts')}")

    # 14. complete sidecar evidence is uploaded verbatim and verified; a
    # sidecar verification failure also flips the archive-failed halt flag
    side = os.path.join(tmp, "authority_sidecar.json")
    with open(side, "wb") as fh:
        fh.write(_json_bytes({"cases": [{"per_position_max_abs": [1, 2, 3]}]}))
    a14 = _mk_args(tmp, out=os.path.join(tmp, "r14.json"))
    arch14 = MockArchiver()
    c14 = new_ctx(a14, arch14)
    e14 = archive_sidecar(c14, "g2", side)
    c14bad = new_ctx(a14, MockArchiver(verify_ok=False))
    archive_sidecar(c14bad, "g2", side)
    with open(side, "rb") as fh:
        raw14 = fh.read()
    check(14, "G2 sidecar uploads verbatim; sidecar failure halts too",
          e14["verified"]
          and e14["remote_path"].endswith("authority_sidecar.json")
          and e14["sha256"] == hashlib.sha256(raw14).hexdigest()
          and e14["bytes"] == len(raw14)
          and c14["archive_failed"] is False
          and c14bad["archive_failed"] is True)

    # 15. first G2 sidecar archival failure -> the second repository's
    # authority subprocess is NEVER invoked, and the STOP is recorded as an
    # archival matrix-incompleteness, not a parity failure
    def _fix_for(model, commit):
        f = _g2_pass_fixture()
        for c in f["cases"]:
            c["model"] = model
        f["provenance"] = {model: {"path": "/snap", "resolved_commit": commit}}
        return f

    class _FakeCK:
        from_local_dir = False

        def __init__(self, commit):
            self._c = commit

        def provenance(self):
            return {"path": "/snap", "resolved_commit": self._c}

    commits = {"siso-187m": "aaa", "mimo-187m": "bbb"}
    launched = []

    def _fake_run(cmd, **kw):
        launched.append(cmd)
        model = cmd[cmd.index("--models") + 1]
        out = cmd[cmd.index("--out") + 1]
        with open(out, "wb") as fh:
            fh.write(_json_bytes(_fix_for(model, commits[model])))
        return argparse.Namespace(returncode=0, stdout="", stderr="")

    g = globals()
    real_resolve, real_bp, real_sub = g["resolve_checkpoint"], fms.build_plan, subprocess.run
    g["resolve_checkpoint"] = (
        lambda name, revision=None, local_only=True: _FakeCK(commits[name]))
    fms.build_plan = lambda ns: ([{"model": ns.models}], [], 2)
    subprocess.run = _fake_run
    try:
        c15 = new_ctx(_mk_args(tmp, out=os.path.join(tmp, "r15.json")),
                      MockArchiver(verify_ok=False))
        v15 = g2_parity(c15)
        d15 = c15["report"]["gates"]["g2"]
    finally:
        g["resolve_checkpoint"], fms.build_plan, subprocess.run = (
            real_resolve, real_bp, real_sub)
    check(15, "first sidecar failure stops G2 before the second subprocess",
          v15 == STOP and len(launched) == 1
          and d15.get("parity_failure") is False
          and d15.get("evidence_matrix_complete") is False
          and d15.get("stop_reason") == "archival failure after authority sidecar"
          and c15["archive_failed"] is True,
          f"launched={len(launched)} data={ {k: d15.get(k) for k in ('parity_failure', 'stop_reason')} }")

    # 16. initial terminal upload succeeds, FINALIZED upload fails -> nonzero
    # exit and the surviving local report truthfully says FAILED
    saved = dict(GATES)
    GATES.clear()
    GATES.update({"t1": lambda c: record(c, "t1", PASS, note="fixture")})
    try:
        a16 = _mk_args(tmp, gates="t1", out=os.path.join(tmp, "r16.json"))
        code16 = run_probe(a16, MockArchiver(verify_ok=[True, True, False]))
        with open(a16.out, "rb") as fh:
            rep16 = _json_loads(fh.read())
        fin16 = [e for e in rep16["archive"]["log"]
                 if e.get("gate") == "final(finalized)"]

        # 17. scientific STOP with healthy archival: the two failure kinds
        # stay distinct in both directions
        GATES.clear()
        GATES.update({"ts": lambda c: record(c, "ts", STOP, note="fixture")})
        a17 = _mk_args(tmp, gates="ts", out=os.path.join(tmp, "r17.json"))
        code17 = run_probe(a17, MockArchiver(verify_ok=True))
        with open(a17.out, "rb") as fh:
            rep17 = _json_loads(fh.read())
    finally:
        GATES.clear()
        GATES.update(saved)
    check(16, "finalized-upload failure -> FAILED on disk + nonzero exit",
          code16 != 0 and rep16["archive"]["status"] == "FAILED"
          and rep16["verdicts"].get("t1") == PASS
          and len(fin16) == 1 and fin16[0]["verified"] is False,
          f"exit={code16} status={rep16['archive']['status']}")
    check(17, "parity STOP with healthy archival stays distinct",
          code17 != 0 and rep17["verdicts"].get("ts") == STOP
          and rep17["archive"]["status"] == "OK")

    ok = all(results)
    print(f"\nself-check {'PASSED' if ok else 'FAILED'} "
          f"({sum(results)}/{len(results)})")
    return ok


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="G1-G7 GPU probe suite (see module docstring)")
    ap.add_argument("--model", default="mimo-187m")
    ap.add_argument("--gates", default=None, help="comma list, e.g. g1,g2")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--revision", default=None,
                    help="exact checkpoint revision for --model/G1 ONLY; "
                         "resolution stays local-only")
    ap.add_argument("--g2-revision-siso", default=None,
                    help="exact commit for the G2 siso-187m repository; "
                         "defaults to its locally resolved snapshot commit")
    ap.add_argument("--g2-revision-mimo", default=None,
                    help="exact commit for the G2 mimo-187m repository; "
                         "defaults to its locally resolved snapshot commit")
    ap.add_argument("--token-contract", default=None,
                    help="validated token_contract.npz; REQUIRED with --all")
    ap.add_argument("--stream", default="b", choices=("a", "b"))
    ap.add_argument("--layers", default=None,
                    help="comma list of capture layers for G5/G7; default all")
    ap.add_argument("--accept-isolated-compile-cost", action="store_true",
                    help="explicitly accept measured JIT overhead and retain "
                         "the validated isolated/unpadded capture policy")
    ap.add_argument("--prompt", default="Mamba-3 is")
    ap.add_argument("--seqlen", type=int, default=256,
                    help="G1 warmup length only; capture gates use the contract")
    ap.add_argument("--hf-repo", default=None,
                    help="Hugging Face repo for archival; MANDATORY on CUDA")
    ap.add_argument("--hf-repo-type", default="dataset")
    ap.add_argument("--hf-path-prefix", default="")
    ap.add_argument("--out", default="gpu_probe_report.json")
    ap.add_argument("--self-check", action="store_true",
                    help="offline checks: no CUDA, no network, mocked archival")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if self_check() else 1)

    if args.hf_repo:
        archiver = HFArchiver(args.hf_repo, args.hf_repo_type,
                              args.hf_path_prefix, os.path.basename(args.out))
        skip = ""
    else:
        archiver = None
        skip = ("no --hf-repo supplied; permitted only for CPU-local dry runs "
                "(a real CUDA run refuses to start without it)")
    sys.exit(run_probe(args, archiver, skip))


if __name__ == "__main__":
    main()
