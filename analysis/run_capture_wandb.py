#!/usr/bin/env python3
"""Run Stage-B capture with live W&B progress and verified HF evacuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import wandb
from huggingface_hub import HfApi, get_token


PROGRESS_RE = re.compile(
    r"block\s+(\d+)/(\d+)\s+([\d,]+)\s+tokens\s+([\d.]+)s"
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def upload_verified(api: HfApi, token: str | None, repo: str, repo_type: str,
                    local: Path, remote: str) -> dict:
    api.upload_file(
        path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo,
        repo_type=repo_type, token=token,
    )
    info = api.get_paths_info(
        repo_id=repo, paths=[remote], repo_type=repo_type, token=token,
    )
    remote_size = int(info[0].size) if info else None
    size = local.stat().st_size
    verified = remote_size == size
    return {
        "local": str(local), "remote": remote, "bytes": size,
        "remote_bytes": remote_size, "sha256": sha256_file(local),
        "verified": verified,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--build-id", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--hf-repo", required=True)
    p.add_argument("--hf-repo-type", default="dataset")
    p.add_argument("--hf-path-prefix", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("capture command is required after --")

    run = wandb.init(
        entity=a.entity, project=a.project,
        name=f"capture-{a.build_id}", group=a.build_id,
        job_type="activation-capture",
        tags=["mamba3", "stage-b", "capture", "rtx6000ada", "stream-b"],
        config={
            "atlas_build_id": a.build_id,
            "stage": "activation-capture",
            "script": "analysis/capture_stage_b.py",
            "capture_command": command,
            "storage": "ephemeral pod staging; HF canonical",
        },
    )
    print(f"W&B LIVE: {run.url}", flush=True)

    token = get_token() or os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        hf_identity = api.whoami(token=token)
    except Exception as e:  # noqa: BLE001
        run.summary["archive/hf_verified"] = False
        run.finish(exit_code=1)
        print(f"[STOP] Hugging Face authentication unavailable: "
              f"{type(e).__name__}: {e}", flush=True)
        return 1
    run.summary["archive/hf_identity"] = hf_identity.get("name", "authenticated")
    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token

    progress_rows: list[list[float]] = []
    t0 = time.time()
    proc = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = PROGRESS_RE.search(line)
        if not m:
            continue
        done, total = int(m.group(1)), int(m.group(2))
        tokens, elapsed = int(m.group(3).replace(",", "")), float(m.group(4))
        fraction = done / total if total else 0.0
        blocks_s = done / elapsed if elapsed > 0 else 0.0
        tokens_s = tokens / elapsed if elapsed > 0 else 0.0
        progress_rows.append([done, total, tokens, elapsed, fraction,
                              blocks_s, tokens_s])
        run.log({
            "progress/blocks_processed": done,
            "progress/blocks_total": total,
            "progress/blocks_fraction": fraction,
            "progress/valid_tokens_processed": tokens,
            "runtime/elapsed_s": elapsed,
            "runtime/blocks_per_s": blocks_s,
            "runtime/tokens_per_s": tokens_s,
        }, step=done)
    returncode = proc.wait()

    capture = Path(a.capture)
    manifest = Path(str(capture).replace(".npz", ".manifest.json"))
    receipts: list[dict] = []
    archive_ok = False
    if returncode == 0 and capture.exists() and manifest.exists():
        prefix = a.hf_path_prefix.strip("/")
        try:
            for local in (manifest, capture):
                remote = f"{prefix}/{local.name}"
                rec = upload_verified(api, token, a.hf_repo,
                                      a.hf_repo_type, local, remote)
                receipts.append(rec)
                if not rec["verified"]:
                    raise RuntimeError(f"remote byte-size mismatch for {remote}")
            archive_ok = True
        except Exception as e:  # noqa: BLE001
            receipts.append({"verified": False,
                             "error": f"{type(e).__name__}: {e}"})

    receipt_path = capture.with_suffix(".archive.json")
    receipt = {
        "atlas_build_id": a.build_id,
        "capture_returncode": returncode,
        "capture_present": capture.exists(),
        "manifest_present": manifest.exists(),
        "hf_repo": a.hf_repo,
        "hf_repo_type": a.hf_repo_type,
        "hf_path_prefix": a.hf_path_prefix,
        "hf_verified": archive_ok,
        "receipts": receipts,
        "wall_s": time.time() - t0,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    if archive_ok:
        remote = f"{a.hf_path_prefix.strip('/')}/{receipt_path.name}"
        receipts.append(upload_verified(api, token, a.hf_repo,
                                        a.hf_repo_type, receipt_path, remote))

    columns = ["blocks_processed", "blocks_total", "valid_tokens_processed",
               "elapsed_s", "blocks_fraction", "blocks_per_s", "tokens_per_s"]
    table = wandb.Table(columns=columns, data=progress_rows)
    run.log({
        "capture_progress": table,
        "capture_throughput": wandb.plot.line(
            table, "blocks_processed", "tokens_per_s",
            title="Stage B capture throughput",
        ),
    })
    run.summary["capture/returncode"] = returncode
    run.summary["capture/wall_s"] = receipt["wall_s"]
    run.summary["capture/progress_points"] = len(progress_rows)
    run.summary["archive/hf_verified"] = archive_ok
    run.summary["archive/hf_repo"] = a.hf_repo
    if capture.exists():
        run.summary["capture/bytes"] = capture.stat().st_size
        run.summary["capture/sha256"] = sha256_file(capture)

    artifact = wandb.Artifact(
        name=f"stage-b-capture-{a.build_id}", type="capture",
        metadata={
            "atlas_build_id": a.build_id,
            "hf_verified": archive_ok,
            "hf_repo": a.hf_repo,
            "hf_path_prefix": a.hf_path_prefix,
            "raw_capture_duplicated_to_wandb": False,
        },
    )
    artifact.add_file(str(receipt_path))
    if manifest.exists():
        artifact.add_file(str(manifest))
    run.log_artifact(artifact)

    ok = returncode == 0 and archive_ok
    if not ok:
        print("DO NOT TERMINATE POD -- capture or HF verification failed", flush=True)
    run.finish(exit_code=0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
