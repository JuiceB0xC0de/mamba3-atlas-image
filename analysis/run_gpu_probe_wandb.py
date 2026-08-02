#!/usr/bin/env python3
"""Run GPU Probe inside a live W&B run and preserve its terminal evidence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import wandb
from huggingface_hub import get_token


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--entity", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--build-id", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command:
        p.error("probe command is required after --")

    run = wandb.init(
        entity=a.entity,
        project=a.project,
        name=f"gpu-probe-{a.build_id}",
        group=a.build_id,
        job_type="gpu-probe",
        tags=["mamba3", "stage-b", "gpu-probe", "h100"],
        config={
            "atlas_build_id": a.build_id,
            "stage": "gpu-probe",
            "script": "analysis/gpu_probe.py",
            "gpu": "NVIDIA H100 80GB HBM3",
            "probe_command": command,
        },
    )
    print(f"W&B LIVE: {run.url}", flush=True)

    env = os.environ.copy()
    if not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = get_token() or ""
    returncode = subprocess.run(command, env=env, check=False).returncode

    report_path = Path(a.report)
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text())
        verdicts = report.get("verdicts", {})
        gates = report.get("gates", {})
        table = wandb.Table(
            columns=["gate_index", "gate", "verdict", "details_json"]
        )
        for index, (gate, verdict) in enumerate(verdicts.items()):
            if gate == "overall":
                continue
            table.add_data(
                index,
                gate,
                verdict,
                json.dumps(gates.get(gate, {}), sort_keys=True, default=str),
            )
        run.log({"gate_verdicts": table})
        run.summary["probe/overall"] = verdicts.get("overall", "UNKNOWN")
        archive = report.get("archive", {})
        run.summary["archive/status"] = archive.get("status", "UNKNOWN")

        artifact = wandb.Artifact(
            name=f"gpu-probe-report-{a.build_id}",
            type="validation",
            metadata={
                "atlas_build_id": a.build_id,
                "overall": verdicts.get("overall", "UNKNOWN"),
                "archive_status": archive.get("status", "UNKNOWN"),
            },
        )
        artifact.add_file(str(report_path))
        for sidecar in sorted(report_path.parent.glob(report_path.name + ".g2_*.json")):
            artifact.add_file(str(sidecar))
        run.log_artifact(artifact)

    run.summary["process/returncode"] = returncode
    run.summary["evidence/report_present"] = report is not None
    run.finish(exit_code=returncode)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
