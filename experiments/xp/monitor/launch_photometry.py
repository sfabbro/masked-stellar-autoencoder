"""Idempotently launch authorised photometric-consistency experiments."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--preflight", action="store_true")
args = parser.parse_args()
record = Path(
    "experiments/xp/launch-photometry-e47-20261013"
    + ("-preflight-r2" if args.preflight else "")
    + ".json"
)
root = "/arc/projects/k-pop/msa_runs/xp-photometry-e47-20261013" + (
    "-preflight-r2" if args.preflight else ""
)
sha = (
    json.loads(record.read_text())["git_sha"]
    if record.exists()
    else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
)
cmd = [
    sys.executable,
    "scripts/launch_xp_experiments.py",
    "--checkpoint",
    "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_47.pth",
    "--output-root",
    root,
    "--git-ref",
    "codex/xp-reconstruction-experiments",
    "--git-sha",
    sha,
    "--experiments",
    "configs/pretrain.xp-photometry.yaml",
    "--benchmark",
    "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5",
    "--train-stars",
    "9000000",
    "--micro-batch-size",
    "512",
    "--record",
    str(record),
]
if args.preflight:
    cmd += ["--photometry-preflight"]
else:
    preflight = json.loads(
        Path("experiments/xp/photometry-preflight-manifest.json").read_text()
    )
    assert preflight["status"] == "complete" and preflight["preflight_only"]
    assert (
        preflight["benchmark_source_id_sha256"]
        == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
    )
    assert preflight["git_sha"] == sha, "Preflight source must match GPU source"
names = (
    ["unchanged_control"]
    if args.preflight
    else [
        "unchanged_control",
        "spectral_auxiliary_mae",
        "observed_photometry_mae",
        "photometry_spectral_mae",
        "anchored_decoder_consistency_mae",
    ]
)
for name in names:
    if record.exists() and any(
        j["run_id"] == name and j["session_ids"]
        for j in json.loads(record.read_text())["jobs"]
    ):
        continue
    subprocess.run(
        cmd + (["--resume-submissions"] if record.exists() else []) + ["--runs", name],
        check=False,
    )
print(
    "ACCEPTED",
    [(j["run_id"], j["session_ids"]) for j in json.loads(record.read_text())["jobs"]],
    flush=True,
)
