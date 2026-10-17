"""Submit only missing experiment IDs, preserving accepted CANFAR sessions."""

import json
import subprocess
import sys
from pathlib import Path

record = Path("experiments/xp/launch-next-e47-20261013.json")
sha = (
    json.loads(record.read_text())["git_sha"]
    if record.exists()
    else subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
)
command = [
    sys.executable,
    "scripts/launch_xp_experiments.py",
    "--checkpoint",
    "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_47.pth",
    "--output-root",
    "/arc/projects/k-pop/msa_runs/xp-next-e47-20261013",
    "--git-ref",
    "codex/xp-reconstruction-experiments",
    "--git-sha",
    sha,
    "--experiments",
    "configs/pretrain.xp-next.yaml",
    "--benchmark",
    "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5",
    "--train-stars",
    "9000000",
    "--micro-batch-size",
    "512",
    "--record",
    str(record),
]
names = [
    "unchanged_control",
    "corrected_signal_weighted_mae",
    "spectral_auxiliary_mae",
    "adam_reset_mae",
    "adam_reset_lr2_mae",
]
for name in names:
    if record.exists() and any(
        j["run_id"] == name and j["session_ids"]
        for j in json.loads(record.read_text())["jobs"]
    ):
        continue
    if record.exists() and any(
        j["run_id"] == name and j.get("retired_session_ids")
        for j in json.loads(record.read_text())["jobs"]
    ):
        continue
    print("SUBMIT", name, flush=True)
    subprocess.run(
        command
        + (["--resume-submissions"] if record.exists() else [])
        + ["--runs", name]
    )
print(
    "ACCEPTED",
    [(j["run_id"], j["session_ids"]) for j in json.loads(record.read_text())["jobs"]],
    flush=True,
)
