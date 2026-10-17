import json
import subprocess
import sys
from pathlib import Path

primary = Path("experiments/xp/launch-next-e47-20261013.json")
record = Path("experiments/xp/launch-next-e47-spectral-r2-20261013.json")
r = json.loads(primary.read_text())
job = next(j for j in r["jobs"] if j["run_id"] == "spectral_auxiliary_mae")
assert job.get("retired_session_ids") == ["x9bn27im"], (
    "Prior spectral session was not confirmed stopped"
)
sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
cmd = [
    sys.executable,
    "scripts/launch_xp_experiments.py",
    "--checkpoint",
    r["checkpoint"],
    "--output-root",
    r["output_root"] + "-spectral-r2",
    "--git-ref",
    r["git_ref"],
    "--git-sha",
    sha,
    "--experiments",
    r["experiments_config"],
    "--benchmark",
    r["benchmark"],
    "--train-stars",
    str(r["train_stars"]),
    "--micro-batch-size",
    str(r["micro_batch_size"]),
    "--record",
    str(record),
    "--runs",
    "spectral_auxiliary_mae",
]
if record.exists():
    cmd += ["--resume-submissions"]
subprocess.run(cmd)
if record.exists():
    replacement = next(
        j
        for j in json.loads(record.read_text())["jobs"]
        if j["run_id"] == "spectral_auxiliary_mae"
    )
    if replacement["session_ids"]:
        job.update(replacement)
        job["git_sha"] = sha
        job["replacement_launch_record"] = str(record)
        primary.write_text(json.dumps(r, indent=2))
print("SPECTRAL", job["session_ids"], flush=True)
