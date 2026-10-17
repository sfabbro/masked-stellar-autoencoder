"""Bounded sequential local CPU verification of already collected checkpoints."""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/xp/results-stellar-improvement"
JOBS = json.loads(
    (ROOT / "experiments/xp/launch-stellar-improvement-20261015.json").read_text()
)["jobs"]
JOBS = [j for j in JOBS if j["run_id"].startswith("prediction_e72")]
STATE = BASE / "prediction-comparison-e72/cpu-replay-progress.json"
state = {
    "status": "running",
    "started_at_utc": datetime.now(UTC).isoformat(),
    "driver_pid": os.getpid(),
    "jobs": {},
}
# The first control is already running in exec session71369/PID42115. Never duplicate it.
first = "prediction_e72_huber_10m_v1"
state["jobs"][first] = {
    "status": "external_verifier_running",
    "exec_session": 71369,
    "pid": 42115,
}


def save():
    for job in JOBS:
        p = BASE / job["run_id"] / "independent-verification.json"
        if p.exists():
            report = json.loads(p.read_text())
            if report.get("status") == "passed":
                state["jobs"][job["run_id"]] = {"status": "passed", "report": str(p)}
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2) + "\n")
    temp.replace(STATE)


save()
for job in JOBS:
    run = job["run_id"]
    if run == first or state["jobs"].get(run, {}).get("status") == "passed":
        continue
    directory = BASE / run
    assert (directory / "checkpoint.pth").exists(), run
    assert json.loads((directory / "manifest.json").read_text())["status"] == "complete"
    log = directory / "independent-verification.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "experiments/xp/monitor/verify_prediction_comparison.py"),
                str(directory),
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        state["jobs"][run] = {"status": "running", "pid": process.pid, "log": str(log)}
        save()
        print(
            json.dumps(
                {
                    "run_id": run,
                    "status": "CPU_verification_running",
                    "pid": process.pid,
                }
            ),
            flush=True,
        )
        code = process.wait()
    if code:
        state["jobs"][run] = {
            "status": "failed_local_verification",
            "exit_code": code,
            "log": str(log),
        }
    save()
    print(json.dumps({"run_id": run, **state["jobs"][run]}), flush=True)
state["status"] = (
    "passed"
    if all(state["jobs"].get(j["run_id"], {}).get("status") == "passed" for j in JOBS)
    else "attention_or_external_verifier_pending"
)
state["finished_at_utc"] = datetime.now(UTC).isoformat()
save()
print(json.dumps({"status": state["status"], "state_path": str(STATE)}), flush=True)
