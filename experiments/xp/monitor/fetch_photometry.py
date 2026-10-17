"""Read-only ARC preservation for the photometric consistency suite."""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--preflight", action="store_true")
args = parser.parse_args()
record = Path(
    "experiments/xp/launch-photometry-e47-20261013"
    + ("-preflight-r2" if args.preflight else "")
    + ".json"
)
jobs = json.loads(record.read_text())["jobs"]
root = Path("experiments/xp/results-photometry-e47")


def fetch(task):
    j, rel = task
    folder = root / ("preflight" if args.preflight else j["run_id"])
    dest = folder / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith("residual_init.npz") and dest.exists():
        return j["run_id"], rel, "already preserved"
    tmp = dest.with_suffix(dest.suffix + ".download")
    remote = j["output"] + "/" + rel
    result = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "120" if rel.endswith(".npz") else "30",
            "--cert",
            "/Users/fabbros/.ssl/cadcproxy.pem",
            "https://ws-uv.canfar.net/arc/files" + remote.removeprefix("/arc"),
            "--output",
            str(tmp),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return j["run_id"], rel, "not available yet"
    if rel.endswith(".npz"):
        with np.load(tmp, allow_pickle=False) as arrays:
            assert "common_prediction" in arrays.files
        tmp.replace(dest)
        return j["run_id"], rel, "saved"
    value = (
        [json.loads(x) for x in tmp.read_text().splitlines() if x.strip()]
        if rel.endswith(".jsonl")
        else json.loads(tmp.read_text())
    )
    tmp.replace(dest)
    if args.preflight and rel == "manifest.json" and value.get("status") == "complete":
        Path("experiments/xp/photometry-preflight-manifest.json").write_text(
            json.dumps(value, indent=2) + "\n"
        )
    if rel == "manifest.json":
        return j["run_id"], rel, value["status"], value.get("error")
    if rel.endswith("metrics.jsonl"):
        return (
            j["run_id"],
            rel,
            [
                {
                    k: x.get(k)
                    for k in [
                        "phase",
                        "rows_seen_total",
                        "optimizer_steps",
                        "xp_only_mae",
                    ]
                }
                for x in value
            ],
        )
    if rel == "photometry_calibration.json":
        return (
            j["run_id"],
            rel,
            value["calibration"]["count"],
            value["observed_xp_baseline"],
        )
    return j["run_id"], rel, "saved"


tasks = []
for j in jobs:
    rels = ["manifest.json", "photometry_calibration.json"]
    if not args.preflight:
        rels += [
            j["run_id"] + "/" + x
            for x in [
                "residual_init.npz",
                "metrics.jsonl",
                "residual_latest.json",
                "checkpoint_verification.json",
                "optimizer_policy.json",
            ]
        ]
    tasks.extend((j, r) for r in rels)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(fetch, tasks):
        print(json.dumps(result), flush=True)
