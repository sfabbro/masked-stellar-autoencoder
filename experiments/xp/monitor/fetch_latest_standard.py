"""Preserve the two authorized latest-standard comparison outputs."""

import argparse
import concurrent.futures
import json
from pathlib import Path

from fetch_followup import fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epoch", type=int, default=57)
    args = parser.parse_args()
    record = json.loads(
        Path(
            f"experiments/xp/launch-latest-standard-e{args.epoch}-20261014.json"
        ).read_text()
    )
    tasks = []
    for job in record["jobs"]:
        if not job["session_ids"]:
            continue
        files = ["manifest.json", "session.log"]
        if job["task"] == "analysis":
            for split in ["identity", "sky"]:
                files += [
                    split + "/" + name
                    for name in [
                        "manifest.json",
                        "results.json",
                        "fits_units_audit.json",
                        "probe_predictions.npz",
                    ]
                ]
        else:
            files += ["photometry_calibration.json"]
            files += [
                job["run_id"] + "/" + name
                for name in [
                    "metrics.jsonl",
                    "residual_latest.json",
                    "residual_latest.npz",
                    "evaluation_verification.json",
                ]
            ]
        root = (
            Path(f"experiments/xp/results-latest-standard-e{args.epoch}")
            / job["run_id"]
        )
        tasks += [(job, name, root / name) for name in files]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for value in pool.map(fetch, tasks):
            if value[1].endswith("manifest.json"):
                print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
