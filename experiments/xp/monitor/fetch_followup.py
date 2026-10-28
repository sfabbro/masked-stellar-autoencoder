"""Preserve authorized follow-up results through authenticated read-only ARC GETs."""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def fetch(task):
    job, relative, destination = task
    destination.parent.mkdir(parents=True, exist_ok=True)
    if relative.endswith("residual_init.npz") and destination.exists():
        return job["run_id"], relative, "preserved"
    remote = job["output"] + (
        ".session.log" if relative == "session.log" else "/" + relative
    )
    temporary = destination.with_suffix(destination.suffix + ".download")
    response = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "120" if relative.endswith(".npz") else "25",
            "--cert",
            "/Users/fabbros/.ssl/cadcproxy.pem",
            "https://ws-uv.canfar.net/arc/files" + remote.removeprefix("/arc"),
            "--output",
            str(temporary),
        ],
        capture_output=True,
        text=True,
    )
    if response.returncode:
        temporary.unlink(missing_ok=True)
        return job["run_id"], relative, "not_available"
    try:
        if relative.endswith(".json"):
            content = json.loads(temporary.read_text())
        elif relative.endswith(".jsonl"):
            content = [
                json.loads(line) for line in temporary.read_text().splitlines() if line
            ]
        else:
            content = None
    except ValueError:
        temporary.unlink(missing_ok=True)
        return job["run_id"], relative, "incomplete_response"
    temporary.replace(destination)
    status = content.get("status", "saved") if isinstance(content, dict) else "saved"
    return job["run_id"], relative, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        nargs="+",
        default=[
            "experiments/xp/launch-followup-e47-20261014.json",
            "experiments/xp/launch-followup-analysis-e47-20261014.json",
        ],
    )
    parser.add_argument("--initial-predictions", action="store_true")
    parser.add_argument("--runs", nargs="+")
    args = parser.parse_args()
    tasks = []
    for record in args.records:
        if not Path(record).exists():
            continue
        for job in json.loads(Path(record).read_text())["jobs"]:
            if args.runs and job["run_id"] not in args.runs:
                continue
            if not job["session_ids"]:
                continue
            files = ["manifest.json", "session.log"]
            if job["settings"]["task"] == "training":
                files += ["photometry_calibration.json"]
                files += [
                    job["run_id"] + "/" + name
                    for name in [
                        "metrics.jsonl",
                        "residual_latest.json",
                        "checkpoint_verification.json",
                        "optimizer_policy.json",
                        "dropout_policy.json",
                    ]
                ]
                if args.initial_predictions:
                    files += [job["run_id"] + "/residual_init.npz"]
                if job["settings"].get("latent_weight", 0):
                    files += [job["run_id"] + "/latent_calibration.json"]
            else:
                for split in job["settings"].get("splits", ["identity"]):
                    files += [
                        split + "/" + name
                        for name in [
                            "manifest.json",
                            "results.json",
                            "fits_units_audit.json",
                        ]
                    ]
                    if args.initial_predictions:
                        name = (
                            "probe_predictions.npz"
                            if job["settings"]["mode"] == "probes"
                            else "latent_refinement.npz"
                        )
                        files += [split + "/" + name]
            folder = Path("experiments/xp/results-followup-e47") / job["run_id"]
            tasks.extend((job, relative, folder / relative) for relative in files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(fetch, tasks):
            if result[1] == "manifest.json" or result[2] not in {
                "saved",
                "preserved",
                "not_available",
            }:
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
