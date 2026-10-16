"""Preserve versioned stellar-improvement outputs via authenticated ARC reads."""

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def fetch(task):
    job, relative, out = task
    out.parent.mkdir(parents=True, exist_ok=True)
    if relative.endswith(".npz") and out.exists():
        return job["run_id"], relative, "preserved"
    remote = job["output"] + (
        ".session.log" if relative == "session.log" else "/" + relative
    )
    temp = out.with_suffix(out.suffix + ".download")
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
            str(temp),
        ],
        capture_output=True,
        text=True,
    )
    if response.returncode:
        temp.unlink(missing_ok=True)
        return job["run_id"], relative, "not_available"
    try:
        if relative.endswith(".json"):
            json.loads(temp.read_text())
        elif relative.endswith(".jsonl"):
            for line in temp.read_text().splitlines():
                json.loads(line)
    except ValueError:
        temp.unlink(missing_ok=True)
        return job["run_id"], relative, "incomplete_response"
    temp.replace(out)
    return job["run_id"], relative, "saved"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--record",
        type=Path,
        default=Path("experiments/xp/launch-stellar-improvement-20261015.json"),
    )
    p.add_argument("--runs", nargs="+")
    p.add_argument("--initial-predictions", action="store_true")
    args = p.parse_args()
    tasks = []
    for job in json.loads(args.record.read_text())["jobs"]:
        if not job["session_ids"] or args.runs and job["run_id"] not in args.runs:
            continue
        task = job["plan"]["task"]
        files = ["manifest.json", "session.log"]
        if task == "catalog":
            files += ["inventory.json", "collection.json", "dataset-design.json"]
            if job["plan"].get("mode") == "age-feature-cohort":
                files += ["age-cohort-report.json"]
            if job["plan"].get("mode") == "enriched-label-view":
                files += [
                    "enriched-view-report.json",
                    "li2025-silver/ingestion-report.json",
                ]
            if job["plan"].get("mode") == "survey-overlap":
                files += [
                    "input-provenance.json",
                    "overlap-summary.json",
                    "overlap-pairs.npz",
                ]
        elif task == "native-age":
            files += [
                "metrics.jsonl",
                "paired-cohort.npz",
                "predictions.npz",
                "checkpoint_verification.json",
            ]
            if args.initial_predictions:
                files += ["initial_predictions.npz"]
        elif task == "pristine-value":
            files += [
                "comparison.json",
                "paired-cohort.npz",
                "predictions.npz",
                "checkpoint_verification.json",
            ]
            files += [
                cohort + "__" + variant + "-head.npz"
                for cohort in [
                    "strict_benchmark_train",
                    "broad_excluding_benchmark_validation",
                ]
                for variant in ["encoder", "encoder_gaia", "encoder_gaia_cahk"]
            ]
            if args.initial_predictions:
                files += ["initial_predictions.npz"]
        elif task == "stellar-labels":
            files += [
                "metrics.jsonl",
                "predictions.npz",
                "checkpoint_verification.json",
            ]
            if args.initial_predictions:
                files += ["initial_predictions.npz"]
        else:
            files += [
                job["run_id"] + "/" + name
                for name in [
                    "metrics.jsonl",
                    "residual_latest.json",
                    "checkpoint_verification.json",
                    "optimizer_policy.json",
                    "training_pairing.json",
                ]
            ]
            if args.initial_predictions:
                files += [job["run_id"] + "/residual_init.npz"]
        root = Path("experiments/xp/results-stellar-improvement") / job["run_id"]
        tasks.extend((job, file, root / file) for file in files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for result in pool.map(fetch, tasks):
            if result[1] == "manifest.json":
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
