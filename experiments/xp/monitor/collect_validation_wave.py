"""Missing-only bounded collector for the immutable 51-run validation wave."""

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import yaml

BASE = Path("experiments/xp/results-stellar-improvement")
SOURCE = "a45ae15759fa158026ff6473b0e02816bb828802"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(path, suffix=None):
    suffix = suffix or path.suffix
    if suffix == ".json":
        json.loads(path.read_text())
    elif suffix == ".jsonl":
        for line in path.read_text().splitlines():
            json.loads(line)
    elif suffix == ".npz":
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"NPZ CRC failed: {bad}")


def inventory(task, manifest, metrics=()):
    if task == "multilabel-validation":
        files = [
            "paired-cohort.npz",
            "predictions.npz",
            "checkpoint.pth",
            "conformal.npz",
            "initial_reconstruction.npz",
            "final_reconstruction.npz",
            "metrics.jsonl",
            "native-provenance.json",
            "xp-unit-audit.json",
        ]
        if manifest["arm"]["kind"] == "flow":
            files += ["flow-moment-memory-preflight.json"]
        if manifest.get("joint_density_conformal"):
            files += [
                "density-calibration-summary.json",
                "density-calibration-arrays.npz",
            ]
        # Read actual producer counters rather than guessing the short boundary batch.
        for row in metrics:
            if row.get("snapshot"):
                files += [f"snapshot-{row['presentations']}.npz"]
        if not any(f.startswith("snapshot-") for f in files):
            files += [
                f"snapshot-{n}.npz"
                for n in (32768, 3014656, 6029312, 9032768, 10000000)
            ]
        return files
    return {
        "density-rescore": [
            "immutable-scoring-policy.json",
            "corrected/manifest.json",
            "corrected/predictions-fp64.npz",
        ],
        "conformal-validation": ["role-identities.npz", "report.json", "metrics.csv"],
        "target-head-pilot": ["checkpoint.pth", "predictions.npz"],
        "catalogue-comparison": ["report.json", "paired-metrics.csv"],
    }[task]


def fetch(job, relative, base=BASE):
    out = base / job["run_id"] / relative
    out.parent.mkdir(parents=True, exist_ok=True)
    expected = job.get("checkpoint_sha256") if relative == "checkpoint.pth" else None

    def receipt(status):
        validate(out)
        sha = digest(out)
        if expected and sha != expected:
            raise ValueError(f"Checkpoint SHA mismatch: {out}")
        return dict(
            run_id=job["run_id"],
            file=relative,
            status=status,
            sha256=sha,
            bytes=out.stat().st_size,
        )

    if out.exists():
        return receipt("preserved")
    temp = out.with_name(out.name + ".download")
    for attempt in range(2):
        result = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--max-time",
                "180" if out.suffix in {".npz", ".pth"} else "30",
                "--cert",
                "/Users/fabbros/.ssl/cadcproxy.pem",
                "https://ws-uv.canfar.net/arc/files"
                + job["output"].removeprefix("/arc")
                + "/"
                + relative,
                "--output",
                str(temp),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            validate(temp, out.suffix)
            break
        temp.unlink(missing_ok=True)
    else:
        return dict(
            run_id=job["run_id"],
            file=relative,
            status="transport_unavailable",
            error=result.stderr[-500:],
        )
    temp.replace(out)
    return receipt("saved")


def deduplicate_pairs(base, receipts):
    """Hardlink only locally verified equal-SHA files; no inferred remote identity."""
    canonical = {}
    links = []
    for receipt in receipts:
        if receipt["file"] != "paired-cohort.npz" or "sha256" not in receipt:
            continue
        path = base / receipt["run_id"] / receipt["file"]
        sha = digest(path)
        if sha != receipt["sha256"]:
            raise ValueError("Paired file changed after preservation")
        if sha not in canonical:
            canonical[sha] = path
            continue
        source = canonical[sha]
        if not os.path.samefile(source, path):
            temporary = path.with_name(path.name + ".hardlink")
            os.link(source, temporary)
            temporary.replace(path)
        links.append(dict(path=str(path), canonical=str(source), sha256=sha))
    return links


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=BASE)
    args = p.parse_args()
    state = args.base / "validation-collection-20261023"
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / "collector.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = yaml.safe_load(
        Path("configs/validation-launch-wave-20261022.yaml").read_text()
    )["jobs"]
    ledger = {
        j["run_id"]: j
        for j in json.loads(
            Path("experiments/xp/launch-stellar-improvement-20261015.json").read_text()
        )["jobs"]
    }
    receipts = []

    def save():
        (state / "fetch-progress.json").write_text(
            json.dumps(dict(updated=time.time(), receipts=receipts), indent=2) + "\n"
        )

    def batch(tasks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch, j, f, args.base) for j, f in tasks]
            for future in concurrent.futures.as_completed(futures):
                receipts.append(future.result())
                save()

    jobs = []
    for entry in plan:
        job = dict(ledger[entry["run_id"]], task=entry["task"])
        # Producer-complete metadata was already preserved by the bounded monitor.
        meta = (
            args.base
            / "validation-runtime-20261023T0405"
            / (
                job["run_id"]
                + (
                    ".report.json"
                    if job["task"] in {"conformal-validation", "catalogue-comparison"}
                    else ".json"
                )
            )
        )
        manifest = json.loads(meta.read_text())
        if manifest["status"] not in {
            "complete",
            "complete_scoring_only",
            "bounded_frozen_embedding_pilot_complete",
            "evaluation_only_complete",
            "paired_evaluation_only_complete",
        }:
            raise ValueError(f"Not producer complete: {job['run_id']}")
        if job.get("source_sha") != SOURCE:
            raise ValueError("Ledger source differs from immutable wave")
        destination = (
            args.base
            / job["run_id"]
            / (
                "report.json"
                if job["task"] in {"conformal-validation", "catalogue-comparison"}
                else "manifest.json"
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(meta, destination)
        job["manifest"] = manifest
        job["checkpoint_sha256"] = manifest.get("checkpoint_sha256")
        jobs.append(job)
    batch(
        [
            (
                j,
                "report.json"
                if j["task"] in {"conformal-validation", "catalogue-comparison"}
                else "manifest.json",
            )
            for j in jobs
        ]
    )
    priority = [
        j
        for j in jobs
        if "purecontrastive_clean_r03_s45" in j["run_id"]
        or "nsf_clean_r03_s45" in j["run_id"]
        or "gaussian_clean_r03_s45" in j["run_id"]
    ]
    for group in [priority, [j for j in jobs if j not in priority]]:
        tasks = []
        for job in group:
            files = inventory(job["task"], job["manifest"])
            tasks += [(job, f) for f in files]
        batch(tasks)
    links = deduplicate_pairs(args.base, receipts)
    (state / "paired-deduplication.json").write_text(json.dumps(links, indent=2) + "\n")
    failures = [r for r in receipts if r["status"] == "transport_unavailable"]
    (state / "state.json").write_text(
        json.dumps(
            dict(
                status="complete_preservation_only"
                if not failures
                else "partial_transport",
                files=len(receipts),
                missing=failures,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
