"""Collect immutable RC/SIMEX outputs, including every contributing checkpoint."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def inventory_files(index):
    """Treat remote JSON as data; reject paths outside the declared replica."""
    records = index["replicas"]
    if index.get("status") != "complete" or len(records) != 17:
        raise ValueError("Expected all17 completed SIMEX replicas")
    result = []
    directories = set()
    for record in records:
        directory = record["directory"]
        if not re.fullmatch(
            r"replica-\d{3}-lambda-(?:0|0\.5|1|1\.5|2)-sim-[0-3]", directory
        ):
            raise ValueError("Unsafe or unexpected replica directory")
        if directory in directories:
            raise ValueError("Duplicate replica directory")
        directories.add(directory)
        if "checkpoint.pth" not in record["files"]:
            raise ValueError("A contributing checkpoint is missing")
        for name, metadata in record["files"].items():
            if Path(name).name != name or name in {".", ".."} or "\\" in name:
                raise ValueError("Unsafe artifact basename")
            if type(metadata["bytes"]) is not int or metadata["bytes"] < 0:
                raise ValueError("Invalid artifact size")
            if not re.fullmatch("[0-9a-f]{64}", metadata["sha256"]):
                raise ValueError("Invalid artifact checksum")
            result.append((directory + "/" + name, metadata))
        result.append(
            (
                directory + "/replica-manifest.json",
                {"sha256": record["manifest_sha256"]},
            )
        )
    return result


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(job, base, relative, expected=None, *, refresh=False):
    path = base / relative
    expected = expected or {}
    receipt = path.with_name(path.name + ".sha256")
    if not expected.get("sha256") and receipt.exists() and not refresh:
        expected = {**expected, "sha256": receipt.read_text().strip()}
    if path.exists() and not refresh and expected.get("sha256"):
        if expected.get("bytes", path.stat().st_size) != path.stat().st_size:
            raise ValueError("Preserved artifact size differs: " + relative)
        if expected["sha256"] != sha(path):
            raise ValueError("Preserved artifact checksum differs: " + relative)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    remote = job["output"] + "/" + relative
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "180",
            "--cert",
            "/Users/fabbros/.ssl/cadcproxy.pem",
            "https://ws-uv.canfar.net/arc/files" + remote.removeprefix("/arc"),
            "--output",
            str(temporary),
        ],
        check=True,
    )
    if expected.get("bytes", temporary.stat().st_size) != temporary.stat().st_size:
        raise ValueError("Downloaded size differs: " + relative)
    actual_sha = sha(temporary)
    if expected.get("sha256", actual_sha) != actual_sha:
        raise ValueError("Downloaded checksum differs: " + relative)
    if relative.endswith(".json"):
        json.loads(temporary.read_text())
    temporary.replace(path)
    staged_receipt = receipt.with_suffix(receipt.suffix + ".download")
    staged_receipt.write_text(actual_sha + "\n")
    staged_receipt.replace(receipt)


def snapshot_files(metrics, budget):
    """Recover actual snapshot names from recorded row counts, including tail batch."""
    names = ["snapshot-0.npz"]
    previous = 0
    for metric in metrics:
        seen = metric["presentations"]
        if not previous < seen <= budget:
            raise ValueError("Non-monotone reconstruction presentations")
        if any(
            previous < target <= seen for target in [3000000, 6000000, 9000000, budget]
        ):
            names.append(f"snapshot-{seen}.npz")
        previous = seen
    if previous != budget:
        raise ValueError("Incomplete metrics for completed run")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    ledger = json.loads(
        (ROOT / "experiments/xp/launch-stellar-improvement-20261015.json").read_text()
    )
    job = next(j for j in ledger["jobs"] if j["run_id"] == args.run_id)
    if not job.get("session_ids") or not args.run_id.endswith("_xp_10m_v2"):
        raise ValueError("Require an accepted XP-correction replacement")
    base = (
        ROOT
        / "experiments/xp/results-stellar-improvement/input-corrections-e72"
        / args.run_id
    )
    base.mkdir(parents=True, exist_ok=True)
    download(job, base, "manifest.json", refresh=True)
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest["source_sha"] != job["source_sha"]:
        raise ValueError("Remote source differs from accepted ledger")
    for name in ["xp-error-source-audit.json", "input-correction.json"]:
        download(job, base, name)
    if manifest["status"] != "complete":
        print(
            json.dumps(
                {
                    "run_id": args.run_id,
                    "status": manifest["status"],
                    "large_artifacts_fetched": False,
                }
            )
        )
        return
    files = [
        (n, None)
        for n in [
            "comparison.json",
            "checkpoint_verification.json",
            "paired-cohort.npz",
            "initial_predictions.npz",
            "predictions.npz",
            "initial_reconstruction.npz",
        ]
    ]
    if manifest.get("replica_count", 1) > 1:
        download(job, base, "replicas.json")
        files += inventory_files(json.loads((base / "replicas.json").read_text()))
    else:
        files += [
            (n, None)
            for n in ["checkpoint.pth", "final_reconstruction.npz", "metrics.jsonl"]
        ]
    for relative, expected in files:
        download(job, base, relative, expected)
    if manifest.get("replica_count", 1) == 1:
        metrics = [
            json.loads(line)
            for line in (base / "metrics.jsonl").read_text().splitlines()
            if line
        ]
        for name in snapshot_files(metrics, manifest["presentations"]):
            download(job, base, name)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "status": "collected_not_yet_independently_verified",
                "files": len(files),
            }
        )
    )


if __name__ == "__main__":
    main()
