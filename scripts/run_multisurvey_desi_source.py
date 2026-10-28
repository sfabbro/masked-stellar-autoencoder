"""One immutable, source-only DESI sidecar preparation on CANFAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from build_multisurvey_desi_sidecar import build_desi_sidecar


def _atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(config: dict, output: Path, *, source_commit: str) -> dict:
    """Prepare a new sidecar once; never replace an existing output directory."""
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("An immutable source commit is required")
    expected = {
        "run_id",
        "raw_path",
        "publisher_path",
        "expected_sha",
        "chunk_rows",
        "snr_min",
        "scratch_dir",
    }
    if set(config) != expected or not config["run_id"]:
        raise ValueError("Unexpected DESI source configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": config["run_id"],
        "status": "preparing",
        "source_commit": source_commit,
        "source_only": True,
        "training_performed": False,
        "scope": "Raw quality-selected DESI rows; no feature join, split assignment, or physical-truth certification",
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    _atomic(output / "manifest.json", manifest)
    try:
        result = build_desi_sidecar(
            output / "desi-raw-observations.h5",
            raw_path=config["raw_path"],
            publisher_path=config["publisher_path"],
            expected_sha=config["expected_sha"],
            chunk_rows=config["chunk_rows"],
            snr_min=config["snr_min"],
            scratch_dir=config["scratch_dir"],
        )
        manifest.update(status="multisurvey_desi_source_complete", result=result)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        _atomic(output / "manifest.json", manifest)
        raise
    _atomic(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=commit)


if __name__ == "__main__":
    main()
