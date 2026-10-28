"""Run a source-only DESI identity/cache census, with no model or training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from masked_stellar_autoencoder.training.multisurvey_census import census


def _write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def run(config, output, *, source_commit):
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("Immutable source commit required")
    required = {
        "run_id",
        "sidecar_manifest",
        "sidecar_path",
        "sidecar_sha",
        "base_manifest",
        "base_path",
        "base_sha",
        "cache_path",
        "original_path",
        "expected_cache_rows",
        "chunk_rows",
    }
    if set(config) != required or not config["run_id"]:
        raise ValueError("Unexpected census configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": config["run_id"],
        "status": "admitting",
        "source_commit": source_commit,
        "source_only": True,
        "training_performed": False,
        "model_loaded": False,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scope": "Exact DESI ID and cache membership only; unknown ancestry or Gaia-XP status remains unknown",
    }
    _write(output / "manifest.json", manifest)
    try:
        source = json.loads(Path(config["sidecar_manifest"]).read_text())
        base = json.loads(Path(config["base_manifest"]).read_text())
        if (
            source.get("run_id") != "multisurvey_desi_dr1_raw_source_v1"
            or source.get("status") != "multisurvey_desi_source_complete"
            or source.get("source_commit") != "06d5438f8a2ac045940c0d5182b14bb2b7627032"
            or source.get("result", {}).get("sha256") != config["sidecar_sha"]
            or source.get("result", {}).get("path") != config["sidecar_path"]
        ):
            raise ValueError("Completed DESI source binding differs")
        if (
            base.get("run_id") != "remediation_source_v3"
            or base.get("status") != "remediation_source_complete"
            or base.get("source_commit") != "d87d7c60d44ae7ea374886959a7bd082ac3e909a"
            or base.get("prepared_sha256") != config["base_sha"]
            or base.get("prepared_path") != config["base_path"]
        ):
            raise ValueError("Completed remediation source binding differs")
        manifest["status"] = "scanning"
        _write(output / "manifest.json", manifest)

        def progress(done, total):
            _write(
                output / "progress.json",
                {"phase": "cache_id_scan", "rows_scanned": done, "rows_total": total},
            )

        result = census(
            config["sidecar_path"],
            config["base_path"],
            config["cache_path"],
            original_path=config["original_path"],
            sidecar_sha=config["sidecar_sha"],
            base_sha=config["base_sha"],
            output=output / "census",
            expected_cache_rows=config["expected_cache_rows"],
            chunk_rows=config["chunk_rows"],
            progress=progress,
        )
        manifest.update(status="multisurvey_desi_census_complete", result=result)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        _write(output / "manifest.json", manifest)
        raise
    _write(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1"
        or not Path("/arc/projects/k-pop").is_dir()
    ):
        raise RuntimeError("Census execution requires CANFAR headless CPU mode")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=sha)


if __name__ == "__main__":
    main()
