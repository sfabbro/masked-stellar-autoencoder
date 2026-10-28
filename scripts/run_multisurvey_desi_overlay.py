"""Materialize a role-preserving DESI selection and observation overlay on CANFAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

from masked_stellar_autoencoder.training.multisurvey_desi_selection import (
    create_overlay,
)


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
        "parent_registry",
    }
    if set(config) != required or not config["run_id"]:
        raise ValueError("Unexpected overlay configuration")
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
        "scope": "Existing-source role-preserving overlay; DESI labels and covariance not physically certified",
    }
    _write(output / "manifest.json", manifest)
    try:
        side = json.loads(Path(config["sidecar_manifest"]).read_text())
        base = json.loads(Path(config["base_manifest"]).read_text())
        if (
            side.get("run_id") != "multisurvey_desi_dr1_raw_source_v1"
            or side.get("status") != "multisurvey_desi_source_complete"
            or side.get("source_commit") != "06d5438f8a2ac045940c0d5182b14bb2b7627032"
            or side.get("result", {}).get("path") != config["sidecar_path"]
            or side.get("result", {}).get("sha256") != config["sidecar_sha"]
        ):
            raise ValueError("Completed DESI source binding differs")
        if (
            base.get("run_id") != "remediation_source_v3"
            or base.get("status") != "remediation_source_complete"
            or base.get("source_commit") != "d87d7c60d44ae7ea374886959a7bd082ac3e909a"
            or base.get("prepared_path") != config["base_path"]
            or base.get("prepared_sha256") != config["base_sha"]
        ):
            raise ValueError("Completed remediation source binding differs")
        registry_path = Path(config["parent_registry"])
        registry_raw = registry_path.read_bytes()
        registry = json.loads(registry_raw)
        manifest.update(
            status="materializing",
            parent_registry_sha256=hashlib.sha256(registry_raw).hexdigest(),
        )
        _write(output / "manifest.json", manifest)
        result = create_overlay(
            config["base_path"],
            config["base_sha"],
            registry,
            config["sidecar_path"],
            config["sidecar_sha"],
            output / "bundle",
        )
        manifest.update(status="multisurvey_desi_overlay_complete", result=result)
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
        raise RuntimeError("Overlay execution requires CANFAR headless CPU mode")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=source)


if __name__ == "__main__":
    main()
