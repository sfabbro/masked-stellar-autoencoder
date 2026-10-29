"""Run one versioned Gaia Fe I/M exact-ID source join on CANFAR CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from gaia_fem_source_join import build

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_gaia_gspspec_fem_join_v1"
SIDECAR = ROOT + "multisurvey_gaia_gspspec_overlay_v1/bundle/gaia-gspspec.h5"
ARCHIVE = ROOT + "gaia_gspspec_fem_archive_20261028/gaia-fem-sorted.npz"
SIDECAR_SHA = "3427de9f51b116bcc59817b015b3dc2965ae8de8cf86baeb696b2a127937ac49"
ARCHIVE_SHA = "e15ff9d995bdf53989102200cf7743f9aae10c33603d6fe85b816c445b7af030"


def write_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def run(output, source_commit, execute=build):
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise ValueError("explicit immutable source commit required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN,
        "status": "admitting",
        "source_commit": source_commit,
        "source_only": True,
        "training_performed": False,
        "model_loaded": False,
        "sidecar_path": SIDECAR,
        "archive_path": ARCHIVE,
        "sidecar_sha256": SIDECAR_SHA,
        "archive_sha256": ARCHIVE_SHA,
    }
    write_manifest(out / "manifest.json", manifest)
    try:
        manifest["status"] = "joining_sources"
        write_manifest(out / "manifest.json", manifest)
        result = execute(
            SIDECAR,
            ARCHIVE,
            out / "gaia-fem-joined.h5",
            expected_sidecar_sha=SIDECAR_SHA,
            expected_archive_sha=ARCHIVE_SHA,
            expected_sidecar_rows=5591594,
            expected_archive_rows=1774698,
        )
        target = out / "gaia-fem-joined.h5"
        if (
            result["output"] != str(target)
            or result["output_bytes"] != target.stat().st_size
        ):
            raise ValueError("output path or size differs")
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != result["output_sha256"]:
            raise ValueError("output hash differs")
        manifest.update(status="gaia_fem_source_join_complete", result=result)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_manifest(out / "manifest.json", manifest)
        raise
    write_manifest(out / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1"
        or not Path("/arc/projects/k-pop").is_dir()
    ):
        raise RuntimeError("requires CANFAR CPU headless mode")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(args.output, commit)


if __name__ == "__main__":
    main()
