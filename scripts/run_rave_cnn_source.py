"""CANFAR source-only RAVE CNN and EDR3 crossmatch producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from rave_cnn_source import build_rave_cnn_source

RUN = "multisurvey_rave_cnn_edr3_source_v1"
CNN = "/arc/projects/k-pop/catalogues/rave_dr6_cnn.fits"
CROSS = "/arc/projects/k-pop/catalogues/rave_dr6_x_gaiaedr3.fits"
CNN_SHA = "8a0232e77e4fa06db6b039509227a95e11b2aeb5014f5653e65dbdbda832d495"
CROSS_SHA = "05b81d2a1f56ee3d4be3da21c70481b983fd779843fbd8973671d9ce16d912b8"


def write_manifest(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def run(output, source_commit, execute=build_rave_cnn_source):
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
        "cnn_path": CNN,
        "crossmatch_path": CROSS,
        "cnn_sha256": CNN_SHA,
        "crossmatch_sha256": CROSS_SHA,
    }
    write_manifest(out / "manifest.json", manifest)
    try:
        manifest["status"] = "building_source"
        write_manifest(out / "manifest.json", manifest)
        result = execute(
            CNN,
            CROSS,
            out / "rave-cnn-edr3.h5",
            expected_cnn_sha256=CNN_SHA,
            expected_xmatch_sha256=CROSS_SHA,
            expected_cnn_rows=426574,
            expected_xmatch_rows=517357,
        )
        target = out / "rave-cnn-edr3.h5"
        if (
            result["output"] != str(target)
            or result["output_bytes"] != target.stat().st_size
        ):
            raise ValueError("source output path or size mismatch")
        with target.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != result["output_sha256"]:
            raise ValueError("source output SHA mismatch")
        manifest.update(status="rave_source_complete", result=result)
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
