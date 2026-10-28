"""Build a four-survey Gaia GSP-Spec overlay and FIT census on CANFAR CPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import yaml

from masked_stellar_autoencoder.training.multisurvey_collation_census import (
    census_fit_collation,
)
from masked_stellar_autoencoder.training.multisurvey_gaia_gspspec import (
    build_gaia_sidecar,
    summarize_gaia_roles,
)
from masked_stellar_autoencoder.training.multisurvey_overlay import (
    build_overlay,
    file_sha256,
    sparse_dataset_from_manifest,
    validate_overlay,
)

RUN = "multisurvey_gaia_gspspec_overlay_v1"


def _write(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    tmp.replace(path)


def _arrays(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **value)
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:
            raise ValueError("Star availability archive CRC failed")
    with np.load(tmp, allow_pickle=False) as saved:
        if set(saved.files) != set(value):
            raise ValueError("Star availability fields changed")
        for key, expected in value.items():
            if not np.array_equal(saved[key], expected):
                raise ValueError("Star availability values changed")
    tmp.replace(path)


def _sha(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def run(config, output, *, source_commit):
    required = {
        "run_id",
        "gaia_path",
        "gaia_sha256",
        "gaia_rows",
        "base_path",
        "base_sha256",
        "parent_registry",
        "desi_producer_manifest",
        "desi_sidecar",
        "desi_sidecar_sha256",
        "expected_fit_ids_sha256",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required
        or config["run_id"] != RUN
        or not _sha(source_commit, 40)
        or any(
            not _sha(config[k], 64)
            for k in (
                "gaia_sha256",
                "base_sha256",
                "desi_sidecar_sha256",
                "expected_fit_ids_sha256",
            )
        )
        or not isinstance(config["gaia_rows"], int)
        or config["gaia_rows"] < 1
    ):
        raise ValueError("Unexpected Gaia overlay configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN,
        "status": "admitting",
        "source_commit": source_commit,
        "source_only": True,
        "training_performed": False,
        "model_loaded": False,
        "scope": "Gaia RVS source materialization and existing-base FIT census only; not a trained model, full feature union or physical-scale certificate",
    }
    _write(output / "manifest.json", manifest)
    dataset = None
    try:
        producer = json.loads(Path(config["desi_producer_manifest"]).read_text())
        selected = producer.get("result", {}).get("selected", {})
        if (
            producer.get("run_id") != "multisurvey_desi_unique_overlay_v1"
            or producer.get("status") != "multisurvey_desi_overlay_complete"
            or producer.get("source_commit")
            != "cf40e37c87d4e724b84fd1dffcac6d9e1363560d"
            or selected.get("path") != config["desi_sidecar"]
            or selected.get("sha256") != config["desi_sidecar_sha256"]
        ):
            raise ValueError("Completed DESI producer binding differs")
        manifest["status"] = "building_gaia_sidecar"
        _write(output / "manifest.json", manifest)
        bundle = output / "bundle"
        bundle.mkdir()
        gaia = build_gaia_sidecar(
            config["gaia_path"],
            config["gaia_sha256"],
            bundle / "gaia-gspspec.h5",
            expected_rows=config["gaia_rows"],
        )
        registry = json.loads(Path(config["parent_registry"]).read_text())
        overlay = build_overlay(
            bundle / "overlay.h5",
            parent_path=config["base_path"],
            parent_sha256=config["base_sha256"],
            parent_registry=registry,
            sidecars=[
                {
                    "path": config["desi_sidecar"],
                    "sha256": config["desi_sidecar_sha256"],
                },
                {"path": gaia["path"], "sha256": gaia["sha256"]},
            ],
            manifest_path=bundle / "overlay-manifest.json",
        )
        overlay_sha = file_sha256(overlay)
        proof = validate_overlay(overlay, expected_manifest_sha256=overlay_sha)
        manifest["status"] = "auditing_fit"
        _write(output / "manifest.json", manifest)
        dataset = sparse_dataset_from_manifest(
            overlay, expected_manifest_sha256=overlay_sha, role="fit"
        )
        census = census_fit_collation(dataset)
        if census["summary"]["fit_ids_sha256"] != config["expected_fit_ids_sha256"]:
            raise ValueError("Gaia overlay altered the existing FIT identities")
        roles = summarize_gaia_roles(config["base_path"], gaia["path"])
        summary = {
            "scope": manifest["scope"],
            "gaia_source": gaia["counts"],
            "gaia_by_existing_role": roles,
            "fit_union": census["summary"],
            "gaia_quality_policy": "Provisional per-target source flag tier and broad sanity limits; source-specific M/H and alpha retained, direct Fe/H missing until Fe I/M is joined.",
        }
        _write(output / "summary.json", summary)
        _arrays(output / "fit-star-availability.npz", census["arrays"])
        dataset._check_source_binding()
        manifest.update(
            status="multisurvey_gaia_gspspec_overlay_complete",
            result={
                "gaia_source_sha256": config["gaia_sha256"],
                "gaia_sidecar_path": gaia["path"],
                "gaia_sidecar_sha256": gaia["sha256"],
                "overlay_manifest_path": str(overlay.resolve()),
                "overlay_manifest_sha256": overlay_sha,
                "overlay_sha256": proof["overlay"]["sha256"],
                "fit_ids_sha256": census["summary"]["fit_ids_sha256"],
                "summary_path": str((output / "summary.json").resolve()),
                "summary_sha256": file_sha256(output / "summary.json"),
                "fit_arrays_path": str(
                    (output / "fit-star-availability.npz").resolve()
                ),
                "fit_arrays_sha256": file_sha256(output / "fit-star-availability.npz"),
                "fit_arrays_bytes": (output / "fit-star-availability.npz")
                .stat()
                .st_size,
                "fit_arrays_crc": "pass",
                "fit_arrays_roundtrip": "pass",
            },
        )
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        _write(output / "manifest.json", manifest)
        raise
    finally:
        if dataset is not None:
            dataset.close()
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
        raise RuntimeError("Gaia overlay requires CANFAR headless CPU mode")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=source)


if __name__ == "__main__":
    main()
