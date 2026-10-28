"""Bounded CPU-only preparation from existing ARC catalogues; never submits jobs."""

import argparse
import hashlib
import json
import signal
from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.io import fits
from materialize_joint_native_labels import materialize, sha
from verify_joint_native_sidecar import verify

from masked_stellar_autoencoder.training.joint_native_labels import (
    SOURCE_COLUMNS,
    apogee_parameter_metadata,
)


def immutable_json(path, data):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise FileExistsError(f"Existing immutable JSON conflicts: {path}")
        return
    with path.open("x") as f:
        f.write(json.dumps(data, indent=2, allow_nan=False) + "\n")


def preflight(config):
    """Fail before output creation or hashing if existing mounts/schema differ."""
    state = readiness(config)
    missing = [
        item["path"]
        for item in state["existing_mount_files"].values()
        if not item["available"]
    ]
    if missing:
        raise FileNotFoundError(
            "Existing source mounts are missing: " + ", ".join(missing)
        )
    for source, spec in config["sources"].items():
        path = Path(spec["path"])
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"{source} bytes differ from accepted inventory")
        with fits.open(path, memmap=True) as hdul:
            if len(hdul[1].data) != spec["rows"]:
                raise ValueError(f"{source} rows differ from accepted inventory")
            schema = SOURCE_COLUMNS[source]
            required = {
                schema["id"],
                *schema["value"],
                *schema["uncertainty"],
                *schema["flags"],
            }
            missing_columns = required - set(hdul[1].columns.names)
            if missing_columns:
                raise ValueError(
                    f"{source} required native columns missing: {sorted(missing_columns)}"
                )
            if source == "apogee":
                apogee_parameter_metadata(hdul)
    return state


def readiness(config):
    paths = {
        "cohort": config["cohort"]["path"],
        "references": config["references"]["path"],
    }
    paths.update({name: spec["path"] for name, spec in config["sources"].items()})
    return {
        "status": "dry_run_no_materialization_no_submission",
        "cpu_only": True,
        "requested_resources": config["requested_resources"],
        "existing_mount_files": {
            name: {"path": path, "available": Path(path).is_file()}
            for name, path in paths.items()
        },
        "source_hash_pins_missing_from_initial_inventory": [
            name for name, spec in config["sources"].items() if spec["sha256"] is None
        ],
        "source_pin_recording_required": True,
        "native_labels_verified": False,
        "training_launch_ready": False,
        "planned_commands": [
            "pixi run python scripts/prepare_joint_native_labels.py --config configs/joint-native-preparation.yaml --record-source-pins",
            "pixi run python scripts/prepare_joint_native_labels.py --config configs/joint-native-preparation.yaml --execute",
        ],
        "notes": config["notes"],
    }


def extract_reference_ids(spec, output):
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError("Reference H5 differs from the actual audited source")
    with h5py.File(spec["path"]) as f:
        ids = f[spec["dataset"]][:]
    if len(ids) != spec["count"] or len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("Reference union count/uniqueness/positive-ID check failed")
    h = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
    if h != spec["ordered_id_sha256"]:
        raise ValueError("Reference union ordered source-ID hash differs")
    output = Path(output)
    if output.exists():
        if not np.array_equal(np.load(output, allow_pickle=False), ids):
            raise FileExistsError(
                "Existing reference-ID publication conflicts with source"
            )
        return ids
    staged = output.with_suffix(output.suffix + ".building")
    with staged.open("xb") as handle:
        np.save(handle, ids, allow_pickle=False)
    staged.replace(output)
    return ids


def record_source_pins(config, output):
    """Record first GALAH mounted checksum honestly, preserving acquired APOGEE pin."""
    pins = {}
    metadata = {}
    for source, spec in config["sources"].items():
        path = Path(spec["path"])
        if path.stat().st_size != spec["bytes"]:
            raise ValueError(f"{source} bytes differ from accepted inventory")
        with fits.open(path, memmap=True) as hdul:
            if len(hdul[1].data) != spec["rows"]:
                raise ValueError(f"{source} rows differ from accepted inventory")
        actual_sha = sha(path)
        if spec["sha256"] is not None and actual_sha != spec["sha256"]:
            raise ValueError(f"{source} differs from accepted raw checksum")
        pins[source] = actual_sha
        metadata[source] = {
            "path": str(path),
            "bytes": spec["bytes"],
            "rows": spec["rows"],
            "sha256": actual_sha,
            "origin": "accepted raw checksum"
            if spec["sha256"] is not None
            else "first local-on-ARC checksum, not a pre-existing publisher checksum",
        }
    immutable_json(output / "source-pins.json", pins)
    immutable_json(output / "source-pin-provenance.json", metadata)
    return pins


def publish_manifest(output, report):
    """Expose completion to the existing monitor without authorizing training."""
    payload = {
        "status": "complete",
        "mode": "joint-native-preparation",
        "cpu_only": True,
        "native_labels_verified": bool(report["native_labels_verified"]),
        "training_launch_ready": False,
        "distance_quality_zero_point_verified": False,
        "galah_logg_parallax_conditioned": True,
        "result": {
            "native_labels_verified": bool(report["native_labels_verified"]),
            "training_launch_ready": False,
            "source_counts": report["source_counts"],
            "sidecar_sha256": report["sidecar_sha256"],
        },
    }
    path = Path(output) / "manifest.json"
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError("Existing preparation manifest conflicts")
        return
    path.write_text(text)


def execute(config, output):
    """Idempotently verify complete existing artifacts; never overwrite conflicts."""
    pinfile = output / "source-pins.json"
    if not pinfile.exists():
        raise FileNotFoundError("Run --record-source-pins before --execute")
    pins = json.loads(pinfile.read_text())
    if set(pins) != set(config["sources"]):
        raise ValueError("Recorded source pins do not match configured sources")
    for source, spec in config["sources"].items():
        if spec["sha256"] is not None and pins[source] != spec["sha256"]:
            raise ValueError(
                "Recorded source pin conflicts with accepted acquisition hash"
            )
    refs = extract_reference_ids(
        config["references"], output / "reference-holdout-source-ids.npy"
    )
    sources = {name: Path(spec["path"]) for name, spec in config["sources"].items()}
    sidecar = output / config["sidecar_filename"]
    if sidecar.exists():
        with h5py.File(sidecar) as f:
            if not bool(f.attrs.get("complete", False)):
                raise FileExistsError(
                    "Existing sidecar is incomplete; retain it and use a new output"
                )
            provenance = json.loads(f.attrs["provenance_json"])
        if (
            provenance["cohort_sha256"] != config["cohort"]["sha256"]
            or provenance["reference_holdout_ordered_id_sha256"]
            != config["references"]["ordered_id_sha256"]
        ):
            raise FileExistsError("Existing sidecar source/reference pins conflict")
    else:
        materialize(
            config["cohort"]["path"],
            sources,
            pins,
            refs,
            sidecar,
            cohort_sha256=config["cohort"]["sha256"],
            reference_sha256=config["references"]["ordered_id_sha256"],
        )
    report = verify(
        sidecar,
        config["cohort"]["path"],
        sources,
        pins,
        output / "native-label-independent-verification.json",
    )
    immutable_json(output / "preparation-report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=Path("configs/joint-native-preparation.yaml")
    )
    p.add_argument(
        "--output-directory",
        type=Path,
        help="New immutable output for a distinct recovery attempt",
    )
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--record-source-pins", action="store_true")
    modes.add_argument("--execute", action="store_true")
    a = p.parse_args()
    config = yaml.safe_load(a.config.read_text())
    if a.output_directory is not None:
        config["output_directory"] = str(a.output_directory)
    if (
        config["schema"] != "joint-native-preparation-v1"
        or config["cpu_only"] is not True
        or config["requested_resources"]["gpus"] != 0
    ):
        raise ValueError("Expected explicit CPU-only preparation configuration")
    if a.dry_run:
        print(json.dumps(readiness(config), indent=2))
        return

    if not 0 < config["requested_resources"]["maximum_runtime_seconds"] <= 3600:
        raise ValueError("CPU preparation requires a positive at-most-one-hour bound")
    preflight(config)

    def timeout(_signal, _frame):
        raise TimeoutError("Bounded native projection exceeded configured wall time")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(int(config["requested_resources"]["maximum_runtime_seconds"]))
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "preparation.lock"
    with lock.open("x"):
        try:
            report = (
                record_source_pins(config, output)
                if a.record_source_pins
                else execute(config, output)
            )
            print(json.dumps(report, indent=2))
            if a.execute:
                publish_manifest(output, report)
        finally:
            lock.unlink()
            signal.alarm(0)


if __name__ == "__main__":
    main()
