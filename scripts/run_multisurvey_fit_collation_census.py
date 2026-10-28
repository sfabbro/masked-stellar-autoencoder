"""Count source-reference masks after FIT collation; never load a stellar model."""

from __future__ import annotations

import argparse
import hashlib
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
from masked_stellar_autoencoder.training.multisurvey_overlay import (
    sparse_dataset_from_manifest,
)

RUN = "multisurvey_fit_collation_census_v1"
PRODUCER = "multisurvey_desi_unique_overlay_v1"
PRODUCER_COMMIT = "cf40e37c87d4e724b84fd1dffcac6d9e1363560d"


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_arrays(path, arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Saved array archive failed CRC verification")
    with np.load(temporary, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("Saved array archive fields differ")
        for key, expected in arrays.items():
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Saved array field differs: {key}")
    temporary.replace(path)


def _is_sha(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def run(config, output, *, source_commit):
    """Write one immutable audit output after validating its source bindings."""
    required = {
        "run_id",
        "overlay_producer_manifest",
        "overlay_manifest",
        "overlay_manifest_sha256",
        "overlay_sha256",
        "expected_fit_ids_sha256",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required
        or config["run_id"] != RUN
        or not _is_sha(source_commit, 40)
        or any(
            not _is_sha(config[key], 64)
            for key in (
                "overlay_manifest_sha256",
                "overlay_sha256",
                "expected_fit_ids_sha256",
            )
        )
        or any(
            not isinstance(config[key], str) or not config[key]
            for key in ("overlay_producer_manifest", "overlay_manifest")
        )
    ):
        raise ValueError("Unexpected FIT collation census configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN,
        "status": "admitting",
        "source_commit": source_commit,
        "source_only": True,
        "model_loaded": False,
        "training_performed": False,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scope": "FIT source-reference availability, not label-quality or physical-scale validation",
    }
    _write_json(output / "manifest.json", manifest)
    dataset = None
    try:
        producer = json.loads(Path(config["overlay_producer_manifest"]).read_text())
        result = producer.get("result", {})
        if (
            producer.get("run_id") != PRODUCER
            or producer.get("status") != "multisurvey_desi_overlay_complete"
            or producer.get("source_commit") != PRODUCER_COMMIT
            or result.get("overlay_manifest_path") != config["overlay_manifest"]
            or result.get("overlay_manifest_sha256")
            != config["overlay_manifest_sha256"]
            or result.get("overlay_sha256") != config["overlay_sha256"]
        ):
            raise ValueError("Completed overlay producer binding differs")
        manifest["status"] = "auditing_fit"
        _write_json(output / "manifest.json", manifest)
        dataset = sparse_dataset_from_manifest(
            config["overlay_manifest"],
            expected_manifest_sha256=config["overlay_manifest_sha256"],
            role="fit",
        )
        census = census_fit_collation(dataset)
        summary, arrays = census["summary"], census["arrays"]
        if (
            summary["source_binding"]
            != config["overlay_sha256"] + ":" + config["overlay_manifest_sha256"]
            or summary["fit_ids_sha256"] != config["expected_fit_ids_sha256"]
        ):
            raise ValueError("Actual overlay or FIT identity binding differs")
        summary_path = output / "summary.json"
        array_path = output / "star-availability.npz"
        _write_json(summary_path, summary)
        _write_arrays(array_path, arrays)
        dataset._check_source_binding()
        manifest.update(
            status="multisurvey_fit_collation_census_complete",
            result={
                "source_binding": summary["source_binding"],
                "registry_sha256": summary["registry_sha256"],
                "fit_ids_sha256": summary["fit_ids_sha256"],
                "eligible_fit_stars": summary["eligible_fit_stars"],
                "summary_path": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "star_arrays_path": str(array_path),
                "star_arrays_sha256": _sha256(array_path),
                "star_arrays_bytes": array_path.stat().st_size,
                "star_arrays_crc": "pass",
                "star_arrays_roundtrip": "pass",
            },
        )
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        _write_json(output / "manifest.json", manifest)
        raise
    finally:
        if dataset is not None:
            dataset.close()
    _write_json(output / "manifest.json", manifest)
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
        raise RuntimeError("FIT collation census requires CANFAR headless CPU mode")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=source)


if __name__ == "__main__":
    main()
