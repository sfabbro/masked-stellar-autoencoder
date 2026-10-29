"""Run a source-only Gaia RVS/cache identity census on CANFAR CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

import yaml

from masked_stellar_autoencoder.training.multisurvey_gaia_cache_census import (
    gaia_cache_census,
)

RUN = "multisurvey_gaia_gspspec_cache_census_v1"
BASE_COMMIT = "d87d7c60d44ae7ea374886959a7bd082ac3e909a"


def _write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value, length):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def run(config, output, *, source_commit, execute=gaia_cache_census):
    required = {
        "run_id",
        "source_path",
        "source_sha",
        "expected_source_rows",
        "base_manifest",
        "base_path",
        "base_sha",
        "cache_path",
        "original_path",
        "expected_cache_rows",
        "chunk_rows",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required
        or config.get("run_id") != RUN
        or not _digest(source_commit, 40)
        or not _digest(config.get("source_sha"), 64)
        or not _digest(config.get("base_sha"), 64)
        or any(
            isinstance(config.get(key), bool)
            or not isinstance(config.get(key), int)
            or config[key] < 1
            for key in ("expected_source_rows", "expected_cache_rows", "chunk_rows")
        )
        or any(
            not isinstance(config.get(key), str) or not config[key]
            for key in (
                "source_path",
                "base_manifest",
                "base_path",
                "cache_path",
                "original_path",
            )
        )
    ):
        raise ValueError("Unexpected Gaia cache census configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": RUN,
        "status": "admitting",
        "source_commit": source_commit,
        "source_only": True,
        "training_performed": False,
        "model_loaded": False,
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "scope": (
            "Exact Gaia IDs and existing cache membership only; outside-base "
            "ancestry and absent-cache XP status remain unknown."
        ),
    }
    _write(output / "manifest.json", manifest)
    try:
        base = json.loads(Path(config["base_manifest"]).read_text())
        if (
            base.get("run_id") != "remediation_source_v3"
            or base.get("status") != "remediation_source_complete"
            or base.get("source_commit") != BASE_COMMIT
            or base.get("prepared_sha256") != config["base_sha"]
            or base.get("prepared_path") != config["base_path"]
        ):
            raise ValueError("Completed remediation source binding differs")
        manifest["status"] = "scanning"
        _write(output / "manifest.json", manifest)
        last = 0

        def progress(done, total):
            nonlocal last
            if (
                isinstance(done, bool)
                or not isinstance(done, int)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total != config["expected_cache_rows"]
                or not last <= done <= total
            ):
                raise ValueError("Cache scan progress differs")
            last = done
            _write(
                output / "progress.json",
                {
                    "run_id": RUN,
                    "source_commit": source_commit,
                    "phase": "cache_id_scan",
                    "rows_scanned": done,
                    "rows_total": total,
                },
            )

        result = execute(
            config["source_path"],
            config["base_path"],
            config["cache_path"],
            original_path=config["original_path"],
            source_sha=config["source_sha"],
            base_sha=config["base_sha"],
            output=output / "census",
            expected_source_rows=config["expected_source_rows"],
            expected_cache_rows=config["expected_cache_rows"],
            chunk_rows=config["chunk_rows"],
            progress=progress,
        )
        arrays = output / "census" / "identities.npz"
        summary = output / "census" / "summary.json"
        if (
            result.get("schema") != "multisurvey-gaia-cache-census-v1"
            or result.get("source_only") is not True
            or result.get("model_loaded") is not False
            or result.get("training_performed") is not False
            or result.get("source_sha256") != config["source_sha"]
            or result.get("base_sha256") != config["base_sha"]
            or result.get("source_rows") != config["expected_source_rows"]
            or result.get("identities_path") != str(arrays.resolve())
            or result.get("identities_crc") != "pass"
            or result.get("identities_roundtrip") != "pass"
            or json.loads(summary.read_text()) != result
            or arrays.stat().st_size != result.get("identities_bytes")
            or _sha(arrays) != result.get("identities_sha256")
            or last != config["expected_cache_rows"]
        ):
            raise ValueError("Gaia cache census result binding differs")
        with zipfile.ZipFile(arrays) as archive:
            if archive.testzip() is not None:
                raise ValueError("Gaia cache census NPZ CRC differs")
        manifest.update(
            status="multisurvey_gaia_cache_census_complete",
            result=result,
            summary_path=str(summary.resolve()),
            summary_sha256=_sha(summary),
        )
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
        raise RuntimeError("Gaia cache census requires CANFAR headless CPU mode")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=commit)


if __name__ == "__main__":
    main()
