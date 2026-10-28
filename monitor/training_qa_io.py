"""Read-only, checksummed CSV snapshots for local training QA.

The publisher writes immutable publication directories and replaces manifest.json
last. A reader never mixes files from two publications or replaces a valid cache
with a partially transferred snapshot. No PyTorch/marimo dependency is required.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
MAX_MANIFEST_BYTES = 128_000
MAX_FILE_BYTES = 100_000_000
MAX_TOTAL_BYTES = 250_000_000
CSV_FIELDS = {
    "losses.csv": {"epoch", "step", "split", "loss", "value"},
    "metrics.csv": {
        "epoch",
        "split",
        "target",
        "subset",
        "metric",
        "value",
        "unit",
        "count",
    },
    "predictions.csv": {
        "epoch",
        "split",
        "target",
        "source_id",
        "truth",
        "prediction",
        "unit",
    },
    "resources.csv": {
        "epoch",
        "step",
        "gpu_utilization_pct",
        "gpu_memory_used_mb",
        "rss_mb",
        "examples_per_second",
    },
}
REQUIRED_FILES = {"losses.csv", "metrics.csv", "predictions.csv"}
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
Copy = Callable[[str, Path, int], None]


@dataclass(frozen=True)
class Snapshot:
    manifest: dict
    directory: Path
    stale: bool
    warning: str | None
    fetched_at_utc: str


def remote_root(value: str) -> str:
    """Only the requested user's VOSpace, without traversal or URI modifiers."""
    if not value.startswith("vos:sfabbro/"):
        raise ValueError("QA remote must start with vos:sfabbro/")
    tail = value.removeprefix("vos:sfabbro/").rstrip("/")
    parts = tail.split("/")
    if not parts or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", p) or p in (".", "..") for p in parts
    ):
        raise ValueError("QA remote contains an unsafe path")
    if tail.startswith("/") or "//" in tail:
        raise ValueError("QA remote contains an unsafe path")
    return "vos:sfabbro/" + tail


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Expected UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Expected ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Timestamp must include UTC timezone")
    return parsed


def validate_manifest(manifest: dict) -> dict:
    """Validate the shared publisher/reader v1 contract; return the same object."""
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("Unsupported training QA schema")
    for key in ("run_id", "publication_id"):
        if not isinstance(manifest.get(key), str) or not TOKEN.fullmatch(manifest[key]):
            raise ValueError(f"Invalid {key}")
    for key, width in (("source_sha", 40), ("parent_sha256", 64)):
        if not isinstance(manifest.get(key), str) or not re.fullmatch(
            rf"[a-f0-9]{{{width}}}", manifest[key]
        ):
            raise ValueError(f"Invalid {key}")
    _utc(manifest.get("generated_at_utc"))
    if manifest.get("status") not in {
        "preparing",
        "running",
        "complete",
        "failed",
        "stopped",
    }:
        raise ValueError("Invalid training status")
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or not files.keys() >= REQUIRED_FILES
        or not files.keys() <= CSV_FIELDS.keys()
    ):
        raise ValueError("QA must contain losses, metrics and predictions CSVs")
    total = 0
    for name, descriptor in files.items():
        if not isinstance(descriptor, dict):
            raise ValueError("Invalid file descriptor")
        if descriptor.get("path") != f"{manifest['publication_id']}/{name}":
            raise ValueError("CSV path must belong to its immutable publication")
        size = descriptor.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 1 <= size <= MAX_FILE_BYTES
        ):
            raise ValueError("CSV size exceeds allowed limits")
        if not isinstance(descriptor.get("sha256"), str) or not re.fullmatch(
            r"[a-f0-9]{64}", descriptor["sha256"]
        ):
            raise ValueError("Invalid CSV checksum")
        total += size
    if total > MAX_TOTAL_BYTES:
        raise ValueError("QA snapshot exceeds total size limit")
    return manifest


def _number(value: str, field: str, *, integer: bool = False) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid numeric CSV field: {field}") from exc
    if not math.isfinite(result) or (
        integer and (result < 0 or not result.is_integer())
    ):
        raise ValueError(f"Invalid numeric CSV field: {field}")
    return result


def validate_csv(name: str, path: Path, run_id: str) -> None:
    """Reject truncation, malformed values and mixed-run snapshots."""
    if name not in CSV_FIELDS:
        raise ValueError("Unknown QA CSV")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = reader.fieldnames
        if (
            fields is None
            or len(fields) != len(set(fields))
            or not CSV_FIELDS[name] <= set(fields)
        ):
            raise ValueError(f"Invalid {name} header")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed {name} row")
            if "run_id" in row and row["run_id"] != run_id:
                raise ValueError("CSV run_id mismatch")
            for key in ("split", "target", "subset", "metric", "loss", "unit"):
                if key in row and not row[key].strip():
                    raise ValueError(f"Empty CSV field: {key}")
            _number(row["epoch"], "epoch", integer=True)
            if "step" in row:
                _number(row["step"], "step", integer=True)
            if "count" in row:
                _number(row["count"], "count", integer=True)
            for key in ("value", "truth", "prediction"):
                if key in row:
                    _number(row[key], key)
            if "source_id" in row and not re.fullmatch(r"[0-9]+", row["source_id"]):
                raise ValueError(
                    "Source IDs must be integer strings, never rounded floats"
                )
            for key in ("lower68", "upper68", "lower95", "upper95"):
                if row.get(key):
                    _number(row[key], key)
            for lower, upper in (("lower68", "upper68"), ("lower95", "upper95")):
                if (
                    row.get(lower)
                    and row.get(upper)
                    and float(row[lower]) > float(row[upper])
                ):
                    raise ValueError("Reversed prediction interval")
            for key in (
                "gpu_utilization_pct",
                "gpu_memory_used_mb",
                "rss_mb",
                "examples_per_second",
            ):
                if row.get(key):
                    value = _number(row[key], key)
                    if value < 0 or (key == "gpu_utilization_pct" and value > 100):
                        raise ValueError("Invalid resource telemetry")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _checked_files(directory: Path, manifest: dict) -> None:
    for name, info in manifest["files"].items():
        path = directory / name
        if path.stat().st_size != info["bytes"] or _sha(path) != info["sha256"]:
            raise ValueError("QA CSV size/checksum mismatch")
        validate_csv(name, path, manifest["run_id"])


def vos_copy(source: str, destination: Path, timeout: int) -> None:
    """Use existing CADC proxy credentials; never expose command output/tokens."""
    args = ["vcp"]
    certificate = os.environ.get("MSA_CADC_CERT")
    if certificate:
        args += ["--certfile", certificate]
    result = subprocess.run(
        args + [source, str(destination)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    if result.returncode or not destination.is_file():
        raise OSError(
            "VOSpace transfer unavailable; check CADC authentication or remote publication"
        )


def _cached(cache: Path, remote: str, *, warning: str | None) -> Snapshot:
    pointer = json.loads((cache / "current.json").read_text())
    if (
        not isinstance(pointer, dict)
        or pointer.get("remote") != remote
        or not isinstance(pointer.get("snapshot"), str)
        or not re.fullmatch(r"[a-f0-9]{64}", pointer["snapshot"])
    ):
        raise ValueError("Invalid cached snapshot pointer")
    directory = cache / pointer["snapshot"]
    if _sha(directory / "manifest.json") != pointer["snapshot"]:
        raise ValueError("Cached manifest checksum mismatch")
    manifest = validate_manifest(json.loads((directory / "manifest.json").read_text()))
    _checked_files(directory, manifest)
    _utc(pointer.get("fetched_at_utc"))
    return Snapshot(manifest, directory, True, warning, pointer["fetched_at_utc"])


def fetch_snapshot(
    remote: str,
    cache_dir: str | Path,
    *,
    offline: bool = False,
    timeout: int = 45,
    copy: Copy | None = None,
) -> Snapshot:
    """Fetch one bounded immutable snapshot, falling back only to verified cache.

    Errors deliberately omit service output and credentials. Stale fallback is
    explicit, never a silent claim that the training session is healthy.
    """
    remote = remote_root(remote)
    if not isinstance(timeout, int) or not 1 <= timeout <= 120:
        raise ValueError("Transfer timeout must be 1–120 seconds")
    namespace = hashlib.sha256(remote.encode()).hexdigest()[:16]
    cache = Path(cache_dir) / namespace
    cache.mkdir(parents=True, exist_ok=True)
    if offline:
        return _cached(
            cache, remote, warning="OFFLINE: showing last verified local snapshot"
        )
    transfer = copy or vos_copy
    try:
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=cache) as temp:
            staging = Path(temp)
            transfer(remote + "/manifest.json", staging / "manifest.json", timeout)
            if (staging / "manifest.json").stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError("Manifest exceeds size limit")
            raw = (staging / "manifest.json").read_bytes()
            manifest = validate_manifest(json.loads(raw))
            snapshot_hash = hashlib.sha256(raw).hexdigest()
            directory = cache / snapshot_hash
            if directory.exists():
                if _sha(directory / "manifest.json") != snapshot_hash:
                    raise ValueError("Cached manifest checksum mismatch")
                _checked_files(directory, manifest)
            else:
                for name, descriptor in manifest["files"].items():
                    transfer(remote + "/" + descriptor["path"], staging / name, timeout)
                _checked_files(staging, manifest)
                # Rename complete directory, then point readers to it atomically.
                try:
                    os.rename(staging, directory)
                except FileExistsError:
                    _checked_files(directory, manifest)
            fetched = datetime.now(UTC).isoformat()
            pointer = {
                "remote": remote,
                "snapshot": snapshot_hash,
                "fetched_at_utc": fetched,
            }
            fd, tmp = tempfile.mkstemp(prefix=".pointer-", dir=cache)
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(pointer, handle)
                os.replace(tmp, cache / "current.json")
            finally:
                Path(tmp).unlink(missing_ok=True)
            return Snapshot(manifest, directory, False, None, fetched)
    except (OSError, ValueError, subprocess.TimeoutExpired, csv.Error):
        try:
            return _cached(
                cache,
                remote,
                warning="STALE: refresh failed; showing last verified local snapshot",
            )
        except (OSError, ValueError, csv.Error) as exc:
            raise OSError(
                "No verified QA snapshot is available; check remote publication and CADC authentication"
            ) from exc


def read_csv(snapshot: Snapshot, name: str) -> list[dict[str, str]]:
    if name not in snapshot.manifest["files"]:
        return []
    with (snapshot.directory / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
