"""Hermetic transfer failures cannot publish a mixed or corrupt QA snapshot."""

import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from monitor.training_qa_io import (
    fetch_snapshot,
    read_csv,
    remote_root,
    validate_manifest,
)

REMOTE = "vos:sfabbro/msa/test-run/qa"


def publication(identifier="epoch_003", prediction="2.1"):
    rows = {
        "losses.csv": (
            ["epoch", "step", "split", "loss", "value"],
            ["3", "6", "train", "masked_mae", "0.5"],
        ),
        "metrics.csv": (
            ["epoch", "split", "target", "subset", "metric", "value", "unit", "count"],
            ["3", "tune", "logg", "all", "mae", "0.1", "dex", "1"],
        ),
        "predictions.csv": (
            ["epoch", "split", "target", "source_id", "truth", "prediction", "unit"],
            ["3", "tune", "logg", "1234567890123456789", "2", prediction, "dex"],
        ),
    }
    content = {}
    files = {}
    for name, data in rows.items():
        output = io.StringIO(newline="")
        csv.writer(output).writerows(data)
        raw = output.getvalue().encode()
        path = identifier + "/" + name
        content[REMOTE + "/" + path] = raw
        files[name] = {
            "path": path,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "run_id": "test-run",
        "publication_id": identifier,
        "generated_at_utc": "2026-10-17T02:00:00Z",
        "status": "running",
        "source_sha": "a" * 40,
        "parent_sha256": "b" * 64,
        "files": files,
    }
    content[REMOTE + "/manifest.json"] = json.dumps(manifest).encode()
    return content, manifest


def copier(content):
    def copy(source, destination, timeout):
        assert timeout <= 120
        Path(destination).write_bytes(content[source])

    return copy


def test_fetch_is_atomic_and_ids_remain_exact(tmp_path):
    content, manifest = publication()
    snapshot = fetch_snapshot(REMOTE, tmp_path, copy=copier(content))
    assert snapshot.manifest == manifest and not snapshot.stale
    assert (
        read_csv(snapshot, "predictions.csv")[0]["source_id"] == "1234567890123456789"
    )
    assert read_csv(snapshot, "predictions.csv")[0]["prediction"] == "2.1"
    assert not list(tmp_path.rglob(".fetch-*"))


def test_failed_new_snapshot_preserves_previous_complete_one(tmp_path):
    content, _ = publication()
    first = fetch_snapshot(REMOTE, tmp_path, copy=copier(content))
    changed, _ = publication("epoch_006", "2.5")
    changed[REMOTE + "/epoch_006/predictions.csv"] = b"broken"
    snapshot = fetch_snapshot(REMOTE, tmp_path, copy=copier(changed))
    assert snapshot.stale and snapshot.warning.startswith("STALE")
    assert snapshot.directory == first.directory
    assert read_csv(snapshot, "predictions.csv")[0]["prediction"] == "2.1"


def test_bad_numeric_value_with_valid_checksum_is_rejected(tmp_path):
    content, _ = publication(prediction="nan")
    with pytest.raises(OSError, match="No verified QA"):
        fetch_snapshot(REMOTE, tmp_path, copy=copier(content))
    assert not list(tmp_path.rglob("current.json"))


def test_offline_never_contacts_remote_and_verifies_cache(tmp_path):
    content, _ = publication()
    first = fetch_snapshot(REMOTE, tmp_path, copy=copier(content))

    def forbidden(*args):
        pytest.fail("Offline refresh must not contact VOSpace")

    result = fetch_snapshot(REMOTE, tmp_path, offline=True, copy=forbidden)
    assert result.stale and result.warning.startswith("OFFLINE")
    (first.directory / "predictions.csv").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        fetch_snapshot(REMOTE, tmp_path, offline=True, copy=forbidden)


@pytest.mark.parametrize(
    "bad",
    [
        "vos:someoneelse/qa",
        "vos:sfabbro/../qa",
        "vos:sfabbro//qa",
        "vos:sfabbro/qa?token=secret",
        "https://example.com/qa",
        "vos:sfabbro/qa#fragment",
    ],
)
def test_rejects_unsafe_remote(bad):
    with pytest.raises(ValueError):
        remote_root(bad)


def test_manifest_limits_and_traversal():
    _, manifest = publication()
    manifest["files"]["losses.csv"]["path"] = "../losses.csv"
    with pytest.raises(ValueError, match="immutable"):
        validate_manifest(manifest)
    _, manifest = publication()
    manifest["files"]["losses.csv"]["bytes"] = 100_000_001
    with pytest.raises(ValueError, match="limits"):
        validate_manifest(manifest)


def test_transfer_failure_does_not_echo_secret_output(tmp_path):
    def fail(*args):
        raise OSError("secret remote credential text")

    with pytest.raises(OSError) as error:
        fetch_snapshot(REMOTE, tmp_path, copy=fail)
    assert "secret" not in str(error.value)


def test_cached_manifest_tampering_rejected(tmp_path):
    content, _ = publication()
    snapshot = fetch_snapshot(REMOTE, tmp_path, copy=copier(content))
    manifest_path = snapshot.directory / "manifest.json"
    manifest_path.write_text(manifest_path.read_text() + " ")
    with pytest.raises(ValueError, match="manifest checksum"):
        fetch_snapshot(REMOTE, tmp_path, offline=True)


def test_unchanged_publication_only_downloads_manifest(tmp_path):
    content, _ = publication()
    fetch_snapshot(REMOTE, tmp_path, copy=copier(content))
    calls = []

    def manifest_only(source, destination, timeout):
        calls.append(source)
        assert source.endswith("/manifest.json")
        destination.write_bytes(content[source])

    result = fetch_snapshot(REMOTE, tmp_path, copy=manifest_only)
    assert not result.stale
    assert calls == [REMOTE + "/manifest.json"]


def test_schema_version_boolean_is_not_v1():
    _, manifest = publication()
    manifest["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        validate_manifest(manifest)
