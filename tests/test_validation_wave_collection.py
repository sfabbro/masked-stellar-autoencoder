"""Collector contracts: missing-only preservation and actual producer inventory."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "collector",
    Path(__file__).resolve().parents[1]
    / "experiments/xp/monitor/collect_validation_wave.py",
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_existing_checkpoint_sha_is_verified_without_network(tmp_path, monkeypatch):
    target = tmp_path / "run" / "checkpoint.pth"
    target.parent.mkdir()
    target.write_bytes(b"immutable bytes")
    job = dict(run_id="run", checkpoint_sha256=collector.digest(target))
    monkeypatch.setattr(
        collector.subprocess,
        "run",
        lambda *a, **k: pytest.fail("network used for preserved file"),
    )
    assert collector.fetch(job, "checkpoint.pth", tmp_path)["status"] == "preserved"
    job["checkpoint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="SHA mismatch"):
        collector.fetch(job, "checkpoint.pth", tmp_path)


def test_inventory_optional_density_and_boundary_counter():
    plain = collector.inventory("multilabel-validation", {"arm": {"kind": "gaussian"}})
    assert "flow-moment-memory-preflight.json" not in plain
    flow = collector.inventory(
        "multilabel-validation",
        {"arm": {"kind": "flow"}, "joint_density_conformal": {"status": "complete"}},
    )
    assert "density-calibration-arrays.npz" in flow
    assert "snapshot-9032768.npz" in flow
    assert "snapshot-9043968.npz" not in flow
    assert collector.inventory("density-rescore", {}) == [
        "immutable-scoring-policy.json",
        "corrected/manifest.json",
        "corrected/predictions-fp64.npz",
    ]


def test_corrupt_npz_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    path.write_bytes(b"not a zip")
    with pytest.raises(collector.zipfile.BadZipFile):
        collector.validate(path)
    path = tmp_path / "bad.json"
    path.write_text("truncated{")
    with pytest.raises(json.JSONDecodeError):
        collector.validate(path)


def test_deduplication_requires_actual_equal_sha(tmp_path):
    receipts = []
    for run, value in [("a", b"one"), ("b", b"one"), ("c", b"different")]:
        path = tmp_path / run / "paired-cohort.npz"
        path.parent.mkdir()
        path.write_bytes(value)
        receipts.append(dict(run_id=run, file=path.name, sha256=collector.digest(path)))
    links = collector.deduplicate_pairs(tmp_path, receipts)
    assert len(links) == 1
    assert (tmp_path / "a/paired-cohort.npz").samefile(tmp_path / "b/paired-cohort.npz")
    assert not (tmp_path / "a/paired-cohort.npz").samefile(
        tmp_path / "c/paired-cohort.npz"
    )
