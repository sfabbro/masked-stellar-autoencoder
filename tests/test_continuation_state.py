"""Failures must leave the last complete checkpoint / QA publication readable."""

import json

import pytest
import torch

from masked_stellar_autoencoder.training.continuation_state import (
    exclusive_run,
    load_checkpoint,
    publish_qa,
    save_checkpoint,
    write_qa_snapshot,
)


def test_same_step_checkpoints_are_immutable_and_corruption_rejected(tmp_path):
    identity = {"source": "a", "policy": "fixed"}
    a = save_checkpoint(
        tmp_path,
        {"next_row": 0, "x": torch.tensor([1.0])},
        epoch=1,
        step=3,
        identity=identity,
    )
    b = save_checkpoint(
        tmp_path,
        {"next_row": 0, "x": torch.tensor([2.0])},
        epoch=1,
        step=3,
        identity=identity,
    )
    assert a["path"] != b["path"] and (tmp_path / a["path"]).exists()
    assert load_checkpoint(tmp_path, identity)["x"].item() == 2
    with pytest.raises(ValueError, match="changed"):
        load_checkpoint(tmp_path, {"source": "b"})
    (tmp_path / b["path"]).write_bytes(b"partial write")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(tmp_path, identity)


def test_interrupted_pointer_keeps_prior_checkpoint(tmp_path, monkeypatch):
    import masked_stellar_autoencoder.training.continuation_state as state

    identity = {"source": "a"}
    prior = save_checkpoint(
        tmp_path, {"next_row": 0, "x": 1}, epoch=1, step=3, identity=identity
    )

    def fail(*a, **k):
        raise OSError("simulated pointer failure")

    monkeypatch.setattr(state, "atomic_json", fail)
    with pytest.raises(OSError):
        save_checkpoint(
            tmp_path, {"next_row": 0, "x": 2}, epoch=1, step=3, identity=identity
        )
    assert json.loads((tmp_path / "latest.json").read_text()) == prior
    assert load_checkpoint(tmp_path, identity)["x"] == 1


def test_exclusive_output(tmp_path):
    with exclusive_run(tmp_path), pytest.raises(RuntimeError, match="owns"):
        with exclusive_run(tmp_path):
            pass


def test_csv_snapshot_publication_manifest_is_last(tmp_path):
    manifest = dict(
        schema_version=1,
        publication_id="epoch003",
        run_id="run",
        source_sha="a" * 40,
        parent_sha256="b" * 64,
        status="running",
        generated_at_utc="2026-10-17T00:00:00Z",
    )
    tables = {
        "losses.csv": (
            ["epoch", "step", "split", "loss", "value"],
            [dict(epoch=3, step=12, split="fit", loss="nll", value=1.0)],
        ),
        "metrics.csv": (
            ["epoch", "split", "target", "subset", "metric", "value", "unit", "count"],
            [],
        ),
        "predictions.csv": (
            ["epoch", "split", "target", "source_id", "truth", "prediction", "unit"],
            [],
        ),
    }
    result = write_qa_snapshot(tmp_path, manifest, tables)
    copied = []
    publish_qa(
        tmp_path, result, "vos:sfabbro/run/qa", copy=lambda a, b: copied.append((a, b))
    )
    assert copied[-1][1] == "vos:sfabbro/run/qa/manifest.json"
    assert len(copied) == 4
    (tmp_path / "epoch003/losses.csv").write_text("broken")
    with pytest.raises(ValueError, match="changed"):
        publish_qa(tmp_path, result, "vos:sfabbro/run/qa", copy=lambda *a: None)
    with pytest.raises(FileExistsError):
        write_qa_snapshot(tmp_path, manifest, tables)


def test_failed_data_upload_never_publishes_manifest(tmp_path):
    result = write_qa_snapshot(
        tmp_path,
        dict(publication_id="epoch003"),
        {"losses.csv": (["value"], [{"value": 1}])},
    )
    copied = []

    def fail(source, destination):
        copied.append(destination)
        raise OSError("service unavailable")

    with pytest.raises(OSError):
        publish_qa(tmp_path, result, "vos:sfabbro/run/qa", copy=fail)
    assert not any(x.endswith("manifest.json") for x in copied)
