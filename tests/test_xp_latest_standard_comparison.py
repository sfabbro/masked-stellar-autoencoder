"""Later standard checkpoints must not be misclassified as matched-budget treatments."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "latest_standard_comparison",
    Path(__file__).parents[1] / "experiments/xp/monitor/compare_latest_standard.py",
)
latest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latest)


def latest_manifest():
    settings = {
        "run_id": "probe_latest_standard57",
        "checkpoint": "/arc/checkpoint_57.pth",
        "checkpoint_epoch": 57,
        "checkpoint_sha256": "a" * 64,
        "mode": "probes",
    }
    manifest = {
        "status": "complete",
        "checkpoint_sha256": settings["checkpoint_sha256"],
        "parent_checkpoint_sha256": None,
        "benchmark_source_id_sha256": latest.paired.BENCHMARK,
        "scaler_sha256": latest.paired.SCALER,
        "arguments": {
            "checkpoint": settings["checkpoint"],
            "checkpoint_sha256": settings["checkpoint_sha256"],
            "seed": 42,
            "mode": "probes",
        },
    }
    return manifest, settings


def test_later_standard_checkpoint_requires_no_experiment_parent():
    manifest, settings = latest_manifest()
    latest.verify_latest_manifest(manifest, settings)
    changed = copy.deepcopy(manifest)
    changed["parent_checkpoint_sha256"] = latest.paired.PARENT
    with pytest.raises(ValueError, match="no experiment-only parent"):
        latest.verify_latest_manifest(changed, settings)


def test_latest_standard_hash_is_pinned_independently_of_earlier_experiment_parent():
    manifest, settings = latest_manifest()
    manifest["checkpoint_sha256"] = latest.paired.PARENT
    with pytest.raises(ValueError, match="Latest standard pinned checksum"):
        latest.verify_latest_manifest(manifest, settings)


def test_latest_config_rejects_multiple_possible_probe_checkpoints(tmp_path):
    config = tmp_path / "latest.yaml"
    config.write_text("analyses:\n- mode: probes\n- mode: probes\n")
    with pytest.raises(ValueError, match="exactly one"):
        latest.latest_settings(config)


def test_latest_epoch_is_declared_in_spec_and_checked_against_checkpoint_path(tmp_path):
    config = tmp_path / "latest.yaml"
    config.write_text(
        "checkpoint_epoch: 57\nanalyses:\n- mode: probes\n  run_id: probe_latest_standard57\n  checkpoint: /arc/checkpoint_57.pth\n  checkpoint_sha256: "
        + "a" * 64
        + "\n"
    )
    assert latest.latest_settings(config)["checkpoint_epoch"] == 57
    config.write_text(
        config.read_text().replace("checkpoint_57.pth", "checkpoint_55.pth")
    )
    with pytest.raises(ValueError, match="differs from checkpoint filename"):
        latest.latest_settings(config)


def test_reconstruction_uses_selected_epoch_and_spec_hash(tmp_path):
    from types import SimpleNamespace

    reconstruction_spec = importlib.util.spec_from_file_location(
        "latest_reconstruction_comparison",
        Path(__file__).parents[1]
        / "experiments/xp/monitor/compare_latest_reconstruction.py",
    )
    reconstruction = importlib.util.module_from_spec(reconstruction_spec)
    reconstruction_spec.loader.exec_module(reconstruction)
    config = tmp_path / "latest57.yaml"
    expected = "b" * 64
    config.write_text(
        "checkpoint_epoch: 57\nanalyses:\n- mode: probes\n  run_id: probe_latest_standard57\n  checkpoint: /arc/checkpoint_57.pth\n  checkpoint_sha256: "
        + expected
        + "\n"
    )
    args = SimpleNamespace(epoch=57, spec=config, root=None, run_id=None)
    root, run_id, settings = reconstruction.resolve_latest(args)
    assert str(root) == "experiments/xp/results-latest-standard-e57"
    assert run_id == "evaluate_latest_standard57"
    assert settings["checkpoint_sha256"] == expected
    args.epoch = 55
    with pytest.raises(ValueError, match="epoch differs"):
        reconstruction.resolve_latest(args)
