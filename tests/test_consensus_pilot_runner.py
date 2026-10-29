"""Synthetic admission tests for the remote-only consensus pilot runner."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "consensus_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_multisurvey_consensus_pilot.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_runner_requires_completed_bound_overlay_and_preserves_failure(tmp_path):
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/multisurvey-consensus-ft-20261028.yaml"
        ).read_text()
    )
    producer = tmp_path / "producer.json"
    config["overlay_producer_manifest"] = str(producer)
    producer.write_text(
        json.dumps({"run_id": "wrong", "status": "multisurvey_desi_overlay_complete"})
    )
    called = []
    with pytest.raises(ValueError, match="producer binding"):
        runner.run(
            config,
            tmp_path / "fail",
            source_commit="a" * 40,
            execute=lambda *x: called.append(x),
        )
    assert not called
    assert (
        json.loads((tmp_path / "fail" / "manifest.json").read_text())["status"]
        == "failed"
    )
    producer.write_text(
        json.dumps(
            {
                "run_id": "multisurvey_desi_unique_overlay_v1",
                "status": "multisurvey_desi_overlay_complete",
                "source_commit": "cf40e37c87d4e724b84fd1dffcac6d9e1363560d",
                "result": {
                    "overlay_manifest_path": config["overlay_manifest"],
                    "overlay_manifest_sha256": config["overlay_manifest_sha256"],
                },
            }
        )
    )

    def execute(c, overlay, commit, output):
        called.append((overlay, commit))
        return {
            "completed_updates": c["updates"],
            "selected_checkpoint_sha256": "a" * 64,
        }

    result = runner.run(
        config, tmp_path / "ok", source_commit="a" * 40, execute=execute
    )
    assert (
        result["status"] == "multisurvey_consensus_pilot_complete"
        and result["model_weights_saved"] is True
    )
    with pytest.raises(FileExistsError):
        runner.run(config, tmp_path / "ok", source_commit="a" * 40, execute=execute)
