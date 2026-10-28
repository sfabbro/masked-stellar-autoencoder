"""Synthetic source-producer admission and immutable output tests."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_runner_requires_matching_completed_sources(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_multisurvey_desi_overlay.py"
    spec = importlib.util.spec_from_file_location("desi_overlay_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    side = tmp_path / "sidecar.h5"
    base = tmp_path / "base.h5"
    source_manifest = tmp_path / "side.json"
    base_manifest = tmp_path / "base.json"
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    source_manifest.write_text(
        json.dumps(
            {
                "run_id": "multisurvey_desi_dr1_raw_source_v1",
                "status": "multisurvey_desi_source_complete",
                "source_commit": "06d5438f8a2ac045940c0d5182b14bb2b7627032",
                "result": {"path": str(side), "sha256": "a" * 64},
            }
        )
    )
    base_manifest.write_text(
        json.dumps(
            {
                "run_id": "remediation_source_v3",
                "status": "remediation_source_complete",
                "source_commit": "d87d7c60d44ae7ea374886959a7bd082ac3e909a",
                "prepared_path": str(base),
                "prepared_sha256": "b" * 64,
            }
        )
    )
    config = {
        "run_id": "test",
        "sidecar_manifest": str(source_manifest),
        "sidecar_path": str(side),
        "sidecar_sha": "a" * 64,
        "base_manifest": str(base_manifest),
        "base_path": str(base),
        "base_sha": "b" * 64,
        "parent_registry": str(registry),
    }
    monkeypatch.setattr(
        module,
        "create_overlay",
        lambda *args, **kwargs: {"selected": {"selected_rows": 3}},
    )
    out = tmp_path / "output"
    result = module.run(config, out, source_commit="c" * 40)
    assert result["status"] == "multisurvey_desi_overlay_complete"
    assert not result["training_performed"] and not result["model_loaded"]
    with pytest.raises(FileExistsError):
        module.run(config, out, source_commit="c" * 40)
    source_manifest.write_text("{}")
    with pytest.raises(ValueError, match="source binding"):
        module.run(config, tmp_path / "failure", source_commit="c" * 40)
    assert (
        json.loads((tmp_path / "failure/manifest.json").read_text())["status"]
        == "failed"
    )


def test_launch_policy_rejects_gpu_and_output_changes():
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    try:
        from multisurvey_desi_overlay_launch_policy import bind_desi_overlay_policy

        plan = yaml.safe_load(
            Path("configs/multisurvey-desi-overlay-launch-20261027.yaml").read_text()
        )["jobs"][0]
        good = {
            "MSA_EXPERIMENT_OUTPUT": "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/multisurvey_desi_unique_overlay_v1"
        }
        assert (
            bind_desi_overlay_policy(plan, dict(good))["MSA_EXPERIMENT_PRESENTATIONS"]
            == "0"
        )
        with pytest.raises(ValueError, match="resource"):
            bind_desi_overlay_policy(
                dict(plan, resources={"cores": 8, "ram": 128, "gpu": 1}), dict(good)
            )
        with pytest.raises(ValueError, match="distinct"):
            bind_desi_overlay_policy(plan, {"MSA_EXPERIMENT_OUTPUT": "/arc/other"})
    finally:
        sys.path.pop(0)
