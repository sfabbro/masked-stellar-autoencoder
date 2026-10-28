"""Source-only census launcher preserves output failures and source bindings."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_runner_binds_completed_sources_and_never_overwrites(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_multisurvey_desi_census.py"
    spec = importlib.util.spec_from_file_location("desi_census_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    side = tmp_path / "sidecar.h5"
    base = tmp_path / "base.h5"
    source_manifest = tmp_path / "side.json"
    base_manifest = tmp_path / "base.json"
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
        "cache_path": str(tmp_path / "cache"),
        "original_path": str(tmp_path / "orig"),
        "expected_cache_rows": 3,
        "chunk_rows": 2,
    }
    monkeypatch.setattr(
        module, "census", lambda *args, **kwargs: {"unique_desi_ids": 3}
    )
    out = tmp_path / "output"
    result = module.run(config, out, source_commit="c" * 40)
    assert result["status"] == "multisurvey_desi_census_complete"
    assert not result["training_performed"] and not result["model_loaded"]
    with pytest.raises(FileExistsError):
        module.run(config, out, source_commit="c" * 40)
    source_manifest.write_text("{}")
    with pytest.raises(ValueError, match="source binding"):
        module.run(config, tmp_path / "failure", source_commit="c" * 40)
    failed = json.loads((tmp_path / "failure/manifest.json").read_text())
    assert failed["status"] == "failed"


def test_launch_policy_rejects_mutated_resources_and_output():
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    try:
        from multisurvey_desi_census_launch_policy import bind_desi_census_policy

        plan = yaml.safe_load(
            Path("configs/multisurvey-desi-census-launch-20261027.yaml").read_text()
        )["jobs"][0]
        good = {
            "MSA_EXPERIMENT_OUTPUT": "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/multisurvey_desi_cache_census_v1"
        }
        env = bind_desi_census_policy(plan, dict(good))
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
        bad = dict(plan, resources={"cores": 8, "ram": 128, "gpu": 1})
        with pytest.raises(ValueError, match="resource"):
            bind_desi_census_policy(bad, dict(good))
        with pytest.raises(ValueError, match="distinct"):
            bind_desi_census_policy(plan, {"MSA_EXPERIMENT_OUTPUT": "/arc/other"})
    finally:
        sys.path.pop(0)
