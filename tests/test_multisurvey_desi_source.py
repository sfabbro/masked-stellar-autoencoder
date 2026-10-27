"""Synthetic-only source runner and launch policy tests."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import multisurvey_desi_launch_policy as policy
import run_multisurvey_desi_source as source
from test_multisurvey_desi_sidecar import fixture


def config_for_fixture(tmp_path):
    params = fixture(tmp_path)
    return {
        "run_id": "synthetic-source",
        "raw_path": str(params["raw_path"]),
        "publisher_path": str(params["publisher_path"]),
        "expected_sha": params["expected_sha"],
        "chunk_rows": 2,
        "snr_min": 10.0,
        "scratch_dir": str(tmp_path),
    }


def test_source_runner_is_source_only_and_never_overwrites(tmp_path):
    config = config_for_fixture(tmp_path)
    out = tmp_path / "source-output"
    result = source.run(config, out, source_commit="a" * 40)
    assert result["status"] == "multisurvey_desi_source_complete"
    assert result["source_only"] and result["training_performed"] is False
    assert result["result"]["quality_rows"] == 5
    assert json.loads((out / "manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        source.run(config, out, source_commit="a" * 40)


def test_source_runner_preserves_failure_without_replacement(tmp_path):
    config = config_for_fixture(tmp_path)
    Path(config["publisher_path"]).write_text("0" * 64)
    out = tmp_path / "failed-output"
    with pytest.raises(ValueError, match="Publisher"):
        source.run(config, out, source_commit="a" * 40)
    result = json.loads((out / "manifest.json").read_text())
    assert result["status"] == "failed" and result["training_performed"] is False
    with pytest.raises(FileExistsError):
        source.run(config, out, source_commit="a" * 40)


def test_launch_policy_rejects_unsafe_overrides():
    cfg = yaml.safe_load(
        Path("configs/multisurvey-desi-launch-20261027.yaml").read_text()
    )
    plan = cfg["jobs"][0]
    env = {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + policy.RUN}
    assert policy.bind_multisurvey_desi_policy(plan, env)[
        "MSA_MULTISURVEY_POLICY"
    ] == str(policy.CONFIG)
    plan["resources"]["gpu"] = 1
    with pytest.raises(ValueError, match="resource"):
        policy.bind_multisurvey_desi_policy(plan, env)
