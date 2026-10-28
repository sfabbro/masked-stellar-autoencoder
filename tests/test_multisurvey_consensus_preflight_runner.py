"""Synthetic admission/receipt checks for the actual-data preflight."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_runner_binds_producer_and_records_failed_attempt(tmp_path):
    path = Path(__file__).parents[1] / "scripts/run_multisurvey_consensus_preflight.py"
    spec = importlib.util.spec_from_file_location("consensus_preflight_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    producer = tmp_path / "producer.json"
    overlay = tmp_path / "overlay.json"
    producer.write_text(
        json.dumps(
            {
                "run_id": "multisurvey_desi_unique_overlay_v1",
                "status": "multisurvey_desi_overlay_complete",
                "source_commit": "cf40e37c87d4e724b84fd1dffcac6d9e1363560d",
                "result": {
                    "overlay_manifest_path": str(overlay),
                    "overlay_manifest_sha256": "a" * 64,
                },
            }
        )
    )
    config = {
        "run_id": "test",
        "overlay_producer_manifest": str(producer),
        "overlay_manifest": str(overlay),
        "overlay_manifest_sha256": "a" * 64,
        "base_path": "base",
        "checkpoint": "ckpt",
        "checkpoint_sha256": "b" * 64,
        "reference_checkpoint": "ref",
        "reference_checkpoint_sha256": "c" * 64,
        "model_config": "config",
        "library_revision": "d" * 40,
        "seed": 1,
    }
    out = tmp_path / "good"
    result = module.run(
        config,
        out,
        source_commit="e" * 40,
        execute=lambda *args: {"updates": 2, "supervised_xp_masking": False},
    )
    assert (
        result["status"] == "multisurvey_consensus_cuda_preflight_pass"
        and result["training_performed"]
    )
    assert not result["model_weights_saved"]
    with pytest.raises(FileExistsError):
        module.run(config, out, source_commit="e" * 40)

    def fail(*args):
        raise ValueError("after updates may have begun")

    with pytest.raises(ValueError, match="updates"):
        module.run(config, tmp_path / "failed", source_commit="e" * 40, execute=fail)
    failed = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert failed["status"] == "failed" and failed["training_performed"] is None
    producer.write_text("{}")
    with pytest.raises(ValueError, match="producer binding"):
        module.run(config, tmp_path / "bad-source", source_commit="e" * 40)


def test_launch_policy_blocks_cpu_and_output_override():
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    try:
        from multisurvey_consensus_preflight_launch_policy import (
            bind_consensus_preflight_policy,
        )

        plan = yaml.safe_load(
            Path(
                "configs/multisurvey-consensus-preflight-launch-20261028.yaml"
            ).read_text()
        )["jobs"][0]
        good = {
            "MSA_EXPERIMENT_OUTPUT": "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/multisurvey_consensus_cuda_preflight_v1"
        }
        assert (
            bind_consensus_preflight_policy(plan, dict(good))[
                "MSA_EXPERIMENT_PRESENTATIONS"
            ]
            == "0"
        )
        with pytest.raises(ValueError, match="resource"):
            bind_consensus_preflight_policy(
                dict(plan, resources={"cores": 4, "ram": 64, "gpu": 0}), dict(good)
            )
        with pytest.raises(ValueError, match="distinct"):
            bind_consensus_preflight_policy(
                plan, {"MSA_EXPERIMENT_OUTPUT": "/arc/other"}
            )
    finally:
        sys.path.pop(0)
