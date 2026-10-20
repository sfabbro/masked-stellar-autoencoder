"""Multilabel production resources must use each arm's real numerical budget."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_stellar_improvement import bind_prediction_policy


def test_multilabel_parent_and_fit_vs_reconstruction_budget(tmp_path, monkeypatch):
    config = tmp_path / "configs"
    config.mkdir()
    (config / "multilabel-screen-20261020.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoint": "/arc/actual/parent86.pth",
                "reconstruction_presentations": 10000000,
                "arms": [
                    {"run_id": "joint", "kind": "gaussian"},
                    {"run_id": "inverse", "kind": "forward"},
                ],
            }
        )
    )
    monkeypatch.chdir(tmp_path)
    for run, expected in [("joint", "10000000"), ("inverse", "0")]:
        env = bind_prediction_policy({"task": "multilabel-screen", "run_id": run}, {})
        assert env == {
            "MSA_PRETRAIN_RESUME": "/arc/actual/parent86.pth",
            "MSA_EXPERIMENT_PRESENTATIONS": expected,
        }
    with pytest.raises(ValueError, match="absent"):
        bind_prediction_policy({"task": "multilabel-screen", "run_id": "unknown"}, {})
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(
            {
                "task": "multilabel-screen",
                "run_id": "joint",
                "env": {"MSA_PRETRAIN_RESUME": "/arc/old57.pth"},
            },
            {},
        )
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(
            {
                "task": "multilabel-screen",
                "run_id": "inverse",
                "env": {"MSA_EXPERIMENT_PRESENTATIONS": "10000000"},
            },
            {},
        )
