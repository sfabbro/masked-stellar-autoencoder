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


def test_long_multilabel_binds_one_noise_density_and_fixed_budget():
    suite = yaml.safe_load(Path("configs/multilabel-long-20261020.yaml").read_text())
    assert len(suite["arms"]) == 1
    arm = suite["arms"][0]
    assert arm["kind"] == "gaussian" and arm["noise_aware"] is True
    assert arm["reconstruction_weight"] == 0.03
    assert suite["reconstruction_presentations"] == 30 * 9_000_000
    assert suite["checkpoint_interval_presentations"] == 9_000_000
    assert suite["selection_policy"] == "reserved_tune_mean_observed_nll"
    env = bind_prediction_policy(dict(task="multilabel-long", run_id=arm["run_id"]), {})
    assert env["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "270000000"
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(
            dict(
                task="multilabel-long",
                run_id=arm["run_id"],
                env={"MSA_PRETRAIN_RESUME": "/wrong"},
            ),
            {},
        )
