import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_stellar_improvement import bind_prediction_policy


def test_validation_training_budget_and_no_silent_parent_override():
    cfg = yaml.safe_load(
        Path("configs/multilabel-validation-wave-20261022.yaml").read_text()
    )
    for arm in cfg["arms"]:
        env = bind_prediction_policy(
            {"task": "multilabel-validation", "run_id": arm["run_id"]}, {}
        )
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "10000000"
        assert env["MSA_PRETRAIN_RESUME"] == cfg["checkpoint"]
    with pytest.raises(ValueError):
        bind_prediction_policy(
            {"task": "multilabel-validation", "run_id": "missing"}, {}
        )


@pytest.mark.parametrize(
    "task",
    [
        "density-rescore",
        "conformal-validation",
        "target-head-pilot",
        "catalogue-comparison",
    ],
)
def test_evaluation_or_frozen_head_has_zero_reconstruction_replay(task):
    env = bind_prediction_policy({"task": task, "run_id": "bound-by-runner-config"}, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    with pytest.raises(ValueError):
        bind_prediction_policy(
            {"task": task, "env": {"MSA_EXPERIMENT_PRESENTATIONS": "10000000"}}, {}
        )


def test_native_pilot_wave_is_explicit_joint_vs_composite_and_not_giant_science():
    cfg = yaml.safe_load(Path("configs/target-pilot-wave-20261022.yaml").read_text())
    assert len(cfg["arms"]) == 6
    assert {a["objective"] for a in cfg["arms"]} == {"joint", "composite"}
    assert {a["seed"] for a in cfg["arms"]} == {42, 43, 44}
    assert len(cfg["input_sha256"]) == len(cfg["manifest_sha256"]) == 64
