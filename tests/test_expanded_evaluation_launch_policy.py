"""Expansion approval permits inference, not another training wave."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expanded_evaluation_launch_policy as policy


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    values = {
        "APPROVAL_CONFIG": {
            "submission_policy": {
                "additional_training_jobs": 0,
                "additional_source_or_inference_jobs": "explicitly_authorized_as_needed_by_user",
            }
        },
        "TRAINING_CONFIG": {
            "checkpoint": "/arc/c544.pth",
            "arms": [{"run_id": "producer"}],
        },
        "SOURCE_CONFIG": {"run_id": policy.SOURCE_RUN},
        "EVALUATION_CONFIG": {
            "arms": [{"run_id": "evaluation", "producer_run_id": "producer"}]
        },
    }
    for key, value in values.items():
        path = tmp_path / (key + ".yaml")
        path.write_text(yaml.safe_dump(value))
        monkeypatch.setattr(policy, key, path)
    return {
        "run_id": "evaluation",
        "task": "expanded-checkpoint-evaluation",
        "depends_on": [policy.SOURCE_RUN, "producer"],
        "resources": {"gpu": 1},
    }


def test_evaluation_depends_on_completed_producer_and_source(pinned):
    env = policy.bind_expanded_evaluation_policy(pinned, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_PRETRAIN_RESUME"] == policy.ROOT + "/producer/checkpoint.pth"
    for change in [
        {"depends_on": ["producer"]},
        {"resources": {"gpu": 0}},
        {"env": {"MSA_EXPERIMENT_PRESENTATIONS": "10000000"}},
    ]:
        with pytest.raises(ValueError):
            policy.bind_expanded_evaluation_policy(dict(pinned, **change), {})


def test_approved_source_preparation_does_not_use_a_gpu(pinned):
    plan = {
        "task": "full-label-evaluation-preparation",
        "run_id": policy.SOURCE_RUN,
        "depends_on": ["five_job_broad_test_preparation_v1"],
        "resources": {"gpu": 0},
    }
    assert (
        policy.bind_expanded_evaluation_policy(plan, {})["MSA_EXPERIMENT_PRESENTATIONS"]
        == "0"
    )
    with pytest.raises(ValueError):
        policy.bind_expanded_evaluation_policy(dict(plan, resources={"gpu": 1}), {})


def test_approval_does_not_authorize_training(pinned):
    data = yaml.safe_load(policy.APPROVAL_CONFIG.read_text())
    data["submission_policy"]["additional_training_jobs"] = 1
    policy.APPROVAL_CONFIG.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="scope"):
        policy.bind_expanded_evaluation_policy(pinned, {})


def test_source_contract_audit_is_distinct_zero_gpu_zero_training(pinned):
    plan = dict(
        run_id="full_label_source_contract_audit_v1",
        task="expanded-source-contract-audit",
        depends_on=["five_job_broad_test_preparation_v1"],
        resources=dict(gpu=0),
    )
    assert (
        policy.bind_expanded_evaluation_policy(plan, {})["MSA_EXPERIMENT_PRESENTATIONS"]
        == "0"
    )
    with pytest.raises(ValueError):
        policy.bind_expanded_evaluation_policy(dict(plan, run_id=policy.SOURCE_RUN), {})
