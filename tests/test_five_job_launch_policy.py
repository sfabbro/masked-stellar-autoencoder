"""The short large-sample experiment wave never expands beyond five attempts."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import five_job_launch_policy as policy


@pytest.fixture
def plans(tmp_path, monkeypatch):
    jobs = [
        {
            "run_id": policy.SOURCE_RUN,
            "task": "five-job-source-preparation",
            "resources": {"gpu": 0},
        }
    ] + [
        {
            "run_id": f"head{i}",
            "task": "five-wave-training",
            "depends_on": [policy.SOURCE_RUN],
            "resources": {"gpu": 1},
        }
        for i in range(4)
    ]
    plan, suite = tmp_path / "plan.yaml", tmp_path / "suite.yaml"
    plan.write_text(yaml.safe_dump({"jobs": jobs}))
    suite.write_text(
        yaml.safe_dump(
            {
                "checkpoint": "/arc/immutable-c544.pth",
                "reconstruction_training_rows": 150_000_000,
                "reconstruction_presentations": 450_000_000,
                "arms": [{"run_id": f"head{i}"} for i in range(4)],
            }
        )
    )
    monkeypatch.setattr(policy, "PLAN", plan)
    monkeypatch.setattr(policy, "TRAINING", suite)
    return jobs


def test_parent_budget_and_source_gate(plans):
    for job in plans:
        env = policy.bind_five_job_policy(job, {})
        assert env["MSA_PRETRAIN_RESUME"] == "/arc/immutable-c544.pth"
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == (
            "0" if job is plans[0] else "450000000"
        )
    with pytest.raises(ValueError, match="immutable"):
        policy.bind_five_job_policy(dict(plans[1], depends_on=[]), {})
    with pytest.raises(ValueError, match="immutable"):
        policy.bind_five_job_policy(
            dict(plans[1], env={"MSA_PRETRAIN_RESUME": "other"}), {}
        )


def test_cap_includes_failed_and_ambiguous_attempts(plans):
    record = {
        "jobs": [dict(j, attempts=1, status="submission_ambiguous") for j in plans]
    }
    policy.enforce_submission_budget(plans, record)
    with pytest.raises(ValueError, match="extra"):
        policy.enforce_submission_budget([dict(plans[1], run_id="replacement")], record)
    record["jobs"][0]["attempts"] = 2
    with pytest.raises(ValueError, match="exhausted"):
        policy.enforce_submission_budget(plans, record)


def test_sixth_configured_job_rejected(plans):
    config = yaml.safe_load(policy.PLAN.read_text())
    config["jobs"].append(dict(plans[1], run_id="extra"))
    policy.PLAN.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="exactly"):
        policy.validated_plans()


def test_actual_five_job_plan_parent_budget_and_dependency():
    jobs = policy.validated_plans()
    assert len(jobs) == 5
    for job in jobs:
        env = policy.bind_five_job_policy(job, {})
        assert env["MSA_PRETRAIN_RESUME"].endswith("checkpoint-004500000000.pth")
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == (
            "450000000" if job["task"] == "five-wave-training" else "0"
        )
    policy.enforce_submission_budget(jobs, {"jobs": []})
