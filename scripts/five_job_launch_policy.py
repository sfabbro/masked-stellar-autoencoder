"""Immutable five-submission budget and parent/budget guards for the large screen."""

from pathlib import Path

import yaml

PLAN = Path("configs/five-job-launch-20261024.yaml")
TRAINING = Path("configs/five-wave-training-20261024.yaml")
SOURCE_RUN = "five_job_broad_test_preparation_v1"


def validated_plans():
    jobs = yaml.safe_load(PLAN.read_text())["jobs"]
    ids = [job["run_id"] for job in jobs]
    source = [j for j in jobs if j["task"] == "five-job-source-preparation"]
    training = [j for j in jobs if j["task"] == "five-wave-training"]
    if len(jobs) != 5 or len(set(ids)) != 5 or len(source) != 1 or len(training) != 4:
        raise ValueError("Five-job wave requires exactly one preparation and four fits")
    if source[0]["run_id"] != SOURCE_RUN:
        raise ValueError("Unknown preparation intent")
    for job in training:
        if job.get("depends_on") != [SOURCE_RUN]:
            raise ValueError(
                "Training requires the actual successful source preparation"
            )
    return jobs


def enforce_submission_budget(plans, record):
    """Count attempts, including failed/ambiguous attempts, against the same cap."""
    jobs = validated_plans()
    ids = {j["run_id"] for j in jobs}
    if any(j["run_id"] not in ids for j in plans):
        raise ValueError("Replacement or extra intent exceeds the five-job policy")
    attempted = [j for j in record.get("jobs", []) if j["run_id"] in ids]
    if sum(int(j.get("attempts", bool(j.get("attempted_at")))) for j in attempted) > 5:
        raise ValueError("Five-submission budget already exhausted")
    if any(int(j.get("attempts", 0)) > 1 for j in attempted):
        raise ValueError("This wave forbids repeat attempts, including ambiguous ones")


def bind_five_job_policy(plan, env):
    """Resolve the big checkpoint and reject a different arm, budget or resource role."""
    jobs = validated_plans()
    actual = next((j for j in jobs if j["run_id"] == plan["run_id"]), None)
    if actual is None or plan != actual:
        raise ValueError("Five-job launch differs from its immutable plan")
    suite = yaml.safe_load(TRAINING.read_text())
    training = plan["task"] == "five-wave-training"
    if training and plan["run_id"] not in {a["run_id"] for a in suite["arms"]}:
        raise ValueError("Unknown large-training head")
    if training and (
        suite["reconstruction_training_rows"] != 150_000_000
        or suite["reconstruction_presentations"] != 450_000_000
        or suite["reconstruction_presentations"]
        != 3 * suite["reconstruction_training_rows"]
    ):
        raise ValueError("Large screen must run three passes over the original 150M")
    if plan.get("resources", {}).get("gpu", 0) != int(training):
        raise ValueError("Preparation must be CPU-only and each fit must have one GPU")
    expected = {
        "MSA_PRETRAIN_RESUME": suite["checkpoint"],
        "MSA_EXPERIMENT_PRESENTATIONS": str(suite["reconstruction_presentations"])
        if training
        else "0",
    }
    for key, value in expected.items():
        if key in plan.get("env", {}) and str(plan["env"][key]) != value:
            raise ValueError("Five-job parent or presentation override differs")
        env[key] = value
    return env
