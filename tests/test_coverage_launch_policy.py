"""Bind the headless coverage wave to declared inputs and distinct outputs."""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "coverage_launcher", ROOT / "scripts/launch_stellar_improvement.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_all_coverage_policies_bind_and_no_old_intent_is_reused():
    jobs = yaml.safe_load(
        (ROOT / "configs/coverage-launch-wave-20261023.yaml").read_text()
    )["jobs"]
    assert len(jobs) == len({j["run_id"] for j in jobs}) == 40
    for job in jobs:
        env = launcher.bind_prediction_policy(job, dict(job.get("env", {})))
        assert env["MSA_PRETRAIN_RESUME"].endswith("_86.pth")
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == (
            "10000000" if job["task"].startswith("multilabel-") else "0"
        )
        altered = dict(
            job, env={**job.get("env", {}), "MSA_EXPERIMENT_PRESENTATIONS": "999"}
        )
        with pytest.raises(ValueError):
            launcher.bind_prediction_policy(altered, {})


def test_target_and_solver_overrides_rejected():
    for task, run, key in [
        ("target-predictions", "posterior_age_distance_nsf_s45_v1", "MSA_TARGET_SEED"),
        ("forward-floweb", "forward_lbfgs_fp64_multistart_v2", "MSA_FORWARD_MODE"),
    ]:
        with pytest.raises(ValueError):
            launcher.bind_prediction_policy(
                dict(task=task, run_id=run, env={key: "invalid"}), {}
            )
    with pytest.raises(ValueError):
        launcher.bind_prediction_policy(
            dict(task="target-source-evidence", run_id="invented"), {}
        )


def test_alpha_jobs_require_declared_independent_source_dependencies():
    jobs = yaml.safe_load(
        (ROOT / "configs/alpha-launch-wave-20261023.yaml").read_text()
    )["jobs"]
    assert len(jobs) == 7
    for job in jobs:
        env = launcher.bind_prediction_policy(job, {})
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
        assert len(job["depends_on"]) == 1
        if job["task"] == "alpha-target-predictions":
            assert env["MSA_TARGET_KIND"] in {"gaussian", "mdn", "nsf"}
            assert len(env["MSA_TARGET_LIBRARY_REV"]) == 40
            assert job["depends_on"] == ["alpha_desi_feature_preparation_v2"]
        else:
            assert job["depends_on"] == ["alpha_desi_source_projection_v1"]
        assert not launcher.eligible(job, None, {})
        assert launcher.eligible(job, None, {job["depends_on"][0]: True})


def test_floweb_recovery_reuses_exact_prior_without_retraining_or_wider_tolerance():
    cfg = yaml.safe_load(
        (ROOT / "configs/floweb-quadrature-recovery-20261023.yaml").read_text()
    )
    assert cfg["policy"]["absolute_log_density_tolerance"] == 0.01
    assert (
        cfg["policy"]["prior_training_updates"] == cfg["policy"]["decoder_updates"] == 0
    )
    for job in yaml.safe_load(
        (ROOT / "configs/floweb-recovery-launch-20261023.yaml").read_text()
    )["jobs"]:
        env = launcher.bind_prediction_policy(job, {})
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
        assert len(env["MSA_FLOWEB_PRIOR_SHA"]) == 64
        altered = dict(job, env={"MSA_FLOWEB_PRIOR_SHA": "bad"})
        with pytest.raises(ValueError):
            launcher.bind_prediction_policy(altered, {})
