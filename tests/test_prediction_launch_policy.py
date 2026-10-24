"""A numerical policy parent must override the generic legacy launch default."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_stellar_improvement import PARENT, bind_prediction_policy


def test_all_prediction_launch_plans_resolve_parent_and_budget():
    suite = yaml.safe_load(Path("configs/prediction-comparison-10m.yaml").read_text())
    plans = yaml.safe_load(Path("configs/stellar-improvement-plan.yaml").read_text())[
        "jobs"
    ]
    for plan in plans:
        if plan["task"] == "prediction-comparison":
            env = bind_prediction_policy(plan, {"MSA_PRETRAIN_RESUME": PARENT})
            assert env["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
            assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "10000000"


def test_conflicting_explicit_parent_is_rejected():
    plan = dict(
        task="prediction-comparison",
        run_id="prediction_e72_corrected_score_xp_10m_v2",
        env={"MSA_PRETRAIN_RESUME": PARENT},
    )
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(plan, {})


def test_forward_eiv_is_explicitly_not_reconstruction_training():
    suite = yaml.safe_load(Path("configs/forward-eiv-screen.yaml").read_text())
    env = bind_prediction_policy(
        dict(task="forward-eiv-screen", run_id=suite["run_id"]), {}
    )
    assert env["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(dict(task="forward-eiv-screen", run_id="wrong"), {})


def test_targeted_overrides_and_testtime_bind_actual_parent():
    suite = yaml.safe_load(Path("configs/targeted-10m-20261019.yaml").read_text())
    for arm in suite["arms"]:
        env = bind_prediction_policy(
            dict(task="targeted-prediction", run_id=arm["run_id"]), {}
        )
        assert env["MSA_PRETRAIN_RESUME"] == arm.get("policy_overrides", {}).get(
            "checkpoint", suite["checkpoint"]
        )
        assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "10000000"
    suite = yaml.safe_load(Path("configs/test-time-screen.yaml").read_text())
    env = bind_prediction_policy(
        dict(task="test-time-screen", run_id=suite["run_id"]), {}
    )
    assert env["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"


def test_form_transport_preserves_repeated_env_special_characters():
    from urllib.parse import parse_qsl

    from launch_stellar_improvement import encode_creation_form

    fields = [
        ("name", "test"),
        ("env", "A=a b&c=2"),
        ("env", "B=λ"),
        ("args", "-u -c code()"),
    ]
    assert parse_qsl(encode_creation_form(fields).decode()) == fields


def test_large_huber_binds_parent86_and_real150m_budget():
    suite = yaml.safe_load(Path("configs/residual-huber-150m.yaml").read_text())
    env = bind_prediction_policy(
        dict(task="large-huber", run_id="residual_huber_e86_150m_30e_v1"), {}
    )
    assert env["MSA_PRETRAIN_RESUME"].endswith("checkpoint_86.pth")
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "4500000000"
    assert suite["reconstruction_training_rows"] == 150_000_000
    assert (
        suite["reconstruction_presentations"]
        == 30 * suite["reconstruction_training_rows"]
    )
    assert suite["checkpoint_interval_presentations"] == 150_000_000
    assert suite["early_checkpoint_presentations"] == [9_000_000, 50_000_000]
    with pytest.raises(ValueError, match="differs"):
        bind_prediction_policy(
            dict(
                task="large-huber",
                run_id="residual_huber_e86_150m_30e_v1",
                env={"MSA_PRETRAIN_RESUME": PARENT},
            ),
            {},
        )


def test_alpha_verification_has_zero_budget_and_rejects_overrides():
    plan = dict(
        task="alpha-head-verification", run_id="alpha_frozen_heads_independent_v1"
    )
    env = bind_prediction_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_PRETRAIN_RESUME"].endswith("checkpoint_86.pth")
    with pytest.raises(ValueError, match="Unknown"):
        bind_prediction_policy(dict(plan, run_id="wrong"), {})
    with pytest.raises(ValueError, match="override"):
        bind_prediction_policy(
            dict(plan, env={"MSA_EXPERIMENT_PRESENTATIONS": "1"}), {}
        )


def test_final_preparation_and_guarded_ttt_have_zero_training_budget():
    for task, run in [
        ("six-target-preparation", "six_target_source_preparation_v1"),
        ("guarded-ttt", "guarded_ttt_encoder_sgd20_holdout_v1"),
    ]:
        plan = dict(task=task, run_id=run)
        result = bind_prediction_policy(plan, {})
        assert result["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
        assert result["MSA_PRETRAIN_RESUME"].endswith("checkpoint_86.pth")
        with pytest.raises(ValueError, match="Unknown|absent"):
            bind_prediction_policy(dict(plan, run_id="unknown"), {})
        with pytest.raises(ValueError, match="override"):
            bind_prediction_policy(
                dict(plan, env={"MSA_EXPERIMENT_PRESENTATIONS": "1"}), {}
            )
