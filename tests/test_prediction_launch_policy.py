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
