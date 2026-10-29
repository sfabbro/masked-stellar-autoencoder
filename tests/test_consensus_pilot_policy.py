"""The two distinct pilot plans are bound to immutable inputs and outputs."""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from multisurvey_consensus_pilot_launch_policy import bind_consensus_pilot_policy


def test_frozen_and_finetuned_plans_are_distinct_and_fail_closed():
    plans = yaml.safe_load(
        (ROOT / "configs/multisurvey-consensus-launch-20261028.yaml").read_text()
    )["jobs"]
    assert len(plans) == 2
    assert plans[0]["run_id"] != plans[1]["run_id"]
    for plan in plans:
        output = (
            "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
            + plan["run_id"]
        )
        result = bind_consensus_pilot_policy(plan, {"MSA_EXPERIMENT_OUTPUT": output})
        config = yaml.safe_load((ROOT / result["MSA_MULTISURVEY_POLICY"]).read_text())
        assert config["run_id"] == plan["run_id"] and config["updates"] == 1000
        with pytest.raises(ValueError):
            bind_consensus_pilot_policy(
                plan, {"MSA_EXPERIMENT_OUTPUT": output + "-duplicate"}
            )
        with pytest.raises(ValueError):
            bind_consensus_pilot_policy(
                {**plan, "resources": {"cores": 4, "ram": 64, "gpu": 0}},
                {"MSA_EXPERIMENT_OUTPUT": output},
            )
