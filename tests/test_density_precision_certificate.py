import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_stellar_improvement import bind_prediction_policy
from run_density_precision_certificate import compare


def test_absolute_density_gate_is_shift_invariant_and_rejects_nonfinite():
    ref = np.array([2.0, 3.0], dtype="f8")
    good = ref + 1e-10
    bad = ref + 1e-7
    for shift in (0.0, 5.4979676):
        assert compare({"density": ref + shift}, {"density": good + shift})["density"][
            "passed"
        ]
        assert not compare({"density": ref + shift}, {"density": bad + shift})[
            "density"
        ]["passed"]
    assert not compare({"density": ref}, {"density": np.array([np.nan, 3.0])})[
        "density"
    ]["passed"]
    with pytest.raises(ValueError, match="Float64"):
        compare({"density": ref}, {"density": ref.astype("f4")})
    with pytest.raises(ValueError, match="Float64"):
        compare({"density": ref}, {"density": ref[:1]})


def test_certificate_zero_training_budget_and_immutable_inputs():
    suite = yaml.safe_load(
        Path("configs/density-precision-certificate-20261022.yaml").read_text()
    )
    plan = {"task": "density-precision-certificate", "run_id": suite["run_id"]}
    env = bind_prediction_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_PRETRAIN_RESUME"] == suite["checkpoint"]
    assert len({case["checkpoint_sha256"] for case in suite["cases"]}) == 3
    assert len({case["paired_sha256"] for case in suite["cases"]}) == 1
    with pytest.raises(ValueError):
        bind_prediction_policy(dict(plan, run_id="wrong"), {})
    for key, value in [
        ("MSA_EXPERIMENT_PRESENTATIONS", "1"),
        ("MSA_PRETRAIN_RESUME", "/wrong"),
    ]:
        with pytest.raises(ValueError):
            bind_prediction_policy(dict(plan, env={key: value}), {})
