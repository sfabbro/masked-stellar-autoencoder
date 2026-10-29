import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "remediation_energy6_launch_policy",
    Path("scripts/remediation_energy6_launch_policy.py"),
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_binding_and_fail_closed(monkeypatch, tmp_path):
    src = Path("configs/remediation-nsf6-energy-20261028.yaml")
    path = tmp_path / "policy.yaml"
    path.write_bytes(src.read_bytes())
    monkeypatch.setattr(policy, "CONFIG", path)
    run = "remediation_nsf6_energy_ft_v1"
    plan = dict(
        run_id=run,
        task="remediation-energy-pilot",
        depends_on=[],
        resources=dict(cores=4, ram=64, gpu=1),
        image="images.canfar.net/astroai/terminal:latest",
    )
    env = {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + run}
    assert policy.bind_energy_pilot_policy(plan, env)["MSA_REMEDIATION_POLICY"] == str(
        path
    )
    for bad in (
        {"run_id": "other"},
        {"resources": {"cores": 4, "ram": 64, "gpu": 0}},
        {"env": {"X": "1"}},
        {"depends_on": ["other"]},
        {"image": "other"},
    ):
        with pytest.raises(ValueError):
            policy.bind_energy_pilot_policy(
                {**plan, **bad}, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + run}
            )
    with pytest.raises(ValueError):
        policy.bind_energy_pilot_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + "other"}
        )
    path.write_text("changed")
    with pytest.raises(ValueError, match="SHA"):
        policy.bind_energy_pilot_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + run}
        )
