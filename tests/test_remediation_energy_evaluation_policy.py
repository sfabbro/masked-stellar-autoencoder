import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "energy_recovery_policy",
    Path("scripts/remediation_energy_evaluation_launch_policy.py"),
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_pinned_contract_rejects_wrong_source_and_output(monkeypatch, tmp_path):
    original = p.CONFIG
    config = tmp_path / "config.yaml"
    config.write_bytes(original.read_bytes())
    monkeypatch.setattr(p, "CONFIG", config)
    run = "remediation_nsf6_energy_frozen_eval_recovery_v1"
    plan = {
        "run_id": run,
        "task": p.TASK,
        "resources": {"cores": 4, "ram": 64, "gpu": 1},
        "image": "images.canfar.net/astroai/terminal:latest",
        "depends_on": [],
    }
    assert p.bind_energy_evaluation_policy(
        plan, {"MSA_EXPERIMENT_OUTPUT": p.ROOT + run}
    )["MSA_REMEDIATION_POLICY"] == str(config)
    with pytest.raises(ValueError):
        p.bind_energy_evaluation_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": p.ROOT + "wrong"}
        )
    with pytest.raises(ValueError):
        p.bind_energy_evaluation_policy(
            {**plan, "resources": {"cores": 4, "ram": 64, "gpu": 0}},
            {"MSA_EXPERIMENT_OUTPUT": p.ROOT + run},
        )
    config.write_text("changed")
    with pytest.raises(ValueError, match="SHA"):
        p.bind_energy_evaluation_policy(plan, {"MSA_EXPERIMENT_OUTPUT": p.ROOT + run})
