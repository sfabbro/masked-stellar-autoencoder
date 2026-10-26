import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import remediation_launch_policy as policy


@pytest.mark.parametrize(
    "task,gpu",
    [
        ("remediation-input-audit", 0),
        ("remediation-source", 0),
        ("remediation-joint-preflight", 1),
        ("remediation-pilot", 1),
    ],
)
def test_remediation_explicit_config_resources_and_pin(
    tmp_path, monkeypatch, task, gpu
):
    f = tmp_path / "policy.yaml"
    f.write_text(
        yaml.safe_dump(
            dict(
                run_id="test",
                arms=[dict(run_id="test")],
                library_revision=policy.LIBRARY,
            )
        )
    )
    monkeypatch.setitem(policy.CONFIGS, task, f)
    plan = dict(
        run_id="test",
        task=task,
        resources=dict(gpu=gpu),
        image="images.canfar.net/astroai/terminal:latest",
    )
    env = policy.bind_remediation_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_REMEDIATION_POLICY"] == str(f)
    for bad in [
        dict(run_id="unknown"),
        dict(resources=dict(gpu=1 - gpu)),
        dict(image="private/image"),
        dict(env=dict(MSA_EXPERIMENT_PRESENTATIONS="450000000")),
    ]:
        with pytest.raises(ValueError):
            policy.bind_remediation_policy(dict(plan, **bad), {})
