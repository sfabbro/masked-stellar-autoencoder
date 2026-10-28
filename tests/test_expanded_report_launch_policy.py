import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import expanded_report_launch_policy as policy


@pytest.fixture
def plan(tmp_path, monkeypatch):
    run, producer = next(iter(policy.PRODUCERS.items()))
    specs = {
        "CONFIG": dict(arms=[dict(run_id=run, producer_run_id=producer)]),
        "APPROVAL": dict(
            submission_policy=dict(
                additional_training_jobs=0,
                additional_source_or_inference_jobs="explicitly_authorized_as_needed_by_user",
            )
        ),
        "TRAINING": dict(checkpoint="/arc/immutable-parent-not-loaded.pth"),
    }
    for key, value in specs.items():
        path = tmp_path / (key + ".yaml")
        path.write_text(yaml.safe_dump(value))
        monkeypatch.setattr(policy, key, path)
    return dict(
        run_id=run,
        task="expanded-report-recovery",
        resources=dict(gpu=0),
        depends_on=[],
    )


def test_report_recovery_zero_gpu_zero_presentations(plan):
    env = policy.bind_report_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_EXPANDED_REPORT_POLICY"] == str(policy.CONFIG)
    for bad in [
        dict(resources=dict(gpu=1)),
        dict(run_id="training"),
        dict(depends_on=["running_gpu"]),
        dict(env=dict(MSA_EXPERIMENT_PRESENTATIONS="10000000")),
    ]:
        with pytest.raises(ValueError):
            policy.bind_report_policy(dict(plan, **bad), {})
