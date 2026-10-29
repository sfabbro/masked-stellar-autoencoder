import copy
import importlib.util
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).parent
SCRIPTS = HERE.parent / "scripts"
if not (SCRIPTS / "multisurvey_rave_cnn_launch_policy.py").exists():
    SCRIPTS = Path("scripts")
spec = importlib.util.spec_from_file_location(
    "policy", SCRIPTS / "multisurvey_rave_cnn_launch_policy.py"
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_policy_accepts_pinned_job():
    plan = {
        "run_id": policy.RUN,
        "task": policy.TASK,
        "resources": {"cores": 8, "ram": 128, "gpu": 0},
        "image": "images.canfar.net/astroai/terminal:latest",
        "depends_on": [],
    }
    env = {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + policy.RUN}
    assert policy.bind_rave_cnn_policy(plan, env)["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    for key, value in [
        ("run_id", "other"),
        ("task", "other"),
        ("image", "bad"),
        ("resources", {"cores": 8, "ram": 128, "gpu": 1}),
        ("depends_on", ["other"]),
        ("env", {"x": "y"}),
    ]:
        bad = copy.deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            policy.bind_rave_cnn_policy(
                bad, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + policy.RUN}
            )
    with pytest.raises(ValueError):
        policy.bind_rave_cnn_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + "other"}
        )


def test_launch_yaml_binds_policy():
    path = HERE.parent / "configs/multisurvey-rave-cnn-source-launch-20261028.yaml"
    if not path.exists():
        path = Path("configs/multisurvey-rave-cnn-source-launch-20261028.yaml")
    p = yaml.safe_load(path.read_text())["jobs"][0]
    policy.bind_rave_cnn_policy(p, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + policy.RUN})
