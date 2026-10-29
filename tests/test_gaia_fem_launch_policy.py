import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import multisurvey_gaia_fem_join_launch_policy as p
import pytest
import yaml


def test_policy_rejects_override():
    plan = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/multisurvey-gaia-fem-source-join-launch-20261028.yaml"
        ).read_text()
    )["jobs"][0]
    env = {"MSA_EXPERIMENT_OUTPUT": p.ROOT + p.RUN}
    assert p.bind_gaia_fem_join_policy(plan, env)["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    for key, value in [
        ("run_id", "other"),
        ("task", "bad"),
        ("image", "bad"),
        ("resources", {"cores": 8, "ram": 128, "gpu": 1}),
        ("depends_on", ["other"]),
        ("env", {"bad": "value"}),
    ]:
        bad = copy.deepcopy(plan)
        bad[key] = value
        with pytest.raises(ValueError):
            p.bind_gaia_fem_join_policy(bad, {"MSA_EXPERIMENT_OUTPUT": p.ROOT + p.RUN})
