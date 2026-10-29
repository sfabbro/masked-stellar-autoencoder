import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "probe", Path("scripts/run_lamost_dr12_network_probe.py")
)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
spec2 = importlib.util.spec_from_file_location(
    "policy", Path("scripts/lamost_dr12_network_probe_policy.py")
)
q = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(q)


def test_records_all_cases_and_never_overwrites(tmp_path):
    seen = []

    def check(url, *, direct):
        seen.append((url, direct))
        return {
            "status": "response",
            "http_status": 206,
            "bytes_read": 4096,
            "gzip_magic": True,
        }

    result = p.run(tmp_path / "out", "a" * 40, check=check)
    assert (
        result["status"] == "lamost_network_probe_complete"
        and len(seen) == 4
        and not result["training_performed"]
    )
    assert json.loads((tmp_path / "out/manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        p.run(tmp_path / "out", "a" * 40, check=check)


def test_launch_policy():
    plan = {
        "run_id": q.RUN,
        "task": q.TASK,
        "resources": {"cores": 8, "ram": 128, "gpu": 0},
        "image": "images.canfar.net/astroai/terminal:latest",
        "depends_on": [],
    }
    assert (
        q.bind_lamost_probe_policy(plan, {"MSA_EXPERIMENT_OUTPUT": q.ROOT + q.RUN})[
            "MSA_EXPERIMENT_PRESENTATIONS"
        ]
        == "0"
    )
    with pytest.raises(ValueError):
        q.bind_lamost_probe_policy(plan, {"MSA_EXPERIMENT_OUTPUT": q.ROOT + "wrong"})
