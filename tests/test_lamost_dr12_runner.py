import importlib.util
import json
import sys

import pytest

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location(
    "run_lamost_dr12_source", "scripts/run_lamost_dr12_source.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = importlib.util.spec_from_file_location(
    "lamost_policy", "scripts/multisurvey_lamost_dr12_launch_policy.py"
)
policy = importlib.util.module_from_spec(p)
p.loader.exec_module(policy)


def test_runner_preserves_result_and_never_overwrites(tmp_path):
    out = tmp_path / "run"

    def fetch(url, path, n):
        path.write_bytes(b"example")
        return "f" * 64

    def execute(**kw):
        kw["output"].write_bytes(b"table")
        kw["progress"](3, 3)
        return {
            "product": "lrs",
            "source_rows": m.PRODUCTS["lrs"]["rows"],
            "output": str(kw["output"].resolve()),
        }

    r = m.run(
        "multisurvey_lamost_dr12_lrs_source_v1",
        out,
        "a" * 40,
        fetch=fetch,
        execute=execute,
    )
    assert r["status"] == "lamost_dr12_source_complete" and not r["training_performed"]
    assert json.loads((out / "progress.json").read_text())["rows_scanned"] == 3
    with pytest.raises(FileExistsError):
        m.run(
            "multisurvey_lamost_dr12_lrs_source_v1",
            out,
            "a" * 40,
            fetch=fetch,
            execute=execute,
        )


def test_failure_and_policy_are_fail_closed(tmp_path):
    out = tmp_path / "failed"

    def fail(*a):
        raise PermissionError("failed download")

    with pytest.raises(PermissionError):
        m.run("multisurvey_lamost_dr12_lrs_source_v1", out, "a" * 40, fetch=fail)
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    run = "multisurvey_lamost_dr12_lrs_source_v1"
    plan = {
        "run_id": run,
        "task": policy.TASK,
        "resources": {"cores": 8, "ram": 128, "gpu": 0},
        "image": "images.canfar.net/astroai/terminal:latest",
        "depends_on": [],
    }
    assert (
        policy.bind_lamost_dr12_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + run}
        )["MSA_EXPERIMENT_PRESENTATIONS"]
        == "0"
    )
    with pytest.raises(ValueError):
        policy.bind_lamost_dr12_policy(
            plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + "other"}
        )
    with pytest.raises(ValueError):
        policy.bind_lamost_dr12_policy(
            {**plan, "env": {"ANY": "x"}}, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + run}
        )
