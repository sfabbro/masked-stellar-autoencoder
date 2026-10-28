"""Network-free preservation inventory for the three new density screens."""

import importlib.util
import json
import sys
from pathlib import Path


def test_density_wave_fetch_inventory_preserves_joint_scores_and_full_state(
    monkeypatch, tmp_path
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts/fetch_stellar_improvement.py"
    )
    spec = importlib.util.spec_from_file_location(
        "density_wave_fetch_inventory", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = "multi_nsf_density_contrastive_s42_v1"
    record = tmp_path / "ledger.json"
    record.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "run_id": name,
                        "session_ids": ["accepted-test-session"],
                        "output": "/arc/test/" + name,
                        "plan": {"task": "multilabel-density-screen"},
                    }
                    for name in (selected, "unselected_density_run")
                ]
            }
        )
    )
    fetched = []

    def capture(task):
        job, relative, output = task
        fetched.append((job["run_id"], relative, output))
        return job["run_id"], relative, "network-free-test"

    class InlineExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, tasks):
            return map(function, tasks)

    monkeypatch.setattr(module, "fetch", capture)
    monkeypatch.setattr(module.concurrent.futures, "ThreadPoolExecutor", InlineExecutor)
    monkeypatch.setattr(
        sys, "argv", [str(script), "--record", str(record), "--runs", selected]
    )
    module.main()
    names = {relative for _, relative, _ in fetched}
    assert names == {
        "manifest.json",
        "session.log",
        "paired-cohort.npz",
        "predictions.npz",
        "checkpoint.pth",
        "initial_reconstruction.npz",
        "final_reconstruction.npz",
        "metrics.jsonl",
        "conformal.npz",
        "native-provenance.json",
        "xp-unit-audit.json",
        "flow-moment-memory-preflight.json",
        "density-calibration-summary.json",
        "density-calibration-arrays.npz",
        "snapshot-32768.npz",
        "snapshot-3014656.npz",
        "snapshot-6029312.npz",
        "snapshot-9032768.npz",
        "snapshot-10000000.npz",
    }
    assert len(names) == len(fetched)
    for name, relative, output in fetched:
        assert name == selected
        assert (
            output
            == Path("experiments/xp/results-stellar-improvement") / selected / relative
        )
