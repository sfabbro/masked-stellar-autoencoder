"""Exercise collection planning without network or large downloads."""

import importlib.util
import json
import sys
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "stellar_fetcher",
    Path(__file__).parents[1] / "scripts/fetch_stellar_improvement.py",
)
FETCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCHER)


def test_targeted_first_job_builds_flow_and_augmentation_inventory(
    tmp_path, monkeypatch
):
    runs = [
        "target_e72_flow_nsf8_s42_10m_v1",
        "target_e72_xpnoise_mask_s42_10m_v1",
    ]
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "run_id": run,
                        "session_ids": ["accepted"],
                        "output": "/arc/example/" + run,
                        "plan": {"task": "targeted-prediction"},
                    }
                    for run in runs
                ]
            }
        )
    )
    captured = []
    monkeypatch.setattr(
        FETCHER,
        "fetch",
        lambda task: captured.append(task) or (task[0]["run_id"], task[1], "test"),
    )
    monkeypatch.setattr(
        sys, "argv", ["fetch", "--record", str(record), "--initial-predictions"]
    )
    FETCHER.main()
    files = {run: {t[1] for t in captured if t[0]["run_id"] == run} for run in runs}
    for run in runs:
        assert {
            "checkpoint.pth",
            "paired-cohort.npz",
            "initial_predictions.npz",
            "reserved-tuning.npz",
        } <= files[run]
    assert "conditional-flow-diagnostics.npz" in files[runs[0]]
    assert "xp-error-source-audit.json" in files[runs[1]]
