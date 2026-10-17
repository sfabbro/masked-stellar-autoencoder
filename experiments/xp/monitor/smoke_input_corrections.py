"""Bounded actual-parent RC/SIMEX integration; original benchmark is a fixture."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = [
    "prediction_e72_rc_shared_xp_10m_v2",
    "prediction_e72_rc_per_star_xp_10m_v2",
    "prediction_e72_simex_xp_10m_v2",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=RUNS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = ROOT / "experiments/xp/results-stellar-improvement"
    parent = results / "prediction-comparison-e72/parent/checkpoint-72.pth"
    cohort = results / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    fixture = (
        results
        / "prediction-e72-local-integration-v2/actual-values-benchmark-SCHEMA-FIXTURE.h5"
    )
    with parent.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == (
            "9de961ce92f0e69838aed174fce6c47179639e23816baf61a45e63d614ff66ec"
        )
    reports = []
    for run in args.runs:
        with (args.output / (run + ".log")).open("x") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_prediction_comparison.py"),
                    "--run-id",
                    run,
                    "--output",
                    str(args.output / run),
                    "--checkpoint",
                    str(parent),
                    "--cohort",
                    str(cohort),
                    "--benchmark",
                    str(fixture),
                    "--smoke",
                ],
                cwd=ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        manifest = json.loads((args.output / run / "manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["smoke"]
        reloads = list((args.output / run).glob("**/checkpoint_verification.json"))
        assert reloads
        for path in reloads:
            check = json.loads(path.read_text())
            assert check["model_and_head_strict_reload"]
            assert check["all_validation_decoder_outputs_byte_exact"]
        reports.append(dict(run_id=run, reload_reports=len(reloads), status="passed"))
    result = dict(
        status="passed",
        arms=reports,
        actual_parent=True,
        actual_cohort=True,
        original_benchmark=False,
        limitation="128 native fit/test rows;512 replay per replica in actual-value SCHEMA FIXTURE. Not original128 XP/error unit certification, full production replay or science.",
        source_status="local working tree; not an immutable production source certificate",
    )
    (args.output / "integration-report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
