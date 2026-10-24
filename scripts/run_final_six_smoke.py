"""One headless CUDA gate for five source-bound actual six-target code paths."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import run_final_six_frozen as frozen
import run_final_six_recipes as runner
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = Path("configs/final-six-recipes-20261024.yaml")
    cfg = yaml.safe_load(config.read_text())
    ids = [
        next(
            a["run_id"]
            for a in cfg["arms"]
            if a["kind"] == kind
            and a.get("auxiliary") == aux
            and bool(a.get("parent86_control")) == control
            and (kind != "mdn" or a.get("ranking_weight", 0))
            and (not mixed or a.get("mixed_weight") == 0.1)
        )
        for kind, aux, control, mixed in [
            ("gaussian", "triloss", False, False),
            ("mdn", None, False, False),
            ("factorized_nsf", None, False, True),
            ("factorized_nsf", "triloss", False, False),
            ("gaussian", None, True, False),
        ]
    ]
    report = dict(
        status="running",
        run_id="final_six_cuda_smoke_v1",
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        actual_native_labels=True,
        production_science_certificate=False,
        shared_runner_sha256=runner.sha(Path(runner.__file__)),
        shared_module_sha256=runner.sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/final_six_recipes.py"
        ),
        case_reports=[],
    )
    runner.native.atomic(args.output / "manifest.json", report)
    try:
        for run_id in ids:
            output = args.output / run_id
            runner.run(
                SimpleNamespace(config=config, run_id=run_id, output=output, smoke=True)
            )
            result = json.loads((output / "manifest.json").read_text())
            assert (
                result["status"] == "complete"
                and result["smoke"]
                and result["training_budget"] == dict(presentations=512, updates=2)
                and result["full_state_reload_equal"]
            )
            report["case_reports"].append(
                dict(
                    run_id=run_id,
                    training_budget=result["training_budget"],
                    checkpoint_sha256=result["checkpoint_sha256"],
                    gpu=result["gpu"],
                )
            )
            runner.native.atomic(args.output / "manifest.json", report)
        control = Path("configs/final-six-frozen-20261024.yaml")
        for run_id in [
            "final6_large_frozen_gaussian_s49_v1",
            "final6_large_frozen_factorized_nsf_mixed01_s49_v1",
        ]:
            output = args.output / run_id
            frozen.run(
                SimpleNamespace(
                    config=control, run_id=run_id, output=output, smoke=True
                )
            )
            result = json.loads((output / "manifest.json").read_text())
            assert (
                result["status"] == "complete"
                and result["smoke"]
                and result["full_state_reload_equal"]
                and result["frozen_model_values_unchanged"]
            )
            assert (
                result["training_budget"]["encoder_updates"] == 0
                and result["training_budget"]["reconstruction_presentations"] == 0
            )
            report["case_reports"].append(
                dict(
                    run_id=run_id,
                    training_budget=result["training_budget"],
                    checkpoint_sha256=result["checkpoint_sha256"],
                )
            )
            runner.native.atomic(args.output / "manifest.json", report)
        report.update(
            status="complete",
            bounded_reconstruction_presentations=512 * len(ids),
            production_science_certificate=False,
        )
        runner.native.atomic(args.output / "manifest.json", report)
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        runner.native.atomic(args.output / "manifest.json", report)
        raise


if __name__ == "__main__":
    main()
