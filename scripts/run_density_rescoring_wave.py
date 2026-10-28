"""One immutable historical checkpoint per evaluation-only CUDA rescoring job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import rescore_density_checkpoint as rescore
import yaml


def make_policy(case, config):
    """Bind published scoring code separately from original fitted-model provenance."""
    policy = dict(case["artifact_policy"])
    policy.update(
        library_revision=config["library_revision"],
        scoring_code_sha256=rescore.digest(rescore.scoring.__file__),
        head_code_sha256=rescore.digest(rescore.heads.__file__),
        architecture_code_sha256=rescore.digest(rescore.architecture.__file__),
        model_config_sha256=rescore.digest(config["model_config"]),
    )
    return policy


def selected_case(config, run_id):
    cases = config["cases"]
    if len({case["run_id"] for case in cases}) != len(cases):
        raise ValueError("duplicate scoring run IDs")
    match = [case for case in cases if case["run_id"] == run_id]
    if len(match) != 1:
        raise ValueError("unknown rescoring run ID")
    if config["evaluation_batch"] != 128 or config["training_updates"] != 0:
        raise ValueError("fixed bounded-memory batch128 and zero SGD policy required")
    return match[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/density-rescoring-wave-20261022.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config.read_text())
    case = selected_case(config, args.run_id)
    # Wrapper never creates files in the fitted producer's output directory.
    source = Path(config["base"]) / case["artifact_policy"]["run_id"]
    if (
        source.resolve() == args.output.resolve()
        or source.resolve() in args.output.resolve().parents
    ):
        raise ValueError("output must be outside original run")
    args.output.mkdir(parents=True, exist_ok=False)
    policy_path = args.output / "immutable-scoring-policy.json"
    policy_path.write_text(json.dumps(make_policy(case, config), indent=2) + "\n")
    manifest = dict(
        run_id=args.run_id,
        status="running",
        training_updates=0,
        config_sha256=rescore.digest(args.config),
        original_run_id=source.name,
        original_failures_cleared=False,
        corrected_bpf_map=False,
        library_revision=config["library_revision"],
        scope="precision-only full-role rescoring; independent verification pending",
    )

    def save():
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    save()
    try:
        subprocess.run(
            [
                sys.executable,
                "scripts/rescore_density_checkpoint.py",
                "--policy",
                str(policy_path),
                "--directory",
                str(source),
                "--parent",
                config["parent"],
                "--model-config",
                config["model_config"],
                "--device",
                "cuda",
                "--output",
                str(args.output / "corrected"),
            ],
            check=True,
        )
        manifest["status"] = "complete_scoring_only"
        manifest["corrected_manifest_sha256"] = rescore.digest(
            args.output / "corrected" / "manifest.json"
        )
        save()
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
