"""CANFAR-only higher-particle refit on the immutable original FIT/TUNE cache."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import yaml
from run_multisurvey_consensus_preflight import atomic, sha
from run_multisurvey_flexible import train


def run(config, output, commit):
    import torch

    if (
        not Path("/arc/projects/k-pop").is_dir()
        or os.environ.get("PYTHONNOUSERSITE") != "1"
        or os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Frozen-cache refinement requires headless CANFAR CUDA")
    installed = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if (
        installed.get("vcs_info", {}).get("commit_id")
        != config["parent"]["library_revision"]
    ):
        raise ValueError("Installed torchregress revision differs")
    original = json.loads(Path(config["frozen_cache"]["producer_manifest"]).read_text())
    receipt = original["cache"]
    if (
        original["source_commit"] != config["frozen_cache"]["source_commit"]
        or receipt["cache_path"] != config["frozen_cache"]["path"]
        or receipt["cache_sha256"] != config["frozen_cache"]["sha256"]
        or sha(receipt["cache_path"]) != receipt["cache_sha256"]
        or receipt["metadata"]["reference_sha256"] != config["density_reference_sha256"]
        or receipt["metadata"]["library_revision"]
        != config["parent"]["library_revision"]
        or original["config"]["parent"] != config["parent"]
    ):
        raise ValueError("Frozen-cache producer/reference contract differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        run_id=config["run_id"],
        source_commit=commit,
        config=config,
        cache=receipt,
        status="refining_frozen_cache",
        scope="Matched higher-particle density fitting on unchanged source-bound FIT/TUNE cache; no encoder, source-map or role changes",
    )
    atomic(output / "manifest.json", manifest)
    try:
        manifest["heads"] = train(config, output, receipt, commit)
        manifest["status"] = "multisurvey_flexible_suite_complete"
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(output / "manifest.json", manifest)
        raise
    atomic(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, commit)


if __name__ == "__main__":
    main()
