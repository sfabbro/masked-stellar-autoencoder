"""Fail-closed role-preserving DESI observation-overlay launch policy."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_desi_unique_overlay_v1"
CONFIG = Path("configs/multisurvey-desi-overlay-20261027.yaml")
EXPECTED = {
    "run_id": RUN,
    "sidecar_manifest": ROOT + "multisurvey_desi_dr1_raw_source_v1/manifest.json",
    "sidecar_path": ROOT
    + "multisurvey_desi_dr1_raw_source_v1/desi-raw-observations.h5",
    "sidecar_sha": "fe09ff1be95bb95884cbba9f008d02bc3e52e0debd143add5a5d28efad934ba3",
    "base_manifest": ROOT + "remediation_source_v3/manifest.json",
    "base_path": ROOT + "remediation_source_v3/prepared.h5",
    "base_sha": "112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c",
    "parent_registry": "configs/multisurvey-parent-registry-20261027.json",
}


def bind_desi_overlay_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != "multisurvey-desi-overlay":
        raise ValueError("Unknown DESI overlay")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("DESI overlay source contract differs")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("DESI overlay resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("DESI overlay requires verified public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError(
            "Overlay checks source producer manifests itself; overrides prohibited"
        )
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("DESI overlay output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
