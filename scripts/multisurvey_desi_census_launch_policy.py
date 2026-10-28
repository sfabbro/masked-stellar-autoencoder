"""Fail-closed, source-only DESI/cached-input identity census policy."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_desi_cache_census_v1"
CONFIG = Path("configs/multisurvey-desi-census-20261027.yaml")
EXPECTED = {
    "run_id": RUN,
    "sidecar_manifest": ROOT + "multisurvey_desi_dr1_raw_source_v1/manifest.json",
    "sidecar_path": ROOT
    + "multisurvey_desi_dr1_raw_source_v1/desi-raw-observations.h5",
    "sidecar_sha": "fe09ff1be95bb95884cbba9f008d02bc3e52e0debd143add5a5d28efad934ba3",
    "base_manifest": ROOT + "remediation_source_v3/manifest.json",
    "base_path": ROOT + "remediation_source_v3/prepared.h5",
    "base_sha": "112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c",
    "cache_path": ROOT + "residual-huber-e86-150m-cache-v1",
    "original_path": "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5",
    "expected_cache_rows": 150000000,
    "chunk_rows": 65536,
}


def bind_desi_census_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != "multisurvey-desi-census":
        raise ValueError("Unknown DESI census")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("DESI census source contract differs")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("DESI census resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("DESI census requires verified public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError(
            "Census producer manifests are checked at runtime; overrides prohibited"
        )
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("DESI census output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
