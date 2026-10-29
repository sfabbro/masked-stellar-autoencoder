"""Fixed, source-only Gaia/cache census launch policy."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_gaia_gspspec_cache_census_v1"
TASK = "multisurvey-gaia-gspspec-cache-census"
CONFIG = Path("configs/multisurvey-gaia-cache-census-20261028.yaml")
EXPECTED = {
    "run_id": RUN,
    "source_path": "/arc/projects/k-pop/catalogues/gaia_rvs_lite_flags.fits.gz",
    "source_sha": "e165c6f1fc4375389f24db86ff093e0c5d6e7c5da7a49b5564998a625f55dd83",
    "expected_source_rows": 5591594,
    "base_manifest": ROOT + "remediation_source_v3/manifest.json",
    "base_path": ROOT + "remediation_source_v3/prepared.h5",
    "base_sha": "112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c",
    "cache_path": ROOT + "residual-huber-e86-150m-cache-v1",
    "original_path": "/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5",
    "expected_cache_rows": 150000000,
    "chunk_rows": 65536,
}


def bind_gaia_cache_census_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != TASK:
        raise ValueError("Unknown Gaia cache census")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("Gaia cache census source contract differs")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("Gaia cache census resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Gaia cache census requires public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Gaia cache census rejects launch overrides")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("Gaia cache census output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
