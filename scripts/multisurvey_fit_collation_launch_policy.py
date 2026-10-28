"""Fail-closed CPU-only launch policy for the FIT collation census."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_fit_collation_census_v1"
CONFIG = Path("configs/multisurvey-fit-collation-census-20261028.yaml")
EXPECTED = {
    "run_id": RUN,
    "overlay_producer_manifest": ROOT
    + "multisurvey_desi_unique_overlay_v1/manifest.json",
    "overlay_manifest": ROOT
    + "multisurvey_desi_unique_overlay_v1/bundle/overlay-manifest.json",
    "overlay_manifest_sha256": "09488a33f21307f3152ce1a6a70a76489b88e55a1b790a44d83e11245955d0ff",
    "overlay_sha256": "60c097676bab21fe03f04afc0b26a608ee1ea0e7ea93a1577725dccb97a7a32a",
    "expected_fit_ids_sha256": "870027a5393713aa5e507534750b048900428b8f9f839475181b7cbb0c010ed8",
}


def bind_fit_collation_census_policy(plan, env):
    if (
        plan.get("run_id") != RUN
        or plan.get("task") != "multisurvey-fit-collation-census"
    ):
        raise ValueError("Unknown FIT collation census")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("FIT collation census source contract differs")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("FIT collation census resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("FIT collation census requires public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Census checks its source producer; overrides prohibited")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("FIT collation census output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
