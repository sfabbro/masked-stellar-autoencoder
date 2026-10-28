"""Fixed source-only Gaia GSP-Spec overlay launch policy."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_gaia_gspspec_overlay_v1"
CONFIG = Path("configs/multisurvey-gaia-gspspec-overlay-20261028.yaml")
EXPECTED = {
    "run_id": RUN,
    "gaia_path": "/arc/projects/k-pop/catalogues/gaia_rvs_lite_flags.fits.gz",
    "gaia_sha256": "e165c6f1fc4375389f24db86ff093e0c5d6e7c5da7a49b5564998a625f55dd83",
    "gaia_rows": 5591594,
    "base_path": ROOT + "remediation_source_v3/prepared.h5",
    "base_sha256": "112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c",
    "parent_registry": "configs/multisurvey-parent-registry-20261027.json",
    "desi_producer_manifest": ROOT + "multisurvey_desi_unique_overlay_v1/manifest.json",
    "desi_sidecar": ROOT + "multisurvey_desi_unique_overlay_v1/bundle/desi-selected.h5",
    "desi_sidecar_sha256": "7a1a40bf07b32cbc1348f75ce64eedc7c7a707c5d7247a06280c835ec6fadef9",
    "expected_fit_ids_sha256": "870027a5393713aa5e507534750b048900428b8f9f839475181b7cbb0c010ed8",
}


def bind_gaia_overlay_policy(plan, env):
    if (
        plan.get("run_id") != RUN
        or plan.get("task") != "multisurvey-gaia-gspspec-overlay"
    ):
        raise ValueError("Unknown Gaia source overlay")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("Gaia overlay source contract differs")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("Gaia overlay resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Gaia overlay requires the public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Gaia overlay rejects launch overrides")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("Gaia overlay output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
