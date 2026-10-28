"""Fail-closed launch policy for the existing DESI DR1 source only."""

from pathlib import Path

import yaml

CONFIG = Path("configs/multisurvey-desi-source-20261027.yaml")
RUN = "multisurvey_desi_dr1_raw_source_v1"
ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RAW = ROOT + "desi_collection_v1/"
EXPECTED = {
    "run_id": RUN,
    "raw_path": RAW + "mwsall-pix-iron.fits",
    "publisher_path": RAW + "desi-publisher.sha256sum",
    "expected_sha": "eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5",
    "chunk_rows": 65536,
    "snr_min": 10.0,
    "scratch_dir": "/scratch",
}


def bind_multisurvey_desi_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != "multisurvey-desi-source":
        raise ValueError("Unknown DESI source run")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError(
            "DESI source policy differs from the published source contract"
        )
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("DESI source resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("DESI source requires the verified public terminal image")
    if plan.get("depends_on", []) != []:
        raise ValueError("DESI raw preparation has no model or feature dependency")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("DESI source output must be distinct")
    if plan.get("env"):
        raise ValueError("DESI source does not permit environment overrides")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
