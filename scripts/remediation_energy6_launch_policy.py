"""Immutable distinct six-target joint NSF sample-score pilot policy."""

import hashlib
from pathlib import Path

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
CONFIG = Path("configs/remediation-nsf6-energy-20261028.yaml")
CONFIG_SHA256 = "059f996a9c91e8faacd73f0cc8983814632e31ce9f4e930f42ab0def5509e3f0"
RUNS = {
    "remediation_nsf6_energy_frozen_v1",
    "remediation_nsf6_energy_ft_v1",
    "remediation_nsf6_energy_ft_recon003_v1",
}


def bind_energy_pilot_policy(plan, env):
    if plan.get("task") != "remediation-energy-pilot" or plan.get("run_id") not in RUNS:
        raise ValueError("Unknown energy pilot")
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Energy pilot configuration SHA differs")
    if plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}:
        raise ValueError("Energy pilot resources differ")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Energy pilot image differs")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Energy pilot rejects dependency and environment overrides")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + plan["run_id"]:
        raise ValueError("Energy pilot output must be distinct")
    env["MSA_REMEDIATION_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
