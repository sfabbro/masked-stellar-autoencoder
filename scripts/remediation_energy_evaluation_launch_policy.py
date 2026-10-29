"""Pin inference-only recoveries to the original failed runs and checkpoints."""

import hashlib
from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
CONFIG = Path("configs/remediation-nsf6-energy-eval-recovery-20261029.yaml")
CONFIG_SHA256 = "5131bd8ab99b1e9c06b16c17e0df027909c01b24bc29b266a75a0b20fe554842"
TASK = "remediation-energy-evaluation-recovery"


def bind_energy_evaluation_policy(plan, env):
    if plan.get("task") != TASK:
        raise ValueError("Invalid recovery task")
    if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("Recovery configuration SHA differs")
    cfg = yaml.safe_load(CONFIG.read_text())
    training = Path(cfg["training_config"])
    if (
        hashlib.sha256(training.read_bytes()).hexdigest()
        != cfg["training_config_sha256"]
    ):
        raise ValueError("Original training configuration SHA differs")
    matches = [r for r in cfg["recoveries"] if r["run_id"] == plan.get("run_id")]
    if len(matches) != 1:
        raise ValueError("Unknown recovery run")
    if (
        plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
    ):
        raise ValueError("Recovery resources or image differ")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Recovery overrides prohibited")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + plan["run_id"]:
        raise ValueError("Recovery requires distinct output")
    env["MSA_REMEDIATION_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
