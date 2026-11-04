"""Bound the overnight non-Gaussian experiment and its project-GPU runtime."""

from pathlib import Path

import yaml

CONFIG = Path("configs/multisurvey-flexible-20261105.yaml")
ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def bind_flexible_policy(plan, env):
    config = yaml.safe_load(CONFIG.read_text())
    if (
        plan["run_id"] != config["run_id"]
        or plan["task"] != "multisurvey-flexible-suite"
        or config["updates"] != 1000
        or config["fit_stars"] != 65536
        or config["tune_stars"] != 2048
        or config["batch_size"] != 64
        or config["micro_batch"] != 8
        or config["particles"] != 256
        or config["evaluation_particles"] != 1024
        or config["density_reference_sha256"]
        != "efd575285789928d2e445d7c11429a7809a35a12e6be29424bd26c48e154c26d"
        or config["parent"]["library_revision"]
        != "2777b0de88121768434725ef227893cd35595d2a"
    ):
        raise ValueError("Flexible density source or numerical contract differs")
    if (
        plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + config["run_id"]
    ):
        raise ValueError("Flexible density resource or output contract differs")
    # This task needs no GaiaXPy flux projection; the tested project GPU lock
    # includes the NF and multi-source library. Preserve the old XP lock.
    env["MSA_XP_EXPERIMENTS"] = "0"
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_PRETRAIN_RESUME"] = config["parent"]["checkpoint"]
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
