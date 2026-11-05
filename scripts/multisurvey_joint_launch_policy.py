"""Bind the authorized full-FIT, three-pass joint reconstruction/mixture run."""

from pathlib import Path

import yaml

CONFIG = Path("configs/multisurvey-joint-20261105.yaml")
ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def bind_joint_policy(plan, env):
    c = yaml.safe_load(CONFIG.read_text())
    if (
        plan["task"] != "multisurvey-joint-fullfit"
        or plan["run_id"] != c["run_id"]
        or c["epochs"] != 3
        or c["batch_size"] != 1024
        or c["micro_batch"] != 128
        or c["reconstruction_weight"] != 0.03
        or c["tune_stars"] != 16384
        or c["reference_sha256"]
        != "efd575285789928d2e445d7c11429a7809a35a12e6be29424bd26c48e154c26d"
        or c["mixture_sha256"]
        != "448b7e00332b5133c55148ff2ed543df82c0e6822c764b3381ca3008e8b47184"
        or c["parent"]["library_revision"] != "2777b0de88121768434725ef227893cd35595d2a"
        or plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + c["run_id"]
    ):
        raise ValueError("Full-FIT reconstruction/mixture contract differs")
    env.update(
        MSA_XP_EXPERIMENTS="0",
        MSA_MULTISURVEY_POLICY=str(CONFIG),
        MSA_PRETRAIN_RESUME=c["parent"]["checkpoint"],
        MSA_EXPERIMENT_PRESENTATIONS="0",
    )
    return env
