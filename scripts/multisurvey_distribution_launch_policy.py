"""Bound the matched multi-survey distribution pilot launch."""

from pathlib import Path

import yaml

CONFIG = Path("configs/multisurvey-distribution-20261104.yaml")
ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def bind_distribution_policy(plan, env):
    arms = {c["run_id"]: c for c in yaml.safe_load(CONFIG.read_text())["experiments"]}
    run = plan["run_id"]
    if run not in arms or plan["task"] != "multisurvey-distribution-pilot":
        raise ValueError("Unknown distribution pilot")
    arm = arms[run]
    if (
        arm["updates"] != 1000
        or arm["batch_size"] != 64
        or arm["parent"]["checkpoint_sha256"]
        != "c544548ba203f249e11509d0e0f9832df0e8366e6f7b0c7a639460752061a2f9"
        or arm["parent"]["overlay_manifest_sha256"]
        != "09488a33f21307f3152ce1a6a70a76489b88e55a1b790a44d83e11245955d0ff"
        or arm["parent"]["library_revision"]
        != "82ca1996cc4f3ca0be3e730d38a99db482a79378"
    ):
        raise ValueError("Distribution source or optimizer contract differs")
    if (
        plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + run
    ):
        raise ValueError("Distribution resource, dependency or output contract differs")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
