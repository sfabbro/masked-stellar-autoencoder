"""Distinct, immutable three-target multi-survey pilot launch policies."""

from pathlib import Path

import yaml
from multisurvey_consensus_preflight_launch_policy import EXPECTED as PREFLIGHT
from multisurvey_consensus_preflight_launch_policy import ROOT

TASK = "multisurvey-consensus-pilot"
RUNS = {
    "multisurvey_consensus_frozen_recon003_pilot_v1": (
        "frozen",
        Path("configs/multisurvey-consensus-frozen-20261028.yaml"),
    ),
    "multisurvey_consensus_ft_recon003_pilot_v1": (
        "finetune",
        Path("configs/multisurvey-consensus-ft-20261028.yaml"),
    ),
}


def bind_consensus_pilot_policy(plan, env):
    run = plan.get("run_id")
    if plan.get("task") != TASK or run not in RUNS:
        raise ValueError("Unknown consensus pilot")
    encoder, config = RUNS[run]
    expected = dict(PREFLIGHT)
    expected.update(
        run_id=run,
        encoder=encoder,
        encoder_lr=0.00000125,
        head_lr=0.001,
        updates=1000,
        batch_size=64,
        tune_every=100,
        panel_size=1024,
        reconstruction_weight=0.03,
    )
    if yaml.safe_load(config.read_text()) != expected:
        raise ValueError("Consensus pilot source/training contract differs")
    if plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}:
        raise ValueError("Consensus pilot resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Consensus pilot requires public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError(
            "Consensus pilot checks producer manifest itself; overrides prohibited"
        )
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + run:
        raise ValueError("Consensus pilot output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(config)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = (
        "0"  # Generic entrypoint budget is unused by this task; actual updates are bound above.
    )
    return env
