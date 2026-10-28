"""Fail-closed two-update multi-survey CUDA preflight launch policy."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_consensus_cuda_preflight_v1"
CONFIG = Path("configs/multisurvey-consensus-preflight-20261028.yaml")
EXPECTED = {
    "run_id": RUN,
    "overlay_producer_manifest": ROOT
    + "multisurvey_desi_unique_overlay_v1/manifest.json",
    "overlay_manifest": ROOT
    + "multisurvey_desi_unique_overlay_v1/bundle/overlay-manifest.json",
    "overlay_manifest_sha256": "09488a33f21307f3152ce1a6a70a76489b88e55a1b790a44d83e11245955d0ff",
    "base_path": ROOT + "remediation_source_v3/prepared.h5",
    "checkpoint": ROOT
    + "residual_huber_e86_150m_30e_v1/checkpoints/checkpoint-004500000000.pth",
    "checkpoint_sha256": "c544548ba203f249e11509d0e0f9832df0e8366e6f7b0c7a639460752061a2f9",
    "reference_checkpoint": "/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth",
    "reference_checkpoint_sha256": "d1ed550e3650817df1e1337b4e405fd50fefa43010866b828faa1aedf5040fb3",
    "model_config": "configs/pretrain.canfar.yaml",
    "library_revision": "82ca1996cc4f3ca0be3e730d38a99db482a79378",
    "seed": 20261028,
}


def bind_consensus_preflight_policy(plan, env):
    if (
        plan.get("run_id") != RUN
        or plan.get("task") != "multisurvey-consensus-preflight"
    ):
        raise ValueError("Unknown multi-survey preflight")
    if yaml.safe_load(CONFIG.read_text()) != EXPECTED:
        raise ValueError("Preflight source contract differs")
    if plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}:
        raise ValueError("Preflight resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Preflight requires verified public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError(
            "Preflight checks the producer manifest itself; overrides prohibited"
        )
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("Preflight output must be distinct")
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
