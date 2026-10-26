"""Explicitly authorized remediation preflights and pilots; immutable first attempts."""

import ast
from collections.abc import Mapping
from pathlib import Path

import yaml

CONFIGS = {
    "remediation-input-audit": Path("configs/remediation-input-audit-20261026-v4.yaml"),
    "remediation-source": Path("configs/remediation-source-20261026-v3.yaml"),
    "remediation-joint-preflight": Path(
        "configs/remediation-joint-preflight-20261026-v3.yaml"
    ),
    "remediation-pilot": Path("configs/remediation-pilots-20261026.yaml"),
}
LIBRARY = "6bd8b93b35fe4fe216319870e72e94eb78429106"
PARENT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/residual_huber_e86_150m_30e_v1/checkpoints/checkpoint-004500000000.pth"


def bind_remediation_policy(plan, env):
    task = plan["task"]
    config = CONFIGS[task]
    cfg = yaml.safe_load(config.read_text())
    runs = (
        {a["run_id"] for a in cfg.get("arms", [])}
        if task == "remediation-pilot"
        else {cfg["run_id"]}
    )
    if plan["run_id"] not in runs:
        raise ValueError("Unknown remediation run")
    expected_gpu = 0 if task in {"remediation-input-audit", "remediation-source"} else 1
    if plan.get("resources", {}).get("gpu") != expected_gpu:
        raise ValueError("Remediation CPU/GPU resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Require the verified public terminal image")
    if (
        task in {"remediation-pilot", "remediation-joint-preflight"}
        and cfg["library_revision"] != LIBRARY
    ):
        raise ValueError("Remediation library pin differs")
    expected = {
        "MSA_REMEDIATION_POLICY": str(config),
        "MSA_PRETRAIN_RESUME": PARENT,
        "MSA_EXPERIMENT_PRESENTATIONS": "0",
    }
    for key, value in expected.items():
        if key in plan.get("env", {}) and str(plan["env"][key]) != value:
            raise ValueError("Remediation source/config/budget override differs")
        env[key] = value
    return env


def assert_project_storage_budget(info, minimum_available_bytes=20_000_000_000):
    """Fail closed before submission when ARC's project quota lacks output space."""
    props = info.get("properties", {})
    if isinstance(props, str):
        props = ast.literal_eval(props)
    if not isinstance(props, Mapping):
        raise ValueError("ARC quota metadata is unavailable")
    quota = int(props["ivo://ivoa.net/vospace/core#quota"])
    used = int(props["ivo://ivoa.net/vospace/core#length"])
    if quota <= 0 or used < 0 or minimum_available_bytes < 0:
        raise ValueError("Invalid ARC quota accounting")
    available = quota - used
    if available < minimum_available_bytes:
        raise ValueError(
            f"ARC project quota blocks remediation submission: used={used}, quota={quota}, available={available}, required_output_reserve={minimum_available_bytes}; no job attempted"
        )
    return dict(
        quota_bytes=quota,
        used_bytes=used,
        available_bytes=available,
        minimum_available_bytes=minimum_available_bytes,
    )
