"""Explicitly authorized remediation preflights and pilots; immutable first attempts."""

import ast
import hashlib
import json
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

NSF_STARTUP_CONFIG = Path("configs/remediation-nsf-qmc256-startupfix-20261027.yaml")
NSF_QMC256_CONFIG = Path("configs/remediation-nsf-qmc256-recovery-20261026.yaml")
NSF_STARTUP_PREVIOUS = {
    "remediation_nsf6_ft_qmc256_startupfix_v3": {
        "run_id": "remediation_nsf6_ft_qmc256_v2",
        "session_id": "yv16nrs1",
        "source_commit": "4b8d0b5b780366517baecf10e00165a9c9414b21",
        "failed_manifest": "experiments/xp/remediation-nsf-startupfix-evidence-20261027/yv16nrs1.json",
        "failed_manifest_sha256": "b5a7414ee9def9a31c3f2ee219597e81271a41d8f8c884448d974f635541dd40",
        "accepted_record": "experiments/xp/remediation-nsf-qmc256-recovery-20261026.json",
        "failure": "Finetuned encoder has no nonzero gradient",
        "completed_updates": 0,
    },
    "remediation_nsf6_ft_parallax_qmc256_startupfix_v3": {
        "run_id": "remediation_nsf6_ft_parallax_qmc256_v2",
        "session_id": "q6lyfrhv",
        "source_commit": "4b8d0b5b780366517baecf10e00165a9c9414b21",
        "failed_manifest": "experiments/xp/remediation-nsf-startupfix-evidence-20261027/q6lyfrhv.json",
        "failed_manifest_sha256": "b9613837e03a21edeba5a1dc4e0f5dc61c3d894ec407d93d042bae7eb181ff5b",
        "accepted_record": "experiments/xp/remediation-nsf-qmc256-recovery-20261026.json",
        "failure": "Finetuned encoder has no nonzero gradient",
        "completed_updates": 0,
    },
}


def validate_nsf_startup_failure(arm, evidence_root=Path(".")):
    """Reconcile one distinct recovery with preserved terminal failure evidence.

    The session ID is bound through the accepted-attempt mapping: remote runner
    manifests contain a run ID but do not contain their hosting session ID.
    Neither absence nor elapsed time is accepted as evidence of failure.
    """
    expected = NSF_STARTUP_PREVIOUS[arm["run_id"]]
    if arm.get("recovery_from") != expected:
        raise ValueError("NSF startup recovery provenance binding differs")
    evidence = evidence_root / expected["failed_manifest"]
    raw = evidence.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected["failed_manifest_sha256"]:
        raise ValueError("NSF terminal failure evidence SHA differs")
    failed = json.loads(raw)
    attempts = json.loads((evidence_root / expected["accepted_record"]).read_text())
    matches = [j for j in attempts["jobs"] if j.get("run_id") == expected["run_id"]]
    if len(matches) != 1 or any(
        matches[0].get(k) != v
        for k, v in {
            "session_id": expected["session_id"],
            "source_sha": expected["source_commit"],
            "submission_status": "accepted",
        }.items()
    ):
        raise ValueError("NSF failed session ID and accepted attempt differ")
    gradient = failed.get("encoder_gradient_contract", {})
    if (
        failed.get("status") != "failed"
        or failed.get("run_id") != expected["run_id"]
        or failed.get("source_commit") != expected["source_commit"]
        or failed.get("error_type") != "FloatingPointError"
        or failed.get("error") != expected["failure"]
        or failed.get("completed_updates") != 0
        or failed.get("training_performed") is not False
        or gradient.get("update") != 1
        or gradient.get("supervised_l2") != 0
        or gradient.get("reconstruction_unweighted_l2") != 0
    ):
        raise ValueError(
            "NSF recovery requires the exact terminal zero-update startup failure"
        )
    return failed


def validate_nsf_startup_config(cfg):
    """Preserve all scientific settings, and recover only the two failed arms."""
    original = yaml.safe_load(NSF_QMC256_CONFIG.read_text())
    if {k: v for k, v in cfg.items() if k != "arms"} != {
        k: v for k, v in original.items() if k != "arms"
    }:
        raise ValueError("NSF startup recovery scientific settings differ")
    arms = cfg.get("arms", [])
    if len(arms) != 2 or {a["run_id"] for a in arms} != set(NSF_STARTUP_PREVIOUS):
        raise ValueError(
            "NSF startup recovery may contain only the two terminal failures"
        )
    for arm in arms:
        expected = NSF_STARTUP_PREVIOUS[arm["run_id"]]
        old = next(a for a in original["arms"] if a["run_id"] == expected["run_id"])
        if {k: v for k, v in arm.items() if k not in {"run_id", "recovery_from"}} != {
            k: v for k, v in old.items() if k != "run_id"
        }:
            raise ValueError("NSF startup recovery arm settings differ")
        validate_nsf_startup_failure(arm)


def bind_remediation_policy(plan, env):
    task = plan["task"]
    config = CONFIGS[task]
    if task == "remediation-pilot" and plan["run_id"] in {
        "remediation_gaussian6_ft_gradientfix_v2",
        "remediation_mdn6_ft_gradientfix_v2",
        "remediation_gaussian6_ft_parallax_gradientfix_v2",
    }:
        config = Path("configs/remediation-pilots-gradient-recovery-20261026.yaml")
    if (
        task == "remediation-pilot"
        and plan["run_id"] == "remediation_nsf6_qmc_probe_v1"
    ):
        config = Path("configs/remediation-nsf-qmc-probe-20261026.yaml")
    if task == "remediation-pilot" and plan["run_id"] in {
        "remediation_nsf6_frozen_qmc256_v2",
        "remediation_nsf6_ft_qmc256_v2",
        "remediation_nsf6_ft_recon003_qmc256_v2",
        "remediation_nsf6_ft_parallax_qmc256_v2",
    }:
        config = Path("configs/remediation-nsf-qmc256-recovery-20261026.yaml")
    startup_recovery = (
        task == "remediation-pilot" and plan["run_id"] in NSF_STARTUP_PREVIOUS
    )
    if startup_recovery:
        config = NSF_STARTUP_CONFIG
    cfg = yaml.safe_load(config.read_text())
    if startup_recovery:
        validate_nsf_startup_config(cfg)
        if plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}:
            raise ValueError("NSF startup recovery resource policy differs")
        required = {
            "remediation_inputs_v4",
            "remediation_source_v3",
            "remediation_joint6_preflight_v3",
            "remediation_nsf6_qmc_probe_v1",
        }
        if set(plan.get("depends_on", [])) != required:
            raise ValueError("NSF startup recovery dependencies differ")
        expected_output = (
            "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
            + plan["run_id"]
        )
        if any(
            str(source["MSA_EXPERIMENT_OUTPUT"]) != expected_output
            for source in (env, plan.get("env", {}))
            if "MSA_EXPERIMENT_OUTPUT" in source
        ):
            raise ValueError("NSF startup recovery output must be distinct")
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
