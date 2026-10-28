"""The approved catalogue expansion adds source/inference jobs, never training."""

from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"
SOURCE_RUN = "full_label_evaluation_preparation_v2"
SOURCE_CONFIG = Path("configs/full-label-source-preparation-20261025-v2.yaml")
EVALUATION_CONFIG = Path("configs/expanded-checkpoint-evaluation-20261025-v2.yaml")
TRAINING_CONFIG = Path("configs/five-wave-training-20261024.yaml")
APPROVAL_CONFIG = Path("configs/full-label-evaluation-expansion-20261024.yaml")


def bind_expanded_evaluation_policy(plan, env):
    approval = yaml.safe_load(APPROVAL_CONFIG.read_text())
    if (
        approval["submission_policy"]["additional_training_jobs"] != 0
        or approval["submission_policy"]["additional_source_or_inference_jobs"]
        != "explicitly_authorized_as_needed_by_user"
    ):
        raise ValueError(
            "Evaluation-only expansion requires the explicit approved scope"
        )
    training = yaml.safe_load(TRAINING_CONFIG.read_text())
    task = plan["task"]
    if task == "expanded-source-contract-audit":
        if plan["run_id"] != "full_label_source_contract_audit_v1" or plan.get(
            "depends_on"
        ) != ["five_job_broad_test_preparation_v1"]:
            raise ValueError("Unknown metadata-only source contract audit")
        parent, gpu = training["checkpoint"], 0
    elif task == "full-label-evaluation-preparation":
        source = yaml.safe_load(SOURCE_CONFIG.read_text())
        if (
            plan["run_id"] != SOURCE_RUN
            or source["run_id"] != SOURCE_RUN
            or plan.get("depends_on") != ["five_job_broad_test_preparation_v1"]
        ):
            raise ValueError("Source assembly differs from the frozen-role preparation")
        parent, gpu = training["checkpoint"], 0
        env["MSA_EXPANDED_SOURCE_POLICY"] = str(SOURCE_CONFIG)
    elif task == "expanded-checkpoint-evaluation":
        suite = yaml.safe_load(EVALUATION_CONFIG.read_text())
        arm = next((a for a in suite["arms"] if a["run_id"] == plan["run_id"]), None)
        if arm is None or arm["producer_run_id"] not in {
            a["run_id"] for a in training["arms"]
        }:
            raise ValueError("Unknown immutable prediction producer")
        producer = arm["producer_run_id"]
        if (
            set(plan.get("depends_on", [])) != {SOURCE_RUN, producer}
            or len(plan["depends_on"]) != 2
        ):
            raise ValueError("Evaluation requires actual complete source and producer")
        parent, gpu = ROOT + "/" + producer + "/checkpoint.pth", 1
        env["MSA_EXPANDED_EVALUATION_POLICY"] = str(EVALUATION_CONFIG)
    else:
        raise ValueError("This binding is only for source preparation and inference")
    if plan.get("resources", {}).get("gpu", 0) != gpu:
        raise ValueError("Source must be CPU-only and inference must use one GPU")
    for key, value in {
        "MSA_PRETRAIN_RESUME": parent,
        "MSA_EXPERIMENT_PRESENTATIONS": "0",
    }.items():
        if key in plan.get("env", {}) and str(plan["env"][key]) != value:
            raise ValueError("Evaluation parent or zero-presentation override differs")
        env[key] = value
    return env
