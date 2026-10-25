"""CPU-only reporting reads completed saved scores, never a model or optimizer."""

from pathlib import Path

import yaml

CONFIG = Path("configs/expanded-report-recovery-20261025.yaml")
APPROVAL = Path("configs/full-label-evaluation-expansion-20261024.yaml")
TRAINING = Path("configs/five-wave-training-20261024.yaml")
HEADS = ("quantile3", "gaussian_noise", "nsf_noise", "nsf_noise_mixed01")
PRODUCERS = {
    f"expanded_report_{head}_s51_v1": f"expanded_eval_{head}_s51_v2" for head in HEADS
}


def bind_report_policy(plan, env):
    scope = yaml.safe_load(APPROVAL.read_text())["submission_policy"]
    if (
        scope["additional_training_jobs"] != 0
        or scope["additional_source_or_inference_jobs"]
        != "explicitly_authorized_as_needed_by_user"
    ):
        raise ValueError(
            "Saved-array report recovery requires authorized evaluation scope"
        )
    cfg = yaml.safe_load(CONFIG.read_text())
    arm = next((a for a in cfg["arms"] if a["run_id"] == plan["run_id"]), None)
    if arm is None or PRODUCERS.get(plan["run_id"]) != arm["producer_run_id"]:
        raise ValueError("Unknown immutable saved-score producer")
    if (
        plan["task"] != "expanded-report-recovery"
        or plan.get("resources", {}).get("gpu") != 0
        or plan.get("depends_on", [])
    ):
        raise ValueError(
            "Reporting is CPU-only; its runtime checks saved-score completion"
        )
    expected = dict(
        MSA_EXPERIMENT_PRESENTATIONS="0",
        MSA_PRETRAIN_RESUME=yaml.safe_load(TRAINING.read_text())["checkpoint"],
        MSA_EXPANDED_REPORT_POLICY=str(CONFIG),
    )
    for key, value in expected.items():
        if key in plan.get("env", {}) and str(plan["env"][key]) != value:
            raise ValueError("Saved-array report policy override differs")
        env[key] = value
    return env
