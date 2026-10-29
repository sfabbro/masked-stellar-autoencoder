"""Fixed distinct LAMOST DR12 source-only jobs."""

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
TASK = "multisurvey-lamost-dr12-source"
RUNS = {
    "multisurvey_lamost_dr12_lrs_source_v1",
    "multisurvey_lamost_dr12_mrs_source_v1",
    "multisurvey_lamost_dr12_lrs_source_direct_v2",
    "multisurvey_lamost_dr12_mrs_source_direct_v2",
}


def bind_lamost_dr12_policy(plan, env):
    if plan.get("task") != TASK or plan.get("run_id") not in RUNS:
        raise ValueError("Unknown DR12 job")
    if (
        plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
    ):
        raise ValueError("Resource/image contract differs")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Launch overrides prohibited")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + plan["run_id"]:
        raise ValueError("Distinct output required")
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
