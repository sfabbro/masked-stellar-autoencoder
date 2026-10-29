ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_lamost_dr12_network_probe_v1"
TASK = "multisurvey-lamost-dr12-network-probe"


def bind_lamost_probe_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != TASK:
        raise ValueError("Unknown network probe")
    if (
        plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
    ):
        raise ValueError("Incorrect resource or image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Overrides rejected")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("Distinct output required")
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
