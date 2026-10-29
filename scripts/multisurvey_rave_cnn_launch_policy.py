"""Fixed source-only RAVE CNN plus Gaia EDR3 launch contract."""

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_rave_cnn_edr3_source_v1"
TASK = "multisurvey-rave-cnn-edr3-source"


def bind_rave_cnn_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != TASK:
        raise ValueError("Unknown RAVE CNN source job")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("RAVE source resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("RAVE source requires public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("RAVE source rejects launch overrides")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("RAVE source output must be distinct")
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
