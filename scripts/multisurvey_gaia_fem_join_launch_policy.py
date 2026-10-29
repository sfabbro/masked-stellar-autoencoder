"""Fixed source-only Gaia Fe I/M exact-ID join launch contract."""

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUN = "multisurvey_gaia_gspspec_fem_join_v1"
TASK = "multisurvey-gaia-gspspec-fem-join"


def bind_gaia_fem_join_policy(plan, env):
    if plan.get("run_id") != RUN or plan.get("task") != TASK:
        raise ValueError("Unknown Gaia Fe/M source join")
    if plan.get("resources") != {"cores": 8, "ram": 128, "gpu": 0}:
        raise ValueError("Gaia Fe/M source resource contract differs")
    if plan.get("image") != "images.canfar.net/astroai/terminal:latest":
        raise ValueError("Gaia Fe/M source join requires public terminal image")
    if plan.get("depends_on", []) != [] or plan.get("env"):
        raise ValueError("Gaia Fe/M source join rejects overrides")
    if env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + RUN:
        raise ValueError("Gaia Fe/M output must be distinct")
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
