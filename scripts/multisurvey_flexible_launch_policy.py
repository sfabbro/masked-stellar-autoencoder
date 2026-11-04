"""Bound the overnight non-Gaussian experiment and its project-GPU runtime."""

from pathlib import Path

import yaml

CONFIG = Path("configs/multisurvey-flexible-20261105.yaml")
CAL_CONFIG = Path("configs/multisurvey-flexible-calibration-20261105.yaml")
ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def bind_flexible_rescore_policy(plan, env):
    path = Path("configs/multisurvey-flexible-rescore-20261105.yaml")
    config = yaml.safe_load(path.read_text())
    suite = yaml.safe_load(
        Path("configs/multisurvey-flexible-refinement-20261105.yaml").read_text()
    )
    if (
        config["suite"] != suite
        or config["run_id"] != "multisurvey_flexible_rescore_20261105_v1"
        or plan["run_id"] != config["run_id"]
        or plan["task"] != "multisurvey-flexible-rescore"
        or config["training_source_commit"]
        != "6f1b3c790b921878e857e79b3d09b4baf3cdebbb"
        or config["fit_cache_sha256"] != suite["frozen_cache"]["sha256"]
        or config["producer_manifest"] != ROOT + suite["run_id"] + "/manifest.json"
        or len(config["producer_manifest_sha256"]) != 64
        or config["particles"] != 4096
        or plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + config["run_id"]
    ):
        raise ValueError("Defensive evaluation-only rescoring contract differs")
    env.update(
        MSA_XP_EXPERIMENTS="0",
        MSA_MULTISURVEY_POLICY=str(path),
        MSA_PRETRAIN_RESUME=suite["parent"]["checkpoint"],
        MSA_EXPERIMENT_PRESENTATIONS="0",
    )
    return env


def bind_flexible_refinement_policy(plan, env):
    path = Path("configs/multisurvey-flexible-refinement-20261105.yaml")
    config = yaml.safe_load(path.read_text())
    original = yaml.safe_load(CONFIG.read_text())
    expected = dict(
        original,
        run_id="multisurvey_flexible_refinement_20261105_v2",
        particles=1024,
        gate_particles=4096,
        evaluation_particles=4096,
        save_all_snapshots=True,
        frozen_cache=dict(
            path=ROOT + original["run_id"] + "/frozen-contexts.h5",
            sha256="99f9e8697c549474867e9f889974e9a5170284a5efb75fa503a4590abf54c5ed",
            producer_manifest=ROOT + original["run_id"] + "/manifest.json",
            source_commit="451cf0dc036b8cacde5fb7fddcd9d86db5817b00",
        ),
    )
    if (
        config != expected
        or plan["run_id"] != config["run_id"]
        or plan["task"] != "multisurvey-flexible-refinement"
        or plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + config["run_id"]
    ):
        raise ValueError("Higher-particle frozen-cache refinement contract differs")
    env["MSA_XP_EXPERIMENTS"] = "0"
    env["MSA_MULTISURVEY_POLICY"] = str(path)
    env["MSA_PRETRAIN_RESUME"] = original["parent"]["checkpoint"]
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env


def bind_flexible_policy(plan, env):
    config = yaml.safe_load(CONFIG.read_text())
    if (
        plan["run_id"] != config["run_id"]
        or plan["task"] != "multisurvey-flexible-suite"
        or config["updates"] != 1000
        or config["fit_stars"] != 65536
        or config["tune_stars"] != 2048
        or config["batch_size"] != 64
        or config["micro_batch"] != 8
        or config["particles"] != 256
        or config["evaluation_particles"] != 1024
        or config["density_reference_sha256"]
        != "efd575285789928d2e445d7c11429a7809a35a12e6be29424bd26c48e154c26d"
        or config["parent"]["library_revision"]
        != "2777b0de88121768434725ef227893cd35595d2a"
    ):
        raise ValueError("Flexible density source or numerical contract differs")
    if (
        plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + config["run_id"]
    ):
        raise ValueError("Flexible density resource or output contract differs")
    # This task needs no GaiaXPy flux projection; the tested project GPU lock
    # includes the NF and multi-source library. Preserve the old XP lock.
    env["MSA_XP_EXPERIMENTS"] = "0"
    env["MSA_MULTISURVEY_POLICY"] = str(CONFIG)
    env["MSA_PRETRAIN_RESUME"] = config["parent"]["checkpoint"]
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env


def bind_flexible_calibration_policy(plan, env):
    config = yaml.safe_load(CAL_CONFIG.read_text())
    original = yaml.safe_load(
        Path("configs/multisurvey-flexible-refinement-20261105.yaml").read_text()
    )
    if (
        config["suite"] != original
        or plan["run_id"] != config["run_id"]
        or plan["task"] != "multisurvey-flexible-calibration"
        or config["cal_stars"] != 2048
        or config["evaluation_particles"] != 4096
        or config["training_source_commit"]
        != "6f1b3c790b921878e857e79b3d09b4baf3cdebbb"
        or config["selection_manifest"]
        != ROOT + "multisurvey_flexible_rescore_20261105_v1/manifest.json"
        or config["integration_proposal"] != "defensive_mixture"
        or config["fit_cache_sha256"] != original["frozen_cache"]["sha256"]
        or not {"gaussian"}
        < set(config["heads"])
        <= {"gaussian", "mixture", "copula", "nsf"}
    ):
        raise ValueError("Fixed-head CAL evaluation contract differs")
    for kind, head in config["heads"].items():
        selected_run = (
            original["run_id"]
            if kind in {"gaussian", "mixture"}
            else "multisurvey_flexible_rescore_20261105_v1"
        )
        if head["path"] != ROOT + selected_run + "/" + kind + "-selected.pth":
            raise ValueError("CAL must evaluate original TUNE-selected checkpoints")
    for digest in [
        config["fit_cache_sha256"],
        config["selection_manifest_sha256"],
        *(h["sha256"] for h in config["heads"].values()),
    ]:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("CAL requires exact checkpoint and cache hashes")
    if (
        plan.get("resources") != {"cores": 4, "ram": 64, "gpu": 1}
        or plan.get("image") != "images.canfar.net/astroai/terminal:latest"
        or plan.get("env")
        or plan.get("depends_on", [])
        or env.get("MSA_EXPERIMENT_OUTPUT") != ROOT + config["run_id"]
    ):
        raise ValueError("CAL resource or output contract differs")
    env["MSA_XP_EXPERIMENTS"] = "0"
    env["MSA_MULTISURVEY_POLICY"] = str(CAL_CONFIG)
    env["MSA_PRETRAIN_RESUME"] = original["parent"]["checkpoint"]
    env["MSA_EXPERIMENT_PRESENTATIONS"] = "0"
    return env
