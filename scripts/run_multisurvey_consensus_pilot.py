"""Run a finite, CANFAR CUDA-only three-target multi-survey consensus pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import yaml


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def native_pilot(config, overlay_manifest, source_commit, output):
    if (
        not Path("/arc/projects/k-pop").is_dir()
        or os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
    ):
        raise RuntimeError("Native preflight requires headless CANFAR CUDA")
    import h5py
    import numpy as np
    import run_xp_representation_probes as native
    import torch
    from torchregress.calibration import RelativeSurveySplineHead

    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
    from masked_stellar_autoencoder.training.multisurvey_consensus_pilot import (
        run_pilot,
    )
    from masked_stellar_autoencoder.training.multisurvey_model import (
        CanonicalAtmosphereModel,
        CanonicalGaussianHead,
    )
    from masked_stellar_autoencoder.training.multisurvey_overlay import (
        sparse_dataset_from_manifest,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Native preflight requires CUDA")
    direct = json.loads(
        importlib.metadata.distribution("torchregress").read_text("direct_url.json")
        or "{}"
    )
    if direct.get("vcs_info", {}).get("commit_id") != config["library_revision"]:
        raise ValueError("Installed torchregress revision differs")
    for key in ("checkpoint", "reference_checkpoint"):
        if sha(config[key]) != config[key + "_sha256"]:
            raise ValueError("Parent checkpoint SHA differs")
    reference = torch_load_trusted(
        config["reference_checkpoint"], map_location="cpu", weights_only=False
    )
    weights = torch_load_trusted(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    if (
        weights.get("presentations") != 4500000000
        or weights.get("completed_updates") != 137340
        or weights.get("parent_sha256") != config["reference_checkpoint_sha256"]
    ):
        raise ValueError("Parent checkpoint lineage differs")
    disk = dict(reference, model_state_dict=weights["model_state_dict"])
    signature = reference["run_signature"]
    model_cfg = yaml.safe_load(Path(config["model_config"]).read_text())
    feature_cols, reconstruction_cols = (
        model_cfg["data"]["feature_cols"],
        model_cfg["data"]["recon_cols"],
    )
    if feature_cols[: len(reconstruction_cols)] != reconstruction_cols:
        raise ValueError("Reconstruction columns must be an input prefix")
    with h5py.File(config["base_path"], "r") as source:
        if (
            source["feature_names"][:].astype("U").tolist() != feature_cols
            or feature_cols != signature["feature_cols"]
        ):
            raise ValueError("Parent input feature order differs")
        for key in ("scaler_center", "scaler_scale"):
            np.testing.assert_array_equal(source[key][:], signature[key])
        fit = (source["role"][:].astype("U") == "fit") & source[
            "primary_input_eligible"
        ][:].astype(bool)
        values = source["value"][:, :3]
        observed = source["observed"][:, :3].astype(bool) & source["quality"][
            :, :3
        ].astype(bool)
        center, scale = [], []
        for j in range(3):
            v = values[fit & observed[:, j] & np.isfinite(values[:, j]), j]
            if len(v) < 100 or not np.isfinite(v).all() or v.std() <= 0:
                raise ValueError(
                    "Insufficient old FIT labels to define preflight scaling"
                )
            center.append(float(v.mean()))
            scale.append(float(v.std()))
    normalization = {
        "center": center,
        "scale": scale,
        "scope": "old FIT labels only; numerical preflight scaling, not physical calibration",
    }
    norm_sha = hashlib.sha256(
        json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda")
    mae = native.load_model(model_cfg, disk, device)
    head = CanonicalGaussianHead(
        model_cfg["model"]["layer_dims"][-1],
        torch.tensor(center, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
        width=128,
    ).to(device=device, dtype=torch.float64)
    model = CanonicalAtmosphereModel(mae, head)
    if config["encoder"] == "frozen":
        model.mae.encoder.requires_grad_(False)
    elif config["encoder"] != "finetune":
        raise ValueError("Unknown encoder training mode")
    knots = torch.tensor(
        [-3.0, -3.0, -3.0, -3.0, -1.0, 0.0, 1.0, 3.0, 3.0, 3.0, 3.0],
        dtype=torch.float64,
        device=device,
    )
    relative = RelativeSurveySplineHead(
        [knots] * 3,
        torch.ones((3, 3), dtype=torch.float64, device=device),
        input_center=torch.tensor(center, dtype=torch.float64, device=device),
        input_scale=torch.tensor(scale, dtype=torch.float64, device=device),
        output_scale=torch.tensor(scale, dtype=torch.float64, device=device),
    ).to(device=device, dtype=torch.float64)
    datasets = [
        sparse_dataset_from_manifest(
            overlay_manifest,
            expected_manifest_sha256=config["overlay_manifest_sha256"],
            role=role,
        )
        for role in ("fit", "tune", "devtest")
    ]
    try:
        floors = torch.tensor(
            [[100.0, 0.1, 0.1]] * len(datasets[0].names),
            dtype=torch.float64,
            device=device,
        )
        return run_pilot(
            model,
            relative,
            *datasets,
            floors,
            output=output,
            parent_sha256=config["checkpoint_sha256"],
            normalization=normalization,
            normalization_sha256=norm_sha,
            library_revision=config["library_revision"],
            msa_revision=source_commit,
            reconstruction_width=len(reconstruction_cols),
            seed=config["seed"],
            encoder_lr=config["encoder_lr"],
            head_lr=config["head_lr"],
            updates=config["updates"],
            batch_size=config["batch_size"],
            tune_every=config["tune_every"],
            panel_size=config["panel_size"],
            reconstruction_weight=config["reconstruction_weight"],
        )
    finally:
        for dataset in datasets:
            dataset.close()


def run(config, output, *, source_commit, execute=native_pilot):
    expected = {
        "run_id",
        "overlay_producer_manifest",
        "overlay_manifest",
        "overlay_manifest_sha256",
        "base_path",
        "checkpoint",
        "checkpoint_sha256",
        "reference_checkpoint",
        "reference_checkpoint_sha256",
        "model_config",
        "library_revision",
        "seed",
        "encoder",
        "encoder_lr",
        "head_lr",
        "updates",
        "batch_size",
        "tune_every",
        "panel_size",
        "reconstruction_weight",
    }
    if (
        set(config) != expected
        or len(source_commit) != 40
        or any(c not in "0123456789abcdef" for c in source_commit)
    ):
        raise ValueError("Preflight configuration or source commit differs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": config["run_id"],
        "status": "admitting",
        "source_commit": source_commit,
        "training_scope": "three-target multi-survey consensus pilot; not physical truth",
        "training_performed": False,
        "model_weights_saved": None,
    }
    atomic(output / "manifest.json", manifest)
    try:
        producer = json.loads(Path(config["overlay_producer_manifest"]).read_text())
        result = producer.get("result", {})
        if (
            producer.get("run_id") != "multisurvey_desi_unique_overlay_v1"
            or producer.get("status") != "multisurvey_desi_overlay_complete"
            or producer.get("source_commit")
            != "cf40e37c87d4e724b84fd1dffcac6d9e1363560d"
            or result.get("overlay_manifest_path") != config["overlay_manifest"]
            or result.get("overlay_manifest_sha256")
            != config["overlay_manifest_sha256"]
        ):
            raise ValueError("Completed overlay producer binding differs")
        manifest.update(
            status="running", training_performed=None, training_attempted=True
        )
        atomic(output / "manifest.json", manifest)
        evidence = execute(config, config["overlay_manifest"], source_commit, output)
        if evidence.get("completed_updates") != config["updates"] or not evidence.get(
            "selected_checkpoint_sha256"
        ):
            raise ValueError("Preflight numerical evidence differs")
        manifest.update(
            status="multisurvey_consensus_pilot_complete",
            model_weights_saved=True,
            training_performed=True,
            result=evidence,
        )
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic(output / "manifest.json", manifest)
        raise
    atomic(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    run(yaml.safe_load(args.config.read_text()), args.output, source_commit=commit)


if __name__ == "__main__":
    main()
