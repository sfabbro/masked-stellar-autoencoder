"""CANFAR CPU metadata export; no model construction, forward pass or fitting."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def extract_normalization(checkpoint):
    center = np.asarray(checkpoint["label_center"], dtype="f8")
    scale = np.asarray(checkpoint["label_scale"], dtype="f8")
    if (
        center.shape != (6,)
        or scale.shape != (6,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("Invalid six-target fitted normalization")
    if list(checkpoint["target_names"]) != [
        "teff",
        "logg",
        "fe_h",
        "alpha",
        "age",
        "distance",
    ]:
        raise ValueError("Target order differs")
    return dict(
        center=center.tolist(),
        scale=scale.tolist(),
        coordinates=["K", "dex", "dex", "dex", "ln(Gyr)", "ln(kpc)"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1"
        or not Path("/arc/projects/k-pop").is_dir()
    ):
        raise RuntimeError("Metadata export runs on CANFAR CPU headless only")
    cfg = yaml.safe_load(
        Path("configs/metalpoor-corner-sampling-20261024.yaml").read_text()
    )
    result = {}
    for model in cfg["models"]:
        if (
            sha(model["checkpoint"]) != model["checkpoint_sha256"]
            or sha(model["independent_report"]) != model["independent_report_sha256"]
        ):
            raise ValueError("Checkpoint/independent proof changed")
        proof = json.loads(Path(model["independent_report"]).read_text())
        if (
            proof["status"] != "complete_strict_pass"
            or proof["bindings"]["checkpoint"] != model["checkpoint_sha256"]
        ):
            raise ValueError("Independent proof does not bind this state")
        disk = torch_load_trusted(
            model["checkpoint"], map_location="cpu", weights_only=False
        )
        result[model["run_id"]] = dict(
            **extract_normalization(disk), checkpoint_sha256=model["checkpoint_sha256"]
        )
        del disk
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "normalization.json"
    if path.exists():
        raise ValueError("Metadata already exported; never overwrite")
    payload = dict(
        status="metadata_only_export_complete",
        models=result,
        model_instantiated=False,
        model_forward=False,
        training=False,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="complete",
                normalization_sha256=sha(path),
                **{k: v for k, v in payload.items() if k != "status"},
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
