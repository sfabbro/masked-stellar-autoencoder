"""CPU-only staging of the large reconstruction sample; never launches training."""

import argparse
import hashlib
from pathlib import Path

import h5py
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.reconstruction_sample import (
    stage_reconstruction_sample,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("reconstruction_source") != "staged_sample":
        raise ValueError("Expected explicit staged_sample configuration")
    parent = Path(cfg["parent"])
    with parent.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != cfg["parent_sha256"]:
        raise ValueError("Parent checkpoint hash differs before data staging")
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    signature = checkpoint["run_signature"]
    original = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
    train_keys = signature["train_keys"]
    validation_keys = original["data"]["valid_keys"]
    with h5py.File(cfg["reconstruction_original"]) as source:
        if set(source) != set(train_keys) | set(validation_keys):
            raise ValueError("Original source shard set differs from parent/config")
        for key, expected in signature["train_rows_by_key"].items():
            if len(source[key]) != expected:
                raise ValueError(f"Original source row count differs: {key}")
    stage_reconstruction_sample(
        cfg["reconstruction_original"],
        train_keys,
        validation_keys,
        signature["feature_cols"],
        signature["scaler_center"],
        signature["scaler_scale"],
        cfg["reconstruction_cache"],
        cfg["reconstruction_rows_per_epoch"],
        seed=cfg["seed"],
        parent_signature=signature,
        parent_sha256=actual_sha,
    )


if __name__ == "__main__":
    main()
