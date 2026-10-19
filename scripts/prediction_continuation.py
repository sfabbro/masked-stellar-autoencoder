"""Explicit optimizer continuation and atomic full epoch checkpoints."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted


def load_segment(path, expected_sha, model, optimizer, *, features, center, scale):
    with open(path, "rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected_sha:
        raise ValueError("Continuation checkpoint digest differs")
    saved = torch_load_trusted(path, map_location="cpu", weights_only=False)
    if saved["arm"]["head"] != "residual" or saved["arm"]["loss"] != "huber":
        raise ValueError("Only the verified residual Huber family may continue")
    if saved["feature_columns"] != features:
        raise ValueError("Continuation feature order differs")
    np.testing.assert_array_equal(
        [saved["label_center"], saved["label_scale"]], [center, scale]
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    return saved


def save_epoch(path, model, head, optimizer, head_optimizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    state = dict(
        model_state_dict=model.state_dict(),
        head_state_dict=head.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        head_optimizer_state_dict=head_optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        **metadata,
    )
    torch.save(state, temporary)
    temporary.replace(path)
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    receipt = dict(
        file=path.name,
        sha256=digest,
        bytes=path.stat().st_size,
        presentations=metadata["presentations"],
        completed_updates=metadata["completed_updates"],
    )
    path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
