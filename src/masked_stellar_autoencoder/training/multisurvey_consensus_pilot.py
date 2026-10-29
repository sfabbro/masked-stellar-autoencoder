"""Finite multi-survey consensus pilot; source-scale diagnostic only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
from masked_stellar_autoencoder.training.multisurvey_training import (
    ConsensusTrainer,
    FitIdentityGate,
)


def eligible_positions(dataset):
    """Uniform star sampling pool containing >=1 observed catalog quantity."""
    h5 = dataset._handle()
    pieces = []
    for name in dataset.names:
        stars, source_rows = dataset._indices[name]
        # h5py fancy-index requests must be sorted; read each bounded source mask
        # sequentially, then use the matching indices.
        mask = np.asarray(
            h5["native_observations"][name]["observed"][:], dtype=bool
        ).any(axis=1)
        pieces.append(stars[mask[source_rows]])
    rows = np.intersect1d(
        dataset.star_rows, np.unique(np.concatenate(pieces)), assume_unique=True
    )
    positions = np.searchsorted(dataset.star_rows, rows)
    if len(positions) < 2:
        raise ValueError("Insufficient stars with actual observed labels")
    return positions


def batch_for(dataset, positions):
    return collate_multisurvey(
        [dataset[int(p)] for p in positions], repeat_policy="lowest_source_row"
    )


def evaluate(model, relative, dataset, positions, scale, batch_size):
    """Save survey-mapped predictions and compute source/target MAD/MAE."""
    model.eval()
    relative.eval()
    result = {
        k: []
        for k in (
            "source_id",
            "source_index",
            "source_row",
            "truth",
            "prediction",
            "mask",
            "canonical",
            "scale_tril",
        )
    }
    device = next(model.mae.parameters()).device
    dtype = next(model.mae.parameters()).dtype
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            batch = batch_for(dataset, positions[start : start + batch_size])
            x = batch["input"].to(device=device, dtype=dtype)
            mean, lower = model(x)
            offsets, _ = relative(mean)
            sources = batch["source_index"].to(device=device)
            safe = sources.clamp(min=0, max=len(dataset.names) - 1)
            mapped = mean[:, None, :] + offsets.gather(
                1, safe[..., None].expand(-1, -1, 3)
            )
            mask = batch["observed"]
            result["source_id"].append(batch["source_id"].numpy())
            result["source_index"].append(batch["source_index"].numpy())
            result["source_row"].append(batch["source_row"].numpy())
            result["truth"].append(batch["value"].numpy())
            result["prediction"].append(mapped.cpu().numpy())
            result["mask"].append(mask.numpy())
            result["canonical"].append(mean.cpu().numpy())
            result["scale_tril"].append(lower.cpu().numpy())
    # Record axis varies across batches; flatten valid observations with stable IDs.
    # Canonical arrays remain aligned with star IDs.
    ids = np.concatenate(result["source_id"])
    canonical = np.concatenate(result["canonical"])
    lower = np.concatenate(result["scale_tril"])
    source = np.concatenate([x.reshape(-1) for x in result["source_index"]])
    row = np.concatenate([x.reshape(-1) for x in result["source_row"]])
    truth = np.concatenate([x.reshape(-1, 3) for x in result["truth"]])
    pred = np.concatenate([x.reshape(-1, 3) for x in result["prediction"]])
    mask = np.concatenate([x.reshape(-1, 3) for x in result["mask"]])
    rec_ids = np.concatenate(
        [
            np.repeat(i, s.shape[1])
            for i, s in zip(result["source_id"], result["source_index"], strict=True)
        ]
    )
    if not np.isfinite(pred[mask]).all() or not np.isfinite(truth[mask]).all():
        raise ValueError("Nonfinite heldout predictions or truth")
    scores = {}
    all_mae = []
    for j, label in enumerate(("teff", "logg", "fe_h")):
        use = mask[:, j]
        if not use.any():
            raise ValueError("Missing TUNE target: " + label)
        mae = float(np.mean(np.abs(pred[use, j] - truth[use, j])))
        all_mae.append(mae / scale[j])
        for source_index, name in enumerate(dataset.names):
            keep = use & (source == source_index)
            if keep.any():
                diff = pred[keep, j] - truth[keep, j]
                scores[name + "/" + label] = {
                    "n": int(keep.sum()),
                    "mae": float(np.abs(diff).mean()),
                    "bias": float(diff.mean()),
                }
    return {
        "macro_normalized_mae": float(np.mean(all_mae)),
        "per_source_target": scores,
    }, {
        "star_source_id": ids,
        "canonical_mean": canonical,
        "canonical_scale_tril": lower,
        "record_source_id": rec_ids,
        "record_source_index": source,
        "record_source_row": row,
        "record_value": truth,
        "record_prediction": pred,
        "record_observed": mask,
    }


def run_pilot(
    model,
    relative,
    fit,
    tune,
    devtest,
    source_floor,
    *,
    output,
    parent_sha256,
    normalization,
    normalization_sha256,
    library_revision,
    msa_revision,
    reconstruction_width,
    seed,
    encoder_lr,
    head_lr,
    updates=1000,
    batch_size=64,
    tune_every=100,
    panel_size=1024,
    reconstruction_weight=0.03,
):
    """Distinct finite pilot, uniform labeled-star FIT and fixed TUNE selection."""
    output = Path(output)
    if (
        updates < 1
        or batch_size < 1
        or tune_every < 1
        or panel_size < batch_size
        or reconstruction_weight <= 0
        or encoder_lr <= 0
        or head_lr <= 0
    ):
        raise ValueError("Invalid consensus pilot training contract")
    rng = np.random.default_rng(seed)
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    fit_pos, tune_pos, dev_pos = (eligible_positions(d) for d in (fit, tune, devtest))
    if any(len(x) < batch_size for x in (fit_pos, tune_pos, dev_pos)):
        raise ValueError("Insufficient eligible stars in a split")
    tune_rng = np.random.default_rng(20261030)
    dev_rng = np.random.default_rng(20261031)
    tune_pos = np.sort(
        tune_rng.choice(tune_pos, min(panel_size, len(tune_pos)), replace=False)
    )
    dev_pos = np.sort(
        dev_rng.choice(dev_pos, min(panel_size, len(dev_pos)), replace=False)
    )
    fit_ids_sha = hashlib.sha256(
        np.asarray(fit.ids[fit.star_rows], dtype="<i8").tobytes()
    ).hexdigest()
    gate = FitIdentityGate(
        fit,
        source_binding=fit.source_binding,
        fit_ids_sha256=fit_ids_sha,
        repeat_policy="lowest_source_row",
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for p in model.mae.parameters() if p.requires_grad],
                "lr": encoder_lr,
            },
            {
                "params": list(model.head.parameters()) + list(relative.parameters()),
                "lr": head_lr,
            },
        ]
    )
    trainer = ConsensusTrainer(
        model,
        relative,
        optimizer,
        gate,
        source_floor,
        parent_sha256=parent_sha256,
        normalization_sha256=normalization_sha256,
        library_revision=library_revision,
        msa_revision=msa_revision,
        assume_diagonal_catalog_covariance=True,
        reconstruction_weight=reconstruction_weight,
        reconstruction_width=reconstruction_width,
        calibration_regularization_weight=0.0001,
    )
    scale = np.asarray(normalization["scale"], dtype="f8")
    best = float("inf")
    best_path = None
    selection = []
    for step in range(1, updates + 1):
        batch = batch_for(fit, rng.choice(fit_pos, batch_size, replace=False))
        mask = (
            torch.rand(batch["input"].shape, generator=mask_generator) < 0.25
        ) & batch["input_available"]
        mask[:, reconstruction_width:] = False
        if not mask.any():
            raise ValueError("No available masked reconstruction feature")
        metrics = trainer.step(batch, reconstruction_mask=mask)
        if step % tune_every == 0 or step == updates:
            evaluation, arrays = evaluate(
                model, relative, tune, tune_pos, scale, batch_size
            )
            selection.append({"step": step, **metrics, **evaluation})
            if evaluation["macro_normalized_mae"] < best:
                best = evaluation["macro_normalized_mae"]
                previous = best_path
                best_path = output / f"selected-step-{step}.pth"
                state = {
                    "numpy_rng": rng.bit_generator.state,
                    "mask_rng": mask_generator.get_state().tolist(),
                    "seed": seed,
                    "updates": step,
                }
                trainer.save(best_path, sampler_state=state)
                np.savez_compressed(output / "tune-selected.npz", **arrays)
                if previous is not None:
                    previous.unlink()
            (output / "progress.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "best_step": int(best_path.stem.split("-")[-1]),
                        "best_metric": best,
                        "history": selection,
                    },
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
    # Reload the best weights from the just-written trusted checkpoint on CANFAR.
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    relative.load_state_dict(selected["relative_head"])
    metrics, arrays = evaluate(model, relative, devtest, dev_pos, scale, batch_size)
    np.savez_compressed(output / "devtest-selected.npz", **arrays)
    with best_path.open("rb") as checkpoint_file:
        checkpoint_sha256 = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
    artifacts = {}
    for file in (
        best_path,
        output / "tune-selected.npz",
        output / "devtest-selected.npz",
        output / "progress.json",
    ):
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        artifacts[file.name] = {"bytes": file.stat().st_size, "sha256": digest}
    return {
        "artifacts": artifacts,
        "completed_updates": trainer.steps,
        "selected_step": selected["steps"],
        "selected_checkpoint": str(best_path),
        "selected_checkpoint_sha256": checkpoint_sha256,
        "tune_selection": selection,
        "devtest": metrics,
        "fit_eligible_stars": len(fit_pos),
        "tune_panel_stars": len(tune_pos),
        "devtest_panel_stars": len(dev_pos),
        "normalization": normalization,
        "scope": "3-target relative survey consensus; devtest previously inspected; not physical truth; explicit diagonal errors and illustrative floors",
    }
