"""Bounded actual-data consensus preflight; no catalog or fit claim."""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .multisurvey_data import collate_multisurvey
from .multisurvey_model import multisurvey_objective
from .multisurvey_training import ConsensusTrainer, FitIdentityGate


def balanced_fit_batch(dataset, *, per_source=4):
    """Choose a small deterministic set of FIT stars with each source represented."""
    if per_source < 1:
        raise ValueError("Positive per-source count required")
    selected = set()
    for name in dataset.names:
        source_rows = dataset._indices[name][0]
        candidates = np.intersect1d(dataset.star_rows, source_rows, assume_unique=False)
        if not len(candidates):
            raise ValueError("A source has no FIT stars: " + name)
        chosen = candidates[
            np.linspace(
                0, len(candidates) - 1, min(per_source, len(candidates)), dtype=int
            )
        ]
        selected.update(int(x) for x in chosen)
    positions = np.searchsorted(
        dataset.star_rows, np.array(sorted(selected), dtype="i8")
    )
    stars = [dataset[int(i)] for i in positions]
    batch = collate_multisurvey(stars, repeat_policy="lowest_source_row")
    active = {
        record["source"]
        for observations in batch["provenance"]
        for record in observations
    }
    if active != set(dataset.names) or not batch["observed"].any():
        raise ValueError("Preflight batch does not cover each survey")
    return batch


def run_two_steps(
    model,
    relative_head,
    dataset,
    source_floor,
    *,
    parent_sha256,
    normalization_sha256,
    library_revision,
    msa_revision,
    reconstruction_width,
    encoder_lr=1.25e-6,
    head_lr=0.001,
):
    """Two actual optimizer updates, supervised encoder checks and masked MAE.

    These updates are a numerical preflight, not a fit or a scientific result.
    Label uncertainties, calibration dependencies and physical scale are not
    certified by this operation.
    """
    if (
        next(model.head.parameters()).dtype != torch.float64
        or next(relative_head.parameters()).dtype != torch.float64
    ):
        raise ValueError("Float64 head and calibration are required for this preflight")
    fit_ids_sha256 = hashlib.sha256(
        np.asarray(dataset.ids[dataset.star_rows], dtype="<i8").tobytes()
    ).hexdigest()
    gate = FitIdentityGate(
        dataset,
        source_binding=dataset.source_binding,
        fit_ids_sha256=fit_ids_sha256,
        repeat_policy="lowest_source_row",
    )
    optimizer = torch.optim.Adam(
        [
            {
                "params": [p for p in model.mae.parameters() if p.requires_grad],
                "lr": encoder_lr,
            },
            {
                "params": list(model.head.parameters())
                + list(relative_head.parameters()),
                "lr": head_lr,
            },
        ]
    )
    trainer = ConsensusTrainer(
        model,
        relative_head,
        optimizer,
        gate,
        source_floor,
        parent_sha256=parent_sha256,
        normalization_sha256=normalization_sha256,
        library_revision=library_revision,
        msa_revision=msa_revision,
        assume_diagonal_catalog_covariance=True,
        reconstruction_weight=0.03,
        reconstruction_width=reconstruction_width,
        calibration_regularization_weight=0.0001,
    )
    batch = balanced_fit_batch(dataset)
    if reconstruction_width < 1 or reconstruction_width > batch["input"].shape[1]:
        raise ValueError("Invalid reconstruction width")
    generator = torch.Generator(device="cpu").manual_seed(20261028)
    mask = (torch.rand(batch["input"].shape, generator=generator) < 0.25) & batch[
        "input_available"
    ]
    mask[:, reconstruction_width:] = False
    if not mask.any():
        raise ValueError("No available features for reconstruction")
    device, dtype = (
        next(model.mae.parameters()).device,
        next(model.mae.parameters()).dtype,
    )
    parameters = [p for p in model.mae.encoder.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("Trainable encoder required for gradient preflight")
    report = []
    for step in range(2):
        verified = gate.verify_batch(batch)
        verified["input"] = verified["input"].to(device=device, dtype=dtype)
        supervised = multisurvey_objective(
            model,
            verified,
            relative_head,
            trainer.source_floor,
            expected_registry_sha256=dataset.registry_sha256,
            expected_source_registry=dataset.names,
            expected_training_source_binding=dataset.source_binding,
            assume_diagonal_catalog_covariance=True,
        )["supervised"]
        grads = torch.autograd.grad(supervised, parameters, allow_unused=True)
        finite = [g for g in grads if g is not None]
        if not finite or not all(torch.isfinite(g).all() for g in finite):
            raise FloatingPointError(
                "Supervised encoder gradients are missing or nonfinite"
            )
        norm = sum(float(g.detach().double().square().sum()) for g in finite) ** 0.5
        if not np.isfinite(norm) or norm <= 0:
            raise FloatingPointError("Supervised encoder gradient is zero")
        metrics = trainer.step(batch, reconstruction_mask=mask)
        encoder_ids = {id(p) for p in model.mae.encoder.parameters()}
        decoder_grads = [
            p.grad
            for p in model.mae.parameters()
            if id(p) not in encoder_ids and p.grad is not None
        ]
        if not decoder_grads or not all(torch.isfinite(g).all() for g in decoder_grads):
            raise FloatingPointError(
                "Reconstruction decoder gradients missing or nonfinite"
            )
        decoder_norm = (
            sum(float(g.detach().double().square().sum()) for g in decoder_grads) ** 0.5
        )
        if not np.isfinite(decoder_norm) or decoder_norm <= 0:
            raise FloatingPointError("Reconstruction decoder gradient is zero")
        report.append(
            {
                "step": step + 1,
                "supervised_encoder_gradient_l2": norm,
                "reconstruction_decoder_gradient_l2": decoder_norm,
                **metrics,
            }
        )
    return {
        "updates": trainer.steps,
        "batch_stars": len(batch["source_id"]),
        "source_registry": list(dataset.names),
        "fit_ids_sha256": fit_ids_sha256,
        "batch_ids_sha256": hashlib.sha256(
            batch["source_id"].numpy().astype("<i8").tobytes()
        ).hexdigest(),
        "repeat_resolution_count": len(batch["repeat_resolution"]),
        "supervised_xp_masking": False,
        "reconstruction_masked_elements": int(mask.sum()),
        "likelihood_precision": "float64"
        if next(model.head.parameters()).dtype == torch.float64
        else str(next(model.head.parameters()).dtype),
        "steps": report,
        "scope": "Two-update FIT-only numerical preflight; not an epoch, calibrated posterior, or physical-label certificate",
    }
