"""Bounded paired age adaptation with all seismic reference IDs held out."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from pristine_quality_views import exact_rows
from run_stellar_label_experiments import BENCH, LEGACY, PARENT, PARENT_SHA
from stellar_age_cohorts import ROOT, cohort_masks, file_digest
from stellar_dataset_views import EXPECTED_IDS

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

CONTENT_ID = "5882eca484ce494596b0f2c33da1cdda35140b60c732ceee5c932a5ee46eac09"
SCALER_SHA = "5fd976729dfe5ccd7b4ec2ce080e679c49f1ca5244f779cad45e148fa4836037"


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_control(
    model, head, x, y, xp_indices, mode, *, steps=2500, warmup=500, batch=256
):
    """Only training inputs/targets enter adaptation; separate fixed row/mask streams."""
    if mode not in ["frozen", "sft", "recon"] or not 0 <= warmup < steps:
        raise ValueError("Invalid bounded adaptation settings")
    model.eval()
    head.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode != "frozen":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": head.parameters(), "lr": 1e-3},
            {"params": model.encoder.parameters(), "lr": 1e-6},
        ],
        weight_decay=1e-4,
    )
    rng = np.random.default_rng(42)
    mask_rng = torch.Generator(device=x.device).manual_seed(4201)
    rows_digest, masks_digest = hashlib.sha256(), hashlib.sha256()
    xp_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    xp_mask[xp_indices] = True
    started = time.monotonic()
    metrics = []
    for step in range(steps):
        index = rng.integers(len(x), size=min(batch, len(x)))
        rows_digest.update(index.astype("<i8").tobytes())
        values = x[index]
        hidden = (
            torch.rand(values.shape, device=x.device, generator=mask_rng) < 0.20
        ) & xp_mask
        hidden &= torch.isfinite(values) & (values != -9999)
        masks_digest.update(hidden.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        if step < warmup or mode == "frozen":
            with torch.no_grad():
                latent = model.encoder(values)
        else:
            latent = model.encoder(values)
        predicted = head(latent).squeeze(1)
        objective = torch.nn.functional.huber_loss(predicted, y[index], delta=1.0)
        if mode == "recon" and step >= warmup:
            decoded = probe.decode(
                model, model.encoder(values.masked_fill(hidden, -9999))
            )
            scored = hidden[:, : decoded.shape[1]]
            if not scored.any():
                raise ValueError("No masked observed XP reconstruction targets")
            objective = (
                objective
                + 0.01
                * (decoded - values[:, : decoded.shape[1]])
                .abs()
                .masked_select(scored)
                .mean()
            )
        if not torch.isfinite(objective):
            raise FloatingPointError("Nonfinite native-age objective")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(
            [*model.encoder.parameters(), *head.parameters()], 1.0
        )
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            metrics.append(
                dict(
                    step=step + 1,
                    loss=float(objective.detach()),
                    elapsed_seconds=time.monotonic() - started,
                )
            )
    model.eval()
    head.eval()
    return dict(
        head_updates=steps,
        encoder_updates=0 if mode == "frozen" else steps - warmup,
        warmup_steps=warmup,
        actual_row_index_sha256=rows_digest.hexdigest(),
        actual_xp_mask_sha256=masks_digest.hexdigest(),
        batch=min(batch, len(x)),
        metrics=metrics,
        training_seconds=time.monotonic() - started,
    )


def predict(encoder, head, x, center, scale):
    with torch.no_grad():
        return np.concatenate(
            [
                torch.exp(head(encoder(block)).squeeze(1) * scale + center)
                .cpu()
                .numpy()
                for block in x.split(512)
            ]
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    modes = {
        "native_age_strict_frozen_v1": "frozen",
        "native_age_strict_sft_v1": "sft",
        "native_age_strict_recon_v1": "recon",
    }
    mode = modes[args.run_id]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        mode=mode,
        fit_cohort="strict_benchmark_train",
        seed=42,
        caveat="StarHorse model/parallax-conditioned posteriors; seismic ages share physics/spectroscopic dependencies. No astroNN target. Reference labels never train/tune. Pretraining overlap not excluded; no model-free age or independent geometric-distance claim.",
    )
    probe.write_json(args.output / "manifest.json", manifest)
    try:
        cohort_path = ROOT / "age_feature_cohort_v1/age-feature-cohort.h5"
        with h5py.File(cohort_path) as f:
            if not f.attrs["complete"] or f.attrs["content_id"] != CONTENT_ID:
                raise ValueError("Unverified age feature cohort")
            content = hashlib.sha256()

            def digest_dataset(name, node):
                if isinstance(node, h5py.Dataset):
                    content.update(name.encode())
                    content.update(node[:].tobytes())

            f.visititems(digest_dataset)
            if content.hexdigest() != CONTENT_ID:
                raise ValueError("Age cohort dataset content changed")
            cohort = {key: f["selected/" + key][:] for key in f["selected"]}
            reference_ids = f["external_reference/source_id"][:]
            reference_rows = f["external_reference/feature_row"][:]
            if file_digest(LEGACY) != f.attrs["legacy_feature_sha256"]:
                raise ValueError("Feature table changed since cohort materialization")
        with h5py.File(BENCH) as f:
            provenance = json.loads(f.attrs["provenance"])
            if provenance["source_id_sha256"] != EXPECTED_IDS:
                raise ValueError("Benchmark provenance changed")
        with h5py.File(ROOT / "dataset_view_v2/dataset-view.h5") as f:
            train_ids = f["train/source_id"][:]
            val_ids = f["validation/source_id"][:]
        actual_hash = hashlib.sha256(
            train_ids.astype("<i8").tobytes() + val_ids.astype("<i8").tobytes()
        ).hexdigest()
        if actual_hash != EXPECTED_IDS:
            raise ValueError("Actual benchmark ordered IDs changed")
        masks = cohort_masks(
            cohort["source_id"],
            np.ones(len(cohort["source_id"]), dtype=bool),
            train_ids,
            val_ids,
            reference_ids,
        )
        for key, value in masks.items():
            np.testing.assert_array_equal(cohort[key], value)
        train = np.flatnonzero(cohort["strict_fit"])
        test = np.flatnonzero(cohort["identity_test"])
        if len(train) != 4440 or len(reference_ids) != 66:
            raise ValueError("Measured strict/reference coverage changed")
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        features = config["data"]["feature_cols"]
        x, _, ids, _, duplicates = probe.read_fits(
            LEGACY, features, ["fe_h"], "g_normalized"
        )
        unit_audit = probe.audit_fits_xp_units(BENCH, features, x, ids, "g_normalized")
        if file_digest(PARENT) != PARENT_SHA:
            raise ValueError("Parent checksum changed")
        checkpoint = torch_load_trusted(PARENT, map_location="cpu", weights_only=False)
        signature = checkpoint["run_signature"]
        center = np.asarray(signature["scaler_center"])
        scale = np.asarray(signature["scaler_scale"])
        if hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest() != SCALER_SHA:
            raise ValueError("Frozen scaler changed")
        reference_available = reference_rows >= 0
        reference_ids = reference_ids[reference_available]
        combined_ids = np.r_[cohort["source_id"], reference_ids]
        matched = exact_rows(ids, combined_ids)
        if (matched < 0).any():
            raise ValueError("Feature availability changed")
        blocked = ["RA", "DEC", "pmra", "pmdec", "PARALLAX", "distance", "age", "e_age"]
        safe = probe.safe_inputs(
            (x[matched] - center) / scale, features, ["age"], blocked
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        values = torch.as_tensor(safe, dtype=torch.float32, device=device)
        y = cohort["age_percentiles"][:, 2]
        label_center = float(np.log(y[train]).mean())
        label_scale = max(float(np.log(y[train]).std()), 1e-4)
        target = torch.as_tensor(
            (np.log(y[train]) - label_center) / label_scale,
            dtype=torch.float32,
            device=device,
        )
        torch.manual_seed(42)
        model = probe.load_model(config, checkpoint, device)
        model.eval()
        head = torch.nn.Sequential(
            torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
        ).to(device)
        head.eval()
        initial_encoder_sha = state_digest(model.encoder)
        initial_decoder_sha = state_digest(model.decoder) + state_digest(
            model.reconstruction_layer
        )
        fixture = values[test[:128]]
        with torch.no_grad():
            latent = model.encoder(fixture)
            initial = head(latent).cpu().numpy()
        initial_state = {
            name: value.detach().cpu().numpy()
            for name, value in head.state_dict().items()
        }
        np.savez_compressed(
            args.output / "initial_predictions.npz",
            source_id=cohort["source_id"][test[:128]],
            latent=latent.cpu().numpy(),
            head_output=initial,
            **initial_state,
        )
        np.savez_compressed(
            args.output / "paired-cohort.npz",
            **cohort,
            train_rows=train,
            test_rows=test,
            reference_source_id=reference_ids,
            reference_feature_row=reference_rows[reference_available],
            label_log_center=label_center,
            label_log_scale=label_scale,
        )
        xp = [i for i, name in enumerate(features) if name.startswith(("bp_", "rp_"))]
        result = fit_control(model, head, values[train], target, xp, mode)
        predictions = predict(
            model.encoder, head, values[test], label_center, label_scale
        )
        external = predict(
            model.encoder, head, values[len(y) :], label_center, label_scale
        )
        if not np.isfinite(predictions).all() or not np.isfinite(external).all():
            raise FloatingPointError("Nonfinite age prediction")
        np.savez_compressed(
            args.output / "predictions.npz",
            test_source_id=cohort["source_id"][test],
            test_truth=y[test],
            test_age_percentiles=cohort["age_percentiles"][test],
            test_survey=cohort["survey"][test],
            test_prediction=predictions,
            reference_source_id=reference_ids,
            reference_prediction=external,
        )
        torch.save(
            dict(
                encoder=model.encoder.state_dict(),
                head=head.state_dict(),
                label_log_center=label_center,
                label_log_scale=label_scale,
            ),
            args.output / "checkpoint.pth",
        )
        restored = torch_load_trusted(
            args.output / "checkpoint.pth", map_location=device, weights_only=False
        )
        encoder = copy.deepcopy(model.encoder)
        encoder.load_state_dict(restored["encoder"])
        other = copy.deepcopy(head)
        other.load_state_dict(restored["head"])
        np.testing.assert_array_equal(
            predict(encoder, other, values[test], label_center, label_scale),
            predictions,
        )
        np.testing.assert_array_equal(
            predict(encoder, other, values[len(y) :], label_center, label_scale),
            external,
        )
        if (
            state_digest(model.decoder) + state_digest(model.reconstruction_layer)
            != initial_decoder_sha
        ):
            raise ValueError("Frozen decoder changed")
        if mode == "frozen" and state_digest(model.encoder) != initial_encoder_sha:
            raise ValueError("Frozen encoder changed")
        probe.write_json(
            args.output / "checkpoint_verification.json",
            dict(
                full_test_and_reference_reload_exact=True,
                parent_unchanged=file_digest(PARENT) == PARENT_SHA,
                decoder_unchanged=True,
                initial_encoder_sha256=initial_encoder_sha,
                final_encoder_sha256=state_digest(model.encoder),
            ),
        )
        with (args.output / "metrics.jsonl").open("w") as f:
            for metric in result.pop("metrics"):
                f.write(json.dumps(metric) + "\n")
        manifest.update(
            status="complete",
            results=probe.score(predictions, y[test]),
            external_predictions_count=len(external),
            external_metrics_pending="Offline audit against every correlated seismic pipeline; no test-based selection",
            unit_audit=unit_audit,
            feature_file_sha256=file_digest(LEGACY),
            cohort_file_sha256=file_digest(cohort_path),
            cohort_content_id=CONTENT_ID,
            parent_sha256=PARENT_SHA,
            scaler_sha256=SCALER_SHA,
            benchmark_source_id_sha256=actual_hash,
            label_uncertainty_policy="All native asymmetric percentiles preserved; Huber on standardized log-age, not calibrated NLL",
            feature_columns=features,
            blocked_features=blocked,
            duplicates_removed=duplicates,
            split_counts=dict(train=len(train), test=len(test)),
            label_log_center=label_center,
            label_log_scale=label_scale,
            **result,
        )
        probe.write_json(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        probe.write_json(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
