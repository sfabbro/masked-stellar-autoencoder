"""Bounded matched frozen-encoder ridge test of processed Pristine photometry."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from astropy.io import fits
from pristine_quality_views import RAW_SHA, exact_rows
from run_stellar_label_experiments import BENCH, LEGACY, PARENT, PARENT_SHA
from stellar_catalog_jobs import atomic_json

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def ridge_fit_predict(x, y, fit, test, alpha=10.0):
    """Train-only standardization; unpenalized intercept and fixed ridge budget."""
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite ridge inputs/labels")
    if len(fit) < 20 or len(test) < 20:
        raise ValueError("Insufficient fixed-cohort labels")
    center = x[fit].mean(0)
    scale = x[fit].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.column_stack([np.ones(len(x)), (x - center) / scale])
    yc = float(y[fit].mean())
    ys = max(float(y[fit].std()), 1e-8)
    a = design[fit]
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(a.T @ a + penalty, a.T @ ((y[fit] - yc) / ys))
    prediction = (design[test] @ coef) * ys + yc
    return prediction, dict(
        center=center, scale=scale, coef=coef, ycenter=yc, yscale=ys
    )


def replay(x, state):
    design = np.column_stack([np.ones(len(x)), (x - state["center"]) / state["scale"]])
    return (design @ state["coef"]) * state["yscale"] + state["ycenter"]


def adjuncts(phot):
    """Only photometry: no inferred Pristine FeH or truth-dependent selection."""
    return np.column_stack(
        [
            phot["Bpmag0"] - phot["Rpmag0"],
            phot["Gmag0"] - phot["Rpmag0"],
            phot["CaHKmag0"] - phot["Gmag0"],
        ]
    ).astype("f8")


def compare_heads(latent, extra, y, split, membership, out):
    results = {}
    arrays = {}
    initial = {}
    test = np.flatnonzero(split == 2)
    for cohort, fitmask in {
        "strict_benchmark_train": (split == 0) & (membership == 0),
        "broad_excluding_benchmark_validation": (split == 0) & (membership != 1),
    }.items():
        fit = np.flatnonzero(fitmask)
        if np.any(membership[fit] == 1):
            raise ValueError("Validation fitting forbidden")
        for variant, nextra in [
            ("encoder", 0),
            ("encoder_gaia", 2),
            ("encoder_gaia_cahk", 3),
        ]:
            x = np.column_stack([latent, extra[:, :nextra]])
            prediction, state = ridge_fit_predict(x, y, fit, test)
            key = cohort + "__" + variant
            np.savez(out / (key + "-head.npz"), **state)
            with np.load(out / (key + "-head.npz")) as saved:
                np.testing.assert_array_equal(prediction, replay(x[test], saved))
            arrays[key] = prediction
            initial[key] = np.full(len(test), state["ycenter"])
            delta = prediction - y[test]
            poor = y[test] < -2
            results[key] = dict(
                fit_rows=len(fit),
                test_rows=len(test),
                mae_dex=float(np.mean(abs(delta))),
                bias_dex=float(delta.mean()),
                metalpoor_n=int(poor.sum()),
                metalpoor_mae_dex=float(np.mean(abs(delta[poor])))
                if poor.any()
                else None,
            )
        arrays[cohort + "__fit_row"] = fit
    return test, arrays, initial, results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy="Frozen epoch57 encoder; six fixed-alpha10 ridge fits (two fit cohorts, three nested feature sets). Train-only standardization; identity test split2. No SGD/encoder updates.",
        caveat="Processed giant-solution dereddened magnitudes; not isolated raw CaHK information. Native spectroscopic labels; unsupervised pretraining overlap not excluded. No sky holdout added by this pilot.",
    )
    atomic_json(args.output / "manifest.json", manifest)
    try:
        if probe.digest(PARENT) != PARENT_SHA:
            raise ValueError("Parent checksum mismatch")
        raw = ROOT / "pristine_full_fetch_v1_r2/pristine-dr1-full.fits"
        view = ROOT / "pristine_quality_view_v1/pristine-quality-view.h5"
        registry = ROOT / "label_registry_v2/labels.h5"
        if probe.digest(raw) != RAW_SHA:
            raise ValueError("Pristine checksum mismatch")
        if (
            probe.digest(view)
            != "b511e1c82910c14c9e50d78c4e8a16c1fb770765935bd2557a960668dec862f9"
        ):
            raise ValueError("QC view checksum mismatch")
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        features = config["data"]["feature_cols"]
        x, _, ids, _, duplicates = probe.read_fits(
            LEGACY, features, ["fe_h"], "g_normalized"
        )
        unit_audit = probe.audit_fits_xp_units(BENCH, features, x, ids, "g_normalized")
        with h5py.File(registry) as f:
            labels = f["selected/labels"][:]
        with h5py.File(view) as f:
            native = f["native"]
            np.testing.assert_array_equal(native["source_id"][:], labels["source_id"])
            eligible = native["input_eligible"][:]
            split = native["label_split"][:]
            membership = native["benchmark_partition"][:]
            rawrow = native["pristine_row"][:]
        feature_row = exact_rows(ids, labels["source_id"])
        take = np.flatnonzero(
            eligible
            & (feature_row >= 0)
            & np.isfinite(labels["value"])
            & np.isfinite(labels["uncertainty"])
            & (labels["uncertainty"] > 0)
        )
        with fits.open(raw, memmap=True) as f:
            extra = adjuncts(f[1].data[rawrow[take]])
            np.testing.assert_array_equal(
                f[1].data["GaiaDR3"][rawrow[take]], labels["source_id"][take]
            )
        finite = np.isfinite(extra).all(1)
        take = take[finite]
        extra = extra[finite]
        checkpoint = torch.load(PARENT, map_location="cpu", weights_only=False)
        center = np.asarray(checkpoint["run_signature"]["scaler_center"])
        scale = np.asarray(checkpoint["run_signature"]["scaler_scale"])
        with h5py.File(BENCH) as f:
            np.testing.assert_array_equal(center, f["reference_scaler/center"][:])
            np.testing.assert_array_equal(scale, f["reference_scaler/scale"][:])
            assert (
                json.loads(f.attrs["provenance"])["source_id_sha256"]
                == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
            )
        inputs = probe.safe_inputs(
            (x[feature_row[take]] - center) / scale,
            features,
            ["fe_h"],
            ["RA", "DEC", "pmra", "pmdec"],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = probe.load_model(config, checkpoint, device)
        for p in model.parameters():
            p.requires_grad_(False)
        latent = probe.encode(model, inputs, device)
        fixture = inputs[:32]
        fixture_before = probe.encode(model, fixture, device)
        reloaded = probe.load_model(config, checkpoint, device)
        np.testing.assert_array_equal(
            fixture_before, probe.encode(reloaded, fixture, device)
        )
        del reloaded
        y = labels["value"][take].astype("f8")
        split = split[take]
        membership = membership[take]
        np.savez_compressed(
            args.output / "paired-cohort.npz",
            source_id=labels["source_id"][take],
            truth=y,
            uncertainty=labels["uncertainty"][take],
            survey=labels["survey"][take],
            split=split,
            benchmark_partition=membership,
            native_row=take,
            pristine_row=rawrow[take],
            latent=latent,
            extra=extra,
        )
        test, arrays, initial, results = compare_heads(
            latent, extra, y, split, membership, args.output
        )
        np.savez_compressed(
            args.output / "predictions.npz",
            source_id=labels["source_id"][take][test],
            truth=y[test],
            **arrays,
        )
        np.savez_compressed(
            args.output / "initial_predictions.npz",
            source_id=labels["source_id"][take][test],
            truth=y[test],
            **initial,
        )
        if probe.digest(PARENT) != PARENT_SHA:
            raise ValueError("Parent changed")
        atomic_json(
            args.output / "checkpoint_verification.json",
            dict(
                all_six_head_reloads_exact=True,
                frozen_encoder_reload_exact=True,
                parent_unchanged=True,
                optimizer_steps=0,
            ),
        )
        manifest.update(
            status="complete",
            parent_sha256=PARENT_SHA,
            unit_audit=unit_audit,
            common_cohort=len(take),
            registry_sha256=probe.digest(registry),
            legacy_sha256=probe.digest(LEGACY),
            pristine_sha256=RAW_SHA,
            duplicates_in_legacy=duplicates,
            results=results,
        )
        atomic_json(args.output / "comparison.json", results)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
