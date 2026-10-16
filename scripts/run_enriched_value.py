"""One fixed frozen-encoder comparison: strict native versus source-aware addition.

Both heads share native-fit standardization and alpha10, with identical zero
initial coefficients. Added PIGS/Li rows change the training population. Each
canonical star contributes once; evaluation retains separate raw source layers.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from run_native_age import SCALER_SHA, state_digest
from run_pristine_value import replay
from run_stellar_label_experiments import BENCH, PARENT, PARENT_SHA
from stellar_age_cohorts import file_digest
from stellar_catalog_jobs import atomic_json, identity_split
from stellar_dataset_views import EXPECTED_IDS

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
COHORT_SHA = "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"
CONTENT = "947e79c4aa27913b9b2693095ed806c9494c0fad85bcfd6b203a2e9edde53e94"
LAYERS = ("native", "PIGS", "Li2025_CaT_silver")
BLOCKED = ["RA", "DEC", "pmra", "pmdec", "PARALLAX", "distance", "age", "e_age"]


def prepare(path):
    """Resolve canonical fits and source-separated test layers without leakage."""
    with h5py.File(path) as f:
        c = {n: f["canonical/" + n][:] for n in f["canonical"]}
        ids = f["features/source_id"][:]
        partition = f["features/benchmark_partition"][:]
        split = identity_split(ids)
        np.testing.assert_array_equal(f["features/label_split"][:], split)
        feature_rows = c["feature_row"]
        np.testing.assert_array_equal(ids[feature_rows], c["source_id"])
        strict = (split[feature_rows] == 0) & (partition[feature_rows] == 0)
        np.testing.assert_array_equal(c["strict_fit"], strict)
        measurements = {n: f["raw_layers/" + n + "/measurements"][:] for n in LAYERS}
        canonical_y = np.empty(len(feature_rows), dtype="f8")
        expected_layer = np.full(len(ids), -1, dtype="i1")
        available = f["features/feature_available"][:]
        tests = {}
        for code, n in enumerate(LAYERS):
            m = measurements[n]
            rows = f["layer_index/" + n + "/feature_row"][:]
            np.testing.assert_array_equal(ids[rows], m["source_id"])
            first = available[rows] & (expected_layer[rows] < 0)
            expected_layer[rows[first]] = code
            test = available[rows] & (split[rows] == 2)
            np.testing.assert_array_equal(
                f["layer_index/" + n + "/identity_test"][:], test
            )
            tests[n] = dict(feature_row=rows[test], measurements=m[test])
            take = c["layer"] == code
            chosen = m[c["layer_row"][take]]
            np.testing.assert_array_equal(chosen["source_id"], c["source_id"][take])
            canonical_y[take] = chosen["value"]
        np.testing.assert_array_equal(c["layer"], expected_layer[feature_rows])
        fits = np.flatnonzero(strict)
        used = np.unique(
            np.r_[feature_rows[fits], *[v["feature_row"] for v in tests.values()]]
        )
        augmented = np.searchsorted(used, feature_rows[fits])
        native = augmented[c["layer"][fits] == 0]
        y = np.full(len(used), np.nan, dtype="f8")
        y[augmented] = canonical_y[fits]
        for n, t in tests.items():
            t["used_row"] = np.searchsorted(used, t["feature_row"])
            assert not np.isin(
                t["measurements"]["source_id"], ids[used[augmented]]
            ).any()
        assert (partition[used[augmented]] == 0).all()
        assert (split[used[augmented]] == 0).all()
        assert np.isfinite(y[augmented]).all()
        return dict(
            source_id=ids[used],
            feature_row=used,
            raw_values=f["features/values"][:][used],
            native_fit=native,
            augmented_fit=augmented,
            fit_truth=y,
            tests=tests,
            feature_columns=json.loads(f.attrs["feature_columns_json"]),
            fit_source_layer=c["layer"][fits],
            center=f["reference_scaler/center"][:],
            scale=f["reference_scaler/scale"][:],
        )


def fit_heads(latent, y, native, augmented):
    """Common native-fit-only scaling, fixed alpha; no evaluation labels enter."""
    latent = np.asarray(latent, dtype="f8")
    native, augmented = np.asarray(native), np.asarray(augmented)
    if not np.isfinite(latent).all() or not np.isfinite(y[augmented]).all():
        raise ValueError("Nonfinite fit inputs")
    if (
        len(native) < 20
        or len(np.unique(augmented)) != len(augmented)
        or not np.isin(native, augmented).all()
    ):
        raise ValueError("Invalid nested fit cohorts")
    center, scale = latent[native].mean(0), latent[native].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    yc, ys = float(y[native].mean()), max(float(y[native].std()), 1e-8)
    design = np.column_stack([np.ones(len(latent)), (latent - center) / scale])
    penalty = np.eye(design.shape[1]) * 10.0
    penalty[0, 0] = 0
    states = {}
    for name, rows in [("native_only", native), ("canonical_augmented", augmented)]:
        a = design[rows]
        coef = np.linalg.solve(a.T @ a + penalty, a.T @ ((y[rows] - yc) / ys))
        states[name] = dict(
            center=center.copy(), scale=scale.copy(), coef=coef, ycenter=yc, yscale=ys
        )
    return states


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy="Frozen epoch57, fixed alpha10 ridge; same native-fit-only scaling and zero initial heads. Native19859 versus canonical20267fits with219PIGS+189Li additions. No SGD. Source-separated tests; added rows change population objective, not matched presentation budgets.",
        caveat="Native FeH/PIGS/Li CaT silver retain distinct calibration and uncertainty tiers; no shifts/averaging/calibrated NLL or unseen-pretraining/SOTA claim.",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        cohort = ROOT / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
        gate = json.loads(
            (ROOT / "enriched_feature_audit_v1/manifest.json").read_text()
        )
        if (
            gate.get("status") != "complete"
            or not gate["result"].get("publication_gate_complete")
            or gate["result"]["cohort_sha256"] != COHORT_SHA
            or gate["result"]["content_id"] != CONTENT
        ):
            raise ValueError("Independent original-source publication gate incomplete")
        if file_digest(cohort) != COHORT_SHA or file_digest(PARENT) != PARENT_SHA:
            raise ValueError("Immutable cohort/parent changed")
        data = prepare(cohort)
        if len(data["native_fit"]) != 19859 or len(data["augmented_fit"]) != 20267:
            raise ValueError("Declared actual fit counts changed")
        features = data["feature_columns"]
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        if features != config["data"]["feature_cols"]:
            raise ValueError("Feature order changed")
        with h5py.File(cohort) as f:
            # Original source audit is complete; still recheck real 128 matched XP units.
            units = probe.audit_fits_xp_units(
                BENCH,
                features,
                f["features/values"][:],
                f["features/source_id"][:],
                "g_normalized",
            )
        checkpoint = torch_load_trusted(PARENT, map_location="cpu", weights_only=False)
        sig = checkpoint["run_signature"]
        center = np.asarray(sig["scaler_center"])
        scale = np.asarray(sig["scaler_scale"])
        if hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest() != SCALER_SHA:
            raise ValueError("Parent scaler changed")
        np.testing.assert_array_equal(center, data["center"])
        np.testing.assert_array_equal(scale, data["scale"])
        safe = probe.safe_inputs(
            (data["raw_values"] - center) / scale, features, ["fe_h"], BLOCKED
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        model = probe.load_model(config, checkpoint, device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        initial_encoder = state_digest(model.encoder)
        latent = probe.encode(model, safe, device)
        states = fit_heads(
            latent, data["fit_truth"], data["native_fit"], data["augmented_fit"]
        )
        paired = {
            k: data[k]
            for k in [
                "source_id",
                "feature_row",
                "raw_values",
                "fit_truth",
                "native_fit",
                "augmented_fit",
                "fit_source_layer",
            ]
        }
        paired.update(safe_inputs=safe, latent=latent)
        for n, t in data["tests"].items():
            paired[n + "__measurements"] = t["measurements"]
            paired[n + "__used_row"] = t["used_row"]
        np.savez_compressed(args.output / "paired-cohort.npz", **paired)
        predictions = {}
        initial = {}
        results = {}
        for name, state in states.items():
            np.savez(args.output / (name + "-head.npz"), **state)
            prediction = replay(latent, state)
            with np.load(args.output / (name + "-head.npz")) as saved:
                np.testing.assert_array_equal(prediction, replay(latent, saved))
            predictions[name] = prediction
            initial[name] = np.full(len(latent), state["ycenter"])
            initial[name + "__coef"] = np.zeros_like(state["coef"])
            results[name] = {}
            for n, t in data["tests"].items():
                target = t["measurements"]["value"]
                pred = prediction[t["used_row"]]
                bins = {
                    "all": np.ones(len(target), bool),
                    "below_minus3": target < -3,
                    "minus3_to_minus2": (target >= -3) & (target < -2),
                    "minus2_to_minus1": (target >= -2) & (target < -1),
                    "at_least_minus1": target >= -1,
                }
                results[name][n] = {
                    key: probe.score(pred[mask], target[mask])
                    for key, mask in bins.items()
                }
                if "survey" in t["measurements"].dtype.names:
                    surveys = t["measurements"]["survey"]
                    results[name][n]["per_survey"] = {
                        str(survey): probe.score(
                            pred[surveys == survey], target[surveys == survey]
                        )
                        for survey in np.unique(surveys)
                    }
        np.testing.assert_array_equal(
            initial["native_only"], initial["canonical_augmented"]
        )
        np.savez_compressed(
            args.output / "initial_predictions.npz",
            source_id=data["source_id"],
            **initial,
        )
        np.savez_compressed(
            args.output / "predictions.npz", source_id=data["source_id"], **predictions
        )
        torch.save(
            dict(encoder=model.encoder.state_dict()), args.output / "encoder.pth"
        )
        restored = probe.load_model(config, checkpoint, device)
        restored.encoder.load_state_dict(
            torch_load_trusted(
                args.output / "encoder.pth", map_location=device, weights_only=False
            )["encoder"],
            strict=True,
        )
        replay_latent = probe.encode(restored, safe, device)
        np.testing.assert_array_equal(replay_latent, latent)
        for name, state in states.items():
            np.testing.assert_array_equal(
                replay(replay_latent, state), predictions[name]
            )
        assert (
            state_digest(model.encoder) == initial_encoder
            and file_digest(PARENT) == PARENT_SHA
        )
        atomic_json(
            args.output / "checkpoint_verification.json",
            dict(
                full_encoder_reload_all_rows_exact=True,
                both_heads_reload_all_rows_exact=True,
                encoder_unchanged=True,
                parent_unchanged=True,
                optimizer_steps=0,
            ),
        )
        atomic_json(args.output / "comparison.json", results)
        manifest.update(
            benchmark_ordered_source_id_sha256=EXPECTED_IDS,
            peak_cpu_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1 if sys.platform == "darwin" else 1024),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
            if device.type == "cuda"
            else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved()
            if device.type == "cuda"
            else 0,
            status="complete",
            results=results,
            unit_audit=units,
            source_gate=gate["result"],
            cohort_sha256=COHORT_SHA,
            cohort_content_id=CONTENT,
            parent_sha256=PARENT_SHA,
            scaler_sha256=SCALER_SHA,
            encoder_state_sha256=initial_encoder,
            encoded_rows=len(safe),
            native_fit_count=len(data["native_fit"]),
            augmented_fit_count=len(data["augmented_fit"]),
            runtime_seconds=time.monotonic() - started,
            device=str(device),
            blocked_features=BLOCKED,
            feature_columns=features,
            alpha=10.0,
            training_updates=0,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        atomic_json(args.output / "manifest.json", manifest)
        raise
    atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
