"""One bounded frozen-encoder parallax-free/assisted distance comparison."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from astropy.io import fits
from pristine_quality_views import exact_rows
from run_pristine_value import replay, ridge_fit_predict
from run_stellar_label_experiments import BENCH, LEGACY, PARENT, PARENT_SHA
from stellar_age_cohorts import ROOT, cohort_masks, file_digest
from stellar_catalog_jobs import atomic_json
from stellar_dataset_views import EXPECTED_IDS

COHORT_SHA = "fe377385e9a5a26a1d3f14ca15ea6c56904d73c34233fb9e17da1b944b76b1ad"
CONTENT_ID = "4af0dbbe9b11fe212538d9e4cbac9842bed6f82af06e9becfeb6339e4897bbce"
SCALER_SHA = "5fd976729dfe5ccd7b4ec2ce080e679c49f1ca5244f779cad45e148fa4836037"
BLOCKED = ["RA", "DEC", "pmra", "pmdec", "distance", "age", "e_age"]


def positive_distance(log_prediction):
    prediction = np.exp(log_prediction)
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise FloatingPointError("Nonfinite or nonpositive predicted distance")
    return prediction


def score_parallax(distance_kpc, measured_parallax_mas, formal_error_mas):
    """No inversion/sign/SNR cut on measurements; formal-error diagnostic only."""
    distance = np.asarray(distance_kpc)
    observed, error = np.asarray(measured_parallax_mas), np.asarray(formal_error_mas)
    if not (np.isfinite(distance).all() and (distance > 0).all()):
        raise ValueError("Invalid predicted kpc distances")
    if not (
        np.isfinite(observed).all() and np.isfinite(error).all() and (error > 0).all()
    ):
        raise ValueError("Invalid measured mas parallaxes/errors")
    return parallax_metrics(1 / distance, observed, error)


def parallax_metrics(predicted_mas, observed, error):
    predicted_mas, observed, error = map(np.asarray, (predicted_mas, observed, error))
    if not (
        np.isfinite(predicted_mas).all()
        and np.isfinite(observed).all()
        and np.isfinite(error).all()
        and (error > 0).all()
    ):
        raise ValueError("Invalid parallax baseline or measurements")
    residual = predicted_mas - observed
    return dict(
        count=len(observed),
        mae_mas=float(np.mean(np.abs(residual))),
        rmse_mas=float(np.sqrt(np.mean(residual**2))),
        bias_mas=float(residual.mean()),
        mean_formal_gaussian_nll=float(
            np.mean(
                0.5 * (residual / error) ** 2 + np.log(error) + 0.5 * np.log(2 * np.pi)
            )
        ),
        formal_sigma_is_total_calibrated_accuracy=False,
        reference_nonpositive_parallaxes_retained=int((observed <= 0).sum()),
    )


def fit_distance_head(latent, distance, train, evaluate):
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError("Native posterior distances must be positive finite kpc")
    log_prediction, state = ridge_fit_predict(
        latent, np.log(distance), train, evaluate, alpha=10.0
    )
    return positive_distance(log_prediction), state


def benchmark_ids(benchmark, index):
    """Use audited ID index; verify the actual nested compound-record benchmark.

    This is the same provenance/index path used by the completed native-age controls.
    Reading ten million projected fields from the compressed 121GB benchmark is
    unnecessary; the independently checked index retains the original ordered IDs.
    """
    with h5py.File(benchmark) as f:
        if json.loads(f.attrs["provenance"])["source_id_sha256"] != EXPECTED_IDS:
            raise ValueError("Immutable benchmark provenance changed")
        counts = []
        for split in ["train", "validation"]:
            records = f[split + "/records"]
            if (
                not isinstance(records, h5py.Dataset)
                or not records.dtype.names
                or "source_id" not in records.dtype.names
            ):
                raise ValueError("Expected original nested compound benchmark records")
            counts.append(len(records))
    with h5py.File(index) as f:
        ids = tuple(f[split + "/source_id"][:] for split in ["train", "validation"])
    if [len(v) for v in ids] != counts:
        raise ValueError("Benchmark index count differs from original records")
    return ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        run_id=args.run_id,
        policy="Two frozen epoch57 encoder input modes; fixed-alpha10 ridge heads on strict4452 native StarHorse log-distance labels. Zero SGD/encoder updates. Same fit/test/reference IDs, train-only head scaling and zero initial standardized heads.",
        caveat="StarHorse model/Gaia-conditioned targets; all99574 Hipparcos/seismic reference IDs excluded fit and tune. Measured Hipparcos parallaxes retained in mas including negatives. Formal errors not total calibrated accuracy; no beyond-Gaia or pretraining independence claim.",
    )
    atomic_json(args.output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        cohort_path = ROOT / "distance_feature_cohort_v1/distance-feature-cohort.h5"
        if file_digest(cohort_path) != COHORT_SHA or file_digest(PARENT) != PARENT_SHA:
            raise ValueError("Verified cohort or parent changed")
        with h5py.File(cohort_path) as f:
            if not f.attrs["complete"] or f.attrs["content_id"] != CONTENT_ID:
                raise ValueError("Distance publication gate not satisfied")
            digest = hashlib.sha256()

            def update(name, node):
                if isinstance(node, h5py.Dataset):
                    digest.update(name.encode())
                    digest.update(node[:].tobytes())

            f.visititems(update)
            if digest.hexdigest() != CONTENT_ID:
                raise ValueError("Cohort content changed")
            c = {key: f["selected/" + key][:] for key in f["selected"]}
            reference = {
                key: f["hipparcos_reference/" + key][:]
                for key in f["hipparcos_reference"]
            }
            holdout = f["ALL_reference_holdout_source_ids"][:]
            if file_digest(LEGACY) != f.attrs["feature_input_sha256"]:
                raise ValueError("Feature table changed")
        bt, bv = benchmark_ids(BENCH, ROOT / "dataset_view_v2/dataset-view.h5")
        bh = hashlib.sha256(
            bt.astype("<i8").tobytes() + bv.astype("<i8").tobytes()
        ).hexdigest()
        if (
            bh != EXPECTED_IDS
            or len(bt) != 9000000
            or len(bv) != 1000000
            or len(np.unique(np.r_[bt, bv])) != 10000000
        ):
            raise ValueError("Immutable benchmark changed")
        expected_masks = cohort_masks(
            c["source_id"], np.ones(len(c["source_id"]), bool), bt, bv, holdout
        )
        for key, value in expected_masks.items():
            np.testing.assert_array_equal(c[key], value)
        train, test = (
            np.flatnonzero(c["strict_fit"]),
            np.flatnonzero(c["identity_test"]),
        )
        ref = np.flatnonzero(reference["evaluation_candidate"])
        if len(train) != 4452 or len(ref) != 4166 or len(holdout) != 99574:
            raise ValueError("Declared paired cohorts changed")
        refids = reference["source_id"][ref]
        combined = np.r_[c["source_id"][train], c["source_id"][test], refids]
        if len(np.unique(combined)) != len(combined):
            raise ValueError("Training/test/reference stars overlap")
        config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
        features = config["data"]["feature_cols"]
        x, _, ids, _, duplicates = probe.read_fits(
            LEGACY, features, ["fe_h"], "g_normalized"
        )
        unit_audit = probe.audit_fits_xp_units(BENCH, features, x, ids, "g_normalized")
        parent = torch.load(PARENT, map_location="cpu", weights_only=False)
        center = np.asarray(parent["run_signature"]["scaler_center"])
        scale = np.asarray(parent["run_signature"]["scaler_scale"])
        if hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest() != SCALER_SHA:
            raise ValueError("Parent scaler changed")
        matched = exact_rows(ids, combined)
        if (matched < 0).any():
            raise ValueError("Declared feature availability changed")
        original_rows = np.r_[
            c["feature_row"][train],
            c["feature_row"][test],
            reference["feature_row"][ref],
        ]
        with fits.open(LEGACY, memmap=True) as f:
            np.testing.assert_array_equal(
                f[1].data["source_id"][original_rows], combined
            )
            raw_parallax = np.asarray(f[1].data["PARALLAX"][original_rows], dtype="f8")
        baseline_gate = json.loads(
            Path("configs/distance-gaia-baseline-verification.json").read_text()
        )
        if (
            hashlib.sha256(refids.astype("<i8").tobytes()).hexdigest()
            != baseline_gate["reference_id_sha256"]
            or hashlib.sha256(
                raw_parallax[len(train) + len(test) :].astype("<f8").tobytes()
            ).hexdigest()
            != baseline_gate["raw_feature_parallax_sha256"]
        ):
            raise ValueError("Independently verified Gaia baseline changed")
        base = probe.safe_inputs(
            (x[matched] - center) / scale, features, ["distance"], BLOCKED
        )
        ntrain, ntest = len(train), len(test)
        fit = np.arange(ntrain)
        evaluate = np.arange(ntrain, len(combined))
        # Reference measurements NEVER become fit targets; their placeholder is unused.
        target = np.r_[
            c["distance_percentiles"][train, 2],
            c["distance_percentiles"][test, 2],
            np.ones(len(ref)),
        ]
        raw = {key: reference[key][ref] for key in reference}
        paired = dict(
            **c,
            train_rows=train,
            test_rows=test,
            reference_source_id=refids,
            combined_source_id=combined,
            input_assisted=base,
            raw_input_parallax=raw_parallax,
            **{
                "reference_" + key: value
                for key, value in raw.items()
                if key != "source_id"
            },
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = probe.load_model(config, parent, device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        baseline_metrics = parallax_metrics(
            raw_parallax[ntrain + ntest :],
            raw["hipparcos_parallax_mas"],
            raw["hipparcos_error_mas"],
        )
        predictions = {}
        initial = {}
        results = {}
        for mode in ["parallax_free", "parallax_assisted"]:
            inputs = base.copy()
            if mode == "parallax_free":
                inputs = probe.safe_inputs(inputs, features, ["distance"], ["PARALLAX"])
            latent = probe.encode(model, inputs, device)
            if not np.isfinite(latent).all():
                raise FloatingPointError("Nonfinite encoder output")
            prediction, state = fit_distance_head(latent, target, fit, evaluate)
            np.savez(args.output / (mode + "-head.npz"), **state)
            with np.load(args.output / (mode + "-head.npz")) as head:
                np.testing.assert_array_equal(
                    prediction, positive_distance(replay(latent[evaluate], head))
                )
            # Full inference replay from the immutable checkpoint in each input mode.
            restored = probe.load_model(config, parent, device)
            restored.eval()
            again = probe.encode(restored, inputs, device)
            del restored
            np.testing.assert_array_equal(latent, again)
            np.testing.assert_array_equal(
                prediction, positive_distance(replay(again[evaluate], state))
            )
            paired["latent_" + mode] = latent
            predictions[mode + "_native_kpc"] = prediction[:ntest]
            predictions[mode + "_reference_kpc"] = prediction[ntest:]
            predictions[mode + "_reference_parallax_mas"] = 1 / prediction[ntest:]
            initial[mode + "_coef"] = np.zeros_like(state["coef"])
            initial[mode + "_native_kpc"] = np.full(ntest, np.exp(state["ycenter"]))
            initial[mode + "_reference_kpc"] = np.full(
                len(ref), np.exp(state["ycenter"])
            )
            results[mode] = dict(
                native_posterior_agreement_kpc=probe.score(
                    prediction[:ntest], target[evaluate[:ntest]]
                ),
                independent_instrument_parallax=score_parallax(
                    prediction[ntest:],
                    raw["hipparcos_parallax_mas"],
                    raw["hipparcos_error_mas"],
                ),
            )
        np.testing.assert_array_equal(
            initial["parallax_free_reference_kpc"],
            initial["parallax_assisted_reference_kpc"],
        )
        np.savez_compressed(args.output / "paired-cohort.npz", **paired)
        common = dict(
            test_source_id=c["source_id"][test],
            test_truth_kpc=c["distance_percentiles"][test, 2],
            reference_source_id=refids,
            reference_hipparcos_parallax_mas=raw["hipparcos_parallax_mas"],
            reference_hipparcos_formal_error_mas=raw["hipparcos_error_mas"],
            reference_raw_gaia_parallax_mas=raw_parallax[ntrain + ntest :],
        )
        np.savez_compressed(args.output / "predictions.npz", **common, **predictions)
        np.savez_compressed(
            args.output / "initial_predictions.npz", **common, **initial
        )
        if file_digest(PARENT) != PARENT_SHA:
            raise ValueError("Parent changed")
        atomic_json(
            args.output / "checkpoint_verification.json",
            dict(
                both_full_encoder_reloads_exact=True,
                both_head_reloads_exact=True,
                parent_unchanged=True,
                optimizer_steps=0,
            ),
        )
        atomic_json(args.output / "comparison.json", results)
        manifest.update(
            status="complete",
            results=results,
            raw_gaia_baseline_parallax_diagnostic=baseline_metrics,
            raw_gaia_baseline_verification=baseline_gate,
            fit_count=ntrain,
            native_test_count=ntest,
            reference_count=len(ref),
            unit_audit=unit_audit,
            feature_columns=features,
            blocked_features=BLOCKED,
            parent_sha256=PARENT_SHA,
            scaler_sha256=SCALER_SHA,
            cohort_sha256=COHORT_SHA,
            cohort_content_id=CONTENT_ID,
            benchmark_source_id_sha256=bh,
            feature_sha256=file_digest(LEGACY),
            duplicates_removed=duplicates,
            ridge_alpha=10.0,
            optimizer_steps=0,
            encoder_updates=0,
            encoded_rows_per_mode=len(combined),
            runtime_seconds=time.monotonic() - started,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        atomic_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
