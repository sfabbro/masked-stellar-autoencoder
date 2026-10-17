"""Independent all-array/source/ridge/CPU replay of the actual distance pilot."""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_xp_representation_probes as probe
from pristine_quality_views import exact_rows
from stellar_age_cohorts import cohort_masks, file_digest

root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
run = root / "distance_value_frozen_v3"
out = run / "verified-comparison"
out.mkdir(exist_ok=True)
m = json.loads((run / "manifest.json").read_text())
assert m["status"] == "complete"
p = np.load(run / "paired-cohort.npz")
prediction = np.load(run / "predictions.npz")
initial = np.load(run / "initial_predictions.npz")
with h5py.File(root / "distance_feature_cohort_v1/distance-feature-cohort.h5") as f:
    for key in f["selected"]:
        np.testing.assert_array_equal(p[key], f["selected/" + key][:], err_msg=key)
    ref = np.flatnonzero(f["hipparcos_reference/evaluation_candidate"][:])
    for key in f["hipparcos_reference"]:
        np.testing.assert_array_equal(
            p["reference_" + key], f["hipparcos_reference/" + key][:][ref], err_msg=key
        )
    holdout = f["ALL_reference_holdout_source_ids"][:]
with h5py.File(root / "dataset_view_v2/dataset-view.h5") as f:
    bt, bv = f["train/source_id"][:], f["validation/source_id"][:]
masks = cohort_masks(
    p["source_id"], np.ones(len(p["source_id"]), bool), bt, bv, holdout
)
for key, value in masks.items():
    np.testing.assert_array_equal(p[key], value, err_msg=key)
train, test = p["train_rows"], p["test_rows"]
np.testing.assert_array_equal(train, np.flatnonzero(p["strict_fit"]))
np.testing.assert_array_equal(test, np.flatnonzero(p["identity_test"]))
nt, nv, nr = len(train), len(test), len(ref)
assert (nt, nv, nr) == (4452, 22932, 4166)
combined = np.r_[p["source_id"][train], p["source_id"][test], p["reference_source_id"]]
np.testing.assert_array_equal(p["combined_source_id"], combined)
assert len(np.unique(combined)) == 31550
for key in ["strict_fit", "broader_fit", "tune"]:
    assert not np.isin(p["source_id"][p[key]], np.r_[holdout, bv]).any()
np.testing.assert_array_equal(
    prediction["test_truth_kpc"], p["distance_percentiles"][test, 2]
)
np.testing.assert_array_equal(prediction["test_source_id"], p["source_id"][test])
for a in [prediction, initial]:
    np.testing.assert_array_equal(a["reference_source_id"], p["reference_source_id"])
    for key in ["hipparcos_parallax_mas", "hipparcos_formal_error_mas"]:
        source = "reference_" + key.replace("formal_", "")
        np.testing.assert_array_equal(a["reference_" + key], p[source])
baseline = np.load(root / "distance-gaia-baseline-v1/official-gaia-baseline.npz")
br = exact_rows(baseline["source_id"], p["reference_source_id"])
assert (br >= 0).all()
gaia = baseline["gaia_parallax_mas"][br]
ge = baseline["gaia_formal_error_mas"][br]
np.testing.assert_array_equal(prediction["reference_raw_gaia_parallax_mas"], gaia)
np.testing.assert_array_equal(p["raw_input_parallax"][nt + nv :], gaia)
assert (gaia <= 0).sum() == 4 and (
    p["reference_hipparcos_parallax_mas"] <= 0
).sum() == 159
for mode in ["parallax_free", "parallax_assisted"]:
    z = p["latent_" + mode]
    state = np.load(run / (mode + "-head.npz"))
    center = z[:nt].mean(0)
    scale = z[:nt].std(0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    np.testing.assert_array_equal(state["center"], center)
    np.testing.assert_array_equal(state["scale"], scale)
    y = np.log(p["distance_percentiles"][train, 2])
    yc = y.mean()
    ys = max(y.std(), 1e-8)
    assert state["ycenter"] == yc and state["yscale"] == ys
    design = np.column_stack([np.ones(len(z)), (z - center) / scale])
    penalty = np.diag(np.r_[0.0, np.full(z.shape[1], np.sqrt(10.0))])
    coef = np.linalg.lstsq(
        np.r_[design[:nt], penalty],
        np.r_[(y - yc) / ys, np.zeros(z.shape[1] + 1)],
        rcond=None,
    )[0]
    np.testing.assert_allclose(coef, state["coef"], rtol=1e-5, atol=1e-7)
    independently = np.exp((design[nt:] @ coef) * ys + yc)
    saved = np.r_[prediction[mode + "_native_kpc"], prediction[mode + "_reference_kpc"]]
    np.testing.assert_allclose(independently, saved, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(initial[mode + "_coef"], np.zeros(z.shape[1] + 1))
    for key, count in [("native", nv), ("reference", nr)]:
        np.testing.assert_array_equal(
            initial[mode + "_" + key + "_kpc"], np.full(count, np.exp(yc))
        )
    np.testing.assert_array_equal(
        1 / prediction[mode + "_reference_kpc"],
        prediction[mode + "_reference_parallax_mas"],
    )
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
features = config["data"]["feature_cols"]
assert features == m["feature_columns"]
parent_path = Path("/tmp/msa-standard-checkpoint-57.pth")
assert file_digest(parent_path) == m["parent_sha256"]
parent = torch.load(parent_path, map_location="cpu", weights_only=False)
signature = parent["run_signature"]
fc = np.asarray(signature["scaler_center"])
fs = np.asarray(signature["scaler_scale"])
assert hashlib.sha256(fc.tobytes() + fs.tobytes()).hexdigest() == m["scaler_sha256"]
with fits.open("/tmp/msa-labels-complete-20261015.fits", memmap=True) as f:
    feature_rows = np.r_[
        p["feature_row"][train], p["feature_row"][test], p["reference_feature_row"]
    ]
    np.testing.assert_array_equal(f[1].data["source_id"][feature_rows], combined)
    raw = f[1].data[feature_rows]
    rawvalues = np.column_stack([np.asarray(raw[n], dtype="f4") for n in features])
    rawvalues[rawvalues == -9999] = np.nan
    np.testing.assert_array_equal(
        np.asarray(raw["PARALLAX"], dtype="f8"), p["raw_input_parallax"]
    )
    expected = probe.safe_inputs(
        (rawvalues - fc) / fs, features, ["distance"], m["blocked_features"]
    )
    np.testing.assert_array_equal(expected, p["input_assisted"])
model = probe.load_model(config, parent, torch.device("cpu"))
model.eval()
replay_errors = {}
started = time.monotonic()
for mode in ["parallax_free", "parallax_assisted"]:
    inputs = p["input_assisted"].copy()
    if mode == "parallax_free":
        inputs = probe.safe_inputs(inputs, features, ["distance"], ["PARALLAX"])
    z = probe.encode(model, inputs, torch.device("cpu"))
    np.testing.assert_allclose(z, p["latent_" + mode], rtol=2e-4, atol=3e-4)
    state = np.load(run / (mode + "-head.npz"))
    design = np.column_stack([np.ones(len(z)), (z - state["center"]) / state["scale"]])
    d = np.exp((design[nt:] @ state["coef"]) * state["yscale"] + state["ycenter"])
    saved = np.r_[prediction[mode + "_native_kpc"], prediction[mode + "_reference_kpc"]]
    np.testing.assert_allclose(d, saved, rtol=5e-4, atol=1e-5)
    replay_errors[mode] = dict(
        max_relative_distance=float(np.max(np.abs(d - saved) / saved)),
        max_absolute_reference_parallax_mas=float(
            np.max(abs(1 / d[nv:] - 1 / saved[nv:]))
        ),
    )
report_reload = json.loads((run / "checkpoint_verification.json").read_text())
assert (
    report_reload["both_full_encoder_reloads_exact"]
    and report_reload["both_head_reloads_exact"]
    and report_reload["parent_unchanged"]
    and report_reload["optimizer_steps"] == 0
)
assert (
    m["unit_audit"]["matched_stars"] == 128
    and m["unit_audit"]["candidates"]["g_normalized"]["passed"]
)
rows = []


def metrics(kind, group, name, truth, pred, error=None):
    delta = pred - truth
    assert np.isfinite(delta).all()
    value = dict(
        kind=kind,
        group=group,
        variant=name,
        count=len(delta),
        mae=float(np.mean(abs(delta))),
        rmse=float(np.sqrt(np.mean(delta**2))),
        bias=float(delta.mean()),
        formal_nll=None,
    )
    if error is not None:
        value["formal_nll"] = float(
            np.mean(
                0.5 * (delta / error) ** 2 + np.log(error) + 0.5 * np.log(2 * np.pi)
            )
        )
    rows.append(value)


hip = p["reference_hipparcos_parallax_mas"]
he = p["reference_hipparcos_error_mas"]
variants = {
    "raw_gaia": gaia,
    **{
        mode: prediction[mode + "_reference_parallax_mas"]
        for mode in ["parallax_free", "parallax_assisted"]
    },
}
for name, pred in variants.items():
    metrics("reference_mas", "all", name, hip, pred, he)
    for label, take in [("hip_nonpositive", hip <= 0), ("hip_positive", hip > 0)]:
        metrics("reference_mas", label, name, hip[take], pred[take], he[take])
for mode in ["parallax_free", "parallax_assisted"]:
    truth = prediction["test_truth_kpc"]
    pred = prediction[mode + "_native_kpc"]
    metrics("native_kpc", "all", mode, truth, pred)
    for survey in np.unique(p["survey"][test]):
        take = p["survey"][test] == survey
        metrics("native_kpc", survey.decode(), mode, truth[take], pred[take])
    for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, np.inf)]:
        take = (truth >= lo) & (truth < hi)
        if take.any():
            metrics("native_kpc", f"posterior_{lo}_{hi}", mode, truth[take], pred[take])
with (out / "conditional-metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with (out / "all-reference-diagnostics.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "source_id",
            "hipparcos_parallax_mas",
            "hipparcos_formal_error_mas",
            "gaia_raw_parallax_mas",
            "gaia_formal_error_mas",
            "free_predicted_parallax_mas",
            "assisted_predicted_parallax_mas",
        ]
    )
    writer.writerows(
        zip(
            p["reference_source_id"],
            hip,
            he,
            gaia,
            ge,
            variants["parallax_free"],
            variants["parallax_assisted"],
        )
    )
combined_error = np.sqrt(he**2 + ge**2)
standardized = (gaia - hip) / combined_error
report = dict(
    all_selected_cohort_arrays_exact=True,
    all_4166_reference_arrays_exact=True,
    all_31550_actual_input_values_and_raw_parallaxes_exact=True,
    all_fit_tune_reference_validation_exclusions_verified=True,
    all_zero_initial_heads_and_constant_predictions_exact=True,
    train_only_scalers_and_augmented_lstsq_coefficients_predictions_verified=True,
    actual_128_XP_unit_audit_passed=True,
    all_native_and_reference_predictions_independently_CPU_replayed=True,
    CPU_CUDA_distance_tolerance="rtol5e-4 atol1e-5kpc; latent rtol2e-4 atol3e-4",
    CPU_replay_errors=replay_errors,
    CPU_replay_seconds=time.monotonic() - started,
    raw_gaia_combined_formal_error_diagnostic=dict(
        mean=float(standardized.mean()),
        standard_deviation=float(standardized.std()),
        fraction_abs_above3=float((abs(standardized) > 3).mean()),
        not_unique_absolute_error_floor=True,
    ),
    retained_nonpositive_hipparcos=159,
    retained_nonpositive_gaia=4,
    no_measurement_parallax_inverted=True,
    no_posthoc_clipping_or_SNR_filter=True,
    formal_error_is_total_calibrated_accuracy=False,
)
(out / "independent-verification.json").write_text(json.dumps(report, indent=2) + "\n")
text = [
    "# Bounded frozen distance comparison",
    "",
    "Matched 4,452 strict native StarHorse fit stars, 22,932 native identity-test stars and all 4,166 Hipparcos reference candidates. All 99,574 reference holdout IDs and every benchmark validation ID were excluded from fitting and tuning. Both fixed-alpha10 heads use a frozen epoch57 encoder and train-only scaling of latent features and log-distance targets.",
    "",
    "Native StarHorse agreement is measured in kpc; external errors are measured directly in parallax mas. No noisy measured parallax is inverted. All 159 nonpositive Hipparcos and four nonpositive Gaia measurements are retained.",
    "",
]
for r in rows:
    if r["group"] == "all":
        text.append(
            "{} {}: MAE {:.6g}, RMSE {:.6g}, bias {:+.6g} (N={}).".format(
                r["kind"], r["variant"], r["mae"], r["rmse"], r["bias"], r["count"]
            )
        )
text += [
    "",
    "The processed native-label gain does not establish improved geometric accuracy. Both heads have much worse Hipparcos parallax agreement than the independently validated raw Gaia baseline. The parallax-free head produces extreme finite extrapolations; these remain in the saved arrays and all-star metrics without clipping. The assisted head also fails to match the raw Gaia baseline. No further distance wave is justified by these results alone.",
    "",
    "StarHorse labels are Gaia/model-conditioned; supervised reference holdouts do not exclude unsupervised pretraining overlap. Hipparcos formal errors do not certify total precision, the weight normalization remains unresolved, and binary/spatial/frame/calibration dependencies remain. Gaussian scores are formal-error diagnostics, not calibrated predictive NLL. Raw Gaia measurements received no zero-point correction. This is a bounded single-seed ridge-head comparison, not SOTA or beyond-Gaia performance.",
]
(out / "comparison.md").write_text("\n".join(text) + "\n")
print(
    json.dumps(
        dict(
            all_star_metrics=[r for r in rows if r["group"] == "all"],
            CPU_replay_errors=replay_errors,
        ),
        indent=2,
    )
)
