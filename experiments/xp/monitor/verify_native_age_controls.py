"""Independent pairing, checkpoint replay and all-pipeline native-age comparison."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_xp_representation_probes as probe
from pristine_quality_views import exact_rows
from run_native_age import state_digest
from stellar_age_cohorts import file_digest

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
RUNS = [
    "native_age_strict_frozen_v1",
    "native_age_strict_sft_v1",
    "native_age_strict_recon_v1",
]
OUT = ROOT / "native-age-comparison"
OUT.mkdir(exist_ok=True)
cohorts = [np.load(ROOT / run / "paired-cohort.npz") for run in RUNS]
initial = [np.load(ROOT / run / "initial_predictions.npz") for run in RUNS]
predictions = [np.load(ROOT / run / "predictions.npz") for run in RUNS]
manifests = [json.loads((ROOT / run / "manifest.json").read_text()) for run in RUNS]
replay = [
    json.loads((ROOT / run / "checkpoint_verification.json").read_text())
    for run in RUNS
]
for other in cohorts[1:]:
    assert set(other.files) == set(cohorts[0].files)
    for key in other.files:
        np.testing.assert_array_equal(other[key], cohorts[0][key], err_msg=key)
for other in initial[1:]:
    assert set(other.files) == set(initial[0].files)
    for key in other.files:
        np.testing.assert_array_equal(other[key], initial[0][key], err_msg=key)
with h5py.File(ROOT / "age_feature_cohort_v1/age-feature-cohort.h5") as f:
    for key in f["selected"]:
        np.testing.assert_array_equal(
            f["selected/" + key][:], cohorts[0][key], err_msg=key
        )
c = cohorts[0]
train, test = c["train_rows"], c["test_rows"]
np.testing.assert_array_equal(train, np.flatnonzero(c["strict_fit"]))
np.testing.assert_array_equal(test, np.flatnonzero(c["identity_test"]))
assert len(train) == 4440 and len(test) == 22830
refs = json.loads(Path("configs/legacy-seismic-reference-ids.json").read_text())[
    "source_id"
]
for mask in ["strict_fit", "broader_fit", "tune"]:
    assert not np.isin(c["source_id"][c[mask]], refs).any()
    assert not (c["benchmark_partition"][c[mask]] == 1).any()
center = float(np.log(c["age_percentiles"][train, 2]).mean())
scale = float(np.log(c["age_percentiles"][train, 2]).std())
assert center == c["label_log_center"] and scale == c["label_log_scale"]
rowhash = hashlib.sha256()
rng = np.random.default_rng(42)
for _ in range(2500):
    rowhash.update(rng.integers(4440, size=256).astype("<i8").tobytes())
for i, m in enumerate(manifests):
    assert (
        m["status"] == "complete"
        and m["head_updates"] == 2500
        and m["warmup_steps"] == 500
    )
    assert m["encoder_updates"] == (0 if i == 0 else 2000)
    assert m["actual_row_index_sha256"] == rowhash.hexdigest()
    assert m["actual_xp_mask_sha256"] == manifests[0]["actual_xp_mask_sha256"]
    for k in [
        "parent_sha256",
        "scaler_sha256",
        "cohort_content_id",
        "feature_file_sha256",
        "benchmark_source_id_sha256",
    ]:
        assert m[k] == manifests[0][k]
    assert (
        replay[i]["full_test_and_reference_reload_exact"]
        and replay[i]["decoder_unchanged"]
        and replay[i]["parent_unchanged"]
    )
    assert replay[i]["initial_encoder_sha256"] == replay[0]["initial_encoder_sha256"]
    p = predictions[i]
    for key, value in dict(
        test_source_id=c["source_id"][test],
        test_truth=c["age_percentiles"][test, 2],
        test_age_percentiles=c["age_percentiles"][test],
        test_survey=c["survey"][test],
        reference_source_id=c["reference_source_id"],
    ).items():
        np.testing.assert_array_equal(p[key], value)
    assert (
        len(p["reference_source_id"]) == 50
        and np.isin(p["reference_source_id"], refs).all()
    )
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
features = config["data"]["feature_cols"]
x, _, ids, _, _ = probe.read_fits(
    "/tmp/msa-labels-complete-20261015.fits", features, ["fe_h"], "g_normalized"
)
parent_path = Path("/tmp/msa-standard-checkpoint-57.pth")
assert file_digest(parent_path) == manifests[0]["parent_sha256"]
parent = torch.load(parent_path, map_location="cpu", weights_only=False)
signature = parent["run_signature"]
fc = np.asarray(signature["scaler_center"])
fs = np.asarray(signature["scaler_scale"])
assert (
    hashlib.sha256(fc.tobytes() + fs.tobytes()).hexdigest()
    == manifests[0]["scaler_sha256"]
)
combined = np.r_[c["source_id"][test], c["reference_source_id"]]
rows = exact_rows(ids, combined)
assert (rows >= 0).all()
safe = probe.safe_inputs(
    (x[rows] - fc) / fs, features, ["age"], manifests[0]["blocked_features"]
)
values = torch.tensor(safe, dtype=torch.float32)
model = probe.load_model(config, parent, torch.device("cpu"))
model.eval()
checkpoints = []
max_errors = []
for i, run in enumerate(RUNS):
    path = ROOT / run / "checkpoint.pth"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(saved["encoder"])
    model.encoder.eval()
    assert state_digest(model.encoder) == replay[i]["final_encoder_sha256"]
    assert saved["label_log_center"] == center and saved["label_log_scale"] == scale
    head = torch.nn.Sequential(
        torch.nn.Linear(256, 128), torch.nn.GELU(), torch.nn.Linear(128, 1)
    )
    head.load_state_dict(saved["head"])
    head.eval()
    actual = []
    with torch.no_grad():
        for block in values.split(512):
            actual.append(
                torch.exp(
                    head(model.encoder(block)).squeeze(1) * scale + center
                ).numpy()
            )
    actual = np.concatenate(actual)
    expected = np.r_[
        predictions[i]["test_prediction"], predictions[i]["reference_prediction"]
    ]
    # CPU versus CUDA GEMMs may differ; the job's same-device full reload was exact.
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=5e-5)
    max_errors.append(float(np.max(np.abs(actual - expected))))
    checkpoints.append(
        dict(run=run, sha256=file_digest(path), bytes=path.stat().st_size)
    )
rows = []


def score(kind, group, truth, predicted, run):
    good = np.isfinite(truth) & np.isfinite(predicted)
    residual = predicted[good] - truth[good]
    rows.append(
        dict(
            kind=kind,
            group=group,
            run=run,
            count=int(good.sum()),
            mae_gyr=float(np.mean(np.abs(residual))),
            rmse_gyr=float(np.sqrt(np.mean(residual**2))),
            bias_gyr=float(np.mean(residual)),
        )
    )


seismic = np.load(ROOT / "legacy_seismic_collection_v1/seismic-model-measurements.npz")
for i, run in enumerate(RUNS):
    p = predictions[i]
    truth = p["test_truth"]
    predicted = p["test_prediction"]
    score("native", "all", truth, predicted, run)
    for survey in np.unique(p["test_survey"]):
        take = p["test_survey"] == survey
        score("native", survey.decode(), truth[take], predicted[take], run)
    for lo, hi in [(0, 2), (2, 5), (5, 8), (8, 14), (14, np.inf)]:
        take = (truth >= lo) & (truth < hi)
        if take.any():
            score("native", f"age_{lo}_{hi}", truth[take], predicted[take], run)
    association = exact_rows(p["reference_source_id"], seismic["source_id"])
    for pipeline in np.unique(seismic["pipeline"]):
        take = (seismic["pipeline"] == pipeline) & (association >= 0)
        assert take.sum() == 50
        assert (seismic["age_lower_width_gyr"][take] > 0).all() and (
            seismic["age_upper_width_gyr"][take] > 0
        ).all()
        score(
            "seismic",
            pipeline.decode(),
            seismic["age_gyr"][take],
            p["reference_prediction"][association[take]],
            run,
        )
with (OUT / "conditional-metrics.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
report = dict(
    all_paired_cohort_and_initial_arrays_exact=True,
    all_native_truth_posteriors_ids_surveys_source_rows_exact=True,
    train_only_log_scaling_exact=True,
    actual_row_stream_independently_regenerated=True,
    actual_cuda_mask_hashes_identical_across_controls=True,
    mask_stream_independent_cpu_regeneration=False,
    strict_fits=4440,
    identity_test=22830,
    all66_references_excluded_fit_and_tune=True,
    external_reference_count=50,
    external_pipelines=6,
    same_device_full_reload_exact_all_jobs=True,
    independent_full_cpu_replay_max_abs_gyr=max_errors,
    independent_cpu_replay_tolerance="rtol=5e-5 atol=5e-5 Gyr, CUDA/CPU arithmetic",
    checkpoints=checkpoints,
    formal_uncertainties_not_calibrated_nll=True,
    pretraining_independence_claim=False,
)
(OUT / "independent-verification.json").write_text(json.dumps(report, indent=2) + "\n")
text = [
    "# Native StarHorse age controls",
    "",
    "All three controls used the same 4,440 strict fit stars, initial states, 2,500 head updates and row/mask streams. SFT and reconstruction adapted the encoder for 2,000 updates after the common 500-update warmup. Reconstruction adds forward passes; compute is not FLOP matched.",
    "",
    "Native identity-test agreement (22,830 stars):",
]
for r in rows:
    if r["kind"] == "native" and r["group"] == "all":
        text.append(
            f"- {r['run']}: MAE {r['mae_gyr']:.4f} Gyr, bias {r['bias_gyr']:+.4f} Gyr."
        )
text += [
    "",
    "External comparison uses all 50 feature-available LEGACY targets held out from both fitting and tuning. All six correlated seismic pipelines are reported separately in conditional-metrics.csv; no averaging or best-pipeline selection.",
    "",
]
for pipeline in np.unique(seismic["pipeline"]):
    selected = [
        r for r in rows if r["kind"] == "seismic" and r["group"] == pipeline.decode()
    ]
    text.append(
        pipeline.decode()
        + ": "
        + ", ".join(
            "{} MAE {:.4f} Gyr (bias {:+.4f})".format(
                r["run"].replace("native_age_strict_", "").replace("_v1", ""),
                r["mae_gyr"],
                r["bias_gyr"],
            )
            for r in selected
        )
        + "."
    )
text += [
    "",
    "StarHorse targets remain model/parallax-conditioned. Seismic ages share stellar physics and spectroscopic constraints, and the six pipelines are correlated. This is a single-seed bounded comparison, not model-free age truth, certified unseen pretraining, metal-poor age validation, geometric-distance accuracy or SOTA. Smaller native-label error can accompany worse external agreement or bias. Native bins and per-survey metrics are retained without changing the fitted objective or selecting test targets.",
]
(OUT / "comparison.md").write_text("\n".join(text) + "\n")
print(
    json.dumps(
        {
            "native": [
                r for r in rows if r["kind"] == "native" and r["group"] == "all"
            ],
            "seismic": [r for r in rows if r["kind"] == "seismic"],
            "replay_max_gyr": max_errors,
        },
        indent=2,
    )
)
