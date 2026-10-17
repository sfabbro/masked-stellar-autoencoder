"""Independent native-source/stream/score checks and full final CPU replay.

Run after actual artifacts and full checkpoints have been fetched. No fitting.
Raw HDF5 remains authoritative for dtype metadata/attributes dropped by NPZ.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_xp_representation_probes as probe  # noqa: E402
from stellar_catalog_jobs import identity_split  # noqa: E402

from masked_stellar_autoencoder.models.checkpoint_load import (
    torch_load_trusted,  # noqa: E402
)
from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_prediction_head,  # noqa: E402
)
from masked_stellar_autoencoder.training.spectrophotometric_inputs import (
    spectrophotometric_inputs,  # noqa: E402
)


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(directory, cohort, parent, *, input_reference=None, manifest_override=None):
    manifest = (
        manifest_override
        if manifest_override is not None
        else json.loads((directory / "manifest.json").read_text())
    )
    input_max_difference = 0.0
    if manifest["status"] != "complete" or manifest["smoke"]:
        raise ValueError("Only completed production artifacts may be verified")
    policy, arm = manifest["policy"], manifest["arm"]
    assert sha(parent) == manifest["parent_sha256"] == policy["checkpoint_sha256"]
    assert (
        sha(cohort)
        == manifest["cohort_sha256"]
        == "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"
    )
    paired = np.load(directory / "paired-cohort.npz", allow_pickle=False)
    final = np.load(directory / "predictions.npz", allow_pickle=False)
    initial = np.load(directory / "initial_predictions.npz", allow_pickle=False)
    flow = arm["loss"] == "contrastive_flow"
    if not flow:
        assert np.all(initial["member_outputs"] == 0)
    with h5py.File(cohort) as f:
        raw = f["raw_layers/native/measurements"][:]
        feature_row = f["layer_index/native/feature_row"][:]
        ids = f["features/source_id"][:]
        np.testing.assert_array_equal(raw["source_id"], ids[feature_row])
        split = identity_split(raw["source_id"])
        available = f["features/feature_available"][:][feature_row]
        partition = f["features/benchmark_partition"][:][feature_row]
        fits = raw[available & (partition == 0) & (split == 0)]
        tests = raw[available & (split == 2)]
        assert len(fits) == 19859 and len(tests) == 100620
        for expected, saved in (
            (fits, paired["fit_measurements"]),
            (tests, paired["test_measurements"]),
        ):
            expected, saved = (a[np.argsort(a["source_id"])] for a in (expected, saved))
            assert expected.dtype.names == saved.dtype.names
            # Structured field values and bytes, not dropped HDF5 dtype attributes.
            for field in expected.dtype.names:
                assert expected[field].tobytes() == saved[field].tobytes(), field
        order = np.argsort(ids)
        for kind in ("fit", "test"):
            wanted = (
                paired["fit_source_id"]
                if kind == "fit"
                else paired["test_measurements"]["source_id"]
            )
            rows = order[np.searchsorted(ids[order], wanted)]
            np.testing.assert_array_equal(ids[rows], wanted)
            raw_values = f["features/values"][:][rows]
            values = (raw_values - f["reference_scaler/center"][:]) / f[
                "reference_scaler/scale"
            ][:]
            names = json.loads(f.attrs["feature_columns_json"])
            safe, _ = spectrophotometric_inputs(
                torch.from_numpy(values.astype("f4")),
                torch.from_numpy(np.isfinite(raw_values)),
                names,
            )
            if input_reference is None:
                np.testing.assert_array_equal(safe.numpy(), paired[kind + "_input"])
            else:
                expected = input_reference(kind, rows, safe.numpy())
                actual = paired[kind + "_input"]
                assert actual.dtype == expected.dtype == np.dtype("f4")
                np.testing.assert_array_equal(expected == -9999, actual == -9999)
                np.testing.assert_allclose(expected, actual, rtol=5e-6, atol=5e-6)
                input_max_difference = max(
                    input_max_difference, float(np.max(np.abs(expected - actual)))
                )
    target = paired["fit_measurements"]["value"]
    np.testing.assert_array_equal(target, paired["fit_target"])
    np.testing.assert_array_equal(
        paired["fit_measurements"]["source_id"], paired["fit_source_id"]
    )
    assert not np.isin(
        paired["fit_source_id"], paired["test_measurements"]["source_id"]
    ).any()
    np.testing.assert_allclose(
        [float(paired["label_center"]), float(paired["label_scale"])],
        [target.mean(), target.std()],
        rtol=1e-14,
    )
    rows_digest = hashlib.sha256()
    for size in (9_000_000, 1_000_000):
        rows_digest.update(np.arange(size, dtype="i8").tobytes())
    assert rows_digest.hexdigest() == manifest["row_sha256"]
    assert (
        manifest["presentations"] == 10_000_000 and manifest["optimizer_updates"] == 306
    )
    rng, label_digest = np.random.default_rng(42), hashlib.sha256()
    for _ in range(500 + 306 * 8):
        label_digest.update(rng.integers(19859, size=256).tobytes())
    assert label_digest.hexdigest() == manifest["supervised_stream_sha256"]
    assert (
        manifest["supervised_warmup_presentations"]
        + manifest["supervised_adaptation_presentations"]
        == 754688
    )
    saved = torch_load_trusted(
        directory / "checkpoint.pth", map_location="cpu", weights_only=False
    )
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    config = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())
    model = probe.load_model(config, checkpoint, torch.device("cpu"))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.eval()
    flow_report = {}
    if flow:
        from contrastive_flow_reference import replay

        parent_model = probe.load_model(config, checkpoint, torch.device("cpu")).eval()
        output, flow_report = replay(
            model, parent_model, saved, paired, initial, final, arm, directory
        )
        del parent_model
    else:
        head = build_prediction_head(arm["head"])
        head.load_state_dict(saved["head_state_dict"], strict=True)
        head.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, 100620, 512):
                predictions.append(
                    head(
                        model.encoder(
                            torch.from_numpy(paired["test_input"][start : start + 512])
                        )
                    ).numpy()
                )
        output = np.concatenate(predictions)
        np.testing.assert_allclose(
            output, final["member_outputs"], rtol=5e-5, atol=5e-5
        )
    reconstruction = np.load(
        directory / "initial_reconstruction.npz", allow_pickle=False
    )
    hidden = np.where(
        reconstruction["mask"] | ~np.isfinite(reconstruction["target"]),
        -9999,
        reconstruction["target"],
    )
    reconstructed = []
    with torch.no_grad():
        for start in range(0, len(hidden), 512):
            reconstructed.append(
                model(torch.from_numpy(hidden[start : start + 512]))[0].numpy()
            )
    actual_reconstruction = np.concatenate(reconstructed)
    with np.load(directory / "final_reconstruction.npz") as expected:
        np.testing.assert_allclose(
            actual_reconstruction, expected["prediction"], rtol=5e-5, atol=5e-5
        )
        reconstruction_diff = float(
            np.max(np.abs(actual_reconstruction - expected["prediction"]))
        )
    yc, ys = float(paired["label_center"]), float(paired["label_scale"])
    raw_point = (
        final["standardized_mle"] if flow else final["member_outputs"][:, :, 0].mean(1)
    )
    point = raw_point * ys + yc
    np.testing.assert_array_equal(point, final["prediction"])
    truth = paired["test_measurements"]["value"]
    report = dict(
        status="passed",
        run_id=manifest["run_id"],
        native_fit=19859,
        native_test=100620,
        cpu_replay_all_test_member_outputs=not flow,
        cpu_replay_all_test_grid_mle=flow,
        cpu_replay_all_validation_decoder_outputs=True,
        cpu_decoder_rows=len(hidden),
        cpu_decoder_max_absolute_difference=reconstruction_diff,
        cpu_max_absolute_member_output_difference=float(
            np.max(
                np.abs(output - final["standardized_mle" if flow else "member_outputs"])
            )
        ),
        cpu_rtol=5e-5,
        cpu_atol=5e-5,
        all_native_raw_fields_bytes_and_inputs_verified=True,
        original_hdf5_metadata_preserved_in_raw_artifact=True,
        corrected_inputs_independently_derived=input_reference is not None,
        corrected_input_max_absolute_difference=input_max_difference,
        corrected_input_rtol=5e-6 if input_reference else 0.0,
        corrected_input_atol=5e-6 if input_reference else 0.0,
        initial_outputs_zero=not flow,
        replay_and_supervised_streams_independently_regenerated=True,
        mask_sha256=manifest["mask_sha256"],
        mask_note="Compare across matched GPU jobs; not independently CPU-regenerated",
        mae_dex=float(np.abs(point - truth).mean()),
        bias_dex=float((point - truth).mean()),
        final_checkpoint_sha256=sha(directory / "checkpoint.pth"),
        source_sha=manifest["source_sha"],
    )
    report.update(flow_report)
    if arm["loss"] in ("gaussian_nll", "gaussian_crps"):
        mu = final["member_outputs"][:, 0, 0].astype("f8") * ys + yc
        sigma = np.exp(final["member_outputs"][:, 0, 1].astype("f8") / 2) * ys
        z = (truth - mu) / sigma
        cdf = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in z]))
        pdf = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        crps = np.mean(sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi)))
        nll = np.mean(0.5 * z * z + np.log(sigma) + 0.5 * math.log(2 * math.pi))
        np.testing.assert_allclose(
            [crps, nll],
            [
                manifest["comparison"]["probabilistic"]["gaussian_crps_dex"],
                manifest["comparison"]["probabilistic"]["gaussian_nll_dex"],
            ],
            rtol=1e-7,
        )
        report.update(gaussian_crps_dex=float(crps), gaussian_nll_dex=float(nll))
    (directory / "independent-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT
        / "experiments/xp/results-stellar-improvement/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    parser.add_argument(
        "--parent",
        type=Path,
        default=ROOT
        / "experiments/xp/results-stellar-improvement/prediction-comparison-e72/parent/checkpoint-72.pth",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(verify(args.directory, args.cohort, args.parent)))
