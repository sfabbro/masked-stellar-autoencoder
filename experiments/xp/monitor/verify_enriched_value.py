"""Independently verify actual enriched GPU outputs against raw cohort and CPU replay."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
LAYERS = ["native", "PIGS", "Li2025_CaT_silver"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="enriched_value_frozen_v1")
    args = p.parse_args()
    out = BASE / args.run
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert (
        manifest["cohort_sha256"]
        == "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"
    )
    assert manifest["source_gate"]["publication_gate_complete"]
    checks = json.loads((out / "checkpoint_verification.json").read_text())
    assert (
        all(
            checks[k]
            for k in [
                "full_encoder_reload_all_rows_exact",
                "both_heads_reload_all_rows_exact",
                "encoder_unchanged",
                "parent_unchanged",
            ]
        )
        and checks["optimizer_steps"] == 0
    )
    report = {}
    with (
        np.load(out / "paired-cohort.npz") as pair,
        h5py.File(BASE / "enriched_feature_cohort_v1/enriched-feature-cohort.h5") as f,
    ):
        ids, rows = pair["source_id"], pair["feature_row"]
        native, aug = pair["native_fit"], pair["augmented_fit"]
        assert len(ids) == 121098 and len(np.unique(ids)) == len(ids)
        np.testing.assert_array_equal(f["features/source_id"][:][rows], ids)
        raw = f["features/values"][:][rows]
        np.testing.assert_array_equal(pair["raw_values"], raw)
        split = f["features/label_split"][:][rows]
        part = f["features/benchmark_partition"][:][rows]
        assert (
            (split[aug] == 0).all()
            and (part[aug] == 0).all()
            and np.isin(native, aug).all()
        )
        canonical = {n: f["canonical/" + n][:] for n in f["canonical"]}
        expected = np.flatnonzero(canonical["strict_fit"])
        np.testing.assert_array_equal(rows[aug], canonical["feature_row"][expected])
        np.testing.assert_array_equal(
            rows[native],
            canonical["feature_row"][expected][canonical["layer"][expected] == 0],
        )
        np.testing.assert_array_equal(
            pair["fit_source_layer"], canonical["layer"][expected]
        )
        truth = np.full(len(ids), np.nan)
        for code, n in enumerate(LAYERS):
            m = f["raw_layers/" + n + "/measurements"][:]
            which = canonical["layer"][expected] == code
            truth[aug[which]] = m["value"][canonical["layer_row"][expected][which]]
            test = f["layer_index/" + n + "/identity_test"][:]
            published = pair[n + "__measurements"]
            assert published.tobytes() == m[test].tobytes()
            positions = pair[n + "__used_row"]
            np.testing.assert_array_equal(ids[positions], published["source_id"])
            assert (split[positions] == 2).all() and not np.isin(
                ids[positions], ids[aug]
            ).any()
        np.testing.assert_array_equal(truth, pair["fit_truth"])
        columns = json.loads(f.attrs["feature_columns_json"])
        center = f["reference_scaler/center"][:]
        scale = f["reference_scaler/scale"][:]
        assert (
            hashlib.sha256(center.tobytes() + scale.tobytes()).hexdigest()
            == manifest["scaler_sha256"]
        )
        safe = np.asarray((raw - center) / scale, dtype="f4")
        blocked = {
            "ra",
            "dec",
            "pmra",
            "pmdec",
            "parallax",
            "distance",
            "age",
            "e_age",
            "fe_h",
            "feh",
            "mh",
            "metallicity",
            "mh_gspphot",
            "mh_gspspec",
        }
        for j, n in enumerate(columns):
            if n.lower() in blocked or n.lower().removeprefix("e_") in blocked:
                safe[:, j] = np.nan
        safe[~np.isfinite(safe)] = -9999
        np.testing.assert_array_equal(safe, pair["safe_inputs"])
        z = pair["latent"].astype("f8")
        c = z[native].mean(0)
        s = z[native].std(0)
        s = np.where(s > 1e-8, s, 1.0)
        yc = float(truth[native].mean())
        ys = max(float(truth[native].std()), 1e-8)
        design = np.column_stack([np.ones(len(z)), (z - c) / s])
        reg = np.eye(design.shape[1]) * np.sqrt(10.0)
        reg[0, 0] = 0
        with (
            np.load(out / "predictions.npz") as preds,
            np.load(out / "initial_predictions.npz") as initial,
        ):
            np.testing.assert_array_equal(preds["source_id"], ids)
            np.testing.assert_array_equal(initial["source_id"], ids)
            for name, fit in [("native_only", native), ("canonical_augmented", aug)]:
                state = np.load(out / (name + "-head.npz"))
                np.testing.assert_array_equal(state["center"], c)
                np.testing.assert_array_equal(state["scale"], s)
                assert state["ycenter"] == yc and state["yscale"] == ys
                a = np.vstack([design[fit], reg])
                b = np.r_[(truth[fit] - yc) / ys, np.zeros(design.shape[1])]
                coef = np.linalg.lstsq(a, b, rcond=None)[0]
                np.testing.assert_allclose(coef, state["coef"], rtol=1e-8, atol=1e-9)
                prediction = (design @ coef) * ys + yc
                np.testing.assert_allclose(
                    prediction, preds[name], rtol=1e-8, atol=1e-9
                )
                np.testing.assert_array_equal(initial[name], np.full(len(ids), yc))
                np.testing.assert_array_equal(
                    initial[name + "__coef"], np.zeros(design.shape[1])
                )
                report[name] = {
                    "max_lstsq_prediction_difference_dex": float(
                        np.max(abs(prediction - preds[name]))
                    )
                }
        # Full CPU reference already encoded every same input and reloaded whole encoder.
        cpu = BASE / "enriched-main-local-smoke-v1"
        cm = json.loads((cpu / "manifest.json").read_text())
        cc = json.loads((cpu / "checkpoint_verification.json").read_text())
        assert (
            cm["status"] == "complete"
            and cc["full_encoder_reload_all_rows_exact"]
            and cc["both_heads_reload_all_rows_exact"]
        )
        assert (
            cm["parent_sha256"] == manifest["parent_sha256"]
            and cm["encoder_state_sha256"] == manifest["encoder_state_sha256"]
        )
        with np.load(cpu / "paired-cohort.npz") as cp:
            for key in pair.files:
                if key != "latent":
                    a, b = pair[key], cp[key]
                    if a.dtype.names:
                        assert a.tobytes() == b.tobytes(), key
                    else:
                        np.testing.assert_array_equal(a, b)
            difference = np.max(abs(z - cp["latent"]))
            np.testing.assert_allclose(z, cp["latent"], rtol=5e-5, atol=5e-5)
            with (
                np.load(out / "predictions.npz") as gp,
                np.load(cpu / "predictions.npz") as cpred,
            ):
                for name in ["native_only", "canonical_augmented"]:
                    np.testing.assert_allclose(
                        gp[name], cpred[name], rtol=5e-5, atol=5e-5
                    )
                    report[name]["max_full_CPU_prediction_difference_dex"] = float(
                        np.max(abs(gp[name] - cpred[name]))
                    )
        if (out / "encoder.pth").exists():
            a = torch.load(out / "encoder.pth", map_location="cpu", weights_only=False)[
                "encoder"
            ]
            b = torch.load(cpu / "encoder.pth", map_location="cpu", weights_only=False)[
                "encoder"
            ]
            assert a.keys() == b.keys()
            for key in a:
                assert torch.equal(a[key], b[key]), key
        else:
            raise ValueError("Preserve actual GPU encoder.pth before full verification")
    report.update(
        all_source_fields_and_inputs_verified=True,
        all_fit_test_and_validation_bans_verified=True,
        common_native_fit_scaling_and_zero_initial_heads_verified=True,
        independent_augmented_lstsq_all_predictions_verified=True,
        all121098_inputs_match_completed_actual_parent_CPU_encoding_and_full_reload=True,
        max_CPU_latent_difference=float(difference),
        CPU_GPU_tolerance="rtol5e-5 atol5e-5 for latents/dex; same-device full reloads exact",
        actual_GPU_encoder_state_byte_exact_to_parent_CPU_reference=True,
        original_benchmark128unit_audit=manifest["unit_audit"],
    )
    (out / "independent-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
