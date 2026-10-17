"""Independent source-row and augmented least-squares replay of saved Pristine heads."""

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
OUT = ROOT / "pristine_value_frozen_v1"


def main():
    m = json.loads((OUT / "manifest.json").read_text())
    assert m["status"] == "complete"
    assert (
        m["parent_sha256"]
        == "9b033ae63a3b87c93dcfeffd989172f7888d029f77143e6cf315cf7a032ab2ca"
    )
    assert m["unit_audit"]["matched_stars"] >= 128
    assert m["unit_audit"]["candidates"]["g_normalized"]["passed"]
    check = json.loads((OUT / "checkpoint_verification.json").read_text())
    assert all(
        check[k]
        for k in [
            "all_six_head_reloads_exact",
            "frozen_encoder_reload_exact",
            "parent_unchanged",
        ]
    )
    assert check["optimizer_steps"] == 0
    a = np.load(OUT / "paired-cohort.npz")
    p = np.load(OUT / "predictions.npz")
    initial = np.load(OUT / "initial_predictions.npz")
    ids = a["source_id"]
    nr = a["native_row"]
    assert len(np.unique(ids)) == len(ids) == m["common_cohort"]
    native = ROOT / "label_registry_v2/labels.h5"
    assert hashlib.sha256(native.read_bytes()).hexdigest() == m["registry_sha256"]
    with h5py.File(native) as f:
        n = f["selected/labels"][:][nr]
        for dest, source in [
            ("source_id", "source_id"),
            ("truth", "value"),
            ("uncertainty", "uncertainty"),
            ("survey", "survey"),
        ]:
            np.testing.assert_array_equal(a[dest], n[source])
    with h5py.File(ROOT / "pristine_quality_view_v1/pristine-quality-view.h5") as f:
        n = f["native"]
        assert n["input_eligible"][:][nr].all()
        for dest, source in [
            ("split", "label_split"),
            ("benchmark_partition", "benchmark_partition"),
            ("pristine_row", "pristine_row"),
        ]:
            np.testing.assert_array_equal(a[dest], n[source][:][nr])
    raw = ROOT / "pristine_full_fetch_v1_r2/pristine-dr1-full.fits"
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == m["pristine_sha256"]
    with fits.open(raw, memmap=True) as f:
        d = f[1].data[a["pristine_row"]]
        np.testing.assert_array_equal(ids, d["GaiaDR3"])
        extra = np.array(
            [
                d["Bpmag0"] - d["Rpmag0"],
                d["Gmag0"] - d["Rpmag0"],
                d["CaHKmag0"] - d["Gmag0"],
            ]
        ).T
        np.testing.assert_array_equal(extra, a["extra"])
    test = np.flatnonzero(a["split"] == 2)
    for saved in [p, initial]:
        np.testing.assert_array_equal(saved["source_id"], ids[test])
        np.testing.assert_array_equal(saved["truth"], a["truth"][test])
    rows = []
    verified = []
    for cohort in ["strict_benchmark_train", "broad_excluding_benchmark_validation"]:
        fit = p[cohort + "__fit_row"]
        expected = (a["split"] == 0) & (
            a["benchmark_partition"] == 0
            if cohort.startswith("strict")
            else a["benchmark_partition"] != 1
        )
        np.testing.assert_array_equal(fit, np.flatnonzero(expected))
        assert not (a["benchmark_partition"][fit] == 1).any()
        for variant, k in [
            ("encoder", 0),
            ("encoder_gaia", 2),
            ("encoder_gaia_cahk", 3),
        ]:
            key = cohort + "__" + variant
            x = np.column_stack([a["latent"], a["extra"][:, :k]])
            with np.load(OUT / (key + "-head.npz")) as head:
                np.testing.assert_array_equal(head["center"], x[fit].mean(0))
                scale = x[fit].std(0)
                scale = np.where(scale > 1e-8, scale, 1.0)
                np.testing.assert_array_equal(head["scale"], scale)
                yc = a["truth"][fit].mean()
                ys = max(a["truth"][fit].std(), 1e-8)
                assert head["ycenter"] == yc and head["yscale"] == ys
                design = np.column_stack(
                    [np.ones(len(x)), (x - head["center"]) / scale]
                )
                # Independent QR/SVD-style augmented least squares, not normal equations.
                penalty = np.eye(design.shape[1]) * np.sqrt(10.0)
                penalty[0, 0] = 0
                coef = np.linalg.lstsq(
                    np.vstack([design[fit], penalty]),
                    np.r_[(a["truth"][fit] - yc) / ys, np.zeros(len(penalty))],
                    rcond=None,
                )[0]
                replay = (design[test] @ coef) * ys + yc
                np.testing.assert_allclose(replay, p[key], rtol=1e-8, atol=1e-8)
                np.testing.assert_array_equal(initial[key], np.full(len(test), yc))
            delta = p[key] - a["truth"][test]
            assert abs(np.mean(abs(delta)) - m["results"][key]["mae_dex"]) < 1e-12
            for group, mask in [
                ("all", np.ones(len(test), dtype=bool)),
                ("FeH_lt_minus2", a["truth"][test] < -2),
                ("FeH_lt_minus3", a["truth"][test] < -3),
            ] + [
                (tag.decode(), a["survey"][test] == tag)
                for tag in np.unique(a["survey"][test])
            ]:
                rows.append(
                    dict(
                        cohort=cohort,
                        variant=variant,
                        group=group,
                        n=int(mask.sum()),
                        mae_dex=float(np.mean(abs(delta[mask])))
                        if mask.any()
                        else None,
                        bias_dex=float(delta[mask].mean()) if mask.any() else None,
                    )
                )
            verified.append(key)
    with (OUT / "conditional-metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        all_native_truth_errors_surveys_ids_and_photometry_exact=True,
        fit_masks_exclude_benchmark_validation=True,
        all_six_train_only_scalers_and_initial_arrays_verified=True,
        independent_augmented_lstsq_predictions_match=True,
        verified=verified,
        common_cohort=len(ids),
        test_rows=len(test),
        metalpoor_test_n=int(sum(a["truth"][test] < -2)),
        extremely_metalpoor_test_n=int(sum(a["truth"][test] < -3)),
    )
    (OUT / "independent-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
