import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
reportpath = root / "survey_overlap_v1/independent-verification.json"
report = json.loads(reportpath.read_text())
summary = json.loads((root / "survey_overlap_v1/overlap-summary.json").read_text())
pairs = np.load(root / "survey_overlap_v1/overlap-pairs.npz")
paths = {
    "native": root / "label_registry_v2/labels.h5",
    "PIGS": root / "pigs_labels_v1/pigs-labels.h5",
    "DESI": root / "desi_labels_v1/desi-labels.h5",
}
for key, path in paths.items():
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    assert h.hexdigest() == summary["input_provenance"][key]["sha256"]
source_checks = []
conditional = []
for name in summary["comparisons"]:
    ln, rn = name.split("_minus_")
    ids = pairs[name + "__source_id"]
    for side, survey in [("left", ln), ("right", rn)]:
        p = paths["native" if survey in ["galah", "apogee"] else survey]
        with h5py.File(p) as f:
            row = pairs[name + "__" + side + "_source_row"]
            if survey == "PIGS":
                d = f["selected/labels"][:]
                order = np.argsort(d["source_id"])
                pos = np.searchsorted(d["source_id"][order], ids)
                actual = d[order[pos]]
                values = actual["value"]
                errors = actual["uncertainty"]
                assert np.array_equal(actual["source_row"], row)
            else:
                unique, inverse = np.unique(row, return_inverse=True)
                ds = f["measurements"]
                actual = ds[unique][inverse]
                values = (
                    actual["feh_corrected"] if survey == "DESI" else actual["value"]
                )
                errors = (
                    actual["feh_err_native"]
                    if survey == "DESI"
                    else actual["uncertainty"]
                )
                if survey == "DESI":
                    assert actual["eligible"].all()
                else:
                    assert actual["quality_pass"].all()
                    assert (actual["survey"] == survey.encode()).all()
                    assert (actual["quantity"] == b"FeH").all()
            np.testing.assert_array_equal(actual["source_id"], ids)
            np.testing.assert_array_equal(values, pairs[name + "__" + side + "_feh"])
            np.testing.assert_array_equal(errors, pairs[name + "__" + side + "_sigma"])
            if survey == "DESI" and len(ids):
                delta = pairs[name + "__left_feh"] - pairs[name + "__right_feh"]
                split = pairs[name + "__split"]
                sig = np.hypot(
                    pairs[name + "__left_sigma"], pairs[name + "__right_sigma"]
                )
                prop = actual["temperature_correction_err_independent"]
                sigtemp = np.hypot(sig, prop)
                for label, mask in [
                    ("all", np.ones(len(ids), bool)),
                    ("Teff_4200_5000", actual["teff"] < 5000),
                    (
                        "Teff_5000_5800",
                        (actual["teff"] >= 5000) & (actual["teff"] < 5800),
                    ),
                    ("Teff_5800_6600", actual["teff"] >= 5800),
                    ("logg_lt4p3", actual["logg"] < 4.3),
                    ("logg_ge4p3", actual["logg"] >= 4.3),
                ]:
                    train = mask & (split == 0)
                    test = mask & (split == 2)
                    v = delta[test]
                    n = len(v)
                    conditional.append(
                        dict(
                            pair=name,
                            subset=label,
                            n_train=int(train.sum()),
                            n_test=n,
                            train_median=float(np.median(delta[train]))
                            if train.any()
                            else None,
                            test_median=float(np.median(v)) if n else None,
                            test_robust_scatter=float(
                                1.4826 * np.median(abs(v - np.median(v)))
                            )
                            if n
                            else None,
                            test_median_random_pair_sigma=float(np.median(sig[test]))
                            if n
                            else None,
                            test_fraction_gt3_random_sigma=float(
                                (abs(v) > 3 * sig[test]).mean()
                            )
                            if n
                            else None,
                            test_median_random_plus_independent_Teff_sigma=float(
                                np.median(sigtemp[test])
                            )
                            if n
                            else None,
                            test_fraction_gt3_including_independent_Teff=float(
                                (abs(v) > 3 * sigtemp[test]).mean()
                            )
                            if n
                            else None,
                        )
                    )
        source_checks.append(
            dict(
                pair=name,
                side=side,
                source=survey,
                rows=len(ids),
                exact_ids_values_errors_original_rows=True,
            )
        )
report.update(
    source_measurement_checks_pending=False,
    source_files_sha256_match=True,
    source_measurement_checks=source_checks,
)
reportpath.write_text(json.dumps(report, indent=2) + "\n")
(root / "survey_overlap_v1/desi-conditional-diagnostics.json").write_text(
    json.dumps(
        {
            "bins": conditional,
            "caveat": "Temperature propagation assumes independent random error. Parameter covariance, logg branch switching, abundance/correction systematics and reference-survey systematics remain unknown; no total uncertainty assertion or fitted correction applied.",
        },
        indent=2,
    )
    + "\n"
)
print("Verified all input SHAs and every paired source row, value and uncertainty.")
print(json.dumps(conditional))
