"""Audit actual FITS source tags and reproduce the saved probe row selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[2]


def identity_split(ids, seed=42):
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(seed)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    b = x % np.uint64(1000)
    return np.where(b < 700, 0, np.where(b < 850, 1, 2))


def counts(tags, values):
    result = {}
    for tag in np.unique(tags):
        z = values[tags == tag]
        finite = np.isfinite(z) & (z != -9999)
        result[str(tag)] = {
            "rows": len(z),
            "finite_metallicity": int(finite.sum()),
            "below_minus2": int((finite & (z < -2)).sum()),
            "below_minus2p5": int((finite & (z < -2.5)).sum()),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fits_path", type=Path)
    args = parser.parse_args()
    with fits.open(args.fits_path, memmap=True) as f:
        expected = (
            f[1].fileinfo()["datLoc"] + f[1].header["NAXIS1"] * f[1].header["NAXIS2"]
        )
        if args.fits_path.stat().st_size < expected:
            raise ValueError("Refuse incomplete catalogue download")
        t = f[1].data
        ids = np.array(t["source_id"], dtype=np.int64)
        tags = np.char.strip(np.asarray(t["spec_source"]).astype(str))
        feh = np.array(t["fe_h"], dtype=np.float64)
        dec = np.array(t["DEC"], dtype=np.float64)
        age = np.array(t["age"], dtype=np.float64)
        eage = np.array(t["e_age"], dtype=np.float64)
    summary = {
        "catalogue_rows": len(ids),
        "raw_sources": counts(tags, feh),
        "label_definition_caveat": "Builder renames APOGEE M_H to fe_h; GALAH fe_h and Li FeH are appended under the same column. Mixed abundance definitions are not homogenized.",
        "notebook": "data/fine-tune-dataset-maker.ipynb",
        "duplicate_policy": "Retain first row per source_id in original FITS order, exactly as probe reader",
        "splits": {},
    }
    galah = (tags == "galah") & np.isfinite(age) & np.isfinite(eage) & (age > 0)
    summary["galah_age_uncertainty_sqrt_age"] = {
        "checked_rows": int(galah.sum()),
        "all_match": bool(
            np.allclose(eage[galah] ** 2, age[galah], rtol=1e-6, atol=1e-6)
        ),
        "meaning": "Notebook assigned e_age=sqrt(age); these are not independently quoted measurement uncertainties.",
    }
    _, first = np.unique(ids, return_index=True)
    keep = np.sort(first)
    summary["duplicate_rows_removed"] = len(ids) - len(keep)
    ids, tags, feh, dec = ids[keep], tags[keep], feh[keep], dec[keep]
    summary["deduplicated_sources"] = counts(tags, feh)
    for split in ["identity", "sky"]:
        partitions = identity_split(ids)
        if split == "sky":
            assert np.isfinite(dec).all()
            partitions = np.where(dec >= 30, 2, np.where(partitions == 1, 1, 0))
        rng = np.random.default_rng(42)
        selections = []
        report = {}
        for k, cap, name in [
            (0, 100000, "train"),
            (1, 20000, "tune"),
            (2, 20000, "test"),
        ]:
            rows = np.flatnonzero(partitions == k)
            report[name + "_before_cap"] = counts(tags[rows], feh[rows])
            rng.shuffle(rows)
            rows = rows[:cap]
            selections.append(rows)
            report[name] = counts(tags[rows], feh[rows])
        saved = np.load(
            ROOT
            / f"experiments/xp/results-followup-e47/probe_parent47/{split}/probe_predictions.npz"
        )
        selected = selections[2]
        finite = np.isfinite(feh[selected]) & (feh[selected] != -9999)
        test = selected[finite]
        np.testing.assert_array_equal(ids[test], saved["fe_h_source_ids"])
        np.testing.assert_allclose(feh[test], saved["fe_h_truth"], rtol=1e-6, atol=1e-6)
        report["saved_test_ids_and_labels_match"] = True
        report["test_source_id_sha256"] = hashlib.sha256(
            ids[test].tobytes()
        ).hexdigest()
        report["li_test_source_ids"] = ids[test][tags[test] == "li_et_al_vmps"].tolist()
        report["per_source_prediction_errors"] = {}
        predictions_paths = list(
            (ROOT / "experiments/xp/results-followup-e47").glob(
                f"probe_*/{split}/probe_predictions.npz"
            )
        )
        predictions_paths.append(
            ROOT
            / f"experiments/xp/results-latest-standard-e57/probe_latest_standard57/{split}/probe_predictions.npz"
        )
        for path in predictions_paths:
            with np.load(path) as predictions:
                np.testing.assert_array_equal(predictions["fe_h_source_ids"], ids[test])
                np.testing.assert_array_equal(
                    predictions["fe_h_truth"], saved["fe_h_truth"]
                )
                keys = [k for k in predictions.files if k.startswith("fe_h_head_")]
                head = max(keys, key=lambda key: int(key.rsplit("_", 1)[1]))
                methods = {}
                for method, key in [("ridge", "fe_h_ridge"), ("frozen_head", head)]:
                    by_source = {}
                    for tag in np.unique(tags[test]):
                        strata = {}
                        for name, match in [
                            ("all", tags[test] == tag),
                            ("below_minus2", (tags[test] == tag) & (feh[test] < -2)),
                        ]:
                            residual = (
                                predictions[key][match]
                                - predictions["fe_h_truth"][match]
                            )
                            strata[name] = {
                                "count": len(residual),
                                "mae_dex": float(np.abs(residual).mean())
                                if len(residual)
                                else None,
                                "bias_dex": float(residual.mean())
                                if len(residual)
                                else None,
                                "low_count": len(residual) < 20,
                            }
                        by_source[str(tag)] = strata
                    methods[method] = by_source
                report["per_source_prediction_errors"][path.parent.parent.name] = (
                    methods
                )
        summary["splits"][split] = report
    out = ROOT / "experiments/xp/science-review-20261015/label-source-provenance.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
