"""Independent full-array audit of age cohorts against raw catalogues and benchmark."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "age-feature-local-v1/age-feature-cohort.h5",
    )
    p.add_argument(
        "--features", type=Path, default=Path("/tmp/msa-labels-complete-20261015.fits")
    )
    args = p.parse_args()
    with (
        h5py.File(args.cohort) as f,
        h5py.File(ROOT / "posterior_labels_v2/posterior-labels.h5") as raw,
        h5py.File(ROOT / "posterior_labels_v2/quality-candidates-v1.h5") as qc,
        h5py.File(ROOT / "dataset_view_v2/dataset-view.h5") as benchmark,
        fits.open(args.features, memmap=True) as legacy,
    ):
        assert f.attrs["complete"]
        feature_table = legacy[1].data
        all_ids = np.asarray(feature_table["source_id"], dtype="i8")
        _, first = np.unique(all_ids, return_index=True)
        first = first[all_ids[first] > 0]
        ids = f["selected/source_id"][:]
        rows = f["selected/source_row"][:]
        surveys = f["selected/survey"][:]
        fr = f["selected/feature_row"][:]
        assert len(np.unique(ids)) == len(ids) and (ids > 0).all()
        np.testing.assert_array_equal(all_ids[fr], ids)
        assert np.isin(fr, first).all()
        xp = json.loads(f.attrs["xp_columns_json"])
        values = np.column_stack([np.asarray(feature_table[col])[fr] for col in xp])
        assert np.isfinite(values).all() and (values != -9999).all()
        for survey in raw:

            def check(name, node):
                if isinstance(node, h5py.Dataset):
                    np.testing.assert_array_equal(
                        node[:], f["raw/" + survey + "/posterior/" + name][:]
                    )

            raw[survey].visititems(check)
            np.testing.assert_array_equal(
                f["source_index/" + survey + "/age_candidate"][:],
                qc[survey + "/conservative_age_candidate"][:],
            )
            take = surveys == survey.encode()
            np.testing.assert_array_equal(
                ids[take], raw[survey + "/source_id"][:][rows[take]]
            )
            np.testing.assert_array_equal(
                f["selected/age_percentiles"][:][take],
                raw[survey + "/Age_percentiles"][:][rows[take]],
            )
            assert qc[survey + "/conservative_age_candidate"][:][rows[take]].all()
        # Independently replay fixed source precedence without looking at age/error.
        expected = {}
        for survey in ["apogee17", "galahdr3"]:
            g = f["source_index/" + survey]
            eligible = g["age_candidate"][:] & g["feature_available"][:]
            for row in np.flatnonzero(eligible):
                expected.setdefault(int(g["source_id"][row]), (survey.encode(), row))
        assert list(ids) == sorted(expected)
        assert all(expected[int(i)] == (s, r) for i, s, r in zip(ids, surveys, rows))
        val = benchmark["validation/source_id"][:]
        train = benchmark["train/source_id"][:]
        reference = np.array(
            json.loads(f.attrs["reference_json"])["source_id"], dtype="i8"
        )
        assert len(reference) == 66
        split = qc["apogee17/label_split"][
            :
        ]  # Source splits already audited independently.
        for survey in raw:
            take = surveys == survey.encode()
            np.testing.assert_array_equal(
                f["selected/label_split"][:][take],
                qc[survey + "/label_split"][:][rows[take]],
            )
        split = f["selected/label_split"][:]
        allowed = ~np.isin(ids, val) & ~np.isin(ids, reference)
        np.testing.assert_array_equal(
            f["selected/broader_fit"][:], allowed & (split == 0)
        )
        np.testing.assert_array_equal(
            f["selected/strict_fit"][:], allowed & (split == 0) & np.isin(ids, train)
        )
        np.testing.assert_array_equal(f["selected/tune"][:], allowed & (split == 1))
        for name in ["strict_fit", "broader_fit", "tune"]:
            assert not np.isin(
                ids[f["selected/" + name][:]], np.r_[reference, val]
            ).any()
        refrows = f["external_reference/feature_row"][:]
        available = refrows >= 0
        np.testing.assert_array_equal(all_ids[refrows[available]], reference[available])
        summary = dict(
            complete=True,
            all_raw_fields_exact=True,
            all_selected_ids_values_source_rows_exact=True,
            all_quality_masks_match_independent_raw_audit=True,
            all_fit_and_tune_masks_reproduced=True,
            all66_reference_ids_excluded_from_training_and_tuning=True,
            canonical_precedence_exact=True,
            selected=len(ids),
            external_feature_available=int(available.sum()),
            strict_fit=int(f["selected/strict_fit"][:].sum()),
            broader_fit=int(f["selected/broader_fit"][:].sum()),
            sha256=hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        )
    (args.cohort.parent / "independent-verification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
