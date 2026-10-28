"""Verify corrected StarHorse fields and build explicitly model-dependent QC views."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import ascii, fits

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pristine_quality_views import exact_rows
from stellar_catalog_jobs import identity_split

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"


def masks(age, dist, ageflag, inputflag, outputflag, ageinout):
    regime = np.isin(ageflag, [b"MSTO", b"SGB"])
    clean = (outputflag == b"") & (ageinout == b"")
    clean &= (np.char.find(inputflag, b"WARN") < 0) & (
        np.char.find(inputflag, b"BAD") < 0
    )

    def valid(a):
        return np.isfinite(a).all(1) & (a > 0).all(1) & (np.diff(a, axis=1) >= 0).all(1)

    agevalid = valid(age)
    distvalid = valid(dist)
    ageprecision = np.zeros(len(age), dtype=bool)
    distanceprecision = ageprecision.copy()
    ageprecision[agevalid] = (age[agevalid, 3] - age[agevalid, 1]) / (
        2 * age[agevalid, 2]
    ) <= 0.3
    distanceprecision[distvalid] = (dist[distvalid, 3] - dist[distvalid, 1]) / (
        2 * dist[distvalid, 2]
    ) <= 0.2
    return dict(
        conservative_age_candidate=clean & regime & agevalid & ageprecision,
        conservative_distance_candidate=clean & distvalid & distanceprecision,
        inputs_include_parallax=np.char.find(inputflag, b"PARALLAX") >= 0,
        age_median_above_13point8=agevalid & (age[:, 2] > 13.8),
    )


def main():
    # Synthetic warning counterexample: AgeFlag SGB alone cannot certify quality.
    a = np.tile([1.0, 2.5, 3.0, 3.5, 5.0], (3, 1))
    d = a / 2
    flag = np.array([b"SGB"] * 3)
    empty = np.array([b""] * 3)
    warning = np.array([b"", b"WARN_diff_inout", b""])
    fixture = masks(a, d, flag, empty, empty, warning)
    assert fixture["conservative_age_candidate"].tolist() == [True, False, True]
    v1 = ROOT / "posterior_labels_v1/posterior-labels.h5"
    v2 = ROOT / "posterior_labels_v2/posterior-labels.h5"
    out = ROOT / "posterior_labels_v2"
    summaries = {}
    with h5py.File(ROOT / "dataset_view_v2/dataset-view.h5") as base:
        benchmark_train = base["train/source_id"][:]
        benchmark_validation = base["validation/source_id"][:]
    ref = ROOT / "apokasc3_collection_v1"
    table = ascii.read(
        ref / "table4.dat", format="cds", readme=str(ref / "apokasc3-ReadMe")
    )
    goldrgb = (np.asarray(table["CatTab"]) == "Gold") & (
        np.asarray(table["AgeCat"]) == "RGB"
    )
    with (
        h5py.File(v1) as original,
        h5py.File(v2) as current,
        h5py.File(out / "quality-candidates-v1.h5", "w") as result,
    ):
        result.attrs["source_sha256"] = hashlib.sha256(v2.read_bytes()).hexdigest()
        result.attrs["policy"] = (
            "Versioned conservative design masks, not publisher certification: empty OutputFlag and AgeInOut; no InputFlag WARN/BAD; age regime MSTO/SGB, finite positive ordered percentiles, half68width/median<=0.30 age or<=0.20 distance. Preserve >13.8 age flag, no clipping. Labels are model-dependent StarHorse posteriors, not independent truth."
        )
        result.attrs["complete"] = False
        for name in current:
            old = original[name]
            g = current[name]

            def same(key, node):
                if isinstance(node, h5py.Dataset):
                    np.testing.assert_array_equal(node[:], g[key][:])

            old.visititems(same)
            path = out / ("raw-" + name + ".fits")
            with fits.open(path, memmap=True) as f:
                raw = f[1].data
                np.testing.assert_array_equal(
                    np.asarray(raw["GaiaEDR3"], dtype="i8"), g["source_id"][:]
                )
                for col in ["AgeInOut", "InputFlag", "OutputFlag", "AgeFlag"]:
                    np.testing.assert_array_equal(
                        np.char.strip(np.asarray(raw[col]).astype("S")),
                        g["flags/" + col][:],
                    )
                for prefix, unit in [("Age", "Gyr"), ("Dist", "kpc")]:
                    for index, suffix in enumerate(["05", "16", "50", "84", "95"]):
                        assert f[1].columns[prefix + suffix].unit == unit
                        np.testing.assert_array_equal(
                            np.asarray(raw[prefix + suffix], dtype="f8"),
                            g[prefix + "_percentiles"][:, index],
                        )
            m = masks(
                g["Age_percentiles"][:],
                g["Dist_percentiles"][:],
                g["flags/AgeFlag"][:],
                g["flags/InputFlag"][:],
                g["flags/OutputFlag"][:],
                g["flags/AgeInOut"][:],
            )
            ids = g["source_id"][:]
            joinable = ids > 0
            assert len(np.unique(ids[joinable])) == int(joinable.sum())
            m["identity_join_valid"] = joinable
            for quantity in ["age", "distance"]:
                m["conservative_" + quantity + "_candidate"] &= joinable
            train = exact_rows(benchmark_train, ids) >= 0
            val = exact_rows(benchmark_validation, ids) >= 0
            split = identity_split(ids)
            split[~joinable] = 255
            m["label_split"] = split
            m["benchmark_partition"] = np.where(train, 0, np.where(val, 1, 255)).astype(
                "u1"
            )
            m["benchmark_partition"][~joinable] = 254
            for quantity in ["age", "distance"]:
                eligible = m["conservative_" + quantity + "_candidate"]
                m["strict_" + quantity + "_fit_eligible"] = (
                    eligible & (split == 0) & train
                )
                m["broader_" + quantity + "_fit_eligible"] = (
                    eligible & (split == 0) & ~val
                )
                assert not (m["broader_" + quantity + "_fit_eligible"] & val).any()
            dest = result.create_group(name)
            dest.create_dataset("source_id", data=g["source_id"][:], compression="gzip")
            for key, val in m.items():
                dest.create_dataset(key, data=val, compression="gzip")
            ids = g["source_id"][:]
            summaries[name] = dict(
                rows=len(ids),
                invalid_missing_gaia_id_rows=int((~joinable).sum()),
                all_v1_fields_unchanged=True,
                all_raw_flags_and_asymmetric_percentiles_exact=True,
                raw_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                age_inout_warnings=int(sum(g["flags/AgeInOut"][:] != b"")),
                **{
                    key: int(val.sum())
                    for key, val in m.items()
                    if val.dtype.kind == "b"
                },
                apokasc3_overlap=len(np.intersect1d(ids, table["GaiaDR3"])),
                apokasc3_gold_recommended_RGB_overlap=len(
                    np.intersect1d(ids, table["GaiaDR3"][goldrgb])
                ),
            )
        result.attrs["complete"] = True
    report = dict(
        posterior_v2_sha256=hashlib.sha256(v2.read_bytes()).hexdigest(),
        sources=summaries,
        independent_truth=False,
        training_performed=False,
        limitation="Current StarHorse subset is MSTO/SGB; APOKASC3 Gold recommended RGB has ZERO exact overlap. The newly acquired giant reference does not directly validate this age subset. No independent age/distance accuracy claim. Parallax-free model inputs do not make parallax-conditioned labels independent.",
    )
    (out / "independent-verification-and-quality.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
