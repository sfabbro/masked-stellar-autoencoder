"""Independent all-row sidecar verification against original native FITS.

Does not import materialization or target-mask helpers; no fitting or download.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify(sidecar, cohort, sources, pins, output):
    with h5py.File(sidecar) as f, h5py.File(cohort) as c:
        assert f.attrs["complete"] and c.attrs["complete"]
        provenance = json.loads(f.attrs["provenance_json"])
        assert digest(cohort) == provenance["cohort_sha256"]
        ids = c["features/source_id"][:]
        np.testing.assert_array_equal(ids, f["source_id"][:])
        value, error, observed = (
            f["targets/" + n][:] for n in ["value", "uncertainty", "observed"]
        )
        assert value.shape == error.shape == observed.shape == (len(ids), 3)
        native = c["raw_layers/native/measurements"][:]
        feature_rows = c["layer_index/native/feature_row"][:]
        np.testing.assert_array_equal(native, f["native_measurements"][:])
        np.testing.assert_array_equal(feature_rows, f["native_feature_row"][:])
        assert np.array_equal(native["source_id"], ids[feature_rows])
        supported = np.zeros(len(ids), bool)
        counts = {}
        for source, mapping in {
            "galah": (
                "gaiadr3_source_id",
                ["teff", "logg", "fe_h"],
                ["e_teff", "e_logg", "e_fe_h"],
            ),
            "apogee": (
                "GAIAEDR3_SOURCE_ID",
                ["TEFF", "LOGG", "FE_H"],
                ["TEFF_ERR", "LOGG_ERR", "FE_H_ERR"],
            ),
        }.items():
            assert (
                digest(sources[source])
                == pins[source]
                == provenance["sources"][source]["sha256"]
            )
            keep = native["survey"] == source.encode()
            record = native[keep]
            row = feature_rows[keep]
            supported[row] = True
            np.testing.assert_array_equal(
                f["targets/source_row"][:][row], record["source_row"]
            )
            np.testing.assert_array_equal(
                f["targets/source_file"][:][row], record["input_file"]
            )
            np.testing.assert_array_equal(
                f["targets/survey"][:][row],
                np.full(len(row), source.encode(), dtype="S32"),
            )
            with fits.open(sources[source], memmap=True) as hdul:
                raw = hdul[1].data[record["source_row"]]
                np.testing.assert_array_equal(raw[mapping[0]], ids[row])
                expected_value = np.stack(
                    [raw[n].astype("f8") for n in mapping[1]], axis=1
                )
                expected_error = np.stack(
                    [raw[n].astype("f8") for n in mapping[2]], axis=1
                )
                np.testing.assert_array_equal(expected_value, value[row])
                np.testing.assert_array_equal(expected_error, error[row])
                good = (
                    np.isfinite(expected_value)
                    & np.isfinite(expected_error)
                    & (expected_error > 0)
                    & (expected_value > -999)
                )
                good[:, 0] &= expected_value[:, 0] > 0
                if source == "galah":
                    good &= ((raw["flag_sp"] == 0) & (raw["flag_sp_fit"] == 0))[:, None]
                    good[:, 2] &= raw["flag_fe_h"] == 0
                else:
                    good &= ((raw["ASPCAPFLAG"] == 0) & (raw["STARFLAG"] == 0))[:, None]
                    good[:, 2] &= raw["FE_H_FLAG"] == 0
                    metadata_hdus = [
                        hdu
                        for hdu in hdul
                        if "PARAM_SYMBOL"
                        in (getattr(getattr(hdu, "columns", None), "names", None) or [])
                    ]
                    assert len(metadata_hdus) == 1, (
                        "Expected unique native parameter metadata table"
                    )
                    assert len(metadata_hdus[0].data) == 1
                    symbols = [
                        str(s.decode() if isinstance(s, bytes) else s).strip().upper()
                        for s in np.asarray(
                            metadata_hdus[0].data["PARAM_SYMBOL"][0]
                        ).reshape(-1)
                    ]
                    assert symbols.count("TEFF") == symbols.count("LOGG") == 1
                    for j, symbol in enumerate(["TEFF", "LOGG"]):
                        flag = raw["PARAMFLAG"][:, symbols.index(symbol)].astype("u8")
                        # Published BAD/fixed bits, rebuilt without the producer helper.
                        bad = np.zeros(len(raw), bool)
                        for bit in [0, 1, 2, 3, 4, 6, 16]:
                            bad |= (flag & np.uint64(2**bit)) != 0
                        good[:, j] &= ~bad
                good[:, 2] &= expected_error[:, 2] < 0.5
                np.testing.assert_array_equal(good, observed[row])
                for name, saved in f["raw/" + source].items():
                    if name not in ["feature_row", "source_row"]:
                        assert raw[name].tobytes() == saved[:].tobytes(), (source, name)
            np.testing.assert_array_equal(
                f["targets/logg_parallax_conditioned"][:][row],
                np.full(len(row), source == "galah"),
            )
            counts[source] = {
                "raw_rows": len(row),
                "complete_targets": int(good.all(1).sum()),
            }
        assert not observed[~supported].any()
        assert np.isnan(value[~supported]).all() and np.isnan(error[~supported]).all()
        refs = f["reference_holdout_source_id"][:]
        assert len(refs) == 99574 and len(np.unique(refs)) == len(refs)
        refsha = hashlib.sha256(np.asarray(refs, dtype="<i8").tobytes()).hexdigest()
        assert refsha == provenance["reference_holdout_ordered_id_sha256"]
        held = np.isin(ids, refs)
        np.testing.assert_array_equal(held, f["roles/reference_holdout"][:])
        partition = c["features/benchmark_partition"][:]
        available = c["features/feature_available"][:]
        h = ids.astype("u8") ^ np.uint64(42)
        with np.errstate(over="ignore"):
            h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            h ^= h >> np.uint64(31)
        split = np.where(h % 1000 < 700, 0, np.where(h % 1000 < 850, 1, 2)).astype("u1")
        np.testing.assert_array_equal(split, f["roles/identity_split"][:])
        train = available & (partition == 0) & ~held
        masks = {
            "fit": train & (split == 0),
            "tune": train & (split == 1) & ((h // 1000) % 2 == 0),
            "calibration": train & (split == 1) & ((h // 1000) % 2 == 1),
            "test": available & (split == 2) & ~held,
        }
        for name, mask in masks.items():
            np.testing.assert_array_equal(mask, f["roles/" + name][:])
        np.testing.assert_array_equal(
            masks["fit"] & observed.all(1), f["roles/fit_complete_targets"][:]
        )
        np.testing.assert_array_equal(
            available & held, f["roles/external_reference"][:]
        )
        np.testing.assert_array_equal(partition, f["roles/benchmark_partition"][:])
        assert not f["astrometry/distance_fit_eligible"][:].any()
        p = json.loads(c.attrs["feature_columns_json"]).index("PARALLAX")
        np.testing.assert_array_equal(
            c["features/values"][:, p], f["astrometry/parallax_mas"][:]
        )
        np.testing.assert_array_equal(
            c["features/reported_feature_errors"][:, p],
            f["astrometry/parallax_error_mas"][:],
        )
        pi = c["features/values"][:, p]
        e_pi = c["features/reported_feature_errors"][:, p]
        expected_pi_mask = (
            c["features/feature_observed"][:, p]
            & c["features/error_observed"][:, p]
            & np.isfinite(pi)
            & np.isfinite(e_pi)
            & (e_pi > 0)
        )
        np.testing.assert_array_equal(expected_pi_mask, f["astrometry/observed"][:])
    report = {
        "status": "passed",
        "native_labels_verified": True,
        "all_original_source_target_values_errors_flags_rows_verified": True,
        "all_source_id_roles_and_reference_bans_verified": True,
        "source_counts": counts,
        "sidecar_sha256": digest(sidecar),
        "cohort_sha256": digest(cohort),
        "reference_sorted_id_sha256": refsha,
        "training_launch_ready": False,
        "distance_quality_zero_point_verified": False,
        "galah_logg_parallax_conditioned": True,
    }
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != report:
            raise FileExistsError("Existing independent-verification report conflicts")
    else:
        staged = output.with_suffix(output.suffix + ".building")
        with staged.open("x") as handle:
            handle.write(json.dumps(report, indent=2) + "\n")
        staged.replace(output)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sidecar", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--galah", type=Path, required=True)
    p.add_argument("--apogee", type=Path, required=True)
    p.add_argument("--source-sha-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(
        json.dumps(
            verify(
                a.sidecar,
                a.cohort,
                {"galah": a.galah, "apogee": a.apogee},
                json.loads(a.source_sha_json.read_text()),
                a.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
