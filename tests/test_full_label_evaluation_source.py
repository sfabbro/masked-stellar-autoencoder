"""Source association/quality contracts; no models or native training execution."""

import gzip
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_full_label_evaluation as source


def block(ids, columns, values, primary=None):
    values = np.asarray(values, float).reshape(len(ids), len(columns))
    valid = np.isfinite(values)
    return dict(
        ids=np.asarray(ids, "i8"),
        columns=columns,
        value=values,
        error=np.full_like(values, 0.1),
        valid=valid,
        primary=valid if primary is None else np.asarray(primary, bool),
        sources=np.full(len(ids), "source_scale"),
        name="synthetic",
    )


def base():
    n = 6
    ids = np.array([10, 11, 12, 20, 21, 22], "i8")
    value = np.full((n, 6), np.nan)
    value[3:, 2] = [-2.3, -2.2, -1.5]
    observed = np.isfinite(value)
    return dict(
        source_id=ids,
        role=np.array(["fit", "tune", "cal", "test", "test", "test"]),
        value=value,
        error=np.where(observed, 0.1, np.nan),
        observed=observed,
        quality=observed.copy(),
        error_observed=observed.copy(),
        label_source=np.full((n, 6), "original_native"),
    )


def external(ids=()):
    return dict(
        ids=np.asarray(ids, "i8"),
        value=np.full(len(ids), 8.0),
        bounds=np.tile([7.0, 8.0, 9.0], (len(ids), 1)),
    )


def test_uncapped_union_excludes_roles_ancestry_preserves_old_and_external_only():
    original = base()
    incoming = block([10, 11, 12, 20, 21, 30, 31, 32], (3,), np.arange(8) / 10)
    out, _, old, positions, census = source.assemble_labels(
        [incoming], original, np.array([21, 31], "i8"), external([40, 10])
    )
    np.testing.assert_array_equal(out["source_id"], [20, 21, 22, 30, 32, 40])
    assert census["current_fit_tune_cal_excluded"] == 3
    assert census["new_ancestor_feh_ids_excluded"] == 1
    np.testing.assert_array_equal(
        out["original_reference_value"][positions], original["value"][old]
    )
    assert not out["strict_unexposed_star"][1]
    assert not out["quality"][1].any() and not out["diagnostic_eligible"][1].any()
    # External seismic-only objects survive even without any six-label primary reference.
    assert out["external_age_observed"][-1]
    assert not out["observed"][-1].any()
    assert out["quality"][0, 2] and out["quality"][0, 3]
    assert out["expanded_reference"][0, 3]
    assert not out["expanded_reference"][0, 2]


def test_desi_metalpoor_is_separate_diagnostic_and_posterior_error_unknown():
    incoming = block([30, 32], (3,), [0.2, 0.3], [[False], [True]])
    incoming["feh"] = np.array([-2.1, -1.5])
    posterior = block([30], (4, 5), [8.0, 0.8])
    posterior["error"][:] = np.nan
    posterior["bounds"] = np.array([[[7.0, 8.0, 9.0], [0.7, 0.8, 0.9]]])
    out, _, _, _, _ = source.assemble_labels(
        [incoming, posterior], base(), np.array([], "i8"), external()
    )
    index = np.flatnonzero(out["source_id"] == 30)[0]
    assert (
        out["diagnostic_eligible"][index, 3]
        and not out["primary_score_eligible"][index, 3]
    )
    assert (
        out["observed"][index, 4:].all() and not out["error_observed"][index, 4:].any()
    )
    np.testing.assert_array_equal(out["posterior_quantiles"][index, 4], [7.0, 8.0, 9.0])


def test_source_flags_precision_and_native_units():
    age = np.tile([6.0, 7.0, 8.0, 9.0, 10.0], (4, 1))
    distance = np.tile([0.6, 0.7, 0.8, 0.9, 1.0], (4, 1))
    good_age, good_dist = source.posterior_quality(
        age,
        distance,
        np.array(["MSTO", "SGB", "RGB", "MSTO"]),
        np.array(["", "", "", "WARN"]),
        np.array([""] * 4),
        np.array([""] * 4),
    )
    np.testing.assert_array_equal(good_age, [True, True, False, False])
    np.testing.assert_array_equal(good_dist, [True, True, True, False])
    with pytest.raises(ValueError):
        source.exact_ids(np.array([123.0]))
    with pytest.raises(ValueError):
        source.exact_ids(np.array([2**63], "u8"))
    np.testing.assert_array_equal(
        source.first_unique(np.array([3, 2, 3, 0], "i8")), [1, 0]
    )


def test_apokasc_signed_offsets_and_integer_identity(tmp_path):
    huge = 600000000000000001
    rows = []
    for cat, mid, low, high in [("RGB", "8", "-1", "2"), ("RC", "5", "-.5", ".7")]:
        record = dict.fromkeys(source.APOKASC3_COLUMNS, "-9999")
        record.update(GaiaDR3=str(huge + len(rows)), AgeCat=cat)
        suffix = "RGB" if cat == "RGB" else "RC"
        record.update(
            {f"Age{suffix}": mid, f"e_Age{suffix}": low, f"E_Age{suffix}": high}
        )
        rows.append(" ".join(record[k] for k in source.APOKASC3_COLUMNS))
    path = tmp_path / "apokasc.dat.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("\n".join(rows))
    result = source.apokasc_panel(
        {"external_apokasc3": dict(path=str(path), sha256=source.sha(path))}
    )
    np.testing.assert_array_equal(result["ids"], [huge, huge + 1])
    np.testing.assert_allclose(result["bounds"], [[7.0, 8.0, 10.0], [4.5, 5.0, 5.7]])


def test_empty_hdf5_datasets_preserve_layout(tmp_path):
    with h5py.File(tmp_path / "empty.h5", "w") as h:
        source._write_rows(h, "empty", np.empty((0, 6)), np.array([], "i8"))
        source._write_rows(h, "strings", np.array(["alpha", "beta"]), np.array([1, 0]))
        assert h["empty"].shape == (0, 6)
        np.testing.assert_array_equal(h["strings"][:], [b"beta", b"alpha"])


def test_desi_real_magnitudes_ids_and_unresolved_release(tmp_path):
    from astropy.io import fits

    ids = np.array([30, 40], dtype="i8")
    columns = [fits.Column(name="SOURCE_ID", format="K", array=ids)]
    for band in ("G", "BP", "RP"):
        columns.append(
            fits.Column(
                name=f"PHOT_{band}_MEAN_MAG",
                format="E",
                unit="mag",
                array=np.array([14.0, 15.0], "f4"),
            )
        )
    raw = tmp_path / "source.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns, name="GAIA")]
    ).writeto(raw)
    sidecar = tmp_path / "projection.h5"
    with h5py.File(sidecar, "w") as h:
        h["measurements"] = np.array([(30,), (40,)], dtype=[("source_id", "i8")])
    config = dict(
        desi_photometry=dict(path=str(raw), sha256=source.sha(raw)),
        alpha=dict(path=str(sidecar)),
    )
    values, proof = source.desi_photometry_fallback(config, ids)
    np.testing.assert_array_equal(values["values"], [[14.0] * 3, [15.0] * 3])
    assert values["identity_verified"].all()
    assert np.isnan(values["errors"]).all()
    assert values["release"] == "unknown" and proof["unknown_release_diagnostic_only"]
    # Equal row counts alone cannot replace exact IDs.
    with h5py.File(sidecar, "r+") as h:
        h["measurements"][1] = (41,)
    with pytest.raises(AssertionError):
        source.desi_photometry_fallback(config, ids)


def test_model_blind_corner_selection_excludes_exposed_and_keeps_empty_cells():
    labels = dict(
        source_id=np.array([8, 4, 2]),
        value=np.array([[0, 0, -2.8]] * 3),
        primary_score_eligible=np.ones((3, 3), bool),
        strict_unexposed_star=np.array([True, True, False]),
    )
    features = dict(
        raw=np.array([[14.0], [14.0], [14.0]]),
        real_xp_present=np.ones(3, bool),
        real_no_xp=np.zeros(3, bool),
    )
    rows, limits, proof = source.corner_selection(
        labels, features, np.arange(3), ["G"], np.array([15.0])
    )
    np.testing.assert_array_equal(rows, [1])
    assert limits.shape == (1, 6, 2) and proof["model_output_used"] is False
    assert sum(c["eligible"] == 0 for c in proof["cells"]) == 11
