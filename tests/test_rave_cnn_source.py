from __future__ import annotations

import sys
from pathlib import Path as _ScriptPath

sys.path.insert(0, str(_ScriptPath(__file__).resolve().parents[1] / "scripts"))

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits
from rave_cnn_source import ID_NULL, TARGETS, build_rave_cnn_source, sha256_file

BIG = 9007199254740993


def fixtures(tmp: Path):
    keys = np.array(
        ["A", "B", "C", "D", "E", "E", "F", "", "G", "H", " H", "a"], dtype="S32"
    )
    cross_keys = np.array(
        ["A", "B", "C", "C", "D", "D", "E", "G", "H", "", "a "], dtype="S32"
    )
    dr2 = np.arange(len(keys), dtype=np.int64) + BIG + 100
    dr2[0] = ID_NULL
    ids = np.array(
        [
            BIG,
            ID_NULL,
            BIG + 2,
            BIG + 4,
            BIG + 6,
            BIG + 6,
            BIG + 8,
            BIG + 10,
            BIG + 10,
            BIG + 12,
            BIG + 14,
        ],
        dtype=np.int64,
    )
    columns = [
        fits.Column(name="rave_obs_id", format="32A", array=keys),
        fits.Column(name="source_id", format="K", null=ID_NULL, array=dr2),
        fits.Column(
            name="snr_cnn", format="E", array=np.full(len(keys), 50, dtype=np.float32)
        ),
    ]
    for index, target in enumerate(TARGETS):
        value = np.full(len(keys), index + 0.25, dtype=np.float32)
        if target == "teff":
            value[:] = 5100.0
            value[8] = -9999.0
        if target == "fe_h":
            value[8] = -9999.0
        if target == "si_fe":
            value[8] = np.nan
        error = np.full(len(keys), 0.2, dtype=np.float32)
        if target == "fe_h":
            error[0] = 0.0
        if target == "alpha_m":
            error[0] = np.nan
        flag = np.zeros(len(keys), dtype=np.float32)
        if target == "logg":
            flag[0] = 1.0
        if target == "ni_fe":
            flag[8] = np.nan
        columns.extend(
            [
                fits.Column(name=f"{target}_cnn", format="E", array=value),
                fits.Column(name=f"{target}_error_cnn", format="E", array=error),
                fits.Column(name=f"{target}_flag_cnn", format="E", array=flag),
            ]
        )
    cnn = tmp / "rave_dr6_cnn.fits"
    cross = tmp / "rave_dr6_x_gaiaedr3.fits"
    fits.BinTableHDU.from_columns(columns).writeto(cnn)
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="rave_obs_id", format="32A", array=cross_keys),
            fits.Column(name="source_id", format="K", null=ID_NULL, array=ids),
            fits.Column(
                name="dist", format="D", array=np.arange(len(ids), dtype=np.float64)
            ),
        ]
    ).writeto(cross)
    return cnn, cross


def produce(cnn: Path, cross: Path, out: Path, **kwargs):
    return build_rave_cnn_source(
        cnn,
        cross,
        out,
        expected_cnn_sha256=sha256_file(cnn),
        expected_xmatch_sha256=sha256_file(cross),
        **kwargs,
    )


def test_raw_rows_identity_selection_and_fits_bytes(tmp_path):
    cnn, cross = fixtures(tmp_path)
    output = tmp_path / "out.h5"
    result = produce(cnn, cross, output, expected_cnn_rows=12, expected_xmatch_rows=11)
    assert result["output_sha256"] == sha256_file(output)
    assert result["counts"]["cnn_rows"] == 12
    assert result["counts"]["crossmatch_rows"] == 11
    assert result["counts"]["identity_candidate_rows"] == 4
    assert result["counts"]["identity_candidate_distinct_stars"] == 3
    with h5py.File(output, "r") as f:
        c, x = f["cnn"], f["crossmatch"]
        assert c["rave_obs_id"].shape == (12,)
        assert x["rave_obs_id"].shape == (11,)
        np.testing.assert_array_equal(c["source_row"][:], np.arange(12))
        np.testing.assert_array_equal(x["source_row"][:], np.arange(11))
        assert int(c["gaia_dr2_source_id_raw"][0]) == ID_NULL
        assert not c["gaia_dr2_id_known"][0]
        assert int(c["gaia_edr3_dr3_source_id"][0]) == BIG
        assert int(x["gaia_edr3_source_id_raw"][1]) == ID_NULL
        assert not x["gaia_edr3_id_known"][1]
        assert int(c["gaia_edr3_dr3_source_id"][1]) == -1
        assert list(c["crossmatch_count"][:]) == [1, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 1]
        assert list(c["identity_candidate"][:]) == [
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            True,
        ]
        assert int(c["crossmatch_row"][2]) == -1
        assert (
            int(c["crossmatch_row"][3]) == -1
        )  # duplicate rows ambiguous, even same Gaia ID
        assert int(c["cnn_observation_count"][4]) == 2
        assert list(c["mapped_distinct_observations_for_star"][[8, 9]]) == [2, 2]
        metadata = json.loads(f.attrs["metadata_json"])
        assert metadata["sources"]["cnn"]["sha256"] == sha256_file(cnn)
        assert metadata["sources"]["crossmatch"]["sha256"] == sha256_file(cross)
        assert "APOGEE DR16" in metadata["dependency"]
        for group, path in ((c, cnn), (x, cross)):
            offset = int(group.attrs["raw_fits_data_offset"])
            nbytes = group["raw_fits_table_and_heap"].size
            with path.open("rb") as inp:
                inp.seek(offset)
                assert bytes(group["raw_fits_table_and_heap"][:]) == inp.read(nbytes)
            assert "XTENSION" in group.attrs["raw_fits_header"]


def test_per_target_masks_preserve_unknown_errors_and_scales(tmp_path):
    cnn, cross = fixtures(tmp_path)
    output = tmp_path / "out.h5"
    produce(cnn, cross, output)
    with h5py.File(output, "r") as f:
        c = f["cnn"]
        fe, alpha, logg, teff, si, ni = (
            TARGETS.index(x)
            for x in ("fe_h", "alpha_m", "logg", "teff", "si_fe", "ni_fe")
        )
        assert c["raw_value_available"][0, fe]
        assert c["provisional_target_candidate"][0, fe]
        assert c["error"][0, fe] == 0.0
        assert not c["positive_finite_error_known"][0, fe]
        assert np.isnan(c["error"][0, alpha])
        assert not c["positive_finite_error_known"][0, alpha]
        assert c["provisional_target_candidate"][0, alpha]
        assert not c["boundary_flag_in_domain"][0, logg]
        assert not c["provisional_target_candidate"][0, logg]
        assert not c["raw_value_available"][8, teff]
        assert not c["raw_value_available"][8, fe]
        assert not c["raw_value_available"][8, si]
        assert c["provisional_target_candidate"][8, logg]
        assert not c["boundary_flag_in_domain"][8, ni]
        assert c["value"][0, alpha] != c["value"][0, fe]
        assert c["value"][0, TARGETS.index("m_h")] != c["value"][0, fe]
        assert json.loads(f.attrs["metadata_json"])["target_order"] == list(TARGETS)


def test_fail_closed_bad_sha_row_count_and_no_overwrite(tmp_path):
    cnn, cross = fixtures(tmp_path)
    out = tmp_path / "out.h5"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_rave_cnn_source(
            cnn,
            cross,
            out,
            expected_cnn_sha256="0" * 64,
            expected_xmatch_sha256=sha256_file(cross),
        )
    assert not out.exists()
    with pytest.raises(ValueError, match="explicit lowercase"):
        build_rave_cnn_source(
            cnn,
            cross,
            out,
            expected_cnn_sha256="",
            expected_xmatch_sha256=sha256_file(cross),
        )
    with pytest.raises(ValueError, match="unexpected CNN row count"):
        produce(cnn, cross, out, expected_cnn_rows=999)
    assert not out.exists()
    produce(cnn, cross, out)
    digest = sha256_file(out)
    with pytest.raises(FileExistsError):
        produce(cnn, cross, out)
    assert sha256_file(out) == digest


def test_fail_closed_wrong_null_and_missing_field(tmp_path):
    cnn, cross = fixtures(tmp_path)
    with fits.open(cross, mode="update") as file:
        ix = file[1].columns.names.index("source_id") + 1
        file[1].header[f"TNULL{ix}"] = -1
    with pytest.raises(ValueError, match="positive int64 TNULL"):
        produce(cnn, cross, tmp_path / "bad_null.h5")
    assert not (tmp_path / "bad_null.h5").exists()
    with fits.open(cnn) as hdus:
        columns = fits.ColDefs(
            [col for col in hdus[1].columns if col.name != "alpha_m_cnn"]
        )
        fits.BinTableHDU.from_columns(columns).writeto(tmp_path / "missing.fits")
    cnn2, cross2 = (
        fixtures(tmp_path / "fresh") if False else (tmp_path / "missing.fits", cross)
    )
    with pytest.raises(ValueError, match="alpha_m_cnn"):
        produce(cnn2, cross2, tmp_path / "bad_missing.h5")
    assert not (tmp_path / "bad_missing.h5").exists()


def test_reject_float_source_id_even_if_it_looks_integral(tmp_path):
    cnn, cross = fixtures(tmp_path)
    with fits.open(cross) as hdus:
        columns = []
        for col in hdus[1].columns:
            if col.name == "source_id":
                columns.append(
                    fits.Column(
                        name="source_id",
                        format="D",
                        array=np.asarray(hdus[1].data["source_id"], dtype=np.float64),
                    )
                )
            else:
                columns.append(col)
        fits.BinTableHDU.from_columns(columns).writeto(tmp_path / "float.fits")
    with pytest.raises(ValueError, match="signed FITS K"):
        produce(cnn, tmp_path / "float.fits", tmp_path / "bad_float.h5")
    assert not (tmp_path / "bad_float.h5").exists()
