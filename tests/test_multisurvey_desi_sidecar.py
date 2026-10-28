"""Synthetic immutable FITS only: no real catalogue reads or downloads."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from masked_stellar_autoencoder.training.multisurvey_overlay import file_sha256

spec = importlib.util.spec_from_file_location(
    "desi_sidecar_adapter",
    Path(__file__).parents[1] / "scripts/build_multisurvey_desi_sidecar.py",
)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def fixture(
    tmp_path, *, gaia_targetid=False, mismatch=None, missing_feh=False, no_good=False
):
    n = 11
    rv = np.zeros(
        n,
        dtype=[
            ("TARGETID", "i8"),
            ("TARGET_RA", "f8"),
            ("TARGET_DEC", "f8"),
            ("TEFF", "f8"),
            ("LOGG", "f8"),
            ("MH" if missing_feh else "FEH", "f8"),
            ("TEFF_ERR", "f8"),
            ("LOGG_ERR", "f8"),
            ("FEH_ERR", "f8"),
            ("SUCCESS", "?"),
            ("PRIMARY", "?"),
            ("RVS_WARN", "i8"),
            ("RR_SPECTYPE", "S6"),
            ("SN_R", "f8"),
        ],
    )
    rv["TARGETID"] = np.arange(n, dtype="i8") + 2**53 + 1
    rv["TARGET_RA"], rv["TARGET_DEC"] = 30.0, 5.0
    rv["TEFF"], rv["LOGG"], rv["MH" if missing_feh else "FEH"] = 3000.0, 0.5, -2.5
    rv["TEFF_ERR"], rv["LOGG_ERR"], rv["FEH_ERR"] = 40.0, 0.2, 0.8
    rv["LOGG_ERR"][0], rv["FEH_ERR"][0] = np.nan, 0
    rv["TEFF_ERR"][1], rv["FEH_ERR"][1] = 0, np.nan
    rv["TEFF_ERR"][2], rv["LOGG_ERR"][2] = np.nan, 0
    rv["LOGG"][0], rv["MH" if missing_feh else "FEH"][1] = np.nan, -9999
    for name in ("TEFF_ERR", "LOGG_ERR", "FEH_ERR"):
        rv[name][3] = np.nan
    rv["SUCCESS"] = rv["PRIMARY"] = True
    rv["RR_SPECTYPE"] = b"STAR"
    rv["SN_R"] = 20
    rv["RVS_WARN"][5] = 1
    rv["SN_R"][6] = 9
    rv["RR_SPECTYPE"][7] = b"QSO"
    rv["SUCCESS"][8] = False
    rv["PRIMARY"][10] = False
    if no_good:
        rv["RVS_WARN"] = 1
    dtype = [("SOURCE_ID", "i8"), ("RA", "f8"), ("DEC", "f8")]
    if gaia_targetid:
        dtype += [("TARGETID", "i8")]
    gaia = np.zeros(n, dtype=dtype)
    gaia["SOURCE_ID"] = 6000000000000000001 + np.arange(n, dtype="i8")
    gaia["SOURCE_ID"][4] = gaia["SOURCE_ID"][0]
    gaia["RA"], gaia["DEC"] = 30.0, 5.0
    gaia["RA"][9] += 2 / 3600
    if gaia_targetid:
        gaia["TARGETID"] = rv["TARGETID"]
        if mismatch == "GAIA":
            gaia["TARGETID"][4] += 1
    fiber = np.zeros(n, dtype=[("TARGETID", "i8")])
    fiber["TARGETID"] = rv["TARGETID"]
    sp = fiber.copy()
    if mismatch == "FIBERMAP":
        fiber["TARGETID"][4] += 1
    if mismatch == "SPTAB":
        sp["TARGETID"][4] += 1
    source = tmp_path / "mwsall-pix-iron.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(rv, name="RVTAB"),
            fits.BinTableHDU(gaia, name="GAIA"),
            fits.BinTableHDU(fiber, name="FIBERMAP"),
            fits.BinTableHDU(sp, name="SPTAB"),
        ]
    ).writeto(source)
    digest = file_sha256(source)
    publisher = tmp_path / "desi-publisher.sha256sum"
    publisher.write_text(digest + "  mwsall-pix-iron.fits\n")
    return dict(
        raw_path=source, publisher_path=publisher, expected_sha=digest, chunk_rows=2
    )


def test_partial_targets_raw_feh_duplicate_and_exact_ids_preserved(tmp_path):
    kwargs = fixture(tmp_path)
    out = tmp_path / "raw-sidecar.h5"
    result = adapter.build_desi_sidecar(out, **kwargs)
    assert result["quality_rows"] == 5
    assert result["observed_counts"] == [5, 4, 4]
    assert result["sha256"] == file_sha256(out)
    with h5py.File(out) as h5:
        group = h5["native_observations/" + adapter.SOURCE_NAME]
        np.testing.assert_array_equal(group["source_row"][:], np.arange(5))
        np.testing.assert_array_equal(
            group["target_id"][:], 2**53 + 1 + np.arange(5, dtype="i8")
        )
        assert group["source_id"][0] == group["source_id"][4] == 6000000000000000001
        assert group["source_id"].dtype.kind == "i"
        assert group["observed"][:].tolist() == [
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        np.testing.assert_array_equal(
            group["value"][:, 2], [-2.5, -9999, -2.5, -2.5, -2.5]
        )
        assert group["error"][2, 2] == 0.8  # No former <0.5 or correction-domain cut.
        assert group["value"][2, 0] == 3000 and group["value"][2, 1] == 0.5
        assert np.isnan(group["error"][3]).all()
        assert np.isnan(group["error"][0, 2]) and group["raw_error"][0, 2] == 0
        assert not group["rvs_warn"][:].any()
        meta = json.loads(group.attrs["registry_json"])
        assert meta["targets"]["fe_h"]["field"] == "FEH"
        provenance = json.loads(h5.attrs["provenance_json"])
        assert "GAIA has no TARGETID" in provenance["alignment"]
        assert "O(N)" in provenance["memory_scope"]
        assert "input" not in h5
    assert file_sha256(kwargs["raw_path"]) == kwargs["expected_sha"]
    with pytest.raises(FileExistsError):
        adapter.build_desi_sidecar(out, **kwargs)


@pytest.mark.parametrize("extension", ["GAIA", "FIBERMAP", "SPTAB"])
def test_exact_alignment_failure_in_later_chunk(tmp_path, extension):
    kwargs = fixture(tmp_path, gaia_targetid=True, mismatch=extension)
    with pytest.raises(ValueError, match="alignment"):
        adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    assert not (tmp_path / "out.h5").exists()


def test_exact_gaia_targetid_receipt_when_present(tmp_path):
    kwargs = fixture(tmp_path, gaia_targetid=True)
    adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    with h5py.File(tmp_path / "out.h5") as h5:
        assert (
            "exact GAIA TARGETID"
            in json.loads(h5.attrs["provenance_json"])["alignment"]
        )


def test_missing_feh_never_replaced_by_mh(tmp_path):
    kwargs = fixture(tmp_path, missing_feh=True)
    with pytest.raises(ValueError, match="Missing native RVS field: FEH"):
        adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)


def test_empty_quality_subset_is_written_without_fabricated_labels(tmp_path):
    kwargs = fixture(tmp_path, no_good=True)
    result = adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    assert result["quality_rows"] == 0 and result["observed_counts"] == [0, 0, 0]
    with h5py.File(tmp_path / "out.h5") as h5:
        assert h5["native_observations/" + adapter.SOURCE_NAME + "/value"].shape == (
            0,
            3,
        )


def test_publisher_and_file_digest_fail_before_output(tmp_path):
    kwargs = fixture(tmp_path)
    kwargs["publisher_path"].write_text("0" * 64 + "  x\n")
    with pytest.raises(ValueError, match="Publisher"):
        adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    kwargs["publisher_path"].write_text(
        kwargs["expected_sha"] + "  mwsall-pix-iron.fits\n"
    )
    with kwargs["raw_path"].open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="differs from publisher"):
        adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    assert not (tmp_path / "out.h5").exists()


def test_source_change_during_checksum_is_rejected(tmp_path, monkeypatch):
    kwargs = fixture(tmp_path)
    real_hash = adapter.file_sha256

    def changed(path):
        digest = real_hash(path)
        with Path(path).open("ab") as stream:
            stream.write(b"changed")
        return digest

    monkeypatch.setattr(adapter, "file_sha256", changed)
    with pytest.raises(ValueError, match="during checksum"):
        adapter.build_desi_sidecar(tmp_path / "out.h5", **kwargs)
    assert not (tmp_path / "out.h5").exists()
