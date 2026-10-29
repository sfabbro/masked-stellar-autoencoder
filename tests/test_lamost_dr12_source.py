import gzip
import hashlib
import sys

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, "scripts")
import lamost_dr12_source as mod


def test_exact_ids_and_unknowns():
    a = np.array(
        [
            b"9007199254740993",
            b"  9007199254740995 ",
            b"-9999",
            b"",
            b"9223372036854775808",
            b"123.0",
            b"0",
        ],
        dtype="S25",
    )
    assert mod.exact_gaia_ids(a).tolist() == [
        9007199254740993,
        9007199254740995,
        0,
        0,
        0,
        0,
        0,
    ]
    with pytest.raises(ValueError):
        mod.exact_gaia_ids(np.array([1.0]))


def test_end_to_end_exact_ids_roles_raw_targets_and_cache(tmp_path, monkeypatch):
    names = [
        "teff",
        "logg",
        "feh",
        "alpha_m",
        "teff_err",
        "logg_err",
        "feh_err",
        "alpha_m_err",
    ]
    cols = [
        fits.Column(
            name="gaia_source_id",
            format="25A",
            array=["9007199254740993", "9007199254740995", "", "9007199254740993"],
        ),
        fits.Column(name="obsid", format="K", array=[11, 12, 13, 14]),
    ]
    for i, n in enumerate(names):
        cols.append(
            fits.Column(
                name=n,
                format="D",
                array=np.array([100.0 + i, -9999.0, np.nan, 101.0 + i]),
            )
        )
    raw = tmp_path / "input.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(raw)
    archive = tmp_path / "input.fits.gz"
    with raw.open("rb") as a, gzip.open(archive, "wb") as b:
        b.write(a.read())
    base = tmp_path / "base.h5"
    cache = tmp_path / "cache"
    cache.mkdir()
    original = tmp_path / "original.h5"
    original.touch()
    for n in ["manifest.json", "features.npy", "source_id.npy", "provenance.npy"]:
        (cache / n).touch()
    all_names = [f"{a}_{i}" for a in ("bp", "rp") for i in range(1, 56)]
    with h5py.File(base, "w") as h:
        h["source_id"] = np.array([9007199254740993, 9007199254740997], dtype="<i8")
        h["role"] = np.array(["fit", "tune"], dtype="S16")
        h["primary_input_eligible"] = np.array([True, True])
        h["known_supervised_ancestor_exposed"] = np.array([False, True])
        h["reserved_reference"] = np.array([False, False])
        h["feature_names"] = np.array(all_names, dtype="S8")
        h["scaler_center"] = np.zeros(110)
        h["scaler_scale"] = np.ones(110)
    with base.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    p = {**mod.PRODUCTS["lrs"], "rows": 4}
    monkeypatch.setitem(mod.PRODUCTS, "lrs", p)
    old = mod.tempfile.TemporaryDirectory
    monkeypatch.setattr(
        mod.tempfile, "TemporaryDirectory", lambda **kw: old(dir=tmp_path)
    )

    class FakeIDs:
        offset = 0

    monkeypatch.setattr(
        mod,
        "_cache_maps",
        lambda *a: (
            np.zeros((1, 110)),
            FakeIDs(),
            np.zeros(1),
            {"files": {"source_id.npy": {"sha256": "hash"}}, "request": {}},
        ),
    )
    monkeypatch.setattr(
        mod,
        "scan_cache",
        lambda ids, *a, **kw: (np.array([3, -1]), np.array([True, False]), "hash", 0),
    )
    monkeypatch.setattr(mod, "verify_cache_associations", lambda *a, **kw: 1)
    out = tmp_path / "joined.h5"
    result = mod.produce(
        product="lrs",
        archive=archive,
        base=base,
        cache=cache,
        original=original,
        base_sha=sha,
        output=out,
        expected_cache_rows=1,
    )
    assert (
        result["source_rows"] == 4
        and result["rows_with_valid_gaia_ids"] == 3
        and result["unique_gaia_ids"] == 2
        and result["matched_base"] == 1
        and result["complete_xp_in_cache"] == 1
    )
    with h5py.File(out, "r") as h:
        assert h["stars/source_id"][:].tolist() == [9007199254740993, 9007199254740995]
        assert h["stars/base_role"][:].tolist() == [b"fit", b"not_in_base"]
        assert h["stars/cache_xp_state"][:].tolist() == [1, -1]
        assert h["observations/source_id"][:].tolist() == [
            9007199254740993,
            9007199254740995,
            0,
            9007199254740993,
        ]
        assert h["observations/lasp/raw_observed"][:].tolist() == [
            [True] * 4,
            [False] * 4,
            [False] * 4,
            [True] * 4,
        ]
    with pytest.raises(FileExistsError):
        mod.produce(
            product="lrs",
            archive=archive,
            base=base,
            cache=cache,
            original=original,
            base_sha=sha,
            output=out,
        )
