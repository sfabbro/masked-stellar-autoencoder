from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import h5py
import numpy as np
import pytest
from gaia_fem_source_join import GROUP, build, sha

BIG = 9007199254740993


def fixtures(tmp):
    side = tmp / "side.h5"
    archive = tmp / "source.npz"
    flags = np.array([b"0" * 41, b"0" * 41, b"0" * 41], dtype="S41")
    flags[1] = b"0" * 27 + b"1" + b"0" * 13
    ids = np.array([BIG + 4, BIG, BIG + 2], dtype="i8")
    with h5py.File(side, "w") as h:
        g = h.create_group(GROUP)
        g["source_id"] = ids
        g["mh_gspspec"] = np.array([-0.2, 0.1, 0.3])
        g["selected_observed_mh"] = [1, 1, 1]
        g["alphafe_gspspec"] = [0.1, 0.2, 0.3]
        g["flags_ascii"] = (
            np.ascontiguousarray(flags[[2, 0, 1]]).view("u1").reshape(-1, 41)
        )
    np.savez(
        archive,
        source_id=np.array([BIG, BIG + 2, BIG + 8], dtype="i8"),
        mh_gspspec=np.array([0.1, 0.3, 0.6]),
        fem_gspspec=np.array([0.2, 0.4, 0.1]),
        fem_gspspec_lower=[0.1, 0.2, 0.0],
        fem_gspspec_upper=[0.3, 0.5, 0.2],
        fem_gspspec_nlines=[1.0, 2.0, 3.0],
        fem_gspspec_linescatter=[0.1, 0.1, 0.2],
        flags_gspspec=flags,
    )
    return side, archive


def produce(side, arc, out):
    return build(
        side,
        arc,
        out,
        expected_sidecar_sha=sha(side),
        expected_archive_sha=sha(arc),
        expected_sidecar_rows=3,
        expected_archive_rows=3,
    )


def test_unordered_match_and_upper_limit_and_unmatched(tmp_path):
    side, arc = fixtures(tmp_path)
    out = tmp_path / "joined.h5"
    result = produce(side, arc, out)
    assert (
        result["counts"]["matched_rows"] == 2
        and result["counts"]["unmatched_rows"] == 1
    )
    with h5py.File(out) as h:
        np.testing.assert_array_equal(h["source_id"][:], [BIG, BIG + 2, BIG + 8])
        np.testing.assert_array_equal(h["sidecar_row"][:], [1, 2, -1])
        np.testing.assert_allclose(h["derived_feh_native"][:], [0.3, 0.7, 0.7])
        np.testing.assert_array_equal(
            h["provisional_derived_feh_candidate"][:], [True, False, False]
        )
        assert h["fe_upper_limit_flag"][1] == ord("1")
        assert "covariance unknown" in h.attrs["dependency"]
    with pytest.raises(FileExistsError):
        produce(side, arc, out)


def test_unmatched_and_flags_mh_conflicts_do_not_go_unnoticed(tmp_path):
    side, arc = fixtures(tmp_path)
    with h5py.File(side, "r+") as h:
        h[GROUP]["mh_gspspec"][1] = 1.0
    with pytest.raises(ValueError, match="M/H differs"):
        produce(side, arc, tmp_path / "bad.h5")
    assert not (tmp_path / "bad.h5").exists()
    side.unlink()
    arc.unlink()
    side, arc = fixtures(tmp_path)
    with h5py.File(side, "r+") as h:
        h[GROUP]["flags_ascii"][1, 27] = ord("2")
    with pytest.raises(ValueError, match="flags differ"):
        produce(side, arc, tmp_path / "bad2.h5")


def test_bad_digest_and_duplicate_ids(tmp_path):
    side, arc = fixtures(tmp_path)
    with pytest.raises(ValueError, match="SHA differs"):
        build(
            side,
            arc,
            tmp_path / "bad.h5",
            expected_sidecar_sha="a" * 64,
            expected_archive_sha=sha(arc),
        )
    with h5py.File(side, "r+") as h:
        h[GROUP]["source_id"][0] = BIG
    with pytest.raises(ValueError, match="unique"):
        produce(side, arc, tmp_path / "bad.h5")
