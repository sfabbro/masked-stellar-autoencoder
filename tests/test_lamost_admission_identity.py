import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "lamost_admission",
    Path(__file__).parents[1] / "scripts/admit_lamost_and_audit_inputs.py",
)
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


def test_exact_native_ids_and_missing_values():
    raw = np.array([b"1152921504606846977", b"", b"-9999", b"9223372036854775808"])
    ids, valid = admission.integer_gaia_ids(raw)
    assert ids[0] == 2**60 + 1
    assert valid.tolist() == [True, False, False, False]
    with pytest.raises(ValueError):
        admission.integer_gaia_ids(np.array([float(2**60 + 1)]))


def test_dr12_admission_keeps_release_and_exact_identity(tmp_path, monkeypatch):
    import json
    import sys

    import h5py
    from astropy.table import Table

    sid = 2**60 + 1
    native = tmp_path / "dr12.fits"
    Table(
        dict(
            gaia_source_id=np.array([sid, sid], dtype="i8"),
            snrg=[35.0, 50.0],
            teff=[5000.0, 5100.0],
            teff_err=[60.0, 50.0],
            logg=[4.0, 4.1],
            logg_err=[0.1, 0.1],
            feh=[-0.2, -0.1],
            feh_err=[0.1, 0.1],
        )
    ).write(native)
    base = tmp_path / "base.h5"
    with h5py.File(base, "w") as f:
        f["source_id"] = np.array([sid], dtype="i8")
        f["role"] = np.array([b"external"])
        f["primary_input_eligible"] = [True]
        f["input"] = [[1.0, 2.0]]
    benchmark = tmp_path / "bench.json"
    benchmark.write_text(json.dumps([sid]))
    output = tmp_path / "candidate"
    monkeypatch.setattr(admission, "BASE", str(base))
    monkeypatch.setitem(admission.RELEASE_ROWS, "dr12v2", 2)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admit",
            "--release",
            "dr12v2",
            "--source",
            str(native),
            "--benchmarks",
            str(benchmark),
            "--output",
            str(output),
        ],
    )
    admission.main()
    with h5py.File(output / "lamost-dr12v2-lrs-candidate.h5") as f:
        group = f["native_observations/lamost_dr12v2_lrs_lasp"]
        assert group["source_id"][:].tolist() == [sid]
        assert group["value"][0, 0] == 5100.0
        assert group["benchmark_reserved"][:].tolist() == [True]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["raw_rows"] == 2
    assert manifest["selected_unique_stars"] == 1
    assert manifest["role_counts"]["external"]["stars"] == 1
    assert not manifest["training_performed"]
