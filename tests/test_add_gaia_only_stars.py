import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/add_gaia_only_stars.py"


def test_unrecovered_candidates_become_no_xp_stars_from_gaia(tmp_path):
    names = ["W1", "W2", "G", "BP", "RP"]
    names += [f"{a}_{j}" for a in ("bp", "rp") for j in range(1, 56)] + ["PARALLAX"]
    run = tmp_path / "root" / "feat"
    run.mkdir(parents=True)
    ids = np.array([10, 20, 30, 40], dtype="i8")  # 10 recovered (XP); 20,30,40 not
    x = np.full((4, len(names)), -9999.0, dtype="f4")
    x[0, :] = 0.1
    path = run / "additional-star-features.h5"
    with h5py.File(path, "w") as f:
        f["source_id"] = ids
        f["feature_names"] = np.array(names, dtype="S8")
        f["scaler_center"] = np.full(len(names), 10.0)
        f["scaler_scale"] = np.full(len(names), 2.0)
        f["input"] = x
        f["visible"] = x != -9999
        f["feature_recoverable"] = [True, False, False, False]
        f["input_has_xp"] = [True, False, False, False]
        f["real_xp_present"] = [True, False, False, False]
        f["real_no_xp"] = np.zeros(4, bool)
        f["original_row"] = np.array([5, -1, -1, -1])
        g = f.create_group("measured_parallax")
        g["parallax_mas"] = np.array([1.0, np.nan, np.nan, np.nan])
        g["parallax_error_mas"] = np.array([0.1, np.nan, np.nan, np.nan])
        g["observed"] = np.array([True, False, False, False])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(
        json.dumps(
            dict(status="additional_full_raw_features_complete", output_sha256=sha)
        )
    )
    gaia = tmp_path / "gaia.parquet"
    pd.DataFrame(
        dict(
            source_id=[20, 30, 40],
            ra=[1.0, 2.0, 3.0],
            dec=[0.0, 0.0, 0.0],
            parallax=[0.5, np.nan, -0.2],
            parallax_error=[0.1, np.nan, 0.3],
            pmra=[1.0, np.nan, 2.0],
            pmdec=[1.0, np.nan, 2.0],
            pmra_error=[0.1, np.nan, 0.2],
            pmdec_error=[0.1, np.nan, 0.2],
            phot_g_mean_mag=[18.0, 19.0, 20.0],
            phot_bp_mean_mag=[18.5, np.nan, np.nan],
            phot_rp_mean_mag=[17.5, np.nan, 19.5],
            has_xp_continuous=[False, False, True],
            ruwe=[1.0, 1.0, 1.0],
        )
    ).to_parquet(gaia)
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path / "root"),
            "--feature-run",
            "feat",
            "--gaia-parquet",
            str(gaia),
            "--output",
            str(out),
        ],
        check=True,
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "additional_full_raw_features_complete"
    assert m["gaia_only_recoverable"] == 2  # 30 has G only -> excluded
    assert m["input_recovered"] == 3 and m["input_with_xp"] == 1
    with h5py.File(out / "additional-star-features.h5") as f:
        assert f["feature_recoverable"][:].tolist() == [True, True, False, True]
        assert f["input_has_xp"][:].tolist() == [True, False, False, False]
        assert f["real_no_xp"][:].tolist() == [False, True, False, False]
        g = names.index("G")
        assert f["input"][1, g] == 4.0  # (18 - 10) / 2
        assert (f["input"][1, 5:115] == -9999).all()  # XP missing
        assert np.array_equal(f["input"][0], x[0])  # recovered row untouched
        assert f["measured_parallax/observed"][:].tolist() == [True, True, False, True]
        assert f["measured_parallax/parallax_mas"][3] == -0.2  # signed kept
    assert (
        hashlib.sha256((out / "additional-star-features.h5").read_bytes()).hexdigest()
        == m["output_sha256"]
    )
    assert (out / "gaia-only" / "000000000.parquet").exists()
