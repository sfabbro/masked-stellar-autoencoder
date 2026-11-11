import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_gaia_unlabelled_pool.py"
spec = importlib.util.spec_from_file_location("pool", SCRIPT)
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


def test_hash_sample_is_deterministic_and_close_to_rate():
    ids = np.arange(1, 400001, dtype="i8") * 7919
    a = pool.in_sample(ids, 0.125)
    assert np.array_equal(a, pool.in_sample(ids, 0.125))
    assert abs(a.mean() - 0.125) < 0.005
    assert pool.in_sample(ids, 1.0).all()
    assert (pool.in_sample(ids, 0.5) | ~a).all()  # nested: rate 0.125 subset of 0.5


def test_select_rows_keeps_sampled_no_xp_with_g_and_drops_heldout():
    sid = np.array([10, 20, 30, 40], dtype="i8")
    rows = pool.select_rows(
        sid,
        [False, True, False, False],
        [15.0, 15.0, np.nan, 18.0],
        1.0,
        np.array([40]),
    )
    assert rows.tolist() == [0]


def test_pool_build_end_to_end(tmp_path):
    gaia = tmp_path / "gaia"
    gaia.mkdir()
    rng = np.random.default_rng(0)
    for k in range(2):
        n = 1000
        with h5py.File(gaia / f"GaiaSource_{k:06d}-{k:06d}.hdf5", "w") as f:
            f["source_id"] = np.arange(k * n + 1, (k + 1) * n + 1, dtype="i8")
            f["has_xp_continuous"] = rng.random(n) < 0.2
            f["phot_g_mean_mag"] = rng.uniform(15, 21, n)
            bp = rng.uniform(15, 22, n)
            bp[::5] = np.nan
            f["phot_bp_mean_mag"] = bp
            f["phot_rp_mean_mag"] = rng.uniform(14, 20, n)
            for c in ("parallax", "pmra", "pmdec", "ra", "dec"):
                f[c] = rng.normal(size=n)
    out = tmp_path / "pool"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gaia-root",
            str(gaia),
            "--rate",
            "0.5",
            "--output",
            str(out),
        ],
        check=True,
    )
    m = json.loads((out / "manifest.json").read_text())
    assert m["status"] == "complete" and m["scanned"] == 2000
    ids = np.load(out / "source_id.npy")
    vals = np.load(out / "values.npy")
    assert vals.shape == (m["stars"], 8) and len(ids) == m["stars"]
    assert 0.3 < m["stars"] / m["no_xp_seen"] < 0.7
    assert np.isfinite(vals[:, 0]).all()  # G always present
    assert 0.7 < m["visible_fraction"]["BP"] < 0.9
