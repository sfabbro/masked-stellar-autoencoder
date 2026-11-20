"""Covariance sidecar: packing, covariance algebra, L21 zero point, lookup, build."""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import healpy
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits

from masked_stellar_autoencoder.data import covariance_sidecar as cs

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "build_covariance_sidecar", SCRIPTS / "build_covariance_sidecar.py"
)
bcs = importlib.util.module_from_spec(spec)
sys.modules["build_covariance_sidecar"] = bcs
spec.loader.exec_module(bcs)


def _random_correlation(rng, n=cs.N_COEFF):
    a = rng.normal(size=(n, n + 5))
    c = a @ a.T
    d = np.sqrt(np.diag(c))
    return c / d[:, None] / d[None, :]


def test_unpack_follows_the_gaia_datamodel_column_major_upper_triangle():
    rng = np.random.default_rng(0)
    packed = rng.uniform(-1, 1, cs.N_CORR).astype(np.float32)
    # Independent construction from the DR3 datamodel text: M[i, j] = C[j(j-1)/2 + i].
    expected = np.eye(cs.N_COEFF)
    for j in range(cs.N_COEFF):
        for i in range(j):
            expected[i, j] = expected[j, i] = packed[j * (j - 1) // 2 + i]
    m = cs.unpack_correlations(packed)
    assert m.dtype == np.float32
    np.testing.assert_array_equal(m, expected.astype(np.float32))
    np.testing.assert_array_equal(cs.pack_correlations(m), packed)
    # datamodel's displayed first elements: M[0,1]=C0, M[0,2]=C1, M[1,2]=C2, M[0,3]=C3
    assert (m[0, 1], m[0, 2], m[1, 2], m[0, 3]) == tuple(packed[:4])
    batch = cs.unpack_correlations(np.stack([packed, -packed]))
    assert batch.shape == (2, 55, 55)
    with pytest.raises(ValueError):
        cs.unpack_correlations(packed[:-1])


def test_xp_covariance_reproduces_errors_and_is_positive_definite():
    rng = np.random.default_rng(1)
    r = _random_correlation(rng)
    e = rng.uniform(0.01, 3.0, cs.N_COEFF)
    s = 1.3
    cov = cs.xp_covariance(e, cs.pack_correlations(r), s)
    np.testing.assert_allclose(np.diag(cov), e**2, rtol=1e-12)
    np.testing.assert_array_equal(cov, cov.T)
    np.linalg.cholesky(cov)
    np.testing.assert_allclose(cs.xp_covariance(e, cs.pack_correlations(r), s, formal=True),
                               cov / s**2, rtol=1e-12)  # fmt: skip
    with pytest.raises(ValueError):
        cs.xp_covariance(e, cs.pack_correlations(r), formal=True)


def test_table_units_and_rank_one_g_flux_term():
    rng = np.random.default_rng(2)
    r = _random_correlation(rng)
    e = rng.uniform(0.01, 3.0, cs.N_COEFF)
    cov = cs.xp_covariance(e, cs.pack_correlations(r))
    assert cs.xp_normalisation_factor(8.5) == 1.0
    np.testing.assert_allclose(cs.xp_normalisation_factor(13.5), 1e-2)
    np.testing.assert_allclose(cs.to_table_units(cov, 13.5), cov * 1e4)
    x = rng.normal(size=cs.N_COEFF)
    full = cs.to_table_units(cov, 13.5, coefficients_table=x, g_flux_over_error=200.0)
    np.testing.assert_allclose(full - cov * 1e4, np.outer(x, x) / 200.0**2, atol=1e-9)
    # Monte Carlo of x = c / f with a noisy G flux reproduces the rank-one term.
    f_true, rel = 1e-2, 1 / 200.0
    c = x * f_true
    flux = 1 + rel * rng.normal(size=200_000)
    sample = c[None, :3] / (f_true * flux[:, None])
    np.testing.assert_allclose(
        np.cov(sample.T), np.outer(x[:3], x[:3]) * rel**2, rtol=0.05, atol=1e-7
    )
    joint = cs.joint_xp_covariance(cov, 2 * cov)
    assert joint.shape == (110, 110) and not joint[:55, 55:].any()


def test_astrometric_covariance_matches_explicit_construction():
    e = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    corr = np.linspace(-0.5, 0.4, 10)
    cov = cs.astrometric_covariance(e, corr)
    names = cs.ASTROMETRY
    lookup = dict(zip(cs.ASTROMETRY_CORRELATIONS, corr, strict=True))
    for i in range(5):
        for j in range(5):
            rho = (
                1.0
                if i == j
                else lookup.get(f"{names[min(i, j)]}_{names[max(i, j)]}_corr")
            )
            assert cov[i, j] == pytest.approx(e[i] * e[j] * rho)
    assert cs.ASTROMETRY_CORRELATIONS[0] == "ra_dec_corr"
    assert cs.ASTROMETRY_CORRELATIONS[-1] == "pmra_pmdec_corr"


# (G, nu_eff, pseudocolour, ecl_lat, astrometric_params_solved, zpt [mas]) from
# gaiadr3_zeropoint 0.1.0 (zero_point.zpt.get_zpt, warnings on, rounded to 1e-6).
L21_REFERENCE = [
    (5.0, 1.55, np.nan, 10.0, 31, -0.038715),
    (6.0, 1.48, np.nan, 0.0, 31, -0.036113),
    (10.9, 1.3, np.nan, -45.0, 31, -0.028482),
    (12.5, 1.6, np.nan, 60.0, 31, -0.019501),
    (13.1, 1.15, np.nan, -80.0, 31, -0.052718),
    (14.7, 1.48, np.nan, 25.0, 31, -0.038787),
    (16.0, 1.75, np.nan, -5.0, 31, -0.034437),
    (17.5, 1.48, np.nan, 0.0, 31, -0.029983),
    (18.3, 1.4, np.nan, 70.0, 31, -0.023107),
    (19.6, 1.52, np.nan, -30.0, 31, -0.015405),
    (20.5, 1.85, np.nan, 15.0, 31, -0.007514),
    (21.4, 1.45, np.nan, -60.0, 31, -0.009416),
    (8.0, np.nan, 1.3, 40.0, 95, -0.030617),
    (12.0, np.nan, 1.48, -10.0, 95, -0.023303),
    (15.95, np.nan, 1.65, 85.0, 95, -0.013415),
    (17.0, np.nan, 1.2, -50.0, 95, -0.058761),
    (19.0, np.nan, 1.8, 5.0, 95, -0.005735),
    (20.7, np.nan, 1.55, -75.0, 95, -0.048418),
]


def test_parallax_zero_point_matches_lindegren2021_reference():
    a = np.array(L21_REFERENCE)
    zpt, ok = cs.parallax_zero_point_l21(
        a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4].astype(int)
    )
    np.testing.assert_allclose(zpt, a[:, 5], atol=6e-7)
    # Hand check at a G node, nu_eff = 1.48, beta = 0: q00 + q02 * (-1/3) (Table 9).
    assert zpt[7] == pytest.approx((-29.18 - 2.41 / 3) * 1e-3, abs=1e-12)
    # out of range: G <= 6 or >= 21, 6p pseudocolour outside (1.24, 1.72)
    assert ok.tolist() == [False, False, True, True, True, True, True, True, True, True,
                           True, False, True, True, True, False, False, True]  # fmt: skip
    z2, ok2 = cs.parallax_zero_point_l21(15.0, 1.5, np.nan, 0.0, 3)
    assert np.isnan(z2[0]) and not ok2[0]


def _partitions():
    return {3: np.array([4, 5, 6, 7]), 2: np.array([0])}


def test_partition_assignment_uses_the_source_id_level12_pixel():
    parts = _partitions()
    pix12 = np.array([0, (1 << 18) - 1, 4 << 18, (6 << 18) + 17, (8 << 18), -1])
    sid = (pix12 << 35) + 12345
    sid[-1] = -5
    k, p = bcs.assign_partitions(sid, parts)
    assert k.tolist() == [2, 2, 3, 3, -1, -1]
    assert p.tolist() == [0, 0, 4, 6, -1, -1]
    # order 29 index of the same pixel is pix12 << 34 (up to sub-pixel bits)
    hp29 = (pix12[2] << 34) + 999
    assert hp29 >> 34 == sid[2] >> 35
    with pytest.raises(ValueError):
        bcs.assign_partitions(sid, {2: np.array([1]), 3: np.array([4])})


def test_partition_files_handles_directory_and_single_file_layouts(tmp_path):
    d = tmp_path / "dataset" / "Norder=6" / "Dir=30000" / "Npix=30227"
    d.mkdir(parents=True)
    (d / "b.parquet").touch()
    (d / "a.parquet").touch()
    f = tmp_path / "dataset" / "Norder=4" / "Dir=0"
    f.mkdir(parents=True)
    (f / "Npix=80.parquet").touch()
    assert [p.name for p in bcs.partition_files(tmp_path, 6, 30227)] == [
        "a.parquet",
        "b.parquet",
    ]
    assert [p.name for p in bcs.partition_files(tmp_path, 4, 80)] == ["Npix=80.parquet"]
    assert bcs.partition_files(tmp_path, 4, 81) == []


def test_hash_subset_is_reproducible_and_order_independent():
    rng = np.random.default_rng(3)
    ids = np.unique(rng.integers(1, 1 << 62, 5000))
    sel = ids[bcs.hash_subset(ids, 700)]
    perm = rng.permutation(len(ids))
    sel2 = ids[perm][bcs.hash_subset(ids[perm], 700)]
    assert len(sel) == 700 and set(sel) == set(sel2)
    assert set(ids[bcs.hash_subset(ids, 700, seed=1)]) != set(sel)
    assert np.all(np.diff(bcs.hash_subset(ids, 700)) > 0)
    assert len(bcs.hash_subset(ids, 10**6)) == len(ids)


# ---------------------------------------------------------------- end to end

ARROW = {"f8": pa.float64(), "f4": pa.float32(), "i2": pa.int16(), "i1": pa.int8(),
         "i8": pa.int64(), "?": pa.bool_()}  # fmt: skip


def _xp_row(rng, sid, hp29):
    row = {
        "source_id": sid,
        "ra": 1.0,
        "dec": 2.0,
        "_healpix_29": hp29,
        "solution_id": 7,
    }
    for c in cs.XP_COLUMNS:
        if c.width == cs.N_CORR:
            row[c.name] = cs.pack_correlations(
                0.9 * _random_correlation(rng) + 0.1 * np.eye(55)
            )
        elif c.width:
            row[c.name] = rng.uniform(0.1, 2.0, c.width)
        else:
            row[c.name] = (
                rng.integers(1, 50) if c.dtype.startswith("i") else rng.uniform(1, 2)
            )
    return row


def _gaia_row(rng, sid, hp29, has_xp=True):
    row = {"source_id": sid, "_healpix_29": hp29, "solution_id": 7}
    for c in cs.GAIA_COLUMNS:
        if c.dtype == "?":
            row[c.name] = has_xp
        elif c.name == "astrometric_params_solved":
            row[c.name] = 31
        elif c.name == "nu_eff_used_in_astrometry":
            row[c.name] = 1.5
        elif c.name == "phot_g_mean_mag":
            row[c.name] = rng.uniform(10, 17)
        else:
            row[c.name] = (
                rng.uniform(-0.2, 0.2)
                if c.name.endswith("_corr")
                else rng.uniform(0.1, 5)
            )
    return row


def _write(path, rows, columns):
    fields = []
    for name in rows[0]:
        col = columns.get(name)
        if col is None:
            fields.append(pa.field(name, pa.int64()))
        elif col.width:
            elem = pa.float64() if col.dtype == "f8" else pa.float32()
            fields.append(pa.field(name, pa.list_(elem)))
        else:
            fields.append(pa.field(name, ARROW[col.dtype]))
    schema = pa.schema(fields)
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.sort_by("_healpix_29"), path, row_group_size=3)


def _catalogues(tmp_path, rng):
    """Two order-3 partitions A (pixel 4) and B (a neighbour of A's first child)."""
    p_a = 4 << 18  # level-12 pixel at a corner of order-3 pixel 4
    nb = healpy.get_all_neighbours(4096, p_a, nest=True)
    q = int(next(n for n in nb if n >= 0 and (n >> 18) != 4))
    part_b = q >> 18
    stars_a = [(p_a + 10 + i) for i in range(6)]  # level-12 pixels inside A
    stars_b = [((part_b << 18) + 50 + i) for i in range(3)]
    sid = lambda pix, j: (pix << 35) + j  # noqa: E731
    xp, gaia, ids = {}, {}, {"a": [], "b": []}
    for j, pix in enumerate(stars_a + stars_b):
        ids["a" if j < len(stars_a) else "b"].append(sid(pix, j + 1))
        s = sid(pix, j + 1)
        hp = (pix << 34) + 5
        xp[s], gaia[s] = _xp_row(rng, s, hp), _gaia_row(rng, s, hp)
    no_xp = sid(stars_a[0], 999)  # in gaia_source only
    gaia[no_xp] = _gaia_row(rng, no_xp, (stars_a[0] << 34) + 6, has_xp=False)
    moved = sid(p_a, 4242)  # source_id says A, J2016 position in B
    xp[moved], gaia[moved] = (
        _xp_row(rng, moved, q << 34),
        _gaia_row(rng, moved, q << 34),
    )
    cols = {c.name: c for c in cs.XP_COLUMNS + cs.GAIA_COLUMNS}
    roots = {"xp": tmp_path / "xp", "gaia": tmp_path / "gaia"}
    in_a = lambda s: (s >> 35) >> 18 == 4 and s != moved  # noqa: E731
    for cat, rows in (("xp", xp), ("gaia", gaia)):
        root = roots[cat]
        root.mkdir()
        (root / "partition_info.csv").write_text(f"Norder,Npix\n3,4\n3,{part_b}\n")
        for npix, members in ((4, [s for s in rows if in_a(s)]),
                              (part_b, [s for s in rows if not in_a(s)])):  # fmt: skip
            base = root / "dataset" / "Norder=3" / "Dir=0"
            if cat == "xp":  # directory of files, split like the ESA CSV chunks
                half = len(members) // 2
                for i, chunk in enumerate((members[:half], members[half:])):
                    if chunk:
                        _write(
                            base / f"Npix={npix}" / f"Xp_{i}.parquet",
                            [rows[s] for s in chunk],
                            cols,
                        )
            else:
                _write(base / f"Npix={npix}.parquet", [rows[s] for s in members], cols)
    return roots, xp, gaia, ids, no_xp, moved, part_b


@pytest.mark.parametrize("workers", [1, 2])
def test_build_end_to_end(tmp_path, workers):
    rng = np.random.default_rng(4)
    roots, xp, gaia, ids, no_xp, moved, part_b = _catalogues(tmp_path, rng)
    xp_ids = sorted(xp)
    a, b = ids["a"], ids["b"]
    absent = (((4 << 18) + 12) << 35) + 777  # in partition A, in neither catalogue
    table_ids = np.array([a[0], moved, no_xp, absent, b[0], a[-1], a[0]])
    fits.BinTableHDU.from_columns(
        [
            fits.Column("source_id", "K", array=table_ids),
            fits.Column("G", "E", array=np.ones(7)),
        ]
    ).writeto(tmp_path / "table.fits")
    np.savez(
        tmp_path / "unl.npz", source_id=np.array(xp_ids[1:6]), features=np.zeros((5, 2))
    )
    out = tmp_path / "out"
    argv = ["--input", f"table={tmp_path / 'table.fits'}", "--input", f"unl={tmp_path / 'unl.npz'}",
            "--sample", "unl=3", "--xp-hats", str(roots["xp"]), "--gaia-hats", str(roots["gaia"]),
            "--output", str(out), "--workers", str(workers), "--batch-size", "2"]  # fmt: skip
    manifest = bcs.build(bcs.parse_args(argv))
    assert manifest["status"] == "complete"
    assert (
        json.loads((out / "manifest.json").read_text())["inputs"]["table"]["found_xp"]
        == 5
    )
    t = manifest["inputs"]["table"]
    assert (t["selected"], t["found_xp"], t["missing_xp"], t["found_astrometry"]) == (
        7,
        5,
        2,
        6,
    )
    assert t["missing_xp_with_has_xp_flag"] == 0
    assert manifest["targets"]["found_in_neighbour_partition"] == {"xp": 1, "gaia": 1}
    assert manifest["reads"]["gaia"]["level12_mismatch"] == 1

    with cs.CovarianceSidecar(out / "sidecar_table.h5") as sc:
        r = sc.read([c.name for c in cs.SCHEMA])
        np.testing.assert_array_equal(r["source_id"], table_ids)
        np.testing.assert_array_equal(r["row_index"], np.arange(7))
        assert r["found_xp"].tolist() == [True, True, False, False, True, True, True]
        assert r["found_astrometry"].tolist() == [
            True,
            True,
            True,
            False,
            True,
            True,
            True,
        ]
        for i, s in enumerate(table_ids):
            if s in xp:
                for c in cs.XP_COLUMNS:
                    np.testing.assert_allclose(
                        r[c.name][i], np.asarray(xp[s][c.name], c.dtype), rtol=1e-6
                    )
            if s in gaia:
                for c in cs.GAIA_COLUMNS:
                    np.testing.assert_allclose(
                        r[c.name][i], np.asarray(gaia[s][c.name], c.dtype), rtol=1e-6
                    )
        assert np.isnan(r["bp_coefficients"][2]).all() and r["bp_n_parameters"][2] == -1
        assert r["hats_npix"].tolist() == [4, part_b, 4, -1, part_b, 4, 4]
        assert r["hats_norder"].tolist() == [3, 3, 3, -1, 3, 3, 3]
        assert np.isfinite(r["parallax_zpt_l21"][r["found_astrometry"]]).all()
        cov = sc.xp_covariance([4, 0], "rp", units="table", g_flux_term=True)
        assert cov.shape == (2, 55, 55)
        np.linalg.cholesky(cov)
        joint = sc.joint_xp_covariance([0], units="table", g_flux_term=True)[0]
        assert np.abs(joint[:55, 55:]).max() > 0
        np.testing.assert_allclose(
            joint[:55, :55], sc.xp_covariance(0, "bp", units="table", g_flux_term=True)
        )
        np.linalg.cholesky(sc.astrometric_covariance([0, 4]))
        zpt, _ = sc.parallax_zero_point([5, 0])
        np.testing.assert_allclose(zpt, r["parallax_zpt_l21"][[5, 0]], rtol=1e-6)
    with h5py.File(out / "sidecar_unl.h5") as f:
        picked = np.array(xp_ids[1:6])[bcs.hash_subset(np.array(xp_ids[1:6]), 3)]
        np.testing.assert_array_equal(f["source_id"][:], picked)
        assert f["found_xp"][:].all()
        assert f["bp_coefficient_correlations"].dtype == np.float32
        assert f["bp_coefficient_correlations"].attrs["description"].startswith("BP")
    with pytest.raises(SystemExit):
        bcs.build(bcs.parse_args(argv))
