"""photometry_v1a builder (``scripts/build_photometry_v1a.py``) on a tiny synthetic sky:
Gaia HATS partitions, unsorted Gaia-keyed FITS match tables (2MASS, SDSS, CatWISE with
``id2``, PS1 ids + chunked photometry files), a SkyMapper HATS tree and small SFD maps.
Checks row order, the exact contract columns, nulls for absent and sentinel values,
de-duplication, the PS1 chunk fallback, out-of-range source ids, SFD against dustmaps,
the manifest checks, resume, and that the v11 loader reads the tree."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]


def _load():
    # importable by name, so spawned pool workers can unpickle its functions
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "build_photometry_v1a", ROOT / "scripts" / "build_photometry_v1a.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()
PARTS = [(1, 0), (1, 1), (2, 8)]
N_PER = 40


def _sfd_maps(d: Path, rng) -> None:
    """64x64 stand-ins with the SFD98 header convention (scale 32 px from b=0 to 90)."""
    d.mkdir(parents=True)
    for pole, n in (("ngp", 1), ("sgp", -1)):
        h = fits.Header()
        h["CRPIX1"] = h["CRPIX2"] = 32.5
        h["CRVAL1"] = h["CRVAL2"] = 90.0 * n
        h["CTYPE1"], h["CTYPE2"] = "GLON-ZEA", "GLAT-ZEA"
        h["CD1_1"] = -0.0395646818624 * 2048 / 32 * n
        h["CD2_2"] = 0.0395646818624 * 2048 / 32 * n
        h["CD1_2"] = h["CD2_1"] = 0.0
        h["LONPOLE"] = 180
        data = rng.uniform(0.0, 0.5, (64, 64)).astype(">f4")
        fits.PrimaryHDU(data, h).writeto(d / f"SFD_dust_4096_{pole}.fits")


def _write_fits(table: Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp.fits")
    table.write(tmp, format="fits")
    tmp.rename(path)  # the PS1 files are plain FITS named .fits.gz


@pytest.fixture(scope="module")
def sky(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sky")
    rng = np.random.default_rng(7)
    gaia = tmp / "gaia"
    rows = {}
    for o, p in PARTS:
        lo, hi = B.pixel_range(o, p)
        pix = rng.integers(lo, hi, N_PER)
        sid = (pix << 35) + rng.integers(1, 1 << 30, N_PER)
        rows[(o, p)] = sid
    # one source whose id HEALPix index lies in (1, 0) but which sits in (1, 1)
    lo0, _ = B.pixel_range(1, 0)
    misplaced = (np.int64(lo0 + 5) << 35) + 12345
    rows[(1, 1)][3] = misplaced
    allg = []
    for (o, p), sid in rows.items():
        n = len(sid)
        g = rng.uniform(10, 18, n).astype(np.float32)
        g[0] = np.nan
        t = pa.table(
            {
                "_healpix_29": pa.array((sid >> 35) << 34, pa.int64()),
                "source_id": pa.array(sid, pa.int64()),
                "ra": rng.uniform(0, 360, n),
                "dec": rng.uniform(-90, 90, n),
                "l": rng.uniform(0, 360, n),
                "b": np.degrees(np.arcsin(rng.uniform(-1, 1, n))),
                "parallax": rng.normal(1, 0.5, n),
                "parallax_error": rng.uniform(0.01, 0.3, n).astype(np.float32),
                "pmra": rng.normal(0, 5, n),
                "pmra_error": rng.uniform(0.01, 0.3, n).astype(np.float32),
                "pmdec": rng.normal(0, 5, n),
                "pmdec_error": rng.uniform(0.01, 0.3, n).astype(np.float32),
                "phot_g_mean_mag": pa.array(g, pa.float32()),
                "phot_g_mean_flux_over_error": rng.uniform(50, 5000, n).astype(
                    np.float32
                ),
                "phot_bp_mean_mag": (g + 0.4).astype(np.float32),
                "phot_bp_mean_flux_over_error": rng.uniform(10, 500, n).astype(
                    np.float32
                ),
                "phot_rp_mean_mag": (g - 0.6).astype(np.float32),
                "phot_rp_mean_flux_over_error": rng.uniform(10, 500, n).astype(
                    np.float32
                ),
                "has_xp_continuous": rng.uniform(size=n) < 0.3,
            }
        )
        f = gaia / "dataset" / B.partition_rel(o, p)
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, f, row_group_size=16)
        allg.append(t)
    gtab = pa.concat_tables(allg)
    (gaia / "partition_info.csv").write_text(
        "Norder,Npix\n" + "".join(f"{o},{p}\n" for o, p in PARTS)
    )
    (gaia / "properties").write_text(f"hats_nrows={gtab.num_rows}\n")
    sid = gtab["source_id"].to_numpy()
    g = gtab["phot_g_mean_mag"].to_numpy(zero_copy_only=False)
    n = len(sid)

    # 2MASS: 60 % of the stars, shuffled; Ks = G - 2; one U, one NaN error, one exact
    # repeat and one ambiguous repeat
    tm = rng.permutation(n)[: int(0.6 * n)]
    tm = tm[np.isfinite(g[tm])]
    ks = (g[tm] - 2 + rng.normal(0, 0.05, len(tm))).astype(np.float32)
    q = np.array(["AAA"] * len(tm), "U3")
    q[1] = "AAU"
    kse = np.full(len(tm), 0.03, np.float32)
    kse[2] = np.nan
    t2 = Table(
        {
            "source_id": sid[tm],
            "tmass_oid": np.arange(len(tm), dtype=np.int64),
            "ph_qual": q,
            "j_m": ks + 0.6,
            "j_msigcom": np.full(len(tm), 0.02, np.float32),
            "h_m": ks + 0.1,
            "h_msigcom": np.full(len(tm), 0.02, np.float32),
            "ks_m": ks,
            "ks_msigcom": kse,
        }
    )
    rep = t2[[4]]  # identical repeat: kept once
    amb = t2[[5]]
    amb["ks_m"] += 1.0  # same id, different values: ambiguous, dropped
    t2 = Table(np.concatenate([t2.as_array(), rep.as_array(), amb.as_array()]))
    adql = tmp / "adql"
    adql.mkdir()
    _write_fits(t2, adql / "twomass.fits")

    # SDSS: 30 %, one -9999 u
    sd = rng.permutation(n)[: int(0.3 * n)]
    sdc = {"source_id": sid[sd], "objid": np.arange(len(sd), dtype=np.int64)}
    sdc["flags"] = np.zeros(len(sd), np.int64)
    for b in "ugriz":
        sdc[f"flags_{b}"] = np.zeros(len(sd), np.int64)
    for b in "ugriz":
        sdc[b] = rng.uniform(15, 22, len(sd)).astype(np.float32)
    for b in "ugriz":
        sdc[f"err_{b}"] = np.full(len(sd), 0.05, np.float32)
    sdc["u"][0] = -9999.0
    sdc["err_u"][0] = -9999.0
    _write_fits(Table(sdc), adql / "sdss.fits")

    # CatWISE: Gaia id in id2, CatWISE name in source_id
    cw = rng.permutation(n)[: int(0.5 * n)]
    cwc = {
        "id2": sid[cw],
        "source_id": np.array([f"0453p000_b0-{i:06d}" for i in range(len(cw))]),
    }
    for k in ("w1mpro", "w1mpro_pm", "w1sigmpro", "w1sigmpro_pm", "w1snr", "w1snr_pm"):
        cwc[k] = rng.uniform(0.02, 15, len(cw))
    for k in ("w2mpro", "w2mpro_pm", "w2sigmpro", "w2sigmpro_pm", "w2snr", "w2snr_pm"):
        cwc[k] = rng.uniform(0.02, 15, len(cw))
    cwc["w1sigmpro"][:] = 0.03
    cwc["w2sigmpro"][:] = 0.04
    _write_fits(Table(cwc), adql / "catwise.fits")

    # PS1: ids table + chunk photometry files (7 ids per chunk); one obj only in the
    # next chunk's file (fallback), one obj nowhere, one -999 magnitude
    ps = rng.permutation(n)[: int(0.7 * n)]
    ext = (10**16 + np.arange(len(ps)) * 7919).astype(np.int64)
    _write_fits(
        Table({"source_id": sid[ps], "original_ext_source_id": ext}),
        adql / "pans_gaia_ids.fits",
    )
    phd = adql / "panstarrsdr1_matches"
    phd.mkdir()
    chunk = 7
    nchunks = -(-len(ps) // chunk)
    obj_files = {k: list(ext[k * chunk : (k + 1) * chunk]) for k in range(nchunks)}
    moved, lost = ext[2], ext[9]  # chunk 0 -> file 1; chunk 1: nowhere
    obj_files[0].remove(moved)
    obj_files[1].append(moved)
    obj_files[1].remove(lost)
    for k, objs in obj_files.items():
        objs = np.sort(np.asarray(objs, np.int64))
        c = {"obj_id": objs}
        for b in "grizy":
            c[f"{b}_mean_psf_mag"] = rng.uniform(14.5, 21, len(objs))
            c[f"{b}_mean_psf_mag_error"] = np.full(len(objs), 0.02)
        if k == 2:
            c["g_mean_psf_mag"][0] = -999.0
        c["obj_info_flag"] = np.zeros(len(objs), np.int32)
        c["quality_flag"] = np.full(len(objs), 52, np.int16)
        _write_fits(Table(c), phd / f"ids_pansdr1_{k}.fits.gz")

    # SkyMapper HATS: 40 % of the stars; a second, farther object for one star; one
    # beyond 2"; one without Gaia id
    sm = rng.permutation(n)[: int(0.4 * n)]
    sid_sm = sid[sm].copy()
    dist = rng.uniform(0.05, 1.0, len(sm))
    sid_sm = np.concatenate([sid_sm, [sid_sm[0], sid_sm[1], 0]])
    dist = np.concatenate([dist, [dist[0] + 0.5, 0.2, np.nan]])
    dist[1] = 3.0  # star 1: only its 0.2" twin counts
    nm = len(sid_sm)
    smc = {
        "object_id": np.arange(nm, dtype=np.int64),
        "flags": np.zeros(nm, np.int32),
        "gaia_dr3_id1": pa.array(np.where(sid_sm > 0, sid_sm, 0), mask=sid_sm == 0),
        "gaia_dr3_dist1": dist,
    }
    for b in "uvgriz":
        smc[f"{b}_psf"] = rng.uniform(14, 20, nm)
        smc[f"e_{b}_psf"] = np.full(nm, 0.03)
    smc["u_psf"][-3] = 99.0  # farther twin of star 0: must not win
    sfd_ref = None
    smt = tmp / "skymapper"
    sf = smt / "dataset" / B.partition_rel(0, 0)
    sf.parent.mkdir(parents=True)
    sfd_dir = tmp / "sfd"
    _sfd_maps(sfd_dir, rng)
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery

    lg = gtab["l"].to_numpy()
    bg = gtab["b"].to_numpy()
    sfd_ref = SFDQuery(map_dir=str(sfd_dir))(
        SkyCoord(lg, bg, unit="deg", frame="galactic")
    )
    pos = {s: i for i, s in enumerate(sid)}
    smc["ebmv_sfd"] = np.array([sfd_ref[pos[s]] if s in pos else 0.1 for s in sid_sm])
    pq.write_table(pa.table(smc), sf)
    (smt / "partition_info.csv").write_text("Norder,Npix\n0,0\n")
    (smt / "properties").write_text(f"hats_nrows={nm}\n")

    out = tmp / "out" / "photometry_v1a"
    args = [
        "--output", str(out), "--work", str(tmp / "work"), "--workers", "2",
        "--gaia", str(gaia), "--twomass", str(adql / "twomass.fits"),
        "--sdss", str(adql / "sdss.fits"), "--catwise", str(adql / "catwise.fits"),
        "--ps1-ids", str(adql / "pans_gaia_ids.fits"), "--ps1-phot", str(phd),
        "--skymapper", str(smt), "--sfd", str(sfd_dir), "--ps1-chunk-rows", str(chunk),
        "--fits-chunk-rows", "13",
    ]  # fmt: skip
    rc = B.main(args)
    return dict(
        rc=rc, out=out, work=tmp / "work", args=args, gtab=gtab, sid=sid, g=g,
        t2=t2, tm=tm, q=q, sd=sd, sdc=sdc, cw=cw, cwc=cwc, ps=ps, ext=ext,
        moved=moved, lost=lost, sid_sm=sid_sm, smc=smc, dist=dist,
        sfd_ref=np.asarray(sfd_ref), misplaced=misplaced,
    )  # fmt: skip


def _out(sky) -> pa.Table:
    parts = [
        pq.read_table(sky["out"] / "dataset" / B.partition_rel(o, p)) for o, p in PARTS
    ]
    return pa.concat_tables(parts)


def _col(t, name):
    return t[name].to_numpy(zero_copy_only=False)


def test_build_completes_with_manifest_checks(sky):
    assert sky["rc"] == 0
    m = json.loads((sky["out"] / "manifest.json").read_text())
    assert m["status"] == "photometry_v1a_complete", m["checks"]
    assert m["rows"] == len(sky["sid"])
    assert m["checks"]["duplicate_source_id"]["out_of_range_ids"] == 1
    assert m["checks"]["duplicate_source_id"]["cross_partition_duplicates"] == 0
    s = m["surveys"]
    assert s["twomass"]["identical_repeats_dropped"] == 1
    assert s["twomass"]["ambiguous_ids_dropped"] == 1
    assert s["twomass"]["staged_rows"] == len(sky["tm"]) - 1
    assert s["ps1"]["fallback_found"] == 1 and s["ps1"]["no_photometry"] == 1
    assert s["sdss"]["bands"]["u"]["sentinel_to_null"] == 1
    assert all(v["staged_rows_not_in_gaia"] == 0 for v in s.values())
    assert m["checks"]["twomass_ks_g"]["pearson_r_ks_g"] > 0.95
    assert m["checks"]["sfd"]["vs_skymapper_ebmv_sfd"]["frac_within_0p01"] == 1.0
    assert m["checks"]["sfd"]["dustmaps_parity"]["max_abs_diff"] == 0.0
    info = (sky["out"] / "partition_info.csv").read_text().split()
    assert info[0] == "Norder,Npix" and len(info) == 1 + len(PARTS)
    assert f"hats_nrows={len(sky['sid'])}" in (sky["out"] / "properties").read_text()


def test_rows_order_and_exact_contract_columns(sky):
    t = _out(sky)
    assert t.column_names == B.output_columns()
    assert np.array_equal(_col(t, "source_id"), sky["sid"])
    assert (
        pq.read_schema(sky["out"] / "dataset" / "_common_metadata").names
        == B.output_columns()
    )
    from masked_stellar_autoencoder.v11.schema import ContractReader, load_registry

    r = ContractReader(load_registry(), "gaia_keyed_v1a")
    xp = {
        v
        for k in ("coefficients", "errors", "correlations")
        for v in r.contract["xp"][k].values()
    }
    assert set(r.columns()) - xp == set(t.column_names) - {"source_id", "_healpix_29"}
    assert t["ph_qual"].type == pa.string()
    assert t["phot_g_mean_mag"].null_count == 1 * len(PARTS)  # NaN G -> null
    assert t["sfd_ebv"].null_count == 0
    assert np.array_equal(_col(t, "sfd_ebv"), sky["sfd_ref"].astype(np.float32))


def test_values_nulls_and_sentinels(sky):
    t = _out(sky)
    pos = {s: i for i, s in enumerate(sky["sid"])}
    # 2MASS
    t2 = sky["t2"]
    ks = _col(t, "ks_m")
    kse = _col(t, "ks_msigcom")
    q = t["ph_qual"].to_pylist()
    first = sky["tm"]
    for j in range(len(first)):
        i = pos[t2["source_id"][j]]
        if j == 5:  # ambiguous repeat: absent
            assert np.isnan(ks[i]) and q[i] is None
            continue
        assert ks[i] == np.float32(t2["ks_m"][j])
        assert q[i] == str(t2["ph_qual"][j])
    assert np.isnan(kse[pos[t2["source_id"][2]]])  # NaN error stays null, value kept
    assert not np.isnan(ks[pos[t2["source_id"][2]]])
    no2 = np.setdiff1d(sky["sid"], t2["source_id"])
    assert all(q[pos[s]] is None for s in no2)
    assert t["ks_m"].null_count == len(no2) + 1
    # SDSS sentinel
    i0 = pos[sky["sdc"]["source_id"][0]]
    assert np.isnan(_col(t, "u")[i0]) and np.isnan(_col(t, "err_u")[i0])
    assert _col(t, "g")[i0] == sky["sdc"]["g"][0]
    assert _col(t, "flags_u")[i0] == 0
    # CatWISE via id2
    i = pos[sky["cwc"]["id2"][3]]
    assert _col(t, "w1mpro")[i] == np.float32(sky["cwc"]["w1mpro"][3])
    assert t["w2sigmpro"].null_count == len(sky["sid"]) - len(sky["cw"])
    # PS1: fallback-filled, lost and -999
    ext = sky["ext"]
    ps_sid = sky["sid"][sky["ps"]]
    assert not np.isnan(_col(t, "r_mean_psf_mag")[pos[ps_sid[2]]])  # moved obj
    assert np.isnan(_col(t, "r_mean_psf_mag")[pos[ps_sid[9]]])  # lost obj
    assert t["quality_flag"].to_pylist()[pos[ps_sid[9]]] is None
    assert t["quality_flag"].to_pylist()[pos[ps_sid[2]]] == 52
    assert t["g_mean_psf_mag"].null_count == len(sky["sid"]) - len(ext) + 1 + 1
    # SkyMapper: nearest within 2"
    sm = sky["sid_sm"]
    u = _col(t, "u_psf")
    assert u[pos[sm[0]]] == np.float32(sky["smc"]["u_psf"][0])  # not the 99 twin
    assert u[pos[sm[1]]] == np.float32(sky["smc"]["u_psf"][-2])  # 0.2" twin, not 3"
    assert t["flags"].null_count == len(sky["sid"]) - (len(sm) - 3)
    # the out-of-range source id is found in every store it is in
    i = pos[sky["misplaced"]]
    for name, ids in (
        ("ks_m", t2["source_id"]),
        ("u", sky["sdc"]["source_id"]),
        ("w1mpro", sky["cwc"]["id2"]),
    ):
        assert np.isnan(_col(t, name)[i]) != (sky["misplaced"] in set(ids))


def test_resume_skips_partitions(sky):
    f = sky["out"] / "dataset" / B.partition_rel(*PARTS[0])
    before = f.stat().st_mtime_ns
    assert B.main(sky["args"]) == 0  # complete: refuses to rebuild
    assert B.main(sky["args"] + ["--force"]) == 0  # stats fingerprints match: skipped
    assert f.stat().st_mtime_ns == before


def test_v11_loader_reads_the_tree(sky, tmp_path):
    from masked_stellar_autoencoder.v11.data import HatsStream, StreamSpec
    from masked_stellar_autoencoder.v11.schema import ContractReader, load_registry

    reg = load_registry()
    reader = ContractReader(reg, "gaia_keyed_v1a")
    asg = tmp_path / "assignment"
    for o, p in PARTS:
        sid = pq.read_table(
            sky["out"] / "dataset" / B.partition_rel(o, p), columns=["source_id"]
        )["source_id"]
        f = asg / B.partition_rel(o, p)
        f.parent.mkdir(parents=True, exist_ok=True)
        n = len(sid)
        pq.write_table(
            pa.table(
                {
                    "source_id": sid,
                    "partition": pa.array(np.zeros(n, np.uint8)),
                    "group_id": pa.array(np.full(n, -1, np.int32)),
                    "input_class": pa.array(np.ones(n, np.uint8)),
                }
            ),
            f,
        )
    spec = StreamSpec(
        trees=[(str(sky["out"]), None)],
        assignment=str(asg),
        partitions=["FIT"],
        reader=reader,
        registry=reg,
        operator=None,
        shuffle=False,
    )
    blocks = list(HatsStream(spec).blocks())
    got = np.concatenate([b.source_id for b in blocks])
    vals = np.concatenate([b.values for b in blocks])
    assert len(got) == len(sky["sid"])
    pos = {s: i for i, s in enumerate(got)}
    t2 = sky["t2"]
    ks = reg.index("tmass_ks")
    assert vals[pos[t2["source_id"][0]], ks] == pytest.approx(t2["ks_m"][0])
    assert np.isnan(vals[pos[t2["source_id"][1]], ks])  # ph_qual U: contract rejects
    assert np.isfinite(vals[:, reg.index("sfd_ebv")]).all()
    assert np.isnan(vals[pos[sky["sdc"]["source_id"][0]], reg.index("sdss_u")])


def test_sfd_matches_dustmaps_formula(tmp_path):
    """Real SFD98 header convention: ZEA pixel of (l, b) = Appendix C formula."""
    rng = np.random.default_rng(3)
    _sfd_maps(tmp_path / "m", rng)
    sfd = B.SFD(tmp_path / "m")
    lon = rng.uniform(0, 360, 1000)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 1000)))
    from scipy.ndimage import map_coordinates

    ref = np.empty(1000, np.float32)
    for pole, n in (("ngp", 1), ("sgp", -1)):
        m = (lat >= 0) if n == 1 else (lat < 0)
        r = np.sqrt(1 - n * np.sin(np.radians(lat[m])))
        x = 32 * r * np.cos(np.radians(lon[m])) + 31.5
        y = -32 * n * r * np.sin(np.radians(lon[m])) + 31.5
        data = np.asarray(
            fits.getdata(tmp_path / "m" / f"SFD_dust_4096_{pole}.fits"), np.float32
        )
        ref[m] = map_coordinates(data, [y, x], order=1, mode="nearest")
    assert np.allclose(sfd(lon, lat), ref, atol=1e-6)


def test_lookup_block_and_outside():
    store = np.array([5, 10, 20, 30, 1000, 2000], np.int64)
    pos, a, b = B.lookup(store, np.array([10, 30, 2000, 7, 999]), 8, 40)
    assert (a, b) == (1, 4)
    assert pos.tolist() == [1, 3, 5, -1, -1]
    col = np.arange(6) * 10
    assert B.gather(col, pos, a, b).tolist() == [10, 30, 50, 0, 0]


def test_contract_column_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "output_columns", lambda: ["source_id", "phot_g_mean_mag"])
    with pytest.raises(SystemExit, match="output columns differ"):
        B.main(["--output", str(tmp_path / "o"), "--gaia", str(tmp_path)])
