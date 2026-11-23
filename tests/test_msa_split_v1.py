"""msa-split-v1 freeze (``scripts/build_msa_split_v1.py``): spec test vectors,
the hash against a scalar reference, fractions on a synthetic Gaia-like sample,
stability under row order, group integrity and reserved roles, and an
end-to-end run on a tiny synthetic HATS tree with stand-in group/reserved
catalogues."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    # importable by name, so spawned pool workers can unpickle its functions
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "build_msa_split_v1", ROOT / "scripts" / "build_msa_split_v1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load()
M64 = (1 << 64) - 1


def _ref_unit(key: int) -> float:
    """Scalar splitmix64 reference in Python integers (datasets.yaml hash)."""
    z = (key ^ S.SALT64) + 0x9E3779B97F4A7C15 & M64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & M64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & M64
    z ^= z >> 31
    return (z >> 11) * 2.0**-53


def _gaia_ids(rng, n):
    """Gaia-like ids: HEALPix level-12 index << 35 plus a small running number."""
    pix = rng.integers(0, 12 * 4**12, n, dtype=np.int64)
    return np.unique((pix << 35) + rng.integers(0, 1 << 22, n, dtype=np.int64))


def test_spec_test_vectors():
    assert f"0x{S.SALT64:016x}" == "0xb6f6e5353645106b"
    for key, u_spec, name in S.TEST_VECTORS:
        u = float(S.unit_of_keys([key])[0])
        assert u == pytest.approx(u_spec, abs=1e-12)
        assert S.PARTITIONS[int(S.partition_of(u))] == name
    assert all(t["ok"] for t in S.check_test_vectors())


def test_unit_matches_scalar_reference_and_key_rules():
    rng = np.random.default_rng(1)
    ids = _gaia_ids(rng, 2000)
    u = S.unit(ids)
    assert np.array_equal(u, [_ref_unit(int(k)) for k in ids])
    assert np.all((u >= 0) & (u < 1))
    k = S.key64("grp:eb21:123")
    assert k >> 63 == 1 and S.key64(123) == 123
    with pytest.raises(ValueError):
        S.key64(-1)
    with pytest.raises(TypeError):
        S.unit(np.array([0.5]))
    # partition edges are half-open on the left: u == 0.75 is DEV
    assert list(S.partition_of([0.0, 0.7499, 0.75, 0.8, 0.88, 0.9999])) == [
        0,
        0,
        1,
        2,
        3,
        3,
    ]
    assert list(S.fit_fold([0.0, 0.149, 0.15, 0.7499, 0.75])) == [0, 0, 1, 4, -1]
    small = S.in_rec_sample(ids, 0.125)
    assert np.all(small <= S.in_rec_sample(ids, 0.25))  # nested samples


def test_fractions_on_synthetic_sample():
    rng = np.random.default_rng(2)
    ids = _gaia_ids(rng, 2_000_000)
    counts = np.bincount(S.partition_of(S.unit(ids)), minlength=4)
    fr = S.fractions(np.r_[counts, 0])
    for v in fr.values():
        assert abs(v["abs_diff"]) < 2e-3
    assert S.fractions_ok(fr, tol=0.05)


def _groups_fixture():
    clusters = pd.DataFrame(
        {
            "source_id": [10, 11, 12, 13, 13, 14],
            "name": [
                "Melotte_22",
                "Melotte_22",
                "Melotte_22",
                "NGC_1",
                "Melotte_22",
                "NGC_1",
            ],
            "prob": [0.9, 0.8, 0.75, 0.95, 0.7, 0.71],
        }
    )
    pairs = pd.DataFrame(
        {
            # 10-20: companion of a cluster member; 30-31 plain pair;
            # 12-14: members of two clusters (cannot both be kept); 40-41-42: triple
            "source_id1": [20, 30, 12, 40, 42],
            "source_id2": [10, 31, 14, 41, 40],
        }
    )
    return clusters, pairs


def test_group_resolution_and_integrity():
    clusters, pairs = _groups_fixture()
    groups, members, pp = S.resolve_groups(clusters, pairs)
    g = groups.set_index("key")
    gid = dict(zip(members.source_id, members.group_id, strict=True))
    key = groups.key.to_numpy()
    part = groups.partition.to_numpy()
    # 13 is in both clusters: the more probable one wins
    assert key[gid[13]] == "grp:hr23:NGC_1"
    # the companion of a cluster member follows the cluster
    assert g.loc["grp:eb21:10", "partition_key"] == "grp:hr23:Melotte_22"
    assert part[gid[20]] == part[gid[10]] == g.loc["grp:hr23:Melotte_22", "partition"]
    # plain pair and the triple share one key
    assert gid[30] == gid[31]
    assert gid[40] == gid[41] == gid[42]
    assert g.loc["grp:hr23:Melotte_22", "unit"] == pytest.approx(
        0.440677168076, abs=1e-12
    )
    # every group's members share the group's partition
    for gi, sub in members.groupby("group_id"):
        assert len({part[j] for j in sub.group_id}) == 1
    over = S.build_overrides(members, groups, {"gbs_v3": np.array([31, 99])})
    over = over.set_index("source_id")
    assert over.loc[31, "partition"] == S.RESERVED
    assert over.loc[31, "hash_partition"] == part[gid[30]]
    assert over.loc[99, "group_id"] == -1
    assert over.loc[99, "hash_partition"] == over.loc[99, "base_partition"]
    assert over.loc[31, "reserved_reason"] == "gbs_v3"
    assert not over.loc[10, "reserved"]


def test_resolution_is_independent_of_row_order():
    clusters, pairs = _groups_fixture()
    g1, m1, _ = S.resolve_groups(clusters, pairs)
    rng = np.random.default_rng(3)
    c2 = clusters.sample(frac=1, random_state=4).reset_index(drop=True)
    p2 = pairs.sample(frac=1, random_state=5).reset_index(drop=True)
    p2[["source_id1", "source_id2"]] = p2[["source_id2", "source_id1"]].to_numpy()
    g2, m2, _ = S.resolve_groups(c2, p2)
    pd.testing.assert_frame_equal(g1, g2)
    pd.testing.assert_frame_equal(m1, m2)
    ids = _gaia_ids(rng, 10_000)
    perm = rng.permutation(len(ids))
    assert np.array_equal(
        S.partition_of(S.unit(ids))[perm], S.partition_of(S.unit(ids[perm]))
    )


def test_unresolved_float_companions_reserve_their_rounding_superset():
    base = 5_000_000 << 35  # float64 spacing 32 here: base+1..base+15 round to base
    gaia = np.array([base + 1, base + 2, base + 40, base + 5000], np.int64)
    clusters = pd.DataFrame({"source_id": [base + 1], "name": ["NGC_1"], "prob": [0.9]})
    groups, members, _ = S.resolve_groups(
        clusters, pd.DataFrame({"source_id1": [], "source_id2": []}, dtype=np.int64)
    )
    over = S.build_overrides(members, groups, {})
    floats = np.array([float(base + 2)])
    S.init_worker(
        over.source_id.to_numpy(),
        over.partition.to_numpy(),
        over.group_id.to_numpy(),
        floats,
        "fp",
    )
    try:
        b, part, group, hit_pos, hit_row, float_row = S.assign(gaia)
    finally:
        empty = (np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.int32))
        S.init_worker(*empty, np.zeros(0), "")
    assert list(float_row) == [0, 1] and list(part[:2]) == [S.RESERVED] * 2
    assert part[2] == b[2] and part[3] == b[3] and group[0] == 0
    over = over.assign(
        in_gaia_dr3=True, written_partition=part[:1], input_class=np.uint8(1)
    )
    new, info = S.add_float_reserved(
        over, gaia[float_row], np.array([1, 2], np.uint8), floats
    )
    new = new.set_index("source_id")
    assert info["added"] == 1 and info["already_listed"] == 1
    assert new.loc[base + 1, "reserved_reason"] == "wd_age_companion_float"
    assert new.loc[base + 1, "group_id"] == 0 and new.loc[base + 2, "group_id"] == -1
    assert (new.partition == S.RESERVED).all() and new.reserved.all()


def test_input_class():
    nan = np.nan
    cls = S.input_class(
        [True, False, False, False, False],
        [15.0, 16.0, 17.0, nan, 18.0],
        [15.5, 16.5, nan, nan, nan],
        [14.5, 15.5, 16.0, nan, nan],
    )
    assert list(cls) == [S.CLS_X, S.CLS_P, S.CLS_G, S.CLS_N, S.CLS_G]


# ── end to end on a tiny synthetic HATS tree ────────────────────────────────


def _write_fits(path, cols):
    from astropy.io import fits

    fits.BinTableHDU.from_columns(
        [fits.Column(name=n, format=f, array=a) for n, f, a in cols]
    ).writeto(path)


def _fixture(tmp, rng, n=40_000):
    ids = _gaia_ids(rng, n)
    rng.shuffle(ids)
    gaia = tmp / "gaia"
    halves = [np.sort(ids[: n // 2]), np.sort(ids[n // 2 :])]
    for npix, sid in zip((3, 7), halves, strict=True):
        d = gaia / "dataset" / "Norder=1" / "Dir=0"
        d.mkdir(parents=True, exist_ok=True)
        m = len(sid)
        g = rng.uniform(10, 21, m)
        g[: m // 50] = np.nan
        bp = np.where(rng.uniform(size=m) < 0.1, np.nan, g + 0.4)
        pq.write_table(
            pa.table(
                {
                    "_healpix_29": np.arange(m, dtype=np.int64),
                    "source_id": sid,
                    "ra": rng.uniform(0, 360, m),
                    "has_xp_continuous": pa.array(
                        rng.uniform(size=m) < 0.15, mask=rng.uniform(size=m) < 0.01
                    ),
                    "phot_g_mean_mag": g.astype("f4"),
                    "phot_bp_mean_mag": pa.array(bp.astype("f4"), from_pandas=True),
                    "phot_rp_mean_mag": (g - 0.5).astype("f4"),
                }
            ),
            d / f"Npix={npix}.parquet",
        )
    (gaia / "partition_info.csv").write_text("Norder,Npix\n1,3\n1,7\n")
    (gaia / "properties").write_text(f"#HATS\nhats_nrows={n}\nhats_order=1\n")
    pool = iter(ids)
    take = lambda k: np.array([next(pool) for _ in range(k)], np.int64)  # noqa: E731
    cl_a, cl_b, cl_bench = take(300), take(200), take(50)
    names = ["Melotte_22"] * 300 + ["NGC_1"] * 200 + ["Melotte_25"] * 50
    mem_ids = np.r_[cl_a, cl_b, cl_bench, cl_a[:5], take(40)]
    mem_names = names + ["NGC_1"] * 5 + ["NGC_1"] * 40
    prob = np.r_[np.full(550, 0.9), np.full(5, 0.95), np.full(40, 0.3)]
    cid = {"Melotte_22": 1, "NGC_1": 2, "Melotte_25": 3}
    cat = tmp / "cat"
    cat.mkdir()
    _write_fits(
        cat / "members.fits",
        [
            ("GaiaDR3", "K", mem_ids),
            ("ID", "J", np.array([cid[x] for x in mem_names])),
            ("Prob", "D", prob),
            ("Name", "20A", np.array(mem_names)),
        ],
    )
    _write_fits(
        cat / "clusters.fits",
        [
            ("ID", "J", np.array([1, 2, 3])),
            ("Name", "20A", np.array(["Melotte_22", "NGC_1", "Melotte_25"])),
            ("Type", "1A", np.array(["o", "o", "o"])),
        ],
    )
    a, b = take(400), take(400)
    a[:3] = cl_a[10:13]  # binaries with a cluster member
    wd, ms = take(20), take(20)
    a = np.r_[a, wd[:10]]
    b = np.r_[b, ms[:10]]
    absent = np.int64(6_000_000_000_000_000_001)  # pair member not in Gaia
    a = np.r_[a, absent]
    b = np.r_[b, take(1)]
    pd.DataFrame(
        {
            "source_id1": a,
            "source_id2": b,
            "R_chance_align": np.r_[np.full(len(a) - 1, 0.01), 0.05],
        }
    ).to_parquet(cat / "eb21.parquet")
    gbs = take(5)
    pd.DataFrame({"GaiaDR3": pd.array(list(gbs) + [None], "Int64")}).to_parquet(
        cat / "gbs.parquet"
    )
    vb = cat / "gc"
    vb.mkdir()
    gc = take(30)
    with (vb / "NGC_104.txt").open("w") as fh:
        fh.write("# source_id\tra\tdec\tqflag\tmemberprob\n")
        for i, s in enumerate(gc):
            fh.write(f"{s}\t1.0\t2.0\t2\t{0.9 if i < 20 else 0.2:.5f}\n")
    gcx = take(10)
    pd.DataFrame(
        {"source_id": gcx, "prob": np.r_[np.full(5, 0.6), np.full(5, 0.1)]}
    ).to_parquet(cat / "gc_extra.parquet")
    pd.DataFrame(
        {"cluster": ["Hyades", "beta Pic MG"], "hr23_name": ["Melotte_25", None]}
    ).to_csv(cat / "priors.csv", index=False)
    kic_ids = take(3)
    _write_fits(cat / "legacy.fits", [("KIC", "K", np.array([101, 101, 102, 103]))])
    _write_fits(
        cat / "kepler_gaia.fits",
        [
            ("kepid", "K", np.array([101, 101, 102, 104])),
            ("source_id", "K", np.r_[kic_ids[0], kic_ids[1], kic_ids[2], take(1)]),
            ("kepler_gaia_ang_dist", "D", np.array([0.1, 0.5, 0.2, 0.1])),
        ],
    )
    pd.DataFrame({"wd_source_id": wd[10:15], "ms_source_id": ms[10:15]}).to_parquet(
        cat / "wdms.parquet"
    )
    wdcomp = take(1)  # companion id not in any pair list: float superset rule
    comp = np.r_[ms[:4].astype(np.float64), np.nan, wdcomp.astype(np.float64)]
    pd.DataFrame(
        {"source_id": np.r_[wd[:5], wd[15]], "companion_source_id": comp}
    ).to_parquet(cat / "wdages.parquet")
    args = [
        "--gaia", str(gaia),
        "--hr23-members", str(cat / "members.fits"),
        "--hr23-clusters", str(cat / "clusters.fits"),
        "--eb21", str(cat / "eb21.parquet"),
        "--gbs", str(cat / "gbs.parquet"),
        "--vb21-dir", str(vb),
        "--vb21-extra", str(cat / "gc_extra.parquet"),
        "--age-clusters", str(cat / "priors.csv"),
        "--kepler-legacy", str(cat / "legacy.fits"),
        "--kepler-gaia", str(cat / "kepler_gaia.fits"),
        "--wdms", str(cat / "wdms.parquet"),
        "--wd-ages", str(cat / "wdages.parquet"),
    ]  # fmt: skip
    return gaia, args, dict(
        cl_a=cl_a, cl_b=cl_b, cl_bench=cl_bench, gbs=gbs, gc=gc, gcx=gcx,
        kic=kic_ids, wd=wd, ms=ms, pairs=(a, b), wdcomp=wdcomp[0],
    )  # fmt: skip


def test_end_to_end_freeze(tmp_path):
    rng = np.random.default_rng(7)
    gaia, args, f = _fixture(tmp_path, rng)
    out = tmp_path / "out"
    common = args + [
        "--output",
        str(out),
        "--workers",
        "2",
        "--fraction-tolerance",
        "0.2",
    ]
    assert S.main(common) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["status"] == "msa_split_v1_complete"
    assert man["salt64"] == "0xb6f6e5353645106b"
    v = man["verification"]
    assert (
        v["ok"]
        and v["rows"] == 40_000
        and v["groups"]["member_partition_mismatches"] == 0
    )
    a = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in sorted((out / "assignment").rglob("Npix=*.parquet"))
        ]
    )
    src = pd.concat(
        [
            pq.read_table(p, columns=["source_id"]).to_pandas()
            for p in sorted((gaia / "dataset").rglob("*.parquet"))
        ]
    )
    assert np.array_equal(
        a.source_id.to_numpy(), src.source_id.to_numpy()
    )  # row order kept
    assert len(a) == 40_000 and a.source_id.is_unique
    part = dict(zip(a.source_id, a.partition, strict=True))
    groups = pd.read_parquet(out / "groups.parquet")
    over = pd.read_parquet(out / "overrides.parquet").set_index("source_id")
    # non-member, non-reserved rows keep the pure hash partition
    plain = ~a.source_id.isin(over.index)
    assert np.array_equal(
        a.partition[plain].to_numpy(),
        S.partition_of(S.unit(a.source_id[plain].to_numpy())),
    )
    # clusters move as one; the companion of a cluster member follows the cluster
    # (the first five Melotte_22 members are more probable NGC_1 members)
    for ids in (f["cl_a"][5:], np.r_[f["cl_b"], f["cl_a"][:5]]):
        p = {part[s] for s in ids if not over.loc[s, "reserved"]}
        assert len(p) == 1
    pa_, pb = f["pairs"]
    for s1, s2 in zip(pa_[:-1], pb[:-1], strict=True):
        if not (over.loc[s1, "reserved"] or over.loc[s2, "reserved"]):
            assert part[s1] == part[s2]
    melotte = groups.set_index("key").loc["grp:hr23:Melotte_22"]
    assert part[f["cl_a"][5]] == melotte.partition
    # reserved objects: never FIT/DEV/CAL/TEST
    reserved = np.r_[
        f["cl_bench"], f["gbs"], f["gc"][:20], f["gcx"][:5], f["kic"][[0, 2]]
    ]
    reserved = np.r_[reserved, f["wd"][:5], f["wd"][10:16], f["ms"][:4], f["ms"][10:15]]
    assert all(part[s] == S.RESERVED for s in reserved)
    assert part[f["gc"][25]] != S.RESERVED and part[f["kic"][1]] != S.RESERVED
    assert part[f["wdcomp"]] == S.RESERVED
    assert over.loc[f["wdcomp"], "reserved_reason"] == "wd_age_companion_float"
    assert man["reserved"]["wd_age_companion_float"]["added"] == 1
    res = man["reserved"]["per_reason"]
    assert res["wd_age_published"]["companions_resolved"] == 4
    assert res["wd_age_published"]["companions_unresolved"] == 1
    assert res["kepler_legacy"]["unresolved_kic"] == [103]
    assert v["group_members_not_in_gaia"] == 1  # the absent pair member
    counts = json.loads((out / "counts.json").read_text())["counts"]
    assert sum(counts["by_partition"].values()) == 40_000
    assert counts["by_partition"]["QA_RESERVED"] == int(over.reserved.sum())
    assert set(a.input_class.unique()) <= {0, 1, 2, 3}
    # a complete freeze is never overwritten; a forced rerun resumes and agrees
    mtime = (out / "manifest.json").stat().st_mtime_ns
    assert S.main(common) == 0
    assert (out / "manifest.json").stat().st_mtime_ns == mtime
    assert S.main(common + ["--force"]) == 0
    b = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in sorted((out / "assignment").rglob("Npix=*.parquet"))
        ]
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
