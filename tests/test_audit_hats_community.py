"""Read-only HATS community audit (``scripts/audit_hats_community.py``) on a tiny tree:
a Gaia anchor partition and one survey with a missing partition, an out-of-range RA, a
duplicated id and a row at the same position twice. Checks every stage's verdict and
that nothing is written under the audited root."""

import importlib.util
import json
import os
from pathlib import Path

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = importlib.util.spec_from_file_location(
    "audit_hats_community",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_hats_community.py",
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _write_tree(base: Path, name: str, props: dict, parts: dict) -> None:
    root = base / name
    (root / "dataset").mkdir(parents=True)
    info = props.pop("_info", list(parts))
    (root / "properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
    with open(root / "partition_info.csv", "w") as fh:
        fh.write("Norder,Npix\n")
        for o, p in info:
            fh.write(f"{o},{p}\n")
    for (o, p), table in parts.items():
        d = root / "dataset" / f"Norder={o}" / f"Dir={p // 10000 * 10000}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / f"Npix={p}.parquet")


def _props(n, ra="ra", dec="dec", order=0):
    return dict(
        obs_collection="t",
        dataproduct_type="object",
        hats_nrows=n,
        hats_col_ra=ra,
        hats_col_dec=dec,
        hats_order=order,
        hats_version="v1.0",
        hats_builder="test",
        hats_creation_date="2026-11-24",
    )


@pytest.fixture
def tree(tmp_path):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(300_000, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    ra = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    dec = np.degrees(np.arcsin(v[:, 2]))
    keep = hp.ang2pix(1, ra, dec, nest=True, lonlat=True) == 4
    ra, dec = ra[keep][:3000], dec[keep][:3000]
    h29 = hp.ang2pix(2**29, ra, dec, nest=True, lonlat=True).astype(np.int64)
    gaia = pa.table(
        dict(
            _healpix_29=h29,
            source_id=np.arange(len(ra), dtype=np.int64),
            ra=ra,
            dec=dec,
            pmra=np.zeros(len(ra)),
            pmdec=np.zeros(len(ra)),
            phot_g_mean_mag=np.full(len(ra), 18.0),
            bp_rp=np.ones(len(ra)),
        )
    )
    hats = tmp_path / "hats"
    _write_tree(hats / "gaia", "dr3", _props(len(ra)), {(0, 4): gaia})
    # survey: rows of child pixel 17 offset by 0.2", one duplicate id, one twin row
    child = hp.ang2pix(2, ra, dec, nest=True, lonlat=True) == 17
    sra, sdec = ra[child] + 0.2 / 3600, dec[child].copy()
    sid = np.arange(len(sra), dtype=np.int64)
    sid[1] = sid[0]
    sra = np.append(sra, sra[5])
    sdec = np.append(sdec, sdec[5])
    sid = np.append(sid, 10**6)
    sra[2] = 360.0
    survey = pa.table(
        dict(
            _healpix_29=hp.ang2pix(
                2**29, sra % 360, sdec, nest=True, lonlat=True
            ).astype(np.int64),
            sourceid=sid,
            ra=sra,
            dec=sdec,
            mag=np.full(len(sra), -999999500.0),
            note=np.array(["1.5"] * len(sra)),
        )
    )
    p = _props(len(sra), order=1)
    p["_info"] = [(1, 17), (1, 18)]
    _write_tree(hats / "vhs", "dr7", p, {(1, 17): survey})
    return hats, tmp_path / "out"


def test_audit_end_to_end(tree):
    hats, out = tree
    before = sorted((str(p), p.stat().st_mtime_ns) for p in hats.rglob("*"))
    assert (
        audit.main(
            [
                "--root",
                str(hats),
                "--output",
                str(out),
                "--workers",
                "2",
                "--small-rows",
                "1e9",
            ]
        )
        == 0
    )
    assert sorted((str(p), p.stat().st_mtime_ns) for p in hats.rglob("*")) == before
    disc = json.loads((out / "discovery.json").read_text())
    assert disc["catalogues"] == ["gaia/dr3", "vhs/dr7"]
    s = json.loads((out / "structure" / "vhs__dr7.json").read_text())
    assert s["missing_on_disk"] == ["1/18"] and s["rows_match_hats_nrows"]
    assert s["n_partitions_coord_stats_out_of_range"] == 1
    assert "mag" in s["sentinel_like_min_max"]
    sm = json.loads((out / "samples" / "vhs__dr7.json").read_text())
    part = sm["partitions"][0]
    assert part["coords"]["out_of_range"] == 1
    assert part["duplicate_id"]["rows_with_dup"] == 2
    assert part["nearest_neighbour"]["n_exact_0p001"] == 2
    assert part["columns"]["note"]["numeric_scalar_frac"] == 1.0
    assert part["gaia"]["frac_gaia_matched"] > 0.95 and part["gaia"][
        "pixel"
    ].startswith("6/")
    assert 0.1 < part["gaia"]["median_sep_arcsec"] < 0.21
    fs = json.loads((out / "fullscan" / "vhs__dr7.json").read_text())
    assert (
        fs["complete"]
        and fs["totals"]["out_of_range"] == 1
        and fs["totals"]["radec_outside"] == 0
    )


def test_refuses_output_under_root(tree):
    hats, _ = tree
    with pytest.raises(SystemExit):
        audit.main(["--root", str(hats), "--output", os.path.join(hats, "x")])
