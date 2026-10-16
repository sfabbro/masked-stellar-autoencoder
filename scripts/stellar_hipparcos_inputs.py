"""Bounded official Hipparcos/Gaia acquisition for distance-reference cohorts."""

import gzip
import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import Table, vstack
from hipparcos_reference import ordinary_reference_mask, propagate_position
from pristine_quality_views import exact_rows

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
RAW_SHA = {
    "hip2.dat.gz": "8e624f843d4254a9b7c2e8dda8e3158dbe825bf98f6bf3dbbae7f0d0b73d6858",
    "ReadMe": "6c017925d658447d40983da3b459ba16bf9789c8f7573e263952c75692622499",
}
MAP_QUERY = "SELECT source_id,original_ext_source_id,angular_distance,number_of_neighbours,xm_flag FROM gaiadr3.hipparcos2_best_neighbour"
ASTROMETRY_QUERY = "SELECT x.source_id,x.original_ext_source_id,g.ra,g.dec,g.pmra,g.pmdec,g.ref_epoch,g.ruwe,g.astrometric_params_solved FROM gaiadr3.hipparcos2_best_neighbour AS x JOIN gaiadr3.gaia_source AS g ON x.source_id=g.source_id"
RANGES = [
    "x.original_ext_source_id<40000",
    "x.original_ext_source_id>=40000 AND x.original_ext_source_id<80000",
    "x.original_ext_source_id>=80000",
]


def complete_votable(path):
    """Check every status, including errors appended AFTER an initial OK/table."""
    status = [
        node.attrib.get("value")
        for node in ET.fromstring(Path(path).read_bytes()).iter()
        if node.tag.split("}")[-1] == "INFO"
        and node.attrib.get("name") == "QUERY_STATUS"
    ]
    if not status or set(status) != {"OK"}:
        raise ValueError(f"Incomplete Gaia query {path}: {status}")
    return Table.read(path, format="votable")


def unique_positive_integer(column, name):
    values = np.asarray(column)
    if (
        values.dtype.kind not in "iu"
        or np.ma.getmaskarray(column).any()
        or (values <= 0).any()
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(f"Invalid or duplicate integer {name}")
    return values.astype("i8")


def download(path, url, limit):
    """One bounded read; preserve the actual response before scientific parsing."""
    if path.exists():
        raise FileExistsError(path)
    request = urllib.request.Request(url, headers={"User-Agent": "MSA-reference-QA/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read(limit + 1)
        path.write_bytes(payload)
        (path.with_suffix(path.suffix + ".http.json")).write_text(
            json.dumps(
                dict(url=url, status=response.status, headers=dict(response.headers)),
                indent=2,
            )
        )
    if len(payload) > limit:
        raise ValueError("Response exceeds bounded acquisition size")


def acquire_geometry(destination, feature_ids, train, validation, cached=False):
    """Return all 99525 mapped references; quality never cuts on parallax SNR/sign.

    cached=True is for independent local reproduction from preserved official bytes.
    Production always fetches fresh raw responses into an exclusive output directory.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=cached)
    for name, expected in RAW_SHA.items():
        p = destination / name
        if not cached:
            download(p, "https://cdsarc.cds.unistra.fr/ftp/I/311/" + name, 10000000)
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Corrected official Hipparcos release checksum changed")
    queries = {
        "gaia-hip2-xmatch": MAP_QUERY,
        "gaia-hip2-count": "SELECT COUNT(*) AS n FROM gaiadr3.hipparcos2_best_neighbour",
        **{
            f"gaia-hip2-astrometry-chunk-{i}": ASTROMETRY_QUERY + " WHERE " + condition
            for i, condition in enumerate(RANGES)
        },
    }
    tables = {}
    for name, query in queries.items():
        p = destination / (name + ".xml")
        if not cached:
            (destination / (name + ".adql")).write_text(query + "\n")
            url = (
                TAP
                + "?"
                + urllib.parse.urlencode(
                    dict(
                        REQUEST="doQuery",
                        LANG="ADQL",
                        FORMAT="votable",
                        MAXREC=200000,
                        QUERY=query,
                    )
                )
            )
            download(p, url, 60000000)
        tables[name] = complete_votable(p)
    count = tables["gaia-hip2-count"]
    x = tables["gaia-hip2-xmatch"]
    g = vstack([tables[f"gaia-hip2-astrometry-chunk-{i}"] for i in range(3)])
    if (
        len(count) != 1
        or int(count["n"][0]) != 99525
        or len(x) != 99525
        or len(g) != 99525
    ):
        raise ValueError("Full corrected Hipparcos/Gaia release row count changed")
    ids = unique_positive_integer(x["SOURCE_ID"], "Gaia map ID")
    hip = unique_positive_integer(x["original_ext_source_id"], "HIP map ID")
    gids = unique_positive_integer(g["SOURCE_ID"], "Gaia astrometry ID")
    lookup = exact_rows(gids, ids)
    if (lookup < 0).any():
        raise ValueError("Missing Gaia astrometry identities")
    g = g[lookup]
    np.testing.assert_array_equal(g["original_ext_source_id"], hip)
    if not (g["ref_epoch"] == 2016).all():
        raise ValueError("Unexpected Gaia epoch")
    for name in ["ra", "dec"]:
        if g[name].unit != u.deg or np.ma.getmaskarray(g[name]).any():
            raise ValueError("Gaia position units/missingness changed")
    schema = (destination / "ReadMe").read_text()
    if schema.count("15F7.2") != 1:
        raise ValueError("Unexpected Hipparcos weight schema")
    (destination / "ReadMe.parser").write_text(schema.replace("15F7.2", "A105  "))
    data = gzip.decompress((destination / "hip2.dat.gz").read_bytes())
    (destination / "hip2.dat").write_bytes(data)
    t = ascii.read(
        destination / "hip2.dat",
        format="cds",
        readme=str(destination / "ReadMe.parser"),
    )
    lines = data.decode("ascii").splitlines()
    if len(t) != 117955 or len(np.unique(t["HIP"])) != len(t):
        raise ValueError("Raw corrected Hipparcos count/uniqueness changed")
    for name, start, stop in [
        ("HIP", 0, 6),
        ("RArad", 15, 28),
        ("DErad", 29, 42),
        ("Plx", 43, 50),
        ("e_Plx", 83, 89),
    ]:
        np.testing.assert_array_equal(
            t[name], np.array([float(line[start:stop]) for line in lines])
        )
    for name, unit in [
        ("RArad", u.rad),
        ("DErad", u.rad),
        ("Plx", u.mas),
        ("e_Plx", u.mas),
        ("pmRA", u.mas / u.yr),
        ("pmDE", u.mas / u.yr),
    ]:
        if t[name].unit != unit:
            raise ValueError("Hipparcos units changed: " + name)
    t["UW"] = np.array(
        [[float(line[171 + 7 * j : 178 + 7 * j]) for j in range(15)] for line in lines]
    )
    t.write(destination / "raw-hip2-parsed.fits", overwrite=cached)
    rows = exact_rows(np.asarray(t["HIP"], dtype="i8"), hip)
    if (rows < 0).any():
        raise ValueError("Gaia map HIP missing from original catalogue")
    a, d, pmra, pmdec = [
        np.asarray(t[n])[rows] for n in ["RArad", "DErad", "pmRA", "pmDE"]
    ]
    moved = propagate_position(a, d, pmra, pmdec, 24.75)
    ar, dr = np.deg2rad(g["ra"]), np.deg2rad(g["dec"])
    xyz = np.column_stack(
        [np.cos(dr) * np.cos(ar), np.cos(dr) * np.sin(ar), np.sin(dr)]
    )
    separation = (
        np.rad2deg(
            np.arctan2(
                np.linalg.norm(np.cross(moved, xyz), axis=1),
                np.einsum("ij,ij->i", moved, xyz),
            )
        )
        * 3600
    )
    parallax, error = np.asarray(t["Plx"])[rows], np.asarray(t["e_Plx"])[rows]
    sn, nc, f2, f1 = [np.asarray(t[n])[rows] for n in ["Sn", "Nc", "F2", "F1"]]
    neighbors, flag = np.asarray(x["number_of_neighbours"]), np.asarray(x["xm_flag"])
    base = (
        (sn == 5)
        & (nc == 1)
        & (f2 < 3)
        & (f1 <= 10)
        & np.isfinite(parallax)
        & np.isfinite(error)
        & (error > 0)
        & (neighbors == 1)
    )
    ordinary = ordinary_reference_mask(sn, nc, f2, f1, parallax, error, neighbors, flag)
    geometry = dict(
        source_id=ids,
        hipparcos_id=hip,
        hip_source_row=rows,
        hipparcos_parallax_mas=parallax.astype("f8"),
        hipparcos_error_mas=error.astype("f8"),
        sn=sn,
        nc=nc,
        f2=f2,
        f1=f1,
        gaia_crossmatch_angular_distance_arcsec=np.asarray(x["angular_distance"]),
        number_of_gaia_neighbors=neighbors,
        xm_flag=flag,
        conservative_reference_candidate=base,
        feature_id_present=np.isin(ids, feature_ids),
        benchmark_train=np.isin(ids, train),
        benchmark_validation=np.isin(ids, validation),
        gaia_ra_deg=np.asarray(g["ra"]),
        gaia_dec_deg=np.asarray(g["dec"]),
        gaia_ruwe=np.asarray(g["ruwe"]),
        gaia_astrometric_params_solved=np.asarray(g["astrometric_params_solved"]),
        epoch_corrected_separation_arcsec=separation,
        ordinary_xm_flag_reference=ordinary,
        position_checked_reference_candidate=ordinary
        & np.isfinite(separation)
        & (separation <= 0.5),
    )
    provenance = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in destination.iterdir()
        if p.is_file()
    }
    (destination / "input-sha256.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return geometry
