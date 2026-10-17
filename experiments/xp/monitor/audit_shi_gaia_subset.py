"""Official Gaia audit of a bounded high-proper-motion Shi candidate subset."""

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.table import Table

root = (
    Path(__file__).resolve().parents[1]
    / "results-stellar-improvement/shi_collection_v1/feature-coordinate-candidates-v1"
)
a = np.load(root / "official-gaia-audit-input.npz")
ids = a["source_id"]
assert len(ids) == len(np.unique(ids)) == 128 and ids.dtype.kind == "i"


def query(name, sql):
    path = root / (name + ".xml")
    (root / (name + ".adql")).write_text(sql + "\n")
    if not path.exists():
        response = requests.post(
            "https://gea.esac.esa.int/tap-server/tap/sync",
            data=dict(
                REQUEST="doQuery",
                LANG="ADQL",
                FORMAT="votable",
                MAXREC="2000",
                QUERY=sql,
            ),
            timeout=(20, 60),
        )
        (root / (name + ".http-status.json")).write_text(
            json.dumps(dict(status=response.status_code, url=response.url)) + "\n"
        )
        path.write_bytes(response.content)
        response.raise_for_status()
    status = [
        n.attrib.get("value")
        for n in ET.fromstring(path.read_bytes()).iter()
        if n.tag.split("}")[-1] == "INFO" and n.attrib.get("name") == "QUERY_STATUS"
    ]
    assert status and set(status) == {"OK"}, status
    return Table.read(io.BytesIO(path.read_bytes()), format="votable")


g = query(
    "official-gaia-identities",
    "SELECT source_id,ra,dec,pmra,pmdec,ref_epoch FROM gaiadr3.gaia_source WHERE source_id IN ("
    + ",".join(map(str, ids))
    + ")",
)
assert len(g) == 128 and g["SOURCE_ID"].dtype.kind == "i"
lookup = {int(v): i for i, v in enumerate(g["SOURCE_ID"])}
g = g[[lookup[int(v)] for v in ids]]
assert (
    (g["ref_epoch"] == 2016).all()
    and g["ra"].unit.is_equivalent(u.deg)
    and g["pmra"].unit.is_equivalent(u.mas / u.yr)
)
gaia = SkyCoord(ra=g["ra"].quantity, dec=g["dec"].quantity)
shi = SkyCoord(a["shi_ra"] * u.deg, a["shi_dec"] * u.deg)
feature = SkyCoord(a["feature_ra"] * u.deg, a["feature_dec"] * u.deg)
sep = shi.separation(gaia).arcsec
feature_sep = feature.separation(gaia).arcsec
conditions = [
    "1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',"
    + repr(float(ra))
    + ","
    + repr(float(dec))
    + ",0.000555555555555556))"
    for ra, dec in zip(a["shi_ra"], a["shi_dec"])
]
n = query(
    "official-gaia-global-neighbors",
    "SELECT source_id,ra,dec FROM gaiadr3.gaia_source WHERE "
    + " OR ".join("(" + c + ")" for c in conditions),
)
neighbor = SkyCoord(n["ra"].quantity, n["dec"].quantity)
other_nearest = []
for coord, sid in zip(shi, ids):
    distance = coord.separation(neighbor).arcsec
    own = np.asarray(n["SOURCE_ID"]) == sid
    assert own.sum() == 1
    other = distance[(~own) & (distance <= 2)]
    other_nearest.append(float(other.min()) if len(other) else np.nan)
other_nearest = np.asarray(other_nearest)
accepted = (sep <= 0.2) & ~np.isfinite(other_nearest)
np.savez_compressed(
    root / "official-subset-associations.npz",
    **{key: a[key] for key in a.files},
    gaia_ra=np.asarray(g["ra"]),
    gaia_dec=np.asarray(g["dec"]),
    gaia_ref_epoch=np.asarray(g["ref_epoch"]),
    shi_gaia_separation_arcsec=sep,
    feature_gaia_separation_arcsec=feature_sep,
    nearest_other_within2arcsec=other_nearest,
    subset_candidate_pass=accepted,
)
report = dict(
    audited_subset=128,
    global_gaia_neighbors_returned=len(n),
    gaia_epoch=2016.0,
    ra_dec_units="degrees verified from official Gaia metadata and coordinate agreement for this subset",
    min_proper_motion_mas_year=float(np.hypot(g["pmra"], g["pmdec"]).min()),
    max_shi_gaia_separation_arcsec=float(sep.max()),
    median_shi_gaia_separation_arcsec=float(np.median(sep)),
    max_feature_gaia_separation_arcsec=float(feature_sep.max()),
    subset_candidates_with_other_gaia_within2arcsec=int(
        np.isfinite(other_nearest).sum()
    ),
    subset_candidate_pass=int(accepted.sum()),
    interpretation="Bounded128 high-proper-motion candidate audit only. Shi FITS does not declare units/epoch; position consistency with GaiaDR3 ref_epoch2016 is observational evidence, not invented publisher metadata. No full82852 or14.5M identity certification. XP metallicity remains weak selection, never independent XP truth.",
    fit_eligible=False,
    query_sha256={
        name: hashlib.sha256((root / (name + ".xml")).read_bytes()).hexdigest()
        for name in ["official-gaia-identities", "official-gaia-global-neighbors"]
    },
)
(root / "official-gaia-subset-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
