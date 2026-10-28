import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time

root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
out = root / "legacy_seismic_collection_v1"
p = out / "raw-source-download"
out.mkdir(exist_ok=True)
for src, dst in [
    ("ReadMe", "legacy-ReadMe"),
    ("table3.dat", "legacy-table3.dat"),
    ("table4.dat", "legacy-table4.dat"),
]:
    shutil.copyfile(p / src, out / dst)
for src in p.iterdir():
    if src.suffix in [".xml", ".txt", ".html"]:
        shutil.copyfile(src, out / src.name)
for file in [
    "simbad-response.xml",
    "gaia-response.xml",
    "gaia-neighborhood-constant-response.xml",
]:
    info = [
        n.attrib.get("value")
        for n in ET.fromstring((p / file).read_bytes()).iter()
        if n.tag.split("}")[-1] == "INFO" and n.attrib.get("name") == "QUERY_STATUS"
    ]
    assert info and set(info) == {"OK"}, (file, info)
a = ascii.read(p / "table3.dat", format="cds", readme=str(p / "ReadMe"))
b = ascii.read(p / "table4.dat", format="cds", readme=str(p / "ReadMe"))
s = Table.read(p / "simbad-response.xml", format="votable")
g = Table.read(p / "gaia-response.xml", format="votable")
n = Table.read(p / "gaia-neighborhood-constant-response.xml", format="votable")
assert len(a) == 66 and len(b) == 396 and len(s) == len(g) == 66 and len(n) < 1000
kic = np.array([int(str(v).removeprefix("KIC ")) for v in s["kic_id"]], dtype="i8")
ids = np.array(
    [int(str(v).removeprefix("Gaia DR3 ")) for v in s["gaia_id"]], dtype="i8"
)
assert (
    set(kic) == set(a["KIC"])
    and len(np.unique(ids)) == 66
    and (ids > 0).all()
    and g["SOURCE_ID"].dtype.kind == "i"
)
lookup = {int(v): i for i, v in enumerate(g["SOURCE_ID"])}
g = g[[lookup[int(sid)] for sid in ids]]
assert (
    (g["ref_epoch"] == 2016).all()
    and g["ra"].unit.is_equivalent(u.deg)
    and g["pmra"].unit.is_equivalent(u.mas / u.yr)
)
c = SkyCoord(
    ra=g["ra"].quantity,
    dec=g["dec"].quantity,
    pm_ra_cosdec=g["pmra"].quantity,
    pm_dec=g["pmdec"].quantity,
    obstime=Time(2016.0, format="jyear"),
)
old = c.apply_space_motion(Time(2000.0, format="jyear"))
sep = old.separation(SkyCoord(ra=s["ra"].quantity, dec=s["dec"].quantity)).arcsec
# Independent tangent-vector propagation (two-dimensional proper motion, zero RV).
al = np.deg2rad(g["ra"])
de = np.deg2rad(g["dec"])
factor = np.deg2rad(1 / 3600000) * -16
r = np.array([np.cos(de) * np.cos(al), np.cos(de) * np.sin(al), np.sin(de)]).T
e_a = np.array([-np.sin(al), np.cos(al), np.zeros(66)]).T
e_d = np.array([-np.sin(de) * np.cos(al), -np.sin(de) * np.sin(al), np.cos(de)]).T
v = r + factor * (
    np.asarray(g["pmra"])[:, None] * e_a + np.asarray(g["pmdec"])[:, None] * e_d
)
v /= np.linalg.norm(v, axis=1)[:, None]
ind = SkyCoord(x=v[:, 0], y=v[:, 1], z=v[:, 2], representation_type="cartesian")
assert old.separation(ind).arcsec.max() < 1e-5 and sep.max() < 0.5
neighbors = SkyCoord(ra=n["ra"].quantity, dec=n["dec"].quantity)
second = []
for coord, sid in zip(c, ids):
    distance = coord.separation(neighbors).arcsec
    own = np.asarray(n["SOURCE_ID"]) == sid
    assert own.sum() == 1 and distance[own].max() < 1e-5
    other = distance[(~own) & (distance <= 10.0)]
    second.append(float(other.min()) if len(other) else np.nan)
second = np.array(second)
np.savez_compressed(
    out / "audited-kic-gaia-map.npz",
    kic=kic,
    source_id=ids,
    epoch_corrected_separation_arcsec=sep,
    nearest_other_within_10arcsec=second,
    crowded_within1arcsec=np.isfinite(second) & (second < 1.0),
)
ids_by_kic = dict(zip(kic, ids))
model_ids = np.array([ids_by_kic[int(k)] for k in b["KIC"]], dtype="i8")
assert (
    b["Age"].unit == u.Gyr
    and b["Dist"].unit == u.pc
    and (b["e_Age"] > 0).all()
    and (b["E_Age"] > 0).all()
)
np.savez_compressed(
    out / "seismic-model-measurements.npz",
    source_id=model_ids,
    kic=np.array(b["KIC"]),
    pipeline=np.array(b["Pipe"]).astype("S6"),
    age_gyr=np.array(b["Age"]),
    age_lower_width_gyr=np.array(b["e_Age"]),
    age_upper_width_gyr=np.array(b["E_Age"]),
    distance_pc=np.array(b["Dist"]),
    distance_lower_width_pc=np.array(b["e_Dist"]),
    distance_upper_width_pc=np.array(b["E_Dist"]),
)
matched = {}
pairs = {}
with (
    h5py.File(root / "posterior_labels_v2/posterior-labels.h5") as f,
    h5py.File(root / "posterior_labels_v2/quality-candidates-v1.h5") as qc,
):
    for name in f:
        star_ids = f[name]["source_id"][:]
        common, li, ri = np.intersect1d(ids, star_ids, return_indices=True)
        matched[name] = dict(
            exact_ID_overlap=len(common),
            conservative_quality_overlap=int(
                qc[name]["conservative_age_candidate"][:][ri].sum()
            ),
        )
        pairs[name + "_source_id"] = common
        pairs[name + "_row"] = ri
        pairs[name + "_kic"] = kic[li]
        pairs[name + "_age_percentiles_gyr"] = f[name]["Age_percentiles"][:][ri]
np.savez_compressed(out / "starhorse-overlap.npz", **pairs)
report = dict(
    published_targets=66,
    model_rows=396,
    pipelines=list(map(str, np.unique(b["Pipe"]))),
    one_to_one_exact_identifier_crossrefs=True,
    official_Gaia_IDs_and_units_verified=True,
    coordinate_reference="SIMBAD ICRS epoch2000 per https://simbad.cds.unistra.fr/Pages/guide/ch15.htx ; Gaia DR3 ref_epoch2016; two-dimensional proper-motion propagation, no radial perspective term",
    independent_vector_propagation_agrees_arcsec=1e-5,
    epoch_corrected_sep_max_arcsec=float(sep.max()),
    neighbors_within1arcsec=int(sum(np.isfinite(second) & (second < 1))),
    neighborhood_rows=len(n),
    neighborhood_bound_arcsec=10,
    starhorse_overlap=matched,
    uncertainty="LEGACY lower/upper errors are positive widths (unlike APOKASC3 signed negative offsets); no pipeline averaging, six models are correlated. Distances are seismic/model-derived pc, not independent geometric truth",
    prediction_evaluation_performed=False,
    hashes={
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in out.iterdir()
        if file.suffix in [".dat", ".xml", ".npz"] or file.name == "legacy-ReadMe"
    },
)
(out / "join-independent-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps({k: v for k, v in report.items() if k != "hashes"}, indent=2))
