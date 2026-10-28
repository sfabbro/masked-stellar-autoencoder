"""Preserve independent-instrument Hipparcos parallaxes; never invert noisy values."""

import gzip
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
from astropy.io import ascii, fits
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pristine_quality_views import exact_rows

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
root = ROOT / "hipparcos2-reference-v1"
with gzip.open(root / "hip2.dat.gz", "rb") as f:
    (root / "hip2.dat").write_bytes(f.read())
# Astropy CDS does not expand the15F7.2 descriptor. Preserve the official
# ReadMe; stage a parsing-only A105 text descriptor for that same105-byte
# field, then decode15components directly from ORIGINAL fixed-width bytes.
schema = (root / "ReadMe").read_text()
assert schema.count("15F7.2") == 1
(root / "ReadMe.parser").write_text(schema.replace("15F7.2", "A105  "))
t = ascii.read(root / "hip2.dat", format="cds", readme=str(root / "ReadMe.parser"))
lines = (root / "hip2.dat").read_text().splitlines()
weights = np.array(
    [[float(line[171 + 7 * j : 178 + 7 * j]) for j in range(15)] for line in lines],
    dtype="f8",
)
assert weights.shape == (117955, 15)
t["UW"] = weights
assert len(t) == 117955 and len(np.unique(t["HIP"])) == len(t)
assert str(t["RArad"].unit) == "rad" and str(t["DErad"].unit) == "rad"
assert str(t["Plx"].unit) == "mas" and str(t["e_Plx"].unit) == "mas"
assert str(t["pmRA"].unit) == "mas / yr"
status = [
    n.attrib.get("value")
    for n in ET.fromstring((root / "gaia-hip2-xmatch.xml").read_bytes()).iter()
    if n.tag.split("}")[-1] == "INFO" and n.attrib.get("name") == "QUERY_STATUS"
]
assert status and set(status) == {"OK"}
x = Table.read(root / "gaia-hip2-xmatch.xml", format="votable")
# ESA's actual table has one more row than the AIP mirror metadata.
# Reconcile against an independent official COUNT, never silently truncate.
count_table = Table.read(root / "gaia-hip2-count.xml", format="votable")
assert (
    int(count_table["n"][0]) == 99525
    and len(x) == 99525
    and x["SOURCE_ID"].dtype.kind == "i"
)
np.testing.assert_array_equal(
    np.asarray(t["HIP"]), np.array([int(line[:6]) for line in lines])
)
for name, start, stop in [
    ("RArad", 15, 28),
    ("DErad", 29, 42),
    ("Plx", 43, 50),
    ("e_Plx", 83, 89),
]:
    np.testing.assert_array_equal(
        np.asarray(t[name]), np.array([float(line[start:stop]) for line in lines])
    )
ids = np.asarray(x["SOURCE_ID"], dtype="i8")
hip = np.asarray(x["original_ext_source_id"], dtype="i8")
rows = exact_rows(np.asarray(t["HIP"], dtype="i8"), hip)
assert (rows >= 0).all() and (ids > 0).all()
unique_ids, counts = np.unique(ids, return_counts=True)
repeat = unique_ids[counts > 1]
parallax = np.asarray(t["Plx"], dtype="f8")[rows]
error = np.asarray(t["e_Plx"], dtype="f8")[rows]
base = (
    (np.asarray(t["Sn"])[rows] == 5)
    & (np.asarray(t["Nc"])[rows] == 1)
    & (np.asarray(t["F2"])[rows] < 3)
    & (np.asarray(t["F1"])[rows] <= 10)
    & np.isfinite(parallax)
    & np.isfinite(error)
    & (error > 0)
)
base &= (np.asarray(x["number_of_neighbours"]) == 1) & ~np.isin(ids, repeat)
assert len(np.unique(hip)) == len(hip)
with fits.open("/tmp/msa-labels-complete-20261015.fits", memmap=True) as f:
    featureids = np.asarray(f[1].data["source_id"], dtype="i8")
feature = np.isin(ids, featureids)
with h5py.File(ROOT / "dataset_view_v2/dataset-view.h5") as f:
    train = np.isin(ids, f["train/source_id"][:])
    val = np.isin(ids, f["validation/source_id"][:])
posterior = {}
with h5py.File(ROOT / "posterior_labels_v2/posterior-labels.h5") as f:
    for survey in f:
        match = exact_rows(
            f[survey + "/source_id"][:][f[survey + "/source_id"][:] > 0], ids
        )
        # Counts only; invalid ID0 never enters identity lookup.
        posterior[survey] = dict(
            all_positive_id_overlap=int((match >= 0).sum()),
            conservative_reference_overlap=int(((match >= 0) & base).sum()),
        )
arrays = dict(
    source_id=ids,
    hipparcos_id=hip,
    hip_source_row=rows,
    hipparcos_parallax_mas=parallax,
    hipparcos_error_mas=error,
    sn=np.asarray(t["Sn"])[rows],
    nc=np.asarray(t["Nc"])[rows],
    f2=np.asarray(t["F2"])[rows],
    f1=np.asarray(t["F1"])[rows],
    gaia_crossmatch_angular_distance_arcsec=np.asarray(x["angular_distance"]),
    number_of_gaia_neighbors=np.asarray(x["number_of_neighbours"]),
    xm_flag=np.asarray(x["xm_flag"]),
    conservative_reference_candidate=base,
    feature_id_present=feature,
    benchmark_train=train,
    benchmark_validation=val,
)
np.savez_compressed(root / "geometric-reference-candidates.npz", **arrays)
t.write(root / "raw-hip2-parsed.fits", overwrite=True)
report = dict(
    official_hip_rows=len(t),
    metadata_discrepancy="AIP mirror metadata99524 vs officialESA99525; rawquery rowcount independentlyconfirmed by officialCOUNT99525, allHIPIDs andGaiaIDs unique and everyHIPpresent in117955rawrows.",
    repeated_format_parser="Original ReadMe15F7.2 unsupported byAstropyCDS; parsing-only ReadMe.parser A105 field, then all15fixed7byte originalcomponents preserved. EveryHIP/RA/DEC/parallax/error independentlychecked fromfixedbyte offsets.",
    gaia_best_neighbor_rows=len(x),
    gaia_ids_unique=len(unique_ids),
    crossmatch_repeated_gaia_id_rows=int(np.isin(ids, repeat).sum()),
    raw_negative_parallaxes=int((np.asarray(t["Plx"]) < 0).sum()),
    raw_zero_parallaxes=int((np.asarray(t["Plx"]) == 0).sum()),
    conservative_reference_candidates=int(base.sum()),
    reference_candidates_negative_or_zero_parallax=int((base & (parallax <= 0)).sum()),
    reference_feature_id_present=int((base & feature).sum()),
    reference_benchmark_train=int((base & train).sum()),
    reference_benchmark_validation=int((base & val).sum()),
    posterior_overlaps=posterior,
    xm_flag_counts={
        str(int(k)): int(v)
        for k, v in zip(*np.unique(x["xm_flag"], return_counts=True))
    },
    quality_policy="Explicit conservative DESIGN, not publishergold: Sn==5(single five-parameter),Nc==1,F2<3,F1<=10,finitepositiveformalerror; uniqueGaiaID and exactlyoneGaia crossmatchneighbor. No positivity/SNRcut on parallax; retain negatives. xm_flag retained pending detailed semantics audit.",
    geometry_policy="Independent-instrument Hipparcos2 trigonometric parallaxes, not Gaia-conditioned/model-derived distance labels. Preserve mas/errors/flags and original rows. Do not invert noisy/negative parallaxes or call uncertainties total calibrated accuracy; spatial correlations/binaries/systematics and Gaia/frame/calibration lineage require audit. Evaluate in parallax space. No beyond-Gaia distance-performance claim.",
    identity_policy="Published official Gaia DR3 covariance-aware Hipparcos2 best-neighbour map, not naive unpropagated sky cone. Hip2 positions radians at epoch1991.25; Gaia DR3 epoch2016. Additional epoch/covariance/neighbor verification remains needed before certification.",
    fit_eligible=False,
    sha256={
        n: hashlib.sha256((root / n).read_bytes()).hexdigest()
        for n in ["ReadMe", "hip2.dat.gz", "gaia-hip2-xmatch.xml"]
    },
)
(root / "acquisition-and-coverage-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
