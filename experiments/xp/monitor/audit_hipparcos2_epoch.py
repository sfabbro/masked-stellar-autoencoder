"""Full matched-row Hipparcos/Gaia epoch and flag audit, no parallax tuning."""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from hipparcos_reference import ordinary_reference_mask, propagate_position

ROOT = (
    Path(__file__).resolve().parents[1]
    / "results-stellar-improvement/hipparcos2-reference-v1"
)
x = np.load(ROOT / "geometric-reference-candidates.npz")
t = Table.read(ROOT / "raw-hip2-parsed.fits")
# The first unchunked sync result ended in QUERY_STATUS=ERROR at90113rows
# (server10second execution limit). Preserve it; consume only three complete
# disjoint ranges whose combined IDs/count match the original full map.
chunks = []
for number in range(3):
    p = ROOT / ("gaia-hip2-astrometry-chunk-" + str(number) + ".xml")
    status = [
        n.attrib.get("value")
        for n in ET.fromstring(p.read_bytes()).iter()
        if n.tag.split("}")[-1] == "INFO" and n.attrib.get("name") == "QUERY_STATUS"
    ]
    assert status and set(status) == {"OK"}
    chunks.append(Table.read(p, format="votable"))
g = vstack(chunks)
assert len(np.unique(g["SOURCE_ID"])) == len(g)
assert len(g) == 99525 and g["SOURCE_ID"].dtype.kind == "i"
lookup = {int(v): i for i, v in enumerate(g["SOURCE_ID"])}
g = g[[lookup[int(v)] for v in x["source_id"]]]
np.testing.assert_array_equal(g["original_ext_source_id"], x["hipparcos_id"])
assert (g["ref_epoch"] == 2016).all() and g["ra"].unit.is_equivalent(u.deg)
rows = x["hip_source_row"]
a = np.asarray(t["RArad"])[rows]
d = np.asarray(t["DErad"])[rows]
pmra = np.asarray(t["pmRA"])[rows]
pmdec = np.asarray(t["pmDE"])[rows]
v = propagate_position(a, d, pmra, pmdec, 24.75)
gaia = SkyCoord(ra=g["ra"].quantity, dec=g["dec"].quantity)
xyz = gaia.cartesian.xyz.value.T
sep = (
    np.rad2deg(
        np.arctan2(
            np.linalg.norm(np.cross(v, xyz), axis=1), np.einsum("ij,ij->i", v, xyz)
        )
    )
    * 3600
)
# Independent propagation audit uniformly spans every published map row.
fixture = np.random.default_rng(42).choice(len(g), 1024, replace=False)
c = SkyCoord(
    ra=a[fixture] * u.rad,
    dec=d[fixture] * u.rad,
    pm_ra_cosdec=pmra[fixture] * u.mas / u.yr,
    pm_dec=pmdec[fixture] * u.mas / u.yr,
    obstime=Time(1991.25, format="jyear"),
)
reference = (
    c.apply_space_motion(Time(2016.0, format="jyear")).separation(gaia[fixture]).arcsec
)
np.testing.assert_allclose(sep[fixture], reference, atol=1e-4, rtol=1e-5)
ordinary = ordinary_reference_mask(
    x["sn"],
    x["nc"],
    x["f2"],
    x["f1"],
    x["hipparcos_parallax_mas"],
    x["hipparcos_error_mas"],
    x["number_of_gaia_neighbors"],
    x["xm_flag"],
)
# Identity sanity threshold is declared before any parallax comparison.
# High-motion/perspective/covariance outliers stay raw and require separate review.
clean = ordinary & np.isfinite(sep) & (sep <= 0.5)
np.savez_compressed(
    ROOT / "geometric-reference-quality-v2.npz",
    **{key: x[key] for key in x.files},
    gaia_ra_deg=np.asarray(g["ra"]),
    gaia_dec_deg=np.asarray(g["dec"]),
    gaia_ruwe=np.asarray(g["ruwe"]),
    gaia_astrometric_params_solved=np.asarray(g["astrometric_params_solved"]),
    epoch_corrected_separation_arcsec=sep,
    ordinary_xm_flag_reference=ordinary,
    position_checked_reference_candidate=clean,
)
# Demonstrate why the15weight elements are not used as absolute covariance.
w = np.asarray(t["UW"])[rows[:128]]
mat = np.zeros((len(w), 5, 5))
k = 0
for col in range(5):
    for row in range(col + 1):
        mat[:, row, col] = w[:, k]
        k += 1
C = np.linalg.inv(np.transpose(mat, (0, 2, 1)) @ mat)
naive_sigma = np.sqrt(np.diagonal(C, axis1=1, axis2=2))
native_sigma = np.column_stack(
    [
        np.asarray(t[key])[rows[:128]]
        for key in ["e_RArad", "e_DErad", "e_Plx", "e_pmRA", "e_pmDE"]
    ]
)
ratio = naive_sigma / native_sigma
assert not np.allclose(ratio, 1, rtol=0.05)
report = dict(
    actual_matched_gaia_rows=len(g),
    all_hip_gaia_identifiers_exact=True,
    gaia_epoch=2016.0,
    hip_epoch=1991.25,
    all1024_independent_epoch_checks_pass=True,
    ordinary_flag8_reference_candidates=int(ordinary.sum()),
    ordinary_candidates_rejected_by_position_sanity=int((ordinary & ~clean).sum()),
    position_checked_reference_candidates=int(clean.sum()),
    retained_nonpositive_parallaxes=int(
        (clean & (x["hipparcos_parallax_mas"] <= 0)).sum()
    ),
    feature_id_present=int((clean & x["feature_id_present"]).sum()),
    benchmark_train=int((clean & x["benchmark_train"]).sum()),
    benchmark_validation=int((clean & x["benchmark_validation"]).sum()),
    ordinary_separation_quantiles_arcsec=np.quantile(
        sep[ordinary], [0.5, 0.9, 0.99, 1]
    ).tolist(),
    xm_flag_semantics="OfficialESA20.4.10:8ordinaryfiveparameter;16twoparameter;72=8+64specialtreatmentforlargeRUWE/ipd_gof_harmonic_amplitude. Maskv2requires8,retainsallothersraw.",
    covariance_caution="NaiveC=inv(UtransposeU)from15publishedcomponents fails to reproduce nativeformalerrors:128row median_naive/native="
    + str(float(np.median(ratio)))
    + ". Normalization/scaling convention needs further authoritative resolution; DO NOT infer normalized positional sigmas, covariance-calibrated identity probabilities or total parallaxaccuracy from this matrix alone.",
    policy="Versioned DESIGN Sn5/Nc1/F2<3/F1<=10,positivefiniteformalerror,oneGaiagoodneighbor,xm_flag8 plus<=0.5arcsec mean-position sanity after2Dzero-RVpropagation. No parallaxSNR/positivity/residualcut. Perspective motion/astrometriccovariance/spatial/binary systematics unresolved. Not full global sky isolation. No distancepredictionvalidationor beyond-Gaia claim. Fiteligiblefalse.",
    fit_eligible=False,
)
(ROOT / "epoch-and-quality-v2-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
