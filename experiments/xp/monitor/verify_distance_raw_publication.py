import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import ascii

sys.path.insert(0, "scripts")
from pristine_quality_views import exact_rows
from stellar_hipparcos_inputs import RAW_SHA, complete_votable

out = Path("experiments/xp/results-stellar-improvement/distance_feature_cohort_v1")
raw = out / "raw-reference"
for name, digest in RAW_SHA.items():
    assert hashlib.sha256((raw / name).read_bytes()).hexdigest() == digest
for name in ["hip2.dat.gz", "ReadMe"]:
    assert (raw / name).read_bytes() == (
        out.parent / "hipparcos2-reference-v1" / name
    ).read_bytes()
t = ascii.read(raw / "hip2.dat", format="cds", readme=str(raw / "ReadMe.parser"))
lines = (raw / "hip2.dat").read_text().splitlines()
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
weights = np.array(
    [[float(line[171 + 7 * j : 178 + 7 * j]) for j in range(15)] for line in lines]
)
from astropy.table import Table, vstack

parsed = Table.read(raw / "raw-hip2-parsed.fits")
np.testing.assert_array_equal(parsed["UW"], weights)
x = complete_votable(raw / "gaia-hip2-xmatch.xml")
count = complete_votable(raw / "gaia-hip2-count.xml")
g = vstack(
    [complete_votable(raw / f"gaia-hip2-astrometry-chunk-{i}.xml") for i in range(3)]
)
assert int(count["n"][0]) == 99525 and len(x) == len(g) == 99525
with h5py.File(out / "distance-feature-cohort.h5") as f:
    r = f["hipparcos_reference"]
    ids = r["source_id"][:]
    xr = exact_rows(x["SOURCE_ID"], ids)
    gr = exact_rows(g["SOURCE_ID"], ids)
    hr = exact_rows(np.asarray(t["HIP"], dtype="i8"), r["hipparcos_id"][:])
    assert (xr >= 0).all() and (gr >= 0).all() and (hr >= 0).all()
    for name, col in [
        ("hipparcos_id", "original_ext_source_id"),
        ("xm_flag", "xm_flag"),
        ("number_of_gaia_neighbors", "number_of_neighbours"),
    ]:
        np.testing.assert_array_equal(r[name][:], x[col][xr])
    for name, col in [
        ("hipparcos_parallax_mas", "Plx"),
        ("hipparcos_error_mas", "e_Plx"),
        ("sn", "Sn"),
        ("nc", "Nc"),
        ("f2", "F2"),
        ("f1", "F1"),
    ]:
        np.testing.assert_array_equal(r[name][:], t[col][hr])
    for name, col in [
        ("gaia_ra_deg", "ra"),
        ("gaia_dec_deg", "dec"),
        ("gaia_ruwe", "ruwe"),
        ("gaia_astrometric_params_solved", "astrometric_params_solved"),
    ]:
        np.testing.assert_array_equal(r[name][:], g[col][gr])
    assert (
        hashlib.sha256((raw / "input-sha256.json").read_bytes()).hexdigest()
        == f.attrs["hipparcos_geometry_input_sha256"]
    )
(out / "raw-source-independent-verification.json").write_text(
    json.dumps(
        dict(
            all_official_sha256s_exact=True,
            all5_raw_fixed_fields_exact=True,
            all15_raw_weight_components_exact=True,
            all_actual_votables_complete=True,
            all99525_raw_id_flags_measurement_associations_exact=True,
            independent_local_release_bytes_exact=True,
        ),
        indent=2,
    )
    + "\n"
)
print("All raw publication checks passed")
