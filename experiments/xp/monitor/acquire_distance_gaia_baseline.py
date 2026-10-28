"""Preserve official raw Gaia parallaxes/errors; validate the existing FITS baseline."""

import json
import sys
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.io import fits
from astropy.table import vstack

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from urllib.parse import urlencode

from pristine_quality_views import exact_rows
from stellar_hipparcos_inputs import (
    RANGES,
    TAP,
    complete_votable,
    download,
    unique_positive_integer,
)

root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
out = root / "distance-gaia-baseline-v1"
out.mkdir(exist_ok=True)
chunks = []
for i, condition in enumerate(RANGES):
    query = (
        "SELECT x.source_id,x.original_ext_source_id,g.parallax,g.parallax_error FROM gaiadr3.hipparcos2_best_neighbour AS x JOIN gaiadr3.gaia_source AS g ON x.source_id=g.source_id WHERE "
        + condition
    )
    path = out / f"gaia-parallax-chunk-{i}.xml"
    if not path.exists():
        (out / f"gaia-parallax-chunk-{i}.adql").write_text(query + "\n")
        download(
            path,
            TAP
            + "?"
            + urlencode(
                dict(
                    REQUEST="doQuery",
                    LANG="ADQL",
                    FORMAT="votable",
                    MAXREC=200000,
                    QUERY=query,
                )
            ),
            30000000,
        )
    chunks.append(complete_votable(path))
g = vstack(chunks)
ids = unique_positive_integer(g["SOURCE_ID"], "Gaia baseline ID")
assert (
    len(g) == 99525
    and g["parallax"].unit == u.mas
    and g["parallax_error"].unit == u.mas
)
with h5py.File(root / "distance_feature_cohort_v1/distance-feature-cohort.h5") as f:
    ref = f["hipparcos_reference"]
    allids = ref["source_id"][:]
    take = np.flatnonzero(ref["evaluation_candidate"][:])
    refids = allids[take]
    rows = ref["feature_row"][:][take]
    assert len(take) == 4166
match = exact_rows(ids, allids)
assert (match >= 0).all()
with fits.open("/tmp/msa-labels-complete-20261015.fits", memmap=True) as f:
    np.testing.assert_array_equal(f[1].data["source_id"][rows], refids)
    legacy = np.asarray(f[1].data["PARALLAX"][rows], dtype="f8")
gr = match[take]
parallax = np.asarray(g["parallax"])[gr]
error = np.asarray(g["parallax_error"])[gr]
assert (
    not np.ma.getmaskarray(g["parallax"][gr]).any()
    and not np.ma.getmaskarray(g["parallax_error"][gr]).any()
)
assert np.isfinite(parallax).all() and np.isfinite(error).all() and (error > 0).all()
np.testing.assert_allclose(legacy, parallax, rtol=1e-6, atol=1e-6)
np.savez_compressed(
    out / "official-gaia-baseline.npz",
    source_id=refids,
    gaia_parallax_mas=parallax,
    gaia_formal_error_mas=error,
    feature_row=rows,
    legacy_parallax=legacy,
)
report = dict(
    all99525_identifiers_exact=True,
    common_reference=4166,
    units="mas",
    all_parallaxes_and_formal_errors_finite=True,
    legacy_baseline_matches_official_gaia=True,
    maximum_abs_parallax_difference_mas=float(np.max(np.abs(legacy - parallax))),
    retained_nonpositive_gaia_parallaxes=int((parallax <= 0).sum()),
    zero_point_correction_applied=False,
    formal_errors_not_total_calibrated_accuracy=True,
    parallax_snr_selection=False,
)
(out / "baseline-independent-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
