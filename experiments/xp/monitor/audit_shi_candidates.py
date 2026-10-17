"""Full Shi-to-feature coordinate candidates, with no unverified Gaia ID promotion."""

import hashlib
import json
import sys
import time
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from stellar_sky_candidates import nearest_two, unambiguous_candidates

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
root = ROOT / "shi_collection_v1"
out = root / "feature-coordinate-candidates-v1"
out.mkdir(exist_ok=False)
start = time.monotonic()
rawpath = root / "shi2026-xp-metals.fits"
assert (
    hashlib.sha256(rawpath.read_bytes()).hexdigest()
    == "665e2952a88db3f1bf5b4646cffca858fbebce7681549d27dbd257dcd96b3183"
)
with (
    fits.open(rawpath, memmap=True) as f,
    fits.open("/tmp/msa-labels-complete-20261015.fits", memmap=True) as feature,
):
    shi = f[1].data
    assert len(shi) == 14536547
    ra, dec = np.asarray(shi["ra"], dtype="f8"), np.asarray(shi["dec"], dtype="f8")
    valid = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & (ra >= 0)
        & (ra < 360)
        & (np.abs(dec) <= 90)
    )
    rawrows = np.flatnonzero(valid)
    table = feature[1].data
    ids = np.asarray(table["source_id"], dtype="i8")
    _, first = np.unique(ids, return_index=True)
    first = np.sort(first[ids[first] > 0])
    qra = np.asarray(table["RA"], dtype="f8")[first]
    qdec = np.asarray(table["DEC"], dtype="f8")[first]
    nearest, separation = nearest_two(ra[valid], dec[valid], qra, qdec)
    nearest = rawrows[nearest]
    candidates = unambiguous_candidates(nearest, separation)
    # Independent Astropy great-circle audit for both neighbors over1024 uniform rows.
    fixture = np.random.default_rng(42).choice(len(first), 1024, replace=False)
    c = SkyCoord(qra[fixture] * u.deg, qdec[fixture] * u.deg)
    for k in range(2):
        other = SkyCoord(
            ra[nearest[fixture, k]] * u.deg, dec[nearest[fixture, k]] * u.deg
        )
        np.testing.assert_allclose(
            c.separation(other).arcsec, separation[fixture, k], atol=1e-7, rtol=1e-7
        )
    pmra = np.asarray(table["pmra"], dtype="f8")[first]
    pmdec = np.asarray(table["pmdec"], dtype="f8")[first]
    proper = np.hypot(pmra, pmdec)
    eligible = np.flatnonzero(
        candidates
        & np.isfinite(proper)
        & (np.abs(pmra) < 9999)
        & (np.abs(pmdec) < 9999)
    )
    auditrows = eligible[np.argsort(proper[eligible])[-128:]]
    np.savez_compressed(
        out / "nearest-two.npz",
        feature_source_id=ids[first],
        feature_original_row=first,
        shi_original_row=nearest,
        separation_arcsec=separation,
        local_candidate=candidates,
    )
    np.savez_compressed(
        out / "official-gaia-audit-input.npz",
        source_id=ids[first[auditrows]],
        feature_original_row=first[auditrows],
        shi_original_row=nearest[auditrows, 0],
        feature_ra=qra[auditrows],
        feature_dec=qdec[auditrows],
        shi_ra=ra[nearest[auditrows, 0]],
        shi_dec=dec[nearest[auditrows, 0]],
        feature_pmra=pmra[auditrows],
        feature_pmdec=pmdec[auditrows],
    )
    selected = nearest[candidates, 0]
    np.savez_compressed(
        out / "candidate-raw-values.npz",
        source_id=ids[first[candidates]],
        shi_original_row=selected,
        **{name: np.asarray(shi[name])[selected] for name in shi.names},
    )
    report = dict(
        shi_raw_rows=len(shi),
        valid_coordinate_rows=int(valid.sum()),
        unique_feature_ids=len(first),
        nearest_within1arcsec=int((separation[:, 0] <= 1).sum()),
        local_isolated_one_to_one_candidates=int(candidates.sum()),
        astropy_independent1024x2_separations_verified=True,
        median_candidate_separation_arcsec=float(np.median(separation[candidates, 0])),
        max_candidate_separation_arcsec=float(separation[candidates, 0].max()),
        runtime_seconds=time.monotonic() - start,
        coordinate_interpretation="Degree hypothesis only until official Gaia audit. Publisher FITS units/epoch absent. Match evidence must establish DR3 coordinate consistency, not invent epoch metadata.",
        scope="Nearest/second-neighbor search in FULL Shi catalogue against1.3M unique existing labelled feature IDs; NOT full Gaia sky or10M/20M new stars. Global Gaia neighborhood ambiguity remains unaudited.",
        label_role="XP-derived weak candidate metadata only, not independent XP truth; no training labels merged or changed.",
        official_gaia_audit_pending=True,
        fit_eligible=False,
    )
(out / "candidate-audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
