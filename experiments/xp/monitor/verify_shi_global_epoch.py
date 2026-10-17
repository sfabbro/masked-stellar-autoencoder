import hashlib
import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time

root = Path(
    "experiments/xp/results-stellar-improvement/shi_collection_v1/feature-coordinate-candidates-v1"
)
a = np.load(root / "official-subset-associations.npz")
g = Table.read(root / "official-gaia-identities.xml", format="votable")
lookup = {int(s): i for i, s in enumerate(g["SOURCE_ID"])}
g = g[[lookup[int(s)] for s in a["source_id"]]]
c = SkyCoord(
    ra=g["ra"].quantity,
    dec=g["dec"].quantity,
    pm_ra_cosdec=g["pmra"].quantity,
    pm_dec=g["pmdec"].quantity,
    obstime=Time(2016.0, format="jyear"),
)
shi = SkyCoord(a["shi_ra"] * u.deg, a["shi_dec"] * u.deg)
old = c.apply_space_motion(Time(2000.0, format="jyear"))
oldsep = old.separation(shi).arcsec
sep = c.separation(shi).arcsec
np.testing.assert_allclose(sep, a["shi_gaia_separation_arcsec"], atol=1e-10)
n = Table.read(root / "official-gaia-global-neighbors.xml", format="votable")
ni = SkyCoord(ra=n["ra"].quantity, dec=n["dec"].quantity)
nearest_ids = []
nearest_sep = []
second = []
for coord in shi:
    ds = coord.separation(ni).arcsec
    order = np.argsort(ds)
    nearest_ids.append(int(n["SOURCE_ID"][order[0]]))
    nearest_sep.append(float(ds[order[0]]))
    # Query is the UNION of cones. Distances>2 are not exhaustive global neighbors.
    second.append(float(ds[order[1]]) if ds[order[1]] <= 2 else np.nan)
nearest_ids = np.array(nearest_ids, dtype="i8")
accepted = a["subset_candidate_pass"]
assert (nearest_ids[accepted] == a["source_id"][accepted]).all()
assert (sep[accepted] < oldsep[accepted]).all()
record = dict(
    subset_n=128,
    subset_pass=int(accepted.sum()),
    provided_feature_id_is_not_nearest_global_gaia=int(
        (nearest_ids != a["source_id"]).sum()
    ),
    all_passed_are_nearest_global_gaia_with_no_other_within2arcsec=True,
    passed_max_2016_separation_arcsec=float(sep[accepted].max()),
    passed_min_2000_separation_arcsec=float(oldsep[accepted].min()),
    all_passed_closer_to_2016_than_2000=True,
    scope="128 highest-proper-motion LOCAL-candidate rows, not random or full-catalogue certification.2Dproper motion/no radial perspective; interpret as DR3 coordinate consistency, not invented Shi epoch metadata. NaNsecond means no second Gaia source within queried2arcsec, not infinite isolation.",
    fit_eligible=False,
)
np.savez_compressed(
    root / "independent-global-neighbor-epoch-audit.npz",
    source_id=a["source_id"],
    nearest_global_gaia_id=nearest_ids,
    nearest_global_separation_arcsec=nearest_sep,
    second_global_within2arcsec=second,
    shi_gaia2016_separation_arcsec=sep,
    shi_gaia2000_separation_arcsec=oldsep,
    subset_pass=accepted,
)
feature = Path("/tmp/msa-labels-complete-20261015.fits")
h = hashlib.sha256()
with feature.open("rb") as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
        h.update(block)
record["feature_file_sha256"] = h.hexdigest()
record["shi_file_sha256"] = (
    "665e2952a88db3f1bf5b4646cffca858fbebce7681549d27dbd257dcd96b3183"
)
(root / "independent-global-neighbor-epoch-audit.json").write_text(
    json.dumps(record, indent=2) + "\n"
)
print(json.dumps(record, indent=2))
