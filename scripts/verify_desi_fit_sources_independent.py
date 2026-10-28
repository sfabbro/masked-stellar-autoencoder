import json
from pathlib import Path

import h5py
import numpy as np

root = (
    Path(__file__).resolve().parents[1] / "experiments/xp/results-stellar-improvement"
)
with np.load(
    root
    / "remediation-wave-20261026/desi-cache-census-preservation-20261027/census/identities.npz",
    allow_pickle=False,
) as z:
    keep = (
        (z["quality_rows"] == 1)
        & (z["base_role"] == "fit")
        & z["base_primary_input"]
        & z["observed"][:, 1]
    )
    census_ids = z["source_id"][keep].copy()
with h5py.File(root / "desi_labels_v1/desi-labels.h5", "r") as h:
    ds = h["measurements"]
    source = ds.fields("source_id")[:]
    raw = ds.fields(["logg", "logg_err_native"])[:]
raw_ids, raw_first, raw_counts = np.unique(
    source, return_index=True, return_counts=True
)
census_unique, census_i, raw_i = np.intersect1d(
    census_ids, raw_ids, return_indices=True
)
assert len(census_unique) == len(census_ids)
chosen_raw = raw_first[raw_i[raw_counts[raw_i] == 1]]
chosen_ids = source[chosen_raw]
g = raw["logg"][chosen_raw]
assert np.isfinite(g).all()
counts = np.histogram(g, bins=[-np.inf, 0, 2, 3, 4, 5, np.inf])[0].tolist()
with h5py.File(root / "joint-native-preparation-v2/joint-native-labels.h5", "r") as h:
    ids = h["source_id"][:]
    mask = h["targets/observed"][:, 1] & h["targets/quality_pass"][:, 1]
    values = h["targets/value"][:, 1]
    surveys = h["targets/survey"][:].astype("U")
    id_match, native_index, desi_index = np.intersect1d(
        ids, chosen_ids, return_indices=True
    )
    native_index, desi_index = (
        native_index[mask[native_index]],
        desi_index[mask[native_index]],
    )
    result = {}
    for survey in ["galah", "apogee"]:
        pick = surveys[native_index] == survey
        ng = values[native_index][pick]
        dg = g[desi_index][pick]
        bins = {}
        for name, lo, hi in [
            ("0_to_2", 0, 2),
            ("2_to_3", 2, 3),
            ("3_to_4", 3, 4),
            ("4_to_5", 4, 5),
        ]:
            inside = (ng >= lo) & (ng < hi)
            bins[name] = {
                "n": int(inside.sum()),
                "median_native_minus_desi": float(np.median((ng - dg)[inside]))
                if inside.any()
                else None,
            }
        result[survey] = {
            "n": int(pick.sum()),
            "median_native_minus_desi": float(np.median(ng - dg)),
            "native_bins": bins,
        }
print(
    json.dumps(
        {
            "unique_full_table_subset": len(chosen_ids),
            "counts": counts,
            "surveys": result,
        },
        sort_keys=True,
        indent=2,
    )
)
