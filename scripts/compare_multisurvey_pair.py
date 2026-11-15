"""Compare two fine-tunes evaluated on the same table and test split.

usage: compare_multisurvey_pair.py DIR CENSUS_MANIFEST

DIR holds <stopgap eval>_predictions.npz, <multi eval>_predictions.npz, their
metrics.json and the table's records.npz (names as below). Reports held-out
seismic log g (all seismic anchors together), [Fe/H] against high-resolution
references by metallicity bin, metrics against the chosen label, and each
large survey's median offset from the common-scale prediction.
"""

import json
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
S = np.load(f"{D}/stopgap_eval_20261113_v3_predictions.npz")
M = np.load(f"{D}/multisurvey_eval_20261113_v1_predictions.npz")
R = np.load(f"{D}/multisurvey_table_20261113_v1_records.npz")
assert (S["table_row"] == M["table_row"]).all()
rows = S["table_row"]
nm = lambda x: 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


def stats(d):
    d = d[np.isfinite(d)]
    return f"n={len(d):6d} bias={np.median(d):+.3f} nmad={nm(d):.3f}"


names = R["survey_names"]
anchor = R["survey_is_anchor"]
census = json.load(open(sys.argv[2]))["census"]["sources"]
highres = {k for k, v in census.items() if v.get("tier") == "highres_reference"}
rec = pd.DataFrame(
    dict(
        row=R["row"],
        s=R["survey"],
        teff=R["value"][:, 0],
        logg=R["value"][:, 1],
        feh=R["value"][:, 2],
    )
)
rec["name"] = names[rec.s]
rec["anchor"] = anchor[rec.s]
test = rec[rec.row.isin(rows)]
pos = pd.Series(np.arange(len(rows)), index=rows)
out = {}
# 1. seismic log g (all anchors together)
seis = test[test.anchor].groupby("row").logg.median()
i = pos[seis.index].to_numpy()
print("Held-out seismic log g (test stars with a seismic record):")
for tag, P in (("stopgap", S), ("multi", M)):
    print(f"  {tag:8s}", stats(P["median"][i, 1] - seis.to_numpy()))
# 2. [Fe/H] vs high-res references
hr = test[test.name.isin(highres)].groupby("row").feh.median().dropna()
i = pos[hr.index].to_numpy()
t = hr.to_numpy()
print("[Fe/H] vs high-res references (test stars):")
for lo, hi, lab in ((-9, -2, "< -2"), (-2, -1, "-2..-1"), (-1, 9, "> -1")):
    k = (t >= lo) & (t < hi)
    for tag, P in (("stopgap", S), ("multi", M)):
        print(f"  {lab:7s} {tag:8s}", stats(P["median"][i[k], 2] - t[k]))
# 3. chosen-label metrics
print("Global NMAD vs the one chosen label per star (stopgap's own training target):")
for tag, f in (
    ("stopgap", "stopgap_eval_20261113_v3"),
    ("multi", "multisurvey_eval_20261113_v1"),
):
    g = json.load(open(f"{D}/{f}_metrics.json"))
    print(
        f"  {tag:8s}",
        {
            k: round(g["global_xp_on"][k]["NMAD"], 3)
            for k in ("teff", "logg", "fe_h", "alpha")
        },
        "feh<-2:",
        round(g["bins_feh_xp_on"]["feh_lt_m2"]["fe_h"]["NMAD"], 3),
        "cov:",
        round(g["calibration_xp_on"]["fe_h"]["coverage_1sigma"], 3),
    )
# 4. per-survey offsets of records from the common-scale prediction
print("Median (survey record - prediction), test stars: teff | logg | [Fe/H]")
top = test[~test.anchor].name.value_counts().head(14).index
for n in top:
    g = test[test.name == n]
    i = pos[g.row].to_numpy()
    cells = []
    for tag, P in (("S", S), ("M", M)):
        d = [
            np.nanmedian(g[c].to_numpy() - P["median"][i, j])
            for j, c in ((0, "teff"), (1, "logg"), (2, "feh"))
        ]
        cells.append(f"{tag} {d[0]:+6.0f} {d[1]:+.3f} {d[2]:+.3f}")
    print(f"  {n:26s} n={len(g):6d}  " + "  ".join(cells))
