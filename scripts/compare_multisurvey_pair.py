"""Compare fine-tunes evaluated on the same table and test split.

usage: compare_multisurvey_pair.py DIR CENSUS_MANIFEST TAG=EVAL [TAG=EVAL ...]
       [--records NAME] [--gbs TAG=FILE ...]

DIR holds each ``<EVAL>_predictions.npz`` (eval_ensemble) and the table's
``<NAME>_records.npz``; an EVAL ending in ``.npz`` is a path to
predict_fits_rows.py output with ``--test-split`` instead. All metrics are
computed from the predictions, so both kinds compare alike. Reports held-out seismic log g (all seismic
anchors together, no giant/dwarf split), [Fe/H] against high-resolution
references by metallicity bin, metrics against the chosen label, each large
survey's median offset from the common-scale prediction, and, with ``--gbs``,
residuals on the Gaia benchmark stars (predict_fits_rows.py output).
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", "All-NaN slice")  # GSP-Spec has no [Fe/H]


def nmad(x):
    return 1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x)))


def stats(d):
    d = d[np.isfinite(d)]
    if not len(d):
        return "n=     0"
    return f"n={len(d):6d} bias={np.median(d):+.3f} nmad={nmad(d):.3f}"


def pairs(items):
    return [tuple(s.split("=", 1)) for s in items]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir")
    p.add_argument("census")
    p.add_argument("models", nargs="+", help="TAG=EVAL_NAME")
    p.add_argument("--records", default="multisurvey_table_20261113_v1")
    p.add_argument("--gbs", nargs="*", default=[], help="TAG=NPZ_PATH")
    a = p.parse_args()
    D = a.dir
    models = pairs(a.models)
    P = {
        t: np.load(e if e.endswith(".npz") else f"{D}/{e}_predictions.npz")
        for t, e in models
    }
    R = np.load(f"{D}/{a.records}_records.npz")
    rows = next(iter(P.values()))["table_row"]
    for t, x in P.items():
        assert (x["table_row"] == rows).all(), f"{t}: different test rows"
    w = max(len(t) for t in P) + 1

    names = R["survey_names"]
    census = json.load(open(a.census))["census"]["sources"]
    highres = {k for k, v in census.items() if v.get("tier") == "highres_reference"}
    v = R["value"]
    rec = pd.DataFrame(
        dict(row=R["row"], s=R["survey"], teff=v[:, 0], logg=v[:, 1], feh=v[:, 2])
    )
    rec["name"] = names[rec.s]
    rec["anchor"] = R["survey_is_anchor"][rec.s]
    test = rec[rec.row.isin(rows)]
    pos = pd.Series(np.arange(len(rows)), index=rows)

    seis = test[test.anchor].groupby("row").logg.median().dropna()
    i = pos[seis.index].to_numpy()
    print("Held-out seismic log g (test stars with a seismic record):")
    for t, x in P.items():
        print(f"  {t:{w}s}", stats(x["median"][i, 1] - seis.to_numpy()))

    hr = test[test.name.isin(highres)].groupby("row").feh.median().dropna()
    i, truth = pos[hr.index].to_numpy(), hr.to_numpy()
    print("[Fe/H] vs high-res references (test stars):")
    for lo, hi, lab in ((-9, -2, "< -2"), (-2, -1, "-2..-1"), (-1, 9, "> -1")):
        k = (truth >= lo) & (truth < hi)
        for t, x in P.items():
            print(f"  {lab:7s} {t:{w}s}", stats(x["median"][i[k], 2] - truth[k]))

    if all("teff_irfm" in x for x in P.values()):
        print("Teff vs photometric IRFM scale (low-reddening test stars):")
        for t, x in P.items():
            print(f"  {t:{w}s}", stats(x["median"][:, 0] - x["teff_irfm"]))

    names4 = ("teff", "logg", "fe_h", "alpha")
    print("NMAD vs the one chosen label per star (the stopgap's training target):")
    for t, x in P.items():
        tr, md, q = x["truth"][:, :4], x["median"][:, :4], x["quantiles"]
        nm = {n: round(float(nmad(md[:, j] - tr[:, j])), 3) for j, n in enumerate(names4)}  # fmt: skip
        mp = tr[:, 2] < -2
        cov = (
            np.isfinite(tr[:, 2]) & (q[:, 2, 0] <= tr[:, 2]) & (tr[:, 2] <= q[:, 2, 2])
        )
        cov = cov.sum() / np.isfinite(tr[:, 2]).sum()
        print(f"  {t:{w}s}", nm, "feh<-2:", round(float(nmad(md[mp, 2] - tr[mp, 2])), 3), "cov:", round(float(cov), 3))  # fmt: skip

    if all("G" in x for x in P.values()):
        for key, edges in (
            ("G", (0, 12, 14, 15.5, 17, 30)),
            ("EBV", (0, 0.05, 0.2, 0.5, 9)),
        ):
            print(f"NMAD vs chosen label by {key} bin: Teff | log g | [Fe/H]")
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                cells = []
                for t, x in P.items():
                    k = (x[key] >= lo) & (x[key] < hi)
                    d = x["median"][k, :3] - x["truth"][k, :3]
                    cells.append(
                        f"{t} {nmad(d[:, 0]):4.0f} {nmad(d[:, 1]):.3f} {nmad(d[:, 2]):.3f}"
                    )
                n = int(((x[key] >= lo) & (x[key] < hi)).sum())
                print(f"  {lo:>5}-{hi:<5} n={n:6d}  " + " | ".join(cells))

    print("Median (survey record - prediction), test stars: Teff | log g | [Fe/H]")
    for n in test[~test.anchor].name.value_counts().head(14).index:
        g = test[test.name == n]
        i = pos[g.row].to_numpy()
        cells = []
        for t, x in P.items():
            d = [
                np.nanmedian(g[c].to_numpy() - x["median"][i, j])
                for j, c in ((0, "teff"), (1, "logg"), (2, "feh"))
            ]
            cells.append(f"{t} {d[0]:+5.0f} {d[1]:+.3f} {d[2]:+.3f}")
        print(f"  {n:26s} n={len(g):6d}  " + " | ".join(cells))

    if a.gbs:
        print("Gaia benchmark stars (prediction - reference): Teff | log g | [Fe/H]")
        for t, f in pairs(a.gbs):
            b = np.load(f)
            d = b["median"][:, :3] - b["truth"]
            cells = [stats(d[:, j]) for j in range(3)]
            print(f"  {t:{w}s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
