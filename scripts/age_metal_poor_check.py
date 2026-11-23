"""Metal-poor consistency check: giants (seismic) vs subgiants/MSTO (isochrone) at
fixed chemistry, in both gauges. Data-driven: at fixed [Fe/H] (and [alpha/M]) the
age distribution of a population should not depend on evolutionary stage."""

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cons, apo, out = sys.argv[1], sys.argv[2], sys.argv[3]
s = pd.read_parquet(
    cons, columns=["source_id", "t_A", "sd_A", "t_C", "sd_C", "stage", "feh", "limit_A"]
)
s = s[np.isfinite(s.t_A) & np.isfinite(s.t_C) & np.isfinite(s.feh) & (s.limit_A == 0)]
s["grp"] = np.where(
    s.stage.isin(["RGB", "RC", "giant"]),
    "giant",
    np.where(s.stage.isin(["SGB", "MSTO"]), "subgiant/MSTO", "other"),
)
s = s[s.grp != "other"]
a = pd.read_parquet(apo, columns=["source_id", "alpha", "fe_h", "snr"])
a = (
    a[(a.source_id > 0) & np.isfinite(a.alpha)]
    .sort_values("snr", ascending=False)
    .drop_duplicates("source_id")
)
s = s.merge(a[["source_id", "alpha"]], on="source_id", how="left")
rng = np.random.default_rng(1)


def med_se(x):
    x = np.asarray(x)
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    bs = [np.median(rng.choice(x, n)) for _ in range(200)]
    return np.median(x), np.std(bs), n


edges = [(-9, -1.0), (-1.0, -0.5), (-0.5, -0.2), (-0.2, 0.1), (0.1, 9)]
rows = []
for lo, hi in edges:
    m = (s.feh >= lo) & (s.feh < hi)
    for sub, mm in (
        ("all", m),
        ("high-alpha", m & (s.alpha > 0.15)),
        ("low-alpha", m & (s.alpha < 0.08)),
    ):
        r = dict(feh=f"[{lo},{hi})", subset=sub)
        for g in ("giant", "subgiant/MSTO"):
            x = s[mm & (s.grp == g)]
            for gauge in "AC":
                v, e, n = med_se(x[f"t_{gauge}"])
                r[f"{g}_{gauge}"], r[f"{g}_{gauge}_se"], r[f"n_{g}"] = v, e, n
        for gauge in "AC":
            r[f"diff_{gauge}"] = r[f"subgiant/MSTO_{gauge}"] - r[f"giant_{gauge}"]
            r[f"diff_{gauge}_se"] = np.hypot(
                r[f"subgiant/MSTO_{gauge}_se"], r[f"giant_{gauge}_se"]
            )
        rows.append(r)
t = pd.DataFrame(rows)
t.to_csv(f"{out}/metal_poor_check.csv", index=False)
print(
    t[
        [
            "feh",
            "subset",
            "n_giant",
            "n_subgiant/MSTO",
            "giant_A",
            "subgiant/MSTO_A",
            "diff_A",
            "diff_A_se",
            "giant_C",
            "subgiant/MSTO_C",
            "diff_C",
            "diff_C_se",
        ]
    ]
    .round(3)
    .to_string(index=False)
)
# plot: median age vs [Fe/H] for giants and subgiants, both gauges; high/low alpha
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
col = {"giant": "#2a6fdb", "subgiant/MSTO": "#e8673c"}
for ax, sub in zip(axs, ("all", "high-alpha", "low-alpha")):
    tt = t[t.subset == sub]
    x = np.arange(len(tt))
    for g in ("giant", "subgiant/MSTO"):
        for gauge, ls, mk in (("A", "-", "o"), ("C", "--", "s")):
            ax.errorbar(
                x + (0.06 if gauge == "C" else -0.06),
                tt[f"{g}_{gauge}"],
                yerr=tt[f"{g}_{gauge}_se"],
                color=col[g],
                ls=ls,
                marker=mk,
                ms=4,
                lw=1.2,
                capsize=2,
                label=f"{g}, gauge {gauge}",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tt.feh, fontsize=7)
    ax.set_xlabel("[Fe/H] bin (dex)")
    ax.set_title(
        f"{sub} ([alpha/M] from APOGEE)" if sub != "all" else "all stars",
        loc="left",
        fontsize=9,
    )
    ax.grid(alpha=0.3)
axs[0].set_ylabel("median consensus log10(age/Gyr)")
axs[0].legend(fontsize=6.5, loc="lower left")
fig.suptitle(
    "Same chemistry, different evolutionary stage: giants (seismic) vs subgiants/MSTO (isochrone). "
    "Equal medians expected if the cross-regime bridge is right",
    x=0.01,
    ha="left",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(f"{out}/c0c_metal_poor_check.png", dpi=100)
