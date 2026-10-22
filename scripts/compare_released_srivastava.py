#!/usr/bin/env python3
"""Same-cohort Fe/H comparison: MSA vs GSP-Phot vs Srivastava+2026 XP catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEH_BINS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6])


def metrics(residual: np.ndarray) -> dict:
    finite = residual[np.isfinite(residual)]
    if not len(finite):
        return {"n": 0, "mae": None, "medad": None, "bias": None, "nmad": None, "rmse": None}
    med = float(np.median(finite))
    return {
        "n": int(len(finite)),
        "mae": float(np.mean(np.abs(finite))),
        "medad": float(np.median(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "nmad": float(1.4826 * np.median(np.abs(finite - med))),
        "rmse": float(np.sqrt(np.mean(finite**2))),
    }


def binned_median(frame: pd.DataFrame, residual: str) -> tuple[list[float], list[float], list[int]]:
    xs, ys, ns = [], [], []
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:]):
        part = frame[(frame["feh_true"] >= lo) & (frame["feh_true"] < hi)]
        xs.append(0.5 * (lo + hi))
        ns.append(int(len(part)))
        ys.append(float(np.median(part[residual])) if len(part) >= 5 else np.nan)
    return xs, ys, ns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--srivastava", type=Path, required=True)
    parser.add_argument("--gspphot-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cohort = np.load(args.cohort, allow_pickle=False)
    rows = cohort["test_rows"]
    source_id = np.asarray(cohort["source_id"][rows], dtype="i8")
    truth = np.asarray(cohort["value"][rows], dtype="f8")
    center = np.asarray(cohort["label_center"], dtype="f8")
    scale = np.asarray(cohort["label_scale"], dtype="f8")
    pred = np.load(args.predictions, allow_pickle=False)
    ours = np.asarray(pred["test_mean"], dtype="f8") * scale + center

    sri = pd.read_parquet(
        args.srivastava, columns=["source_id", "feh_pred", "feh_pred_std", "phot_g_mean_mag", "bp_rp", "ruwe"]
    )
    gsp = pd.read_csv(args.gspphot_cache, usecols=["source_id", "teff_gspphot", "logg_gspphot", "mh_gspphot"])
    gsp["source_id"] = pd.to_numeric(gsp["source_id"], errors="coerce").astype("Int64")
    gsp["mh_gspphot"] = pd.to_numeric(gsp["mh_gspphot"], errors="coerce")
    frame = pd.DataFrame({"source_id": source_id, "feh_true": truth[:, 2], "feh_ours": ours[:, 2]})
    frame = frame.merge(sri, on="source_id", how="left").merge(gsp[["source_id", "mh_gspphot"]], on="source_id", how="left")
    frame["dfeh_ours"] = frame["feh_ours"] - frame["feh_true"]
    frame["dfeh_gsp"] = frame["mh_gspphot"] - frame["feh_true"]
    frame["dfeh_sri"] = frame["feh_pred"] - frame["feh_true"]
    common = frame[np.isfinite(frame["feh_ours"]) & np.isfinite(frame["feh_pred"])].copy()
    tail = common["feh_true"] < -2.0
    summary = {
        "n_test": int(len(source_id)),
        "n_common_srivastava": int(len(common)),
        "tail_n": int(tail.sum()),
        "ours": {"all": metrics(common["dfeh_ours"].to_numpy()), "tail": metrics(common.loc[tail, "dfeh_ours"].to_numpy())},
        "srivastava": {"all": metrics(common["dfeh_sri"].to_numpy()), "tail": metrics(common.loc[tail, "dfeh_sri"].to_numpy())},
    }
    if common["dfeh_gsp"].notna().any():
        summary["gspphot"] = {
            "all": metrics(common["dfeh_gsp"].to_numpy()),
            "tail": metrics(common.loc[tail, "dfeh_gsp"].to_numpy()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    common.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    fig.suptitle(
        f"Same-cohort [Fe/H] comparison: MSA long vs GSP-Phot vs Srivastava+2026\n"
        f"common n={len(common):,}; tail [Fe/H]<-2 n={int(tail.sum())}; local preliminary",
        fontsize=11.5,
    )
    ax = axes[0, 0]
    for column, label, color in (
        ("dfeh_ours", "MSA long", "#c44e52"),
        ("dfeh_gsp", "GSP-Phot", "#4c72b0"),
        ("dfeh_sri", "Srivastava XGBoost", "#55a868"),
    ):
        xs, ys, ns = binned_median(common, column)
        ax.plot(xs, ys, "o-", label=label, color=color, lw=1.3)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.axvline(-2, color="0.55", ls=":", lw=1.0)
    ax.set_xlabel("true [Fe/H]")
    ax.set_ylabel("median residual [dex]")
    ax.set_title("binned median residual", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    bins = np.linspace(-1.5, 1.5, 80)
    for column, label, color in (
        ("dfeh_ours", "MSA long", "#c44e52"),
        ("dfeh_gsp", "GSP-Phot", "#4c72b0"),
        ("dfeh_sri", "Srivastava XGBoost", "#55a868"),
    ):
        values = common[column].dropna().to_numpy()
        ax.hist(values, bins=bins, histtype="step", density=True, label=f"{label} (n={len(values)})", color=color, lw=1.3)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.set_xlabel("pred − true [Fe/H]")
    ax.set_ylabel("density")
    ax.set_title("residual distribution", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.hexbin(common["feh_true"], common["feh_pred"], gridsize=50, bins="log", cmap="viridis", mincnt=1)
    lims = [-2.8, 0.8]
    ax.plot(lims, lims, color="0.4", lw=0.8)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("true [Fe/H]")
    ax.set_ylabel("Srivastava feh_pred")
    ax.set_title("Srivastava vs native labels", fontsize=10)

    ax = axes[1, 1]
    ax.axis("off")
    rows_text = []
    for who, key in (("MSA long", "ours"), ("Srivastava", "srivastava"), ("GSP-Phot", "gspphot")):
        if key not in summary:
            continue
        a = summary[key]["all"]
        t = summary[key]["tail"]
        def fmt(value, signed=False):
            if value is None:
                return "—"
            return f"{value:+.3f}" if signed else f"{value:.3f}"

        rows_text.append(
            f"{who:12s} all n={a['n']:5d} MedAD={fmt(a['medad'])} MAE={fmt(a['mae'])} bias={fmt(a['bias'], True)}\n"
            f"{'':12s} tail n={t['n']:3d} MedAD={fmt(t['medad'])} MAE={fmt(t['mae'])} bias={fmt(t['bias'], True)}"
        )
    ax.text(0.0, 1.0, "\n\n".join(rows_text), va="top", family="monospace", fontsize=9)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    print(json.dumps({"status": "complete", "png": str(args.output), "n_common": len(common), "tail_n": int(tail.sum())}))


if __name__ == "__main__":
    main()
