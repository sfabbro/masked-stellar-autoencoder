#!/usr/bin/env python3
"""Fair same-cohort comparison against the released Gaia DR3 GSP-Phot values.

Fetches `teff_gspphot`, `logg_gspphot`, `mh_gspphot` (and 16/84 bounds) for the
exact MSA test source IDs via Gaia TAP, then compares our point predictions and
GSP-Phot on the same stars/labels. This is the first leg of the released-catalogue
cross-check; Andrae-175M and Srivastava-2026 can be added later.

The query uses a VOTable upload of source IDs; chunks are resumable through a CSV
cache. No credentials are required for the public Gaia TAP service.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import astropy.table as at
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyvo.dal import TAPService

LABELS = ("T_eff", "log g", "[Fe/H]")
UNITS = ("K", "dex", "dex")
FEH_BINS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6])
GSP_COLUMNS = [
    "source_id",
    "teff_gspphot",
    "teff_gspphot_lower",
    "teff_gspphot_upper",
    "logg_gspphot",
    "logg_gspphot_lower",
    "logg_gspphot_upper",
    "mh_gspphot",
    "mh_gspphot_lower",
    "mh_gspphot_upper",
]


def metrics(residual: np.ndarray) -> dict:
    finite = residual[np.isfinite(residual)]
    if not len(finite):
        return {
            "n": 0,
            "mae": None,
            "medad": None,
            "bias": None,
            "nmad": None,
            "rmse": None,
        }
    med = float(np.median(finite))
    return {
        "n": int(len(finite)),
        "mae": float(np.mean(np.abs(finite))),
        "medad": float(np.median(np.abs(finite))),
        "bias": float(np.mean(finite)),
        "nmad": float(1.4826 * np.median(np.abs(finite - med))),
        "rmse": float(np.sqrt(np.mean(finite**2))),
    }


def fetch_gspphot(ids: np.ndarray, cache: Path, chunk: int = 10000) -> pd.DataFrame:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        cached = pd.read_csv(cache)
        done = set(cached["source_id"].astype("i8").tolist())
    else:
        cached = pd.DataFrame(columns=GSP_COLUMNS)
        done = set()
    pending = np.array([x for x in ids if int(x) not in done], dtype="i8")
    if not len(pending):
        return cached
    service = TAPService("https://gea.esac.esa.int/tap-server/tap")
    query = """SELECT a.source_id,
        a.teff_gspphot, a.teff_gspphot_lower, a.teff_gspphot_upper,
        a.logg_gspphot, a.logg_gspphot_lower, a.logg_gspphot_upper,
        a.mh_gspphot, a.mh_gspphot_lower, a.mh_gspphot_upper
        FROM gaiadr3.astrophysical_parameters AS a
        JOIN TAP_UPLOAD.ids AS i ON a.source_id = i.source_id"""
    for start in range(0, len(pending), chunk):
        block = pending[start : start + chunk]
        upload = at.Table({"source_id": block})
        for attempt in range(3):
            try:
                result = service.run_sync(query, uploads={"ids": upload})
                frame = result.to_table().to_pandas()
                break
            except Exception as error:  # noqa: BLE001 - retry TAP transient errors
                if attempt == 2:
                    raise
                print(f"retry {start} attempt {attempt + 1}: {error}", flush=True)
                time.sleep(2 + 2 * attempt)
        cached = pd.concat([cached, frame], ignore_index=True)
        cached = cached.drop_duplicates("source_id", keep="last")
        cached.to_csv(cache, index=False)
        print(f"fetched {len(block)} -> cached {len(cached)}/{len(ids)}", flush=True)
    return cached


def binned(
    frame: pd.DataFrame, column: str
) -> tuple[list[float], list[float], list[int]]:
    xs, ys, ns = [], [], []
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:]):
        part = frame[(frame["feh_true"] >= lo) & (frame["feh_true"] < hi)]
        xs.append(0.5 * (lo + hi))
        ns.append(int(len(part)))
        ys.append(float(np.median(part[column])) if len(part) >= 5 else np.nan)
    return xs, ys, ns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--title", default="MSA long vs released Gaia DR3 GSP-Phot, same test cohort"
    )
    args = parser.parse_args(argv)

    cohort = np.load(args.cohort, allow_pickle=False)
    rows = cohort["test_rows"]
    source_id = np.asarray(cohort["source_id"][rows], dtype="i8")
    truth = np.asarray(cohort["value"][rows], dtype="f8")
    center = np.asarray(cohort["label_center"], dtype="f8")
    scale = np.asarray(cohort["label_scale"], dtype="f8")
    prediction = np.load(args.predictions, allow_pickle=False)
    if "test_source_id" in prediction and not np.array_equal(
        np.asarray(prediction["test_source_id"], dtype="i8"), source_id
    ):
        raise ValueError("prediction/cohort source IDs differ")
    ours = np.asarray(prediction["test_mean"], dtype="f8") * scale + center

    gsp = fetch_gspphot(source_id, args.cache)
    gsp = gsp.drop_duplicates("source_id", keep="last")
    frame = pd.DataFrame(
        {
            "source_id": source_id,
            "teff_true": truth[:, 0],
            "logg_true": truth[:, 1],
            "feh_true": truth[:, 2],
            "teff_ours": ours[:, 0],
            "logg_ours": ours[:, 1],
            "feh_ours": ours[:, 2],
        }
    ).merge(gsp, on="source_id", how="left")
    for column in GSP_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = (
        np.isfinite(frame["teff_gspphot"])
        & np.isfinite(frame["logg_gspphot"])
        & np.isfinite(frame["mh_gspphot"])
    )
    frame = frame[finite].copy()
    frame["dteff_ours"] = frame["teff_ours"] - frame["teff_true"]
    frame["dlogg_ours"] = frame["logg_ours"] - frame["logg_true"]
    frame["dfeh_ours"] = frame["feh_ours"] - frame["feh_true"]
    frame["dteff_gsp"] = frame["teff_gspphot"] - frame["teff_true"]
    frame["dlogg_gsp"] = frame["logg_gspphot"] - frame["logg_true"]
    frame["dfeh_gsp"] = frame["mh_gspphot"] - frame["feh_true"]
    tail = frame["feh_true"] < -2.0
    summary = {
        "n_test": int(len(source_id)),
        "n_common_gspphot": int(len(frame)),
        "tail_n": int(tail.sum()),
        "ours": {
            "all": {
                k: metrics(frame[f"d{k}_ours"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
            "tail": {
                k: metrics(frame.loc[tail, f"d{k}_ours"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
        },
        "gspphot": {
            "all": {
                k: metrics(frame[f"d{k}_gsp"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
            "tail": {
                k: metrics(frame.loc[tail, f"d{k}_gsp"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2), constrained_layout=True)
    fig.suptitle(
        f"{args.title}\ncommon n={len(frame):,}; tail [Fe/H]<-2 n={int(tail.sum())}; local preliminary",
        fontsize=12,
    )
    for col, (label, unit) in enumerate(zip(LABELS, UNITS)):
        for row, prefix, name in ((0, "ours", "MSA long"), (1, "gsp", "GSP-Phot")):
            ax = axes[row, col]
            xs, med, ns = binned(
                frame,
                f"d{'teff' if col == 0 else 'logg' if col == 1 else 'feh'}_{prefix}",
            )
            ax.plot(xs, med, "o-", color="#c44e52" if row == 0 else "#4c72b0", lw=1.3)
            ax.axhline(0, color="0.3", lw=0.8)
            ax.axvline(-2.0, color="0.55", ls=":", lw=1.0)
            ax.set_title(f"{name}: median {label} residual", fontsize=10)
            ax.set_xlabel("true [Fe/H]")
            ax.set_ylabel(f"pred − true [{unit}]")
            stat_all = summary["gspphot" if prefix == "gsp" else prefix]["all"][
                "teff" if col == 0 else "logg" if col == 1 else "feh"
            ]
            ax.text(
                0.01,
                0.98,
                f"n={stat_all['n']}  MAE={stat_all['mae']:.3g}\nbias={stat_all['bias']:+.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0),
            )
            for x, y, nbin in zip(xs, med, ns):
                if np.isfinite(y):
                    ax.annotate(
                        str(nbin),
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=7,
                    )
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    print(
        json.dumps(
            {
                "status": "complete",
                "png": str(args.output),
                "n_common": len(frame),
                "tail_n": int(tail.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
