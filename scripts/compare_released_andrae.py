#!/usr/bin/env python3
"""Same-cohort comparison against the Andrae+2023 175M XGBoost catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from comparison_pairing import bound_predictions

LABELS = ("T_eff", "log g", "[M/H]")
UNITS = ("K", "dex", "dex")
FEH_BINS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6])


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


def binned(
    frame: pd.DataFrame, residual: str
) -> tuple[list[float], list[float], list[int]]:
    xs, ys, ns = [], [], []
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:]):
        part = frame[(frame["feh_true"] >= lo) & (frame["feh_true"] < hi)]
        xs.append(0.5 * (lo + hi))
        ns.append(int(len(part)))
        ys.append(float(np.median(part[residual])) if len(part) >= 5 else np.nan)
    return xs, ys, ns


def scan_fits(path: Path, ids: np.ndarray, chunk: int = 5_000_000) -> pd.DataFrame:
    ids = np.asarray(ids, dtype="i8")
    order = np.argsort(ids)
    sorted_ids = ids[order]
    parts = []
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        hdu = hdus[1]
        nrows = int(hdu.header["NAXIS2"])
        print(f"scanning {nrows:,} rows in chunks of {chunk:,}", flush=True)
        for start in range(0, nrows, chunk):
            stop = min(start + chunk, nrows)
            data = hdu.data[start:stop]
            sid = np.asarray(data["source_id"], dtype="i8")
            pos = np.searchsorted(sorted_ids, sid)
            safe = pos < len(sorted_ids)
            match = np.zeros(len(sid), dtype=bool)
            match[safe] = sorted_ids[pos[safe]] == sid[safe]
            if match.any():
                part = pd.DataFrame(
                    {
                        "source_id": sid[match],
                        "teff_andrae": np.asarray(data["teff_xgboost"], dtype="f8")[
                            match
                        ],
                        "logg_andrae": np.asarray(data["logg_xgboost"], dtype="f8")[
                            match
                        ],
                        "mh_andrae": np.asarray(data["mh_xgboost"], dtype="f8")[match],
                        "andrae_training_sample": np.asarray(
                            data["in_training_sample"], dtype=bool
                        )[match],
                    }
                )
                parts.append(part)
            if start % (chunk * 5) == 0:
                print(
                    f"  scanned {stop:,}/{nrows:,}; matches {sum(len(p) for p in parts):,}",
                    flush=True,
                )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    cohort = np.load(args.cohort, allow_pickle=False)
    rows = np.asarray(cohort["test_rows"])
    ids = np.asarray(cohort["source_id"][rows], dtype="i8")
    truth = np.asarray(cohort["value"][rows], dtype="f8")
    pred = np.load(args.predictions, allow_pickle=False)
    ours = bound_predictions(cohort, pred, "test")

    andrae = scan_fits(args.fits, ids)
    if andrae["source_id"].duplicated().any():
        raise ValueError("duplicate source IDs in Andrae catalogue")
    frame = pd.DataFrame(
        {
            "source_id": ids,
            "teff_true": truth[:, 0],
            "logg_true": truth[:, 1],
            "feh_true": truth[:, 2],
            "teff_ours": ours[:, 0],
            "logg_ours": ours[:, 1],
            "feh_ours": ours[:, 2],
        }
    ).merge(andrae, on="source_id", how="inner", validate="one_to_one")
    for key in ("teff", "logg", "feh"):
        frame[f"d{key}_ours"] = frame[f"{key}_ours"] - frame[f"{key}_true"]
    for key, col in (
        ("teff", "teff_andrae"),
        ("logg", "logg_andrae"),
        ("feh", "mh_andrae"),
    ):
        frame[f"d{key}_andrae"] = frame[col] - frame[f"{key}_true"]
    # A single finite subset for both systems and all three targets.
    paired_columns = [
        f"d{k}_{model}" for k in ("teff", "logg", "feh") for model in ("ours", "andrae")
    ]
    frame = frame[np.isfinite(frame[paired_columns]).all(axis=1)].copy()
    tail = frame["feh_true"] < -2.0
    held = ~frame["andrae_training_sample"].astype(bool)
    summary = {
        "target_caveat": "Andrae [M/H] is a proxy for native [Fe/H], not an identical target",
        "n_test": int(len(ids)),
        "n_common_andrae": int(len(frame)),
        "n_not_training_sample": int(held.sum()),
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
        "andrae": {
            "all": {
                k: metrics(frame[f"d{k}_andrae"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
            "tail": {
                k: metrics(frame.loc[tail, f"d{k}_andrae"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
        },
        "ours_heldout": {
            "all": {
                k: metrics(frame.loc[held, f"d{k}_ours"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
            "tail": {
                k: metrics(frame.loc[tail & held, f"d{k}_ours"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
        },
        "andrae_heldout": {
            "all": {
                k: metrics(frame.loc[held, f"d{k}_andrae"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
            "tail": {
                k: metrics(frame.loc[tail & held, f"d{k}_andrae"].to_numpy())
                for k in ("teff", "logg", "feh")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2), constrained_layout=True)
    fig.suptitle(
        f"Same-cohort Andrae+2023 175M vs MSA: common n={len(frame):,}; "
        f"tail [Fe/H]<-2 n={int(tail.sum())}; local preliminary",
        fontsize=11.5,
    )
    for col, (key, label, unit) in enumerate(
        zip(("teff", "logg", "feh"), LABELS, UNITS)
    ):
        for row, prefix, name, color in (
            (0, "ours", "MSA long", "#c44e52"),
            (1, "andrae", "Andrae XGBoost", "#55a868"),
        ):
            ax = axes[row, col]
            xs, ys, ns = binned(frame, f"d{key}_{prefix}")
            ax.plot(xs, ys, "o-", color=color, lw=1.3)
            ax.axhline(0, color="0.3", lw=0.8)
            ax.axvline(-2, color="0.55", ls=":", lw=1.0)
            ax.set_xlabel("true [Fe/H]")
            ax.set_ylabel(f"pred − true [{unit}]")
            ax.set_title(f"{name}: {label} residual", fontsize=10)
            stats = summary[prefix]["all"][key]
            ax.text(
                0.01,
                0.98,
                f"n={stats['n']} MAE={stats['mae']:.3g}\nbias={stats['bias']:+.3g}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0),
            )
            for x, y, nbin in zip(xs, ys, ns):
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
