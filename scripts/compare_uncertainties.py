#!/usr/bin/env python3
"""Compare release-quoted 16/84 intervals with MSA split-conformal intervals.

MSA intervals: calibration rows (2,046; separate role) produce normalized
conformal scores |y_cal - pred_cal| / sigma_cal; the exact finite-sample
ceil((n+1)(1-alpha)) rank defines a per-star interval
pred_test +/- q_alpha * sigma_test. GSP-Phot uses its released 16/84 bounds.
Coverage and width are evaluated on the same test stars with finite GSP-Phot values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ("T_eff", "log g", "[Fe/H]")
UNITS = ("K", "dex", "dex")
FEH_BINS = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.6])


def exact_rank(n: int, alpha: float) -> int:
    return int(np.ceil((n + 1) * (1.0 - alpha))) - 1


def split_conformal(
    pred_cal: np.ndarray, sigma_cal: np.ndarray, y_cal: np.ndarray, pred_test: np.ndarray, sigma_test: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float]:
    score = np.abs(y_cal - pred_cal) / np.maximum(sigma_cal, 1e-12)
    score = score[np.isfinite(score)]
    rank = min(max(exact_rank(len(score), alpha), 0), len(score) - 1)
    q = float(np.sort(score)[rank])
    return pred_test - q * sigma_test, pred_test + q * sigma_test, q


def coverage(frame: pd.DataFrame, lo: str, hi: str, truth: str) -> dict:
    finite = np.isfinite(frame[lo]) & np.isfinite(frame[hi]) & np.isfinite(frame[truth])
    part = frame[finite]
    if not len(part):
        return {"n": 0, "coverage": None, "median_width": None, "mean_width": None, "coverage_poisson_se": None}
    covered = (part[truth] >= part[lo]) & (part[truth] <= part[hi])
    width = part[hi] - part[lo]
    return {
        "n": int(len(part)),
        "coverage": float(covered.mean()),
        "median_width": float(np.median(width)),
        "mean_width": float(np.mean(width)),
        "coverage_poisson_se": float(np.sqrt((covered.mean() * (1 - covered.mean())) / len(part))),
    }


def binned_coverage(frame: pd.DataFrame, lo: str, hi: str, truth: str) -> tuple[list[float], list[float], list[int]]:
    xs, ys, ns = [], [], []
    for a, b in zip(FEH_BINS[:-1], FEH_BINS[1:]):
        part = frame[(frame["feh_true"] >= a) & (frame["feh_true"] < b)]
        xs.append(0.5 * (a + b))
        ns.append(int(len(part)))
        if len(part) >= 5:
            covered = (part[truth] >= part[lo]) & (part[truth] <= part[hi])
            ys.append(float(covered.mean()))
        else:
            ys.append(np.nan)
    return xs, ys, ns


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--gspphot-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.3173, help="nominal two-sided tail (16/84 ~= 0.3173)")
    args = parser.parse_args(argv)

    cohort = np.load(args.cohort, allow_pickle=False)
    center = np.asarray(cohort["label_center"], dtype="f8")
    scale = np.asarray(cohort["label_scale"], dtype="f8")
    test_rows = np.asarray(cohort["test_rows"])
    cal_rows = np.asarray(cohort["calibration_rows"])
    test_truth = np.asarray(cohort["value"][test_rows], dtype="f8")
    cal_truth = np.asarray(cohort["value"][cal_rows], dtype="f8")
    pred = np.load(args.predictions, allow_pickle=False)
    test_source_id = np.asarray(pred["test_source_id"], dtype="i8")
    pred_test = np.asarray(pred["test_mean"], dtype="f8") * scale + center
    sigma_test = np.sqrt(np.maximum(np.asarray(pred["test_variance"], dtype="f8"), 0.0)) * scale
    pred_cal = np.asarray(pred["calibration_mean"], dtype="f8") * scale + center
    sigma_cal = np.sqrt(np.maximum(np.asarray(pred["calibration_variance"], dtype="f8"), 0.0)) * scale

    gsp = pd.read_csv(args.gspphot_cache)
    for column in gsp.columns:
        if column != "source_id":
            gsp[column] = pd.to_numeric(gsp[column], errors="coerce")
    frame = pd.DataFrame(
        {
            "source_id": test_source_id,
            "teff_true": test_truth[:, 0],
            "logg_true": test_truth[:, 1],
            "feh_true": test_truth[:, 2],
            "teff_ours": pred_test[:, 0],
            "logg_ours": pred_test[:, 1],
            "feh_ours": pred_test[:, 2],
        }
    ).merge(gsp, on="source_id", how="left")
    for j, key in enumerate(("teff", "logg", "feh")):
        lo, hi, q = split_conformal(
            pred_cal[:, j], sigma_cal[:, j], cal_truth[:, j], pred_test[:, j], sigma_test[:, j], args.alpha
        )
        frame[f"{key}_conf_lo"] = lo
        frame[f"{key}_conf_hi"] = hi
        frame[f"{key}_q"] = q

    frame = frame[
        np.isfinite(frame["teff_gspphot"]) & np.isfinite(frame["logg_gspphot"]) & np.isfinite(frame["mh_gspphot"])
    ].copy()
    summary = {"alpha": args.alpha, "nominal_coverage": 1.0 - args.alpha, "calibration_n": int(len(cal_rows))}
    for key, label in zip(("teff", "logg", "feh"), LABELS):
        gsp_lo, gsp_hi = f"{key}_gspphot_lower", f"{key}_gspphot_upper"
        if key == "feh":
            gsp_lo, gsp_hi = "mh_gspphot_lower", "mh_gspphot_upper"
        truth = f"{key}_true"
        summary[key] = {
            "msa_conformal": coverage(frame, f"{key}_conf_lo", f"{key}_conf_hi", truth),
            "gspphot": coverage(frame, gsp_lo, gsp_hi, truth),
        }
        tail = frame[frame["feh_true"] < -2.0]
        summary[key]["msa_conformal_tail"] = coverage(tail, f"{key}_conf_lo", f"{key}_conf_hi", truth)
        summary[key]["gspphot_tail"] = coverage(tail, gsp_lo, gsp_hi, truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), constrained_layout=True)
    fig.suptitle(
        f"Released 16/84 vs MSA split conformal; common n={len(frame):,}; "
        f"calibration n={len(cal_rows):,}; nominal={1-args.alpha:.3f}",
        fontsize=11.5,
    )
    for col, (key, label, unit) in enumerate(zip(("teff", "logg", "feh"), LABELS, UNITS)):
        ax = axes[0, col]
        msa = summary[key]["msa_conformal"]["coverage"]
        gsp = summary[key]["gspphot"]["coverage"]
        ax.bar([0, 1], [msa if msa is not None else 0, gsp if gsp is not None else 0],
               color=["#c44e52", "#4c72b0"], alpha=0.85)
        ax.axhline(1 - args.alpha, color="0.3", ls="--", lw=0.9)
        ax.set_xticks([0, 1], ["MSA conformal", "GSP-Phot"])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("empirical 68.3% coverage")
        ax.set_title(f"{label}: coverage (n={summary[key]['msa_conformal']['n']})", fontsize=10)
        ax.text(0, (msa or 0) + 0.02, f"MSA {msa:.3f}" if msa is not None else "MSA —", ha="center", fontsize=8)
        ax.text(1, (gsp or 0) + 0.02, f"GSP {gsp:.3f}" if gsp is not None else "GSP —", ha="center", fontsize=8)
        ax = axes[1, col]
        xs, ys, ns = binned_coverage(frame, f"{key}_conf_lo", f"{key}_conf_hi", f"{key}_true")
        ax.plot(xs, ys, "o-", color="#c44e52", label="MSA conformal", lw=1.2)
        gsp_lo, gsp_hi = f"{key}_gspphot_lower", f"{key}_gspphot_upper"
        if key == "feh":
            gsp_lo, gsp_hi = "mh_gspphot_lower", "mh_gspphot_upper"
        xs2, ys2, _ = binned_coverage(frame, gsp_lo, gsp_hi, f"{key}_true")
        ax.plot(xs2, ys2, "s--", color="#4c72b0", label="GSP-Phot", lw=1.2)
        ax.axhline(1 - args.alpha, color="0.3", ls=":", lw=0.9)
        ax.axvline(-2, color="0.55", ls=":", lw=0.8)
        ax.set_xlabel("true [Fe/H]")
        ax.set_ylabel(f"coverage of true {label}")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{label}: coverage vs [Fe/H]", fontsize=10)
        ax.legend(frameon=False, fontsize=7)
    fig.savefig(args.output, dpi=170)
    fig.savefig(args.output.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    print(json.dumps({"status": "complete", "png": str(args.output), "n_common": len(frame), "calibration_n": len(cal_rows)}))


if __name__ == "__main__":
    main()
