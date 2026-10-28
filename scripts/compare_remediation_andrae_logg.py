"""Compare saved pilot and Andrae logg predictions on the same development stars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_ANDRAE_SHA = "4ade1351106a8ac8340ba1ed68dcbf9fe71def51a9ae9382031e6e4e80b17b52"
MODELS = ("Frozen spline", "Fine-tuned spline", "Andrae")
RUNS = ("remediation_copula_nsf6_frozen_v1", "remediation_copula_nsf6_ft_v1")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare(copula_dir, andrae_file, output):
    copula_dir, andrae_file, output = map(Path, (copula_dir, andrae_file, output))
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads((copula_dir / "receipts.json").read_text())
    expected = {(r["run_id"], r["name"]): r for r in receipt["artifacts"]}
    paths = [copula_dir / name / "development-predictions.npz" for name in RUNS]
    for run, path in zip(RUNS, paths, strict=True):
        record = expected[(run, path.name)]
        if not record["producer_sha_verified"] or sha(path) != record["sha256"]:
            raise ValueError("Pilot prediction artifact digest differs")
    if sha(andrae_file) != EXPECTED_ANDRAE_SHA:
        raise ValueError("Preserved Andrae array digest differs")
    first, second = [load(path) for path in paths]
    other = load(andrae_file)
    for key in ("source_id", "role", "observed", "real_xp_present", "label_source"):
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_allclose(
        first["value"], second["value"], rtol=0, atol=0, equal_nan=True
    )
    for sample in (first, other):
        ids = sample["source_id"]
        if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
            raise ValueError("Each source must contain unique integer star identities")
    ids, pilot_idx, other_idx = np.intersect1d(
        first["source_id"], other["source_id"], return_indices=True
    )
    reference = first["value"][pilot_idx, 1]
    predictions = [
        first["native_median"][pilot_idx, 1],
        second["native_median"][pilot_idx, 1],
        other["andrae_teff_logg_mh"][other_idx, 1],
    ]
    eligible = (
        first["real_xp_present"][pilot_idx]
        & (first["role"][pilot_idx] == "devtest")
        & other["membership_known"][other_idx]
        & ~other["in_training"][other_idx]
        & first["observed"][pilot_idx, 1]
        & np.isfinite(reference)
    )
    for values in predictions:
        eligible &= np.isfinite(values)
    specs = [
        ("all", None, None),
        ("<1", None, 1.0),
        ("1–2", 1.0, 2.0),
        ("2–3", 2.0, 3.0),
        ("3–4", 3.0, 4.0),
        ("4–4.5", 4.0, 4.5),
        ("≥4.5", 4.5, None),
    ]
    rows = []
    for label, lower, upper in specs:
        mask = eligible.copy()
        if lower is not None:
            mask &= reference >= lower
        if upper is not None:
            mask &= reference < upper
        for model, pred in zip(MODELS, predictions, strict=True):
            residual = pred[mask] - reference[mask]
            rows.append(
                {
                    "bin": label,
                    "model": model,
                    "n": int(mask.sum()),
                    "mae_dex": float(np.mean(np.abs(residual)))
                    if residual.size
                    else None,
                    "bias_dex": float(np.mean(residual)) if residual.size else None,
                    "rmse_dex": float(np.sqrt(np.mean(residual**2)))
                    if residual.size
                    else None,
                }
            )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired-logg.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_names, source_counts = np.unique(
        first["label_source"][pilot_idx, 1][eligible], return_counts=True
    )
    summary = {
        "scope": "Same-star XP-present development comparison, known non-training in cached Andrae membership; not an untouched MSA test or physical truth.",
        "unique_intersection": int(len(ids)),
        "paired_stars": int(eligible.sum()),
        "reference_source_counts": dict(
            zip(source_names.tolist(), source_counts.tolist(), strict=True)
        ),
        "reference_scope": "Current pilot catalog logg, which may differ from earlier report references for the same star",
        "andrae_scope": "Cached Andrae Teff/logg/MH estimates; membership not known outside cached subset",
        "pilot_prediction_sha256": {
            run: sha(path) for run, path in zip(RUNS, paths, strict=True)
        },
        "andrae_array_sha256": sha(andrae_file),
        "rows": rows,
    }
    (output / "paired-logg.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    x = np.arange(len(specs))
    width = 0.24
    colors = ("#1878a0", "#d54b2d", "#44763c")
    for j, (model, color) in enumerate(zip(MODELS, colors, strict=True)):
        model_rows = [row for row in rows if row["model"] == model]
        for ax, column in zip(axes, ("mae_dex", "bias_dex"), strict=True):
            ax.bar(
                x + (j - 1) * width,
                [
                    row[column] if row[column] is not None else np.nan
                    for row in model_rows
                ],
                width,
                color=color,
                label=model,
            )
    axes[0].set_ylabel("Mean absolute error (dex)")
    axes[1].set_ylabel("Mean residual (dex)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    counts = [
        next(row["n"] for row in rows if row["bin"] == label) for label, _, _ in specs
    ]
    axes[1].set_xticks(
        x, [f"{label}\n(n={n})" for (label, _, _), n in zip(specs, counts, strict=True)]
    )
    axes[1].set_xlabel("Catalog logg reference bin (dex)")
    axes[0].legend(frameon=False, ncol=3)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Same-star logg comparison: XP-present development sample")
    fig.text(
        0.5,
        0.01,
        "Known non-training in cached Andrae membership; catalog reference, mostly GALAH; not a blind physical test.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(output / "paired-logg.png", dpi=140)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("copula_dir", type=Path)
    parser.add_argument("andrae_file", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.copula_dir, args.andrae_file, args.output), indent=2))


if __name__ == "__main__":
    main()
