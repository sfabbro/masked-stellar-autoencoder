"""Evaluate saved common distributions and native labels on updated GBS references."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from validate_multisurvey_predictions import compare, digest, join

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    summarize_residuals,
)
from masked_stellar_autoencoder.training.survey_expansion_sources import (
    exact_optional_ids,
)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    if (
        digest(args.predictions)
        != "aca9420f31ec08fc1a3d8a10cece8989e532f371d877f2c0150e16df4ff89335"
    ):
        raise ValueError("Expected frozen external-role predictions")
    b = pd.read_csv(args.benchmarks, dtype={"source_id": str}, keep_default_na=False)
    b["source_id"] = exact_optional_ids(b.source_id)
    b = b[b.source_id.notna()].copy()
    with h5py.File(args.predictions) as f:
        data = {k: f[k][:] for k in f}
        checkpoint = str(f.attrs["selected_checkpoint_sha256"])
    rr, jj = join(data["source_id"], b.source_id.to_numpy(dtype="i8"))
    b = b.iloc[jj].copy()
    values = b[["Teff", "logg", "[FeI/H]"]].to_numpy(dtype=float)
    errors = b[["e_Teff", "e_logg", "e_[FeI/H]"]].to_numpy(dtype=float)
    primary = (
        b.direct_diameter.eq(True).to_numpy() & b.spectral_warning.eq(False).to_numpy()
    )
    result = dict(
        checkpoint_sha256=checkpoint,
        prediction_role="external",
        matched=len(b),
        primary=int(primary.sum()),
        training_performed=False,
        benchmark_sha256=digest(args.benchmarks),
        primary_definition="direct angular diameter; no published spectral warning",
        iron_definition="Casamiquela2026 recommended FeI/H; FeII/H retained separately, not averaged",
        scope="reserved from current joint adaptation; inherited training/calibration exposure not independently certified; shared Gaia input dependencies retained",
        native_vs_benchmark={},
        matched_comparisons={},
    )
    for label, k in [("all", np.ones(len(b), dtype=bool)), ("primary", primary)]:
        if k.any():
            result["common_" + label] = compare(
                data,
                rr[k],
                values[k],
                errors[k],
                np.isfinite(values[k]),
                "GBSv3-common-" + label,
                output,
            )
    for s, survey in enumerate(["APOGEE", "DESI", "GALAH"]):
        per_target = {}
        matched = {}
        for t, name in enumerate(["Teff", "logg", "FeH"]):
            k = (
                primary
                & data["native_observed"][rr, s, t].astype(bool)
                & np.isfinite(values[:, t])
            )
            if k.any():
                per_target[name] = summarize_residuals(
                    data["native_value"][rr[k], s, t], values[k, t]
                )
                matched[name] = dict(
                    native=per_target[name],
                    common=summarize_residuals(data["mean"][rr[k], t], values[k, t]),
                )
            b[survey + "_" + name] = np.where(
                data["native_observed"][rr, s, t],
                data["native_value"][rr, s, t],
                np.nan,
            )
        result["native_vs_benchmark"][survey] = per_target
        result["matched_comparisons"][survey] = matched
    if args.mapped is not None:
        if args.mapped_sha256 is None or digest(args.mapped) != args.mapped_sha256:
            raise ValueError("Mapped export content binding differs")
        with h5py.File(args.mapped) as mapped:
            if str(mapped.attrs["checkpoint_sha256"]) != checkpoint:
                raise ValueError("Mapped checkpoint differs")
            np.testing.assert_array_equal(mapped["source_id"][:], data["source_id"])
            weights = mapped["weights"][:][rr]
            components = mapped["component_mean"][:][rr]
            source_mean = (weights[:, None, :, None] * components).sum(2)
        result["mapped_prediction_sha256"] = args.mapped_sha256
        result["mapped_benchmark_triangle"] = {}
        for source_index, survey in enumerate(["APOGEE", "DESI", "GALAH"]):
            result["mapped_benchmark_triangle"][survey] = {}
            for target_index, name in enumerate(["Teff", "logg", "FeH"]):
                k = primary & data["native_observed"][
                    rr, source_index, target_index
                ].astype(bool)
                b["mapped_" + survey + "_" + name] = source_mean[
                    :, source_index, target_index
                ]
                if k.any():
                    prediction = source_mean[k, source_index, target_index]
                    native = data["native_value"][rr[k], source_index, target_index]
                    benchmark = values[k, target_index]
                    result["mapped_benchmark_triangle"][survey][name] = dict(
                        mapped_vs_native=summarize_residuals(prediction, native),
                        native_vs_benchmark=summarize_residuals(native, benchmark),
                        mapped_vs_benchmark=summarize_residuals(prediction, benchmark),
                    )
    if args.mapped is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        for target_index, name in enumerate(["Teff", "logg", "FeH"]):
            keep = primary & data["native_observed"][rr, 0, target_index].astype(bool)
            reference = values[keep, target_index]
            ax = axes[target_index]
            for label, prediction in [
                ("APOGEE catalogue", data["native_value"][rr[keep], 0, target_index]),
                ("Common coordinate", data["mean"][rr[keep], target_index]),
                ("Prediction mapped to APOGEE", source_mean[keep, 0, target_index]),
            ]:
                ax.scatter(reference, prediction, s=20, alpha=0.8, label=label)
            if len(reference):
                lo, hi = np.min(reference), np.max(reference)
                ax.plot([lo, hi], [lo, hi], color="grey", ls="--")
            ax.set(
                xlabel="Updated benchmark " + name,
                ylabel="Reported / predicted " + name,
                title=f"Same {int(keep.sum())} primary benchmark stars",
            )
        axes[0].legend(fontsize=7)
        fig.suptitle(
            "Benchmark triangle: physical reference, common coordinate and APOGEE observation prediction"
        )
        fig.savefig(output / "benchmark-triangle-apogee.png", dpi=160)
        plt.close(fig)
    for t, name in enumerate(["Teff", "logg", "FeH"]):
        b["common_" + name] = data["mean"][rr, t]
        b["common_sigma_" + name] = np.sqrt(data["covariance"][rr, t, t])
    b.to_csv(output / "matched-stars.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mapped", type=Path)
    p.add_argument("--mapped-sha256")
    run(p.parse_args())


if __name__ == "__main__":
    main()
