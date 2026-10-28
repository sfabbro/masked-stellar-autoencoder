"""Lightweight saved-array reports; never import or instantiate neural models."""

import json
from pathlib import Path

import numpy as np


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def external_report(predictions, output):
    """APOKASC is an external age scale, never pooled with the teacher posterior."""
    output.mkdir(exist_ok=True)
    if "external_age_observed" not in predictions:
        atomic(
            output / "summary.json",
            dict(status="not_available", pooled_with_teacher=False),
        )
        return
    mask = np.array(predictions["external_age_observed"], dtype=bool, copy=True)
    target = np.asarray(predictions["external_age_value"])
    mask &= np.isfinite(target) & (target > 0)
    if not mask.any():
        atomic(
            output / "summary.json",
            dict(status="no_eligible_references", pooled_with_teacher=False),
        )
        return
    predicted = np.asarray(predictions["point_native"])[mask, 4]
    target = target[mask]
    residual = predicted - target
    result = dict(
        status="source_scale_point_comparison",
        count=len(target),
        mae_gyr=float(np.abs(residual).mean()),
        bias_gyr=float(residual.mean()),
        rmse_gyr=float(np.sqrt(np.mean(residual**2))),
        pooled_with_teacher=False,
        scope="APOKASC3 seismic/model-dependent reference age, not physical truth or a StarHorse posterior target",
    )
    if "external_age_intrinsic_pit" in predictions:
        pit = np.asarray(predictions["external_age_intrinsic_pit"])[mask]
        if not np.isfinite(pit).all() or ((pit < 0) | (pit > 1)).any():
            raise FloatingPointError("Invalid external age CDF")
        result["mean_crps_gyr"] = float(
            np.asarray(predictions["external_age_intrinsic_crps_gyr"])[mask].mean()
        )
    atomic(output / "summary.json", result)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    take = np.arange(len(target))
    if len(take) > 20000:
        take = take[np.linspace(0, len(take) - 1, 20000, dtype=int)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(target[take], predicted[take], s=3, alpha=0.25)
    lo, hi = np.quantile(np.r_[target, predicted], [0.005, 0.995])
    axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[0].set(
        xlabel="APOKASC3 age reference [Gyr]",
        ylabel="Predicted conditional median [Gyr]",
    )
    axes[1].scatter(target[take], residual[take], s=3, alpha=0.25)
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set(
        xlabel="APOKASC3 age reference [Gyr]", ylabel="Prediction minus reference [Gyr]"
    )
    fig.suptitle(
        "External seismic/model age scale; not pooled with StarHorse teacher labels"
    )
    fig.tight_layout()
    fig.savefig(output / "age-reference.png", dpi=180)
    fig.savefig(output / "age-reference.pdf")
    plt.close(fig)
