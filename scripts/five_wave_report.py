"""Saved-array reporting for the five-job, large-sample prediction comparison.

Called inside each training job after checkpoint selection. It never loads a
model or chooses a checkpoint. Figures are exploratory comparisons against
source-qualified reported labels, not latent physical truth.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")
LABELS = (
    "Teff [K]",
    "log g [dex]",
    "[Fe/H] [dex]",
    "[alpha/Fe] [dex]",
    "Age [Gyr]",
    "Distance [kpc]",
)
PLOT_LABELS = (*LABELS[:4], "ln(age / Gyr)", "ln(distance / kpc)")
POINT_ONLY = {"quantile", "quantile3", "three_quantile", "mse", "baseline_mse"}
STATUS = {
    "complete_xp": "xp_present",
    "real_xp_present": "xp_present",
    "xp_present": "xp_present",
    "official_no_xp": "no_xp",
    "real_no_xp": "no_xp",
    "no_xp": "no_xp",
    "official_xp_unknown": "unknown",
    "unknown": "unknown",
    "partial_xp": "partial_xp",
}
COLORS = {
    "natural_xp": "#2478b4",
    "natural_no_xp": "#e18b25",
    "xp_masked_ablation": "#9963b2",
    "natural_unknown": "#777777",
    "natural_partial_xp": "#577f71",
}
REFERENCE_NOTE = "Reported-label references, not physical truth. Age/distance may be catalogue-posterior medians. Missing labels are never imputed."


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _point_metrics(point, target):
    if not len(point):
        return dict(
            count=0,
            mae=None,
            rmse=None,
            bias=None,
            median_absolute_error=None,
            nmad=None,
            r2=None,
        )
    error = point - target
    total = np.sum((target - target.mean()) ** 2)
    return dict(
        count=len(error),
        mae=float(np.abs(error).mean()),
        rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),
        median_absolute_error=float(np.median(np.abs(error))),
        nmad=float(1.4826 * np.median(np.abs(error - np.median(error)))),
        r2=float(1 - np.sum(error**2) / total) if total else None,
    )


def _ks(pit):
    if not len(pit):
        return None
    values = np.sort(pit)
    ranks = np.arange(len(values)) / len(values)
    return float(max(np.max(values - ranks), np.max(ranks + 1 / len(values) - values)))


def _stable_take(ids, rows, maximum=20000):
    """Outcome-independent plotting sample, identical across head variants."""
    if len(rows) <= maximum:
        return rows
    with np.errstate(over="ignore"):
        key = np.asarray(ids[rows], dtype="u8") + np.uint64(0x9E3779B97F4A7C15)
        key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        key ^= key >> np.uint64(31)
    return rows[np.argsort(key, kind="stable")[:maximum]]


def validate_predictions(predictions, head_kind):
    data = {k: np.asarray(v) for k, v in predictions.items()}
    ids = data["source_id"]
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or (ids <= 0).any():
        raise ValueError("Exact positive integer Gaia IDs required")
    n = len(ids)
    for key in ("target_native", "observed"):
        if data[key].shape != (n, 6):
            raise ValueError(f"{key} must be [N,6]")
    point = data.get("point_native", data.get("mean_native"))
    if point is None or point.shape != (n, 6) or not np.isfinite(point).all():
        raise ValueError("Finite six-target point predictions required")
    data["point_native"] = point
    mask = np.asarray(data["observed"], dtype=bool)
    if not np.isfinite(data["target_native"][mask]).all():
        raise ValueError("An observed reference must be finite")
    if (data["target_native"][:, 4:][mask[:, 4:]] <= 0).any():
        raise ValueError("Observed age/distance must be positive")
    data["observed"] = mask
    mode = data.get("evaluation_mode", np.full(n, "natural")).astype(str)
    if mode.ndim == 0:
        mode = np.full(n, str(mode))
    if mode.shape != (n,) or not np.isin(mode, ["natural", "xp_masked_ablation"]).all():
        raise ValueError(
            "Separate natural and paired xp_masked_ablation modes required"
        )
    if len(set(zip(ids.tolist(), mode.tolist(), strict=True))) != n:
        raise ValueError("Duplicate source ID within an evaluation mode")
    if "xp_status" in data:
        status = np.array([STATUS.get(str(x), "invalid") for x in data["xp_status"]])
        if len(status) != n or (status == "invalid").any():
            raise ValueError("Unknown source-certified XP status")
    elif "real_xp_present" in data and "real_no_xp" in data:
        present, absent = (
            data["real_xp_present"].astype(bool),
            data["real_no_xp"].astype(bool),
        )
        if present.shape != (n,) or absent.shape != (n,) or (present & absent).any():
            raise ValueError("Contradictory original XP availability")
        status = np.where(present, "xp_present", np.where(absent, "no_xp", "unknown"))
    else:
        raise ValueError(
            "Original certified XP status required; input missingness is not proof of no XP"
        )
    if ((mode == "xp_masked_ablation") & (status != "xp_present")).any():
        raise ValueError("Artificial XP masking is only a paired XP-star ablation")
    data["evaluation_mode"], data["xp_status"] = mode, status
    data["label_source"] = data.get(
        "label_source", np.full((n, 6), "unspecified")
    ).astype(str)
    if data["label_source"].shape != (n, 6):
        raise ValueError("Label-source matrix must be [N,6]")
    for aliases in (
        ("pit_intrinsic", "intrinsic_pit"),
        ("pit_observed", "observed_pit"),
    ):
        available = next((data[k] for k in aliases if k in data), None)
        if available is not None:
            if head_kind in POINT_ONLY:
                raise ValueError(
                    "Three quantiles/MSE do not define a unique predictive CDF or PIT"
                )
            default_valid = np.isfinite(available)
            if aliases[0] == "pit_observed" and "error_observed" in data:
                default_valid &= data["error_observed"].astype(bool)
            valid = mask & data.get(aliases[0] + "_valid", default_valid).astype(bool)
            if valid.shape != (n, 6):
                raise ValueError("PIT eligibility mask must be [N,6]")
            if (
                available.shape != (n, 6)
                or not np.isfinite(available[valid]).all()
                or ((available[valid] < 0) | (available[valid] > 1)).any()
            ):
                raise ValueError("Invalid PIT values on eligible reference labels")
            data[aliases[0]] = available
            data[aliases[0] + "_valid"] = valid
    if "quantiles_native" in data:
        q = data["quantiles_native"]
        levels = data.get("quantile_levels")
        if (
            levels is None
            or levels.ndim != 1
            or not np.all(np.diff(levels) > 0)
            or (levels <= 0).any()
            or (levels >= 1).any()
        ):
            raise ValueError("Increasing explicit quantile levels in (0,1) required")
        if (
            q.shape != (n, len(levels), 6)
            or not np.isfinite(q).all()
            or (np.diff(q, axis=1) < 0).any()
        ):
            raise ValueError("Finite ordered [N,Q,6] quantiles required")
    return data


def populations(data, *, ancestry=False):
    mode, xp = data["evaluation_mode"], data["xp_status"]
    result = {
        "natural_xp": (mode == "natural") & (xp == "xp_present"),
        "natural_no_xp": (mode == "natural") & (xp == "no_xp"),
        "xp_masked_ablation": mode == "xp_masked_ablation",
    }
    for label, status in (
        ("natural_unknown", "unknown"),
        ("natural_partial_xp", "partial_xp"),
    ):
        if ((mode == "natural") & (xp == status)).any():
            result[label] = (mode == "natural") & (xp == status)
    if ancestry:
        original = list(result.items())
        for field in (
            "ancestor_feh_supervised_fit_exposed",
            "ancestor_150m_reconstruction_exposed",
        ):
            if field not in data:
                continue
            flag = np.asarray(data[field], dtype=bool)
            if flag.shape != mode.shape:
                raise ValueError("Ancestry flags must be source-bound [N]")
            for name, mask in original:
                for label, subset in (("exposed", flag), ("not_exposed", ~flag)):
                    result[name + "__" + field + "__" + label] = mask & subset
    return result


def metric_rows(data, run_id):
    rows = []
    y, point, mask = data["target_native"], data["point_native"], data["observed"]
    metalpoor = mask[:, 2] & (y[:, 2] < -2)
    for population, sample in populations(data, ancestry=True).items():
        for scope, subset in (("global", sample), ("metal_poor", sample & metalpoor)):
            for c, name in enumerate(NAMES):
                eligible = subset & mask[:, c]
                for survey in [
                    "all_sources",
                    *sorted(set(data["label_source"][eligible, c])),
                ]:
                    valid = (
                        eligible
                        if survey == "all_sources"
                        else eligible & (data["label_source"][:, c] == survey)
                    )
                    row = dict(
                        run_id=run_id,
                        population=population,
                        scope=scope,
                        target=name,
                        label_source=survey,
                        **_point_metrics(point[valid, c], y[valid, c]),
                    )
                    for key in ("pit_intrinsic", "pit_observed"):
                        if key in data:
                            pit_valid = valid & data[key + "_valid"][:, c]
                            row[key + "_count"] = int(pit_valid.sum())
                            row[key + "_ks"] = _ks(data[key][pit_valid, c])
                            row[key + "_descriptive_only"] = True
                    if "quantiles_native" in data:
                        for j, level in enumerate(data["quantile_levels"]):
                            delta = y[valid, c] - data["quantiles_native"][valid, j, c]
                            row[f"pinball_{level:g}"] = (
                                float(
                                    np.maximum(
                                        level * delta, (level - 1) * delta
                                    ).mean()
                                )
                                if valid.any()
                                else None
                            )
                        for coverage, lo, hi in ((90, 0.05, 0.95), (95, 0.025, 0.975)):
                            low = np.flatnonzero(
                                np.isclose(data["quantile_levels"], lo)
                            )
                            high = np.flatnonzero(
                                np.isclose(data["quantile_levels"], hi)
                            )
                            if len(low) and len(high):
                                lower = data["quantiles_native"][valid, low[0], c]
                                upper = data["quantiles_native"][valid, high[0], c]
                                row[f"predictive_{coverage}_coverage"] = (
                                    float(
                                        (
                                            (y[valid, c] >= lower)
                                            & (y[valid, c] <= upper)
                                        ).mean()
                                    )
                                    if valid.any()
                                    else None
                                )
                                row[f"predictive_{coverage}_mean_width"] = (
                                    float((upper - lower).mean())
                                    if valid.any()
                                    else None
                                )
                    rows.append(row)
    return rows


def distribution_score_rows(data, run_id):
    """Aggregate supplied family-specific scores; unknown measurement errors excluded.

    Native CRPS/pinball/interval scores keep per-coordinate units. Joint scores
    use the runner's FIT-standardized log-age/log-distance coordinates only.
    """
    result = []
    n = len(data["source_id"])
    tail = data["observed"][:, 2] & (data["target_native"][:, 2] < -2)
    groups = {"native3": [0, 1, 2], "age_distance": [4, 5], "full6": list(range(6))}
    for key, values in data.items():
        if (
            not key.startswith(("intrinsic_", "observed_"))
            or "pit" in key
            or "quantile_" in key
        ):
            continue
        if not any(
            t in key
            for t in (
                "crps_",
                "pinball_",
                "predictive_",
                "energy_",
                "variogram_",
                "vario_rho1_",
                "log_prob_native",
            )
        ):
            continue
        if values.shape not in {(n,), (n, 6)}:
            raise ValueError(f"Score {key} has invalid row shape")
        for population, keep in populations(data, ancestry=True).items():
            for scope, subset in (("global", keep), ("metal_poor", keep & tail)):
                targets = range(6) if values.ndim == 2 else [None]
                for c in targets:
                    if c is None:
                        group = next((g for g in groups if "_" + g + "_" in key), None)
                        if group is None:
                            raise ValueError(
                                f"Joint score {key} requires an explicit reference block"
                            )
                        eligible = data["observed"][:, groups[group]].all(1)
                        if key.startswith("observed_"):
                            eligible &= data.get(
                                "pit_observed_valid", np.zeros((n, 6), bool)
                            )[:, groups[group]].all(1)
                        scores = values
                        name = group
                    else:
                        eligible = data["observed"][:, c].copy()
                        if key.startswith("observed_"):
                            eligible &= data.get(
                                "pit_observed_valid", np.zeros((n, 6), bool)
                            )[:, c]
                        scores = values[:, c]
                        name = NAMES[c]
                    if key.endswith("log_prob_native"):
                        scores = -scores
                    valid = subset & eligible & np.isfinite(scores)
                    result.append(
                        dict(
                            run_id=run_id,
                            population=population,
                            scope=scope,
                            target=name,
                            metric=key.replace("log_prob_native", "nll_native"),
                            count=int(valid.sum()),
                            mean=float(scores[valid].mean()) if valid.any() else None,
                            measurement_error_scope="known_quoted_errors_only"
                            if key.startswith("observed_")
                            else "intrinsic_predictive_law",
                        )
                    )
    return result


def _limits(reference):
    if not len(reference):
        return (0.0, 1.0)
    lo, hi = np.quantile(reference, [0.005, 0.995])
    width = hi - lo
    pad = 0.08 * width if width > 0 else max(abs(lo) * 0.05, 0.1)
    return float(lo - pad), float(hi + pad)


def _save(fig, stem):
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_points(data, output, run_id, metadata):
    figures = []
    y, point, observed = data["target_native"], data["point_native"], data["observed"]
    metalpoor = observed[:, 2] & (y[:, 2] < -2)
    for name, tail, residual in (
        ("predicted-vs-reference", False, False),
        ("residual-vs-reference-metalpoor", True, True),
        ("residual-vs-feh-metalpoor", True, True),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for c, ax in enumerate(axes.flat):
            all_rows = observed[:, c] & ((metalpoor) if tail else np.ones(len(y), bool))
            all_rows &= data["evaluation_mode"] == "natural"
            limits = _limits(
                y[all_rows, 2 if name == "residual-vs-feh-metalpoor" else c]
            )
            for population, subset in populations(data).items():
                valid = subset & observed[:, c] & (metalpoor if tail else True)
                rows = _stable_take(data["source_id"], np.flatnonzero(valid))
                x = y[rows, 2] if name == "residual-vs-feh-metalpoor" else y[rows, c]
                value = point[rows, c] - y[rows, c] if residual else point[rows, c]
                if len(rows):
                    ax.scatter(
                        x,
                        value,
                        s=4,
                        alpha=0.16,
                        color=COLORS[population],
                        rasterized=True,
                        label=f"{population}: n={int(valid.sum()):,}",
                    )
            ax.set_xlim(limits)
            if residual:
                residuals = point[all_rows, c] - y[all_rows, c]
                ax.set_ylim(_limits(residuals))
                ax.axhline(0, color="black", ls="--", lw=0.8)
                ax.set_ylabel("Prediction − reference: " + LABELS[c])
            else:
                ax.set_ylim(limits)
                ax.plot(limits, limits, "k--", lw=0.8)
                ax.set_ylabel("Predicted " + LABELS[c])
            ax.set_xlabel(
                "Reference [Fe/H] [dex]"
                if name == "residual-vs-feh-metalpoor"
                else "Reference " + LABELS[c]
            )
            ax.set_title(NAMES[c] + f" · natural eligible n={int(all_rows.sum()):,}")
            if not all_rows.any():
                ax.text(
                    0.5,
                    0.5,
                    "No eligible natural references",
                    transform=ax.transAxes,
                    ha="center",
                )
            ax.grid(alpha=0.15)
            if c == 0:
                ax.legend(fontsize=7)
        fig.suptitle(
            run_id
            + (
                " · metal-poor reference FeH < −2"
                if tail
                else " · six source-qualified targets"
            ),
            fontsize=11,
        )
        fig.text(
            0.02,
            0.012,
            REFERENCE_NOTE
            + " Axes use central 99% of references/residuals; full-population metrics include all finite eligible rows.\nNatural no-XP is official availability; masked-XP is a separate stress ablation. Point functional: "
            + str(metadata.get("point_functional", "conditional mean")),
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
        _save(fig, output / name)
        figures.append(name)
    return figures


def plot_pits(data, output, run_id):
    figures = []
    for key in ("pit_intrinsic", "pit_observed"):
        if key not in data:
            continue
        for tail in (False, True):
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            for c, ax in enumerate(axes.flat):
                for population, sample in populations(data).items():
                    valid = sample & data["observed"][:, c] & data[key + "_valid"][:, c]
                    if tail:
                        valid &= data["observed"][:, 2] & (
                            data["target_native"][:, 2] < -2
                        )
                    if valid.any():
                        ax.hist(
                            data[key][valid, c],
                            bins=np.linspace(0, 1, 21),
                            density=True,
                            histtype="step",
                            lw=1.4,
                            color=COLORS[population],
                            label=f"{population}: n={int(valid.sum()):,}",
                        )
                ax.set_xlim(0, 1)
                if not tail:
                    ax.axhline(1, color="gray", ls="--", lw=0.7)
                ax.set_title(NAMES[c])
                ax.set_xlabel(key)
                ax.set_ylabel("Density")
                ax.grid(alpha=0.15)
                if c == 0:
                    ax.legend(fontsize=7)
            title = key + (
                " · metal-poor conditional subset (descriptive only)"
                if tail
                else " · global reported labels"
            )
            fig.suptitle(run_id + " · " + title, fontsize=11)
            fig.text(
                0.02,
                0.015,
                "No latent-truth claim. Tail selection on observed FeH changes the PIT null: no tail-uniformity hypothesis test.\nQuoted-error observed CDF and intrinsic CDF are different; genuine no-XP and artificial XP masking are never pooled.",
                fontsize=8,
            )
            fig.tight_layout(rect=(0, 0.055, 1, 0.95))
            stem = key + ("-metalpoor" if tail else "-global")
            _save(fig, output / stem)
            figures.append(stem)
    return figures


def andrae_comparison(data, output, run_id):
    if "andrae_prediction_native" not in data:
        return dict(
            status="unavailable",
            reason="No source-ID-bound Andrae atmospheric-three projection supplied",
        )
    a = data["andrae_prediction_native"]
    if a.shape != (len(data["source_id"]), 3):
        raise ValueError("Andrae projection must be [N,3] Teff/logg/MH")
    membership = data.get("andrae_in_training", np.zeros(len(a), bool)).astype(bool)
    known = data.get(
        "andrae_membership_known",
        data.get("andrae_training_membership_known", np.zeros(len(a), bool)),
    ).astype(bool)
    matched = data.get("andrae_matched", np.isfinite(a).all(1)).astype(bool)
    if (matched & (data["xp_status"] == "no_xp")).any():
        raise ValueError(
            "Andrae175M XP catalogue unexpectedly matched a certified no-XP source"
        )
    paired = matched & np.isfinite(a).all(1) & data["observed"][:, :3].all(1)
    stats = []
    for population, sample in populations(data).items():
        for training, heldout in (
            ("all_matched", np.ones(len(a), bool)),
            ("not_in_andrae_training", known & ~membership),
        ):
            for tail in (False, True):
                valid = paired & sample & heldout
                if tail:
                    valid &= data["target_native"][:, 2] < -2
                for c, name in enumerate(NAMES[:3]):
                    for system, pred in (("ours", data["point_native"]), ("andrae", a)):
                        stats.append(
                            dict(
                                run_id=run_id,
                                population=population,
                                comparison_subset=training,
                                scope="metal_poor" if tail else "global",
                                target=name,
                                system=system,
                                mh_vs_feh_proxy=bool(c == 2),
                                **_point_metrics(
                                    pred[valid, c], data["target_native"][valid, c]
                                ),
                            )
                        )
    np.savez_compressed(
        output / "andrae-paired.npz",
        source_id=data["source_id"][paired],
        evaluation_mode=data["evaluation_mode"][paired],
        reference_native=data["target_native"][paired, :3],
        ours_native=data["point_native"][paired, :3],
        andrae_teff_logg_mh=a[paired],
        in_training=membership[paired],
        membership_known=known[paired],
    )
    save_json(output / "andrae-comparison.json", stats)
    figure_names = []
    for tail in (False, True):
        eligible = paired & (data["evaluation_mode"] == "natural")
        if tail:
            eligible &= data["target_native"][:, 2] < -2
        selected = _stable_take(data["source_id"], np.flatnonzero(eligible))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for column in range(3):
            target = data["target_native"][selected, column]
            bounds = _limits(data["target_native"][eligible, column])
            for system, prediction, color in (
                ("ours", data["point_native"], "#2478b4"),
                ("Andrae", a, "#e18b25"),
            ):
                value = prediction[selected, column]
                axes[0, column].scatter(
                    target,
                    value,
                    s=4,
                    alpha=0.15,
                    color=color,
                    label=system,
                    rasterized=True,
                )
                axes[1, column].scatter(
                    target,
                    value - target,
                    s=4,
                    alpha=0.15,
                    color=color,
                    label=system,
                    rasterized=True,
                )
            axes[0, column].plot(bounds, bounds, "k--", lw=0.8)
            axes[0, column].set_ylim(bounds)
            axes[1, column].axhline(0, color="black", ls="--", lw=0.8)
            all_residuals = np.concatenate(
                [
                    data["point_native"][eligible, column]
                    - data["target_native"][eligible, column],
                    a[eligible, column] - data["target_native"][eligible, column],
                ]
            )
            axes[1, column].set_ylim(_limits(all_residuals))
            for i in range(2):
                axes[i, column].set_xlim(bounds)
                axes[i, column].set_xlabel("Reported reference " + LABELS[column])
                axes[i, column].grid(alpha=0.15)
            axes[0, column].set_title(
                NAMES[column] + (" · M/H vs Fe/H proxy" if column == 2 else "")
            )
            axes[0, column].set_ylabel("Prediction")
            axes[1, column].set_ylabel("Prediction − reference")
        axes[0, 0].legend()
        fig.suptitle(
            run_id
            + " · same-object Andrae comparison · "
            + ("metal poor" if tail else "global")
            + f" · n={int(eligible.sum()):,}"
        )
        fig.text(
            0.02,
            0.015,
            "Native atmospheric-three only. Andrae M/H differs from Fe/H. Training-membership-stratified statistics are in CSV/JSON; figures include all matched natural XP sources. Missing alpha/age/distance are not inferred.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.045, 1, 0.95))
        stem = "andrae-paired-" + ("metalpoor" if tail else "global")
        _save(fig, output / stem)
        figure_names.append(stem)
    return dict(
        status="paired_comparison_complete",
        paired_rows=int(paired.sum()),
        figures=figure_names,
        known_nontraining_rows=int((paired & known & ~membership).sum()),
        rows=stats,
        scope="Teff/logg direct; Andrae M/H versus reference Fe/H is explicitly a proxy, not identity. No Andrae alpha/age/distance truth. XP-masked comparison has unequal available inputs.",
    )


def histogram_mass_levels(counts, total_samples, probabilities=(0.68, 0.95)):
    """Mass levels refer to ALL draws, not a renormalized plotting window."""
    counts = np.asarray(counts, dtype="f8")
    if (
        counts.ndim != 2
        or (counts < 0).any()
        or not np.isfinite(counts).all()
        or total_samples < counts.sum()
        or total_samples <= 0
    ):
        raise ValueError("Invalid full-sample histogram accounting")
    ordered = np.sort(counts.ravel())[::-1]
    cumulative = np.cumsum(ordered) / total_samples
    return [
        None
        if not len(ordered) or cumulative[-1] < p
        else float(ordered[min(np.searchsorted(cumulative, p), len(ordered) - 1)])
        for p in probabilities
    ]


def corner_figures(corners, output, run_id, head_kind, metadata):
    if head_kind in POINT_ONLY:
        if corners is not None:
            raise ValueError(
                "Quantile/MSE baseline has no normalized joint law; no invented corners"
            )
        return dict(
            status="unsupported",
            reason="Three marginal quantiles or MSE points do not define a unique joint predictive distribution",
        )
    if corners is None:
        return dict(
            status="not_provided", reason="No saved actual-model joint draws supplied"
        )
    d = {k: np.asarray(v) for k, v in corners.items()}
    samples = d["samples_native"]
    if (
        samples.ndim != 3
        or samples.shape[2] != 6
        or not np.isfinite(samples).all()
        or (samples[:, :, 4:] <= 0).any()
    ):
        raise ValueError("Finite positive-support actual six-target draws required")
    k, s, _ = samples.shape
    if s < 256 or d["source_id"].shape != (k,) or len(np.unique(d["source_id"])) != k:
        raise ValueError("At least 256 draws per unique fixed source required")
    plotted = samples.copy()
    plotted[:, :, 4:] = np.log(plotted[:, :, 4:])
    if "samples_plot_coordinates" in d and not np.allclose(
        d["samples_plot_coordinates"], plotted, rtol=1e-10, atol=1e-10
    ):
        raise ValueError("Corner coordinates inconsistent with native/log support")
    values = d.get("target_native", np.full((k, 6), np.nan))
    observed = d.get("observed", np.zeros((k, 6), bool)).astype(bool)
    if (
        values.shape != (k, 6)
        or observed.shape != (k, 6)
        or not np.isfinite(values[observed]).all()
        or (values[:, 4:][observed[:, 4:]] <= 0).any()
    ):
        raise ValueError("Invalid source-qualified corner references")
    fixed = "plot_limits" in d
    limits = d.get("plot_limits")
    if fixed:
        if limits.shape == (6, 2):
            limits = np.broadcast_to(limits, (k, 6, 2))
        if (
            limits.shape != (k, 6, 2)
            or not np.isfinite(limits).all()
            or not (limits[:, :, 1] > limits[:, :, 0]).all()
        ):
            raise ValueError("Fixed per-source plotted limits must be [K,6,2]")
    receipt = []
    directory = output / "corners"
    directory.mkdir(exist_ok=True)
    for row, sid in enumerate(d["source_id"]):
        x = plotted[row]
        bounds = limits[row] if fixed else np.quantile(x, [0.002, 0.998], axis=0).T
        pad = np.maximum(np.diff(bounds, axis=1)[:, 0] * 0.04, 1e-7)
        if not fixed:
            bounds = bounds + np.stack([-pad, pad], 1)
        reference = values[row].copy()
        reference[4:] = np.log(np.where(observed[row, 4:], reference[4:], 1))
        fig, axes = plt.subplots(6, 6, figsize=(14, 14))
        outside = np.mean((x < bounds[:, 0]) | (x > bounds[:, 1]), axis=0)
        for i in range(6):
            for j in range(6):
                ax = axes[i, j]
                if i < j:
                    ax.set_visible(False)
                    continue
                if i == j:
                    bins = np.linspace(*bounds[i], 45)
                    weight = np.full(s, 1 / (s * (bins[1] - bins[0])))
                    ax.hist(x[:, i], bins=bins, weights=weight, color="#557fa2")
                    ax.text(
                        0.97,
                        0.96,
                        f"outside {100 * outside[i]:.1f}%",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=6,
                    )
                else:
                    counts, xx, yy = np.histogram2d(
                        x[:, j], x[:, i], bins=42, range=[bounds[j], bounds[i]]
                    )
                    ax.pcolormesh(xx, yy, counts.T, shading="auto", cmap="Blues")
                    levels = [
                        v
                        for v in histogram_mass_levels(counts, s)
                        if v is not None and v > 0
                    ]
                    levels = np.unique(levels)
                    if len(levels):
                        ax.contour(
                            (xx[:-1] + xx[1:]) / 2,
                            (yy[:-1] + yy[1:]) / 2,
                            counts.T,
                            levels=levels,
                            colors="#315879",
                            linewidths=0.7,
                        )
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "68/95% full mass outside view",
                            transform=ax.transAxes,
                            ha="center",
                            fontsize=6,
                        )
                if observed[row, j]:
                    ax.axvline(reference[j], color="#b12d2d", lw=0.8)
                if i != j and observed[row, i]:
                    ax.axhline(reference[i], color="#b12d2d", lw=0.8)
                ax.set_xlim(bounds[j])
                if i != j:
                    ax.set_ylim(bounds[i])
                if i == 5:
                    ax.set_xlabel(PLOT_LABELS[j], fontsize=8)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0 and i > 0:
                    ax.set_ylabel(PLOT_LABELS[i], fontsize=8)
                elif i != j:
                    ax.tick_params(labelleft=False)
                ax.tick_params(labelsize=6)
        dependence = metadata.get(
            "distribution_dependence",
            "Independent conditional coordinates"
            if head_kind != "mdn"
            else "Shared mixture component; sparse coupling is not identified physical dependence",
        )
        fig.suptitle(
            f"Gaia {int(sid)} · {run_id}\n{dependence}; {s:,} actual predictive draws; 68/95% mass in plotted coordinates",
            fontsize=10,
        )
        fig.text(
            0.02,
            0.015,
            "Red: eligible reported reference. Missing labels absent. Positive age/distance plotted in natural logs.\nPredictive draws are not conformal sets or a physical posterior. Truncated-view mass is reported; raw draws are retained.",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
        stem = directory / str(int(sid))
        _save(fig, stem)
        np.savez_compressed(
            directory / f"{int(sid)}-samples.npz",
            samples_native=samples[row],
            samples_plot_coordinates=plotted[row],
            source_id=sid,
            target_native=values[row],
            observed=observed[row],
            plot_limits=bounds,
            label_source=d.get("label_source", np.full((k, 6), "unspecified"))[row],
        )
        receipt.append(
            dict(
                source_id=int(sid),
                draws=s,
                fixed_source_axes=fixed,
                outside_fraction=outside.tolist(),
                png_sha256=sha(stem.with_suffix(".png")),
                pdf_sha256=sha(stem.with_suffix(".pdf")),
                samples_sha256=sha(directory / f"{int(sid)}-samples.npz"),
            )
        )
    return dict(
        status="saved_draw_figures_complete_visual_review_pending",
        models_or_checkpoints_loaded=False,
        rows=receipt,
        common_axes_source_frozen=fixed,
    )


def make_report(predictions, output, *, run_id, head_kind, corners=None, metadata=None):
    """Render all six labels and return a JSON-safe report manifest.

    Quantiles are [N,Q,6]. Original XP status must be certified source metadata.
    Optional Andrae prediction_native is [N,3] Teff/logg/MH and has membership
    flags. Corner references are source-qualified; missing coordinates absent.
    """
    metadata = {} if metadata is None else dict(metadata)
    data = validate_predictions(predictions, head_kind)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (
        head_kind in POINT_ONLY
        and "median" not in metadata.get("point_functional", "median")
        and head_kind not in {"mse", "baseline_mse"}
    ):
        raise ValueError("Three-quantile baseline point functional must be median")
    metrics = metric_rows(data, run_id)
    save_json(output / "metrics.json", metrics)
    keys = sorted(set().union(*(r.keys() for r in metrics))) if metrics else []
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(metrics)
    scores = distribution_score_rows(data, run_id)
    save_json(output / "distribution-scores.json", scores)
    with (output / "distribution-scores.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id",
                "population",
                "scope",
                "target",
                "metric",
                "count",
                "mean",
                "measurement_error_scope",
            ],
        )
        writer.writeheader()
        writer.writerows(scores)
    figures = plot_points(data, output, run_id, metadata) + plot_pits(
        data, output, run_id
    )
    andrae = andrae_comparison(data, output, run_id)
    corners_info = corner_figures(corners, output, run_id, head_kind, metadata)
    summary = dict(
        status="report_generated_visual_review_pending",
        run_id=run_id,
        head_kind=head_kind,
        rows=len(data["source_id"]),
        metadata=metadata,
        figures=figures,
        population_counts={k: int(v.sum()) for k, v in populations(data).items()},
        six_target_complete_reference_count=int(data["observed"].all(1).sum()),
        pit_status="unsupported_point_or_quantile_baseline"
        if head_kind in POINT_ONLY
        else "provided"
        if any(k in data for k in ["pit_intrinsic", "pit_observed"])
        else "not_provided",
        corners=corners_info,
        andrae=andrae,
        reference_scope=REFERENCE_NOTE,
        no_xp_scope="Natural no-XP requires original official availability; masked-XP is a distinct feature-limited stress control. Artificially masked CAL is not a genuine no-XP exchangeability guarantee.",
        pit_scope="Tail PIT KS descriptive only; conditioning on measured FeH changes the uniform null. No physical-truth p-values or coverage guarantees.",
        no_checkpoint_or_model_loaded=True,
        ancestry_scope="FeH supervised ancestry and unsupervised 150M reconstruction exposure are separate source-ID flags; exposure is not automatically leakage for all six target labels.",
        score_scope="NLL = negative mean native log probability; lower CRPS/NLL/energy/variogram/pinball/interval score is better. Vario rho1 = negative CRPS is higher better. Predictive intervals are not automatically conformal intervals.",
    )
    save_json(output / "report.json", summary)
    body = [
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>"
        + html.escape(run_id)
        + "</title><style>body{font:15px/1.5 system-ui;max-width:1300px;margin:auto;padding:25px}img{max-width:100%}article{margin:20px 0}code{overflow-wrap:anywhere}.warn{border-left:4px solid #b77830;padding:10px;background:#fff8e9}</style><h1>"
        + html.escape(run_id)
        + "</h1><p class='warn'>"
        + html.escape(REFERENCE_NOTE)
        + " Natural no-XP and paired XP masking are separate populations. This saved-output report is not a complete independent science certificate.</p><p><a href='metrics.csv'>All point/interval/PIT metric rows CSV</a> · <a href='metrics.json'>JSON</a> · <a href='distribution-scores.csv'>Family-specific proper scores</a> · <a href='report.json'>Scope and provenance</a></p>"
    ]
    for fig in [*figures, *andrae.get("figures", [])]:
        body.append(
            "<article><h2>"
            + html.escape(fig)
            + "</h2><a href='"
            + fig
            + ".pdf'>Vector PDF</a><img src='"
            + fig
            + ".png' alt='"
            + html.escape(fig)
            + "'></article>"
        )
    if summary["pit_status"].startswith("unsupported"):
        body.append(
            "<h2>PIT and normalized predictive density unavailable</h2><p>Three quantiles/MSE do not define a unique CDF or joint predictive distribution; no PIT, PDF or full corner plot is invented.</p>"
        )
    if andrae["status"] == "paired_comparison_complete":
        body.append(
            "<h2>Andrae atmospheric-three comparison</h2><p>"
            + html.escape(andrae["scope"])
            + "</p><a href='andrae-comparison.json'>Same-object metric rows</a> · <a href='andrae-paired.npz'>Paired arrays</a>"
        )
    else:
        body.append(
            "<h2>Andrae comparison unavailable</h2><p>"
            + html.escape(andrae["reason"])
            + "</p>"
        )
    for row in corners_info.get("rows", []):
        stem = "corners/" + str(row["source_id"])
        body.append(
            "<article><h2>Gaia "
            + str(row["source_id"])
            + " full six-target conditional draws</h2><a href='"
            + stem
            + ".pdf'>PDF</a> · <a href='"
            + stem
            + "-samples.npz'>Raw samples</a><img src='"
            + stem
            + ".png' alt='Six-target corner'></article>"
        )
    body.append("</html>")
    (output / "index.html").write_text("\n".join(body) + "\n")
    return _jsonable(summary)
