"""Read-only training QA, sourced only from verified published CSV snapshots."""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from monitor.training_qa_io import fetch_snapshot, read_csv

    return UTC, datetime, fetch_snapshot, mo, os, pd, plt, read_csv


@app.cell
def _(mo, os):
    remote = mo.ui.text(
        value=os.environ.get("MSA_QA_REMOTE", "vos:sfabbro/msa/joint_native_30e_v1/qa"),
        label="Published QA directory",
    )
    offline = mo.ui.checkbox(
        value=os.environ.get("MSA_QA_OFFLINE") == "1",
        label="Offline: use verified cache only",
    )
    refresh = mo.ui.refresh(
        options=["1m", "5m", "15m"], default_interval="5m", label="Refresh"
    )
    mo.vstack(
        [mo.md("# Joint stellar training QA"), mo.hstack([remote, offline, refresh])]
    )
    return offline, refresh, remote


@app.cell
def _(UTC, datetime, fetch_snapshot, mo, offline, os, pd, read_csv, refresh, remote):
    _tick = refresh.value
    snapshot = None
    fetch_error = None
    frames = {}
    try:
        snapshot = fetch_snapshot(
            remote.value.strip(),
            os.environ.get("MSA_QA_CACHE", "experiments/xp/training-qa-cache"),
            offline=offline.value,
        )
        for _name in snapshot.manifest["files"]:
            frames[_name] = pd.DataFrame(read_csv(snapshot, _name))
            for _column in (
                "epoch",
                "step",
                "value",
                "truth",
                "prediction",
                "lower68",
                "upper68",
                "lower95",
                "upper95",
                "count",
                "gpu_utilization_pct",
                "gpu_memory_used_mb",
                "rss_mb",
                "examples_per_second",
            ):
                if _column in frames[_name]:
                    frames[_name][_column] = pd.to_numeric(
                        frames[_name][_column], errors="coerce"
                    )
    except (OSError, ValueError):
        fetch_error = "No verified snapshot available. Check the run publication and CADC proxy, or choose an existing cache."
    if snapshot is not None:
        _manifest = snapshot.manifest
        _age = max(
            0,
            (
                datetime.now(UTC)
                - datetime.fromisoformat(
                    _manifest["generated_at_utc"].replace("Z", "+00:00")
                )
            ).total_seconds()
            / 60,
        )
        _notice = (
            snapshot.warning or "Verified current publication fetched successfully"
        )
        _view = mo.vstack(
            [
                mo.md(f"**{_notice}**"),
                mo.md(
                    f"Run `{_manifest['run_id']}` · status `{_manifest['status']}` · publication `{_manifest['publication_id']}` · published {_age:.1f} minutes ago"
                ),
                mo.md(
                    f"Source `{_manifest['source_sha']}` · parent `{_manifest['parent_sha256']}`"
                ),
                mo.md(
                    "Snapshot age is shown explicitly. Successful download does not establish that a training session is healthy. QA uses tuning/monitoring roles; do not repeatedly select on a locked final test."
                ),
            ]
        )
    else:
        _view = mo.md(f"**{fetch_error}**")
    _view
    return frames, snapshot


@app.cell
def _(frames, mo):
    _predictions = frames.get("predictions.csv")
    _metrics = frames.get("metrics.csv")
    _targets = sorted(
        set(
            (
                _predictions["target"].tolist()
                if _predictions is not None and not _predictions.empty
                else []
            )
            + (
                _metrics["target"].tolist()
                if _metrics is not None and not _metrics.empty
                else []
            )
        )
    )
    target = mo.ui.dropdown(
        _targets, value=_targets[0] if _targets else None, label="Target"
    )
    _epochs = (
        sorted(_predictions["epoch"].dropna().unique().astype(int).tolist())
        if _predictions is not None and not _predictions.empty
        else []
    )
    epoch = mo.ui.dropdown(
        _epochs, value=_epochs[-1] if _epochs else None, label="Prediction epoch"
    )
    _splits = (
        sorted(_predictions["split"].unique().tolist())
        if _predictions is not None and not _predictions.empty
        else []
    )
    split = mo.ui.dropdown(
        _splits, value=_splits[0] if _splits else None, label="Prediction split"
    )
    mo.hstack([target, epoch, split])
    return epoch, split, target


@app.cell
def _(frames, mo, plt):
    _view = None
    _losses = frames.get("losses.csv")
    if _losses is not None and not _losses.empty:
        _names = sorted(_losses["loss"].unique())
        _fig, _axes = plt.subplots(
            len(_names),
            1,
            figsize=(11, max(3, 2.6 * len(_names))),
            squeeze=False,
            layout="constrained",
        )
        for _axis, _name in zip(_axes.flat, _names, strict=True):
            for _split, _data in _losses[_losses["loss"] == _name].groupby("split"):
                _data = _data.sort_values(["epoch", "step"])
                _axis.plot(_data["epoch"], _data["value"], label=_split, marker=".")
            _axis.set(
                title=f"Loss: {_name}",
                xlabel="Completed reconstruction epoch",
                ylabel="Reported loss (see loss definition)",
            )
            _axis.legend()
        _view = mo.vstack([mo.md("## Loss history"), _fig])
        plt.close(_fig)
    _view
    return


@app.cell
def _(frames, mo, plt, target):
    _view = None
    _metrics = frames.get("metrics.csv")
    if _metrics is not None and not _metrics.empty and target.value is not None:
        _data = _metrics[_metrics["target"] == target.value]
        _names = sorted(_data["metric"].unique())
        _fig, _axes = plt.subplots(
            len(_names),
            2,
            figsize=(12, max(3, 2.7 * len(_names))),
            squeeze=False,
            layout="constrained",
        )
        for _i, _metric in enumerate(_names):
            _part = _data[_data["metric"] == _metric]
            for (_split, _subset, _unit), _group in _part.groupby(
                ["split", "subset", "unit"]
            ):
                _group = _group.sort_values("epoch")
                _label = f"{_split} / {_subset} [{_unit}]"
                _axes[_i, 0].plot(
                    _group["epoch"], _group["value"], label=_label, marker="."
                )
                _baseline = _group.iloc[0]["value"]
                _axes[_i, 1].plot(
                    _group["epoch"],
                    _group["value"] - _baseline,
                    label=_label,
                    marker=".",
                )
            _axes[_i, 0].set(
                title=f"{target.value}: {_metric}",
                xlabel="Epoch",
                ylabel="Metric value (legend units)",
            )
            _axes[_i, 1].axhline(0, color="gray", linestyle="--")
            _axes[_i, 1].set(
                title="Change from first published evaluation",
                xlabel="Epoch",
                ylabel="Current − first (legend units)",
            )
            _axes[_i, 0].legend(fontsize=7)
            _axes[_i, 1].legend(fontsize=7)
        _view = mo.vstack(
            [
                mo.md("## Metric history and changes"),
                mo.md(
                    "Negative change improves error metrics; positive change improves coverage only when moving toward its nominal level. No universal direction is assumed."
                ),
                _fig,
            ]
        )
        plt.close(_fig)
    _view
    return


@app.cell
def _(epoch, frames, mo, plt, split, target):
    _view = None
    _predictions = frames.get("predictions.csv")
    if (
        _predictions is not None
        and not _predictions.empty
        and epoch.value is not None
        and target.value is not None
    ):
        _data = _predictions[
            (_predictions["epoch"] == epoch.value)
            & (_predictions["target"] == target.value)
            & (_predictions["split"] == split.value)
        ].copy()
        if not _data.empty:
            _units = _data["unit"].unique()
            if len(_units) != 1:
                raise ValueError("One target/split/epoch contains inconsistent units")
            _unit = _units[0]
            _residual = _data["prediction"] - _data["truth"]
            _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
            _axes[0].hexbin(
                _data["truth"], _data["prediction"], gridsize=45, mincnt=1, bins="log"
            )
            _lo = min(_data["truth"].min(), _data["prediction"].min())
            _hi = max(_data["truth"].max(), _data["prediction"].max())
            _axes[0].plot([_lo, _hi], [_lo, _hi], color="gray", linestyle="--")
            _axes[0].set(
                xlabel=f"Reported reference [{_unit}]",
                ylabel=f"Prediction [{_unit}]",
                title="Regression; logarithmic count density",
            )
            _axes[1].hexbin(
                _data["truth"], _residual, gridsize=45, mincnt=1, bins="log"
            )
            _axes[1].axhline(0, color="gray", linestyle="--")
            _axes[1].set(
                xlabel=f"Reported reference [{_unit}]",
                ylabel=f"Prediction − reference [{_unit}]",
                title="Residual versus reference",
            )
            _axes[2].hist(_residual, bins=60)
            _axes[2].set(
                xlabel=f"Prediction − reference [{_unit}]",
                ylabel="Number of stars",
                title=f"Residual distribution · N={len(_data):,}",
            )
            _view = mo.vstack(
                [
                    mo.md(f"## {target.value}: epoch {epoch.value}, {split.value}"),
                    mo.md(
                        "Regression plots show the published monitoring sample; full-cohort metrics may have a larger count. Reference-label agreement is not independent physical accuracy."
                    ),
                    _fig,
                ]
            )
            plt.close(_fig)
    _view
    return


@app.cell
def _(frames, mo, plt):
    _view = None
    _resources = frames.get("resources.csv")
    if _resources is not None and not _resources.empty:
        _fig, _axes = plt.subplots(2, 2, figsize=(12, 6), layout="constrained")
        for _axis, _column, _title, _unit in zip(
            _axes.flat,
            [
                "gpu_utilization_pct",
                "gpu_memory_used_mb",
                "rss_mb",
                "examples_per_second",
            ],
            [
                "GPU utilization",
                "PyTorch peak allocated GPU memory",
                "Process peak resident memory",
                "Training throughput",
            ],
            ["%", "MiB", "MiB", "examples / second"],
            strict=True,
        ):
            if _resources[_column].notna().any():
                _axis.plot(_resources["step"], _resources[_column])
            else:
                _axis.text(
                    0.5,
                    0.5,
                    "Not measured",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=_axis.transAxes,
                )
            _axis.set(title=_title, xlabel="Optimizer step", ylabel=_unit)
        _view = mo.vstack([mo.md("## Resource telemetry"), _fig])
        plt.close(_fig)
    _view
    return


if __name__ == "__main__":
    app.run()
