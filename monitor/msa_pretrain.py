import marimo

__generated_with = "0.1.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import re
    import sys
    import tempfile
    from datetime import UTC, datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from msa_monitor_io import load_jsonl, sync_run_outputs

    return (
        UTC,
        json,
        Path,
        datetime,
        load_jsonl,
        mo,
        np,
        os,
        pd,
        plt,
        re,
        sys,
        sync_run_outputs,
        tempfile,
    )


@app.cell
def _(mo, os, sys):
    run_id = mo.ui.text(
        label="Run ID",
        value=os.environ.get("MSA_MONITOR_RUN_ID", ""),
    )
    output_root = mo.ui.text(
        label="ARC run directory",
        value=os.environ.get("MSA_MONITOR_OUTPUT_ROOT", ""),
    )
    session_name = mo.ui.text(
        label="CANFAR session name",
        value=os.environ.get("MSA_MONITOR_SESSION_NAME", ""),
    )
    refresh = mo.ui.refresh(
        options=["30s", "1m"],
        default_interval="30s",
        label="Auto refresh",
    )
    runtime = mo.md(
        f"**Monitor runtime:** `{sys.executable}` · Python `{sys.version.split()[0]}`"
    )
    mo.vstack(
        [
            mo.md("# MSA pretraining monitor"),
            mo.hstack([run_id, output_root]),
            session_name,
            refresh,
            runtime,
        ]
    )
    return output_root, refresh, run_id, session_name


@app.cell
def _(
    UTC,
    json,
    Path,
    datetime,
    load_jsonl,
    output_root,
    pd,
    refresh,
    re,
    run_id,
    session_name,
    sync_run_outputs,
    tempfile,
):
    _ = refresh.value
    _selected_run_id = run_id.value.strip()
    _selected_root = output_root.value.strip()
    errors = {}
    metrics_records = []
    residual_records = []
    progress_records = []

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", _selected_run_id):
        errors["Configuration"] = (
            "Enter a run ID using letters, numbers, dots, underscores, or hyphens."
        )
    elif not _selected_root:
        errors["Configuration"] = "Enter the run directory on `/arc/projects`."
    else:
        _cache_dir = Path(tempfile.gettempdir()) / "msa-monitor" / _selected_run_id
        _paths, errors = sync_run_outputs(_selected_root, _cache_dir)
        metrics_records = load_jsonl(_paths["metrics.jsonl"])
        residual_records = load_jsonl(_paths["residual_stats.jsonl"])
        progress_records = load_jsonl(_paths["progress.jsonl"])
        latest_path = _paths["residual_latest.json"]
        if latest_path.is_file():
            try:
                latest_residual = json.loads(latest_path.read_text())
                if (
                    latest_residual.get("run_id") == _selected_run_id
                    and latest_residual.get("record_type") == "interval"
                    and (
                        not residual_records
                        or latest_residual.get("timestamp_utc", "")
                        > residual_records[-1].get("timestamp_utc", "")
                    )
                ):
                    residual_records.append(latest_residual)
            except (OSError, json.JSONDecodeError) as exc:
                errors["residual_latest.json"] = str(exc)
        metrics_records = [
            row for row in metrics_records if row.get("run_id") == _selected_run_id
        ]
        residual_records = [
            row for row in residual_records if row.get("run_id") == _selected_run_id
        ]
        progress_records = [
            row for row in progress_records if row.get("run_id") == _selected_run_id
        ]
    metrics = pd.DataFrame(metrics_records)
    residuals = pd.DataFrame(residual_records)
    progress = pd.DataFrame(progress_records)
    for _frame in (progress, metrics, residuals):
        if not _frame.empty and "timestamp_utc" in _frame:
            _frame["timestamp_utc"] = pd.to_datetime(
                _frame["timestamp_utc"], utc=True, errors="coerce"
            )

    _last_times = []
    for _frame in (progress, metrics, residuals):
        if not _frame.empty and "timestamp_utc" in _frame:
            _valid = _frame["timestamp_utc"].dropna()
            if not _valid.empty:
                _last_times.append(_valid.max())
    last_update = max(_last_times) if _last_times else None
    stale_minutes = (
        max(0.0, (datetime.now(UTC) - last_update.to_pydatetime()).total_seconds() / 60)
        if last_update is not None
        else None
    )
    return errors, last_update, metrics, progress, residuals, stale_minutes


@app.cell
def _(errors, last_update, metrics, mo, pd, progress, session_name, stale_minutes):
    _session_text = session_name.value.strip() or "not configured"
    if not progress.empty:
        _latest_progress = progress.iloc[-1]
        _details = []
        for _field, _detail_label in (
            ("key", "shard"),
            ("shard_index", "shard index"),
            ("shard_count", "shards"),
            ("overall_rows_completed", "rows scanned"),
            ("overall_rows_total", "rows to scan"),
            ("rows_seen_total", "training rows seen"),
            ("epoch_rows_seen", "rows in epoch"),
            ("interval_rows", "interval rows"),
            ("train_loss", "interval train loss"),
            ("sampled_val_loss", "sample validation loss"),
            ("rows_per_second", "rows/s"),
        ):
            _value = _latest_progress.get(_field)
            if _value is not None and pd.notna(_value):
                _details.append(f"{_detail_label}: {_value}")
        _phase_text = f"`{_latest_progress.get('stage', 'unknown')}`"
        if _details:
            _phase_text += " · " + " · ".join(_details)
    else:
        _phase_text = "waiting for progress records"

    _last_text = last_update.isoformat() if last_update is not None else "no output yet"
    _stale_text = f"{stale_minutes:.1f} min ago" if stale_minutes is not None else "—"
    if stale_minutes is not None and stale_minutes > 45:
        _health = "⚠️ No persisted update for over 45 minutes"
    elif errors:
        _health = "⚠️ Some files could not be fetched; cached snapshots remain visible"
    elif last_update is not None:
        _health = "✅ Monitoring data is updating"
    else:
        _health = "Waiting for the first scan or training event"

    if not metrics.empty:
        _latest_metric = metrics.iloc[-1]
        _epoch_text = (
            f"{int(_latest_metric['epoch'])}/{int(_latest_metric['total_epochs'])}"
        )
        _elapsed_hours = (
            metrics["wall_time_s"].sum() / 3600 if "wall_time_s" in metrics else 0
        )
        _average_epoch = (
            metrics["wall_time_s"].mean() if "wall_time_s" in metrics else 0
        )
        _remaining = max(
            0, int(_latest_metric["total_epochs"]) - int(_latest_metric["epoch"])
        )
        _eta_text = (
            f"{_remaining * _average_epoch / 3600:.1f} h"
            if _average_epoch
            else "estimating"
        )
        _best_val = (
            metrics["val_loss"].dropna().min()
            if "val_loss" in metrics
            else float("nan")
        )
        _best_text = f"{_best_val:.6g}" if pd.notna(_best_val) else "—"
    else:
        _epoch_text, _elapsed_hours, _eta_text, _best_text = "—", 0.0, "—", "—"

    _status = mo.md(
        f"## Run health\n\n{_health}\n\n"
        f"**CANFAR session name:** `{_session_text}`  \n"
        f"**Phase:** {_phase_text}  \n"
        f"**Epoch:** {_epoch_text} · **Elapsed:** {_elapsed_hours:.1f} h · "
        f"**ETA:** {_eta_text} · **Best validation loss:** {_best_text}  \n"
        f"**Last update:** {_last_text} ({_stale_text})"
    )
    _fetch_details = None
    if errors:
        _error_lines = []
        for _name, _message in errors.items():
            _single_line = " ".join(str(_message).split()).replace("`", "'")
            if len(_single_line) > 240:
                _single_line = _single_line[:237] + "..."
            _error_lines.append(f"- **{_name}:** `{_single_line}`")
        _fetch_details = mo.md("### Fetch details\n\n" + "\n".join(_error_lines))
    _health_content = [_status]
    if _fetch_details is not None:
        _health_content.append(_fetch_details)
    mo.vstack(_health_content)
    return


@app.cell
def _(metrics, mo, pd, plt):
    _content = []
    if metrics.empty:
        _content.append(
            mo.md("## Training curves\nWaiting for the first training metric.")
        )
    else:
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
        _interval_mask = metrics.get(
            "record_type", pd.Series("epoch", index=metrics.index)
        ).eq("interval")
        _intervals = metrics[_interval_mask]
        _epochs = metrics[~_interval_mask]
        if not _intervals.empty and "rows_seen_total" in _intervals:
            _x = _intervals["rows_seen_total"] / 1e6
            _axes[0].plot(_x, _intervals["train_loss"], ".-", label="train interval")
            if (
                "sampled_val_loss" in _intervals
                and _intervals["sampled_val_loss"].notna().any()
            ):
                _axes[0].plot(
                    _x,
                    _intervals["sampled_val_loss"],
                    ".-",
                    label="validation sample",
                )
            if not _epochs.empty and "rows_seen_total" in _epochs:
                _axes[0].scatter(
                    _epochs["rows_seen_total"] / 1e6,
                    _epochs["train_loss"],
                    marker="s",
                    label="epoch train",
                )
                if "val_loss" in _epochs and _epochs["val_loss"].notna().any():
                    _axes[0].scatter(
                        _epochs["rows_seen_total"] / 1e6,
                        _epochs["val_loss"],
                        marker="s",
                        label="full validation",
                    )
            _axes[0].set_xlabel("Training rows seen (millions)")
        else:
            _axes[0].plot(
                _epochs["epoch"], _epochs["train_loss"], "o-", label="train", ms=3
            )
            if "val_loss" in _epochs and _epochs["val_loss"].notna().any():
                _axes[0].plot(
                    _epochs["epoch"],
                    _epochs["val_loss"],
                    "s-",
                    label="validation",
                    ms=3,
                )
            _axes[0].set_xlabel("Epoch")
        _axes[0].set(ylabel="Masked reconstruction loss", title="Loss")
        _axes[0].legend()
        if "lr" in _epochs:
            _axes[1].plot(_epochs["epoch"], _epochs["lr"], color="tab:orange")
        _axes[1].set(xlabel="Epoch", ylabel="Learning rate", title="Schedule")
        _fig.tight_layout()
        _content.extend([mo.md("## Training curves"), _fig])
    mo.vstack(_content)
    return


@app.cell
def _(metrics, mo, pd, plt):
    _resource_columns = {
        "current_host_rss_bytes": "host RSS",
        "peak_host_rss_bytes": "peak host RSS",
        "current_gpu_allocated_bytes": "GPU allocated",
        "peak_gpu_allocated_bytes": "peak GPU allocated",
        "peak_gpu_reserved_bytes": "peak GPU reserved",
    }
    _available = [name for name in _resource_columns if name in metrics]
    if metrics.empty or not _available:
        _content = [mo.md("## Resources\nWaiting for epoch resource metrics.")]
    else:
        _interval_mask = metrics.get(
            "record_type", pd.Series("epoch", index=metrics.index)
        ).eq("interval")
        _intervals = metrics[_interval_mask]
        _epochs = metrics[~_interval_mask]
        _rows_axis = not _intervals.empty and "rows_seen_total" in _intervals
        _fig, _ax = plt.subplots(figsize=(10, 4))
        _resource_data = _intervals if _rows_axis else _epochs
        _x = (
            _resource_data["rows_seen_total"] / 1e6
            if _rows_axis
            else _resource_data["epoch"]
        )
        for _resource_name in _available:
            _ax.plot(
                _x,
                _resource_data[_resource_name] / 1024**3,
                "o-",
                ms=3,
                label=_resource_columns[_resource_name],
            )
        _ax.set(
            xlabel="Training rows seen (millions)" if _rows_axis else "Epoch",
            ylabel="GiB",
            title="Host and GPU memory",
        )
        _ax.legend(ncol=2)
        _fig.tight_layout()
        _content = [mo.md("## Resources"), _fig]
    mo.vstack(_content)
    return


@app.cell
def _(metrics, mo, pd, plt, progress):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    if not metrics.empty:
        _residual_x = (
            metrics["rows_seen_total"] / 1e6
            if "rows_seen_total" in metrics and metrics["rows_seen_total"].notna().any()
            else metrics["epoch"]
        )
        for _column, _metric_label in (
            ("residual_xp_mae", "XP MAE"),
            ("residual_xp_p84", "XP p84"),
            ("residual_photo_mae", "Photometry MAE"),
            ("residual_overall_mae", "Overall MAE"),
        ):
            if _column in metrics:
                _axes[0].plot(
                    _residual_x, metrics[_column], "o-", ms=3, label=_metric_label
                )
        _axes[0].set(
            xlabel="Training rows seen (millions)"
            if "rows_seen_total" in metrics and metrics["rows_seen_total"].notna().any()
            else "Epoch",
            ylabel="Absolute residual",
            title="Validation residuals",
        )
        if _axes[0].lines:
            _axes[0].legend(fontsize="small")
    if not progress.empty and "stage" in progress:
        _train_rows = progress[progress["stage"] == "train_shard_finished"].copy()
        if not _train_rows.empty and "timestamp_utc" in _train_rows:
            _axes[1].plot(
                _train_rows["timestamp_utc"], _train_rows["rows_per_second"], ".-"
            )
            _axes[1].tick_params(axis="x", labelrotation=30)
    _axes[1].set(xlabel="Time", ylabel="Rows / s", title="Training shard throughput")
    _fig.tight_layout()
    mo.vstack([mo.md("## Residuals and throughput"), _fig])
    return


@app.cell
def _(mo, np, pd, plt, re, residuals):
    _content = []
    if residuals.empty or "feature_names" not in residuals:
        _content.append(
            mo.md(
                "## Per-feature validation QA\nWaiting for the first residual sample."
            )
        )
    else:
        _latest = residuals.iloc[-1]
        _names = _latest["feature_names"]
        _mae = _latest.get("feature_mae", [])
        _p84 = _latest.get("feature_p84", [])
        _valid_fraction = _latest.get("feature_valid_fraction", [])
        if not _names or not _mae:
            _content.append(
                mo.md(
                    "## Per-feature validation QA\nNo per-feature summaries in the latest residual record."
                )
            )
        else:
            _row_text = (
                f" · {int(_latest['rows_seen_total']):,} training rows seen"
                if pd.notna(_latest.get("rows_seen_total"))
                else ""
            )
            _values = np.asarray(
                [np.nan if value is None else value for value in _mae], dtype=float
            )
            _tails = np.asarray(
                [
                    np.nan if i >= len(_p84) or _p84[i] is None else _p84[i]
                    for i in range(len(_names))
                ],
                dtype=float,
            )
            _fractions = np.asarray(
                [
                    np.nan if i >= len(_valid_fraction) else _valid_fraction[i]
                    for i in range(len(_names))
                ],
                dtype=float,
            )
            _xp_indices = [
                i
                for i, _name in enumerate(_names)
                if re.fullmatch(r"(?:bp|rp)_\d+", _name)
            ]
            _xp_set = set(_xp_indices)
            _other_indices = [i for i in range(len(_names)) if i not in _xp_set]

            _fig, _axes = plt.subplots(2, 1, figsize=(12, 7))
            if _xp_indices:
                _labels = [_names[i] for i in _xp_indices]
                _axes[0].plot(
                    range(len(_xp_indices)), _values[_xp_indices], label="MAE"
                )
                _axes[0].plot(range(len(_xp_indices)), _tails[_xp_indices], label="p84")
                _axes[0].set_xticks(
                    range(len(_xp_indices)), _labels, rotation=90, fontsize=6
                )
                _axes[0].set(
                    title="XP residual by coefficient", ylabel="Absolute residual"
                )
                _axes[0].legend()
            if _other_indices:
                _ranked = sorted(
                    _other_indices,
                    key=lambda i: _values[i] if np.isfinite(_values[i]) else -1,
                    reverse=True,
                )[:20]
                _ranked.reverse()
                _axes[1].barh(
                    [_names[i] for i in _ranked], _values[_ranked], label="MAE"
                )
                _axes[1].set(title="Largest non-XP feature residuals", xlabel="MAE")
            _fig.tight_layout()
            _content.extend(
                [
                    mo.md(
                        f"## Per-feature validation QA · epoch {int(_latest['epoch'])}"
                        f"{_row_text} · "
                        f"{int(_latest.get('sampled_rows', 0)):,} sampled rows across "
                        f"{int(_latest.get('sampled_validation_shards', 0))} validation shards"
                    ),
                    _fig,
                ]
            )
    mo.vstack(_content)
    return


if __name__ == "__main__":
    app.run()
