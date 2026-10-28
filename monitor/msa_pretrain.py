import marimo

__generated_with = "0.1.0"
app = marimo.App(width="full")


@app.cell
def _():
    import hashlib
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
    from msa_monitor_io import (
        experiment_comparison,
        load_jsonl,
        load_snapshot,
        sync_run_outputs,
    )

    return (
        UTC,
        experiment_comparison,
        hashlib,
        json,
        load_snapshot,
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
    monitor_mode = mo.ui.dropdown(
        ["Training run", "Experiment suite"],
        value=os.environ.get("MSA_MONITOR_MODE", "Experiment suite"),
        label="View",
    )
    run_id = mo.ui.text(
        label="Run ID",
        value=os.environ.get("MSA_MONITOR_RUN_ID", ""),
    )
    output_root = mo.ui.text(
        label="ARC run / experiment suite directory",
        value=os.environ.get(
            "MSA_MONITOR_OUTPUT_ROOT",
            "/arc/projects/k-pop/msa_runs/experiments-e21-full1m-20261001T0522Z",
        ),
    )
    session_name = mo.ui.text(
        label="CANFAR session name",
        value=os.environ.get("MSA_MONITOR_SESSION_NAME", ""),
    )
    refresh = mo.ui.refresh(
        options=["5m", "10m"],
        default_interval="5m",
        label="Auto refresh",
    )
    runtime = mo.md(
        f"**Monitor runtime:** `{sys.executable}` · Python `{sys.version.split()[0]}`"
    )
    mo.vstack(
        [
            mo.md("# MSA pretraining monitor"),
            monitor_mode,
            mo.hstack([run_id, output_root]),
            session_name,
            refresh,
            runtime,
        ]
    )
    return monitor_mode, output_root, refresh, run_id, session_name


@app.cell
def _(
    UTC,
    json,
    Path,
    datetime,
    load_jsonl,
    monitor_mode,
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

    if monitor_mode.value == "Experiment suite":
        pass
    elif _selected_run_id in (".", "..") or not re.fullmatch(
        r"[A-Za-z0-9_.-]+", _selected_run_id
    ):
        errors["Configuration"] = (
            "Enter a run ID using letters, numbers, dots, underscores, or hyphens."
        )
    elif not _selected_root:
        errors["Configuration"] = "Enter the run directory on `/arc/projects`."
    else:
        _cache_dir = Path(tempfile.gettempdir()) / "msa-monitor" / _selected_run_id
        try:
            _paths, errors = sync_run_outputs(_selected_root, _cache_dir)
            metrics_records = load_jsonl(_paths["metrics.jsonl"])
            residual_records = load_jsonl(_paths["residual_stats.jsonl"])
            progress_records = load_jsonl(_paths["progress.jsonl"])
        except (ValueError, OSError) as exc:
            errors["Read"] = str(exc)
            _paths = {"residual_latest.json": _cache_dir / "residual_latest.json"}
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
def _(
    errors,
    last_update,
    metrics,
    mo,
    monitor_mode,
    pd,
    progress,
    session_name,
    stale_minutes,
):
    mo.stop(monitor_mode.value == "Experiment suite")
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
def _(metrics, mo, monitor_mode, pd, plt):
    mo.stop(monitor_mode.value == "Experiment suite")
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
def _(metrics, mo, monitor_mode, pd, plt):
    mo.stop(monitor_mode.value == "Experiment suite")
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
def _(metrics, mo, monitor_mode, pd, plt, progress):
    mo.stop(monitor_mode.value == "Experiment suite")
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
def _(mo, monitor_mode, np, pd, plt, re, residuals):
    mo.stop(monitor_mode.value == "Experiment suite")
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
                        f"## Per-feature validation QA · epoch {_latest.get('epoch', '—')}"
                        f"{_row_text} · "
                        f"{int(_latest.get('sampled_rows', 0)):,} sampled rows across "
                        f"{int(_latest.get('sampled_validation_shards', 0))} validation shards"
                    ),
                    _fig,
                ]
            )
    mo.vstack(_content)
    return


@app.cell
def _(
    UTC,
    Path,
    datetime,
    experiment_comparison,
    hashlib,
    load_jsonl,
    load_snapshot,
    mo,
    monitor_mode,
    output_root,
    refresh,
    re,
    sync_run_outputs,
    tempfile,
):
    mo.stop(monitor_mode.value != "Experiment suite")
    _ = refresh.value
    suite_manifest = {}
    suite_records = {}
    suite_snapshots = {}
    suite_errors = {}
    suite_files = []
    _suite_root = output_root.value.strip()
    if not _suite_root:
        suite_errors["Configuration"] = (
            "Enter the experiment suite directory on /arc/projects."
        )
    else:
        _cache = (
            Path(tempfile.gettempdir())
            / "msa-monitor"
            / hashlib.sha256(_suite_root.encode()).hexdigest()[:16]
        )
        try:
            _paths, _errors = sync_run_outputs(
                _suite_root, _cache, filenames=("manifest.json",), subdir=""
            )
            suite_errors.update(_errors)
            suite_manifest = load_snapshot(_paths["manifest.json"])
            for _entry in suite_manifest.get("arms", []):
                _arm = _entry if isinstance(_entry, dict) else {"arm_id": _entry}
                _name = _arm.get("arm_id", _arm.get("arm", _arm.get("name", "")))
                if not re.fullmatch(r"[A-Za-z0-9_-]+", _name):
                    suite_errors["Manifest"] = f"Invalid experiment arm name: {_name!r}"
                    continue
                _arm_paths, _errors = sync_run_outputs(
                    _suite_root,
                    _cache,
                    filenames=(
                        f"{_name}/metrics.jsonl",
                        f"{_name}/residual_latest.json",
                    ),
                    subdir="",
                )
                suite_errors.update(_errors)
                _paths.update(_arm_paths)
                try:
                    suite_records[_name] = load_jsonl(
                        _arm_paths[f"{_name}/metrics.jsonl"]
                    )
                    suite_snapshots[_name] = load_snapshot(
                        _arm_paths[f"{_name}/residual_latest.json"]
                    )
                except (ValueError, OSError) as exc:
                    suite_errors[_name] = str(exc)
            for _name, _path in _paths.items():
                suite_files.append(
                    {
                        "file": _name,
                        "snapshot fetched UTC": datetime.fromtimestamp(
                            _path.stat().st_mtime, UTC
                        ).isoformat()
                        if _path.is_file()
                        else None,
                        "fetch": "failed / cached"
                        if _name in suite_errors
                        else "available"
                        if _path.is_file()
                        else "pending",
                    }
                )
        except (ValueError, OSError) as exc:
            suite_errors["Read"] = str(exc)
    suite_comparison = experiment_comparison(
        suite_manifest, suite_records, suite_snapshots
    )
    return (
        suite_comparison,
        suite_errors,
        suite_files,
        suite_manifest,
        suite_records,
        suite_snapshots,
    )


@app.cell
def _(mo):
    get_arm_choice, set_arm_choice = mo.state("")
    get_regime_choice, set_regime_choice = mo.state("common")
    return get_arm_choice, get_regime_choice, set_arm_choice, set_regime_choice


@app.cell
def _(
    UTC,
    datetime,
    json,
    mo,
    pd,
    suite_comparison,
    suite_errors,
    suite_files,
    suite_manifest,
    suite_records,
    suite_snapshots,
):
    _items = [mo.md("## What this comparison tells us")]
    _details = []
    if not suite_manifest:
        _items.append(mo.md("Waiting for the experiment manifest and saved results."))
    else:
        _sample = suite_manifest.get("sample", {})
        _expected_id = suite_manifest.get("validation_id")
        _expected_rows = _sample.get("validation_rows")
        _arm_names = [
            entry.get("arm_id", entry.get("arm", entry.get("name", "")))
            if isinstance(entry, dict)
            else entry
            for entry in suite_manifest.get("arms", [])
        ]
        _valid_full_summaries = all(
            (qa := suite_snapshots.get(name, {})).get("validation_id") == _expected_id
            and qa.get("validation_rows_total") == _expected_rows
            and qa.get("regimes", {}).get("common", {}).get("overall", {}).get("finite")
            is True
            for name in _arm_names
        )
        _status = suite_manifest.get("status", "unknown")
        _readout = None
        if (
            _status == "complete"
            and _valid_full_summaries
            and {"legacy_control", "stable_embedding"}.issubset(_arm_names)
        ):
            _control = suite_snapshots["legacy_control"]["regimes"]["common"]["overall"]
            _stable = suite_snapshots["stable_embedding"]["regimes"]["common"][
                "overall"
            ]
            _control_mae = _control.get("mae")
            _stable_mae = _stable.get("mae")
            _control_skill = _control.get("group_skill_median")
            _stable_skill = _stable.get("group_skill_median")
            _improvement = (
                100 * (_control_mae - _stable_mae) / _control_mae
                if _control_mae is not None
                and _control_mae != 0
                and _stable_mae is not None
                else None
            )
            _skill_gain = (
                100 * (_stable_skill - _control_skill)
                if _stable_skill is not None and _control_skill is not None
                else None
            )
            _control_metrics = suite_records.get("legacy_control", [])
            _stable_metrics = suite_records.get("stable_embedding", [])
            _control_final = _control_metrics[-1] if _control_metrics else {}
            _stable_final = _stable_metrics[-1] if _stable_metrics else {}
            _stable_previous = max(
                (
                    row
                    for row in _stable_metrics[:-1]
                    if isinstance(row.get("group_skill_median"), int | float)
                ),
                key=lambda row: row.get("rows_seen_total", 0),
                default=None,
            )
            _plateau = ""
            if _stable_previous is not None and _stable_skill is not None:
                _previous_skill = _stable_previous["group_skill_median"]
                _skill_change = 100 * (_stable_skill - _previous_skill)
                _previous_views = _stable_previous.get("rows_seen_total", 0) / 1_000_000
                _final_views = _stable_final.get("rows_seen_total", 0) / 1_000_000
                if abs(_skill_change) < 0.1:
                    _plateau = (
                        f" Its equal-group score was essentially flat from "
                        f"{_previous_views:g}M to {_final_views:g}M presentations."
                    )
                else:
                    _direction = "rose" if _skill_change > 0 else "fell"
                    _plateau = (
                        f" From {_previous_views:g}M to {_final_views:g}M presentations, "
                        f"equal-group skill {_direction} by {abs(_skill_change):.1f} percentage points."
                    )
            _frequency_before = _control_final.get("frequency_gradient_fraction")
            _frequency_after = _stable_final.get("frequency_gradient_fraction")
            _clip_before = _control_final.get("clipping_rate")
            _clip_after = _stable_final.get("clipping_rate")
            _improvement_text = (
                f"{_improvement:.1f}%" if _improvement is not None else "unavailable"
            )
            _skill_gain_text = (
                f"{_skill_gain:+.1f} percentage points"
                if _skill_gain is not None
                else "unavailable"
            )
            _stability = ""
            if all(
                value is not None
                for value in (
                    _frequency_before,
                    _frequency_after,
                    _clip_before,
                    _clip_after,
                )
            ):
                _stability = (
                    f" The frequency-gradient share fell from "
                    f"{100 * _frequency_before:.0f}% to {100 * _frequency_after:.1f}%, "
                    f"and clipping from {100 * _clip_before:.0f}% to "
                    f"{100 * _clip_after:.0f}%."
                )
            _readout = (
                "### Promising stability fix; modest quality gain, then a plateau\n\n"
                f"On the shared full validation set, stable embedding lowered common MAE by "
                f"{_improvement_text} and raised equal-group skill by {_skill_gain_text} "
                "over control."
                f"{_stability}{_plateau}"
            )
        elif _status != "complete":
            _readout = (
                f"### Comparison is {_status}\n\n"
                "Wait for every arm to finish the same presentation budget before choosing a winner."
            )
        elif not _valid_full_summaries:
            _readout = (
                "### Full-validation evidence is incomplete\n\n"
                "The saved summaries do not yet agree with the manifest's fixed validation ID "
                "and row count."
            )
        else:
            _readout = (
                "### The suite finished\n\n"
                "Use the plots below to compare the completed arms."
            )
        _items.append(mo.md(_readout))

        _next = [
            "### What to do next",
            "**Do not scale this straight to full pretraining yet.** First run a 2×2 comparison "
            "of group-balanced loss and lower mask ratios while holding stable embedding, "
            "checkpoint, data rows, validation IDs and budgets fixed. The current `lower_mask` "
            "arm changes all three together, so it cannot identify which change helped.",
        ]
        _latest = [records[-1] for records in suite_records.values() if records]
        _at_memory_limit = any(
            isinstance(row.get("cgroup_memory_limit_bytes"), int | float)
            and row["cgroup_memory_limit_bytes"] > 0
            and isinstance(row.get("cgroup_memory_peak_bytes"), int | float)
            and row["cgroup_memory_peak_bytes"] >= row["cgroup_memory_limit_bytes"]
            for row in _latest
        )
        if _at_memory_limit:
            _next.append(
                "Before that run, reduce or profile full-validation peak memory: every arm "
                "reached the reported cgroup limit. No OOM was recorded, but the run had no "
                "measured headroom."
            )
        _items.append(mo.md("\n\n".join(_next)))
        _train_rows = _sample.get("train_rows")
        _presentation_budget = suite_manifest.get("presentations_per_arm")
        _items.append(
            mo.md(
                "**Scope:** "
                f"{f'{_train_rows:,}' if isinstance(_train_rows, int) else '—'} bounded "
                "training-prefix rows · "
                f"{f'{_expected_rows:,}' if isinstance(_expected_rows, int) else '—'} fixed "
                "validation stars · "
                f"{f'{_presentation_budget:,}' if isinstance(_presentation_budget, int) else '—'} "
                "presentations per arm. Validation JSON metrics and plots summarize all fixed "
                "validation rows; residual-array and latent diagnostics use the labeled 100K "
                "diagnostic sample. Training rows are sequential per-shard prefixes, not a "
                "population-random sample. Scores are in saved RobustScaler units; uncertainty "
                "units are unverified."
            )
        )

        _training_config = suite_manifest.get("config", {}).get("training", {})
        _cache_cap = suite_manifest.get("cache_limit_bytes")
        if _cache_cap is None:
            _cache_cap = _sample.get(
                "cache_cap_bytes", _training_config.get("io_cache_max_bytes")
            )

        def _gib(value):
            if isinstance(value, int | float) and not isinstance(value, bool):
                return round(value / (1024**3), 2)
            return None

        _sample_summary = {
            "training prefix rows": _sample.get("train_rows"),
            "fixed validation rows": _sample.get("validation_rows"),
            "presentations per arm": suite_manifest.get("presentations_per_arm"),
            "cache used (GiB)": _gib(_sample.get("cache_bytes")),
            "cache limit (GiB)": _gib(_cache_cap),
            "sampling policy": _sample.get("policy", "pending"),
        }
        _details.append(mo.ui.table([_sample_summary], selection=None))
        _availability_rows = []
        _feature_availability = []
        for _label, _key in (
            ("training", "train_availability"),
            ("validation", "validation_availability"),
        ):
            _availability = _sample.get(_key, {})
            if not isinstance(_availability, dict):
                continue
            if any(
                name in _availability
                for name in (
                    "xp_all_finite_rows",
                    "xp_none_finite_rows",
                    "xp_partial_finite_rows",
                )
            ):
                _availability_rows.append(
                    {
                        "sample": _label,
                        "XP features": _availability.get("xp_feature_count"),
                        "XP all finite rows": _availability.get("xp_all_finite_rows"),
                        "XP none finite rows": _availability.get("xp_none_finite_rows"),
                        "XP partial finite rows": _availability.get(
                            "xp_partial_finite_rows"
                        ),
                    }
                )
            _names = _availability.get("feature_names", [])
            _counts = _availability.get("finite_feature_counts", [])
            if isinstance(_names, list) and isinstance(_counts, list):
                _feature_availability.extend(
                    {
                        "sample": _label,
                        "feature": _name,
                        "finite rows": _count,
                    }
                    for _name, _count in zip(_names, _counts)
                )
        if _availability_rows:
            _details.append(mo.md("#### Natural XP and finite input availability"))
            _details.append(
                mo.ui.table(_availability_rows, selection=None, pagination=False)
            )
        if _feature_availability:
            _details.append(
                mo.accordion(
                    {
                        "Per-feature finite input counts": mo.ui.table(
                            pd.DataFrame(_feature_availability), selection=None
                        )
                    }
                )
            )
        if suite_comparison:
            _comparison = pd.DataFrame(suite_comparison)
            for _column in _comparison.select_dtypes(include="number"):
                _comparison[_column] = _comparison[_column].round(3)
            _details.append(mo.ui.table(_comparison, selection=None, pagination=False))
        _provenance = json.dumps(suite_manifest, indent=2, sort_keys=True)
        _details.append(
            mo.accordion(
                {
                    "Configuration and code provenance": mo.md(
                        f"```json\n{_provenance}\n```"
                    )
                }
            )
        )
    _times = pd.to_datetime(
        [row.get("updated") for row in suite_comparison], utc=True, errors="coerce"
    )
    _times = _times.dropna()
    _age = (
        (datetime.now(UTC) - _times.max().to_pydatetime()).total_seconds() / 60
        if len(_times)
        else None
    )
    if _age is not None and _age > 45:
        _details.append(mo.md("⚠️ No new persisted metric for over 45 minutes."))
    if suite_errors:
        _items.append(mo.md("⚠️ Fetch/read errors; cached data may be stale."))
        _details.append(
            mo.ui.table(
                [{"file": key, "error": value} for key, value in suite_errors.items()],
                selection=None,
            )
        )
    if suite_files:
        _details.append(
            mo.accordion(
                {
                    "Fetched-file freshness": mo.ui.table(
                        suite_files, selection=None, pagination=False
                    )
                }
            )
        )
    _items.append(mo.accordion({"Setup and exact values": mo.vstack(_details)}))
    mo.vstack(_items)
    return


@app.cell
def _(mo, np, pd, plt, suite_manifest, suite_records, suite_snapshots):
    _colors = {
        "legacy_control": "#777777",
        "stable_embedding": "#0072B2",
        "lower_mask": "#D55E00",
    }
    _labels = {
        "legacy_control": "Control",
        "stable_embedding": "Stable embedding",
        "lower_mask": "Stable + balanced loss + lower mask",
    }
    _fig, _axes = plt.subplots(3, 2, figsize=(13, 13), constrained_layout=True)
    _axes = _axes.ravel()
    _plot_count = 0
    _latest_rows = []

    # Compare only observed endpoints; sparse checkpoints do not support a learning curve.
    _change_ax = _axes[0]
    _change_points = []
    _tick_positions, _tick_labels = [], []
    _y = 0
    _change_arm_labels = {
        "legacy_control": "Control",
        "stable_embedding": "Stable embedding",
        "lower_mask": "Combined arm",
    }
    _arm_order = ["legacy_control", "stable_embedding", "lower_mask"]
    _arm_order.extend(_name for _name in suite_records if _name not in _arm_order)
    for _name in _arm_order:
        _records = suite_records.get(_name, [])
        if _records:
            _latest_rows.append({"arm": _name, **_records[-1]})
        _ordered = sorted(
            (row for row in _records if isinstance(row.get("rows_seen_total"), int)),
            key=lambda row: row["rows_seen_total"],
        )
        _first_train = next(
            (
                row["rows_seen_total"]
                for row in _ordered
                if isinstance(row.get("train_loss"), int | float)
                and np.isfinite(row["train_loss"])
            ),
            None,
        )
        for _field, _label, _marker in (
            ("train_loss", "training loss", "o"),
            ("common_scaled_mae", "validation MAE", "s"),
        ):
            _points = [
                row
                for row in _ordered
                if _first_train is not None
                and row["rows_seen_total"] >= _first_train
                and isinstance(row.get(_field), int | float)
                and np.isfinite(row[_field])
            ]
            if len(_points) >= 2 and _points[0][_field] != 0:
                _change = 100 * (_points[-1][_field] / _points[0][_field] - 1)
                _change_points.append((_change, _y, _name, _marker))
                _tick_positions.append(_y)
                _tick_labels.append(
                    f"{_change_arm_labels.get(_name, _name)} · {_label}"
                )
                _y += 1
        if _ordered and _first_train is None:
            _tick_positions.append(_y)
            _tick_labels.append(
                f"{_change_arm_labels.get(_name, _name)} · awaiting checkpoints"
            )
            _y += 1
    for _change, _position, _name, _marker in _change_points:
        _change_ax.scatter(
            _change,
            _position,
            marker=_marker,
            s=48,
            color=_colors.get(_name, "#9467bd"),
            zorder=3,
        )
        _change_ax.annotate(
            f"{_change:+.1f}%" if abs(_change) >= 0.05 else "≈0%",
            (_change, _position),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    _change_ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    _change_ax.set_yticks(_tick_positions, _tick_labels)
    _change_ax.invert_yaxis()
    _change_ax.set_xlabel("Change from first to last checkpoint (%)")
    _change_ax.set_title("Validation residuals barely move from 10M to 20M", loc="left")
    if _change_points:
        _changes = [point[0] for point in _change_points]
        _change_ax.set_xlim(
            min(-1.0, min(_changes) - 0.8), max(1.0, max(_changes) + 1.0)
        )
        _plot_count += len(_change_points)
    else:
        _change_ax.text(0.5, 0.5, "Need two checkpoints to show change", ha="center")
    # Relative group MAE makes small photometry values comparable to XP/astrometry.
    _group_ax = _axes[1]
    _groups = (("xp", "Gaia XP"), ("photo", "Photometry"), ("astro", "Astrometry"))
    _control_blocks = (
        suite_snapshots.get("legacy_control", {})
        .get("regimes", {})
        .get("common", {})
        .get("blocks", {})
    )
    _group_offsets = {"stable_embedding": -0.12, "lower_mask": 0.12}
    for _arm in ("stable_embedding", "lower_mask"):
        _blocks = (
            suite_snapshots.get(_arm, {})
            .get("regimes", {})
            .get("common", {})
            .get("blocks", {})
        )
        for _index, (_key, _label) in enumerate(_groups):
            _base = _control_blocks.get(_key, {}).get("mae")
            _value = _blocks.get(_key, {}).get("mae")
            if _base and _value is not None:
                _gain = 100 * (_base - _value) / _base
                _group_ax.scatter(
                    _gain,
                    _index + _group_offsets[_arm],
                    color=_colors[_arm],
                    s=45,
                    label=_labels[_arm] if _index == 0 else None,
                    zorder=3,
                )
                _group_ax.annotate(
                    f"{_gain:+.1f}%",
                    (_gain, _index + _group_offsets[_arm]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                    color=_colors[_arm],
                )
                _plot_count += 1
    _group_ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
    _group_ax.set_yticks(range(len(_groups)), [label for _, label in _groups])
    _group_ax.invert_yaxis()
    _group_ax.set_xlabel("MAE reduction vs control (%) · right is better")
    _group_ax.set_title(
        "Stable helps each group; combined arm shifts the trade-off", loc="left"
    )
    _group_ax.legend(frameon=False, fontsize=8, loc="lower right")

    def _find_feature(qa, regime, feature_name):
        _features = qa.get("regimes", {}).get(regime, {}).get("features", [])
        if isinstance(_features, dict):
            return _features.get(feature_name)
        return next(
            (
                row
                for row in _features
                if str(row.get("name", "")).casefold() == feature_name.casefold()
            ),
            None,
        )

    _regimes = ("common", "xp_on", "xp_off")
    _regime_labels = ("Common", "XP visible*", "XP hidden*")
    _parallax_axes = (_axes[2], _axes[3])
    for _ax, _stat, _title in zip(
        _parallax_axes,
        ("mae", "bias"),
        (
            "Parallax error is lower with XP visible",
            "Parallax bias moves toward zero with XP visible",
        ),
        strict=True,
    ):
        for _arm, _qa in suite_snapshots.items():
            _values = [
                (_find_feature(_qa, regime, "PARALLAX") or {}).get(_stat)
                for regime in _regimes
            ]
            if any(value is not None for value in _values):
                _ax.plot(
                    range(len(_regimes)),
                    [float("nan") if value is None else value for value in _values],
                    marker="o",
                    linewidth=2,
                    color=_colors.get(_arm, "#9467bd"),
                    label=_labels.get(_arm, _arm),
                )
                _plot_count += 1
        if _stat == "bias":
            _ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
        _ax.set_xticks(range(len(_regimes)), _regime_labels)
        _ax.set_ylabel("RobustScaler units")
        _ax.set_title(_title, loc="left")
        _ax.legend(frameon=False, fontsize=8)
    _axes[3].text(
        0.02,
        0.02,
        "*XP visibility is changed artificially for this diagnostic.",
        transform=_axes[3].transAxes,
        fontsize=8,
    )

    _health_ax = _axes[4]
    _health_arms = [
        row
        for row in _latest_rows
        if row.get("frequency_gradient_fraction") is not None
        and row.get("clipping_rate") is not None
    ]
    _x = range(len(_health_arms))
    _width = 0.34
    for _offset, _key, _label in (
        (-_width / 2, "frequency_gradient_fraction", "Frequency-gradient share"),
        (_width / 2, "clipping_rate", "Steps clipped"),
    ):
        _bars = _health_ax.bar(
            [position + _offset for position in _x],
            [100 * row[_key] for row in _health_arms],
            _width,
            label=_label,
            color="#4C78A8" if _key == "frequency_gradient_fraction" else "#F2CF5B",
        )
        for _bar in _bars:
            _health_ax.annotate(
                f"{_bar.get_height():.0f}%",
                (_bar.get_x() + _bar.get_width() / 2, _bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    _health_ax.set_xticks(
        list(_x),
        [_labels.get(row["arm"], row["arm"]) for row in _health_arms],
        rotation=15,
        ha="right",
    )
    _health_ax.set_ylim(0, 112)
    _health_ax.set_ylabel("Optimizer steps (%)")
    _health_ax.set_title("Stable embeddings reduced gradient imbalance", loc="left")
    _health_ax.legend(frameon=False, fontsize=8)

    _memory_ax = _axes[5]
    _memory_rows = [
        row
        for row in _latest_rows
        if row.get("cgroup_memory_peak_bytes") is not None
        and row.get("cgroup_memory_limit_bytes")
    ]
    if _memory_rows:
        _peaks = [row["cgroup_memory_peak_bytes"] / 1024**3 for row in _memory_rows]
        _limits = [row["cgroup_memory_limit_bytes"] / 1024**3 for row in _memory_rows]
        _memory_ax.bar(
            range(len(_memory_rows)),
            _peaks,
            color=[_colors.get(row["arm"], "#9467bd") for row in _memory_rows],
        )
        _memory_ax.axhline(
            max(_limits), color="#B22222", linestyle="--", label="Hard limit"
        )
        for _index, _peak in enumerate(_peaks):
            _memory_ax.annotate(
                f"{_peak:.1f}",
                (_index, _peak),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        _memory_ax.set_xticks(
            range(len(_memory_rows)),
            [_labels.get(row["arm"], row["arm"]) for row in _memory_rows],
            rotation=15,
            ha="right",
        )
        _memory_ax.set_ylabel("Peak cgroup memory (GiB)")
        _memory_ax.set_title("Evaluation reached the session RAM limit", loc="left")
        _memory_ax.legend(frameon=False, fontsize=8)
        _plot_count += 1
    else:
        _memory_ax.text(0.5, 0.5, "Memory telemetry unavailable", ha="center")
        _memory_ax.set_axis_off()

    for _ax in _axes:
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)
        _ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.6)
        _ax.set_axisbelow(True)

    _content = [
        mo.md("## Compare the evidence"),
        mo.md(
            "Left: first-to-last change after training begins (10M→20M here); negative means lower. "
            "Objective loss is shown relative to each arm's own first checkpoint, so compare "
            "its direction only. "
            "Right: group MAE relative to control. Validation scores use the same fixed 1M-star set; "
            "the lower-mask arm combines stable embeddings, group-balanced loss and lower mask ratios."
        ),
    ]
    if _plot_count:
        _content.append(_fig)
    else:
        plt.close(_fig)
        _content.append(
            mo.md("Waiting for comparable results from the experiment arms.")
        )
    mo.vstack(_content)
    return


@app.cell
def _(
    get_arm_choice,
    get_regime_choice,
    mo,
    set_arm_choice,
    set_regime_choice,
    suite_snapshots,
):
    _arm_names = list(suite_snapshots)
    selected_arm = mo.ui.dropdown(
        _arm_names or ["Waiting for manifest"],
        value=get_arm_choice()
        if get_arm_choice() in _arm_names
        else _arm_names[0]
        if _arm_names
        else "Waiting for manifest",
        label="Candidate",
        on_change=set_arm_choice,
    )
    selected_regime = mo.ui.dropdown(
        ["common", "xp_on", "xp_off"],
        value=get_regime_choice(),
        label="Input regime",
        on_change=set_regime_choice,
    )
    mo.vstack(
        [
            mo.md("### Inspect residuals and feature-level QA"),
            mo.md(
                "Choose an arm and input regime to inspect residual shapes, XP-order "
                "errors, and validation bins."
            ),
            mo.hstack([selected_arm, selected_regime]),
        ]
    )
    return selected_arm, selected_regime


@app.cell
def _(json, mo, np, pd, plt, selected_arm, selected_regime, suite_snapshots):
    _qa = suite_snapshots.get(selected_arm.value, {})
    _regime = _qa.get("regimes", {}).get(selected_regime.value, {})
    _items = [mo.md(f"## Residuals · {selected_arm.value} · {selected_regime.value}")]
    if not _regime:
        _items.append(
            mo.md("Waiting for validation results for this candidate and input regime.")
        )
    else:
        _diagnostic = _qa.get("diagnostic_snapshot", {})
        _validation_rows = _qa.get("validation_rows_total")
        _sample_rows = _diagnostic.get("rows")
        _selection = _diagnostic.get("selection", "selection policy unavailable")
        if (
            isinstance(_validation_rows, int)
            and not isinstance(_validation_rows, bool)
            and isinstance(_sample_rows, int)
            and not isinstance(_sample_rows, bool)
        ):
            _items.append(
                mo.md(
                    f"Full-validation scores and residual histograms use all {_validation_rows:,} stars. "
                    f"The {_sample_rows:,}-star deterministic sample ({_selection}) is used only for "
                    "the saved residual arrays and latent-activity summaries."
                )
            )
        else:
            _items.append(
                mo.md(
                    "Validation and diagnostic sample sizes are unavailable; interpret the plots "
                    "as exploratory until the saved row counts are confirmed."
                )
            )
        _overall = _regime.get("overall", {})
        _blocks = _regime.get("blocks", {})
        _features = _regime.get("features", [])
        _features = (
            [{"name": name, **value} for name, value in _features.items()]
            if isinstance(_features, dict)
            else _features
        )
        _histograms = _regime.get("histograms", {})
        _fig, _axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        _axes = _axes.ravel()
        _colors = {
            "xp": "#4C78A8",
            "photo": "#F58518",
            "astro": "#54A24B",
        }
        _hist_ax = _axes[0]
        _hist_drawn = False
        _tails = []
        for _name in ("xp", "photo", "astro"):
            _hist = _histograms.get(_name, {})
            _edges, _counts = _hist.get("edges", []), _hist.get("counts", [])
            if len(_edges) == len(_counts) + 1 and _counts and sum(_counts) > 0:
                _hist_ax.stairs(
                    [100 * count / sum(_counts) for count in _counts],
                    _edges,
                    label={
                        "xp": "Gaia XP",
                        "photo": "Photometry",
                        "astro": "Astrometry",
                    }[_name],
                    color=_colors[_name],
                    linewidth=1.8,
                )
                _hist_drawn = True
                _tails.append(_hist.get("outside_range_count", 0))
        _hist_ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
        _hist_ax.set(
            xlabel="Prediction − target (RobustScaler units)",
            ylabel="Targets in bin (%)",
            title="Residuals cluster around zero; tails extend beyond ±5",
        )
        if _hist_drawn:
            _hist_ax.legend(frameon=False, fontsize=8)
            _hist_ax.text(
                0.02,
                0.95,
                f"Outside plotted range: {sum(_tails):,} targets",
                transform=_hist_ax.transAxes,
                va="top",
                fontsize=8,
            )
        else:
            _hist_ax.text(0.5, 0.5, "Residual histogram unavailable", ha="center")
            _hist_ax.set_axis_off()

        _orders = [
            (name, value.get("mae"))
            for name, value in _blocks.items()
            if name.startswith("xp_") and isinstance(value, dict)
        ]
        if _orders:
            _xp_ax = _axes[1]
            _xp_labels = {
                "xp_1_10": "Coefficients 1–10",
                "xp_11_30": "Coefficients 11–30",
                "xp_31_55": "Coefficients 31–55",
            }
            _bars = _xp_ax.barh(
                [_xp_labels.get(name, name) for name, _ in _orders],
                [np.nan if value is None else value for _, value in _orders],
                color="#4C78A8",
            )
            for _bar, (_, _value) in zip(_bars, _orders, strict=True):
                if _value is not None:
                    _xp_ax.annotate(
                        f"{_value:.2f}",
                        (_bar.get_width(), _bar.get_y() + _bar.get_height() / 2),
                        xytext=(4, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8,
                    )
            _xp_ax.set_xlabel("MAE (RobustScaler units)")
            _xp_ax.set_title(
                "Later XP coefficients are harder to reconstruct", loc="left"
            )
        else:
            _axes[1].text(0.5, 0.5, "XP-order summaries unavailable", ha="center")
            _axes[1].set_axis_off()

        _feature_rows = [
            row
            for row in _features
            if isinstance(row, dict) and isinstance(row.get("mae"), int | float)
        ]
        _feature_rows.sort(key=lambda row: row["mae"])
        _feature_rows = _feature_rows[-12:]
        if _feature_rows:
            _feature_ax = _axes[2]
            _bars = _feature_ax.barh(
                [row.get("name", "?") for row in _feature_rows],
                [row["mae"] for row in _feature_rows],
                color=[
                    "#D55E00"
                    if row.get("name", "").casefold() == "parallax"
                    else "#72A0C1"
                    for row in _feature_rows
                ],
            )
            for _bar in _bars:
                _feature_ax.annotate(
                    f"{_bar.get_width():.2f}",
                    (_bar.get_width(), _bar.get_y() + _bar.get_height() / 2),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=8,
                )
            _feature_ax.set_xlabel("MAE (RobustScaler units)")
            _feature_ax.set_title("Largest per-feature errors", loc="left")
        else:
            _axes[2].text(0.5, 0.5, "Per-feature scores unavailable", ha="center")
            _axes[2].set_axis_off()

        _bins = _regime.get("bins", {})
        _magnitude_rows = _bins.get("magnitude", [])
        if _magnitude_rows:
            _mag_ax = _axes[3]
            _mag_labels = [
                f"{row.get('lower', '?')}–{row.get('upper', '?')}"
                for row in _magnitude_rows
            ]
            _mag_ax.plot(
                _mag_labels,
                [row.get("mae", np.nan) for row in _magnitude_rows],
                marker="o",
                color="#4C78A8",
                label="MAE",
            )
            _mag_ax.plot(
                _mag_labels,
                [row.get("p95", np.nan) for row in _magnitude_rows],
                marker="o",
                color="#E45756",
                label="95th percentile",
            )
            _mag_ax.set_xlabel("Magnitude bin")
            _mag_ax.set_ylabel("Absolute error (RobustScaler units)")
            _mag_ax.set_title("Error grows for fainter bins", loc="left")
            _mag_ax.tick_params(axis="x", rotation=20)
            _mag_ax.legend(frameon=False, fontsize=8)
        else:
            _axes[3].text(0.5, 0.5, "Magnitude-bin QA unavailable", ha="center")
            _axes[3].set_axis_off()

        for _ax in _axes:
            if _ax.axison:
                _ax.spines["top"].set_visible(False)
                _ax.spines["right"].set_visible(False)
                _ax.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.6)
                _ax.set_axisbelow(True)
        _items.append(_fig)

        _block_rows = [
            {"group": name, **value}
            for name, value in _blocks.items()
            if isinstance(value, dict)
        ]
        _rounded_overall = {
            key: round(value, 3) if isinstance(value, int | float) else value
            for key, value in _overall.items()
        }
        _details = [mo.ui.table([_rounded_overall], selection=None)] if _overall else []
        if _block_rows:
            _details.append(
                mo.ui.table(pd.DataFrame(_block_rows).round(3), selection=None)
            )
        if _features:
            _details.append(
                mo.accordion(
                    {
                        "Per-feature MAE, bias and tails": mo.ui.table(
                            pd.DataFrame(_features).round(3), selection=None
                        )
                    }
                )
            )
        for _name in ("magnitude", "snr"):
            _bin_rows = _bins.get(_name, [])
            if _bin_rows:
                _title = (
                    "Magnitude-bin summaries"
                    if _name == "magnitude"
                    else f"SNR-bin summaries ({_qa.get('snr_units', 'unverified units')})"
                )
                _details.append(
                    mo.accordion(
                        {
                            _title: mo.ui.table(
                                pd.DataFrame(_bin_rows).round(3), selection=None
                            )
                        }
                    )
                )
        _latent = _qa.get("latent", {}) if selected_regime.value == "common" else {}
        if isinstance(_latent, dict) and _latent:
            _latent_summary = {
                key: round(value, 3) if isinstance(value, int | float) else value
                for key, value in _latent.items()
            }
            _details.append(
                mo.accordion(
                    {
                        "Latent activity (100K diagnostic rows)": mo.ui.table(
                            [_latent_summary], selection=None
                        )
                    }
                )
            )
        _uncertainty = {
            "units": _qa.get("uncertainty_units", "unverified"),
            "verified_features": _qa.get("verified_sigma_features", []),
            "SNR units": _qa.get("snr_units", "unverified"),
            "diagnostic": _overall.get(
                "uncertainty_normalized", _regime.get("uncertainty", {})
            ),
        }
        _items.append(
            mo.md(
                "**Calibration check:** Disabled because source uncertainty units are unverified. "
                "The SNR-bin diagnostic is also provisional."
            )
        )
        _details.append(
            mo.accordion(
                {
                    "Uncertainty-unit record": mo.md(
                        f"```json\n{json.dumps(_uncertainty, indent=2)}\n```"
                    )
                }
            )
        )
        _items.append(
            mo.accordion({"Exact QA summaries and units": mo.vstack(_details)})
        )
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
