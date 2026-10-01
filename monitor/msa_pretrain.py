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
        value=os.environ.get("MSA_MONITOR_MODE", "Training run"),
        label="View",
    )
    run_id = mo.ui.text(
        label="Run ID",
        value=os.environ.get("MSA_MONITOR_RUN_ID", ""),
    )
    output_root = mo.ui.text(
        label="ARC run / experiment suite directory",
        value=os.environ.get("MSA_MONITOR_OUTPUT_ROOT", ""),
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
        label="Inspect candidate",
        on_change=set_arm_choice,
    )
    selected_regime = mo.ui.dropdown(
        ["common", "xp_on", "xp_off"],
        value=get_regime_choice(),
        label="Validation regime",
        on_change=set_regime_choice,
    )
    mo.hstack([selected_arm, selected_regime])
    return selected_arm, selected_regime


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
):
    _items = [mo.md("## Experiment comparison")]
    if not suite_manifest:
        _items.append(
            mo.md(
                "Waiting for manifest.json. Fixed validation and baselines appear before the first training interval."
            )
        )
    else:
        _items.append(
            mo.md(
                f"**Suite:** `{suite_manifest.get('suite_id', suite_manifest.get('run_id', '—'))}` · "
                f"**Status:** `{suite_manifest.get('status', '—')}` · "
                f"**Common validation:** `{suite_manifest.get('validation_id', 'unavailable')}`\n\n"
                "Rank uses equal-group skill against the median baseline over every row in the "
                "common fixed validation set: "
                "mean(1 − group MAE / group median MAE) across XP, photometry and astrometry. Higher is better. "
                "Only completed candidates with equal star-view and optimizer-step budgets are ranked. "
                "Raw unweighted MAE in RobustScaler units and live unfinished candidates remain visible. "
                "A bounded pilot prefix is not a full-population evaluation."
            )
        )
        _sample = suite_manifest.get("sample", {})
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
            "star views per arm": suite_manifest.get("presentations_per_arm"),
            "cache used (GiB)": _gib(_sample.get("cache_bytes")),
            "cache limit (GiB)": _gib(_cache_cap),
            "loader": "bounded prefix cache" if _sample else "pending",
            "sampling policy": _sample.get("policy", "pending"),
        }
        _items.append(mo.ui.table([_sample_summary], selection=None, pagination=False))
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
            _items.append(mo.md("### Natural XP and finite input availability"))
            _items.append(
                mo.ui.table(_availability_rows, selection=None, pagination=False)
            )
        if _feature_availability:
            _items.append(
                mo.accordion(
                    {
                        "Per-feature finite input counts": mo.ui.table(
                            _feature_availability, selection=None
                        )
                    }
                )
            )
        if suite_comparison:
            _items.append(
                mo.ui.table(
                    pd.DataFrame(suite_comparison), selection=None, pagination=False
                )
            )
        _provenance = json.dumps(suite_manifest, indent=2, sort_keys=True)
        _items.append(
            mo.accordion(
                {
                    "Configuration, checkpoint, sample counts and code provenance": mo.md(
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
    _items.append(
        mo.md(
            f"**Latest persisted metric:** {f'{max(0, _age):.1f} min ago' if _age is not None else 'pending'} · "
            "Refresh defaults to 5 minutes; click the refresh button for an immediate fetch. "
            "Fetch times below describe cached snapshots, not training progress."
        )
    )
    if _age is not None and _age > 45:
        _items.append(mo.md("⚠️ No new persisted metric for over 45 minutes."))
    if suite_errors:
        _items.append(mo.md("⚠️ Fetch/read errors; cached data may be stale."))
        _items.append(
            mo.ui.table(
                [{"file": key, "error": value} for key, value in suite_errors.items()],
                selection=None,
            )
        )
    if suite_files:
        _items.append(
            mo.accordion(
                {
                    "Snapshot freshness": mo.ui.table(
                        suite_files, selection=None, pagination=False
                    )
                }
            )
        )
    mo.vstack(_items)
    return


@app.cell
def _(mo, pd, plt, suite_records):
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4))
    _has_data = False
    _latest_rows = []
    for _name, _records in suite_records.items():
        _frame = pd.DataFrame(_records)
        if _frame.empty:
            continue
        _latest_rows.append({"arm": _name, **_records[-1]})
        if "optimizer_steps" not in _frame:
            continue
        for _ax, _column in zip(
            _axes,
            (
                "group_skill_median",
                "sampled_validation_mae",
                "train_loss",
                "gradient_norm_mean",
            ),
            strict=True,
        ):
            if _column in _frame and _frame[_column].notna().any():
                _ax.plot(_frame["optimizer_steps"], _frame[_column], ".-", label=_name)
                _has_data = True
    for _ax, _title in zip(
        _axes,
        (
            "Equal-group skill vs median",
            "Fixed validation MAE (scaled units)",
            "Optimized training objective",
            "Gradient norm before clipping",
        ),
        strict=True,
    ):
        _ax.set(xlabel="Optimizer steps", title=_title)
        if _ax.lines:
            _ax.legend(fontsize="small")
    _fig.tight_layout()
    _content = [mo.md("## Pilot learning and stability")]
    if _has_data:
        _content.append(_fig)
    else:
        plt.close(_fig)
        _content.append(mo.md("Waiting for initial validation and interval metrics."))
    if _latest_rows:
        _telemetry = pd.DataFrame(_latest_rows)
        _performance_columns = [
            name
            for name in (
                "arm",
                "event",
                "optimizer_steps",
                "rows_seen_total",
                "rows_per_second",
                "validation_seconds",
                "optimizer_batch_size",
                "train_loss",
                "learning_rate",
                "gradient_norm_mean",
                "frequency_gradient_fraction",
                "clipping_rate",
            )
            if name in _telemetry
        ]
        if _performance_columns:
            _performance_labels = {
                "rows_per_second": "Rows/s",
                "validation_seconds": "Validation seconds",
                "optimizer_batch_size": "Optimizer batch size",
            }
            _content.append(
                mo.ui.table(
                    _telemetry[_performance_columns].rename(
                        columns=_performance_labels
                    ),
                    selection=None,
                    pagination=False,
                )
            )

        _byte_columns = {
            "cache_bytes": "Cache used (GiB)",
            "projected_cache_bytes": "Projected cache (GiB)",
            "current_host_rss_bytes": "Host RSS current (GiB)",
            "peak_host_rss_bytes": "Host RSS peak (GiB)",
            "cgroup_memory_current_bytes": "Cgroup memory current (GiB)",
            "cgroup_memory_peak_bytes": "Cgroup memory peak (GiB)",
            "cgroup_memory_limit_bytes": "Cgroup hard limit (GiB)",
            "current_gpu_allocated_bytes": "GPU allocated current (GiB)",
            "peak_gpu_allocated_bytes": "GPU allocated peak (GiB)",
            "current_gpu_reserved_bytes": "GPU reserved current (GiB)",
            "peak_gpu_reserved_bytes": "GPU reserved peak (GiB)",
        }
        _empty_values = pd.Series(index=_telemetry.index, dtype="float64")
        for _source, _label in _byte_columns.items():
            _values = pd.to_numeric(
                _telemetry.get(_source, _empty_values),
                errors="coerce",
            )
            _telemetry[_label] = [
                round(float(_value) / (1024**3), 2) if pd.notna(_value) else None
                for _value in _values
            ]

        _cgroup_peak = pd.to_numeric(
            _telemetry.get("cgroup_memory_peak_bytes", _empty_values),
            errors="coerce",
        )
        _cgroup_limit = pd.to_numeric(
            _telemetry.get("cgroup_memory_limit_bytes", _empty_values),
            errors="coerce",
        )
        if ((_cgroup_limit > 0) & (_cgroup_peak >= _cgroup_limit)).fillna(False).any():
            _content.append(
                mo.md(
                    "**Cgroup memory peak reached or exceeded the reported hard limit; "
                    "peak headroom was zero.**"
                )
            )
        _resource_columns = [
            "arm",
            "Cache used (GiB)",
            "Projected cache (GiB)",
            "Host RSS current (GiB)",
            "Host RSS peak (GiB)",
            "Cgroup memory current (GiB)",
            "Cgroup memory peak (GiB)",
            "Cgroup hard limit (GiB)",
            "GPU allocated current (GiB)",
            "GPU allocated peak (GiB)",
            "GPU reserved current (GiB)",
            "GPU reserved peak (GiB)",
        ]
        _content.append(
            mo.md(
                "### Memory and cache resources\n"
                "Byte values use GiB (2³⁰ bytes). Cache use is compared with the run-specific limit above; "
                "filesystem-wide free space is not shown as session scratch headroom."
            )
        )
        _content.append(
            mo.ui.table(_telemetry[_resource_columns], selection=None, pagination=False)
        )
    mo.vstack(_content)
    return


@app.cell
def _(json, mo, np, pd, plt, selected_arm, selected_regime, suite_snapshots):
    _qa = suite_snapshots.get(selected_arm.value, {})
    _regime = _qa.get("regimes", {}).get(selected_regime.value, {})
    _items = [
        mo.md(f"## Validation QA · {selected_arm.value} · {selected_regime.value}")
    ]
    if not _regime:
        _items.append(
            mo.md(
                "Waiting for the fixed validation snapshot for this candidate/regime."
            )
        )
    else:
        _items.append(
            mo.md(
                f"**Snapshot:** `{_qa.get('timestamp_utc', '—')}` · "
                f"**Star views:** {_qa.get('rows_seen_total', '—')} · "
                f"**Units:** {_qa.get('units', 'saved RobustScaler units')}  \n"
                "Only artificially hidden, finite targets are scored. XP on leaves the entire XP block visible; "
                "XP off hides the entire XP block. A group with count zero has no scored targets."
            )
        )
        _diagnostic = _qa.get("diagnostic_snapshot", {})
        _validation_rows = _diagnostic.get("validation_rows")
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
                    f"**Full-validation JSON summaries:** {_validation_rows:,} rows; "
                    "overall, feature, group, histogram, and bin summaries use all validation rows.  \n"
                    f"**Bounded diagnostic sample:** {_sample_rows:,} of {_validation_rows:,} rows "
                    f"({_selection}). Residual NPZ arrays and latent activity use this sample only; "
                    "any NPZ-based charts are sample views, separate from the full-validation scores."
                )
            )
        else:
            _items.append(
                mo.md(
                    "Validation and diagnostic sample sizes are unavailable in this snapshot; "
                    "do not infer full-population coverage from its QA tables."
                )
            )
        _overall = _regime.get("overall", {})
        if _overall:
            _items.append(mo.ui.table([_overall], selection=None, pagination=False))
        _latent = _qa.get("latent", {}) if selected_regime.value == "common" else {}
        if isinstance(_latent, dict) and _latent:
            _items.append(mo.md("### Latent activity on diagnostic sample"))
            _items.append(
                mo.ui.table(
                    [
                        {
                            "latent std min": _latent.get("std_min"),
                            "latent std median": _latent.get("std_median"),
                            "latent std max": _latent.get("std_max"),
                            "effective rank": _latent.get("effective_rank"),
                            "near-constant dimensions": _latent.get(
                                "near_constant_dimensions"
                            ),
                            "sample rows": _latent.get("sample_rows"),
                            "sample policy": _latent.get("sample_policy"),
                        }
                    ],
                    selection=None,
                    pagination=False,
                )
            )
        _blocks = _regime.get("blocks", {})
        if _blocks:
            _block_rows = [
                {"group": name, **value}
                for name, value in _blocks.items()
                if isinstance(value, dict)
            ]
            _items.append(
                mo.md(
                    "### XP, photometry, astrometry and XP order groups\nSigned bias, RMSE and absolute tails use the same masked validation targets."
                )
            )
            _items.append(mo.ui.table(_block_rows, selection=None, pagination=False))
        _features = _regime.get("features", [])
        _features = (
            [{"name": name, **value} for name, value in _features.items()]
            if isinstance(_features, dict)
            else _features
        )
        if _features:
            _items.append(
                mo.accordion(
                    {
                        "Per-feature MAE, bias, RMSE and tails": mo.ui.table(
                            _features, selection=None
                        )
                    }
                )
            )
        _histograms = _regime.get("histograms", {})
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))
        _drawn = False
        for _name, _hist in _histograms.items():
            _edges, _counts = _hist.get("edges", []), _hist.get("counts", [])
            if len(_edges) == len(_counts) + 1 and _counts:
                _axes[0].stairs(_counts, _edges, label=_name)
                _drawn = True
        _orders = [
            (name, value.get("mae"))
            for name, value in _blocks.items()
            if name.startswith("xp_") and isinstance(value, dict)
        ]
        if _orders:
            _axes[1].bar(
                [name for name, _ in _orders],
                [np.nan if value is None else value for _, value in _orders],
            )
            _drawn = True
        _axes[0].set(
            xlabel="Signed prediction − target (scaled units)",
            ylabel="Masked entries",
            title="Signed residual distribution",
        )
        if _axes[0].patches:
            _axes[0].legend(fontsize="small")
        _axes[1].set(ylabel="MAE (scaled units)", title="XP coefficient order")
        _fig.tight_layout()
        if _drawn:
            _items.append(_fig)
        else:
            plt.close(_fig)
            _items.append(
                mo.md(
                    "Signed histograms and XP order plots await compact QA summaries."
                )
            )
        _bins = _regime.get("bins", {})
        for _name in ("magnitude", "snr"):
            _bin_rows = _bins.get(_name, [])
            if _bin_rows:
                _bin_title = _name.title()
                if _name == "snr":
                    _bin_title += f" bins · units: {_qa.get('snr_units', 'unverified')}"
                else:
                    _bin_title += " bins"
                _items.append(mo.md(f"### {_bin_title}"))
                _items.append(mo.ui.table(_bin_rows, selection=None, pagination=False))
            else:
                _items.append(
                    mo.md(f"{_name.title()} bin QA unavailable for this snapshot.")
                )
        _bin_fig, _bin_axes = plt.subplots(1, 2, figsize=(13, 4))
        _bins_drawn = False
        for _ax, _name in zip(_bin_axes, ("magnitude", "snr"), strict=True):
            _rows = _bins.get(_name, [])
            if _rows:
                _labels = [
                    f"{row.get('lower', '?')}–{row.get('upper', '?')}" for row in _rows
                ]
                for _stat in ("mae", "p95"):
                    _values = [row.get(_stat) for row in _rows]
                    if any(value is not None for value in _values):
                        _ax.plot(
                            _labels,
                            [np.nan if value is None else value for value in _values],
                            ".-",
                            label=_stat,
                        )
                        _bins_drawn = True
                _ax.tick_params(axis="x", rotation=25)
                if _ax.lines:
                    _ax.legend()
            _ax.set(
                title=f"{_name.title()} binned residuals",
                ylabel="Absolute residual (scaled units)",
            )
        _bin_fig.tight_layout()
        if _bins_drawn:
            _items.append(_bin_fig)
        else:
            plt.close(_bin_fig)
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
                "### Uncertainty residual diagnostics\nCalibration requires verified uncertainty units and finite positive sigma. XP uncertainty units in older HDF files may be unverified; count zero means the diagnostic is disabled."
            )
        )
        _items.append(
            mo.md(
                f"**Source sigma units:** `{_uncertainty['units']}` · "
                f"**SNR bins:** `{_uncertainty['SNR units']}`. "
                "Unverified sigma and SNR values are provisional and do not support calibration claims."
            )
        )
        _items.append(mo.md(f"```json\n{json.dumps(_uncertainty, indent=2)}\n```"))
    mo.vstack(_items)
    return


if __name__ == "__main__":
    app.run()
