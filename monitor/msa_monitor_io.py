"""Small, read-only helpers used by the local CANFAR training monitor."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def load_jsonl(path: str | Path) -> list[dict]:
    """Read complete JSONL records and ignore a possibly partial final write."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines, start=1):
        if not line.endswith(("\n", "\r")) and index == len(lines):
            break
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL record at {path}:{index}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSONL object at {path}:{index}")
        records.append(record)
    return records


def _arc_output_path(output_root: str | Path) -> PurePosixPath:
    path = PurePosixPath(str(output_root))
    if (
        not path.is_absolute()
        or path.parts[:3] != ("/", "arc", "projects")
        or ".." in path.parts
    ):
        raise ValueError("output_root must be an absolute /arc/projects path")
    return path


def sync_run_outputs(
    output_root: str | Path,
    cache_dir: str | Path,
    *,
    filenames: tuple[str, ...] = (
        "progress.jsonl",
        "metrics.jsonl",
        "residual_stats.jsonl",
        "residual_latest.json",
    ),
    subdir: str = "pretrain",
) -> tuple[dict[str, Path], dict[str, str]]:
    """Fetch only small monitoring files from the CADC ARC VOSpace with vcp."""
    root = _arc_output_path(output_root)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    relative_dir = PurePosixPath(subdir)
    if relative_dir.is_absolute() or ".." in relative_dir.parts:
        raise ValueError("monitor subdir must be a relative path without '..'")
    for filename in filenames:
        relative = PurePosixPath(filename)
        if relative.is_absolute() or ".." in relative.parts or not relative.name:
            raise ValueError("monitor filenames must be relative paths without '..'")
        destination = cache / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        paths[filename] = destination
        remote_path = root / relative_dir / relative
        remote_uri = f"arc:{str(remote_path).removeprefix('/arc')}"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{relative.name}.", dir=destination.parent
        )
        os.close(fd)
        temp_path = Path(temp_name)
        temp_path.unlink()
        try:
            result = subprocess.run(
                ["vcp", remote_uri, str(temp_path)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode or not temp_path.is_file():
                message = result.stderr.strip() or "vcp did not create a local file"
                if "NodeNotFound" not in message:
                    errors[filename] = message
                continue
            os.replace(temp_path, destination)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors[filename] = str(exc)
        finally:
            temp_path.unlink(missing_ok=True)
    return paths, errors


def load_snapshot(path: str | Path) -> dict:
    """Load a compact JSON object; missing output is pending, malformed output fails."""
    path = Path(path)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return value


def experiment_comparison(
    manifest: dict, records: dict[str, list[dict]], snapshots: dict[str, dict]
) -> list[dict]:
    """Rank completed equal-budget arms by equal-group skill on common validation."""
    rows = []
    expected_validation = manifest.get("validation_id")
    for arm in manifest.get("arms", []):
        info = arm if isinstance(arm, dict) else {"arm": arm}
        name = info.get("arm_id", info.get("arm", info.get("name", "")))
        arm_records = records.get(name, [])
        latest = arm_records[-1:] or [{}]
        metric = latest[0]
        qa = snapshots.get(name, {})
        overall = (
            qa.get("regimes", {})
            .get("common", {})
            .get("overall", qa.get("overall", {}))
        )
        score = metric.get("sampled_validation_mae", overall.get("mae"))
        baseline = metric.get("median_mae", overall.get("median_mae"))
        finite = (
            isinstance(score, int | float)
            and math.isfinite(score)
            and metric.get("finite", True) is not False
            and qa.get("finite", True) is not False
        )
        validation = metric.get("validation_id", qa.get("validation_id"))
        comparable = bool(
            expected_validation
            and validation == expected_validation
            and qa.get("validation_id", expected_validation) == expected_validation
        )
        initial = next((row for row in arm_records if row.get("event") == "init"), {})
        initial_mae = initial.get("sampled_validation_mae")
        initial_skill = initial.get("group_skill_median")
        group_skill = metric.get(
            "group_skill_median", overall.get("group_skill_median")
        )
        initial_valid = initial.get("validation_id") == expected_validation
        rows.append(
            {
                "arm": name,
                "status": info.get("status", metric.get("event", "pending")),
                "rank": None,
                "group skill vs median": group_skill,
                "initial group skill": initial_skill if initial_valid else None,
                "group skill improvement": group_skill - initial_skill
                if initial_valid
                and isinstance(group_skill, int | float)
                and isinstance(initial_skill, int | float)
                and math.isfinite(group_skill)
                and math.isfinite(initial_skill)
                else None,
                "ranking eligibility": "pending",
                "raw validation MAE": score,
                "initial raw MAE": initial_mae if initial_valid else None,
                "MAE improvement": initial_mae - score
                if initial_valid
                and finite
                and isinstance(initial_mae, int | float)
                and math.isfinite(initial_mae)
                else None,
                "median MAE": baseline,
                "skill vs median": (
                    1 - score / baseline
                    if finite
                    and isinstance(baseline, int | float)
                    and math.isfinite(baseline)
                    and baseline > 0
                    else None
                ),
                "group MAE": metric.get("group_mae"),
                "finite": finite,
                "common validation": comparable,
                "optimizer steps": metric.get("optimizer_steps"),
                "star views": metric.get("rows_seen_total"),
                "budget progress %": (
                    100 * metric["rows_seen_total"] / manifest["presentations_per_arm"]
                    if isinstance(metric.get("rows_seen_total"), int | float)
                    and isinstance(manifest.get("presentations_per_arm"), int | float)
                    and manifest["presentations_per_arm"] > 0
                    else None
                ),
                "learning rate": metric.get("learning_rate"),
                "updated": metric.get("timestamp_utc", qa.get("timestamp_utc")),
            }
        )
    eligible = []
    for row in rows:
        skill = row["group skill vs median"]
        if (
            not row["finite"]
            or not isinstance(skill, int | float)
            or not math.isfinite(skill)
        ):
            reason = "missing / nonfinite score"
        elif not row["common validation"]:
            reason = "different validation sample"
        elif row["status"] != "complete":
            reason = "unfinished"
        elif row["star views"] is None or row["optimizer steps"] is None:
            reason = "missing budget"
        elif (
            manifest.get("presentations_per_arm") is not None
            and row["star views"] != manifest["presentations_per_arm"]
        ):
            reason = "different star-view budget"
        else:
            reason = "eligible"
            eligible.append(row)
        row["ranking eligibility"] = reason
    # ponytail: with unequal completed step budgets, display scores but withhold ranks.
    if len({(row["star views"], row["optimizer steps"]) for row in eligible}) > 1:
        for row in eligible:
            row["ranking eligibility"] = "different update budgets"
        eligible = []
    eligible.sort(key=lambda row: row["group skill vs median"], reverse=True)
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    return sorted(rows, key=lambda row: row["rank"] or math.inf)
