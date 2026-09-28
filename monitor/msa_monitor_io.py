"""Small, read-only helpers used by the local CANFAR training monitor."""

from __future__ import annotations

import json
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
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL record at {path}:{index}") from exc
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
    ),
) -> tuple[dict[str, Path], dict[str, str]]:
    """Fetch only small monitoring files from the CADC ARC VOSpace with vcp."""
    root = _arc_output_path(output_root)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    errors: dict[str, str] = {}
    for filename in filenames:
        if Path(filename).name != filename:
            raise ValueError("monitor filenames must be simple file names")
        destination = cache / filename
        paths[filename] = destination
        remote_uri = f"arc:{str(root).removeprefix('/arc')}/pretrain/{filename}"
        fd, temp_name = tempfile.mkstemp(prefix=f".{filename}.", dir=cache)
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
