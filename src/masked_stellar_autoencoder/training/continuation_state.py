"""Atomic, hashed continuation checkpoints and immutable QA snapshots.

Only trusted local checkpoints are loaded. Remote publication is a separate,
bounded operation: a failure must never corrupt or discard local training QA.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, content):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(content, handle, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_run(directory):
    """One process per output; OS releases lock after an interruption."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns this training output") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_checkpoint(directory, payload, *, epoch, step, identity):
    """Write complete state first, then atomically replace a checksum pointer.

    Keep epoch-boundary files and the latest two intermediate files. A killed
    writer leaves the previous pointer usable; temporary files are not resumed.
    """
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    boundary = payload["next_row"] == 0
    # Never replace a file referenced by the current pointer: QA can commit
    # another payload at the same optimizer step, and a kill may occur anytime.
    name = (
        f"{'epoch' if boundary else 'step'}-{epoch:03d}-{step:09d}-{time.time_ns()}.pth"
    )
    path = directory / name
    temporary = path.with_suffix(".pth.tmp")
    with temporary.open("wb") as handle:
        torch.save({**payload, "identity": identity}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    pointer = dict(
        path=name, sha256=sha256(path), epoch=epoch, step=step, identity=identity
    )
    atomic_json(directory / "latest.json", pointer)
    for stale in sorted(directory.glob("step-*.pth"))[:-2]:
        if stale != path:
            stale.unlink()
    return pointer


def load_checkpoint(directory, identity):
    from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

    directory = Path(directory)
    pointer = json.loads((directory / "latest.json").read_text())
    name = pointer["path"]
    if Path(name).name != name or not name.endswith(".pth"):
        raise ValueError("Invalid checkpoint pointer")
    if pointer["identity"] != identity:
        raise ValueError("Cannot resume with changed source/data/configuration")
    path = directory / name
    if sha256(path) != pointer["sha256"]:
        raise ValueError(
            "Checkpoint checksum mismatch; choose a verified prior checkpoint explicitly"
        )
    state = torch_load_trusted(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity:
        raise ValueError("Checkpoint identity differs from pointer")
    return state


def write_qa_snapshot(directory, manifest, tables):
    """Write CSVs into a unique immutable publication, manifest last."""
    directory = Path(directory)
    publication = directory / manifest["publication_id"]
    publication.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, (columns, rows) in tables.items():
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("QA files must be named CSVs")
        path = publication / name
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        files[name] = dict(
            path=f"{publication.name}/{name}",
            bytes=path.stat().st_size,
            sha256=sha256(path),
        )
    result = {**manifest, "files": files}
    atomic_json(publication / "manifest.json", result)
    atomic_json(directory / "manifest.json", result)
    return result


def publish_qa(directory, manifest, remote_root, *, copy=None, timeout=60):
    """Bounded VOS publication. Never publish the pointer before all data.

    Requires a pre-created run QA container. vcp performs authenticated copies;
    no secrets, shell interpolation or prediction data encoded in logs.
    """
    if not remote_root.startswith("vos:sfabbro/") or any(
        c in remote_root for c in "\r\n?#"
    ):
        raise ValueError("QA publication must use a plain vos:sfabbro/ run path")
    directory = Path(directory)
    if copy is None:
        auth = (
            ["--certfile", os.environ["MSA_CADC_CERT"]]
            if os.environ.get("MSA_CADC_CERT")
            else []
        )
        subprocess.run(
            [
                "vmkdir",
                *auth,
                "-p",
                remote_root.rstrip("/") + "/" + manifest["publication_id"],
            ],
            check=True,
            timeout=timeout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def copy(source, destination):
            subprocess.run(
                ["vcp", *auth, str(source), destination],
                check=True,
                timeout=timeout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    for item in manifest["files"].values():
        path = item["path"]
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Unsafe QA path")
        local = directory / path
        if sha256(local) != item["sha256"] or local.stat().st_size != item["bytes"]:
            raise ValueError("QA snapshot changed before publication")
        copy(local, remote_root.rstrip("/") + "/" + path)
    # Use the immutable manifest, not a mutable file another publisher may edit.
    copy(
        directory / manifest["publication_id"] / "manifest.json",
        remote_root.rstrip("/") + "/manifest.json",
    )
