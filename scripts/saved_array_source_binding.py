"""Bind saved-array reports to the exact prepared source used by their producer."""

import json
from pathlib import Path


def prepare_report_output(output, waiting_status, reuse=False):
    if not output.exists():
        output.mkdir(parents=True, exist_ok=False)
        return
    if not reuse:
        raise FileExistsError("Report output already exists")
    previous = json.loads((output / "manifest.json").read_text())
    allowed = (
        list(waiting_status) if isinstance(waiting_status, tuple) else [waiting_status]
    )
    allowed.append("upstream_failed")
    if (
        previous.get("status") not in allowed
        or list(output.glob("*.npz"))
        or list(output.glob("*.png"))
    ):
        raise ValueError("Cannot replace a report after array processing starts")
    attempt = 1
    while (output / f"pre-array-attempt-{attempt}.json").exists():
        attempt += 1
    with (output / f"pre-array-attempt-{attempt}.json").open("xb") as stream:
        stream.write((output / "manifest.json").read_bytes())


def load_bound_source(producer, sha):
    if not producer.get("source_manifest_path") or not producer.get("source_sha256"):
        raise ValueError("Producer must declare its actual source manifest and digest")
    source = json.loads(Path(producer["source_manifest_path"]).read_text())
    if (
        source.get("prepared_sha256") != producer["source_sha256"]
        or sha(source["prepared_path"]) != producer["source_sha256"]
    ):
        raise ValueError("Saved-array producer and native-reference source differ")
    return source


def population_panel_masks(source, panel_rows, *, count=2048, seed=2026110742):
    """Recover the producer's fixed uniform panels, excluding added QA-only stars."""
    import h5py
    import numpy as np

    rows = np.asarray(panel_rows, dtype="i8")
    if np.any(np.diff(rows) <= 0):
        raise ValueError("Unique sorted producer rows required")
    with h5py.File(source["prepared_path"]) as data:
        roles = data["role"][:]
        eligible = data["primary_input_eligible"][:].astype(bool)
    masks = {}
    for role in ["fit", "devtest"]:
        candidates = np.flatnonzero((roles == role.encode()) & eligible)
        selected = np.sort(
            np.random.default_rng(seed).choice(candidates, count, replace=False)
        )
        if not np.isin(selected, rows).all():
            raise ValueError(
                "Producer uniform-panel contract differs; do not label augmented QA rows a population"
            )
        masks["population_" + role] = np.isin(rows, selected)
    return masks
