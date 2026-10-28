"""Deterministic, model-blind representative metal-poor corner-plot selection.

CPU headless CANFAR data arithmetic only. No predictions are read and no model
is loaded. Fixed FeH/brightness cells and source-diverse hash selection prevent
choosing aesthetically favorable objects after looking at a predictor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml

TARGETS = ("teff", "logg", "fe_h", "alpha", "age", "distance")
EDGES = (-np.inf, -3.0, -2.5, -2.0)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashed_ids(ids, salt):
    return sorted(
        range(len(ids)),
        key=lambda row: hashlib.sha256(f"{salt}:{int(ids[row])}".encode()).digest(),
    )


def select_rows(data, *, salt="msa-metalpoor-corners-v1"):
    ids = np.asarray(data["source_id"])
    roles = np.asarray(data["role"])
    quality = np.asarray(data["quality"], bool) & np.asarray(data["observed"], bool)
    target = np.asarray(data["value"], dtype="f8")
    if (
        ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or np.any(ids <= 0)
    ):
        raise ValueError("Exact positive unique source IDs required")
    if (
        tuple(data["target_names"]) != TARGETS
        or target.shape != (len(ids), 6)
        or quality.shape != target.shape
    ):
        raise ValueError("Six-target array/order contract differs")
    source_roles = np.asarray(data["label_source_role"])
    if source_roles.shape != target.shape or np.any(
        quality & (source_roles != roles[:, None])
    ):
        raise ValueError("Scoring includes a label promoted from another role")
    names = list(data["feature_names"])
    if names.count("G") != 1:
        raise ValueError("Unique raw G feature required")
    g_index = names.index("G")
    g = np.asarray(data["raw"], dtype="f8")[:, g_index]
    visible = np.asarray(data["visible"], bool)[:, g_index]
    available = visible & np.isfinite(g) & (g != -9999)
    fit = roles == "fit"
    if not np.any(fit & available):
        raise ValueError("No finite FIT-only raw G for brightness split")
    median = float(np.median(g[fit & available]))
    feh = target[:, 2]
    eligible = (
        (roles == "test")
        & quality[:, 2]
        & (source_roles[:, 2] == "test")
        & np.isfinite(feh)
        & (feh < -2)
        & available
    )
    sources = np.char.lower(np.asarray(data["label_source"]).astype("U64")[:, 2])
    cells = []
    selected = []
    for lower, upper in zip(EDGES[:-1], EDGES[1:], strict=True):
        label = f"[{lower},{upper})"
        for brightness in ("bright", "faint"):
            brightness_mask = g < median if brightness == "bright" else g >= median
            candidates = np.flatnonzero(
                eligible & (feh >= lower) & (feh < upper) & brightness_mask
            )
            ranking = np.asarray(
                hashed_ids(ids[candidates], salt + label + brightness), dtype="i8"
            )
            ordered = candidates[ranking]
            survey_rows = {
                name: ordered[np.char.find(sources[ordered], name) >= 0]
                for name in ("apogee", "galah")
            }
            if all(len(rows) for rows in survey_rows.values()):
                chosen = np.array(
                    [survey_rows["apogee"][0], survey_rows["galah"][0]], dtype="i8"
                )
            else:
                chosen = ordered[:2]
            selected.extend(chosen.tolist())
            cells.append(
                dict(
                    feh_interval=label,
                    brightness=brightness,
                    candidate_count=len(candidates),
                    selected_source_ids=ids[chosen].astype("i8").tolist(),
                    selected_sources=sources[chosen].tolist(),
                    empty=len(candidates) == 0,
                    source_diversity_preference_applied=all(
                        len(rows) for rows in survey_rows.values()
                    ),
                )
            )
    selected = np.asarray(selected, dtype="i8")
    if len(np.unique(ids[selected])) != len(selected) or len(selected) > 12:
        raise ValueError("Representative selection repeated IDs or exceeded 12")
    pair_counts = quality[fit].astype("i8").T @ quality[fit].astype("i8")
    return selected, dict(
        fit_only_G_median=median,
        fit_G_count=int((fit & available).sum()),
        eligible_metalpoor_test_count=int(eligible.sum()),
        cells=cells,
        fit_pair_observation_counts=pair_counts.tolist(),
        target_names=TARGETS,
        selected_count=len(selected),
    )


def run(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError(
            "Representative native selection must execute on CANFAR CPU headless"
        )
    if output.exists():
        raise FileExistsError(output)
    prepared = Path(config["prepared"])
    if sha(prepared) != config["prepared_sha256"]:
        raise ValueError("Prepared source hash changed")
    manifest = json.loads(Path(config["prepared_manifest"]).read_text())
    if (
        manifest["status"] != "sparse_six_target_source_assembly_complete"
        or manifest["input_sha256"] != config["prepared_sha256"]
        or manifest["source_commit"] != config["prepared_source_commit"]
    ):
        raise ValueError("Prepared source manifest differs")
    with np.load(prepared, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    rows, report = select_rows(data, salt=config["hash_salt"])
    output.mkdir(parents=True, exist_ok=False)
    selected = {
        key: data[key][rows]
        for key in (
            "source_id",
            "role",
            "input",
            "visible",
            "raw",
            "feature_errors",
            "value",
            "observed",
            "quality",
            "error",
            "error_observed",
            "posterior_quantiles",
            "label_source",
            "label_source_role",
        )
    }
    selected.update(
        prepared_row=rows,
        target_names=np.asarray(TARGETS),
        feature_names=data["feature_names"],
        fit_pair_observation_counts=np.asarray(report["fit_pair_observation_counts"]),
        fit_only_G_median=np.array(report["fit_only_G_median"]),
    )
    np.savez_compressed(output / "selection.npz", **selected)
    report.update(
        status="representative_metalpoor_corner_selection_complete",
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
        bindings=dict(
            policy=config,
            prepared_sha256=sha(prepared),
            preparation_manifest_sha256=sha(config["prepared_manifest"]),
            selector_sha256=sha(Path(__file__)),
            selection_sha256=sha(output / "selection.npz"),
        ),
        model_or_prediction_output_used=False,
        scope="Fixed model-blind representative metal-poor native TEST stars; not a weighted population sample",
        missing_reference_policy="Overlay only original-role eligible coordinates; unknown errors absent; StarHorse quantiles not measurement errors",
        alpha_warning="DESI alpha at FeH below -2 is diagnostic only, not recommended physical abundance truth",
    )
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
