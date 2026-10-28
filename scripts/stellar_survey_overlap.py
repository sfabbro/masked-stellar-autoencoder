"""Read-only survey overlap diagnostics; no label correction is applied."""

import hashlib
import itertools
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
DTYPE = [
    ("source_id", "<i8"),
    ("value", "<f8"),
    ("sigma", "<f8"),
    ("source_row", "<i8"),
]


def unique_best(records):
    if not len(records):
        return records
    order = np.lexsort((records["sigma"], records["source_id"]))
    ordered = records[order]
    return ordered[np.r_[True, np.diff(ordered["source_id"]) != 0]]


def paired(left, right):
    from stellar_catalog_jobs import identity_split

    left, right = unique_best(left), unique_best(right)
    ids, li, ri = np.intersect1d(
        left["source_id"], right["source_id"], assume_unique=True, return_indices=True
    )
    return left[li], right[ri], identity_split(ids)


def statistics(delta, sigma):
    if not len(delta):
        return {"n": 0}
    median = float(np.median(delta))
    z = delta / sigma
    return dict(
        n=len(delta),
        median_delta_dex=median,
        mae_delta_dex=float(np.mean(abs(delta))),
        robust_scatter_dex=float(1.4826 * np.median(abs(delta - median))),
        median_reported_pair_sigma_dex=float(np.median(sigma)),
        fraction_abs_delta_gt_3_reported_sigma=float(np.mean(abs(z) > 3)),
    )


def build_audit(output, scratch, root=ROOT):
    from stellar_catalog_jobs import atomic_json

    paths = {
        "native": root / "label_registry_v2/labels.h5",
        "PIGS": root / "pigs_labels_v1/pigs-labels.h5",
        "DESI": root / "desi_labels_v1/desi-labels.h5",
    }
    provenance = {}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as src:
            while chunk := src.read(8 * 1024 * 1024):
                digest.update(chunk)
        provenance[name] = dict(
            path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()
        )
    atomic_json(output / "input-provenance.json", provenance)
    surveys = {}
    with h5py.File(paths["native"]) as f:
        d = f["measurements"].fields(
            ["source_id", "value", "uncertainty", "survey", "quality_pass", "quantity"]
        )[:]
        for survey in [b"galah", b"apogee"]:
            mask = (
                (d["survey"] == survey) & d["quality_pass"] & (d["quantity"] == b"FeH")
            )
            a = np.zeros(int(mask.sum()), dtype=DTYPE)
            for dest, source in [
                ("source_id", "source_id"),
                ("value", "value"),
                ("sigma", "uncertainty"),
            ]:
                a[dest] = d[source][mask]
            a["source_row"] = np.flatnonzero(mask)
            surveys[survey.decode()] = unique_best(a)
    with h5py.File(paths["PIGS"]) as f:
        d = f["selected/labels"][:]
        a = np.zeros(len(d), dtype=DTYPE)
        for dest, source in [
            ("source_id", "source_id"),
            ("value", "value"),
            ("sigma", "uncertainty"),
            ("source_row", "source_row"),
        ]:
            a[dest] = d[source]
        surveys["PIGS"] = unique_best(a)
    with h5py.File(paths["DESI"]) as f:
        d = f["selected/measurements"].fields(
            ["source_id", "feh_corrected", "feh_err_native", "source_row"]
        )[:]
        a = np.zeros(len(d), dtype=DTYPE)
        for dest, source in [
            ("source_id", "source_id"),
            ("value", "feh_corrected"),
            ("sigma", "feh_err_native"),
            ("source_row", "source_row"),
        ]:
            a[dest] = d[source]
        surveys["DESI"] = unique_best(a)
    results = {}
    arrays = {}
    for ln, rn in itertools.combinations(surveys, 2):
        name = ln + "_minus_" + rn
        matched_left, r, split = paired(surveys[ln], surveys[rn])
        delta = matched_left["value"] - r["value"]
        sigma = np.hypot(matched_left["sigma"], r["sigma"])
        train = split == 0
        offset = float(np.median(delta[train])) if train.any() else None
        summary = dict(
            pairs=len(matched_left), train_only_median_offset_dex=offset, bins={}
        )
        for si, sl in [(0, "train"), (1, "tune"), (2, "test")]:
            for label, selection in [
                ("all", np.ones(len(matched_left), bool)),
                ("left_FeH_lt_minus2", matched_left["value"] < -2),
                ("right_FeH_lt_minus2", r["value"] < -2),
                (
                    "both_FeH_lt_minus2",
                    (matched_left["value"] < -2) & (r["value"] < -2),
                ),
            ]:
                mask = (split == si) & selection
                stat = statistics(delta[mask], sigma[mask])
                stat["after_train_offset"] = (
                    statistics(delta[mask] - offset, sigma[mask])
                    if offset is not None
                    else None
                )
                summary["bins"][sl + "_" + label] = stat
        results[name] = summary
        for key, data in [
            ("source_id", matched_left["source_id"]),
            ("split", split),
            ("left_feh", matched_left["value"]),
            ("right_feh", r["value"]),
            ("left_sigma", matched_left["sigma"]),
            ("right_sigma", r["sigma"]),
            ("left_source_row", matched_left["source_row"]),
            ("right_source_row", r["source_row"]),
        ]:
            arrays[name + "__" + key] = data
    np.savez_compressed(output / "overlap-pairs.npz", **arrays)
    report = dict(
        comparisons=results,
        input_provenance=provenance,
        survey_selected_counts={k: len(v) for k, v in surveys.items()},
        caveats=[
            "Exact GaiaDR3 matches; native quality-selected stars; smallest reported error per duplicate survey ID.",
            "Offsets are descriptive and fitted on train identities only; no source label is changed or merged.",
            "Neither catalogue is absolute truth; overlap selection is not the Galactic population.",
            "DESI sigma is native random FEH_ERR only; unknown correction systematics, covariance and temperature/gravity uncertainty remain separate.",
            "PIGS published uncertainties already contain its0.13dex floor; it is not added again.",
            "Quadrature normalization assumes independent errors and is not total uncertainty calibration.",
            "Native source_row references registry measurements; PIGS and DESI source_row reference their original tables.",
        ],
    )
    atomic_json(output / "overlap-summary.json", report)
    return report
