"""Versioned StarHorse age cohorts; external seismic IDs never train or tune."""

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from pristine_quality_views import exact_rows
from stellar_catalog_jobs import atomic_json, identity_split
from stellar_dataset_views import EXPECTED_IDS, publish

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
POSTERIOR_SHA = "0d24f9a748b9664d6ae6888cb38ec0ca50fe91b1d685cd8f377fa09378e2ae36"


def quality_masks(age, distance, ageflag, inputflag, outputflag, ageinout):
    """Conservative design thresholds, never publisher certification or truth."""
    clean = (outputflag == b"") & (ageinout == b"")
    clean &= (np.char.find(inputflag, b"WARN") < 0) & (
        np.char.find(inputflag, b"BAD") < 0
    )

    def valid(a, precision):
        ok = np.isfinite(a).all(1) & (a > 0).all(1) & (np.diff(a, axis=1) >= 0).all(1)
        relative = np.full(len(a), np.inf)
        relative[ok] = (a[ok, 3] - a[ok, 1]) / (2 * a[ok, 2])
        return ok & (relative <= precision)

    return dict(
        age_candidate=clean & np.isin(ageflag, [b"MSTO", b"SGB"]) & valid(age, 0.30),
        distance_candidate=clean & valid(distance, 0.20),
        inputs_include_parallax=np.char.find(inputflag, b"PARALLAX") >= 0,
        age_median_above_13point8=np.isfinite(age[:, 2]) & (age[:, 2] > 13.8),
    )


def cohort_masks(
    ids, feature_available, benchmark_train, benchmark_validation, reference_ids
):
    positive = ids > 0
    train = np.isin(ids, benchmark_train) & positive
    validation = np.isin(ids, benchmark_validation) & positive
    if (train & validation).any():
        raise ValueError("Benchmark splits overlap")
    reference = np.isin(ids, reference_ids) & positive
    split = identity_split(ids)
    split[~positive] = 255
    allowed = positive & feature_available & ~reference & ~validation
    return dict(
        label_split=split,
        benchmark_partition=np.where(
            ~positive, 254, np.where(train, 0, np.where(validation, 1, 255))
        ).astype("u1"),
        external_reference=reference,
        strict_fit=allowed & train & (split == 0),
        broader_fit=allowed & (split == 0),
        tune=allowed & (split == 1),
        identity_test=positive & feature_available & ~reference & (split == 2),
    )


def feature_index(path, xp_columns):
    """First original feature row per positive ID; complete finite110 XP, no labels read."""
    with fits.open(path, memmap=True) as f:
        hdu = next(h for h in f if isinstance(h, fits.BinTableHDU))
        ids = np.asarray(hdu.data["source_id"], dtype="i8")
        _, first = np.unique(ids, return_index=True)
        first = np.sort(first[ids[first] > 0])
        available = np.ones(len(first), dtype=bool)
        for name in xp_columns:
            values = np.asarray(hdu.data[name])[first]
            available &= np.isfinite(values) & (values != -9999)
            null = hdu.columns[name].null
            if null is not None:
                available &= values != null
    return ids[first], first, available, len(ids) - len(first)


def materialize(output, scratch, root=ROOT, legacy=None):
    import yaml

    started = time.monotonic()
    if legacy is None:
        legacy = Path(
            "/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits"
        )
    posterior = root / "posterior_labels_v2/posterior-labels.h5"
    reference_path = Path("configs/legacy-seismic-reference-ids.json")
    reference = json.loads(reference_path.read_text())
    reference_ids = np.asarray(reference["source_id"], dtype="i8")
    if (
        len(reference_ids) != 66
        or len(np.unique(reference_ids)) != 66
        or (reference_ids <= 0).any()
    ):
        raise ValueError("Require all66 audited reference IDs")
    digest = hashlib.sha256(posterior.read_bytes()).hexdigest()
    if digest != POSTERIOR_SHA:
        raise ValueError("Audited posterior v2 checksum changed")
    with h5py.File(root / "dataset_view_v2/dataset-view.h5") as f:
        train, val = f["train/source_id"][:], f["validation/source_id"][:]
    ordered = hashlib.sha256(
        train.astype("<i8").tobytes() + val.astype("<i8").tobytes()
    ).hexdigest()
    if (
        ordered != EXPECTED_IDS
        or len(train) != 9000000
        or len(val) != 1000000
        or len(np.unique(np.r_[train, val])) != 10000000
    ):
        raise ValueError("Immutable benchmark IDs changed")
    config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
    xp = [
        name
        for name in config["data"]["feature_cols"]
        if name.startswith(("bp_", "rp_")) and name.split("_")[-1].isdigit()
    ]
    if len(xp) != 110:
        raise ValueError("Require exactly110 XP columns")
    feature_ids, feature_rows, feature_valid, duplicate_excess = feature_index(
        legacy, xp
    )
    staged = scratch / "age-feature-cohort.h5.building"
    summary = {}
    with h5py.File(posterior) as source, h5py.File(staged, "x") as out:
        out.attrs["policy"] = (
            "StarHorse model-dependent posterior labels; positive GaiaIDs only. Conservative flag/regime/precision design, not publisher gold. Canonical APOGEE17 before GALAHDR3 among quality+feature-eligible rows; no averaging or error-minimization. All66 LEGACY seismic IDs excluded from train AND tuning. Every benchmark validation ID excluded from fits and tuning. Complete finite110 XP required, original first feature-row rule. No astroNN/legacy invented age errors. Pretraining overlap not excluded."
        )
        out.attrs["posterior_sha256"] = digest
        out.attrs["benchmark_source_id_sha256"] = ordered
        out.attrs["reference_json"] = json.dumps(reference)
        out.attrs["legacy_feature_sha256"] = file_digest(legacy)
        out.attrs["xp_columns_json"] = json.dumps(xp)
        candidates = []
        ages = {}
        for survey in ["apogee17", "galahdr3"]:
            source.copy(survey, out.create_group("raw/" + survey), name="posterior")
            g = source[survey]
            ids = g["source_id"][:]
            ages[survey] = g["Age_percentiles"][:]
            if len(np.unique(ids[ids > 0])) != int((ids > 0).sum()):
                raise ValueError("Ambiguous within-survey identity")
            m = quality_masks(
                g["Age_percentiles"][:],
                g["Dist_percentiles"][:],
                *[
                    g["flags/" + col][:]
                    for col in ["AgeFlag", "InputFlag", "OutputFlag", "AgeInOut"]
                ],
            )
            fr = exact_rows(feature_ids, ids)
            available = (fr >= 0) & feature_valid[np.maximum(fr, 0)]
            m["age_candidate"] &= ids > 0
            m["distance_candidate"] &= ids > 0
            group = out.create_group("source_index/" + survey)
            rows = np.where(available, feature_rows[np.maximum(fr, 0)], -1)
            for key, value in dict(
                source_id=ids, feature_row=rows, feature_available=available, **m
            ).items():
                group.create_dataset(key, data=value, compression="gzip")
            eligible = np.flatnonzero(m["age_candidate"] & available)
            candidates.extend(
                (int(ids[row]), survey, int(row), int(rows[row])) for row in eligible
            )
            summary[survey] = dict(
                raw_rows=len(ids),
                age_candidates=int(m["age_candidate"].sum()),
                feature_available_age_candidates=len(eligible),
            )
        # Fixed survey precedence; it is independent of ages/errors/test outcomes.
        canonical = {}
        for candidate in candidates:
            canonical.setdefault(candidate[0], candidate)
        selected = sorted(canonical.values())
        ids = np.array([row[0] for row in selected], dtype="i8")
        surveys = np.array([row[1] for row in selected], dtype="S16")
        rows = np.array([row[2] for row in selected], dtype="i8")
        age = np.array([ages[row[1]][row[2]] for row in selected])
        masks = cohort_masks(
            ids, np.ones(len(ids), dtype=bool), train, val, reference_ids
        )
        dest = out.create_group("selected")
        for key, value in dict(
            source_id=ids,
            survey=surveys,
            source_row=rows,
            feature_row=np.array([row[3] for row in selected], dtype="i8"),
            age_percentiles=age,
            **masks,
        ).items():
            ds = dest.create_dataset(key, data=value, compression="gzip")
            if key == "age_percentiles":
                ds.attrs.update(unit="Gyr", percentiles=[5, 16, 50, 84, 95])
        refrow = exact_rows(feature_ids, reference_ids)
        ref_available = (refrow >= 0) & feature_valid[np.maximum(refrow, 0)]
        refgroup = out.create_group("external_reference")
        refgroup.create_dataset("source_id", data=reference_ids)
        refgroup.create_dataset(
            "feature_row",
            data=np.where(ref_available, feature_rows[np.maximum(refrow, 0)], -1),
        )
        refgroup.create_dataset(
            "selected_label_row", data=exact_rows(ids, reference_ids)
        )
        content = hashlib.sha256()

        def update(name, node):
            if isinstance(node, h5py.Dataset):
                content.update(name.encode())
                content.update(node[:].tobytes())

        out.visititems(update)
        out.attrs.update(complete=True, content_id=content.hexdigest())
        report = dict(
            sources=summary,
            unique_selected=len(ids),
            cross_survey_duplicate_candidates=len(candidates) - len(ids),
            feature_duplicate_excess=duplicate_excess,
            masks={
                key: int(value.sum())
                for key, value in masks.items()
                if value.dtype.kind == "b"
            },
            all66_reference_feature_available=int(ref_available.sum()),
            reference_with_quality_age_label=int(
                (exact_rows(ids, reference_ids) >= 0).sum()
            ),
            input_posterior_sha256=digest,
            benchmark_ordered_id_sha256=ordered,
            content_id=content.hexdigest(),
            training_performed=False,
            independent_truth=False,
            runtime_seconds=time.monotonic() - started,
        )
    target = output / "age-feature-cohort.h5"
    publish(staged, target)
    report.update(bytes=target.stat().st_size, sha256=file_digest(target))
    atomic_json(output / "age-cohort-report.json", report)
    return report


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
