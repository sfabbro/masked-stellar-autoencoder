"""Versioned StarHorse distance cohorts with separate measured-parallax holdouts."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from pristine_quality_views import exact_rows
from stellar_age_cohorts import (
    POSTERIOR_SHA,
    ROOT,
    cohort_masks,
    feature_index,
    file_digest,
    quality_masks,
)
from stellar_dataset_views import EXPECTED_IDS, publish
from stellar_hipparcos_inputs import acquire_geometry


def materialize(out, scratch, root=ROOT, legacy=None, cached_reference=None):
    out, scratch, root = Path(out), Path(scratch), Path(root)
    posterior = root / "posterior_labels_v2/posterior-labels.h5"
    legacy = Path(
        legacy
        or "/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits"
    )
    if file_digest(posterior) != POSTERIOR_SHA:
        raise ValueError("Audited posterior v2 checksum changed")
    if (
        file_digest(legacy)
        != "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
    ):
        raise ValueError("Original feature FITS checksum changed")
    with h5py.File(root / "dataset_view_v2/dataset-view.h5") as f:
        train, validation = f["train/source_id"][:], f["validation/source_id"][:]
    ordered = hashlib.sha256(
        train.astype("<i8").tobytes() + validation.astype("<i8").tobytes()
    ).hexdigest()
    if (
        ordered != EXPECTED_IDS
        or len(train) != 9000000
        or len(validation) != 1000000
        or len(np.unique(np.r_[train, validation])) != 10000000
    ):
        raise ValueError("Immutable benchmark changed")
    config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
    xp = [
        n
        for n in config["data"]["feature_cols"]
        if n.startswith(("bp_", "rp_")) and n.split("_")[-1].isdigit()
    ]
    if len(xp) != 110:
        raise ValueError("Require exactly 110 XP columns")
    feature_ids, feature_rows, valid, _ = feature_index(legacy, xp)
    reference_metadata = json.loads(
        Path("configs/legacy-seismic-reference-ids.json").read_text()
    )
    seismic = np.asarray(reference_metadata["source_id"], dtype="i8")
    if len(seismic) != 66 or len(np.unique(seismic)) != 66 or (seismic <= 0).any():
        raise ValueError("Require all 66 seismic reference IDs")
    if cached_reference is not None:
        import shutil

        shutil.copytree(cached_reference, out / "raw-reference")
    geometry = acquire_geometry(
        out / "raw-reference",
        feature_ids,
        train,
        validation,
        cached=cached_reference is not None,
    )
    holdout = np.union1d(geometry["source_id"], seismic)
    summary = {}
    path = scratch / "distance-feature-cohort.h5.building"
    with (
        h5py.File(posterior) as raw,
        h5py.File(path, "x") as result,
    ):
        result.attrs["complete"] = False
        result.attrs["policy"] = (
            "Versioned production cohort; model/parallax-conditioned StarHorse distanceskpc as training targets, never called independentgeometry. FixedAPO-beforeGALAH precedence, allrawsources/flags/asymmetricpercentiles retained. All99525HipparcosGaiaIDs plusall66seismicIDs excluded fromfit/tune. NativeformalHipparcosparallaxesmas separatevalidationreference;negativevaluesretained,notinverted. Ordinaryflag8/positioncheckedmeans identityQC,notabsolutecovariancecalibration. No modeltrained. Futureparallax-free/assisted inputmodes must shareIDs/budgets and referenceholdouts."
        )
        canonical = {}
        for survey in ["apogee17", "galahdr3"]:
            g = raw[survey]
            raw.copy(survey, result.create_group("raw/" + survey), name="posterior")
            ids = g["source_id"][:]
            if len(np.unique(ids[ids > 0])) != int((ids > 0).sum()):
                raise ValueError("Ambiguous within-survey identities")
            masks = quality_masks(
                g["Age_percentiles"][:],
                g["Dist_percentiles"][:],
                *[
                    g["flags/" + n][:]
                    for n in ["AgeFlag", "InputFlag", "OutputFlag", "AgeInOut"]
                ],
            )
            fr = exact_rows(feature_ids, ids)
            available = (fr >= 0) & valid[np.maximum(fr, 0)] & (ids > 0)
            eligible = masks["distance_candidate"] & available
            for row in np.flatnonzero(eligible):
                canonical.setdefault(
                    int(ids[row]), (survey, int(row), int(feature_rows[fr[row]]))
                )
            summary[survey] = dict(quality_feature_candidates=int(eligible.sum()))
        ids = np.array(sorted(canonical), dtype="i8")
        surveys = np.array([canonical[int(i)][0] for i in ids], dtype="S16")
        source_rows = np.array([canonical[int(i)][1] for i in ids], dtype="i8")
        features = np.array([canonical[int(i)][2] for i in ids], dtype="i8")
        arrays = {survey: raw[survey + "/Dist_percentiles"][:] for survey in raw}
        distance = np.array(
            [arrays[canonical[int(i)][0]][canonical[int(i)][1]] for i in ids]
        )
        masks = cohort_masks(
            ids, np.ones(len(ids), dtype=bool), train, validation, holdout
        )
        selected = result.create_group("selected")
        for key, value in dict(
            source_id=ids,
            survey=surveys,
            source_row=source_rows,
            feature_row=features,
            distance_percentiles=distance,
            **masks,
        ).items():
            node = selected.create_dataset(key, data=value, compression="gzip")
            if key == "distance_percentiles":
                node.attrs.update(unit="kpc", percentiles=[5, 16, 50, 84, 95])
        reference = result.create_group("hipparcos_reference")
        for key in geometry:
            reference.create_dataset(key, data=geometry[key], compression="gzip")
        fr = exact_rows(feature_ids, geometry["source_id"])
        available = (fr >= 0) & valid[np.maximum(fr, 0)]
        eligible = geometry["position_checked_reference_candidate"] & available
        reference.create_dataset(
            "feature_row",
            data=np.where(available, feature_rows[np.maximum(fr, 0)], -1),
            compression="gzip",
        )
        reference.create_dataset(
            "complete_xp_available", data=available, compression="gzip"
        )
        reference.create_dataset(
            "evaluation_candidate", data=eligible, compression="gzip"
        )
        result.create_dataset(
            "ALL_reference_holdout_source_ids", data=holdout, compression="gzip"
        )
        # Independent eligibility reconstruction; no label/reference parallax entersfit.
        expected = (ids > 0) & ~np.isin(ids, validation) & ~np.isin(ids, holdout)
        np.testing.assert_array_equal(
            masks["broader_fit"], expected & (masks["label_split"] == 0)
        )
        np.testing.assert_array_equal(
            masks["strict_fit"],
            expected & (masks["label_split"] == 0) & np.isin(ids, train),
        )
        np.testing.assert_array_equal(
            masks["tune"], expected & (masks["label_split"] == 1)
        )
        for key in ["strict_fit", "broader_fit", "tune"]:
            assert not np.isin(ids[masks[key]], np.r_[holdout, validation]).any()
        for survey in raw:
            take = surveys == survey.encode()
            np.testing.assert_array_equal(
                distance[take], raw[survey + "/Dist_percentiles"][:][source_rows[take]]
            )
            np.testing.assert_array_equal(
                ids[take], raw[survey + "/source_id"][:][source_rows[take]]
            )
        content = hashlib.sha256()

        def add_dataset(name, node):
            if isinstance(node, h5py.Dataset):
                content.update(name.encode())
                content.update(node[:].tobytes())

        result.visititems(add_dataset)
        result.attrs["content_id"] = content.hexdigest()
        result.attrs["pretraining_overlap_excluded"] = False
        result.attrs["benchmark_source_id_sha256"] = ordered
        result.attrs["reference_json"] = json.dumps(reference_metadata)
        result.attrs["xp_columns_json"] = json.dumps(xp)
        result.attrs["parallax_modes"] = (
            "Paired parallax-free/assisted inputs planned; blocking parallax does not establish independent training labels"
        )
        result.attrs["posterior_input_sha256"] = (
            "0d24f9a748b9664d6ae6888cb38ec0ca50fe91b1d685cd8f377fa09378e2ae36"
        )
        result.attrs["feature_input_sha256"] = (
            "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
        )
        result.attrs["hipparcos_geometry_input_sha256"] = file_digest(
            out / "raw-reference/input-sha256.json"
        )
        result.attrs["complete"] = True
    target = out / "distance-feature-cohort.h5"
    publish(path, target)
    path = target
    report = dict(
        independent_remote_publication_verification_pending=True,
        content_id=content.hexdigest(),
        unsupervised_pretraining_overlap_excluded=False,
        training_performed=False,
        unique_selected_distance_labels=len(ids),
        source_candidates=summary,
        holdout_union_ids=len(holdout),
        strict_fit=int(masks["strict_fit"].sum()),
        broader_fit=int(masks["broader_fit"].sum()),
        tune=int(masks["tune"].sum()),
        identity_test=int(masks["identity_test"].sum()),
        hipparcos_complete_xp_reference_candidates=int(eligible.sum()),
        hipparcos_reference_nonpositive_parallaxes_retained=int(
            (eligible & (geometry["hipparcos_parallax_mas"] <= 0)).sum()
        ),
        all_reference_and_validation_fit_tune_exclusions_independently_checked=True,
        asymmetric_target_units="kpc",
        reference_units="mas",
        parallax_conversion="predicted_parallax_mas=1/distance_kpc; neverinvertnoisymeasuredparallax",
        no_claim_beyond_gaia_accuracy=True,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
    )
    (out / "distance-cohort-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
