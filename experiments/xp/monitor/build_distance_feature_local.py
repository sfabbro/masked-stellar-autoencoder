"""Local distance cohort design: native model targets, separate Hipparcos tests."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pristine_quality_views import exact_rows
from stellar_age_cohorts import cohort_masks, feature_index, quality_masks

ROOT = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
out = ROOT / "distance-feature-local-v1"
out.mkdir(exist_ok=False)
geometry = np.load(ROOT / "hipparcos2-reference-v1/geometric-reference-quality-v2.npz")
seismic = np.asarray(
    json.loads(Path("configs/legacy-seismic-reference-ids.json").read_text())[
        "source_id"
    ],
    dtype="i8",
)
holdout = np.union1d(geometry["source_id"], seismic)
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
xp = [n for n in config["data"]["feature_cols"] if n.startswith(("bp_", "rp_"))]
feature_ids, feature_rows, valid, _ = feature_index(
    "/tmp/msa-labels-complete-20261015.fits", xp
)
with h5py.File(ROOT / "dataset_view_v2/dataset-view.h5") as f:
    train = f["train/source_id"][:]
    validation = f["validation/source_id"][:]
summary = {}
path = out / "distance-feature-cohort.h5"
with (
    h5py.File(ROOT / "posterior_labels_v2/posterior-labels.h5") as raw,
    h5py.File(path, "x") as result,
):
    result.attrs["complete"] = False
    result.attrs["policy"] = (
        "LOCAL design only; model/parallax-conditioned StarHorse distanceskpc as training targets, never called independentgeometry. FixedAPO-beforeGALAH precedence, allrawsources/flags/asymmetricpercentiles retained. All99525HipparcosGaiaIDs plusall66seismicIDs excluded fromfit/tune. NativeformalHipparcosparallaxesmas separatevalidationreference;negativevaluesretained,notinverted. Ordinaryflag8/positioncheckedmeans identityQC,notabsolutecovariancecalibration. No modeltrained. Futureparallax-free/assisted inputmodes must shareIDs/budgets and referenceholdouts."
    )
    canonical = {}
    for survey in ["apogee17", "galahdr3"]:
        g = raw[survey]
        raw.copy(survey, result.create_group("raw/" + survey), name="posterior")
        ids = g["source_id"][:]
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
    masks = cohort_masks(ids, np.ones(len(ids), dtype=bool), train, validation, holdout)
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
    for key in geometry.files:
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
    reference.create_dataset("evaluation_candidate", data=eligible, compression="gzip")
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
    np.testing.assert_array_equal(masks["tune"], expected & (masks["label_split"] == 1))
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
    result.attrs["parallax_modes"] = (
        "Futurepairedparallax-free/assistedinputs;no inference oflabelindependencefromfeatureblocking"
    )
    result.attrs["posterior_input_sha256"] = (
        "0d24f9a748b9664d6ae6888cb38ec0ca50fe91b1d685cd8f377fa09378e2ae36"
    )
    result.attrs["feature_input_sha256"] = (
        "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
    )
    result.attrs["hipparcos_geometry_input_sha256"] = hashlib.sha256(
        (
            ROOT / "hipparcos2-reference-v1/geometric-reference-quality-v2.npz"
        ).read_bytes()
    ).hexdigest()
    result.attrs["complete"] = True
report = dict(
    LOCAL_not_ARC_published=True,
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
(out / "distance-cohort-design-audit.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
