import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training import label_overlap as lo
from masked_stellar_autoencoder.training import master_labels as ml


def test_model_selection_prefers_simple_maps_and_detects_curvature():
    rng = np.random.default_rng(0)
    y = rng.uniform(-3.5, 0.3, 3000)
    noise = rng.normal(0, 0.08, y.size)
    # survey reads 0.1 dex high everywhere -> offset suffices
    offset = lo.model_selection(y + 0.1 + noise, y)
    assert offset["chosen"] == "offset"
    assert abs(offset["coefficients"][0] + 0.1) < 0.01
    # survey compresses the metal-poor tail (floor-like curvature) -> nonlinear
    squashed = np.where(y < -2.0, -2.0 + 0.4 * (y + 2.0), y) + noise
    assert lo.model_selection(squashed, y)["chosen"] == "piecewise_linear"
    # a tiny overlap cannot support more than an offset
    small = lo.model_selection(y[:12] + 0.1 + noise[:12], y[:12])
    assert small["chosen"] in ("identity", "offset")
    assert "affine" not in small["cv_median_abs_residual"]


def test_floor_detection_and_metal_poor_gain():
    values = np.r_[np.full(50, -2.5), np.linspace(-2.4, 0.3, 5000)]
    floor = lo.floor_stats(values)
    assert floor["min"] == -2.5 and floor["pile_up"]
    old = pd.Series([-2.5, -1.0, -2.2], index=[1, 2, 3])
    new = pd.Series([-3.2, -1.1, -2.9], index=[1, 2, 4])
    gain = lo.metal_poor_gain(old, new)
    assert gain["overlap"] == 2 and gain["crosses_below_-2.5"] == 1
    assert gain["crosses_below_-3.0"] == 1 and gain["lowered_by_0p5_and_below_-2"] == 1


def _obs(source, tier, group, ids, feh):
    frame = ml.empty_frame(
        len(ids),
        source=source,
        tier=tier,
        release="r",
        pipeline="p",
        dependency_group=group,
    )
    frame["native_row"] = np.arange(len(ids))
    frame["source_id"] = ids
    frame["fe_h"] = feh
    frame["quality"] = True
    return ml.finalize(frame)


def test_run_uses_leave_own_group_out_reference_for_highres_sources():
    obs = pd.concat(
        [
            _obs("saga", "highres_reference", "lit", [1, 2, 3], [-3.0, -2.0, -1.0]),
            _obs("jina", "highres_reference", "lit", [1, 2], [-3.1, -2.1]),
            _obs("roed", "highres_reference", "roed", [1, 3], [-2.9, -1.1]),
            _obs("sspp", "survey", "sspp", [1, 2, 3, 4], [-2.7, -1.9, -0.9, -0.5]),
        ],
        ignore_index=True,
    )
    existing = {"lamost": pd.Series([-2.5, -1.2], index=[1, 3])}
    report = lo.run(obs, existing, log=lambda m: None)
    assert report["reference"]["stars"] == 3
    # saga/jina (same group) are compared only against roederer's 2 stars
    assert report["master_sources"]["saga"]["reference"]["overlap"] == 2
    assert report["master_sources"]["sspp"]["reference"]["overlap"] == 3
    gain = report["staged_surveys"]["lamost"]["metal_poor_gain"]
    assert gain["crosses_below_-2.5"] == 1


def test_reference_drops_republished_copies_of_the_tested_value():
    rows = pd.DataFrame(
        dict(
            source_id=[1, 1, 2],
            fe_h=[-3.00, -3.20, -2.00],
            source=["saga", "roed", "saga"],
            dependency_group=["lit", "roed", "lit"],
        )
    )
    tested = pd.Series([-3.00, -2.50], index=[1, 2])
    ref = lo.independent_reference(tested, rows)
    assert ref.loc[1] == -3.20  # saga's copy of the tested paper is dropped
    assert ref.loc[2] == -2.00
