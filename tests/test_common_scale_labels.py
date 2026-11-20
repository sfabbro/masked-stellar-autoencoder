import math

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from masked_stellar_autoencoder.training.anchored_survey_map import (
    FULL,
    IDENTITY,
    OFFSET,
    AnchoredSurveyMaps,
)
from masked_stellar_autoencoder.training.common_scale_labels import (
    build_common_scale_labels,
    combine_inverse_variance,
    frozen_core_reference,
    map_to_common_scale,
)

NAMES = ["anchor_seis", "big", "saga_mw", "bensby2014", "excluded", "other"]


def _maps(seed=0, scale=0.3):
    torch.manual_seed(seed)
    tier = np.array([[IDENTITY] * 3, [FULL] * 3, [OFFSET] * 3, [OFFSET] * 3,
                     [OFFSET] * 3, [OFFSET] * 3])  # fmt: skip
    core = np.zeros((6, 3), bool)
    core[[2, 3], 2] = True
    maps = AnchoredSurveyMaps(tier, core, np.ones((6, 3)))
    with torch.no_grad():
        for p in (maps.bias, maps.linear, maps.mix):
            p.uniform_(-scale, scale)
        maps.log_sigma.fill_(math.log(0.1))
    return maps.double().eval()


def test_frozen_core_reference_priority_and_core_only():
    rec = dict(
        survey_names=np.array(NAMES), parameters=np.array(["teff", "logg", "fe_h"]),
        row=np.array([0, 0, 1, 1, 2, 3]), survey=np.array([2, 3, 2, 5, 5, 2]),
        value=np.array([[0, 0, -2.0], [0, 0, -2.2], [0, 0, np.nan], [0, 0, -1.0],
                        [0, 0, -0.5], [0, 0, -3.0]]),
        error=np.full((6, 3), 0.1),
    )  # fmt: skip
    ref, err, src = frozen_core_reference(rec, ("saga_", "bensby"), n_rows=5)
    assert ref[0] == -2.2 and src[0] == 3  # bensby2014 before saga_mw
    assert np.isnan(ref[1]) and src[1] == -1  # NaN core record, non-core ignored
    assert np.isnan(ref[2])  # only a non-core survey
    assert ref[3] == -3.0 and src[3] == 2
    assert np.isnan(ref[4]) and err.shape == (5,)


def test_map_inversion_recovers_truth_and_head_point_evaluation():
    maps = _maps()
    rng = np.random.default_rng(1)
    R = 200
    x = rng.normal(size=(R, 3))
    s = torch.full((R,), 1)
    with torch.no_grad():
        y, jac = maps(torch.as_tensor(x), s)
    y = y.numpy()
    x0 = x + rng.normal(scale=0.5, size=(R, 3))  # a biased head prediction
    obs = np.ones((R, 3), bool)
    obs[:50, 0] = False  # Teff not reported by these records
    val = np.where(obs, y, np.nan)
    err = np.full((R, 3), 0.05)
    xs, var, conv = map_to_common_scale(maps, s.numpy(), val, err, obs, x0)
    assert conv.all()
    # Fully observed records invert exactly; partial ones hold Teff at the head.
    np.testing.assert_allclose(xs[50:], x[50:], atol=1e-8)
    assert np.isnan(xs[:50, 0]).all() and np.isnan(var[:50, 0]).all()
    with torch.no_grad():
        yp, _ = maps(
            torch.as_tensor(np.column_stack([x0[:50, 0], xs[:50, 1:]])), s[:50]
        )
    np.testing.assert_allclose(yp.numpy()[:, 1:], y[:50, 1:], atol=1e-8)
    # Variance: J^-1 diag(e^2 + sigma^2) J^-T at the solution.
    j = torch.linalg.inv(jac[60]).numpy()
    expect = (j * j) @ np.full(3, 0.05**2 + 0.1**2)
    np.testing.assert_allclose(var[60], expect, rtol=1e-6)
    # iterations=0: the map evaluated at the head's prediction.
    x0s, _, _ = map_to_common_scale(maps, s.numpy(), val, err, obs, x0, iterations=0)
    with torch.no_grad():
        y0, _ = maps(torch.as_tensor(x0), s)
    np.testing.assert_allclose(x0s[50:], (y - (y0.numpy() - x0))[50:], atol=1e-10)
    assert np.abs(x0s[50:] - x[50:]).max() > 1e-3  # the head's bias leaks in


def test_offset_and_identity_maps_need_no_start():
    maps = _maps()
    R = 5
    val = np.tile([0.3, -0.2, 1.1], (R, 1))
    obs = np.ones((R, 3), bool)
    err = np.zeros((R, 3))
    x0 = np.random.default_rng(0).normal(size=(R, 3))
    xi, vi, _ = map_to_common_scale(maps, np.zeros(R, int), val, err, obs, x0)
    np.testing.assert_allclose(xi, val)  # identity
    np.testing.assert_allclose(vi, 0.1**2)
    xo, _, _ = map_to_common_scale(maps, np.full(R, 2), val, err, obs, x0)
    bias = maps.coefficients()[0][2].detach().numpy()
    np.testing.assert_allclose(xo, val - bias, atol=1e-12)


def test_combine_inverse_variance():
    row = np.array([0, 0, 2])
    x = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 7.0]])
    var = np.array([[1.0, np.nan], [3.0, 4.0], [0.5, 2.0]])
    mean, err, n = combine_inverse_variance(row, x, var, 4)
    np.testing.assert_allclose(mean[0], [(1 + 1) / (1 + 1 / 3), 2.0])
    np.testing.assert_allclose(err[0], [1 / math.sqrt(1 + 1 / 3), 2.0])
    assert n[0].tolist() == [2, 1] and n[1].tolist() == [0, 0]
    assert np.isnan(mean[1]).all() and np.isnan(mean[3]).all()
    np.testing.assert_allclose(mean[2], [5.0, 7.0])


def _state_and_records(maps):
    names = ["teff", "logg", "fe_h", "ebv_star"]
    scalers = []
    for mu, sd in ((5000.0, 500.0), (3.0, 1.0), (-0.5, 0.5), (0.05, 0.03)):
        sc = StandardScaler().fit(np.array([[mu - sd], [mu + sd]]))
        scalers.append(sc)
    state = dict(
        label_names=names, label_scalers=scalers, survey_names=NAMES,
        map_tier=maps.tier.numpy(), map_core=maps.core.numpy(),
        map_counts=np.ones((6, 3)), map_kind="anchored",
        survey_maps_state_dict={k: v.float() for k, v in maps.state_dict().items()},
    )  # fmt: skip
    nan = np.nan
    #       row survey  teff     logg  fe_h   (physical)
    recs = [
        (0, 0, nan, 4.4, nan),  # seismic anchor
        (0, 3, 5800, 4.3, -2.0),  # bensby (core)
        (0, 1, 5700, 4.2, -1.9),  # big survey
        (1, 1, 4800, 2.5, -2.9),  # big survey, censored [Fe/H] (< -2.2)
        (1, 4, 4900, 2.6, 5.0),  # excluded survey
        (1, 5, 4850, 2.4, -2.6),  # other
        (2, 2, nan, nan, -1.0),  # saga (core)
    ]  # fmt: skip
    rec = dict(
        survey_names=np.array(NAMES), survey_is_anchor=np.array([1, 0, 0, 0, 0, 0], bool),
        parameters=np.array(["teff", "logg", "fe_h", "alpha"]),
        row=np.array([r[0] for r in recs]), survey=np.array([r[1] for r in recs]),
        value=np.array([[r[2], r[3], r[4], nan] for r in recs]),
        error=np.array([[50.0, 0.05, 0.05, nan]] * len(recs)),
    )  # fmt: skip
    config = dict(multisurvey=dict(
        fe_h_core=["saga_", "bensby"], censor={"big": {"fe_h": -2.2}},
        exclude_surveys=["excluded"],
    ))  # fmt: skip
    return config, state, rec


def test_build_labels_excludes_censored_and_combines():
    maps = _maps(scale=0.1)
    config, state, rec = _state_and_records(maps)
    head = np.zeros((3, 3))
    lab = build_common_scale_labels(config, state, rec, head, log=lambda m: None)
    loaded = _maps(scale=0.1)
    sd = np.array([500.0, 1.0, 0.5])
    mu = np.array([5000.0, 3.0, -0.5])
    # Row 1: the censored big-survey [Fe/H] and the excluded survey are ignored,
    # so [Fe/H] comes from "other" alone (offset map).
    bias = loaded.coefficients()[0].detach().numpy()
    expect = (-2.6 - mu[2]) / sd[2] - bias[5, 2]
    assert lab["n_feh"][1] == 1
    np.testing.assert_allclose(lab["fe_h"][1], expect * sd[2] + mu[2], atol=1e-5)
    assert lab["n_teff"][1] == 2 and lab["n_logg"][1] == 2
    # Row 0: three log g records combined by inverse variance (anchor tightest).
    assert lab["n_logg"][0] == 3 and lab["n_feh"][0] == 2
    assert lab["e_logg"][0] < math.sqrt(0.05**2 + 0.1**2)
    # Frozen core: bensby for row 0 (priority), saga for row 2, none for row 1.
    assert lab["has_core_feh"].tolist() == [True, False, True]
    np.testing.assert_allclose(
        lab["fe_h_core"][2], ((-1.0 - mu[2]) / sd[2] - bias[2, 2]) * sd[2] + mu[2],
        atol=1e-5,
    )  # fmt: skip
    assert lab["fe_h_ref"][0] == -2.0 and lab["fe_h_ref_survey"][0] == 3
    np.testing.assert_allclose(
        lab["e_fe_h_core"][2], math.sqrt(0.05**2 + (0.1 * sd[2]) ** 2), rtol=1e-4
    )
    # Row 2 has only a core [Fe/H]: no Teff / log g.
    assert np.isnan(lab["teff"][2]) and lab["n_teff"][2] == 0


def test_unmapped_survey_fails_closed():
    maps = _maps()
    config, state, rec = _state_and_records(maps)
    state["survey_names"] = NAMES[:-1] + ["renamed"]
    with pytest.raises(ValueError, match="without a map"):
        build_common_scale_labels(config, state, rec, np.zeros((3, 3)),
                                  log=lambda m: None)  # fmt: skip
