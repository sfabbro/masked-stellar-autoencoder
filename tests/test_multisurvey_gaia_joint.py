import hashlib

import h5py
import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
    GaiaNativeObservationMap,
    build_gaia_records,
)
from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
    masked_energy_score,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, bad_assoc=False, bad_mh=False):
    a = 2**53 + 100
    sidecar = tmp_path / "sidecar.h5"
    joined = tmp_path / "join.h5"
    ids = np.array([a + 2, a, a + 1], dtype="i8")
    with h5py.File(sidecar, "w") as f:
        g = f.create_group("native_observations/gaia_dr3_gspspec")
        g["source_id"] = ids
        g["source_row"] = [20, 21, 22]
        g["value"] = np.array(
            [[5000.0, 2.0, np.nan], [5100.0, 3.0, np.nan], [5200.0, 4.0, np.nan]]
        )
        g["observed"] = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]], dtype="?")
        g["percentile16"] = np.array(
            [[4900, 1.9, np.nan], [5000, 2.9, np.nan], [5100, 3.9, np.nan]]
        )
        g["percentile84"] = np.array(
            [[5100, 2.1, np.nan], [5200, 3.1, np.nan], [5300, 4.1, np.nan]]
        )
        for name, values in [("mh", [-0.4, -0.8, -0.2]), ("alphafe", [0.2, 0.1, 0.05])]:
            g[name + "_gspspec"] = values
            g[name + "_gspspec_lower"] = np.array(values) - 0.1
            g[name + "_gspspec_upper"] = np.array(values) + 0.1
        g["selected_observed_mh"] = [1, 1, 1]
        g["selected_observed_alpha"] = [1, 0, 1]
    side_sha = digest(sidecar)
    with h5py.File(joined, "w") as f:
        f.attrs["schema"] = "gaia-gspspec-fem-exact-join-v1"
        f.attrs["sidecar_sha256"] = side_sha
        f["source_id"] = np.array([a, a + 2], dtype="i8")
        f["sidecar_matched"] = [1, 1]
        f["sidecar_row"] = [0 if bad_assoc else 1, 0]
        f["mh_gspspec"] = [-0.6 if bad_mh else -0.8, -0.4]
        f["fem_gspspec"] = [0.15, 0.25]
        f["fem_gspspec_lower"] = [0.1, 0.2]
        f["fem_gspspec_upper"] = [0.2, 0.3]
        f["fe_upper_limit_flag"] = [ord("0"), ord("1")]
        f["fe_uncertainty_flag"] = [ord("1"), ord("0")]
        f["provisional_derived_feh_candidate"] = [1, 0]
        f["fe_detection_flag_zero"] = [1, 0]
    return sidecar, joined, side_sha, digest(joined), a


def test_unsorted_exact_ids_and_native_dependent_labels(tmp_path):
    side, joined, sh, jh, a = fixture(tmp_path)
    ids = np.array([a + 2, a + 999, a, a + 1], dtype="i8")
    r = build_gaia_records(side, joined, ids, sidecar_sha256=sh, joined_sha256=jh)
    np.testing.assert_array_equal(r["source_matched"], [1, 0, 1, 1])
    np.testing.assert_array_equal(r["source_row"], [20, -1, 21, 22])
    np.testing.assert_array_equal(
        r["observed"],
        [[1, 1, 1, 1, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 1], [0, 1, 1, 1, 0]],
    )
    assert r["value"][0, 4] == 0.25 and r["fe_upper_limit_flag_ascii"][0] == ord("1")
    assert r["value"][2, 2] == -0.8 and r["value"][2, 4] == 0.15
    assert np.isnan(r["error"]).all() and not r["error_known"].any()
    assert r["fields"] == (
        "teff_gspspec",
        "logg_gspspec",
        "mh_gspspec",
        "alphafe_gspspec",
        "fem_gspspec",
    )
    assert "derived" in r["scope"] and r["dependency_id"] == "gaia-dr3-gspspec-rvs"


def test_join_mismatch_fails_closed(tmp_path):
    for bad in ("bad_assoc", "bad_mh"):
        sub = tmp_path / bad
        sub.mkdir()
        side, joined, sh, jh, a = fixture(sub, **{bad: True})
        with pytest.raises(ValueError):
            build_gaia_records(
                side,
                joined,
                np.array([a], dtype="i8"),
                sidecar_sha256=sh,
                joined_sha256=jh,
            )
    side, joined, sh, jh, a = fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA256"):
        build_gaia_records(
            side,
            joined,
            np.array([a], dtype="i8"),
            sidecar_sha256="0" * 64,
            joined_sha256=jh,
        )
    with pytest.raises(ValueError, match="64-bit"):
        build_gaia_records(
            side, joined, np.array([float(a)]), sidecar_sha256=sh, joined_sha256=jh
        )
    with pytest.raises(ValueError, match="unique"):
        build_gaia_records(
            side,
            joined,
            np.array([a, a], dtype="i8"),
            sidecar_sha256=sh,
            joined_sha256=jh,
        )


def test_shared_sample_observation_map_identity_and_gradient():
    mapper = GaiaNativeObservationMap(
        [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0], [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    )
    latent = torch.tensor(
        [
            [[5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]],
            [[5200.0, 4.0, -0.5, 0.3, 6.0, 1200.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    with torch.no_grad():
        mapper.correction.bias.copy_(torch.tensor([0.2, -0.1, 0.05]))
    out = mapper(latent)
    torch.testing.assert_close(out[..., 2], latent[..., 2] + 0.2)
    torch.testing.assert_close(out[..., 2] + out[..., 4], latent[..., 2] - 0.1)
    assert out.shape == (2, 1, 5)
    out.square().sum().backward()
    assert (
        torch.isfinite(latent.grad).all()
        and torch.isfinite(mapper.correction.weight.grad).all()
    )


def test_mh_fem_scored_together_with_one_joint_record():
    mapper = GaiaNativeObservationMap(
        [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0], [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    )
    x = torch.tensor(
        [
            [[5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]],
            [[5100.0, 3.2, -0.8, 0.3, 5.0, 1000.0]],
        ],
        dtype=torch.float64,
    )
    draw = mapper(x)
    target = torch.tensor([[5000.0, 3.0, -0.9, 0.2, 0.1]], dtype=torch.float64)
    mask = torch.tensor([[False, False, True, False, True]])
    score = masked_energy_score(draw, target, mask, scale=[1, 1, 1, 1, 1])
    projected = draw[:, 0, [2, 4]].detach().numpy()
    actual = target[0, [2, 4]].numpy()
    reference = (
        np.linalg.norm(projected - actual, axis=1).mean()
        - np.linalg.norm(projected[0] - projected[1]) / 2
    )
    np.testing.assert_allclose(float(score.detach()), reference, atol=1e-12)


def test_joint_noise_approximation_requires_explicit_consent_and_floors(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
        gaia_native_energy_score,
    )

    side, joined, sh, jh, a = fixture(tmp_path)
    r = build_gaia_records(
        side, joined, np.array([a], dtype="i8"), sidecar_sha256=sh, joined_sha256=jh
    )
    mapper = GaiaNativeObservationMap(
        [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0], [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    )
    latent = torch.tensor(
        [
            [[5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]],
            [[5100.0, 3.2, -0.8, 0.3, 5.0, 1000.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    args = dict(
        scale=[500.0, 1.0, 1.0, 1.0, 1.0],
        noise_generator=torch.Generator().manual_seed(1),
        unknown_error_floor=[30.0, 0.1, 0.1, 0.1, 0.1],
    )
    with pytest.raises(ValueError, match="explicit"):
        gaia_native_energy_score(
            latent, mapper, r, assume_independent_gaussian_errors=False, **args
        )
    score = gaia_native_energy_score(
        latent, mapper, r, assume_independent_gaussian_errors=True, **args
    )
    score.backward()
    assert torch.isfinite(score) and torch.isfinite(latent.grad).all()
    args["unknown_error_floor"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="positive"):
        gaia_native_energy_score(
            latent, mapper, r, assume_independent_gaussian_errors=True, **args
        )


def test_refuses_mislabelled_derived_iron_as_independent(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
        gaia_native_energy_score,
    )

    side, joined, sh, jh, a = fixture(tmp_path)
    record = build_gaia_records(
        side, joined, np.array([a], dtype="i8"), sidecar_sha256=sh, joined_sha256=jh
    )
    record["fields"] = (
        "teff_gspspec",
        "logg_gspspec",
        "mh_gspspec",
        "alphafe_gspspec",
        "derived_feh_native",
    )
    mapper = GaiaNativeObservationMap(
        [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0], [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    )
    latent = torch.tensor(
        [
            [[5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]],
            [[5100.0, 3.2, -0.8, 0.3, 5.0, 1000.0]],
        ],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="field order"):
        gaia_native_energy_score(
            latent,
            mapper,
            record,
            scale=[500, 1, 1, 1, 1],
            noise_generator=torch.Generator().manual_seed(1),
            unknown_error_floor=[30, 0.1, 0.1, 0.1, 0.1],
            assume_independent_gaussian_errors=True,
        )


def test_three_target_gauge_and_shared_draw_gradients():
    from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
        ThreeTargetSurveyMap,
    )

    center = [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]
    scale = [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    weights = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
    mapper = ThreeTargetSurveyMap(center, scale, weights)
    latent = torch.tensor(
        [
            [[5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]],
            [[5100.0, 3.2, -0.8, 0.3, 5.0, 1200.0]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    with torch.no_grad():
        mapper.correction.bias.copy_(
            torch.tensor([0.2, 0.3, 0.4, -0.4, -0.2, 0.5], dtype=torch.float64)
        )
    out = mapper(latent)
    assert out.shape == (2, 1, 2, 3)
    weighted_mean = (out * weights).sum(-2) / weights.sum(0)
    torch.testing.assert_close(weighted_mean, latent[..., :3])
    out.square().sum().backward()
    assert (
        torch.isfinite(latent.grad).all()
        and torch.isfinite(mapper.correction.weight.grad).all()
    )


def test_joint_energy_mixed_sparse_records_and_implicit_noise_rejected():
    from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
        DEPENDENCY,
        FIELDS,
        GaiaNativeObservationMap,
        ThreeTargetSurveyMap,
        joint_multisurvey_energy_score,
    )

    center = [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0]
    scale = [500.0, 1.0, 1.0, 0.2, 3.0, 500.0]
    survey_map = ThreeTargetSurveyMap(center, scale, [[1.0, 1.0, 1.0]])
    gaia_map = GaiaNativeObservationMap(center, scale)
    latent = torch.tensor(
        [
            [
                [5000.0, 3.0, -1.0, 0.2, 5.0, 1000.0],
                [5100.0, 3.0, -0.8, 0.1, 6.0, 900.0],
                [4900.0, 2.0, -2.0, 0.4, 7.0, 1200.0],
            ],
            [
                [5100.0, 3.1, -0.9, 0.3, 5.0, 1100.0],
                [5000.0, 3.1, -1.0, 0.2, 6.0, 1000.0],
                [5000.0, 2.2, -1.8, 0.3, 7.0, 1300.0],
            ],
            [
                [4900.0, 2.9, -1.1, 0.1, 5.0, 900.0],
                [5200.0, 2.9, -0.7, 0.0, 6.0, 800.0],
                [4800.0, 1.9, -2.2, 0.5, 7.0, 1100.0],
            ],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    obs = torch.tensor([[[1, 0, 1]], [[0, 0, 0]], [[0, 1, 0]]], dtype=torch.bool)
    base = {
        "source_id": torch.tensor([101, 102, 103], dtype=torch.int64),
        "observed": obs,
        "value": torch.tensor(
            [[[5000.0, 0.0, -1.0]], [[0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0]]],
            dtype=torch.float64,
        ),
        "error": torch.full((3, 1, 3), float("nan"), dtype=torch.float64),
        "error_known": torch.zeros((3, 1, 3), dtype=torch.bool),
        "source_index": torch.tensor([[0], [-1], [-1]], dtype=torch.int64),
        "is_anchor": torch.tensor([[False], [False], [True]]),
        "source_registry": ["survey"],
    }
    gaia = {
        "source_id": np.array([101, 102, 103], dtype=np.int64),
        "fields": FIELDS,
        "dependency_id": DEPENDENCY,
        "value": np.array(
            [[np.nan] * 5, [np.nan, 3.0, np.nan, np.nan, np.nan], [np.nan] * 5]
        ),
        "observed": np.array([[0] * 5, [0, 1, 0, 0, 0], [0] * 5], dtype=bool),
        "error": np.full((3, 5), np.nan),
        "error_known": np.zeros((3, 5), dtype=bool),
    }
    args = dict(
        gaia_map=gaia_map,
        survey_map=survey_map,
        source_floor=[[50.0, 0.2, 0.2]],
        anchor_floor=[50.0, 0.2, 0.2],
        gaia_floor=[50.0, 0.2, 0.2, 0.2, 0.2],
        survey_scale=[500.0, 1.0, 1.0],
        gaia_scale=[500.0, 1.0, 1.0, 1.0, 1.0],
        noise_generator=torch.Generator().manual_seed(3),
    )
    with pytest.raises(ValueError, match="[Ee]xplicit"):
        joint_multisurvey_energy_score(
            latent,
            {"base": base, "gaia": gaia},
            assume_independent_gaussian_errors=False,
            **args,
        )
    score = joint_multisurvey_energy_score(
        latent,
        {"base": base, "gaia": gaia},
        assume_independent_gaussian_errors=True,
        **args,
    )
    assert torch.isfinite(score)
    score.backward()
    assert torch.isfinite(latent.grad).all()
    assert all(torch.linalg.vector_norm(latent.grad[:, i, :]) > 0 for i in range(3))
    assert torch.isfinite(gaia_map.correction.weight.grad).all()
    assert torch.isfinite(survey_map.correction.weight.grad).all()
    with pytest.raises(ValueError, match="identities"):
        joint_multisurvey_energy_score(
            latent,
            {"base": base, "gaia": {**gaia, "source_id": np.array([101, 103, 102])}},
            assume_independent_gaussian_errors=True,
            **args,
        )


def test_fit_batch_replaces_gaia_without_touching_other_surveys(tmp_path):
    from masked_stellar_autoencoder.training.multisurvey_data import SparseSurveyDataset
    from masked_stellar_autoencoder.training.multisurvey_gaia_joint import (
        GaiaFitView,
        prepare_joint_fit_batch,
    )
    from masked_stellar_autoencoder.training.multisurvey_training import FitIdentityGate

    side, joined, sh, jh, a = fixture(tmp_path)
    source = tmp_path / "fit.h5"
    with h5py.File(source, "w") as f:
        f["source_id"] = np.array([a, a + 1, a + 2, a + 3], dtype="i8")
        f["input"] = np.ones((4, 2))
        f["role"] = np.array([b"fit", b"fit", b"fit", b"tune"])
        f["primary_input_eligible"] = np.ones(4, dtype=bool)
        root = f.create_group("native_observations")
        g = root.create_group("gaia_dr3_gspspec")
        g.attrs["source_fields"] = ["teff_gspspec", "logg_gspspec", ""]
        g["source_id"] = np.array([a, a + 1, a + 2], dtype="i8")
        g["source_row"] = [21, 22, 20]
        g["value"] = np.array(
            [[5100.0, 3.0, np.nan], [5200.0, 4.0, np.nan], [5000.0, 2.0, np.nan]]
        )
        g["observed"] = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=bool)
        g["error"] = np.full((3, 3), np.nan)
        g = root.create_group("other")
        g.attrs["source_fields"] = ["teff", "logg", "fe_h"]
        g["source_id"] = np.array([a, a + 2], dtype="i8")
        g["source_row"] = [4, 5]
        g["value"] = np.array([[5000.0, 3.0, -1.0], [5000.0, 2.0, -0.4]])
        g["observed"] = np.ones((2, 3), dtype=bool)
        g["error"] = np.full((2, 3), 0.1)
    fields = lambda teff, logg, fe: {
        "teff": {"field": teff, "quantity": "teff", "unit": "K"},
        "logg": {"field": logg, "quantity": "logg", "unit": "dex(cm/s2)"},
        "fe_h": None
        if fe is None
        else {"field": fe, "quantity": "fe_h", "unit": "dex"},
    }
    registry = {
        "gaia_dr3_gspspec": {
            "release": "DR3",
            "pipeline": "GSP-Spec",
            "dependency_id": "gaia-dr3-gspspec-rvs",
            "targets": fields("teff_gspspec", "logg_gspspec", None),
        },
        "other": {
            "release": "DR",
            "pipeline": "spec",
            "dependency_id": "other-spectrum",
            "targets": fields("teff", "logg", "fe_h"),
        },
    }
    source_sha = digest(source)
    dataset = SparseSurveyDataset(
        source, registry, role="fit", expected_source_sha256=source_sha
    )
    ids_sha = hashlib.sha256(
        np.asarray(dataset.ids[dataset.star_rows], dtype="<i8").tobytes()
    ).hexdigest()
    gate = FitIdentityGate(dataset, source_binding=source_sha, fit_ids_sha256=ids_sha)
    view = GaiaFitView(gate, side, joined, sidecar_sha256=sh, joined_sha256=jh)
    batch = prepare_joint_fit_batch(gate, view, np.array([a, a + 2], dtype="i8"))
    assert batch["base"]["source_id"].tolist() == [a, a + 2]
    assert batch["base"]["observed"].sum().item() == 6
    assert batch["original"]["observed"].sum().item() == 9
    assert batch["gaia"]["observed"].tolist() == [[1, 0, 1, 0, 1], [1, 1, 1, 1, 0]]
    with pytest.raises(ValueError, match="outside"):
        prepare_joint_fit_batch(gate, view, np.array([a + 3], dtype="i8"))
    dataset.close()
