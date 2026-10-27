"""Small synthetic checks, no native MSA weights or models."""

import pytest
import torch
from torch import nn

from masked_stellar_autoencoder.training.multisurvey_model import (
    CanonicalAtmosphereModel,
    CanonicalGaussianHead,
)
from masked_stellar_autoencoder.training.multisurvey_model import (
    pack_multisurvey_observations as pack_impl,
)

calibration = pytest.importorskip(
    "torchregress.calibration.multisource",
    reason="new torchregress revision not installed",
)
losses = pytest.importorskip(
    "torchregress.losses.multisource", reason="new torchregress revision not installed"
)
CommonScaleSplineHead = calibration.CommonScaleSplineHead
RelativeSurveySplineHead = calibration.RelativeSurveySplineHead
MarginalizedMultiSourceGaussianNLLLoss = losses.MarginalizedMultiSourceGaussianNLLLoss


def pack_multisurvey_observations(*args, **kwargs):
    kwargs.setdefault("expected_registry_sha256", "a" * 64)
    kwargs.setdefault("expected_source_registry", ("a", "b", "anchor"))
    return pack_impl(*args, **kwargs)


def spline_setup():
    k = torch.tensor(
        [-3.0, -3.0, -3.0, -3.0, 0.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.double
    )
    center = torch.tensor([5000.0, 2.5, -1.0], dtype=torch.double)
    scale = torch.tensor([1000.0, 1.0, 1.0], dtype=torch.double)
    weights = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.double
    )
    rel = RelativeSurveySplineHead(
        [k, k, k],
        weights,
        input_center=center,
        input_scale=scale,
        output_scale=torch.tensor([50.0, 0.1, 0.1]),
    ).double()
    common = CommonScaleSplineHead(
        [k, k, k],
        input_center=center,
        input_scale=scale,
        output_scale=torch.tensor([50.0, 0.1, 0.1]),
    ).double()
    return center, scale, rel, common


def batch():
    value = torch.tensor(
        [
            [[5100.0, 2.2, -0.5], [5200.0, 2.1, -0.4]],
            [[4800.0, 3.0, -1.4], [4900.0, 2.8, -1.3]],
        ],
        dtype=torch.double,
    )
    error = torch.tensor([50.0, 0.1, 0.1], dtype=torch.double).expand(2, 2, 3).clone()
    observed = torch.tensor(
        [[[1, 1, 1], [1, 1, 0]], [[1, 0, 1], [1, 1, 0]]], dtype=torch.bool
    )
    return {
        "value": value,
        "error": error,
        "observed": observed,
        "error_known": observed.clone(),
        "source_index": torch.tensor([[0, 1], [1, -1]]),
        "is_anchor": torch.tensor([[False, False], [False, True]]),
        "target_names": ("teff", "logg", "fe_h"),
        "source_registry": ("a", "b", "anchor"),
        "registry_sha256": "a" * 64,
        "source_binding": "b" * 64,
        "provenance": [
            [
                {
                    "release": "r1",
                    "pipeline": "p",
                    "dependency_id": f"{i}-{j}",
                    "target_definitions": {
                        name: {"field": name, "quantity": name, "unit": unit}
                        for name, unit in (
                            ("teff", "K"),
                            ("logg", "dex(cm/s2)"),
                            ("fe_h", "dex"),
                        )
                    },
                }
                for j in range(2)
            ]
            for i in range(2)
        ],
    }


def test_full_pipeline_gradients_and_unchanged_reconstruction():
    torch.manual_seed(44)
    center, scale, relative, common = spline_setup()

    class ToyMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 5)
            self.decoder = nn.Linear(5, 4)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    mae = ToyMAE().double()
    head = CanonicalGaussianHead(5, center, scale, width=8).double()
    model = CanonicalAtmosphereModel(mae, head)
    x = torch.randn(2, 4, dtype=torch.double)
    torch.testing.assert_close(model.reconstruct(x), mae(x))
    mean, lower = model(x)
    assert lower.shape == (2, 3, 3) and (lower.diagonal(dim1=-2, dim2=-1) > 0).all()
    assert torch.count_nonzero(lower[:, 1, 0]) > 0
    floor = torch.tensor(
        [[5.0, 0.02, 0.02], [5.0, 0.02, 0.02], [5.0, 0.02, 0.02]], dtype=torch.double
    )
    packed = pack_multisurvey_observations(
        mean,
        batch(),
        relative,
        floor,
        common_head=common,
        assume_diagonal_catalog_covariance=True,
    )
    loss = MarginalizedMultiSourceGaussianNLLLoss(method="information")(
        mean, lower, packed
    )
    loss.backward()
    for parameters in (
        mae.encoder.parameters(),
        head.parameters(),
        relative.parameters(),
        common.parameters(),
    ):
        grad = [p.grad for p in parameters if p.grad is not None]
        assert (
            grad
            and all(torch.isfinite(g).all() for g in grad)
            and sum(g.abs().sum() for g in grad) > 0
        )
    # The anchor is an identity map, not a second application of survey bias.
    torch.testing.assert_close(packed.mapped_mean[1, 3:6], mean[1])
    torch.testing.assert_close(
        packed.jacobian[1, 3:6], torch.eye(3, dtype=torch.double)
    )


def test_unknown_errors_and_diagonal_assumption_are_explicit():
    center, _, relative, _ = spline_setup()
    mean = center[None].repeat(2, 1)
    b = batch()
    floor = torch.zeros(3, 3, dtype=torch.double)
    with pytest.raises(ValueError, match="diagonal"):
        pack_multisurvey_observations(mean, b, relative, floor)
    b["error_known"][0, 0, 0] = False
    b["error"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="positive floor"):
        pack_multisurvey_observations(
            mean, b, relative, floor, assume_diagonal_catalog_covariance=True
        )
    floor[0, 0] = 20.0
    packed = pack_multisurvey_observations(
        mean, b, relative, floor, assume_diagonal_catalog_covariance=True
    )
    assert packed.measurement_covariance[0, 0, 0] == 400


def test_unavailable_survey_target_rejected():
    center, _, relative, _ = spline_setup()
    b = batch()
    b["is_anchor"][1, 1] = False  # zero weight was intentionally reserved for an anchor
    b["source_index"][1, 1] = 2
    with pytest.raises(ValueError, match="not declared comparable"):
        pack_multisurvey_observations(
            center[None].repeat(2, 1),
            b,
            relative,
            torch.ones(3, 3),
            assume_diagonal_catalog_covariance=True,
        )


def test_joint_covariance_must_agree_with_catalog_error_diagonal():
    center, _, relative, _ = spline_setup()
    mean = center[None].repeat(2, 1)
    b = batch()
    b["joint_covariance"] = torch.diag_embed(
        torch.ones(2, 6, dtype=torch.double)
    ).reshape(2, 2, 3, 2, 3)
    with pytest.raises(ValueError, match="diagonal differs"):
        pack_multisurvey_observations(mean, b, relative, torch.ones(3, 3))


def test_incompatible_metallicity_is_rejected_before_identity_mapping():
    center, _, relative, _ = spline_setup()
    b = batch()
    b["provenance"][0][0]["target_definitions"]["fe_h"]["field"] = "M_H"
    with pytest.raises(ValueError, match="not be treated"):
        pack_multisurvey_observations(
            center[None].repeat(2, 1),
            b,
            relative,
            torch.ones(3, 3),
            assume_diagonal_catalog_covariance=True,
        )


def test_joint_objective_preserves_separate_reconstruction_and_fit_roles():
    from masked_stellar_autoencoder.training.multisurvey_model import (
        multisurvey_objective,
    )

    torch.manual_seed(5)
    center, scale, relative, _ = spline_setup()

    class ToyMAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 5)
            self.decoder = nn.Linear(5, 4)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = CanonicalAtmosphereModel(
        ToyMAE().double(), CanonicalGaussianHead(5, center, scale, width=8).double()
    )
    b = batch()
    b["input"] = torch.randn(2, 4, dtype=torch.double)
    b["role"] = ["fit", "fit"]
    reconstruction = {
        "masked_input": torch.zeros(3, 4, dtype=torch.double),
        "target": torch.ones(3, 4, dtype=torch.double),
        "mask": torch.tensor([[1, 0, 0, 1]] * 3, dtype=torch.bool),
        "role": ["fit"] * 3,
        "source_id": torch.tensor([11, 12, 13]),
        "source_binding": "c" * 64,
    }
    out = multisurvey_objective(
        model,
        b,
        relative,
        torch.ones(3, 3),
        reconstruction=reconstruction,
        reconstruction_weight=0.03,
        assume_diagonal_catalog_covariance=True,
        expected_registry_sha256="a" * 64,
        expected_source_registry=("a", "b", "anchor"),
        expected_training_source_binding="b" * 64,
        expected_reconstruction_source_binding="c" * 64,
    )
    torch.testing.assert_close(
        out["total"], out["supervised"] + 0.03 * out["reconstruction"]
    )
    out["total"].backward()
    assert model.mae.encoder.weight.grad.abs().sum() > 0
    assert model.mae.decoder.weight.grad.abs().sum() > 0
    reconstruction["role"][1] = "tune"
    with pytest.raises(ValueError, match="fit-role"):
        multisurvey_objective(
            model,
            b,
            relative,
            torch.ones(3, 3),
            reconstruction=reconstruction,
            reconstruction_weight=0.03,
            assume_diagonal_catalog_covariance=True,
            expected_registry_sha256="a" * 64,
            expected_source_registry=("a", "b", "anchor"),
            expected_training_source_binding="b" * 64,
            expected_reconstruction_source_binding="c" * 64,
        )
    reconstruction["role"][1] = "fit"
    reconstruction["source_binding"] = "e" * 64
    with pytest.raises(ValueError, match="pinned source"):
        multisurvey_objective(
            model,
            b,
            relative,
            torch.ones(3, 3),
            reconstruction=reconstruction,
            reconstruction_weight=0.03,
            assume_diagonal_catalog_covariance=True,
            expected_registry_sha256="a" * 64,
            expected_source_registry=("a", "b", "anchor"),
            expected_training_source_binding="b" * 64,
            expected_reconstruction_source_binding="c" * 64,
        )
    b["role"][1] = "tune"
    with pytest.raises(ValueError, match="fit-role"):
        multisurvey_objective(
            model,
            b,
            relative,
            torch.ones(3, 3),
            assume_diagonal_catalog_covariance=True,
            expected_registry_sha256="a" * 64,
            expected_source_registry=("a", "b", "anchor"),
            expected_training_source_binding="b" * 64,
        )


def test_common_scale_cannot_train_without_audited_anchor_contract():
    from masked_stellar_autoencoder.training.multisurvey_model import (
        multisurvey_objective,
    )

    center, scale, relative, common = spline_setup()

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(4, 5)

        def forward(self, x):
            return x

    model = CanonicalAtmosphereModel(
        Dummy().double(), CanonicalGaussianHead(5, center, scale).double()
    )
    b = batch()
    b["input"] = torch.zeros(2, 4, dtype=torch.double)
    b["role"] = ["fit", "fit"]
    with pytest.raises(ValueError, match="anchor contract"):
        multisurvey_objective(
            model,
            b,
            relative,
            torch.ones(3, 3),
            common_head=common,
            assume_diagonal_catalog_covariance=True,
            expected_registry_sha256="a" * 64,
            expected_source_registry=("a", "b", "anchor"),
            expected_training_source_binding="b" * 64,
        )


def test_common_anchor_contract_must_match_pinned_content_and_training_split():
    import hashlib
    import json

    from masked_stellar_autoencoder.training.multisurvey_model import (
        multisurvey_objective,
    )

    center, scale, relative, common = spline_setup()
    mae = nn.Module()
    mae.encoder = nn.Linear(4, 5).double()
    model = CanonicalAtmosphereModel(
        mae, CanonicalGaussianHead(5, center, scale).double()
    )
    b = batch()
    b["input"] = torch.zeros(2, 4, dtype=torch.double)
    b["role"] = ["fit", "fit"]
    contract = {
        "independence_audited": True,
        "training_anchor_sha256": "b" * 64,
        "independence_evidence_sha256": "c" * 64,
        "training_split_sha256": "d" * 64,
        "training_source_binding": "b" * 64,
        "registry_sha256": "a" * 64,
        "covered_targets": ["logg"],
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    kwargs = dict(
        common_head=common,
        anchor_contract=contract,
        expected_anchor_contract_sha256=digest,
        expected_training_split_sha256="d" * 64,
        calibration_regularization_weight=0.01,
        assume_diagonal_catalog_covariance=True,
        expected_registry_sha256="a" * 64,
        expected_source_registry=("a", "b", "anchor"),
        expected_training_source_binding="b" * 64,
    )
    assert torch.isfinite(
        multisurvey_objective(model, b, relative, torch.ones(3, 3), **kwargs)["total"]
    )
    contract["covered_targets"] = ["teff"]
    with pytest.raises(ValueError, match="digest"):
        multisurvey_objective(model, b, relative, torch.ones(3, 3), **kwargs)
    contract["training_anchor_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="anchor contract"):
        multisurvey_objective(model, b, relative, torch.ones(3, 3), **kwargs)


def test_batch_registry_mismatch_is_rejected():
    center, _, relative, _ = spline_setup()
    b = batch()
    b["registry_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="registry bindings"):
        pack_multisurvey_observations(
            center[None].repeat(2, 1),
            b,
            relative,
            torch.ones(3, 3),
            assume_diagonal_catalog_covariance=True,
        )
