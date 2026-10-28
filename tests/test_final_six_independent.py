"""Synthetic independent six-target numerics and source/CP invariants."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/xp/monitor"))
from verify_final_six_heads import (
    compare,
    finite_quantile,
    independent_coordinates,
    independent_head,
)

from masked_stellar_autoencoder.training.final_six_recipes import SparseSixHead


@pytest.mark.parametrize("kind", ["gaussian", "mdn", "factorized_nsf"])
def test_independent_native_moments_sparse_noise_and_joint_scores(kind):
    torch.manual_seed(814)
    head = SparseSixHead(kind, latent_dim=4).double()
    z = torch.randn(3, 4, dtype=torch.double)
    target = torch.randn(3, 6, dtype=torch.double)
    sigma = torch.full_like(target, 0.15)
    sigma[:, 4:] = 0
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[1, 3] = False
    mask[2, 2] = False
    center = torch.tensor([4900, 2.5, -0.4, 0.05, 1.2, 0.7], dtype=torch.double)
    scale = torch.tensor([150, 0.2, 0.3, 0.1, 0.2, 0.3], dtype=torch.double)
    jacobian = scale.log().expand_as(target).clone()
    jacobian[:, 4:] += center[4:] + scale[4:] * target[:, 4:]
    actual = independent_head(
        z, target, sigma, mask, jacobian, center, scale, head.state_dict(), kind
    )
    mean, variance = head.native_moments(z, center, scale)
    np.testing.assert_allclose(
        actual["mean_native"], mean.detach().numpy(), rtol=1e-11, atol=1e-10
    )
    np.testing.assert_allclose(
        actual["variance_native"], variance.detach().numpy(), rtol=1e-10, atol=1e-10
    )
    for column in range(6):
        single = torch.zeros_like(mask)
        single[:, column] = True
        for key, observed in [("intrinsic", False), ("observed", True)]:
            expected = (
                head.row_log_prob(z, target, sigma, single, observed=observed)
                - jacobian[:, column]
            )
            expected = (
                expected.masked_fill(~mask[:, column], torch.nan).detach().numpy()
            )
            np.testing.assert_allclose(
                actual[key + "_log_prob_native"][:, column],
                expected,
                rtol=1e-11,
                atol=1e-10,
            )
    native3 = torch.zeros_like(mask)
    native3[:, :3] = mask[:, :3]
    for key, observed in [("intrinsic", False), ("observed", True)]:
        expected = head.row_log_prob(
            z, target, sigma, native3, observed=observed
        ) - jacobian[:, :3].sum(-1)
        expected = expected.masked_fill(~mask[:, :3].all(1), torch.nan).detach().numpy()
        np.testing.assert_allclose(
            actual["native3_" + key + "_joint_log_prob_native"],
            expected,
            rtol=1e-11,
            atol=1e-10,
        )


def test_independent_source_coordinates_do_not_impute_or_treat_posteriors_as_errors():
    values = (
        np.tile([4900, 2, -0.5, 0.2, 5, 2], (25, 1)) + np.arange(25)[:, None] * 0.01
    )
    mask = np.ones_like(values, dtype=bool)
    mask[0, 3] = False
    values[0, 3] = np.nan
    errors = np.full_like(values, np.nan)
    known = np.zeros_like(mask)
    errors[:, 0] = 100
    known[:, 0] = True
    y, sigma, quality, jacobian, center, scale = independent_coordinates(
        values, mask, errors, known, np.ones(25, bool)
    )
    assert y[0, 3] == 0 and not quality[0, 3]
    assert np.all(sigma[:, 4:] == 0)
    np.testing.assert_allclose(jacobian[:, 4], np.log(scale[4]) + np.log(values[:, 4]))
    assert center[4] == pytest.approx(np.log(values[:, 4]).mean())
    known[:, 4] = True
    errors[:, 4] = 1
    with pytest.raises(ValueError, match="Posterior"):
        independent_coordinates(values, mask, errors, known, np.ones(25, bool))


def test_strict_scores_and_finite_sample_conformal_no_tolerance_relaxation():
    assert compare(np.array([1.0, np.nan]), np.array([1.0, np.nan]))["passed"]
    assert not compare(np.array([1.0, 0.0]), np.array([1.0, np.nan]))["passed"]
    assert not compare(np.array([0.001]), np.array([0.0]))["passed"]
    assert finite_quantile(np.arange(9), 0.1) == 8
    with pytest.raises(ValueError, match="unavailable"):
        finite_quantile(np.arange(4), 0.05)
    assert np.isinf(finite_quantile(np.arange(4), 0.05, unbounded=True))


@pytest.mark.parametrize(
    "kind,mixed",
    [("gaussian", 0), ("mdn", 0), ("factorized_nsf", 0), ("factorized_nsf", 0.1)],
)
def test_independent_sparse_objective_and_both_context_gradients(kind, mixed):
    from verify_final_six_heads import independent_criterion, validate_qmc_nodes

    torch.manual_seed(913)
    head = SparseSixHead(kind, latent_dim=4, mixed_weight=mixed).double()
    z = torch.randn(4, 4, dtype=torch.double, requires_grad=True)
    y = torch.randn(4, 6, dtype=torch.double)
    sigma = torch.zeros_like(y)
    sigma[:, :4] = 0.15
    mask = torch.ones_like(y, dtype=torch.bool)
    mask[0, [2, 4]] = False
    mask[1, [0, 5]] = False
    ids = torch.arange(4)
    actual = independent_criterion(
        z, y, sigma, mask, ids, head.state_dict(), kind, mixed
    )
    expected = head.criterion(z, y, sigma, mask, ids)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    actual_gradient = torch.autograd.grad(actual, z, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, z)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10
    )
    validate_qmc_nodes(head.state_dict(), kind)


def test_source_role_eligibility_and_photometry_xp_inputs_reject_bypass():
    from verify_final_six_heads import source_layout_checks

    ids = np.arange(2)
    roles = np.array(["fit", "cal"])
    observed = np.ones((2, 6), bool)
    source_role = np.tile(roles[:, None], (1, 6))
    source_role[0, 3] = "cal"
    quality = observed & (source_role == roles[:, None])
    raw = np.array([[12.0, 10.0, 0.1], [11.0, np.nan, 0.2]])
    visible = np.array([[True, True, False], [True, False, False]])
    data = dict(
        source_id=ids,
        role=roles,
        observed=observed,
        label_source_role=source_role,
        quality=quality,
        original_role_eligible=quality.copy(),
        benchmark_partition=np.zeros(2),
        feature_names=np.array(["G", "bp_1", "PARALLAX"]),
        raw=raw,
        scaler_center=np.zeros(3),
        scaler_scale=np.ones(3),
        visible=visible,
        input=np.where(visible, raw, -9999).astype("f4"),
    )
    prep = dict(
        observed_per_role_target={
            role: quality[roles == role].sum(0).tolist() for role in roles
        }
    )
    assert source_layout_checks(data, prep)["original_role_eligibility_pass"]
    data["quality"][0, 3] = True
    with pytest.raises(ValueError, match="eligibility"):
        source_layout_checks(data, prep)


def test_independent_stream_and_selected_adam_counters_reject_tampering():
    import copy
    import hashlib
    import json

    from verify_final_six_heads import check_state_and_streams

    policy = dict(
        reconstruction_presentations=512,
        effective_batch=256,
        warmup_updates=2,
        supervised_batches_per_update=1,
    )
    data = dict(role=np.full(24, "fit"))
    mask = np.ones((24, 6), bool)
    rng = np.random.default_rng(41)
    label_hash = hashlib.sha256()
    metrics = []
    row_hash = hashlib.sha256()
    for draw in range(4):
        rows = np.unique(
            np.concatenate(
                [rng.choice(np.arange(24), 24, replace=False) for _ in range(6)]
            )
        )
        label_hash.update(np.asarray(rows, dtype="<i8").tobytes())
        if draw >= 2:
            update = draw - 1
            row_hash.update(
                np.arange((update - 1) * 256, update * 256, dtype="<i8").tobytes()
            )
            metrics.append(
                dict(
                    updates=update,
                    presentations=update * 256,
                    reconstruction_stream_sha256=row_hash.hexdigest(),
                    label_stream_sha256=label_hash.hexdigest(),
                    mask_stream_sha256="fixed-receipt",
                )
            )
    metrics[-1]["tune_macro_native_nll"] = 1.5
    arm = dict(seed=41)
    checkpoint = dict(
        completed_updates=2,
        completed_presentations=512,
        source_sha="source",
        arm=arm,
        supervised_rng_state=json.loads(json.dumps(rng.bit_generator.state)),
        model_optimizer_state_dict=dict(
            state={
                0: dict(
                    step=torch.tensor(2.0),
                    exp_avg=torch.tensor(0.0),
                    exp_avg_sq=torch.tensor(1.0),
                )
            }
        ),
        head_optimizer_state_dict=dict(
            state={
                0: dict(
                    step=torch.tensor(4.0),
                    exp_avg=torch.tensor(0.0),
                    exp_avg_sq=torch.tensor(1.0),
                )
            }
        ),
    )
    for key in [
        "rng_state",
        "cuda_rng_state",
        "numpy_rng_state",
        "python_rng_state",
        "mask_rng_state",
        "auxiliary_view_rng_state",
    ]:
        checkpoint[key] = "present"
    manifest = dict(
        source_sha="source",
        arm=arm,
        training_budget=dict(updates=2),
        selected_model_counters=dict(updates=2),
        row_stream_sha256=row_hash.hexdigest(),
        label_stream_sha256=label_hash.hexdigest(),
        mask_stream_sha256="fixed-receipt",
    )
    report = check_state_and_streams(checkpoint, manifest, data, mask, metrics, policy)
    assert (
        report["reconstruction_position_stream_pass"] and report["head_adam_steps"] == 4
    )
    tampered = copy.deepcopy(metrics)
    tampered[0]["label_stream_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Intermediate FIT"):
        check_state_and_streams(checkpoint, manifest, data, mask, tampered, policy)


def test_independent_cp_checks_both_density_estimands_and_unbounded_familywise(
    tmp_path,
):
    from run_final_six_recipes import calibrate
    from verify_final_six_heads import check_calibration

    rng = np.random.default_rng(93)
    cal = dict(
        mean_native=rng.normal(size=(20, 6)),
        target_native=rng.normal(size=(20, 6)),
        intrinsic_log_prob_native=rng.normal(size=(20, 6)),
        observed_log_prob_native=rng.normal(size=(20, 6)),
        observed=np.ones((20, 6), bool),
        source_id=np.arange(20),
    )
    test = dict(
        mean_native=rng.normal(size=(7, 6)),
        target_native=rng.normal(size=(7, 6)),
        intrinsic_log_prob_native=rng.normal(size=(7, 6)),
        observed_log_prob_native=rng.normal(size=(7, 6)),
        observed=np.ones((7, 6), bool),
        source_id=np.arange(20, 27),
    )
    calibrate(cal, test, tmp_path)
    saved = dict(np.load(tmp_path / "conformal.npz", allow_pickle=False))
    checks = check_calibration(cal, test, saved)
    assert all(value["passed"] for value in checks.values())
    saved["90_observed_density_age_threshold"] = (
        saved["90_observed_density_age_threshold"] + 0.01
    )
    assert not all(
        value["passed"] for value in check_calibration(cal, test, saved).values()
    )


def test_frozen_head_stream_uses_real_zero_model_budget_and_selected_counts():
    import copy
    import hashlib
    import json

    from verify_final_six_heads import check_frozen_state_and_streams

    control = dict(
        warmup_updates=2, head_optimizer_updates=2, supervised_batches_per_update=1
    )
    data = dict(role=np.full(24, "fit"))
    mask = np.ones((24, 6), bool)
    rng = np.random.default_rng(41)
    label_hash = hashlib.sha256()
    counts = np.zeros(6, dtype="i8")
    metrics = []
    for draw in range(4):
        rows = np.unique(
            np.concatenate(
                [rng.choice(np.arange(24), 24, replace=False) for _ in range(6)]
            )
        )
        counts += mask[rows].sum(0)
        label_hash.update(np.asarray(rows, dtype="<i8").tobytes())
        if draw >= 2:
            metrics.append(
                dict(
                    updates=draw - 1,
                    head_adam_steps=draw + 1,
                    label_stream_sha256=label_hash.hexdigest(),
                    encoder_updates=0,
                    reconstruction_presentations=0,
                )
            )
    metrics[-1]["tune_macro_native_nll"] = 1.3
    arm = dict(seed=41)
    budget = dict(
        control, encoder_updates=0, decoder_updates=0, reconstruction_presentations=0
    )
    checkpoint = dict(
        completed_updates=2,
        completed_head_adam_steps=4,
        completed_presentations=0,
        encoder_updates=0,
        decoder_updates=0,
        reconstruction_presentations=0,
        source_sha="source",
        arm=arm,
        supervised_rng_state=json.loads(json.dumps(rng.bit_generator.state)),
        label_stream_sha256=label_hash.hexdigest(),
        supervised_label_presentations=counts,
        frozen_model_parameter_sha256="digest",
        head_optimizer_state_dict=dict(
            state={
                0: dict(
                    step=torch.tensor(4.0),
                    exp_avg=torch.tensor(0.0),
                    exp_avg_sq=torch.tensor(1.0),
                )
            }
        ),
    )
    for key in ["rng_state", "cuda_rng_state", "numpy_rng_state", "python_rng_state"]:
        checkpoint[key] = "present"
    manifest = dict(
        source_sha="source",
        arm=arm,
        training_budget=budget,
        no_10m_training_claim=True,
        selected_head_counters=dict(head_updates=2, head_adam_steps=4),
        label_stream_sha256=label_hash.hexdigest(),
        supervised_label_presentations=counts.tolist(),
        frozen_model_values_unchanged=True,
        frozen_model_parameter_sha256="digest",
        frozen_model_parameter_sha256_after="digest",
    )
    report = check_frozen_state_and_streams(
        checkpoint, manifest, data, mask, metrics, control
    )
    assert report["encoder_updates"] == report["reconstruction_presentations"] == 0
    assert report["head_adam_steps"] == 4
    bad = copy.deepcopy(checkpoint)
    bad["completed_presentations"] = 10000000
    with pytest.raises(ValueError, match="10M"):
        check_frozen_state_and_streams(bad, manifest, data, mask, metrics, control)


def test_shared_frozen_encoder_cache_reuses_exact_outputs_without_forward(tmp_path):
    from verify_final_six_heads import encode_inputs

    class TinyEncoder:
        calls = 0

        def encoder(self, x):
            self.calls += 1
            return 2 * x

    model = TinyEncoder()
    data = dict(
        source_id=np.array([11, 12, 13]), input=np.arange(9).reshape(3, 3).astype("f4")
    )
    rows = np.array([0, 1])
    first = encode_inputs(model, data, rows, tmp_path)
    second = encode_inputs(model, data, rows, tmp_path)
    assert model.calls == 1
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.dtype == torch.float64
