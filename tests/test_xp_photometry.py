import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.xp_photometry import (
    PhotometryCalibration,
    PhotometryProjection,
    fit_calibration,
    normalized_observed_flux,
    photometry_consistency_sum,
)


def calibration(reference=(1.0, 2.0)):
    return PhotometryCalibration(
        np.array(reference),
        np.array([0.1, -0.1]),
        np.array([0.2, 0.3]),
        np.array([1000, 1000]),
    )


def fixture():
    prediction = torch.tensor(
        [[3.0, 99.0, 5.0], [4.0, 5.0, 9.0]], dtype=torch.float64, requires_grad=True
    )
    target = torch.tensor([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0]], dtype=torch.float64)
    hidden = torch.tensor([[True, False, False], [False, True, True]])
    matrix = np.array([[1.0, 2.0], [2.0, 1.0], [1.0, -1.0]])
    center, scale = np.array([0.1, 0.2, 0.3]), np.array([2.0, 3.0, 4.0])
    flux = np.array([[20.0, 30.0], [30.0, 20.0]])
    errors = np.array([[0.1, 0.1], [0.2, 0.2]])
    return prediction, target, hidden, center, scale, matrix, flux, errors


def test_huber_matches_independent_numpy_and_hidden_gradients():
    args = fixture()
    result, count = photometry_consistency_sum(*args, calibration())
    pred, target, hidden, center, scale, matrix, flux, errors = args
    combined = np.where(hidden.numpy(), pred.detach().numpy(), target.numpy())
    ref = calibration().reference_flux
    observed = flux / ref
    z = ((combined * scale + center) @ matrix) / ref - observed - calibration().bias
    z /= np.sqrt((observed * np.log(10) / 2.5 * errors) ** 2 + calibration().floor ** 2)
    huber = np.where(abs(z) <= 1, 0.5 * z * z, abs(z) - 0.5)
    expected = (huber.mean(axis=1) * hidden.sum(axis=1).numpy()).sum()
    assert result.item() == pytest.approx(expected)
    assert count == 4
    result.backward()
    assert torch.equal(pred.grad[~hidden], torch.zeros(3, dtype=torch.float64))
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad[hidden] != 0).all()


def test_physical_flux_units_do_not_change_loss_or_gradients():
    args = fixture()
    first, _ = photometry_consistency_sum(*args, calibration())
    first.backward()
    gradient = args[0].grad.clone()
    args = list(fixture())
    args[5] *= 1e-24
    args[6] *= 1e-24
    second, _ = photometry_consistency_sum(*args, calibration((1e-24, 2e-24)))
    second.backward()
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(gradient, args[0].grad)


def test_invalid_rows_missing_photometry_and_empty_mask():
    args = list(fixture())
    args[1][0, 1] = float("nan")
    args[6][1, 0] = float("nan")
    result, count = photometry_consistency_sum(*args, calibration())
    assert count == 1 and torch.isfinite(result)
    result.backward()
    assert torch.equal(args[0].grad[0], torch.zeros(3, dtype=torch.float64))
    args = list(fixture())
    args[2].fill_(False)
    result, count = photometry_consistency_sum(*args, calibration())
    assert count == 0 and result.item() == 0
    args = list(fixture())
    args[0].data[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite predicted"):
        photometry_consistency_sum(*args, calibration())


def test_decoder_target_detached_and_band_mask_honoured():
    args = fixture()
    decoder = torch.tensor([[20.0, 30.0], [30.0, 20.0]], requires_grad=True)
    result, count = photometry_consistency_sum(
        *args,
        calibration(),
        decoder_flux=decoder,
        band_mask=torch.tensor([[True, False], [False, False]]),
    )
    result.backward()
    assert count == 1
    assert decoder.grad is None
    assert torch.equal(args[0].grad[1], torch.zeros(3, dtype=torch.float64))


def test_training_calibration_units_and_known_bias():
    projection = PhotometryProjection(
        np.eye(2), np.array([0.0, 0.0]), ("g", "r"), ("g", "r")
    )
    mags = np.zeros((1000, 2))
    g = np.full(1000, 8.5)
    np.testing.assert_allclose(
        normalized_observed_flux(mags, g, projection.zero_points), 1
    )
    coeff = np.tile([1.1, 0.8], (1000, 1))
    fitted = fit_calibration(coeff, mags, g, projection, min_count=1000)
    np.testing.assert_allclose(fitted.bias, [0.1, -0.2])
    np.testing.assert_allclose(fitted.floor, [0.02, 0.02])
    scaled = PhotometryProjection(
        projection.matrix * 1e-24,
        projection.zero_points - 60,
        projection.columns,
        projection.bands,
    )
    fit_scaled = fit_calibration(coeff, mags, g, scaled, min_count=1000)
    np.testing.assert_allclose(fitted.bias, fit_scaled.bias)
    np.testing.assert_allclose(fitted.floor, fit_scaled.floor)
    with pytest.raises(ValueError, match="only 999"):
        fit_calibration(coeff[:-1], mags[:-1], g[:-1], projection, min_count=1000)


def test_bundled_projection_npz_roundtrip_and_raw_band_contraction(tmp_path):
    path = "experiments/xp/photometry-projection.npz"
    projection = PhotometryProjection.load(path)
    assert projection.matrix.shape == (110, 9)
    assert projection.columns == (
        "G_SDSS",
        "R_SDSS",
        "I_SDSS",
        "G_PS1",
        "R_PS1",
        "I_PS1",
        "G_SMSS",
        "R_SMSS",
        "I_SMSS",
    )
    projection.save(tmp_path / "projection.npz")
    recovered = PhotometryProjection.load(tmp_path / "projection.npz")
    np.testing.assert_array_equal(projection.matrix, recovered.matrix)


def test_microbatch_gradient_uses_global_scored_denominator():
    args = fixture()
    full, _ = photometry_consistency_sum(*args, calibration())
    denominator = 17.0  # XP plus non-XP scored columns in the optimiser batch.
    (full / denominator).backward()
    expected = args[0].grad.clone()
    args = fixture()
    for row in range(2):
        micro = tuple(
            value[row : row + 1] if index in (0, 1, 2, 6, 7) else value
            for index, value in enumerate(args)
        )
        numerator, _ = photometry_consistency_sum(*micro, calibration())
        (numerator / denominator).backward()
    torch.testing.assert_close(args[0].grad, expected)
