"""XP input-error corrections for the matched FeH arms.

Only the 110 Gaia XP coefficients are corrected. Their reported errors are
divided by the parent feature scale. Photometry, EBV and sentinel entries are
left unchanged. Each coefficient is calibrated on its own: the measurement
noise is diagonal, and a joint 110-d reliability matrix is ill-conditioned.

Shared regression calibration uses one rms error per coefficient, estimated on
the fit stars. Per-star calibration uses that same signal variance and each
star's own error. SIMEX adds Gaussian noise at the library's default lambda
grid and extrapolates the dex predictions with the library polynomial.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torchregress.algorithms.rc import RegressionCalibration
from torchregress.algorithms.simex import SIMEX

_NOISE = "fit_rms_standardized_diagonal"
_FEATURES = "xp_coefficients"
_SIMEX_SIMULATIONS = 4
_SIMEX_ORDER = 2


def xp_indices(names: list[str]) -> np.ndarray:
    indices = [
        i
        for i, name in enumerate(names)
        if name.startswith(("bp_", "rp_")) and name.split("_")[-1].isdigit()
    ]
    if len(indices) != 110:
        raise ValueError(f"expected 110 XP coefficients, found {len(indices)}")
    return np.asarray(indices, dtype=int)


def standardized_xp_errors(
    raw_error: np.ndarray, scale: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    if raw_error.ndim != 2 or scale.shape != (raw_error.shape[1],):
        raise ValueError("errors and scale do not describe the same columns")
    sigma = np.asarray(raw_error[:, columns], dtype="f8") / np.asarray(
        scale[columns], dtype="f8"
    )
    if not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("XP errors must be finite and positive in standardized units")
    return sigma


def shared_rms(sigma: np.ndarray) -> np.ndarray:
    if sigma.ndim != 2 or sigma.shape[1] != 110:
        raise ValueError("shared XP noise expects [row, 110] standardized errors")
    rms = np.sqrt(np.mean(np.square(sigma), axis=0))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("XP rms noise must be finite and positive")
    return rms


def _locked(spec: dict, method: str) -> None:
    if spec.get("features") != _FEATURES or spec.get("noise") != _NOISE:
        raise ValueError("XP input correction spec changed")
    if method == "simex" and (
        spec.get("simex_n_simulations") != _SIMEX_SIMULATIONS
        or spec.get("simex_extrapolation_order") != _SIMEX_ORDER
    ):
        raise ValueError("SIMEX replication spec changed")


def fit_univariate_rc(
    values: np.ndarray, sigma_std: np.ndarray
) -> list[RegressionCalibration]:
    if values.ndim != 2 or sigma_std.shape != (values.shape[1],):
        raise ValueError("RC inputs and per-coefficient sigma do not match")
    if not np.isfinite(values).all():
        raise ValueError("RC cannot be fit on nonfinite XP coefficients")
    calibrators = []
    for column, sigma in enumerate(sigma_std):
        calibrator = RegressionCalibration(float(sigma))
        calibrator.fit(
            torch.as_tensor(values[:, column : column + 1], dtype=torch.float64)
        )
        reliability = float(calibrator.reliability_matrix)
        if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
            raise ValueError(f"XP coefficient {column} reliability is {reliability}")
        calibrators.append(calibrator)
    return calibrators


def _as_float32(values: torch.Tensor, rows: int) -> np.ndarray:
    array = values.detach().cpu().numpy().astype("f4", copy=False)
    if (
        array.ndim != 2
        or array.shape[0] != rows
        or not np.isfinite(array).all()
        or np.any(np.abs(array) > 50)
    ):
        raise ValueError("calibrated XP left the finite standardized range")
    return array


def apply_shared_rc(
    calibrators: list[RegressionCalibration], values: np.ndarray
) -> np.ndarray:
    corrected = []
    for column, calibrator in enumerate(calibrators):
        block = torch.as_tensor(values[:, column : column + 1], dtype=torch.float64)
        corrected.append(calibrator.transform(block))
    return _as_float32(torch.cat(corrected, dim=1), len(values))


def apply_per_star_rc(
    calibrators: list[RegressionCalibration],
    values: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    if sigma.shape != values.shape:
        raise ValueError("per-star XP errors must match the coefficient matrix")
    corrected = []
    for column, calibrator in enumerate(calibrators):
        block = torch.as_tensor(values[:, column : column + 1], dtype=torch.float64)
        noise = torch.as_tensor(sigma[:, column : column + 1], dtype=torch.float64)
        mean, _covariance = calibrator.posterior(block, sigma_u=noise)
        corrected.append(mean)
    return _as_float32(torch.cat(corrected, dim=1), len(values))


def simex_library_weights() -> tuple[list[float], np.ndarray]:
    """Polynomial weights from torchregress SIMEX, independent of the data."""
    probe = SIMEX(
        model_factory=lambda: torch.nn.Identity(),
        train_func=lambda model, _values, _target: model,
        sigma_u=1.0,
        n_simulations=1,
        extrapolation_order=_SIMEX_ORDER,
    )
    probe.fit(torch.zeros(4, 1), torch.zeros(4, 1))
    return [
        float(value) for value in probe.lambdas_used
    ], probe.extrapolation_weights.detach().cpu().numpy()


def rational_extrapolate(lambdas: list[float], predictions: np.ndarray) -> np.ndarray:
    """Extrapolate each column to lambda=-1 with a/(c+lambda).

    The library's quadratic gives 1.49, not 2, for the classical attenuation
    curve 2/(2+lambda). This one-parameter family matches that curve.
    """
    lam = np.asarray(lambdas, dtype="f8")
    values = np.asarray(predictions, dtype="f8")
    if values.ndim != 2 or values.shape[0] != len(lam):
        raise ValueError("SIMEX predictions must be [lambda, row]")
    best_resid = np.full(values.shape[1], np.inf)
    best = np.empty(values.shape[1], dtype="f8")
    for offset in np.linspace(1.05, 8.0, 400):
        weight = 1.0 / (offset + lam)
        scale = (weight @ values) / np.dot(weight, weight)
        resid = np.sum((scale * weight[:, None] - values) ** 2, axis=0)
        estimate = scale / (offset - 1.0)
        take = resid < best_resid
        best_resid[take] = resid[take]
        best[take] = estimate[take]
    if not np.isfinite(best).all():
        raise ValueError("nonfinite rational SIMEX extrapolation")
    return best


def perturb_xp(
    matrix: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    sigma_std: np.ndarray,
    lam: float,
    seed: int,
) -> np.ndarray:
    """Add N(0, lambda Sigma) to selected XP entries with a private generator."""
    if lam < 0:
        raise ValueError("SIMEX lambda must be non-negative")
    out = np.array(matrix, copy=True)
    if lam == 0:
        return out
    block = torch.as_tensor(out[np.ix_(rows, columns)], dtype=torch.float64)
    covariance = torch.diag(torch.as_tensor(sigma_std, dtype=torch.float64) ** 2)
    factor = torch.linalg.cholesky(covariance)
    generator = torch.Generator().manual_seed(int(seed))
    noise = (
        torch.randn(block.shape, generator=generator, dtype=torch.float64) @ factor.T
    )
    out[np.ix_(rows, columns)] = (
        (block + math.sqrt(lam) * noise).numpy().astype(out.dtype)
    )
    return out


def correction_report(
    method: str, spec: dict, rms: np.ndarray, reliability: np.ndarray
) -> dict:
    _locked(spec, method)
    return dict(
        method=method,
        features=_FEATURES,
        noise=_NOISE,
        correlations_omitted=True,
        photometry_and_ebv_unchanged=True,
        pretraining_not_retrained=True,
        rms_min=float(rms.min()),
        rms_max=float(rms.max()),
        reliability_min=float(reliability.min()),
        reliability_max=float(reliability.max()),
        coefficients_below_reliability_0_05=int(np.sum(reliability < 0.05)),
    )


def prepare_replicas(
    spec: dict,
    matrix: np.ndarray,
    raw_error: np.ndarray,
    scale: np.ndarray,
    names: list[str],
    fit_rows: np.ndarray,
) -> tuple[list[dict], dict]:
    method = spec.get("method")
    if method not in {"rc_shared", "rc_per_star", "simex"}:
        raise ValueError(f"unknown input correction {method}")
    columns = xp_indices(list(names))
    sigma = standardized_xp_errors(raw_error, scale, columns)
    rms = shared_rms(sigma[fit_rows])
    values = np.asarray(matrix[:, columns], dtype="f8")
    if not np.isfinite(values).all() or np.any(np.isclose(values, -9999)):
        raise ValueError("XP inputs are missing; sentinels are not calibrated")
    calibrators = fit_univariate_rc(values[fit_rows], rms)
    reliability = np.array(
        [float(item.reliability_matrix) for item in calibrators], dtype="f8"
    )
    report = correction_report(method, spec, rms, reliability)
    if method == "rc_shared":
        corrected = np.array(matrix, copy=True)
        corrected[:, columns] = apply_shared_rc(calibrators, values)
        replicas = [dict(lambda_value=0.0, simulation=0, matrix=corrected)]
    elif method == "rc_per_star":
        corrected = np.array(matrix, copy=True)
        corrected[:, columns] = apply_per_star_rc(calibrators, values, sigma)
        replicas = [dict(lambda_value=0.0, simulation=0, matrix=corrected)]
    else:
        lambdas, _weights = simex_library_weights()
        rows = np.arange(len(matrix))
        replicas = []
        for index, lam in enumerate(lambdas):
            simulations = 1 if lam == 0.0 else _SIMEX_SIMULATIONS
            for simulation in range(simulations):
                replicas.append(
                    dict(
                        lambda_value=float(lam),
                        simulation=simulation,
                        matrix=perturb_xp(
                            matrix,
                            rows,
                            columns,
                            rms,
                            lam,
                            20_000 + index * 100 + simulation,
                        ),
                    )
                )
        report.update(
            lambdas=lambdas,
            replicas=len(replicas),
            extrapolant="a/(c+lambda), fit independently per star",
            library_quadratic_extrapolant_rejected=True,
        )
    return replicas, report


def mean_prediction_by_lambda(
    replicas: list[dict], predictions: list[np.ndarray]
) -> tuple[list[float], np.ndarray]:
    order: list[float] = []
    grouped: dict[float, list[np.ndarray]] = {}
    if len(replicas) != len(predictions):
        raise ValueError("SIMEX replicas and predictions differ in length")
    for replica, prediction in zip(replicas, predictions):
        lam = float(replica["lambda_value"])
        grouped.setdefault(lam, []).append(np.asarray(prediction, dtype="f8"))
        if lam not in order:
            order.append(lam)
    means = np.stack([np.mean(grouped[lam], axis=0) for lam in order])
    return order, means
