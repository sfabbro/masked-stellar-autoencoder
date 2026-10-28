"""XP input-error corrections for the matched FeH arms.

Only the 110 Gaia XP coefficients are corrected. Their reported errors are
converted from explicitly declared source units to G-normalised coefficient
units and divided by the parent feature scale. Photometry, EBV and sentinel entries are
left unchanged. Each coefficient is calibrated on its own: the measurement
noise is diagonal, and a joint 110-d reliability matrix is ill-conditioned.

Shared regression calibration uses one rms error per coefficient, estimated on
the fit stars. Per-star calibration uses that same signal variance and each
star's own error. SIMEX adds Gaussian noise at the library's default lambda
grid and extrapolates dex predictions with an affine rational curve (not the library polynomial).
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
    raw_error: np.ndarray,
    scale: np.ndarray,
    columns: np.ndarray,
    *,
    g_magnitude: np.ndarray,
    error_convention: str | np.ndarray,
) -> np.ndarray:
    """Map reported diagonal XP errors to the parent network coefficient units.

    ``raw_xp`` means the original Gaia archive coefficient-error units: divide
    by ``10**((8.5-G)/2.5)`` before the parent RobustScaler scale. Already
    G-normalised values must explicitly declare ``g_normalized_xp`` and are
    divided by the parent scale only. G itself is an apparent magnitude, never
    its scaled network input. Per-row conventions permit audited mixed sources.

    This conditions on observed G. Its uncertainty and the resulting common
    normalisation covariance are omitted, as are native BP/RP correlations.
    No physical error floor is invented.
    """
    raw_error, scale = np.asarray(raw_error), np.asarray(scale)
    columns = np.asarray(columns, dtype=int)
    if raw_error.ndim != 2 or scale.shape != (raw_error.shape[1],):
        raise ValueError("errors and scale do not describe the same columns")
    if columns.ndim != 1 or np.any(columns < 0) or np.any(columns >= len(scale)):
        raise ValueError("invalid XP column indices")
    selected_scale = np.asarray(scale[columns], dtype="f8")
    if not np.isfinite(selected_scale).all() or np.any(selected_scale <= 0):
        raise ValueError("XP parent scales must be finite and positive")
    g = np.asarray(g_magnitude, dtype="f8")
    if g.shape != (len(raw_error),) or not np.isfinite(g).all() or np.any(g == -9999):
        raise ValueError("finite apparent G magnitudes are required for XP errors")
    conventions = np.asarray(error_convention)
    if conventions.ndim == 0:
        conventions = np.full(len(raw_error), conventions.item())
    if (
        conventions.shape != (len(raw_error),)
        or not np.isin(conventions, ["raw_xp", "g_normalized_xp"]).all()
    ):
        raise ValueError(
            "XP error convention must be explicitly raw_xp or g_normalized_xp"
        )
    raw_rows = conventions == "raw_xp"
    factor = np.ones(len(raw_error), dtype="f8")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        factor[raw_rows] = 10.0 ** ((8.5 - g[raw_rows]) / 2.5)
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError("invalid G flux normalisation factor")
    sigma = np.asarray(raw_error[:, columns], dtype="f8") / factor[:, None]
    sigma /= selected_scale
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
        spec.get("simex_extrapolant") != "affine_rational"
        or spec.get("simex_n_simulations") != _SIMEX_SIMULATIONS
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
    if array.ndim != 2 or array.shape[0] != rows or not np.isfinite(array).all():
        raise ValueError("calibrated XP must have a finite two-dimensional row matrix")
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
    """Fit b + a/(c+lambda), then evaluate at -1 (raw prediction units).

    Profile c on the declared fixed grid [1.05,8] and solve intercept/amplitude
    by least squares. Unlike a/(c+lambda), this preserves constants and changes
    of label origin. It remains a bounded, empirical SIMEX model: the known
    linear attenuation example does not certify nonlinear stellar corrections.
    """
    lam = np.asarray(lambdas, dtype="f8")
    values = np.asarray(predictions, dtype="f8")
    if (
        lam.ndim != 1
        or len(lam) < 3
        or not np.isfinite(lam).all()
        or np.any(lam < 0)
        or len(np.unique(lam)) != len(lam)
        or 0 not in lam
    ):
        raise ValueError(
            "SIMEX needs at least three distinct finite nonnegative lambdas including zero"
        )
    if values.ndim != 2 or values.shape[0] != len(lam) or not np.isfinite(values).all():
        raise ValueError("SIMEX predictions must be finite [lambda, row]")
    # Remove an arbitrary label origin before fitting; every fit has an intercept.
    anchor = values[0]
    shifted = values - anchor
    mean = shifted.mean(axis=0)
    centered = shifted - mean
    best_resid = np.full(values.shape[1], np.inf)
    best = np.zeros(values.shape[1], dtype="f8")
    for offset in np.linspace(1.05, 8.0, 400):
        weight = 1.0 / (offset + lam)
        centered_weight = weight - weight.mean()
        amplitude = (centered_weight @ centered) / np.dot(
            centered_weight, centered_weight
        )
        residual = centered_weight[:, None] * amplitude - centered
        resid = np.sum(residual**2, axis=0)
        estimate = mean + amplitude * (1.0 / (offset - 1.0) - weight.mean())
        take = resid < best_resid
        best_resid[take] = resid[take]
        best[take] = estimate[take]
    best += anchor
    if not np.isfinite(best).all():
        raise ValueError("nonfinite affine rational SIMEX extrapolation")
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
    *,
    g_magnitude: np.ndarray,
    error_convention: str | np.ndarray,
) -> tuple[list[dict], dict]:
    method = spec.get("method")
    if method not in {"rc_shared", "rc_per_star", "simex"}:
        raise ValueError(f"unknown input correction {method}")
    columns = xp_indices(list(names))
    sigma = standardized_xp_errors(
        raw_error,
        scale,
        columns,
        g_magnitude=g_magnitude,
        error_convention=error_convention,
    )
    rms = shared_rms(sigma[fit_rows])
    values = np.asarray(matrix[:, columns], dtype="f8")
    if not np.isfinite(values).all() or np.any(np.isclose(values, -9999)):
        raise ValueError("XP inputs are missing; sentinels are not calibrated")
    calibrators = fit_univariate_rc(values[fit_rows], rms)
    reliability = np.array(
        [float(item.reliability_matrix) for item in calibrators], dtype="f8"
    )
    report = correction_report(method, spec, rms, reliability)
    report.update(
        xp_error_conventions=np.unique(np.asarray(error_convention)).tolist(),
        xp_error_normalization="raw_xp: reported_sigma / 10**((8.5-G)/2.5) / parent_scale; g_normalized_xp: reported_sigma / parent_scale",
        apparent_G_conditioned_on=True,
        apparent_G_uncertainty_omitted=True,
        shared_G_normalization_covariance_omitted=True,
        native_XP_correlations_omitted=True,
        invented_physical_sigma_floor=False,
    )
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
            extrapolant="b+a/(c+lambda), fit independently per star; c grid[1.05,8]",
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


def audit_xp_error_sources(
    cohort_path,
    benchmark_path,
    legacy_path,
    feature_rows,
    features,
    error_names=None,
    *,
    common_rows=128,
    block_size=8192,
):
    """Read-only launch gate: verify actual XP errors against original records.

    The immutable benchmark's raw-XP error convention is independently established
    by the original Flatiron matched-source audit. Here, common stars transfer
    that convention to the legacy FITS by testing all 110 error columns, rather
    than inferring error units from coefficient agreement. Every selected cohort
    row is then matched back to its declared source row, including G and XP
    coefficients. This reads bounded record blocks, never the whole FITS table.
    """
    import hashlib
    import json

    import h5py
    from astropy.io import fits

    rows = np.unique(np.asarray(feature_rows, dtype="i8"))
    columns = xp_indices(list(features))
    g_index = list(features).index("G")

    def clean(value):
        value = np.asarray(value, dtype="f4").copy()
        value[(value == -9999) | ~np.isfinite(value)] = np.nan
        return value

    def matches(actual, expected, name):
        if not np.array_equal(clean(actual), clean(expected), equal_nan=True):
            raise ValueError(f"XP source error-unit audit mismatch: {name}")

    with h5py.File(cohort_path) as cohort, h5py.File(benchmark_path) as benchmark:
        if error_names is None:
            error_names = json.loads(cohort.attrs["error_columns_json"])
        if len(error_names) != len(features) or any(
            error_names[j] is None for j in columns
        ):
            raise ValueError("Require the actual original XP error column mapping")
        error_fields = [error_names[j] for j in columns]
        xp_fields = [features[j] for j in columns]
        all_ids = cohort["features/source_id"][:]
        all_codes = cohort["features/input_source"][:]
        all_positions = cohort["features/input_original_row"][:]
        if not len(rows) or (rows < 0).any() or (rows >= len(all_ids)).any():
            raise ValueError("No valid selected source rows for XP error audit")
        if not np.isin(all_codes[rows], [0, 1, 2]).all():
            raise ValueError("Unknown/unavailable input source in XP error audit")
        fields = ["source_id", "G", *xp_fields, *error_fields]

        def records_for(selected):
            codes, positions = all_codes[selected], all_positions[selected]
            original_g = np.empty(len(selected), dtype="f4")
            original_values = np.empty((len(selected), 110), dtype="f4")
            original_errors = np.empty_like(original_values)
            for code in np.unique(codes):
                source = (
                    legacy
                    if code == 2
                    else benchmark[
                        ("train" if code == 0 else "validation") + "/records"
                    ]
                )
                which = np.flatnonzero(codes == code)
                if (positions[which] < 0).any() or (
                    positions[which] >= len(source)
                ).any():
                    raise ValueError("Source row outside original record table")
                for cell in np.unique(positions[which] // block_size):
                    take = which[positions[which] // block_size == cell]
                    start = int(cell) * block_size
                    if code == 2:
                        raw = source[start : min(start + block_size, len(source))]
                    else:
                        raw = source.fields(fields)[
                            start : min(start + block_size, len(source))
                        ]
                    raw = raw[positions[take] - start]
                    if not np.array_equal(
                        np.asarray(raw["source_id"], dtype="i8"),
                        all_ids[selected[take]],
                    ):
                        raise ValueError("XP source row / Gaia ID mismatch")
                    original_g[take] = clean(raw["G"])
                    original_values[take] = np.column_stack(
                        [clean(raw[k]) for k in xp_fields]
                    )
                    original_errors[take] = np.column_stack(
                        [clean(raw[k]) for k in error_fields]
                    )
            return original_g, original_values, original_errors

        with fits.open(legacy_path, memmap=True) as hdus:
            table = next(h for h in hdus if isinstance(h, fits.BinTableHDU))
            legacy = table.data
            if not set(fields).issubset(table.columns.names):
                raise ValueError("Original legacy FITS lacks required XP fields")
            legacy_ids = np.asarray(legacy["source_id"], dtype="i8")
            unique_ids, first = np.unique(legacy_ids, return_index=True)
            candidates = np.flatnonzero(np.isin(all_codes, [0, 1]))
            common = []
            legacy_matches = []
            for start in range(0, len(candidates), block_size):
                candidate = candidates[start : start + block_size]
                at = np.searchsorted(unique_ids, all_ids[candidate])
                hit = at < len(unique_ids)
                hit &= (
                    unique_ids[np.minimum(at, len(unique_ids) - 1)]
                    == all_ids[candidate]
                )
                candidate, matched = candidate[hit], first[at[hit]]
                errors = cohort["features/reported_feature_errors"][candidate][
                    :, columns
                ]
                finite = np.isfinite(errors).all(1) & (errors > 0).all(1)
                common.extend(candidate[finite].tolist())
                legacy_matches.extend(matched[finite].tolist())
                if len(common) >= common_rows:
                    break
            if len(common) < common_rows:
                raise ValueError(
                    f"Only {len(common)} common stars for XP error-unit audit"
                )
            common = np.asarray(common[:common_rows], dtype="i8")
            legacy_matches = np.asarray(legacy_matches[:common_rows], dtype="i8")
            original_g, original_values, original_errors = records_for(common)
            legacy_common = legacy[legacy_matches]
            for j, field in enumerate(error_fields):
                matches(
                    legacy_common[field], original_errors[:, j], f"common raw {field}"
                )
            for j, field in enumerate(xp_fields):
                matches(
                    legacy_common[field],
                    original_values[:, j],
                    f"common normalized {field}",
                )
            matches(legacy_common["G"], original_g, "common apparent G")
            # The cohort itself must match each original source; do this for
            # every row used by the actual fit/evaluation, not a schema fixture.
            error_digest = hashlib.sha256()
            g_digest = hashlib.sha256()
            for start in range(0, len(rows), block_size):
                selected = rows[start : start + block_size]
                original_g, original_values, original_errors = records_for(selected)
                cohort_values = cohort["features/values"][selected]
                cohort_errors = cohort["features/reported_feature_errors"][selected]
                matches(
                    cohort_errors[:, columns], original_errors, "selected raw errors"
                )
                matches(
                    cohort_values[:, columns],
                    original_values,
                    "selected normalized coefficients",
                )
                matches(cohort_values[:, g_index], original_g, "selected apparent G")
                error_digest.update(original_errors.tobytes())
                g_digest.update(original_g.tobytes())
    return dict(
        status="verified",
        error_convention="raw_xp",
        benchmark_path=str(benchmark_path),
        legacy_path=str(legacy_path),
        cohort_path=str(cohort_path),
        common_stars=len(common),
        common_source_id_sha256=hashlib.sha256(all_ids[common].tobytes()).hexdigest(),
        common_source_id=all_ids[common].tolist(),
        common_legacy_row=legacy_matches.tolist(),
        all110_errors_match_original_benchmark_and_legacy=True,
        all_selected_rows_match_actual_source=True,
        selected_rows=len(rows),
        selected_source_counts={
            str(k): int((all_codes[rows] == k).sum()) for k in (0, 1, 2)
        },
        selected_original_error_sha256=error_digest.hexdigest(),
        selected_apparent_G_sha256=g_digest.hexdigest(),
        normalization="original raw XP errors / 10**((8.5-G)/2.5) / parent scale",
        common_G_uncertainty_and_covariance_omitted=True,
        no_physical_error_floor=True,
    )
