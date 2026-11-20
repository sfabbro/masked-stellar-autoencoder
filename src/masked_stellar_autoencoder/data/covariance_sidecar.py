"""Gaia DR3 XP and astrometric covariance sidecar: schema, reader and covariance algebra.

The sidecar (``scripts/build_covariance_sidecar.py``) is an HDF5 file with one row
per input star, in the input order (``source_id`` and ``row_index`` give the
alignment; ``found_xp`` / ``found_astrometry`` say whether the star was found in
the HATS ``xp_continuous_mean_spectrum`` / ``gaia_source`` catalogues). Every
dataset is a top-level ``[N]`` or ``[N, k]`` array named as in the Gaia DR3
archive; missing values are NaN (floats), -1 (integers) or False (booleans).

XP correlation packing (Gaia DR3 datamodel, ``xp_continuous_mean_spectrum``):
``bp_coefficient_correlations`` holds the ``n(n-1)/2 = 1485`` off-diagonal
elements of the correlation matrix ``M`` (unit diagonal implicit) in
column-major order of the *upper* triangle: ``M[i, j] = C[j(j-1)/2 + i]`` for
``i < j``. This is the same sequence as numpy's row-major strict lower triangle,
``M[np.tril_indices(n, -1)] = C`` (the convention of GaiaXPy's
``array_to_symmetric_matrix``).

XP covariance convention. The published ``bp_coefficient_errors`` already
include the solution's ``bp_standard_deviation`` s (``s = sqrt(chi2/dof)``). The
covariance at the level of the published errors is::

    Sigma = diag(e) R diag(e)                         (default, ``formal=False``)

and the formal (unscaled least-squares) covariance is ``Sigma / s**2``
(``formal=True``). Source: GaiaXPy 2.2.0, ``spectrum/utils.py::
correlation_to_covariance_dr3int5`` builds ``D R D`` with ``D = diag(e / s)``
and ``spectrum/absolute_sampled_spectrum.py`` / ``sampled_spectrum.
_sample_error`` multiply that covariance by ``s**2`` (flux errors by ``s``)
when propagating to sampled spectra; ``cholesky.py`` notes "the formal errors
need to be scaled by the inverse standard deviation". The archive datamodel
only defines s as the "standard deviation of the solution"; see also Carrasco
et al. (2021, A&A 652, A86) and De Angeli et al. (2023, A&A 674, A2). Do not
multiply ``Sigma`` by ``s**2`` again.

Table units. The training tables store XP coefficients normalised to G = 8.5:
``table = raw / f`` with ``f = 10**((8.5 - G) / 2.5)``
(``docs/xp-covariance-experiments.md``), so ``Sigma_table = Sigma_raw / f**2``.
Optionally the G-flux uncertainty of ``f`` adds the rank-one term
``x x^T / (phot_g_mean_flux_over_error)**2`` (``x`` = normalised coefficients),
which also couples BP and RP; it is off unless requested.

Astrometry: 5x5 covariance in the order (ra, dec, parallax, pmra, pmdec), units
(mas, mas, mas, mas/yr, mas/yr); ``ra_error`` is sigma(alpha*) =
sigma(alpha) cos(dec). Parallax zero point: Lindegren et al. (2021, A&A 649, A4)
Tables 9 (5-parameter) and 10 (6-parameter), implemented here (the
``gaiadr3-zeropoint`` package is not in the environment; values checked against
its version 0.1.0 in ``tests/test_covariance_sidecar.py``). Subtract it:
``parallax_corrected = parallax - zpt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = "1"
N_COEFF = 55
N_CORR = N_COEFF * (N_COEFF - 1) // 2  # 1485
BANDS = ("bp", "rp")
NORMALISATION_G = 8.5

XP_PACKING = (
    "Gaia DR3 datamodel: off-diagonal correlations, column-major upper triangle, "
    "M[i,j] = C[j*(j-1)/2 + i] for i < j, unit diagonal implicit; equals "
    "M[numpy.tril_indices(n, -1)] = C (GaiaXPy array_to_symmetric_matrix)"
)
XP_COVARIANCE_CONVENTION = (
    "Sigma = diag(e) R diag(e) with e = published coefficient_errors (already "
    "scaled by standard_deviation s); formal covariance = Sigma / s**2 "
    "(GaiaXPy 2.2.0 correlation_to_covariance_dr3int5 and sampled-spectrum "
    "propagation: s**2 * D R D, D = diag(e/s))"
)
TABLE_NORMALISATION = (
    "table XP = raw / f, f = 10**((8.5 - phot_g_mean_mag) / 2.5); "
    "Sigma_table = Sigma_raw / f**2 (+ optional rank-one G-flux term "
    "x x^T / phot_g_mean_flux_over_error**2)"
)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    unit: str
    description: str
    width: int = 0  # 0 = scalar per star


def _xp_columns(band: str) -> list[Column]:
    b = band.upper()
    return [
        Column(f"{band}_coefficients", "f8", "e-/s",
               f"{b} basis-function coefficients (raw Gaia units)", N_COEFF),
        Column(f"{band}_coefficient_errors", "f4", "e-/s",
               f"{b} coefficient errors (published; include {band}_standard_deviation)",
               N_COEFF),
        Column(f"{band}_coefficient_correlations", "f4", "",
               f"{b} coefficient correlations, packed: {XP_PACKING}", N_CORR),
        Column(f"{band}_standard_deviation", "f4", "",
               f"{b} solution standard deviation s = sqrt(chi2/dof)"),
        Column(f"{band}_chi_squared", "f4", "", f"{b} chi-squared of the solution"),
        Column(f"{band}_n_parameters", "i2", "", f"{b} number of coefficients"),
        Column(f"{band}_n_relevant_bases", "i2", "", f"{b} number of relevant bases"),
        Column(f"{band}_basis_function_id", "i2", "", f"{b} basis-function set id"),
        Column(f"{band}_degrees_of_freedom", "i2", "", f"{b} degrees of freedom"),
        Column(f"{band}_n_measurements", "i2", "", f"{b} number of measurements"),
        Column(f"{band}_n_rejected_measurements", "i2", "",
               f"{b} number of rejected measurements"),
        Column(f"{band}_relative_shrinking", "f4", "", f"{b} relative shrinking"),
    ]  # fmt: skip


XP_COLUMNS: list[Column] = _xp_columns("bp") + _xp_columns("rp")

ASTROMETRY = ("ra", "dec", "parallax", "pmra", "pmdec")
ASTROMETRY_UNITS = ("mas", "mas", "mas", "mas/yr", "mas/yr")
# (i, j) pairs of np.triu_indices(5, 1), i.e. the Gaia correlation column order.
ASTROMETRY_CORRELATIONS = tuple(
    f"{ASTROMETRY[i]}_{ASTROMETRY[j]}_corr" for i, j in zip(*np.triu_indices(5, 1))
)

GAIA_COLUMNS: list[Column] = [
    Column("ra", "f8", "deg", "right ascension (ICRS, epoch J2016.0)"),
    Column("dec", "f8", "deg", "declination (ICRS, epoch J2016.0)"),
    Column("ra_error", "f4", "mas", "sigma(ra*) = sigma(ra) cos(dec)"),
    Column("dec_error", "f4", "mas", "sigma(dec)"),
    Column("parallax", "f8", "mas", "parallax (no zero-point correction)"),
    Column("parallax_error", "f4", "mas", "parallax error"),
    Column("pmra", "f8", "mas/yr", "proper motion in ra*"),
    Column("pmra_error", "f4", "mas/yr", "pmra error"),
    Column("pmdec", "f8", "mas/yr", "proper motion in dec"),
    Column("pmdec_error", "f4", "mas/yr", "pmdec error"),
    *[Column(c, "f4", "", "astrometric correlation") for c in ASTROMETRY_CORRELATIONS],
    Column("astrometric_params_solved", "i1", "", "3 = 2p, 31 = 5p, 95 = 6p"),
    Column("nu_eff_used_in_astrometry", "f4", "1/um", "effective wavenumber (5p)"),
    Column("pseudocolour", "f4", "1/um", "astrometric pseudocolour (6p)"),
    Column("ecl_lat", "f8", "deg", "ecliptic latitude"),
    Column("ruwe", "f4", "", "renormalised unit weight error"),
    Column("phot_g_mean_mag", "f4", "mag", "G magnitude (used for table normalisation)"),
    Column("phot_bp_mean_mag", "f4", "mag", "BP magnitude"),
    Column("phot_rp_mean_mag", "f4", "mag", "RP magnitude"),
    Column("phot_g_mean_flux", "f8", "e-/s", "G flux"),
    Column("phot_g_mean_flux_error", "f4", "e-/s", "G flux error"),
    Column("phot_g_mean_flux_over_error", "f4", "", "G flux / error"),
    Column("phot_bp_mean_flux_over_error", "f4", "", "BP flux / error"),
    Column("phot_rp_mean_flux_over_error", "f4", "", "RP flux / error"),
    Column("phot_bp_rp_excess_factor", "f4", "", "BP/RP flux excess factor"),
    Column("has_xp_continuous", "?", "", "gaia_source flag: XP continuous spectrum published"),
]  # fmt: skip

DERIVED_COLUMNS: list[Column] = [
    Column("parallax_zpt_l21", "f4", "mas",
           "Lindegren+2021 parallax zero point (subtract from parallax); NaN for 2p"),
    Column("hats_norder", "i1", "", "HATS partition order where the star was found"),
    Column("hats_npix", "i4", "", "HATS partition pixel where the star was found"),
]  # fmt: skip

INDEX_COLUMNS: list[Column] = [
    Column("source_id", "i8", "", "Gaia DR3 source_id (input order)"),
    Column("row_index", "i8", "", "row of the star in the input file"),
    Column("found_xp", "?", "", "found in the HATS xp_continuous_mean_spectrum"),
    Column("found_astrometry", "?", "", "found in the HATS gaia_source"),
]

SCHEMA: list[Column] = INDEX_COLUMNS + XP_COLUMNS + GAIA_COLUMNS + DERIVED_COLUMNS


def fill_value(dtype: str):
    """Missing-value fill for a column dtype (NaN, -1 or False)."""
    kind = np.dtype(dtype).kind
    if kind == "f":
        return np.nan
    if kind == "b":
        return False
    return -1


# --------------------------------------------------------------------------- XP


def unpack_correlations(packed: np.ndarray, n: int = N_COEFF) -> np.ndarray:
    """Packed Gaia correlations ``[..., n(n-1)/2]`` -> symmetric ``[..., n, n]``."""
    packed = np.asarray(packed)
    if packed.shape[-1] != n * (n - 1) // 2:
        raise ValueError(
            f"expected {n * (n - 1) // 2} packed values, got {packed.shape}"
        )
    dtype = np.result_type(packed.dtype, np.float32)
    out = np.zeros(packed.shape[:-1] + (n, n), dtype)
    i, j = np.tril_indices(n, -1)
    out[..., i, j] = packed
    out[..., j, i] = packed
    d = np.arange(n)
    out[..., d, d] = 1
    return out


def pack_correlations(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_correlations` (strict lower triangle, row-major)."""
    matrix = np.asarray(matrix)
    n = matrix.shape[-1]
    i, j = np.tril_indices(n, -1)
    return matrix[..., i, j]


def xp_covariance(
    errors: np.ndarray,
    packed_correlations: np.ndarray,
    standard_deviation: np.ndarray | float | None = None,
    *,
    formal: bool = False,
) -> np.ndarray:
    """Dense XP coefficient covariance ``[..., 55, 55]`` in float64, raw Gaia units.

    ``formal=False`` (default) returns ``diag(e) R diag(e)``, the covariance at the
    level of the published errors (they already include the standard deviation).
    ``formal=True`` returns the formal covariance ``diag(e/s) R diag(e/s)``.
    """
    e = np.asarray(errors, np.float64)
    if formal:
        if standard_deviation is None:
            raise ValueError("formal=True needs standard_deviation")
        e = e / np.asarray(standard_deviation, np.float64)[..., None]
    r = unpack_correlations(np.asarray(packed_correlations, np.float64), e.shape[-1])
    return (e[..., :, None] * e[..., None, :]) * r  # exactly symmetric


def xp_normalisation_factor(g_mag: np.ndarray | float) -> np.ndarray:
    """``f = 10**((8.5 - G) / 2.5)``: raw XP = table XP x f."""
    return 10.0 ** ((NORMALISATION_G - np.asarray(g_mag, np.float64)) / 2.5)


def to_table_units(
    covariance_raw: np.ndarray,
    g_mag: np.ndarray | float,
    *,
    coefficients_table: np.ndarray | None = None,
    g_flux_over_error: np.ndarray | float | None = None,
) -> np.ndarray:
    """Raw covariance -> G = 8.5 normalised (table) units: ``Sigma / f**2``.

    With ``coefficients_table`` (normalised coefficients ``x`` matching the
    covariance's last axis, e.g. 55 for one band or 110 for BP+RP) and
    ``g_flux_over_error``, the G-flux uncertainty of ``f`` is added as the
    rank-one term ``x x^T / foe**2``.
    """
    f = xp_normalisation_factor(g_mag)
    out = np.asarray(covariance_raw, np.float64) / (f**2)[..., None, None]
    if (coefficients_table is None) != (g_flux_over_error is None):
        raise ValueError(
            "pass both coefficients_table and g_flux_over_error, or neither"
        )
    if coefficients_table is not None:
        x = np.asarray(coefficients_table, np.float64)
        rel = 1.0 / np.asarray(g_flux_over_error, np.float64)
        out = out + (rel**2)[..., None, None] * (x[..., :, None] * x[..., None, :])
    return out


def joint_xp_covariance(bp_cov: np.ndarray, rp_cov: np.ndarray) -> np.ndarray:
    """Block-diagonal ``[..., 110, 110]`` from per-band covariances (no BP/RP cross term
    is published)."""
    shape = bp_cov.shape[:-2]
    n_bp, n_rp = bp_cov.shape[-1], rp_cov.shape[-1]
    out = np.zeros(shape + (n_bp + n_rp,) * 2, np.result_type(bp_cov, rp_cov))
    out[..., :n_bp, :n_bp] = bp_cov
    out[..., n_bp:, n_bp:] = rp_cov
    return out


# -------------------------------------------------------------------- astrometry


def astrometric_covariance(errors: np.ndarray, correlations: np.ndarray) -> np.ndarray:
    """5x5 covariance from ``errors [..., 5]`` (ra*, dec, parallax, pmra, pmdec) and
    ``correlations [..., 10]`` in :data:`ASTROMETRY_CORRELATIONS` order."""
    e = np.asarray(errors, np.float64)
    c = np.asarray(correlations, np.float64)
    r = np.zeros(e.shape[:-1] + (5, 5))
    i, j = np.triu_indices(5, 1)
    r[..., i, j] = c
    r[..., j, i] = c
    r[..., range(5), range(5)] = 1.0
    return (e[..., :, None] * e[..., None, :]) * r  # exactly symmetric


# Lindegren et al. (2021, A&A 649, A4), Tables 9 and 10 (coefficients in uas).
# Identical to gaiadr3_zeropoint 0.1.0 coefficients/z5_200720.txt, z6_200720.txt.
_L21_G = np.array(
    [6.0, 10.8, 11.2, 11.8, 12.2, 12.9, 13.1, 15.9, 16.1, 17.5, 19.0, 20.0, 21.0]
)
_L21_Z5_JK = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0), (4, 0))
_L21_Z5 = np.array([
    [-26.98, -9.62, 27.40, -25.1, -0.0, -1257, 0.0, 0.0],
    [-27.23, -3.07, 23.04, 35.3, 15.7, -1257, 0.0, 0.0],
    [-30.33, -9.23, 9.08, -88.4, -11.8, -1257, 0.0, 0.0],
    [-33.54, -10.08, 13.28, -126.7, 11.6, -1257, 0.0, 0.0],
    [-13.65, -0.07, 9.35, -111.4, 40.6, -1257, 0.0, 0.0],
    [-19.53, -1.64, 15.86, -66.8, 20.6, -1257, 0.0, 0.0],
    [-37.99, 2.63, 16.14, -5.7, 14.0, -1257, 107.9, 104.3],
    [-38.33, 5.61, 15.42, 0.0, 18.7, -1189, 243.8, 155.2],
    [-31.05, 2.83, 8.59, 0.0, 15.5, -1404, 105.5, 170.7],
    [-29.18, -0.09, 2.41, 0.0, 24.5, -1165, 189.7, 325.0],
    [-18.40, 5.98, -6.46, 0.0, 5.5, 0.0, 0.0, 276.6],
    [-12.65, -4.57, -7.46, 0.0, 97.9, 0.0, 0.0, 0.0],
    [-18.22, -15.24, -18.54, 0.0, 128.2, 0.0, 0.0, 0.0],
])  # fmt: skip
_L21_Z6_JK = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0))
_L21_Z6 = np.array([
    [-27.85, -7.78, 27.47, -32.1, 14.4, 9.5, -67],
    [-28.91, -3.57, 22.92, 7.7, 12.6, 1.6, -572],
    [-26.72, -8.74, 9.36, -30.3, 5.6, 17.2, -1104],
    [-29.04, -9.69, 13.63, -49.4, 36.3, 17.7, -1129],
    [-12.39, -2.16, 10.23, -92.6, 19.8, 27.6, -365],
    [-18.99, -1.93, 15.90, -57.2, -8.0, 19.9, -554],
    [-38.29, 2.59, 16.20, -10.5, 1.4, 0.4, -960],
    [-36.83, 4.20, 15.76, 22.3, 11.1, 10.0, -1367],
    [-28.37, 1.99, 9.28, 50.4, 17.2, 13.7, -1351],
    [-24.68, -1.37, 3.52, 86.8, 19.8, 21.3, -1380],
    [-15.32, 4.01, -6.03, 29.2, 14.1, 0.4, -563],
    [-13.73, -10.92, -8.30, -74.4, 196.4, -42.0, 536],
    [-29.53, -20.34, -18.74, -39.5, 326.8, -262.3, 1598],
])  # fmt: skip


def _l21(g, colour, sin_beta, table, jk):
    c = (
        np.ones_like(colour),
        np.clip(colour - 1.48, -0.24, 0.24),
        np.clip(1.48 - colour, 0.0, 0.24) ** 3,
        np.minimum(colour - 1.24, 0.0),
        np.maximum(colour - 1.72, 0.0),
    )
    b = (np.ones_like(sin_beta), sin_beta, sin_beta**2 - 1.0 / 3.0)
    n = len(_L21_G)
    ig = np.clip(np.searchsorted(_L21_G, g, side="right") - 1, 0, n - 2)
    h = np.clip((g - _L21_G[ig]) / (_L21_G[ig + 1] - _L21_G[ig]), 0.0, 1.0)
    q = (1.0 - h)[:, None] * table[ig] + h[:, None] * table[ig + 1]
    return sum(q[:, m] * c[j] * b[k] for m, (j, k) in enumerate(jk))


def parallax_zero_point_l21(
    phot_g_mean_mag,
    nu_eff_used_in_astrometry,
    pseudocolour,
    ecl_lat,
    astrometric_params_solved,
) -> tuple[np.ndarray, np.ndarray]:
    """Lindegren+2021 parallax zero point in mas, and an in-range mask.

    5p sources (``astrometric_params_solved == 31``) use ``nu_eff_used_in_astrometry``,
    6p (95) ``pseudocolour``; other solutions return NaN. Outside 6 < G < 21,
    1.1 < nu_eff < 1.9 (5p) or 1.24 < pseudocolour < 1.72 (6p) the value is the
    clamped extrapolation of the published functions (as ``gaiadr3_zeropoint``
    returns with warnings enabled) and ``in_range`` is False.
    """
    g = np.atleast_1d(np.asarray(phot_g_mean_mag, np.float64))
    nu = np.broadcast_to(np.asarray(nu_eff_used_in_astrometry, np.float64), g.shape)
    pc = np.broadcast_to(np.asarray(pseudocolour, np.float64), g.shape)
    beta = np.broadcast_to(np.asarray(ecl_lat, np.float64), g.shape)
    aps = np.broadcast_to(np.asarray(astrometric_params_solved), g.shape)
    sin_beta = np.sin(np.deg2rad(beta))
    zpt = np.full(g.shape, np.nan)
    five, six = aps == 31, aps == 95
    if five.any():
        zpt[five] = _l21(g[five], nu[five], sin_beta[five], _L21_Z5, _L21_Z5_JK)
    if six.any():
        zpt[six] = _l21(g[six], pc[six], sin_beta[six], _L21_Z6, _L21_Z6_JK)
    with np.errstate(invalid="ignore"):
        in_range = (
            (g > 6.0)
            & (g < 21.0)
            & ((five & (nu > 1.1) & (nu < 1.9)) | (six & (pc > 1.24) & (pc < 1.72)))
        )
    return zpt * 1e-3, in_range


# ------------------------------------------------------------------------ reader


class CovarianceSidecar:
    """Row access to a covariance sidecar HDF5 file.

    >>> sc = CovarianceSidecar("sidecar_table.h5")
    >>> cov = sc.xp_covariance([0, 5, 9], "bp", units="table")  # [3, 55, 55]
    """

    def __init__(self, path: str | Path):
        import h5py

        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        self.attrs = dict(self._file.attrs)
        if str(self.attrs.get("format_version")) != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported sidecar format {self.attrs}")

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __len__(self) -> int:
        return int(self._file["source_id"].shape[0])

    @property
    def columns(self) -> list[str]:
        return list(self._file.keys())

    def read(self, columns, index=None) -> dict[str, np.ndarray]:
        """Read ``columns`` for rows ``index`` (any order; None = all rows)."""
        if isinstance(columns, str):
            columns = [columns]
        if index is None:
            return {c: self._file[c][...] for c in columns}
        index = np.asarray(index, np.int64)
        scalar = index.ndim == 0
        index = np.atleast_1d(index)
        uniq, inverse = np.unique(index, return_inverse=True)
        out = {}
        for c in columns:
            ds = self._file[c]
            if len(uniq) and uniq[-1] - uniq[0] + 1 == len(uniq):
                block = ds[int(uniq[0]) : int(uniq[-1]) + 1]
            else:
                block = ds[uniq]
            values = block[inverse]
            out[c] = values[0] if scalar else values
        return out

    @staticmethod
    def _band_columns(band: str) -> list[str]:
        return [f"{band}_coefficient_errors", f"{band}_coefficient_correlations",
                f"{band}_standard_deviation", f"{band}_coefficients"]  # fmt: skip

    def xp_covariance(
        self,
        index,
        band: str,
        *,
        units: str = "raw",
        formal: bool = False,
        g_flux_term: bool = False,
    ) -> np.ndarray:
        """Per-band ``[n, 55, 55]`` covariance; ``units`` = ``"raw"`` or ``"table"``.

        ``g_flux_term`` (table units only) adds the band block of the rank-one
        G-flux term. Rows without XP are NaN.
        """
        return self._xp(index, (band,), units, formal, g_flux_term)

    def joint_xp_covariance(
        self,
        index,
        *,
        units: str = "raw",
        formal: bool = False,
        g_flux_term: bool = False,
    ) -> np.ndarray:
        """``[n, 110, 110]`` (BP then RP). Gaia publishes no BP/RP cross covariance;
        the only cross-band term is the optional G-flux rank-one term."""
        return self._xp(index, BANDS, units, formal, g_flux_term)

    def _xp(self, index, bands, units, formal, g_flux_term):
        for band in bands:
            if band not in BANDS:
                raise ValueError(band)
        if units not in ("raw", "table"):
            raise ValueError(f"units must be 'raw' or 'table', got {units!r}")
        if g_flux_term and units != "table":
            raise ValueError("g_flux_term applies to table units")
        cols = [c for b in bands for c in self._band_columns(b)]
        r = self.read(cols + ["phot_g_mean_mag", "phot_g_mean_flux_over_error"], index)
        covs = [
            xp_covariance(*(r[c] for c in self._band_columns(b)[:3]), formal=formal)
            for b in bands
        ]
        cov = covs[0] if len(covs) == 1 else joint_xp_covariance(*covs)
        if units == "raw":
            return cov
        g = r["phot_g_mean_mag"]
        if not g_flux_term:
            return to_table_units(cov, g)
        f = xp_normalisation_factor(g)
        x = (
            np.concatenate([r[f"{b}_coefficients"] for b in bands], axis=-1)
            / f[..., None]
        )
        return to_table_units(
            cov,
            g,
            coefficients_table=x,
            g_flux_over_error=r["phot_g_mean_flux_over_error"],
        )

    def table_coefficients(self, index, band: str) -> np.ndarray:
        """XP coefficients in table (G = 8.5 normalised) units."""
        r = self.read([f"{band}_coefficients", "phot_g_mean_mag"], index)
        f = xp_normalisation_factor(r["phot_g_mean_mag"])
        return r[f"{band}_coefficients"] / f[..., None]

    def astrometric_covariance(self, index) -> np.ndarray:
        errs = [f"{p}_error" for p in ASTROMETRY]
        r = self.read(errs + list(ASTROMETRY_CORRELATIONS), index)
        e = np.stack([r[c] for c in errs], axis=-1)
        c = np.stack([r[c] for c in ASTROMETRY_CORRELATIONS], axis=-1)
        return astrometric_covariance(e, c)

    def parallax_zero_point(self, index) -> tuple[np.ndarray, np.ndarray]:
        cols = ["phot_g_mean_mag", "nu_eff_used_in_astrometry", "pseudocolour",
                "ecl_lat", "astrometric_params_solved"]  # fmt: skip
        r = self.read(cols, index)
        return parallax_zero_point_l21(*(r[c] for c in cols))
