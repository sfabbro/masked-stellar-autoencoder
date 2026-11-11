"""Full record passes and fixed-parent target coordinates for expanded training."""

import numpy as np


def epoch_record_batches(lengths, batch_size, rng):
    lengths = np.asarray(lengths, dtype="i8")
    if (
        lengths.ndim != 1
        or (lengths < 0).any()
        or not lengths.sum() > 0
        or batch_size < 1
    ):
        raise ValueError("Nonnegative source sizes, positive total and batch required")
    bounds = np.cumsum(lengths)
    offsets = np.r_[0, bounds[:-1]]
    order = rng.permutation(int(bounds[-1]))
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        source = np.searchsorted(bounds, indices, side="right")
        yield source, indices - offsets[source]


def fixed_coordinates(value, error, observed, known, center, scale):
    value = np.asarray(value, dtype="f8")
    error = np.asarray(error, dtype="f8")
    observed = np.asarray(observed, bool)
    known = np.asarray(known, bool)
    center = np.asarray(center)
    scale = np.asarray(scale)
    if (
        value.ndim != 2
        or value.shape[1] != 6
        or any(v.shape != value.shape for v in [error, observed, known])
        or center.shape != (6,)
        or scale.shape != (6,)
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Aligned six-target records and fixed finite positive normalization required"
        )
    if (
        not np.isfinite(value[observed]).all()
        or (known & ~observed).any()
        or (known[:, 4:]).any()
        or not np.isfinite(error[known]).all()
        or (error[known] <= 0).any()
    ):
        raise ValueError(
            "Invalid observed labels, known errors, or posterior widths mislabelled as errors"
        )
    valid = observed.copy()
    valid[:, 4:] &= value[:, 4:] > 0
    coordinate = np.where(valid, value, 0.0)
    coordinate[:, 4:] = np.log(np.where(valid[:, 4:], value[:, 4:], 1.0))
    y = np.where(valid, (coordinate - center) / scale, 0).astype("f4")
    sigma = np.where(known, np.where(np.isfinite(error), error, 0) / scale, 0).astype(
        "f4"
    )
    return y, sigma, valid, known & valid


def reconstruction_block_start(length, batch_size, rng):
    """Sample a bounded contiguous mmap block across the full cache, not a prefix."""
    if batch_size < 1 or length < batch_size:
        raise ValueError("Reconstruction cache must fit the declared batch")
    return int(rng.integers(0, length - batch_size + 1))


XP_INPUT_NAMES = frozenset(f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56))
MISSING_TOKEN = -9999.0


def xp_input_columns(feature_names):
    """Indices of the 110 Gaia XP coefficient inputs; fail on a partial layout."""
    cols = np.array([i for i, n in enumerate(feature_names) if n in XP_INPUT_NAMES])
    if len(cols) != 110:
        raise ValueError("Complete 110-column XP input layout required")
    return cols


# Inputs a genuine no-XP Gaia star actually provides (Gaia DR3 columns only).
NOXP_GAIA_INPUTS = frozenset(
    {"G", "BP", "RP", "PARALLAX", "pmra", "pmdec", "RA", "DEC"}
)


def noxp_view_spec(feature_names, *, bprp_drop=0.19, astrometry_drop=0.18):
    """Columns to hide so an XP-hidden view matches a real no-XP Gaia star.

    Real has_xp_continuous=False reservoir stars carry only Gaia columns; about
    19% lack BP/RP and 18% lack parallax/proper motions (2-parameter
    solutions), so those groups are dropped at the same rates.
    """
    names = list(feature_names)
    xp = xp_input_columns(names)
    always = np.array(
        [
            i
            for i, n in enumerate(names)
            if n not in NOXP_GAIA_INPUTS and i not in set(xp)
        ]
    )
    groups = [
        (np.array([names.index(n) for n in ("BP", "RP") if n in names]), bprp_drop),
        (
            np.array(
                [names.index(n) for n in ("PARALLAX", "pmra", "pmdec") if n in names]
            ),
            astrometry_drop,
        ),
    ]
    return dict(xp=xp, always=always, groups=groups)


def xp_dropout_inputs(
    inputs, xp_cols, *, hide_all=False, rate=0.0, seed=None, view=None
):
    """Copy of encoder inputs with whole-XP modality dropout per star.

    Hidden XP takes the encoder's missing token, exactly as a genuine no-XP star.
    ``rate`` hides each row's XP independently; ``hide_all`` hides every row.
    With ``view`` (``noxp_view_spec``), hidden rows also lose every non-Gaia
    input and, at the measured rates, BP/RP and astrometry, so they look like
    real no-XP Gaia inputs. The input array is never modified in place.
    """
    if not (0.0 <= rate < 1.0):
        raise ValueError("XP dropout rate must lie in [0, 1)")
    if not hide_all and rate == 0.0:
        return inputs
    x = np.array(inputs, copy=True)
    rng = np.random.default_rng(seed)
    hide = np.ones(len(x), bool) if hide_all else rng.random(len(x)) < rate
    x[np.ix_(hide, xp_cols)] = MISSING_TOKEN
    if view is not None:
        rows = np.flatnonzero(hide)
        if len(view["always"]):
            x[np.ix_(rows, view["always"])] = MISSING_TOKEN
        for cols, prob in view["groups"]:
            if len(cols) and prob > 0:
                drop = rows[rng.random(len(rows)) < prob]
                x[np.ix_(drop, cols)] = MISSING_TOKEN
    return x
