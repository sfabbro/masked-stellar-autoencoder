"""Source-only remediation primitives: sparse targets and identity-safe development roles."""

from __future__ import annotations

import numpy as np

from .joint_native_labels import SOURCE_COLUMNS, identity_hash

TARGET_NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")
ROLE_NAMES = ("fit", "tune", "cal", "devtest", "external", "excluded")


def exact_ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("Exact integer source IDs required")
    if values.dtype.kind == "u" and np.any(values > np.iinfo("i8").max):
        raise ValueError("ID outside int64")
    return values.astype("i8", copy=False)


def development_roles(ids, eligible, reserved_ids=(), historical_fit_ids=()):
    """Uncapped 70/10/10/10 development split; known exposure never held out.

    The inspected original test is consumed development data, not a fresh test.
    Reserved reference identities remain external; exposed reserved IDs excluded.
    """
    ids = exact_ids(ids)
    eligible = np.asarray(eligible, bool)
    if (
        np.any(ids <= 0)
        or len(np.unique(ids)) != len(ids)
        or eligible.shape != ids.shape
    ):
        raise ValueError("Unique positive IDs and aligned eligibility required")
    h = identity_hash(ids) % 1000
    role = np.select(
        [h < 700, h < 800, h < 900], ["fit", "tune", "cal"], "devtest"
    ).astype("S8")
    exposed = np.isin(ids, np.asarray(historical_fit_ids, dtype="i8"))
    reserved = np.isin(ids, np.asarray(reserved_ids, dtype="i8"))
    role[exposed] = b"fit"
    role[reserved] = b"external"
    role[~eligible | (reserved & exposed)] = b"excluded"
    return role


def project_raw_native(table, source, parameter_symbols=None):
    """Rebuild quality masks from original rows, preserving each survey's scale.

    GALAH's problematic flag_fe_h is deliberately unused; flag_sp and explicitly
    finite CCD3 S/N >30 are mandatory. APOGEE parameter order comes from metadata.
    """
    spec = SOURCE_COLUMNS[source]
    required = [spec["id"], *spec["value"], *spec["uncertainty"]]
    required += ["flag_sp", "snr_px_ccd3"] if source == "galah" else list(spec["flags"])
    missing = set(required) - set(table.dtype.names or ())
    if missing:
        raise ValueError(f"Missing raw {source} columns: {sorted(missing)}")
    ids = exact_ids(table[spec["id"]])
    value = np.column_stack([table[k] for k in spec["value"]]).astype("f8")
    error = np.column_stack([table[k] for k in spec["uncertainty"]]).astype("f8")
    observed = np.isfinite(value) & (value > -999) & np.isfinite(error) & (error > 0)
    observed[:, 0] &= value[:, 0] > 0
    if source == "galah":
        snr = np.asarray(table["snr_px_ccd3"])
        observed &= ((table["flag_sp"] == 0) & np.isfinite(snr) & (snr > 30))[:, None]
    else:
        observed &= ((table["ASPCAPFLAG"] == 0) & (table["STARFLAG"] == 0))[:, None]
        observed[:, 2] &= table["FE_H_FLAG"] == 0
        symbols = [
            str(s.decode() if isinstance(s, bytes) else s).strip().upper()
            for s in (parameter_symbols or [])
        ]
        if symbols.count("TEFF") != 1 or symbols.count("LOGG") != 1:
            raise ValueError("APOGEE parameter metadata required")
        flags = np.asarray(table["PARAMFLAG"])
        if flags.ndim != 2 or flags.shape[1] != len(symbols):
            raise ValueError("PARAMFLAG metadata shape mismatch")
        rejected = sum(1 << b for b in [0, 1, 2, 3, 4, 6, 16])
        for j, symbol in enumerate(("TEFF", "LOGG")):
            observed[:, j] &= (
                flags[:, symbols.index(symbol)].astype("u8") & rejected
            ) == 0
    observed &= (ids > 0)[:, None]
    observed[:, 2] &= error[:, 2] < 0.5
    return ids, value, error, observed


def first_eligible_rows(ids, eligible):
    """First eligible original row, independently for each target; no error ranking."""
    ids = exact_ids(ids)
    rows = np.flatnonzero(np.asarray(eligible, bool) & (ids > 0))
    _, first = np.unique(ids[rows], return_index=True)
    return rows[first]


def native_assignment(feature_ids, ids, observed):
    """Return per-target exact feature/measurement row pairs."""
    feature_ids, ids = exact_ids(feature_ids), exact_ids(ids)
    if np.any(np.diff(feature_ids) <= 0):
        raise ValueError("Sorted unique feature identities required")
    result = []
    for j in range(3):
        rows = first_eligible_rows(ids, observed[:, j])
        dest = np.searchsorted(feature_ids, ids[rows])
        good = dest < len(feature_ids)
        good[good] &= feature_ids[dest[good]] == ids[rows[good]]
        result.append((dest[good], rows[good]))
    return result


def role_counts(role, observed):
    return {
        name: {
            "identities": int(np.sum(role == name.encode())),
            "targets": dict(
                zip(
                    TARGET_NAMES,
                    np.sum(observed[role == name.encode()], axis=0)
                    .astype(int)
                    .tolist(),
                    strict=True,
                )
            ),
        }
        for name in ROLE_NAMES
    }


def phot_xp_columns(names):
    """Label/astrometry features cannot leak into the photometry+XP control."""
    names = np.asarray(names).astype("U").tolist()
    if len(names) != len(set(names)):
        raise ValueError("Unique feature names required")
    xp = {f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)}
    if len(xp.intersection(names)) != 110:
        raise ValueError("Complete parent 110-column XP order required")
    return np.array(
        [name in xp or name in {"W1", "W2", "G", "BP", "RP"} for name in names]
    )
