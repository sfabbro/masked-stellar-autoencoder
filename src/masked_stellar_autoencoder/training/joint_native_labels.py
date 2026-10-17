"""Strict native-label projection and source-ID roles for joint stellar training.

No target imputation, abundance homogenization or missing-covariance invention.
"""

from __future__ import annotations

import numpy as np

TARGET_NAMES = ("teff", "logg", "fe_h")
TARGET_UNITS = ("K", "dex(cm/s2)", "dex")
SOURCE_COLUMNS = {
    "galah": {
        "id": "gaiadr3_source_id",
        "value": ("teff", "logg", "fe_h"),
        "uncertainty": ("e_teff", "e_logg", "e_fe_h"),
        "flags": ("flag_sp", "flag_sp_fit", "flag_fe_h"),
    },
    "apogee": {
        "id": "GAIAEDR3_SOURCE_ID",
        "value": ("TEFF", "LOGG", "FE_H"),
        "uncertainty": ("TEFF_ERR", "LOGG_ERR", "FE_H_ERR"),
        "flags": ("ASPCAPFLAG", "STARFLAG", "FE_H_FLAG", "PARAMFLAG"),
    },
}


def identity_hash(ids):
    """Match the accepted experiment identity hash exactly."""
    x = np.asarray(ids, dtype=np.uint64) ^ np.uint64(42)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def assign_roles(ids, partition, available, reference_ids):
    """Retain accepted 70/15/15 roles; split prior tuning IDs into tune/calibration.

    All supervised fit/tune/calibration IDs require original benchmark TRAIN.
    These are deterministic roles, not a claim that inspected test stars are fresh.
    """
    ids, partition, available = map(np.asarray, (ids, partition, available))
    if ids.ndim != 1 or partition.shape != ids.shape or available.shape != ids.shape:
        raise ValueError("IDs, partitions and availability must be aligned vectors")
    if not np.issubdtype(ids.dtype, np.integer) or np.any(ids <= 0):
        raise ValueError("Gaia IDs must be positive integers")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate feature IDs cannot have independent roles")
    refs = np.asarray(reference_ids)
    if refs.ndim != 1 or not np.issubdtype(refs.dtype, np.integer):
        raise ValueError("Reference IDs must be an integer vector")
    h = identity_hash(ids)
    split = np.where(h % 1000 < 700, 0, np.where(h % 1000 < 850, 1, 2)).astype("u1")
    held = np.isin(ids, refs)
    trainable = available.astype(bool) & (partition == 0) & ~held
    # A different hash quotient splits the pre-existing tuning group, not its labels.
    calibration = (h // np.uint64(1000)) % 2 == 1
    return {
        "identity_split": split,
        "reference_holdout": held,
        "benchmark_partition": partition,
        "fit": trainable & (split == 0),
        "tune": trainable & (split == 1) & ~calibration,
        "calibration": trainable & (split == 1) & calibration,
        "test": available.astype(bool) & (split == 2) & ~held,
        "external_reference": available.astype(bool) & held,
    }


def project_catalogue(table, source, selected_measurements, *, parameter_symbols=None):
    """Project exact accepted original rows and reject changed ID/FeH associations.

    Returns raw values/errors unchanged plus masks. Unusable entries stay raw;
    downstream code must use ``observed``. Alpha/M is deliberately not alpha/Fe.
    """
    if source not in SOURCE_COLUMNS:
        raise ValueError("Only explicit GALAH/APOGEE native schemas are supported")
    spec = SOURCE_COLUMNS[source]
    names = table.dtype.names
    required = [spec["id"], *spec["value"], *spec["uncertainty"], *spec["flags"]]
    missing = set(required) - set(names or ())
    if missing:
        raise ValueError(f"Missing native {source} columns: {sorted(missing)}")
    rows = np.asarray(selected_measurements["source_row"])
    if (
        not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows >= len(table))
    ):
        raise ValueError("Original source rows are out of bounds")
    raw = table[rows]
    if not np.array_equal(raw[spec["id"]], selected_measurements["source_id"]):
        raise ValueError("Original source-row Gaia IDs differ from accepted native IDs")
    value = np.column_stack([raw[k] for k in spec["value"]]).astype("f8")
    uncertainty = np.column_stack([raw[k] for k in spec["uncertainty"]]).astype("f8")
    if not np.array_equal(value[:, 2], selected_measurements["value"], equal_nan=True):
        raise ValueError("Original FeH values differ from accepted native measurements")
    if not np.array_equal(
        uncertainty[:, 2], selected_measurements["uncertainty"], equal_nan=True
    ):
        raise ValueError(
            "Original FeH uncertainties differ from accepted native measurements"
        )
    observed = np.isfinite(value) & np.isfinite(uncertainty) & (uncertainty > 0)
    observed &= value > -999  # Native -9999 placeholders are retained raw, never used.
    observed[:, 0] &= value[:, 0] > 0
    quality = np.ones_like(observed)
    if source == "galah":
        global_good = (raw["flag_sp"] == 0) & (raw["flag_sp_fit"] == 0)
        quality &= global_good[:, None]
        quality[:, 2] &= raw["flag_fe_h"] == 0
    else:
        global_good = (raw["ASPCAPFLAG"] == 0) & (raw["STARFLAG"] == 0)
        quality &= global_good[:, None]
        quality[:, 2] &= raw["FE_H_FLAG"] == 0
        # Metadata establishes native PARAMFLAG ordering; never guess vector indices.
        symbols = [
            str(s.decode() if isinstance(s, bytes) else s).strip().upper()
            for s in ([] if parameter_symbols is None else parameter_symbols)
        ]
        if symbols.count("TEFF") != 1 or symbols.count("LOGG") != 1:
            raise ValueError("APOGEE PARAM_SYMBOL must identify TEFF and LOGG uniquely")
        flags = raw["PARAMFLAG"]
        if flags.ndim != 2 or flags.shape[1] != len(symbols):
            raise ValueError("APOGEE PARAMFLAG dimensions differ from PARAM_SYMBOL")
        # Reject bad/fixed parameters; retain informational RC/RGB calibration bits.
        rejected_bits = sum(1 << b for b in [0, 1, 2, 3, 4, 6, 16])
        for j, symbol in enumerate(["TEFF", "LOGG"]):
            quality[:, j] &= (
                flags[:, symbols.index(symbol)].astype("u8") & rejected_bits
            ) == 0
    observed &= quality
    observed[:, 2] &= uncertainty[:, 2] < 0.5
    return {
        "value": value,
        "uncertainty": uncertainty,
        "observed": observed,
        "quality_pass": quality,
        "raw": {name: np.asarray(raw[name]).copy() for name in required},
        "logg_parallax_conditioned": np.full(len(raw), source == "galah", dtype=bool),
    }
