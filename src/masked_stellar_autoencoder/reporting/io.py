"""Strict JSON/CSV I/O for the C6 metric contract.

Two invariants drive this module:

1. **No bare NaN / Infinity ever reaches a file.**  A non-finite number is a
   failed computation, and the only legal way to publish it is ``null`` plus a
   :class:`~masked_stellar_autoencoder.reporting.schema.NullReason`.  Writing a
   ``NaN`` raises; writing a non-finite float through :func:`build_row` requires
   the caller to name the reason.
2. **Failed verdicts are append-only.**  :func:`update_certificate_verdicts`
   merges new gate results into an existing certificate without dropping an
   earlier failure (the plan's "never erase an old failed verdict").

The emitted directory is exactly the section 7.1 contract::

    <dir>/metrics.csv, joint.csv, distance.csv, age.csv
    <dir>/prediction-certificate.json
    <dir>/schema.json
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import schema
from .schema import (
    TABLE_FIELDS,
    TABLE_FILENAMES,
    null_reason_ok,
)

#: Row fields that do not count towards "a null cell needs a null_reason":
#: pure metadata (identity, notes, derived flag), optional counts, and the
#: additive R1 calibration/theory annotations.  The last group is deliberately
#: excluded so a complete 1.0.0 row (no Mondrian/theory product) still validates
#: unchanged after the optional columns were added; a producer that does run the
#: product fills them, and a producer that does not leaves them blank rather than
#: inventing a null_reason for a metric it never computed.
_NON_DATA_FIELDS = frozenset(
    {
        "null_reason",
        "notes",
        "failure_id",
        "verified",
        "model_id",
        "bootstrap_reps",
        "gt10Gyr_n",
        "calibration_kind",
        "pooled_strata",
        "theory_sigma",
        "gamma",
        "theory_envelope_low",
        "theory_envelope_high",
    }
)


class ContractError(ValueError):
    """Base class for a schema/contract violation."""


class NonFiniteValueError(ContractError):
    """A ``NaN``/``Infinity`` would have been written or validated."""


class NullReasonError(ContractError):
    """A null cell lacks a valid reason, or a reason is stale."""


def _fields(table: str) -> tuple[schema.Field, ...]:
    if table not in TABLE_FIELDS:
        raise ContractError(
            f"unknown table {table!r}; expected one of {sorted(TABLE_FIELDS)}"
        )
    return TABLE_FIELDS[table]


def _field_map(table: str) -> dict[str, schema.Field]:
    return {field.name: field for field in _fields(table)}


def _columns(table: str) -> tuple[str, ...]:
    return tuple(field.name for field in _fields(table))


def _required_columns(table: str) -> list[str]:
    return [field.name for field in _fields(table) if field.required]


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_range(field: schema.Field, number: float) -> None:
    if field.valid_range is None:
        return
    low, high = field.valid_range
    tol = 1e-9 * max(
        1.0,
        abs(low) if low is not None else 1.0,
        abs(high) if high is not None else 1.0,
    )
    if low is not None and number < low - tol:
        raise ContractError(f"{field.name}: {number!r} below valid_range low {low!r}")
    if high is not None and number > high + tol:
        raise ContractError(f"{field.name}: {number!r} above valid_range high {high!r}")


def _coerce(field: schema.Field, value):
    value = _python_scalar(value)
    if value is None:
        return None
    if field.dtype == "str":
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            raise ContractError(
                f"{field.name}: expected str, got {type(value).__name__}"
            )
        return value
    if field.dtype == "bool":
        if isinstance(value, bool | np.bool_):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in ("true", "1", "yes"):
                return True
            if token in ("false", "0", "no"):
                return False
        if isinstance(value, int | np.integer) and not isinstance(value, bool):
            if int(value) in (0, 1):
                return bool(int(value))
        raise ContractError(f"{field.name}: expected bool, got {value!r}")
    if field.dtype == "int":
        if isinstance(value, bool):
            raise ContractError(f"{field.name}: bool is not an int")
        if (
            isinstance(value, int | np.integer)
            or isinstance(value, float)
            and value.is_integer()
        ):
            number = int(value)
        elif isinstance(value, str):
            try:
                number = int(value)
            except ValueError as error:
                raise ContractError(f"{field.name}: {value!r} is not an int") from error
        else:
            raise ContractError(
                f"{field.name}: expected int, got {type(value).__name__}"
            )
        _check_range(field, number)
        return number
    if field.dtype == "float":
        if isinstance(value, bool):
            raise ContractError(f"{field.name}: bool is not a float")
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError as error:
                raise ContractError(
                    f"{field.name}: {value!r} is not a float"
                ) from error
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"{field.name}: expected float, got {type(value).__name__}"
            ) from error
        if not math.isfinite(number):
            raise NonFiniteValueError(
                f"{field.name}: non-finite value {number!r}; emit null with a null_reason"
            )
        _check_range(field, number)
        return number
    raise ContractError(f"unknown dtype {field.dtype!r} for field {field.name!r}")


def _check_enum(field: schema.Field, value) -> None:
    if value is None or field.enum is None:
        return
    if value not in field.enum:
        raise ContractError(
            f"{field.name}: {value!r} not in allowed values {field.enum}"
        )


def _data_fields(table: str) -> list[str]:
    return [
        field.name for field in _fields(table) if field.name not in _NON_DATA_FIELDS
    ]


def finite_or_null(value, *, null_reason: str) -> tuple[float | None, str]:
    """Convert a non-finite metric to ``None`` and pair it with its reason.

    Returns ``(value_or_None, null_reason)``.  A finite value keeps the reason
    unchanged (usually ``""``).  This is the low-level hook used by row
    builders; a producer that has a ``NaN`` must say *why*.
    """
    if value is None:
        return None, null_reason
    number = float(_python_scalar(value))
    if not math.isfinite(number):
        return None, null_reason
    return number, null_reason


def validate_row(table: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one row, returning a normalised copy (schema column order).

    Enforces: known keys only, required keys present, dtypes/enums/ranges,
    no bare ``NaN``, ``null_reason`` iff some metric cell is null, and the
    cross-field provenance rules (failed verdict -> ``failure_id``,
    ``verified`` consistent with ``evidence_status``, ``nll`` paired with
    ``nll_space``).
    """
    field_map = _field_map(table)
    extra = sorted(set(row) - set(field_map))
    if extra:
        raise ContractError(f"{table}: unknown columns {extra}")
    missing = [name for name in _required_columns(table) if name not in row]
    if missing:
        raise ContractError(f"{table}: missing required columns {missing}")

    normalised: dict[str, Any] = {}
    for name, field in field_map.items():
        if name not in row:
            normalised[name] = None
            continue
        value = _coerce(field, row[name])
        if value is None and not field.nullable:
            raise ContractError(f"{table}.{name}: required value is null")
        if name != "null_reason":  # validated below with the enum spelled out
            _check_enum(field, value)
        normalised[name] = value

    # Derived provenance flag, consistent by construction when inferred.
    evidence = normalised.get("evidence_status")
    verified = normalised.get("verified")
    if verified is None and evidence is not None:
        normalised["verified"] = evidence == "verified"
        verified = normalised["verified"]
    elif verified is not None and evidence is not None:
        if verified != (evidence == "verified"):
            raise ContractError(
                f"{table}.verified={verified!r} contradicts evidence_status={evidence!r}"
            )
    if evidence == "strict_replay_failed" and not normalised.get("failure_id"):
        raise ContractError(
            f"{table}: evidence_status='strict_replay_failed' requires a failure_id back-reference"
        )

    null_reason = normalised.get("null_reason")
    if null_reason is not None and not null_reason_ok(null_reason):
        raise ContractError(
            f"{table}.null_reason={null_reason!r} not in the enum {schema.NULL_REASONS}"
        )
    nulls = [name for name in _data_fields(table) if normalised.get(name) is None]
    if nulls and not null_reason:
        raise NullReasonError(
            f"{table}: null cells {nulls} require a null_reason from {schema.NULL_REASONS}"
        )
    if not nulls and null_reason:
        raise NullReasonError(
            f"{table}: null_reason={null_reason!r} set but no metric cell is null"
        )

    if table == "metrics":
        if normalised.get("nll") is not None and normalised.get("nll_space") is None:
            raise ContractError(
                "metrics: an NLL row must state nll_space (intrinsic or observed_label)"
            )
        if normalised.get("nll") is None and normalised.get("nll_space") is not None:
            raise ContractError("metrics: nll_space set without an nll value")
        if (
            normalised.get("cov_90") is not None
            and normalised.get("cov_nominal") is None
        ):
            raise ContractError("metrics: cov_90 requires cov_nominal")

    if table == "joint":
        method = normalised.get("levelset_method")
        if (
            method not in (None, "none")
            and normalised.get("region_volume_native") is None
        ):
            if not null_reason:
                raise ContractError(
                    "joint: a level-set method requires region_volume_native (or a null_reason)"
                )
    return normalised


def validate_rows(
    table: str, rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate every row of a table; returns the normalised rows."""
    return [validate_row(table, row) for row in rows]


def build_row(
    table: str, values: Mapping[str, Any] | None = None, **extra: Any
) -> dict[str, Any]:
    """Fill one contract row from partial ``values`` and validate it.

    Missing columns become ``None``; non-finite floats become ``None`` and
    require ``null_reason`` (either supplied or via ``values["null_reason"]``).
    Unknown keys raise.  ``verified`` is derived from ``evidence_status`` when
    omitted.
    """
    field_map = _field_map(table)
    payload: dict[str, Any] = dict(values or {})
    payload.update(extra)
    unknown = sorted(set(payload) - set(field_map))
    if unknown:
        raise ContractError(f"{table}: unknown fields {unknown}")

    row: dict[str, Any] = {}
    for name, field in field_map.items():
        value = payload.get(name)
        value = _python_scalar(value)
        if field.dtype in ("float", "int") and value is not None:
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    f"{table}.{name}: {value!r} is not numeric"
                ) from error
            if not math.isfinite(number):
                row[name] = None
                continue
        row[name] = value
    return validate_row(table, row)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def _format_cell(field: schema.Field, value) -> str:
    if value is None:
        return ""
    if field.dtype == "bool":
        return "true" if value else "false"
    if field.dtype == "int":
        return str(int(value))
    if field.dtype == "float":
        return repr(float(value))
    return str(value)


def _parse_cell(field: schema.Field, text: str):
    if text == "":
        return None
    return text


def write_csv(path, table: str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Validate then atomically write one contract table.

    Every schema column is written (required ones must be present); nulls are
    empty cells, never ``nan``/``inf``.  The file is replaced atomically so a
    half-written table cannot be read as a valid one.
    """
    fields = _fields(table)
    normalised = validate_rows(table, rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([field.name for field in fields])
        for row in normalised:
            writer.writerow([_format_cell(field, row[field.name]) for field in fields])
    os.replace(tmp, target)
    return target


def read_csv(path, table: str, *, validate: bool = True) -> list[dict[str, Any]]:
    """Read one contract table, reindexing to schema order and validating."""
    field_map = _field_map(table)
    source = Path(path)
    with source.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        unknown = sorted(set(header) - set(field_map))
        if unknown:
            raise ContractError(f"{source}: unknown columns {unknown}")
        missing = [name for name in _required_columns(table) if name not in header]
        if missing:
            raise ContractError(f"{source}: missing required columns {missing}")
        rows = []
        for record in reader:
            row = {
                name: _parse_cell(field, record.get(name, "") or "")
                for name, field in field_map.items()
            }
            rows.append(row)
    return validate_rows(table, rows) if validate else rows


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def write_json(path, payload) -> Path:
    """Atomically write JSON, refusing ``NaN``/``Infinity`` outright."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = json.dumps(payload, indent=2, allow_nan=False, sort_keys=False)
    except ValueError as error:
        raise NonFiniteValueError(
            f"{target}: JSON payload contains non-finite numbers ({error}); "
            "use null + null_reason"
        ) from error
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_schema_json(path, *, provenance: Mapping[str, str] | None = None) -> Path:
    """Write the versioned ``schema.json`` (fields + enums + gates + provenance)."""
    return write_json(path, schema.schema_document(provenance=provenance))


def validate_certificate_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Validate (and return) a certificate document."""
    schema.validate_certificate(doc)
    return dict(doc)


def write_certificate(path, certificate: Mapping[str, Any]) -> Path:
    """Validate then atomically write ``prediction-certificate.json``."""
    schema.validate_certificate(certificate)
    return write_json(path, certificate)


def read_certificate(path, *, validate: bool = True) -> dict[str, Any]:
    doc = read_json(path)
    if validate:
        schema.validate_certificate(doc)
    return doc


def update_certificate_verdicts(
    path,
    updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Read-modify-write gate verdicts, preserving every earlier failure.

    ``updates`` maps gate -> status string or entry mapping (see
    :func:`schema.merge_verdicts`).  An old ``fail`` stays in ``history`` and in
    ``failed_verdicts`` forever, so a later pass cannot erase it.
    """
    certificate = read_certificate(path)
    schema.merge_verdicts(certificate, updates)
    schema.validate_certificate(certificate)
    write_json(path, certificate)
    return certificate


# ---------------------------------------------------------------------------
# whole-contract directory
# ---------------------------------------------------------------------------
def write_contract_directory(
    directory,
    *,
    metrics: Iterable[Mapping[str, Any]] | None = None,
    joint: Iterable[Mapping[str, Any]] | None = None,
    distance: Iterable[Mapping[str, Any]] | None = None,
    age: Iterable[Mapping[str, Any]] | None = None,
    certificate: Mapping[str, Any] | None = None,
    schema_provenance: Mapping[str, str] | None = None,
    write_schema: bool = True,
) -> dict[str, Path]:
    """Write the complete section 7.1 contract into one immutable run directory.

    Any table left as ``None`` is simply not written; every supplied table is
    validated first.  Returns a mapping of emitted names to paths.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, rows in (
        ("metrics", metrics),
        ("joint", joint),
        ("distance", distance),
        ("age", age),
    ):
        if rows is not None:
            written[name] = write_csv(root / TABLE_FILENAMES[name], name, rows)
    if certificate is not None:
        written["certificate"] = write_certificate(
            root / "prediction-certificate.json", certificate
        )
    if write_schema:
        written["schema"] = write_schema_json(
            root / "schema.json", provenance=schema_provenance
        )
    return written
