"""FIT-source availability after the sparse collator's repeat resolution.

This is a source/reference census, not a validation of survey errors, labels,
independence, astrophysical truth, or the completeness of source ancestry.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .multisurvey_data import SparseSurveyDataset
from .multisurvey_model import _is_source_binding
from .multisurvey_overlay import TARGET_NAMES


def _boolean_column(h5, name, rows, n):
    if name not in h5:
        return None
    data = np.asarray(h5[name][:])
    if (
        data.shape != (n,)
        or data.dtype.kind not in "biu"
        or np.any((data != 0) & (data != 1))
    ):
        raise ValueError(f"{name} must be an aligned boolean column")
    return data[rows].astype(bool, copy=False)


def _target_counts(mask):
    return {name: int(mask[:, j].sum()) for j, name in enumerate(TARGET_NAMES)}


def _overlap_counts(active, names):
    bit = np.zeros(len(active), dtype=np.uint64)
    for j in range(len(names)):
        bit |= active[:, j].astype(np.uint64) << np.uint64(j)
    unique, counts = np.unique(bit, return_counts=True)
    return {
        "+".join(names[j] for j in range(len(names)) if int(value) & (1 << j))
        or "none": int(count)
        for value, count in zip(unique, counts, strict=True)
    }


def census_fit_collation(dataset: SparseSurveyDataset) -> dict:
    """Count the exact selected and all-repeat reference masks for FIT stars.

    ``dataset`` must be a digest-bound FIT reader restricted to primary-input
    eligible rows. The selection mirrors ``lowest_source_row``: only active
    records (at least one observed target) enter repeat resolution; the smallest
    original source row wins. The union uses all active same-source repeats and
    is a diagnostic of information removed by that policy, not a recommended
    procedure for treating dependent repeats as independent observations.

    This reads source arrays, never a model. Per-star arrays permit independent
    comparisons and retain exact source identities. ``input_has_xp=False`` is a
    statement about staged input features, never proof of official XP absence.
    """
    if not isinstance(dataset, SparseSurveyDataset):
        raise ValueError("A verified sparse survey reader is required")
    if not _is_source_binding(dataset.source_binding) or (
        dataset._direct_source_stat is None and dataset._verified_overlay_access is None
    ):
        raise ValueError("A content-verified source or overlay is required")
    if not 0 < len(dataset.names) <= 63:
        raise ValueError("Between one and 63 distinct sources are required")
    dataset._check_source_binding()
    h5 = dataset._handle()
    n_base = len(dataset.ids)
    roles = np.asarray(h5["role"][:]).astype("U")
    primary = _boolean_column(h5, "primary_input_eligible", np.arange(n_base), n_base)
    if primary is None or roles.shape != (n_base,):
        raise ValueError("Explicit aligned role and primary-input masks are required")
    fit_rows = np.flatnonzero((roles == "fit") & primary)
    if not np.array_equal(dataset.star_rows, fit_rows):
        raise ValueError("Reader must contain exactly primary-eligible FIT stars")
    ids = np.asarray(dataset.ids[fit_rows], dtype="<i8")
    if len(ids) != len(np.unique(ids)) or np.any(ids <= 0) or np.any(np.diff(ids) <= 0):
        raise ValueError("FIT source identities must be positive, sorted, and unique")
    n, sources = len(ids), len(dataset.names)
    selected = np.zeros((n, sources, len(TARGET_NAMES)), dtype=bool)
    repeated_union = np.zeros_like(selected)
    selected_error_known = np.zeros_like(selected)
    selected_row = np.full((n, sources), -1, dtype=np.int64)
    active_repeats = {}
    seen_dependencies = {}
    for j, name in enumerate(dataset.names):
        star_rows, native_rows = dataset._indices[name]
        positions = np.searchsorted(fit_rows, star_rows)
        in_fit = positions < n
        matched = np.flatnonzero(in_fit)
        in_fit[matched] &= fit_rows[positions[matched]] == star_rows[matched]
        native = native_rows[in_fit]
        pos = positions[in_fit]
        group = h5["native_observations"][name]
        raw_rows = np.asarray(group["source_row"][:])
        if raw_rows.dtype.kind not in "iu" or (
            raw_rows.dtype.kind == "u" and np.any(raw_rows > np.iinfo("i8").max)
        ):
            raise ValueError(f"{name} source rows must be exact int64 identities")
        row = raw_rows[native].astype(np.int64, copy=False)
        if np.any(row < 0):
            raise ValueError(f"{name} source rows must be nonnegative")
        observed = np.asarray(group["observed"][:], dtype=bool)[native]
        values = np.asarray(group["value"][:], dtype=np.float64)[native]
        errors = np.asarray(group["error"][:], dtype=np.float64)[native]
        comparable = np.array(
            [
                dataset.registry[name]["targets"][target] is not None
                for target in TARGET_NAMES
            ]
        )
        if np.any(observed & ~np.isfinite(values)) or np.any(observed & ~comparable):
            raise ValueError(f"{name} has invalid or unmapped observed labels")
        active = np.any(observed, axis=1)
        pos, row, observed, errors = (
            pos[active],
            row[active],
            observed[active],
            errors[active],
        )
        for target in range(len(TARGET_NAMES)):
            np.logical_or.at(repeated_union[:, j, target], pos, observed[:, target])
        dependency = dataset.registry[name]["dependency_id"]
        source_presence = np.zeros(n, dtype=bool)
        source_presence[pos] = True
        if dependency in seen_dependencies and np.any(
            source_presence & seen_dependencies[dependency]
        ):
            raise ValueError("Unresolved dependent observations across sources")
        if dependency in seen_dependencies:
            seen_dependencies[dependency] |= source_presence
        else:
            seen_dependencies[dependency] = source_presence
        order = np.lexsort((row, pos))
        ordered_pos, ordered_row = pos[order], row[order]
        same_star = ordered_pos[1:] == ordered_pos[:-1]
        if np.any(same_star & (ordered_row[1:] == ordered_row[:-1])):
            raise ValueError("Repeated measurements have duplicate source rows")
        first = np.r_[True, ~same_star] if len(order) else np.zeros(0, bool)
        chosen = order[first]
        selected[pos[chosen], j] = observed[chosen]
        selected_row[pos[chosen], j] = row[chosen]
        selected_error_known[pos[chosen], j] = (
            observed[chosen] & np.isfinite(errors[chosen]) & (errors[chosen] > 0)
        )
        run_lengths = np.diff(np.r_[np.flatnonzero(first), len(order)])
        active_repeats[name] = int(np.count_nonzero(run_lengths > 1))
    active_sources = selected.any(axis=2)
    selected_union = selected.any(axis=1)
    all_union = repeated_union.any(axis=1)
    if np.any(selected & ~repeated_union):
        raise AssertionError("Selected observations must be a subset of repeat union")
    input_xp = _boolean_column(h5, "input_has_xp", fit_rows, n_base)
    real_xp = _boolean_column(h5, "real_xp_present", fit_rows, n_base)
    real_no_xp = _boolean_column(h5, "real_no_xp", fit_rows, n_base)
    if (real_xp is None) != (real_no_xp is None):
        raise ValueError("Actual XP status requires both explicit status masks")
    if real_xp is not None and np.any(real_xp & real_no_xp):
        raise ValueError("Explicit XP and no-XP masks must be disjoint")
    by_input_xp = None
    if input_xp is not None:
        by_input_xp = {
            key: {
                "stars": int(mask.sum()),
                "selected_targets": _target_counts(selected_union[mask]),
                "source_combinations": _overlap_counts(
                    active_sources[mask], dataset.names
                ),
            }
            for key, mask in (
                ("input_xp_present", input_xp),
                ("input_xp_not_staged_or_missing", ~input_xp),
            )
        }
    status = None
    if real_xp is not None:
        status = {
            "explicit_xp_present": int(real_xp.sum()),
            "explicit_no_xp": int(real_no_xp.sum()),
            "other_or_unknown": int((~(real_xp | real_no_xp)).sum()),
        }
    per_source = {}
    for j, name in enumerate(dataset.names):
        per_source[name] = {
            "stars_any_selected_target": int(active_sources[:, j].sum()),
            "active_repeated_stars": active_repeats[name],
            "selected_targets": _target_counts(selected[:, j]),
            "all_repeat_union_targets": _target_counts(repeated_union[:, j]),
            "targets_lost_by_selection": _target_counts(
                repeated_union[:, j] & ~selected[:, j]
            ),
            "selected_known_error_targets": _target_counts(selected_error_known[:, j]),
            "selected_unknown_error_targets": _target_counts(
                selected[:, j] & ~selected_error_known[:, j]
            ),
        }
    dataset._check_source_binding()
    summary = {
        "schema": "multisurvey-fit-collation-census-v1",
        "source_binding": dataset.source_binding,
        "registry_sha256": dataset.registry_sha256,
        "fit_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "selection": "lowest original source row among active same-source repeats",
        "scope": "FIT and primary-input-eligible source reference availability only; not a quality, independence, physical-scale or model certificate",
        "eligible_fit_stars": n,
        "stars_any_selected_target": int(selected_union.any(axis=1).sum()),
        "sources": list(dataset.names),
        "per_source": per_source,
        "source_combinations": _overlap_counts(active_sources, dataset.names),
        "selected_targets_any_source": _target_counts(selected_union),
        "all_repeat_union_targets_any_source": _target_counts(all_union),
        "targets_lost_all_sources": _target_counts(all_union & ~selected_union),
        "input_xp_modes": by_input_xp,
        "explicit_xp_status_counts": status,
        "xp_scope": "Input XP absence is not evidence of official no XP; unknown official status stays unknown",
    }
    arrays = {
        "source_id": ids,
        "selected_source_row": selected_row,
        "selected_observed": selected,
        "all_repeat_union_observed": repeated_union,
        "selected_error_known": selected_error_known,
    }
    if input_xp is not None:
        arrays["input_has_xp"] = input_xp
    if real_xp is not None:
        arrays.update(real_xp_present=real_xp, real_no_xp=real_no_xp)
    return {"summary": summary, "arrays": arrays}
