"""Deterministic star/group split construction for multi-survey experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np

ROLES = ("fit", "tune", "cal", "test")


def build_group_split(
    source_ids: Sequence[int],
    groups: Mapping[str, Sequence[int]],
    *,
    role_fractions: Mapping[str, float],
    salt: str,
    excluded_ids: Sequence[int] = (),
    fixed_roles: Mapping[int, str] | None = None,
) -> dict:
    """Split unique stars without splitting any declared cluster/family group.

    Group IDs must be explicitly namespaced by the caller (e.g. ``cluster:X``
    versus ``anchor-family:Y``). Transitive overlaps form one component. Any
    component containing an explicitly excluded identity is entirely excluded.
    Existing roles can be locked for supplied stars; conflicting locked roles in
    the same component are rejected. Exclusions override locked roles explicitly
    and are counted. This does not discover missing ancestors or certify group membership.
    Fractions describe hash probabilities, not guaranteed realized counts.
    A changed group component can change its assigned role: freeze the
    complete manifest before fitting and do not append sources mid-experiment.
    """
    ids = np.asarray(source_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("source_ids must be one-dimensional exact integers")
    if ids.dtype.kind == "u" and np.any(ids > np.iinfo("i8").max):
        raise ValueError("source_id exceeds signed int64")
    ids = ids.astype("i8", copy=False)
    if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("source_ids must be positive and unique")
    ids = np.sort(ids)
    if not isinstance(salt, str) or not salt:
        raise ValueError("An explicit nonempty split salt is required")
    if set(role_fractions) != set(ROLES):
        raise ValueError("All and only fit/tune/cal/test fractions are required")
    weights = np.asarray([role_fractions[r] for r in ROLES], dtype=np.float64)
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("Role fractions must be finite, nonnegative and sum to one")
    index = {int(v): i for i, v in enumerate(ids)}
    parent = np.arange(len(ids), dtype="i8")

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    canonical_groups = []
    for name, members in sorted(groups.items()):
        if (
            not isinstance(name, str)
            or ":" not in name
            or name.startswith(":")
            or name.endswith(":")
        ):
            raise ValueError("Group names require a nonempty namespace and name")
        raw = np.asarray(members)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise ValueError("Group members must be exact integer source IDs")
        if raw.dtype.kind == "u" and np.any(raw > np.iinfo("i8").max):
            raise ValueError("Group source ID exceeds signed int64")
        member_ids = np.unique(raw.astype("i8", copy=False))
        if not len(member_ids) or any(int(v) not in index for v in member_ids):
            raise ValueError("Groups must contain known source IDs")
        positions = [index[int(v)] for v in member_ids]
        for pos in positions[1:]:
            union(positions[0], pos)
        canonical_groups.append([name, member_ids.tolist()])
    excluded = np.asarray(excluded_ids)
    if excluded.size and (excluded.ndim != 1 or excluded.dtype.kind not in "iu"):
        raise ValueError("Excluded IDs must be exact integers")
    excluded_set = {int(v) for v in excluded}
    if any(v <= 0 for v in excluded_set):
        raise ValueError("Excluded IDs must be positive")
    fixed = {}
    for ident, role in (fixed_roles or {}).items():
        if isinstance(ident, bool | np.bool_) or not isinstance(
            ident, int | np.integer
        ):
            raise ValueError("Fixed-role identities must be exact integers")
        ident = int(ident)
        if ident not in index:
            raise ValueError("Fixed-role identities must be in the candidate universe")
        if role not in (*ROLES, "devtest", "external", "excluded"):
            raise ValueError("Unknown fixed role")
        fixed[ident] = role
    components = {}
    for pos, ident in enumerate(ids):
        components.setdefault(find(pos), []).append(int(ident))
    cumulative = np.cumsum(weights)
    by_id = {}
    excluded_matches = 0
    overridden_fixed = 0
    for component in components.values():
        intersects = any(ident in excluded_set for ident in component)
        excluded_matches += sum(ident in excluded_set for ident in component)
        locked = {fixed[ident] for ident in component if ident in fixed}
        if intersects or "excluded" in locked:
            role = "excluded"
            overridden_fixed += sum(
                fixed.get(ident) not in (None, "excluded") for ident in component
            )
        elif len(locked) > 1:
            raise ValueError("A declared group connects conflicting fixed roles")
        elif locked:
            role = locked.pop()
        else:
            payload = json.dumps([salt, component], separators=(",", ":")).encode()
            random_value = (
                int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
            )
            role = ROLES[
                min(int(np.searchsorted(cumulative, random_value, side="right")), 3)
            ]
        by_id.update({ident: role for ident in component})
    assignments = [[int(ident), by_id[int(ident)]] for ident in ids]
    contract = {
        "schema": "multisurvey-star-group-split-v1",
        "salt": salt,
        "role_fractions": {r: float(role_fractions[r]) for r in ROLES},
        "groups": canonical_groups,
        "excluded_ids": sorted(excluded_set),
        "fixed_roles": [[ident, fixed[ident]] for ident in sorted(fixed)],
        "assignments": assignments,
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "contract": contract,
        "sha256": digest,
        "counts": {
            r: sum(role == r for _, role in assignments)
            for r in (*ROLES, "devtest", "external", "excluded")
        },
        "components": len(components),
        "matched_excluded_ids": excluded_matches,
        "overridden_fixed_ids": overridden_fixed,
        "unmatched_excluded_ids": len(excluded_set) - excluded_matches,
        "scope": "Declared groups/exclusions only; no independent ancestry or physical-anchor certification",
    }
