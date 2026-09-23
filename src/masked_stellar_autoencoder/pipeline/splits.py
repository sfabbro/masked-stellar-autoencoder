"""Splits are by source_id. A star is in one split only."""

from __future__ import annotations

import numpy as np

SPLIT_NAMES = ("train", "select", "calibrate", "test")


class SourceLeak(ValueError):
    """A source_id landed in train and in another split."""


def refuse_train_overlap(
    train_source_ids: np.ndarray, other_source_ids: np.ndarray
) -> None:
    overlap = np.intersect1d(np.asarray(train_source_ids), np.asarray(other_source_ids))
    if overlap.size:
        raise SourceLeak(f"source_id {overlap[0]} is already in train")


def four_way_split(
    source_id: np.ndarray,
    *,
    fractions: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Disjoint train / select / calibrate / test masks. Select is not calibrate."""
    if len(fractions) != 4 or any(f <= 0 for f in fractions):
        raise ValueError(f"fractions must be four positive numbers, got {fractions}")
    source_id = np.asarray(source_id)
    uniq = np.unique(source_id)
    if uniq.size < 4:
        raise ValueError(f"four-way split needs at least 4 source_ids, got {uniq.size}")
    order = np.random.default_rng(seed).permutation(uniq)
    n = int(order.size)
    n_test = max(1, int(round(fractions[3] / sum(fractions) * n)))
    n_cal = max(1, int(round(fractions[2] / sum(fractions) * n)))
    n_sel = max(1, int(round(fractions[1] / sum(fractions) * n)))
    if n_test + n_cal + n_sel >= n:
        n_test = n_cal = n_sel = 1
    chunks = {
        "test": order[:n_test],
        "calibrate": order[n_test : n_test + n_cal],
        "select": order[n_test + n_cal : n_test + n_cal + n_sel],
        "train": order[n_test + n_cal + n_sel :],
    }
    for name in ("select", "calibrate", "test"):
        refuse_train_overlap(chunks["train"], chunks[name])
    return {name: np.isin(source_id, ids) for name, ids in chunks.items()}


def leave_one_survey_out(
    source_id: np.ndarray,
    survey: np.ndarray,
    held_out: str,
    *,
    train_source_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Test is every star that appears in the held-out survey.

    A star observed in two surveys is not left in train. Pass ``train_source_ids``
    to reject a caller-supplied train set that already contains one of those stars.
    """
    source_id = np.asarray(source_id)
    survey = np.asarray(survey).astype(str)
    test_ids = np.unique(source_id[survey == held_out])
    if train_source_ids is None:
        train_ids = np.unique(source_id[~np.isin(source_id, test_ids)])
    else:
        train_ids = np.unique(np.asarray(train_source_ids))
    refuse_train_overlap(train_ids, test_ids)
    train_idx = np.flatnonzero(np.isin(source_id, train_ids))
    test_idx = np.flatnonzero(np.isin(source_id, test_ids))
    return train_idx, test_idx


def healpix_holdout(healpix: np.ndarray, *, fraction: float, seed: int) -> np.ndarray:
    """Hold out whole healpix pixels. Not the first rows of a file."""
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    healpix = np.asarray(healpix)
    uniq = np.unique(healpix)
    if uniq.size < 2:
        raise ValueError("healpix holdout needs at least 2 pixels")
    n_hold = max(1, int(round(uniq.size * fraction)))
    n_hold = min(n_hold, uniq.size - 1)
    chosen = np.random.default_rng(seed).choice(uniq, size=n_hold, replace=False)
    return np.isin(healpix, chosen)
