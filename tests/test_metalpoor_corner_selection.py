"""Synthetic representative-selection invariants; no models or native arrays."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from select_metalpoor_corners import TARGETS, select_rows


def fixture():
    feh, magnitude, source = [], [], []
    for metal in (-3.3, -2.8, -2.3):
        for g in (8.0, 12.0):
            for survey in ("galah", "galah", "apogee", "apogee"):
                feh.append(metal)
                magnitude.append(g)
                source.append(survey)
    n = len(feh)
    role = np.array(["test"] * n + ["fit", "fit"])
    target = np.ones((n + 2, 6))
    target[:n, 2] = feh
    target[n:, 2] = -0.3
    label_source = np.full(target.shape, "native", dtype="U16")
    label_source[:n, 2] = source
    return dict(
        source_id=np.arange(100, 100 + n + 2),
        role=role,
        value=target,
        observed=np.ones_like(target, bool),
        quality=np.ones_like(target, bool),
        feature_names=np.array(["G", "bp_1"]),
        raw=np.column_stack([magnitude + [9.0, 11.0], np.zeros(n + 2)]),
        visible=np.ones((n + 2, 2), bool),
        label_source=label_source,
        label_source_role=np.tile(role[:, None], (1, 6)),
        target_names=np.asarray(TARGETS),
    )


def test_selection_is_model_blind_stable_source_diverse_and_fit_G_only():
    data = fixture()
    rows, report = select_rows(data)
    assert len(rows) == len(np.unique(rows)) == 12
    assert report["fit_only_G_median"] == 10.0
    assert report["fit_pair_observation_counts"] == np.full((6, 6), 2).tolist()
    for cell in report["cells"]:
        assert cell["candidate_count"] == 4
        assert set(cell["selected_sources"]) == {"galah", "apogee"}
    # Row reordering cannot change exact ID choices.
    permutation = np.arange(len(data["source_id"]))[::-1]
    shuffled = {
        key: values[permutation]
        if getattr(values, "shape", ()) and len(values) == len(permutation)
        else values
        for key, values in data.items()
    }
    again, _ = select_rows(shuffled)
    assert set(data["source_id"][rows]) == set(shuffled["source_id"][again])
    # TEST brightness/outcomes never set the split threshold.
    data["raw"][:24, 0] += 1.0
    _, changed = select_rows(data)
    assert changed["fit_only_G_median"] == 10.0


def test_empty_cells_and_missing_feh_quality_remain_explicit():
    data = fixture()
    data["quality"][:8, 2] = False
    rows, report = select_rows(data)
    assert len(rows) == 8
    assert [cell["empty"] for cell in report["cells"]] == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert report["cells"][0]["selected_source_ids"] == []
