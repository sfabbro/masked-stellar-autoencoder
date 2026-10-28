"""Synthetic checks for FIT-only post-collation availability counts."""

import hashlib

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.multisurvey_collation_census import (
    census_fit_collation,
)
from masked_stellar_autoencoder.training.multisurvey_data import (
    SparseSurveyDataset,
    collate_multisurvey,
)


def registry():
    targets = {
        target: {"field": target, "quantity": target, "unit": unit}
        for target, unit in (("teff", "K"), ("logg", "dex(cm/s2)"), ("fe_h", "dex"))
    }
    return {
        name: {
            "release": "r1",
            "pipeline": "p1",
            "dependency_id": name,
            "targets": targets,
        }
        for name in ("a", "b")
    }


def create(path):
    with h5py.File(path, "w") as h:
        h["source_id"] = np.array([11, 22, 33, 44, 55, 66, 77], dtype="i8")
        h["input"] = np.zeros((7, 2))
        h["role"] = np.array(
            [b"fit", b"fit", b"fit", b"tune", b"excluded", b"fit", b"fit"]
        )
        h["primary_input_eligible"] = np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool)
        h["input_has_xp"] = np.array([1, 0, 1, 1, 1, 1, 0], dtype=bool)
        h["real_xp_present"] = np.array([1, 0, 1, 1, 1, 1, 0], dtype=bool)
        h["real_no_xp"] = np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool)
        native = h.create_group("native_observations")
        for name, ids, rows, observed in (
            (
                "a",
                [11, 11, 22, 33, 33, 44, 55, 66, 77],
                [10, 11, 20, 30, 31, 40, 50, 60, 70],
                [
                    [1, 0, 0],
                    [0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ],
            ),
            ("b", [22, 33, 77], [80, 81, 82], [[1, 1, 1], [0, 1, 0], [0, 0, 0]]),
        ):
            g = native.create_group(name)
            g.attrs["source_fields"] = ["teff", "logg", "fe_h"]
            g["source_id"] = np.asarray(ids, dtype="i8")
            g["source_row"] = np.asarray(rows, dtype="i8")
            g["value"] = np.full((len(ids), 3), 2.0)
            errors = np.full((len(ids), 3), 0.1)
            if name == "a":
                errors[0, 0] = np.nan
            g["error"] = errors
            g["observed"] = np.asarray(observed, dtype=bool)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_census_matches_actual_collator_and_preserves_lost_target_and_modes(tmp_path):
    path = tmp_path / "source.h5"
    create(path)
    ds = SparseSurveyDataset(
        path, registry(), role="fit", expected_source_sha256=digest(path)
    )
    result = census_fit_collation(ds)
    summary, arrays = result["summary"], result["arrays"]
    assert arrays["source_id"].tolist() == [11, 22, 33, 77]
    assert summary["eligible_fit_stars"] == 4
    assert summary["per_source"]["a"]["active_repeated_stars"] == 1
    assert summary["per_source"]["a"]["targets_lost_by_selection"] == {
        "teff": 0,
        "logg": 1,
        "fe_h": 1,
    }
    assert summary["targets_lost_all_sources"] == {"teff": 0, "logg": 1, "fe_h": 1}
    assert summary["selected_targets_any_source"] == {"teff": 4, "logg": 2, "fe_h": 1}
    assert summary["source_combinations"] == {"a": 2, "a+b": 2}
    assert summary["explicit_xp_status_counts"] == {
        "explicit_xp_present": 2,
        "explicit_no_xp": 1,
        "other_or_unknown": 1,
    }
    assert summary["input_xp_modes"]["input_xp_not_staged_or_missing"]["stars"] == 2
    assert summary["per_source"]["a"]["selected_unknown_error_targets"]["teff"] == 1
    assert arrays["selected_source_row"][2, 0] == 31  # inactive row 30 is not selected
    for i in range(len(ds)):
        batch = collate_multisurvey([ds[i]], repeat_policy="lowest_source_row")
        for j in range(len(ds.names)):
            source_mask = batch["source_index"][0].numpy() == j
            actual = batch["observed"][0].numpy()[source_mask].any(axis=0)
            np.testing.assert_array_equal(actual, arrays["selected_observed"][i, j])
    ds.close()


def test_unverified_or_non_fit_dataset_cannot_be_counted(tmp_path):
    path = tmp_path / "source.h5"
    create(path)
    raw = SparseSurveyDataset(path, registry(), role="fit")
    with pytest.raises(ValueError, match="content-verified"):
        census_fit_collation(raw)
    raw.close()
    tune = SparseSurveyDataset(
        path, registry(), role="tune", expected_source_sha256=digest(path)
    )
    with pytest.raises(ValueError, match="exactly primary-eligible FIT"):
        census_fit_collation(tune)
    tune.close()
    no_filter = SparseSurveyDataset(
        path,
        registry(),
        role="fit",
        require_primary_input=False,
        expected_source_sha256=digest(path),
    )
    with pytest.raises(ValueError, match="exactly primary-eligible FIT"):
        census_fit_collation(no_filter)
    no_filter.close()


def test_dependency_conflicts_and_duplicate_active_source_rows_match_collator(tmp_path):
    path = tmp_path / "source.h5"
    create(path)
    shared = registry()
    shared["b"]["dependency_id"] = "a"
    ds = SparseSurveyDataset(
        path, shared, role="fit", expected_source_sha256=digest(path)
    )
    with pytest.raises(ValueError, match="dependent observations across sources"):
        census_fit_collation(ds)
    with pytest.raises(ValueError, match="dependent or repeated"):
        collate_multisurvey([ds[1]], repeat_policy="lowest_source_row")
    ds.close()
    with h5py.File(path, "r+") as h:
        h["native_observations/a/source_row"][1] = 10
    ds = SparseSurveyDataset(
        path, registry(), role="fit", expected_source_sha256=digest(path)
    )
    with pytest.raises(ValueError, match="duplicate source rows"):
        census_fit_collation(ds)
    with pytest.raises(ValueError, match="duplicate source rows"):
        collate_multisurvey([ds[0]], repeat_policy="lowest_source_row")
    ds.close()


def test_verified_external_link_overlay_and_later_mutation(tmp_path):
    import importlib.util
    import os
    from pathlib import Path

    from masked_stellar_autoencoder.training.multisurvey_overlay import (
        file_sha256,
        sparse_dataset_from_manifest,
    )

    spec = importlib.util.spec_from_file_location(
        "synthetic_overlay", Path(__file__).with_name("test_multisurvey_overlay.py")
    )
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _, side, manifest = fixture.fixture(tmp_path)
    ds = sparse_dataset_from_manifest(
        manifest, expected_manifest_sha256=file_sha256(manifest), role="fit"
    )
    result = census_fit_collation(ds)
    assert result["arrays"]["source_id"].tolist() == [11, 33]
    assert result["summary"]["selected_targets_any_source"] == {
        "teff": 0,
        "logg": 0,
        "fe_h": 1,
    }
    assert result["summary"]["source_combinations"] == {"none": 1, "extra": 1}
    assert result["summary"]["input_xp_modes"] is None
    assert result["summary"]["explicit_xp_status_counts"] is None
    stat = os.stat(side["path"])
    os.utime(side["path"], ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="attachment changed"):
        census_fit_collation(ds)
    ds.close()
