"""Small unit checks for the A0 age-data audit helpers (no catalogue downloads)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _audit():
    path = Path(__file__).parents[1] / "scripts" / "age_data_audit.py"
    spec = importlib.util.spec_from_file_location("age_data_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_median_nmad_known_values_and_empty():
    audit = _audit()
    values = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    result = audit.median_nmad(values)
    assert result["n"] == 4
    assert result["median"] == 2.5
    assert 0.0 < result["nmad"] < 2.0
    empty = audit.median_nmad(np.array([np.nan]))
    assert empty["n"] == 0 and np.isnan(empty["median"])


def test_grouped_offsets_skips_unclassified_groups():
    audit = _audit()
    frame = pd.DataFrame(
        {
            "age_apokasc": [5.0, 4.0, 7.0, 6.0],
            "age_Dnu": [4.5, 4.5, 6.0, 5.0],
            "group": ["a", "a", "b", None],
        }
    )
    result = audit.grouped_offsets(frame, "age_Dnu", "group")
    assert [row["group"] for row in result] == ["a", "b"]
    assert all(np.isfinite(row["median"]) for row in result)


def test_constant_identifier_column_detected(tmp_path):
    path = tmp_path / "cat_TESS_APG.csv"
    path.write_text("TESS_ID,GAIADR3_ID,age_Dnu\n1,42,3.0\n2,42,4.0\n")
    frame = pd.read_csv(path, dtype=str)
    values = frame["GAIADR3_ID"].dropna().astype(str)
    assert values.nunique() == 1 and values.iloc[0] == "42"
