import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.table import MaskedColumn, Table

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "pigs", Path(__file__).parents[1] / "scripts/pigs_stellar_labels.py"
)
pigs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pigs)


def test_native_quality_flags_and_total_errors_are_not_reinflated():
    table = Table(
        {
            "GaiaDR3": [11, 22, 33],
            "double": ["false", "true", "false"],
            "goodF": ["true", "true", "false"],
            "goodU": ["false", "true", "false"],
            "[Fe/H]F": [-2.4, -3.0, -1.0],
            "e_[Fe/H]Ftot": [0.2] * 3,
            "[Fe/H]U": [-2.3, -2.9, -1.0],
            "e_[Fe/H]Utot": [0.21] * 3,
        }
    )
    records = pigs.spectroscopy_measurements(table)
    selected = records[records["quality_pass"]]
    assert selected["source_id"].tolist() == [11]
    np.testing.assert_array_equal(selected["uncertainty"], [0.2])
    assert selected["pipeline"].tolist() == [b"FERRE"]


def test_distance_join_rejects_ambiguous_ids_and_checks_repeated_feh():
    table = Table(
        {
            "Star": ["a", "a", "b", "b"],
            "GaiaDR3": [11, 11, 22, 33],
            "[Fe/H]F": [-2.4] * 4,
        }
    )
    distances = Table(
        {
            "Star": ["a", "b", "a", "a"],
            "[Fe/H]": [-2.3991097, -2.4, -1.0, -2.4],
            "dist16": [1.0, 1.0, 1.0, 3.0],
            "dist50": [2.0] * 4,
            "dist84": [3.0] * 4,
        }
    )
    ids, rows, percentiles, valid = pigs.join_distances(table, distances)
    assert ids.tolist() == [11, 0, 0, 11]
    assert rows.tolist() == [0, -1, -1, 0]
    assert valid.tolist() == [True, False, False, False]
    assert percentiles[0].tolist() == [1.0, 2.0, 3.0]


def test_raw_cds_fields_and_missingness_are_preserved():
    table = Table({"Star": ["a", "b"]})
    table["[Fe/H]F"] = MaskedColumn([-2.4, 0.0], mask=[False, True])
    values, missing = pigs.raw_table(table)
    assert values.dtype["Star"].kind == "S"
    assert values["Star"].tolist() == [b"a", b"b"]
    assert np.isnan(values["[Fe/H]F"][1])
    assert missing.tolist() == [[False, False], [False, True]]
