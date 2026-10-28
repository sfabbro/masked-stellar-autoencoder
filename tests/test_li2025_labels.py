import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from li2025_labels import canonical_rows, records_from_table


def fixture():
    return Table(
        dict(
            Gaia=np.array([2**60 + 1, 2**60 + 1, 9, 8], dtype="i8"),
            obsid=[11, 12, 13, 14],
            RAdeg=[1.0, 1.0, 2.0, 3.0],
            DEdeg=[0.0, 0.0, 1.0, 2.0],
            snr=[30.0, 40.0, 20.0, 50.0],
            **{
                "[Fe/H]": [-2.8, -2.1, -3.0, -2.4],
                "e_[Fe/H]": [0.01, 0.2, 0.1, 0.1],
                "n_[Fe/H]": ["mag", "color", "mag", "mag"],
                "q_EW1": [0, 1, 0, 7],
            },
        )
    )


def test_both_index_flags_are_valid_and_duplicates_do_not_average():
    r = records_from_table(fixture())
    assert r["silver_eligible"].tolist() == [True, True, False, False]
    assert canonical_rows(r).tolist() == [1]
    assert r["source_id"][1] == 2**60 + 1
    assert not r["gold_nll_eligible"].any()
    assert (r["published_ca_fe_allowance_dex"] == 0.2).all()
    assert r["reported_uncertainty"][0] == 0.01
    assert r["split"][0] == r["split"][1]


def test_rejects_inexact_ids_and_no_eligible_rows():
    t = fixture()
    t["Gaia"] = np.asarray(t["Gaia"], dtype=float)
    with pytest.raises(ValueError, match="exact"):
        records_from_table(t)
    t = fixture()
    t["snr"] = 0.0
    assert canonical_rows(records_from_table(t)).size == 0
