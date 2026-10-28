import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_age_anchor_labels import APOKASC3_COLUMNS, build_apokasc3


def test_signed_catalogue_errors_become_age_bounds_without_float_id_roundtrip(tmp_path):
    rows = []
    for age, lower, upper in [(5.0, -1.0, 2.0), (5.0, -9999.0, 2.0), (1.0, -2.0, 1.0)]:
        row = {name: "0" for name in APOKASC3_COLUMNS}
        row.update(
            AgeCat="RGB",
            AgeRGB=str(age),
            e_AgeRGB=str(lower),
            E_AgeRGB=str(upper),
            GaiaDR3="1234567890123456789",
        )
        rows.append(" ".join(row[k] for k in APOKASC3_COLUMNS))
    path = tmp_path / "table4.dat"
    path.write_text("\n".join(rows))
    result = build_apokasc3(path)
    assert result.source_id.iloc[0] == 1234567890123456789
    assert result.age16.iloc[0] == pytest.approx(4.0)
    assert result.age84.iloc[0] == pytest.approx(7.0)
    assert result.age_bounds_valid.tolist() == [True, False, False]
    assert np.isnan(result.age16.iloc[1:]).all()
