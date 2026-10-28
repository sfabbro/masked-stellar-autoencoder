import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_hipparcos_inputs import complete_votable, unique_positive_integer


def test_query_guard_rejects_trailing_error_and_overflow(tmp_path):
    p = tmp_path / "response.xml"
    Table(dict(source_id=np.array([2**60 + 1], dtype="i8"))).write(p, format="votable")
    original = p.read_text().replace(
        '<RESOURCE type="results">',
        '<RESOURCE type="results"><INFO name="QUERY_STATUS" value="OK"/>',
    )
    # Astropy versions may use RESOURCE without a type attribute.
    if 'name="QUERY_STATUS"' not in original:
        original = original.replace(
            "<RESOURCE>", '<RESOURCE><INFO name="QUERY_STATUS" value="OK"/>'
        )
    p.write_text(original)
    assert int(complete_votable(p)["source_id"][0]) == 2**60 + 1
    for status in ["ERROR", "OVERFLOW"]:
        p.write_text(
            original.replace(
                "</RESOURCE>",
                f'<INFO name="QUERY_STATUS" value="{status}"/></RESOURCE>',
            )
        )
        with pytest.raises(ValueError, match="Incomplete Gaia"):
            complete_votable(p)
    p.write_text(original.replace('<INFO name="QUERY_STATUS" value="OK"/>', ""))
    with pytest.raises(ValueError, match="Incomplete Gaia"):
        complete_votable(p)


def test_ids_cannot_be_float_masked_zero_or_duplicated():
    exact = np.array([2**60 + 1, 2**60 + 3], dtype="i8")
    np.testing.assert_array_equal(unique_positive_integer(exact, "id"), exact)
    for values in [
        exact.astype(float),
        [1, 1],
        [0, 2],
        MaskedColumn([1, 2], mask=[False, True]),
    ]:
        with pytest.raises(ValueError, match="Invalid or duplicate"):
            unique_positive_integer(values, "id")
