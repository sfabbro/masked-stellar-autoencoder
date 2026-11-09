import importlib.util
from pathlib import Path

import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("xmatcher")
SPEC = importlib.util.spec_from_file_location(
    "mastar_identity", Path(__file__).parents[1] / "scripts/audit_mastar_gaia_join.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_native_join_preserves_exact_gaia_and_refuses_ambiguous_ids():
    left = pl.DataFrame({"mangaid": ["a", "b"], "parameter_row": [0, 1]})
    right = pl.DataFrame(
        {"mangaid": ["b", "a"], "gaia_edr3_source_id": [2**60 + 1, 2**60 + 3]}
    )
    out = audit.join_native_ids(left, right).sort("parameter_row")
    assert out["gaia_edr3_source_id"].to_list() == [2**60 + 3, 2**60 + 1]
    with pytest.raises(ValueError, match="unique"):
        audit.join_native_ids(left, pl.concat([right, right]))
    with pytest.raises(ValueError, match="absent"):
        audit.join_native_ids(left, right.head(1))
