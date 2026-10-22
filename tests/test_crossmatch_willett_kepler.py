"""Exact identifier namespace and ambiguity contracts."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "crossmatch", Path(__file__).parents[1] / "scripts/crossmatch_willett_kepler.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact64bit_identity_and_kepler_namespace_only(tmp_path):
    gaia = 2050237616959273729
    anchors = pd.DataFrame({"kic": [42], "source_id": pd.array([gaia], dtype="Int64")})
    willett = pd.DataFrame(
        {
            "survey": ["kepler", "tess", "k2_cor", "kepler"],
            "kic": ["42", "42", "42", "999"],
            "gaia_id_raw": ["123"] * 4,
            "age50": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = module.crossmatch(anchors, willett)
    assert int(result.source_id.iloc[0]) == gaia
    assert result.source_id.iloc[1:].isna().all()
    assert result.age50.equals(willett.age50)
    path = tmp_path / "identities.parquet"
    result.to_parquet(path)
    assert pd.read_parquet(path).source_id.equals(result.source_id)


@pytest.mark.parametrize(
    "anchors",
    [
        pd.DataFrame({"kic": [42, 42], "source_id": [1, 2]}),
        pd.DataFrame({"kic": [42, 43], "source_id": [1, 1]}),
    ],
)
def test_ambiguous_reference_map_fails(anchors):
    with pytest.raises(ValueError):
        module.crossmatch(anchors, pd.DataFrame({"survey": ["kepler"], "kic": ["42"]}))


def test_duplicate_kepler_rows_fail_without_aggregation():
    with pytest.raises(ValueError, match="duplicate Willett"):
        module.crossmatch(
            pd.DataFrame({"kic": [42], "source_id": [1]}),
            pd.DataFrame({"survey": ["kepler", "kepler"], "kic": ["42", "42"]}),
        )


def test_float_identity_rejected_not_silently_rounded():
    assert module.exact_positive_integer(float(2050237616959273729)) is None
    assert module.exact_positive_integer("2050237616959273729") == 2050237616959273729
