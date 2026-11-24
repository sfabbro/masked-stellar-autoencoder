"""data/msa_split.py: the one msa-split-v1 hash of the package."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from masked_stellar_autoencoder.data import msa_split as M

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    # reuse the module test_msa_split_v1 registered: replacing it in sys.modules
    # would break the pickling of its pool workers
    if "build_msa_split_v1" in sys.modules:
        return sys.modules["build_msa_split_v1"]
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "build_msa_split_v1", ROOT / "scripts" / "build_msa_split_v1.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_spec_test_vectors_and_salt():
    M.check_test_vectors()
    assert hex(M.SALT64) == "0xb6f6e5353645106b"


def test_agrees_with_frozen_builder_and_v11():
    from masked_stellar_autoencoder.v11 import data as v11

    B = _builder()
    rng = np.random.default_rng(0)
    keys = (rng.integers(0, 12 * 4**12, 20000) << 35) + rng.integers(0, 2**30, 20000)
    np.testing.assert_array_equal(M.unit(keys), B.unit(keys))
    np.testing.assert_array_equal(M.hash_partition(keys), B.partition_of(B.unit(keys)))
    np.testing.assert_array_equal(
        M.unit(keys, M.REC_SALT64), B.unit(keys, B.salt64(B.REC_SALT))
    )
    assert v11.split_unit is M.unit and v11.split_partition is M.partition_of
    frac = np.bincount(M.hash_partition(keys), minlength=4) / len(keys)
    np.testing.assert_allclose(frac, [0.75, 0.05, 0.08, 0.12], atol=0.01)


def test_float_ids_are_refused():
    with pytest.raises(TypeError):
        M.unit(np.array([1.0e18]))


def test_overrides_take_precedence(tmp_path):
    keys = np.array([4185355646116022528, 4398046511104, 4295806720, 4295806753])
    assert M.hash_partition(keys).tolist() == [0, 1, 2, 3]
    pd.DataFrame(
        dict(
            source_id=keys[:2], partition=np.array([3, 4], np.uint8),
            group_id=np.array([7, -1], np.int32), input_class=np.array([0, 1], np.uint8),
            reserved_reason=["", "gbs_v3"],
        )
    ).to_parquet(tmp_path / "overrides.parquet")  # fmt: skip
    split = M.MsaSplit.from_freeze(tmp_path)
    assert split.partition(keys).tolist() == [3, 4, 2, 3]
    assert split.group_id(keys).tolist() == [7, -1, -1, -1]
    assert split.reserved_reason(keys).tolist() == ["", "gbs_v3", "", ""]
