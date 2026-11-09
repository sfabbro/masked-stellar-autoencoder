import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

pytest.importorskip("polars")
pytest.importorskip("xmatcher")
spec = importlib.util.spec_from_file_location(
    "local_gaia_match",
    Path(__file__).parents[1] / "scripts/run_decision_local_gaia_eso.py",
)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def test_local_retrieval_accounts_for_motion_and_ra_wrap():
    ra = np.array([359.9999])
    dec = np.array([0.0])
    pm = np.array([1000.0])
    zero = np.array([0.0])
    moved = local.directions(ra, dec, pm, zero, 16.0)
    norm = np.linalg.norm(moved, axis=1)
    np.testing.assert_allclose(norm, 1.0)
    expected = local.directions((ra + 16.0 / 3600) % 360, dec)
    assert np.linalg.norm(moved - expected) < 1e-9
    tree = cKDTree(expected)
    distance, _ = tree.query(local.directions(ra, dec))
    assert distance[0] > np.deg2rad(10 / 3600)
    distance, _ = tree.query(moved)
    assert distance[0] < np.deg2rad(1 / 3600)
