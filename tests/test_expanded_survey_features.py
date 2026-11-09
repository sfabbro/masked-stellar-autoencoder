import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "expanded_feature_builder",
    Path(__file__).parents[1] / "scripts/build_expanded_survey_features.py",
)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.mark.parametrize("release", ["dr8v2", "dr12v2"])
def test_source_union_preserves_base_and_remains_unassigned(
    tmp_path, monkeypatch, release
):
    def sidecar(path, name, ids):
        with h5py.File(path, "w") as f:
            g = f.create_group("native_observations/" + name)
            g["source_id"] = np.array(ids, dtype="i8")
            g["observed"] = np.ones((len(ids), 3), dtype=bool)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    gaia = tmp_path / "gaia.h5"
    lamost = tmp_path / "lamost.h5"
    base = tmp_path / "base.h5"
    gs = sidecar(gaia, "gaia_dr3_gspspec", [1, 2])
    name = "lamost_" + release + "_lrs_lasp"
    ls = sidecar(lamost, name, [2, 3])
    with h5py.File(base, "w") as f:
        f["source_id"] = np.array([1], dtype="i8")
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text(
        json.dumps(
            {"request": {"features": ["G", "bp_1"], "center": [0, 0], "scale": [1, 1]}}
        )
    )
    output = tmp_path / "output"
    c = dict(
        output=str(output),
        training_enabled=False,
        assign_roles=False,
        scan_original=False,
        allow_unknown_photometry=False,
        gaia_sidecar=str(gaia),
        gaia_sha256=gs,
        lamost_sidecar=str(lamost),
        lamost_sha256=ls,
        lamost_source_name=name,
        base=str(base),
        cache=str(cache),
        legacy="fixture",
        legacy_sha256="fixture",
        cache_request_sha256="fixture",
        original="fixture",
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(c))

    def lookup(ids, names, center, scale, **kwargs):
        np.testing.assert_array_equal(ids, [2, 3])
        assert not kwargs["scan_original"] and not kwargs["allow_unknown_photometry"]
        a = dict(
            input=np.array([[1.0, 2.0], [3.0, 4.0]], dtype="f4"),
            visible=np.ones((2, 2), bool),
            feature_recoverable=np.ones(2, bool),
            input_has_xp=np.ones(2, bool),
            real_xp_present=np.ones(2, bool),
            real_no_xp=np.zeros(2, bool),
            cache_row=np.array([5, 6]),
            original_row=np.array([50, 60]),
            original_shard=np.array([0, 0]),
            input_xp_missing_publication_unknown=np.zeros(2, bool),
            xp_status=np.array(["complete_xp"] * 2),
        )
        return a, dict(counts={"complete_xp": 2})

    monkeypatch.setattr(builder, "lookup_evaluation_features", lookup)
    monkeypatch.setattr(
        builder.shutil, "disk_usage", lambda p: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(
        sys, "argv", ["build", "--config", str(config), "--output", str(output)]
    )
    builder.main()
    assert hashlib.sha256(base.read_bytes()).hexdigest() == base_sha
    with h5py.File(output / "new-star-features.h5") as f:
        np.testing.assert_array_equal(f["source_id"][:], [2, 3])
        assert "role" not in f
    result = json.loads((output / "manifest.json").read_text())
    assert result["recovered_stars"] == 2 and not result["training_performed"]
    c["training_enabled"] = True
    config.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="source-only"):
        builder.main()
