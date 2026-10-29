import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hashlib
import json

import pytest
from run_gaia_fem_source_join import ARCHIVE_SHA, RUN, SIDECAR_SHA, run


def test_runner_accepts_and_preserves_manifest(tmp_path):
    def fake(side, archive, output, **kwargs):
        assert (
            kwargs["expected_sidecar_sha"] == SIDECAR_SHA
            and kwargs["expected_archive_sha"] == ARCHIVE_SHA
        )
        output.write_bytes(b"example")
        return {
            "output": str(output),
            "output_bytes": 7,
            "output_sha256": hashlib.sha256(b"example").hexdigest(),
            "counts": {},
        }

    p = tmp_path / "out"
    m = run(p, "a" * 40, fake)
    assert m["run_id"] == RUN and m["status"] == "gaia_fem_source_join_complete"
    assert json.loads((p / "manifest.json").read_text()) == m
    with pytest.raises(FileExistsError):
        run(p, "a" * 40, fake)


def test_runner_preserves_failure(tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("bad source")

    p = tmp_path / "out"
    with pytest.raises(ValueError):
        run(p, "b" * 40, fail)
    assert json.loads((p / "manifest.json").read_text())["status"] == "failed"
