import sys
from pathlib import Path as _ScriptPath

sys.path.insert(0, str(_ScriptPath(__file__).resolve().parents[1] / "scripts"))
import hashlib
import json

import pytest
from run_rave_cnn_source import CNN_SHA, CROSS_SHA, RUN, run


def test_complete_and_no_overwrite(tmp_path):
    def fake(cnn, cross, output, **kwargs):
        assert kwargs["expected_cnn_sha256"] == CNN_SHA
        assert kwargs["expected_xmatch_sha256"] == CROSS_SHA
        output.write_bytes(b"example")
        return {
            "output": str(output),
            "output_bytes": 7,
            "output_sha256": hashlib.sha256(b"example").hexdigest(),
            "counts": {},
        }

    p = tmp_path / "run"
    m = run(p, "a" * 40, fake)
    assert m["run_id"] == RUN and m["status"] == "rave_source_complete"
    assert json.loads((p / "manifest.json").read_text()) == m
    with pytest.raises(FileExistsError):
        run(p, "a" * 40, fake)


def test_failure_is_durable(tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("bad source")

    p = tmp_path / "run"
    with pytest.raises(ValueError):
        run(p, "b" * 40, fail)
    m = json.loads((p / "manifest.json").read_text())
    assert m["status"] == "failed" and m["error_type"] == "ValueError"
