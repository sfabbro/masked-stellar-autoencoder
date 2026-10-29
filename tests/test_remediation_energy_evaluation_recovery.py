import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path("scripts")))
spec = importlib.util.spec_from_file_location(
    "energy_recovery", "scripts/run_remediation_energy_evaluation_recovery.py"
)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Head:
    def __init__(self):
        self.calls = []

    def sample(self, z, n, generator):
        self.calls.append(n)
        return torch.randn((n, len(z), 6), generator=generator)


def test_actual_whole_vector_chunks_and_seed():
    h = Head()
    out = r.whole_draws(h, torch.zeros((3, 2)), 19, 7, torch.Generator().manual_seed(7))
    assert out.shape == (19, 3, 6) and h.calls == [7, 7, 5] and np.isfinite(out).all()
    h2 = Head()
    np.testing.assert_array_equal(
        out,
        r.whole_draws(h2, torch.zeros((3, 2)), 19, 7, torch.Generator().manual_seed(7)),
    )
    for n, c in [(1, 2), (3, 0)]:
        with pytest.raises(ValueError):
            r.whole_draws(h, torch.zeros((3, 2)), n, c, torch.Generator())


def test_requires_exact_terminal_failure_and_checkpoint(tmp_path):
    p = tmp_path / "manifest.json"
    checkpoint = tmp_path / "selected"
    checkpoint.write_bytes(b"weights")
    digest = r.prior.sha(checkpoint)
    m = {
        "run_id": "old",
        "source_commit": "f70aedca512925481f29a5022bf0184084282374",
        "status": "failed",
        "error_type": "OutOfMemoryError",
        "completed_updates": 1000,
        "selected_update": 900,
    }
    p.write_text(json.dumps(m))
    assert r.admission(p, "old", checkpoint, digest, 900) == m
    for change in (
        {"status": "running"},
        {"error_type": "ValueError"},
        {"completed_updates": 999},
        {"selected_update": 1000},
        {"source_commit": "a" * 40},
    ):
        p.write_text(json.dumps({**m, **change}))
        with pytest.raises(ValueError):
            r.admission(p, "old", checkpoint, digest, 900)
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        r.admission(p, "old", checkpoint, "0" * 64, 900)
