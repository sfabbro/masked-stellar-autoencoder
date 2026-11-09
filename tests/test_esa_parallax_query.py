import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "esa_recovery",
    Path(__file__).resolve().parents[1] / "scripts/recover_esa_parallax_gaps.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_esa_join_keeps_exact_id_negative_parallax_and_rejects_other_star(monkeypatch):
    sid = 6000000000000000001
    monkeypatch.setattr(
        recovery.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(
            f"source_id,parallax,parallax_error\n{sid},-0.1,0.02\n".encode()
        ),
    )
    body, frame = recovery.query(np.array([sid], dtype="i8"))
    assert int(frame.source_id.iloc[0]) == sid
    assert frame.parallax.iloc[0] == -0.1
    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="association"):
        recovery.query(np.array([sid + 2], dtype="i8"))
