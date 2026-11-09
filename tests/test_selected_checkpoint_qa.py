import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "selected_qa", Path(__file__).parents[1] / "scripts/run_expanded_checkpoint_qa.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_final_qa_requires_completed_run_and_explicit_selected_digest():
    digest = "a" * 64
    completed = {"status": "large_six_candidate_complete", "selected_checkpoint_sha256": digest}
    m.verify_final_snapshot(completed, digest, digest)
    with pytest.raises(ValueError, match="completed"):
        m.verify_final_snapshot({**completed, "status": "training"}, digest, digest)
    with pytest.raises(ValueError, match="explicitly bound"):
        m.verify_final_snapshot(completed, "b" * 64, digest)
    with pytest.raises(ValueError, match="explicitly bound"):
        m.verify_final_snapshot(completed, digest, None)
