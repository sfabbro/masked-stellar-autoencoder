import importlib.util
import sys
from pathlib import Path

import numpy as np

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "seismic_qa", scripts / "run_seismic_gravity_qa.py"
)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
sys.path.pop(0)


def test_quoted_reference_noise_is_not_intrinsic_model_coverage():
    w = np.ones((3, 1))
    mu = np.array([[2.0], [3.0], [4.0]])
    y = mu[:, 0] + 0.1
    result = qa.summarize(w, mu, np.full((3, 1), 0.01**2), y, np.full(3, 0.1))
    assert result["intrinsic_vs_reference"]["coverage90"] == 0
    assert result["convolved_with_quoted_seismic_error"]["coverage90"] == 1
    np.testing.assert_allclose(result["point"]["mae"], 0.1)
    assert qa.summarize(w[:0], mu[:0], mu[:0], y[:0], y[:0]) == {"n": 0}
