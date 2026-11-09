import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits

SPEC = importlib.util.spec_from_file_location(
    "survey_inventory", Path(__file__).parents[1] / "scripts/audit_survey_inventory.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_header_only_schema_preserves_slashes_and_bounded_inventory(tmp_path):
    path = tmp_path / "lamost-native.fits"
    column = fits.Column(
        name="Fe/H", format="D", unit="dex(cm/s2)", array=np.array([-0.1, -0.2])
    )
    fits.BinTableHDU.from_columns([column]).writeto(path)
    schema = audit.fits_schema(path)
    assert schema["rows"] == 2
    assert schema["columns"] == ["Fe/H"]
    assert schema["units"]["Fe/H"] == "dex(cm/s2)"
    assert schema["content_sha256_verified"] is False
    assert audit.card_value("'a''b/c' / comment") == "a'b/c"
    (tmp_path / "other.txt").write_text("not a survey")
    found = audit.inventory(tmp_path)
    assert found["complete"] and len(found["sources"]) == 1
    partial = audit.inventory(tmp_path, max_files=1)
    assert not partial["complete"]
