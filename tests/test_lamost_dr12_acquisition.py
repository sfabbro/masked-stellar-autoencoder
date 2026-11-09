import gzip
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from astropy.table import Table

SPEC = importlib.util.spec_from_file_location(
    "lamost_dr12_acquisition",
    Path(__file__).parents[1] / "scripts/acquire_lamost_dr12.py",
)
acquire = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acquire)


@pytest.mark.parametrize("truncate", [False, True])
def test_native_download_requires_complete_gzip_and_schema(
    tmp_path, monkeypatch, truncate
):
    table = Table({name: [1] for name in acquire.REQUIRED})
    fits_buffer = io.BytesIO()
    table.write(fits_buffer, format="fits")
    payload = gzip.compress(fits_buffer.getvalue())
    if truncate:
        payload = payload[:-8]

    class Response(io.BytesIO):
        status = 200
        url = acquire.URL
        headers = {"Content-Type": "image/fits", "Content-Length": str(len(payload))}

    monkeypatch.setattr(
        acquire.urllib.request, "urlopen", lambda *a, **k: Response(payload)
    )
    monkeypatch.setattr(
        acquire.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12)
    )
    monkeypatch.setattr(acquire, "EXPECTED_ROWS", 1)
    out = tmp_path / "raw"
    monkeypatch.setattr(sys, "argv", ["acquire", "--output", str(out)])
    if truncate:
        with pytest.raises(EOFError):
            acquire.main()
        assert (
            json.loads((out / "acquisition.json").read_text())["status"]
            != "verified_native_source"
        )
    else:
        acquire.main()
        result = json.loads((out / "acquisition.json").read_text())
        assert result["status"] == "verified_native_source"
        assert result["gzip_crc_verified"] and result["rows"] == 1
        assert not result["publisher_checksum_verified"]
