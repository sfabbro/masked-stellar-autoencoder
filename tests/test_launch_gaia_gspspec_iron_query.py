"""One-shot TAP query submission is recorded before the external request."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    p = Path(__file__).parents[1] / "scripts/launch_gaia_gspspec_iron_query.py"
    s = importlib.util.spec_from_file_location("gaia_iron_query", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class Response:
    def __init__(
        self, status=303, location="https://gaia.ari.uni-heidelberg.de/tap/async/1234"
    ):
        self.status_code = status
        self.headers = {"Location": location}
        self.text = "bounded error"


class Client:
    def __init__(self, record, response=None, error=None):
        self.record = record
        self.response = response or Response()
        self.error = error
        self.calls = 0

    def post(self, url, **kwargs):
        self.calls += 1
        assert json.loads(self.record.read_text())["status"] == "submitting"
        assert (
            kwargs["data"]["MAXREC"] == "10000000" and kwargs["data"]["PHASE"] == "RUN"
        )
        assert "fem_gspspec IS NOT NULL" in kwargs["data"]["QUERY"]
        if self.error:
            raise self.error
        return self.response


def test_accepted_submission_is_recorded_once(tmp_path):
    m = module()
    p = tmp_path / "record.json"
    client = Client(p)
    result = m.submit(p, source_sha="a" * 40, client=client)
    assert result["status"] == "submitted" and result["job_id"] == "1234"
    assert json.loads(p.read_text()) == result and client.calls == 1
    with pytest.raises(FileExistsError):
        m.submit(p, source_sha="a" * 40, client=client)
    assert client.calls == 1


def test_transport_ambiguity_is_recorded_without_retry(tmp_path):
    m = module()
    p = tmp_path / "record.json"
    client = Client(p, error=TimeoutError("timeout"))
    with pytest.raises(TimeoutError):
        m.submit(p, source_sha="a" * 40, client=client)
    assert (
        json.loads(p.read_text())["status"] == "submission_ambiguous"
        and client.calls == 1
    )


@pytest.mark.parametrize(
    "response,status",
    [
        (Response(400, ""), "rejected"),
        (Response(303, "https://example.com/tap/async/1234"), "submission_ambiguous"),
    ],
)
def test_bad_response_preserves_evidence(tmp_path, response, status):
    m = module()
    p = tmp_path / "record.json"
    client = Client(p, response=response)
    with pytest.raises(RuntimeError):
        m.submit(p, source_sha="a" * 40, client=client)
    assert json.loads(p.read_text())["status"] == status and client.calls == 1
