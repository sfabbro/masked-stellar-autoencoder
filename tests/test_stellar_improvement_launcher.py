import datetime as dt
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "stellar_launch",
    Path(__file__).parents[1] / "scripts/launch_stellar_improvement.py",
)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def test_deferred_jobs_cannot_start_early_even_when_dependencies_complete():
    now = dt.datetime(2026, 10, 15, 21, tzinfo=dt.UTC)
    plan = {"not_before_utc": "2026-10-16T00:47:00+00:00", "depends_on": ["registry"]}
    assert not launch.eligible(plan, now, {"registry": True})
    assert launch.eligible(plan, now + dt.timedelta(hours=4), {"registry": True})


def test_registry_completion_is_not_enough_without_native_quality_measurements():
    now = dt.datetime.now(dt.UTC)
    plan = {"require_native_registry": True, "depends_on": ["label_registry_v1"]}
    assert not launch.eligible(plan, now, {"label_registry_v1": True})
    assert launch.eligible(
        plan,
        now,
        {"label_registry_v1": True, "label_registry_v1:native_labels_ready": True},
    )


def test_native_readiness_is_specific_to_the_requested_registry_version():
    now = dt.datetime.now(dt.UTC)
    plan = {"require_native_registry": True, "depends_on": ["label_registry_v2"]}
    assert not launch.eligible(
        plan,
        now,
        {
            "label_registry_v2": True,
            "label_registry_v1:native_labels_ready": True,
            "label_registry_v2:native_labels_ready": False,
        },
    )
    assert launch.eligible(
        plan,
        now,
        {
            "label_registry_v2": True,
            "label_registry_v1:native_labels_ready": False,
            "label_registry_v2:native_labels_ready": True,
        },
    )


def test_catalogue_hosts_are_explicitly_added_without_disabling_tls(monkeypatch):
    monkeypatch.setattr(launch, "dns_addresses", lambda: {"github.com": "192.0.2.1"})
    monkeypatch.setattr(launch.socket, "gethostbyname", lambda host: "192.0.2.2")
    addresses = launch.launch_addresses()
    assert addresses["data.desi.lbl.gov"] == "192.0.2.2"
    assert addresses["zenodo.org"] == "192.0.2.2"
    assert addresses["github.com"] == "192.0.2.1"
    assert "unconfigured.example" not in addresses


def test_server_rejection_body_is_preserved_bounded_without_retry():
    from types import SimpleNamespace

    error = RuntimeError("request rejected")
    error.response = SimpleNamespace(
        status_code=400, content=b"unsupported field" + b"x" * 9000
    )
    evidence = launch.submission_error_evidence(error)
    assert evidence["http_status"] == 400
    assert evidence["response_body"].startswith("unsupported field")
    assert len(evidence["response_body"]) == 8192
    assert evidence["response_body_truncated"] is True
    assert evidence["error_type"] == "RuntimeError"
    assert launch.submission_error_evidence(TimeoutError("transport")) == {
        "error": "transport",
        "error_type": "TimeoutError",
    }


def test_explicit_public_terminal_image_preserves_old_default():
    import pytest

    assert launch.launch_image({}) == "images.canfar.net/astroai/base:latest"
    assert (
        launch.launch_image(dict(image="images.canfar.net/astroai/terminal:latest"))
        == "images.canfar.net/astroai/terminal:latest"
    )
    with pytest.raises(ValueError):
        launch.launch_image(dict(image="skaha/unreviewed:latest"))
