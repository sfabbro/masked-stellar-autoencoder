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
        plan, now, {"label_registry_v1": True, "native_labels_ready": True}
    )
