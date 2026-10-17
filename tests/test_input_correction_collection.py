"""SIMEX artifact discovery must retain every model and reject unsafe paths."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "experiments/xp/monitor/fetch_input_corrections.py"
SPEC = importlib.util.spec_from_file_location("correction_collector", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def inventory():
    records = []
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for simulation in range(1 if lam == 0 else 4):
            records.append(
                dict(
                    directory=f"replica-{len(records):03d}-lambda-{lam:g}-sim-{simulation}",
                    files={"checkpoint.pth": dict(bytes=410000000, sha256="a" * 64)},
                    manifest_sha256="b" * 64,
                )
            )
    return dict(status="complete", replicas=records)


def test_all17_actual_checkpoint_paths_are_preserved():
    paths = collector.inventory_files(inventory())
    assert len(paths) == 34
    assert len({p for p, _ in paths}) == 34
    assert sum(p.endswith("/checkpoint.pth") for p, _ in paths) == 17


@pytest.mark.parametrize(
    "change", ["traversal", "nested", "duplicate", "missing", "checksum"]
)
def test_remote_inventory_is_not_a_path_authority(change):
    data = copy.deepcopy(inventory())
    record = data["replicas"][0]
    if change == "traversal":
        record["directory"] = "../other"
    elif change == "nested":
        record["files"]["../checkpoint.pth"] = record["files"]["checkpoint.pth"]
    elif change == "duplicate":
        data["replicas"][1]["directory"] = record["directory"]
    elif change == "missing":
        record["files"] = {}
    else:
        record["files"]["checkpoint.pth"]["sha256"] = "incorrect"
    with pytest.raises(ValueError):
        collector.inventory_files(data)


def test_snapshot_names_follow_actual_batches_and_final_tail():
    metrics = [
        {"presentations": n} for n in [2999999, 3000100, 6000100, 9000100, 10000000]
    ]
    assert collector.snapshot_files(metrics, 10000000) == [
        "snapshot-0.npz",
        "snapshot-3000100.npz",
        "snapshot-6000100.npz",
        "snapshot-9000100.npz",
        "snapshot-10000000.npz",
    ]
    with pytest.raises(ValueError):
        collector.snapshot_files(metrics[:-1], 10000000)
