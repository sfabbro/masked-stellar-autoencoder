import subprocess

import pytest

from monitor.msa_monitor_io import (
    load_jsonl,
    sync_run_outputs,
)


def test_load_jsonl_ignores_only_an_incomplete_trailing_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"epoch": 1}\n{"epoch": 2')

    assert load_jsonl(path) == [{"epoch": 1}]

    path.write_text('{"epoch": 1}\nnot-json\n')
    with pytest.raises(ValueError, match=":2"):
        load_jsonl(path)


def test_sync_run_outputs_downloads_to_atomic_local_snapshots(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        destination = args[2]
        with open(destination, "w") as stream:
            stream.write('{"epoch": 1}\n')
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths, errors = sync_run_outputs(
        "/arc/projects/k-pop/msa_runs/run-1",
        tmp_path,
        filenames=("metrics.jsonl",),
    )

    assert not errors
    assert load_jsonl(paths["metrics.jsonl"]) == [{"epoch": 1}]
    assert calls[0][:2] == [
        "vcp",
        "arc:/projects/k-pop/msa_runs/run-1/pretrain/metrics.jsonl",
    ]
    assert calls[0][2].startswith(str(tmp_path))
    assert calls[0][2] != str(paths["metrics.jsonl"])


def test_sync_run_outputs_treats_not_yet_written_metrics_as_pending(
    tmp_path, monkeypatch
):
    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr="ERROR:: NodeNotFound: metrics.jsonl",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths, errors = sync_run_outputs(
        "/arc/projects/k-pop/msa_runs/run-1",
        tmp_path,
        filenames=("metrics.jsonl",),
    )

    assert not errors
    assert load_jsonl(paths["metrics.jsonl"]) == []


def test_sync_run_outputs_rejects_non_arc_or_escaping_paths(tmp_path):
    with pytest.raises(ValueError, match="/arc/projects"):
        sync_run_outputs("/arc/home/user/run", tmp_path)
    with pytest.raises(ValueError, match="/arc/projects"):
        sync_run_outputs("/arc/projects/k-pop/msa_runs/../other", tmp_path)
