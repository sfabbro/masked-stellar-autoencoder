import subprocess

import pytest

from monitor.msa_monitor_io import (
    experiment_comparison,
    load_jsonl,
    load_snapshot,
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


def test_suite_outputs_allow_nested_files_but_reject_traversal(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        with open(args[2], "w") as stream:
            stream.write('{"finite": true}')
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    paths, errors = sync_run_outputs(
        "/arc/projects/k-pop/pilot",
        tmp_path,
        filenames=("arm-a/residual_latest.json",),
        subdir="",
    )
    assert not errors
    assert calls[0][1] == "arc:/projects/k-pop/pilot/arm-a/residual_latest.json"
    assert load_snapshot(paths["arm-a/residual_latest.json"]) == {"finite": True}
    for filename in ("../private.json", "/absolute.json", "arm/../../private.json"):
        with pytest.raises(ValueError, match="relative"):
            sync_run_outputs(
                "/arc/projects/k-pop/pilot",
                tmp_path,
                filenames=(filename,),
                subdir="",
            )
    with pytest.raises(ValueError, match="relative"):
        sync_run_outputs("/arc/projects/k-pop/pilot", tmp_path, subdir="../")


def test_experiment_ranking_requires_common_finite_validation():
    manifest = {
        "validation_id": "fixed-v1",
        "arms": [
            {"arm_id": name, "status": "complete"}
            for name in ("better", "worse", "missing", "nan", "mismatch", "flagged")
        ],
    }
    records = {
        name: [
            {
                "validation_id": "fixed-v1",
                "sampled_validation_mae": score,
                "median_mae": 2.0,
                "optimizer_steps": 100,
                "rows_seen_total": 200,
                "group_skill_median": 1 - score,
            }
        ]
        for name, score in (
            ("better", 0.5),
            ("worse", 1.0),
            ("nan", float("nan")),
            ("mismatch", 0.1),
            ("flagged", 0.1),
        )
    }
    records["mismatch"][0]["validation_id"] = "other-v1"
    snapshots = {"flagged": {"finite": False}}
    rows = {
        row["arm"]: row for row in experiment_comparison(manifest, records, snapshots)
    }
    assert rows["better"]["rank"] == 1
    assert rows["worse"]["rank"] == 2
    assert rows["better"]["skill vs median"] == 0.75
    for name in ("missing", "nan", "mismatch", "flagged"):
        assert rows[name]["rank"] is None
    assert rows["missing"]["finite"] is False
    assert rows["nan"]["finite"] is False
    assert rows["mismatch"]["common validation"] is False
    assert experiment_comparison({}, {}, {}) == []


def test_failed_fetch_preserves_previous_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "manifest.json"
    snapshot.write_text('{"status": "running"}')
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="network unavailable"
        ),
    )
    paths, errors = sync_run_outputs(
        "/arc/projects/k-pop/pilot",
        tmp_path,
        filenames=("manifest.json",),
        subdir="",
    )
    assert load_snapshot(paths["manifest.json"])["status"] == "running"
    assert errors["manifest.json"] == "network unavailable"


def test_ranking_uses_group_skill_and_excludes_unfinished_or_unequal_budgets():
    manifest = {
        "validation_id": "v1",
        "presentations_per_arm": 200,
        "arms": [
            {"arm_id": name, "status": "running" if name == "running" else "complete"}
            for name in ("group_winner", "raw_winner", "running", "partial")
        ],
    }
    records = {
        name: [
            {
                "validation_id": "v1",
                "finite": True,
                "sampled_validation_mae": mae,
                "group_skill_median": skill,
                "rows_seen_total": views,
                "optimizer_steps": 100,
            }
        ]
        for name, mae, skill, views in (
            ("group_winner", 0.8, 0.7, 200),
            ("raw_winner", 0.2, 0.4, 200),
            ("running", 0.1, 0.9, 200),
            ("partial", 0.1, 0.9, 100),
        )
    }
    rows = {row["arm"]: row for row in experiment_comparison(manifest, records, {})}
    assert rows["group_winner"]["rank"] == 1
    assert rows["raw_winner"]["rank"] == 2
    assert rows["running"]["rank"] is None
    assert rows["partial"]["rank"] is None
    records["raw_winner"][0]["optimizer_steps"] = 101
    rows = experiment_comparison(manifest, records, {})
    assert all(row["rank"] is None for row in rows)
    assert rows[0]["ranking eligibility"] == "different update budgets"


def test_comparison_shows_initial_improvement_and_checks_both_validation_ids():
    manifest = {"validation_id": "v1", "arms": [{"arm_id": "a", "status": "complete"}]}
    records = {
        "a": [
            {
                "event": "init",
                "validation_id": "v1",
                "sampled_validation_mae": 1.0,
                "group_skill_median": 0.2,
            },
            {
                "event": "complete",
                "validation_id": "v1",
                "sampled_validation_mae": 0.5,
                "group_skill_median": 0.7,
                "optimizer_steps": 10,
                "rows_seen_total": 100,
            },
        ]
    }
    row = experiment_comparison(manifest, records, {"a": {"validation_id": "v1"}})[0]
    assert row["rank"] == 1
    assert row["MAE improvement"] == 0.5
    assert row["group skill improvement"] == pytest.approx(0.5)
    records["a"][-1]["validation_id"] = "other"
    row = experiment_comparison(manifest, records, {"a": {"validation_id": "v1"}})[0]
    assert row["rank"] is None
    assert row["common validation"] is False
