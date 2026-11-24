"""Bounded checks for the P1 pilot helpers (scripts/gaia_master_pilot.py)."""

import importlib.util
import json
import os
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gaia_master_pilot.py"
spec = importlib.util.spec_from_file_location("gaia_master_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_snapshot_diff_groups_changes_by_tree(tmp_path):
    root = tmp_path / "hats"
    (root / "gaia" / "dr3" / "dataset").mkdir(parents=True)
    (root / "ps1" / "otmo").mkdir(parents=True)
    (root / "gaia" / "dr3" / "properties").write_text("hats_nrows=1\n")
    (root / "ps1" / "otmo" / "properties").write_text("x\n")
    pilot.snapshot(root, tmp_path / "a.tsv")
    same = pilot.diff(tmp_path / "a.tsv", tmp_path / "a.tsv", 2)
    assert same["unchanged"] and same["before_entries"] == 7

    later = time.time() + 10
    os.utime(root / "ps1" / "otmo" / "properties", (later, later))
    (root / "ps1" / "otmo" / "new.parquet").write_text("")
    pilot.snapshot(root, tmp_path / "b.tsv")
    changed = pilot.diff(tmp_path / "a.tsv", tmp_path / "b.tsv", 2)
    assert not changed["unchanged"]
    assert set(changed["trees"]) == {"ps1/otmo"}
    assert changed["trees"]["ps1/otmo"]["added"] == 1
    assert changed["trees"]["ps1/otmo"]["modified"] >= 1


def test_summary_reports_fractions_and_task_stats(tmp_path):
    hats = tmp_path / "pilot.hats"
    hats.mkdir()
    (hats / "properties").write_text(
        "hats_nrows=10\nxmatcher_anchor_partitions=2/69\n"
        "xmatcher_invalid_survey_rows=sdss:3 vhs:0\n"
    )
    rel = "Norder=2/Dir=0/Npix=69.parquet"
    stats = {
        "anchor_rows": 10,
        "seconds": 12.5,
        "peak_rss_gib": 3.2,
        "surveys": {"sdss": {"partitions": 2, "rows": 100, "matched": 4}},
    }
    (hats / "resume.state").write_text(json.dumps({"task_stats": {rel: stats}}))
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "rows": 10,
                "advertised_rows": 10,
                "surveys": {
                    "sdss": {
                        "rows": 10,
                        "matched": 4,
                        "crowded": 1,
                        "pm_warn": 2,
                        "quality": {
                            "good": 2,
                            "ambiguous": 1,
                            "pm_warn": 1,
                            "no_match": 6,
                        },
                    }
                },
            }
        )
    )
    out = tmp_path / "summary.json"

    class Args:
        pass

    args = Args()
    args.validation, args.hats, args.out = validation, hats, out
    result = pilot.summary(args)
    row = result["surveys"][0]
    assert row["match_fraction"] == 0.4
    assert row["ambiguous_fraction"] == 0.1
    assert row["invalid_rows_dropped"] == 3
    assert row["tasks"][rel]["rows"] == 100
    assert result["tasks"][rel]["peak_rss_gib"] == 3.2
    assert "| sdss | 4 |" in out.with_suffix(".md").read_text()
