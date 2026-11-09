"""Wait for immutable decision arrays, verify SHA and render all head-family corners."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(a.config.read_text())
    # Keep Gaussian then mixture parallax as the first two for continuity of
    # the selected population plot; all eight within-star plots use common axes.
    order = [
        "decision6_gaussian_parallax_20261106_v1",
        "decision6_mdn_parallax_20261106_v1",
    ]
    order += [x["run_id"] for x in cfg["arms"] if x["run_id"] not in order]
    pending = set(order)
    ready = {}
    failures = {}
    deadline = time.monotonic() + 24 * 3600

    def save(status="waiting_for_arrays"):
        t = a.output / "manifest.tmp"
        t.write_text(
            json.dumps(
                dict(
                    status=status,
                    pending=sorted(pending),
                    ready=ready,
                    failures=failures,
                    training_performed=False,
                    model_loaded=False,
                ),
                indent=2,
            )
            + "\n"
        )
        t.replace(a.output / "manifest.json")

    save()
    while pending and time.monotonic() < deadline:
        for rid in sorted(pending):
            path = ROOT / rid / "manifest.json"
            if not path.exists():
                continue
            try:
                m = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if m.get("status") == "failed":
                failures[rid] = m.get("error", "upstream failed")
                pending.remove(rid)
                save()
            elif m.get("status") == "remediation_pilot_complete":
                artifacts = {}
                for name in ["whole-vector-corners.npz", "development-predictions.npz"]:
                    source = ROOT / rid / name
                    if sha(source) != m["artifacts"][name]["sha256"]:
                        raise ValueError("Producer SHA differs: " + rid + "/" + name)
                    shutil.copyfile(source, a.output / (rid + "-" + name))
                    artifacts[name] = dict(
                        sha256=sha(source), bytes=source.stat().st_size
                    )
                ready[rid] = dict(
                    selected_checkpoint_sha256=m["selected_checkpoint_sha256"],
                    artifacts=artifacts,
                )
                pending.remove(rid)
                save()
        if pending:
            time.sleep(45)
    if pending:
        raise TimeoutError("Pending runs retained in manifest")
    if any(rid not in ready for rid in order[:2]):
        raise ValueError("Baseline arrays unavailable; failures preserved")
    runs = [rid for rid in order if rid in ready]
    save("rendering_saved_array_corners")
    subprocess.run(
        [
            sys.executable,
            "scripts/plot_decision_six_corners.py",
            "--directory",
            str(a.output),
            "--runs",
            *runs,
        ],
        check=True,
    )
    save("complete_saved_array_corners")


if __name__ == "__main__":
    main()
