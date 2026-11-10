"""Read-only CANFAR status for the handoff; run with pixi's monitor environment."""

import argparse
import datetime as dt
import json
from pathlib import Path

from canfar.sessions import Session
from canfar.storage import filesystem

ROOT = "projects/k-pop/msa_runs/stellar-improvement-20261015/"
RUNS = [
    ("expanded_six_nsf_fullfit_20261107_v1", "lohnjtr7"),
    ("msa_morning_report_20261107_v3", "l44pw7oe"),
    ("full_gaia_gaia_stage_20261108_v2", "v1kyrpx1"),
    ("full_gaia_photometry_stage_20261108_v1", "fxc6kwlb"),
    ("full_gaia_alpha_stage_20261108_v1", "lbhf0ucu"),
    ("full_gaia_ages_stage_20261108_v1", "fvll9eut"),
    ("full_gaia_contract_pilot_20261108_v1", "hshxs209"),
    ("full_gaia_corrected_sources_20261109_v2", "eo0ywcd8"),
    ("expanded_selected_checkpoint_qa_20261109_v1", "gqc68dfi"),
    ("expanded_selected_checkpoint_report_20261109_v1", "fwib7zhu"),
    ("native_desi_lamost_population_20261110_v1", "dolylw69"),
    ("full_gaia_native_parent_refresh_20261110_v1", "m9pwdvrf"),
    ("tess_selected_checkpoint_qa_20261110_v1", "vmuapd4t"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, help="Private runtime token file; never copied to reports")
    parser.add_argument("--session-url", default="https://ws-uv.canfar.net/skaha/v1")
    args = parser.parse_args()
    token = args.token_file.read_text().strip() if args.token_file else None
    if args.token_file and not token:
        raise ValueError("Runtime token file is empty")
    print("Checked UTC:", dt.datetime.now(dt.UTC).isoformat(), flush=True)
    fs = filesystem("arc", **({"token": token} if token else {}))
    keys = [
        "status",
        "error",
        "updated_utc",
        "completed_updates",
        "completed_full_fit_epochs",
        "current_epoch",
        "candidate_stars",
        "completed_gaia_shards",
        "matched_unique_candidates",
        "all_gaia_rows_scanned",
        "real_no_xp_ssl_rows",
        "current_step",
        "current_stage",
        "completed_partitions",
        "total_partitions",
    ]
    for run, sid in RUNS:
        try:
            m = json.loads(fs.cat_file(ROOT + run + "/manifest.json"))
            print(
                json.dumps(
                    dict(run=run, session=sid, **{k: m[k] for k in keys if k in m})
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(dict(run=run, session=sid, read_error=str(exc)[:300])),
                flush=True,
            )
    fs.close()
    with Session(timeout=60, **({"token": token, "url": args.session_url} if token else {})) as sessions:
        live = sessions.fetch()
    wanted = {sid for _, sid in RUNS}
    print(
        "Retained session records:",
        json.dumps(
            [
                {
                    k: s.get(k)
                    for k in [
                        "id",
                        "name",
                        "status",
                        "startTime",
                        "requestedGPUCores",
                        "cpuCoresInUse",
                        "ramInUse",
                    ]
                }
                for s in live
                if s.get("id") in wanted
            ]
        ),
        flush=True,
    )
    print("Completed sessions may disappear; durable manifests remain authoritative.")


if __name__ == "__main__":
    main()
