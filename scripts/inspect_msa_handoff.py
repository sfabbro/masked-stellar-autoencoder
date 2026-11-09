"""Read-only CANFAR status for the handoff; run with pixi's monitor environment."""
import datetime as dt
import json
from canfar.sessions import Session
from canfar.storage import filesystem

ROOT = 'projects/k-pop/msa_runs/stellar-improvement-20261015/'
RUNS = [
    ('expanded_six_nsf_fullfit_20261107_v1', 'lohnjtr7'),
    ('msa_morning_report_20261107_v3', 'l44pw7oe'),
    ('full_gaia_gaia_stage_20261108_v2', 'v1kyrpx1'),
    ('full_gaia_photometry_stage_20261108_v1', 'fxc6kwlb'),
    ('full_gaia_alpha_stage_20261108_v1', 'lbhf0ucu'),
    ('full_gaia_ages_stage_20261108_v1', 'fvll9eut'),
    ('full_gaia_contract_pilot_20261108_v1', 'hshxs209'),
]


def main():
    print('Checked UTC:', dt.datetime.now(dt.UTC).isoformat(), flush=True)
    fs = filesystem('arc')
    keys = ['status', 'error', 'updated_utc', 'completed_updates',
            'completed_full_fit_epochs', 'current_epoch', 'candidate_stars',
            'completed_gaia_shards', 'matched_unique_candidates',
            'all_gaia_rows_scanned', 'real_no_xp_ssl_rows']
    for run, sid in RUNS:
        try:
            m = json.loads(fs.cat_file(ROOT + run + '/manifest.json'))
            print(json.dumps(dict(run=run, session=sid,
                                  **{k: m[k] for k in keys if k in m})), flush=True)
        except Exception as exc:
            print(json.dumps(dict(run=run, session=sid, read_error=str(exc)[:300])), flush=True)
    live = Session(timeout=60).fetch()
    wanted = {sid for _, sid in RUNS}
    print('Retained session records:', json.dumps([
        {k: s.get(k) for k in ['id', 'name', 'status', 'startTime',
                              'requestedGPUCores', 'cpuCoresInUse', 'ramInUse']}
        for s in live if s.get('id') in wanted]), flush=True)
    print('Completed sessions may disappear; durable manifests remain authoritative.')


if __name__ == '__main__':
    main()
