# Gaia DR3 master: data and verified run procedure (P1)

Runbook for programme step P1 (`docs/programme/README.md` §6): the
Gaia-anchored, full-wavelength master built with the
[xmatcher `ray-anchor` engine](https://github.com/astroai/xmatcher/blob/main/docs/ray-anchor.md).
The engine keeps every Gaia row and adds, per survey, the best counterpart
within 1.5″ plus `sep_arcsec`, `sep_second`, `n_within`, `crowded`, `pm_warn`
and `quality`. Status on 2026-11-24: pilot verified (three CANFAR runs, below);
the full run is proposed, not started.

## Output-location rules

- MSA writes only under `/arc/projects/astroai/msa` (`vos:astroai/msa`): the
  master at `/arc/projects/astroai/msa/xmatcher/gaia_master.hats`, pilots next
  to it (`xmatcher/pilot_gaia_master*.hats`), reports under `runs/<run_id>/`.
- `/arc/projects/hats` is reserved for the community to read HATS catalogues:
  read the trees there in place, never write there (no outputs, no new
  mirrors or syncs). xmatcher refuses every write under it.
- xmatcher's catalogue cache (mirrors of remote HATS inputs) lives under
  `/arc/projects/astroai/hats` (`XMATCHER_CACHE_ROOT`, xmatcher's CANFAR
  default); cached catalogues there may later be promoted to community-ready
  HATS. The master needs no mirror: run with `--no-sync`, so a tree that cannot
  be read in place is an error, never a download.

## Inputs

Anchor: `gaia_lsdb_hats`, read in place from `/arc/projects/hats/gaia/dr3`
(1,811,709,771 rows, 2,016 partitions at orders 2–6, epoch J2016.0). Every
survey is a community tree under `/arc/projects/hats` with a bundled xmatcher
entry (`local_paths` or a `PENSUS_SYNC.json` record, pinned `hats_nrows` and
`obs_collection`, `default_columns`). Rows and selected-column sizes are from
the partition footers read by the pilot's estimate pass.

| Entry | Tree | Rows | Partitions | Selected columns (compressed) | Epoch |
|---|---|--:|--:|--:|--:|
| `twomass_psc_hats` | `2mass/psc` (sync record) | 470,992,970 | 219 | 13.8 GB | 2000.0 |
| `allwise_hats` | `allwise/catalog` | 747,634,026 | 1,071 | 24.7 GB | 2010.0 |
| `catwise2020_catalog` | `catwise/2020` | 1,890,715,640 | 4,080 | 138.1 GB | 2015.4 |
| `skymapper_dr4_hats` | `skymapper/dr4` | 724,329,344 | 356 | 23.9 GB | 2015.0 |
| `sdss_dr16_photoprimary` | `sdss/photoprimary_dr13` | 469,050,917 | 1,247 | 43.8 GB | 2000.0 |
| `ls_dr11_sweep` | `legacysurvey/dr11` | 3,967,618,454 | 9,222 | 261.6 GB | 2017.0 |
| `galex_guvcat_ais` | `galex/guvcat_ais_gr67` | 82,992,062 | 158 | 3.6 GB | 2007.0 |
| `vhs_dr7` | `vhs/dr7` | 1,496,104,720 | 3,274 | 55.5 GB | 2014.0 |
| `ukidss_dr11_las` | `ukidss/dr11_las` | 77,685,267 | 188 | 4.3 GB | 2008.0 |
| `ukidss_dr11_gcs` | `ukidss/dr11_gcs` | 64,835,919 | 151 | 3.4 GB | 2008.0 |
| `viking_dr5` | `viking/dr5` | 82,063,585 | 159 | 4.9 GB | 2013.0 |
| `vvvx_dr1` | `vvvx/dr1` | 886,574,093 | 1,663 | 35.1 GB | 2017.0 |
| `vmc_dr7` | `vmc/dr7` | 64,359,516 | 149 | 3.2 GB | 2014.0 |
| `splus_dr4_dual` | `splus/dr4` | 80,155,547 | 218 | 13.4 GB | 2018.0 |
| `pristine_dr2` | `pristine/dr2` | 25,224,342 | 63 | 2.7 GB | 2016.0 |
| `hsc_pdr3_forced_{g,r,i,z,y}` | `hsc/pdr3/forced_*` | 731–784 M each | 2,010–2,574 each | 43–46 GB each | 2016.0 |
| `euclid_dr1_mer_wide` | `euclid/dr1/mer_catalogue_wide_survey/…-hats` | 888,152,165 | 768 (`Npix=N/` directories) | 83.6 GB | 2024.0 |

Left out of the pilot and of the proposed full run: PS1 OTMO, DES Y6 Gold,
DELVE DR3 Gold and UHS DR3 (Pensus was rebuilding or re-importing them:
sessions `pensus-ps1-otmo-rawrap`, `pensus-ps1-import-arm`,
`pensus-des-delve-import`; UHS rebuilt with RA wrapped), and J-PLUS/J-PAS
(string-typed magnitudes). Add a tree back only once its Pensus session has
finished and `xmatcher describe <entry>` resolves it in place; a rebuilt tree
whose row count changed fails the entry's `hats_nrows` pin instead of being
read silently.

No survey is moved to J2016.0: only Gaia carries usable motions (its
`ref_epoch` is already 2016.0). AllWISE `pmra`/`pmdec` are kept as data but are
not used as proper motions (xmatcher ffa5e23; see caveats). `pm_warn` flags a
Gaia source whose motion over the Gaia–survey epoch baseline exceeds half the
radius, or with no usable motion when the baseline exceeds five years.

## Procedure

Preconditions:

1. No input is mid-rebuild:
   `curl -sSL --cert ~/.ssl/cadcproxy.pem https://ws-uv.canfar.net/skaha/v1/session`
   (look for `pensus-*` sessions touching the trees above). Do not touch other
   projects' sessions.
2. xmatcher is the sfabbro fork at 266e85c or later (community trees read in
   place, `--anchor-partitions`, per-task stats and memory reservations).
   The repository is private, so headless jobs receive it as a tarball under
   `/arc/projects/astroai/msa/launcher/xmatcher_src/`, checked by sha256.
3. The MSA HEAD is clean and pushed (the launcher refuses otherwise); commit and
   push the receipt it writes under `experiments/xp/decision-wave-20261106/`.

Headless job: `scripts/run_gaia_master_pilot.sh` (launcher steps
`gaia-master-*` in `scripts/launch_master_label_training_chain.py`). It

- unpacks and checks the xmatcher tarball, installs xmatcher's Pixi env;
- snapshots `/arc/projects/hats` and the cache root (type, mtime, size of every
  entry) before and after, and diffs them by tree;
- runs (subset or full)

  ```bash
  xmatcher -v match gaia_lsdb_hats \
    twomass_psc_hats allwise_hats catwise2020_catalog skymapper_dr4_hats \
    sdss_dr16_photoprimary ls_dr11_sweep galex_guvcat_ais vhs_dr7 \
    ukidss_dr11_las ukidss_dr11_gcs viking_dr5 vvvx_dr1 vmc_dr7 \
    splus_dr4_dual pristine_dr2 \
    hsc_pdr3_forced_g hsc_pdr3_forced_r hsc_pdr3_forced_i hsc_pdr3_forced_z hsc_pdr3_forced_y \
    euclid_dr1_mer_wide \
    --engine ray-anchor -r 1.5 --target-epoch 2016.0 \
    [--anchor-partitions 2/69,6/42837] \
    --no-sync --cache-root /arc/projects/astroai/hats \
    --task-rows 4000000000 --chunk-memory-gb 4096 --no-color \
    -o /arc/projects/astroai/msa/xmatcher/<name>.hats
  ```

  with a local Ray cluster on the session (`RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.05`,
  so ~95% of the memory is schedulable for the per-task reservations);
- greps the log for the "reading … in place (read-only; no mirror)" lines (one
  per input), runs xmatcher's `scripts/validate_anchor_output.py`, and writes
  `pilot_summary.{json,md}` (per survey: match, crowded, ambiguous, `pm_warn`
  fractions, invalid rows dropped; per task: seconds, peak RSS, rows read) and,
  unless `--skip-estimate`, `task_estimates.parquet` (candidate partitions,
  rows and compressed bytes per anchor partition and survey, from footers).

Pilot (verified):

```bash
pixi run -e monitor --no-install python scripts/launch_master_label_training_chain.py --step gaia-master-pilot-v3
```

Full run (proposed; launch only when asked):

```bash
pixi run -e monitor --no-install python scripts/launch_master_label_training_chain.py --step gaia-master-full
```

This is `run_gaia_master_pilot.sh --anchor-partitions all --out
/arc/projects/astroai/msa/xmatcher/gaia_master.hats --expected-rows 1811709771
--skip-estimate` on 16 cores / 128 GiB, CPU only. A restart after an
interruption reruns the step as a new attempt with `--resume`: finished
partitions are kept when the fingerprint (inputs, columns, radius, epoch)
matches; changing any of them starts over. From an interactive session with
xmatcher on PATH, `scripts/run_gaia_master.sh` runs the same command
(`--dry-run` prints it).

Validate (the job does it; to rerun):

```bash
python <xmatcher>/scripts/validate_anchor_output.py \
  /arc/projects/astroai/msa/xmatcher/gaia_master.hats \
  --expected-rows 1811709771 --json gaia_master_validation.json
```

It fails on a null or repeated `source_id` within a partition, on
`hats_nrows` different from the partition sum, and on a row count other than
the expected one.

## Pilot results (2026-11-24)

Runs (reports under `/arc/projects/astroai/msa/runs/<run_id>/`):

| Run | Session | xmatcher | Partitions | Outcome |
|---|---|---|---|---|
| `gaia_master_pilot_20261124` | `ozv1t6ad` | 9b2d810 | 2/69, 6/42837 | all 22 inputs read in place; stopped in the first task: AllWISE's motion fit moved a source by 1,630″ (halo check) |
| `gaia_master_pilot_20261124_v2` | `ufvwfkmc` | ffa5e23 | 2/69, 6/42837 | passed; `xmatcher/pilot_gaia_master.hats` |
| `gaia_master_estimate_20261124` | `zm3mjlkm` | ffa5e23 | none (estimates) | per-partition candidate rows/bytes for all 2,016 anchor partitions |
| `gaia_master_pilot_20261124_v3` | `d0680ko2` | 6d65857 | 2/69, 2/133, 5/8283 | passed; `xmatcher/pilot_gaia_master_v3.hats` |

Checks, every run:

- Each of the 22 inputs logged "reading local_paths copy …" (2MASS: "community
  copy", via its sync record) "in place (read-only; no mirror)".
- `/arc/projects/hats` before/after: the only differences were new partitions
  in `des/y6_gold` written by the running `pensus-des-delve-import` session
  (DES is not an input); nothing else added, removed or modified.
- Cache root `/arc/projects/astroai/hats`: only `registry/community_index.json`
  (the index of community sync records) appeared in the first run; later runs
  left it unchanged. Nothing was mirrored.
- Validator: 0 null and 0 duplicate `source_id`; `hats_nrows` equals the
  partition sum (v2: 1,513,619 rows; v3: 2,193,149). 2/69 written by v2 and by
  v3 has the same rows and the same file size (565,159,810 bytes).
- Invalid survey rows dropped (non-finite or |Dec| > 90): 0 for every survey.

Partitions: 2/69 (coarse, RA 34°, Dec 0°: 14 surveys including all HSC bands),
6/42837 (fine, bulge, VVVX), 2/133 (largest estimated task: 289 M Euclid
candidate rows), 5/8283 (LMC outskirts: VMC, Euclid).

Match fraction (% of Gaia rows with a counterpart within 1.5″) and, in
brackets, crowded = ambiguous fraction (`n_within` > 1). "–": no survey
partition overlaps; "0": partitions read, no match.

| Survey | 2/69 (637,729) | 6/42837 (875,890) | 2/133 (1,424,509) | 5/8283 (130,911) |
|---|--:|--:|--:|--:|
| 2MASS PSC | 50.5 (0.01) | 11.7 (0.01) | 44.3 (0.01) | 13.5 (0.00) |
| AllWISE | 77.0 (0.00) | 3.2 (0.00) | 73.9 (0.00) | 27.9 (0.00) |
| CatWISE2020 | 86.7 (0.07) | 9.7 (0.02) | 89.1 (0.07) | 47.5 (0.02) |
| SkyMapper DR4 | 86.3 (0.04) | 25.9 (0.18) | 89.0 (0.05) | 59.9 (0.10) |
| SDSS DR16 | 97.4 (0.60) | – | 0 | – |
| LS DR11 | 100.0 (23.46) | – | 100.0 (5.55) | 100.0 (9.77) |
| GALEX | 12.3 (0.00) | 0 | 10.0 (0.00) | 1.5 (0.00) |
| VHS DR7 | 62.8 (0.40) | 66.6 (1.26) | 98.9 (0.47) | 85.7 (0.39) |
| UKIDSS LAS | 61.8 (0.79) | – | – | – |
| UKIDSS GCS | – | – | – | – |
| VIKING DR5 | 4.1 (0.04) | – | 0 | – |
| VVVX DR1 | – | 94.7 (13.13) | – | – |
| VMC DR7 | – | – | – | 39.8 (0.57) |
| S-PLUS DR4 | 26.5 (0.45) | 0 | 12.9 (1.39) | 55.7 (4.48) |
| Pristine DR2 | 49.3 (0.12) | – | 0 | – |
| HSC g / r / i / z / y | 67.9 / 71.1 / 59.7 / 67.1 / 58.7 (52–63) | – | – | – |
| Euclid DR1 MER | – | – | 93.7 (16.34) | 22.1 (6.02) |

`pm_warn`: 13–23% of rows for surveys ≥ 5 years from 2016 (2MASS, AllWISE,
SDSS, GALEX, UKIDSS, Euclid) at high latitude, 45% in the bulge, ~0 for the
others; nearly all are Gaia two-parameter sources (no proper motion) on a long
baseline. UKIDSS GCS overlaps none of the pilot partitions (resolution only).

Tasks (seconds and peak RSS of the Ray worker; candidate rows read):

| Partition | Rows read | v2 (ffa5e23) | v3 (6d65857) | Estimate (266e85c model) |
|---|--:|--:|--:|--:|
| 2/69 | 598 M | 657.6 s, 19.4 GiB | 213.1 s, 19.6 GiB | 27.9 GiB |
| 6/42837 | 34 M | 43.2 s, 4.1 GiB | – | 7.9 GiB |
| 2/133 | 564 M | – | 202.1 s, 59.5 GiB | 64.9 GiB |
| 5/8283 | 39 M | – | 29.7 s, 3.3 GiB | 5.4 GiB |

v2 → v3 on 2/69: reads 433 → 110 s (8 partition files read concurrently),
time outside read/match 170 → 7 s (surveys without motion are no longer
propagated), match 55 → 96 s (it shared the 16 cores with 2/133).

## Full run: extrapolation

From the estimate pass over all 2,016 anchor partitions and the v3 per-survey
costs:

- **Reads.** 90.9 B candidate rows and 4.63 TB of compressed selected columns
  (3.9× the 23.6 B survey rows: each task's cone circumscribes its HEALPix cell
  plus a 600″ halo, and every overlapping survey partition is read whole).
  The 87 order-2 anchor partitions (4.8% of Gaia rows) take 31% of the bytes.
- **Time.** Fitted v3 costs: read 1.35 s/GB + 0.04 s per partition file;
  match 0.14 s per M survey rows + 0.76 s per M anchor rows per survey; other
  3.9 s per M anchor rows + 6 s. That gives 20 task-hours; with small tasks
  running 50% slower than the fit (5/8283) and 5 s of worker start per task,
  about **32 task-hours** (median task ≈ 45 s, largest ≈ 4–5 min).
- **Memory.** Task estimates (conservative: 1.1× the measured 2/133 peak, 1.4×
  2/69): median 7.6 GiB, p90 12.8, p99 27.3, max 64.9 GiB (2/133); 13 tasks
  above 32 GiB. Ray reserves each estimate (capped at 90% of the node), so
  heavy tasks are not packed together.
- **Wall time on 16 cores / 128 GiB.** CPU bound ≥ 2.0 h; memory-reservation
  bound ≈ 330 GiB·h / ~109 GiB ≈ 3 h. Expect **3–5 h**; the plan (footers of
  ~36k selected partitions, read concurrently) adds a few minutes. Two such
  sessions as one Ray cluster would roughly halve it; not needed.
- **Output.** Gaia columns ≈ 80–110 B/row; each survey adds 45–190 B per
  matched row (2MASS 52, CatWISE 70–85, LS DR11 85, SDSS 117, S-PLUS 189) and
  ≈ 0 for unmatched rows. The pilot partitions span 192 B/row (bulge, 7
  surveys) to 886 B/row (2/69, 14 surveys). With ~0.47 B 2MASS, ~0.68 B
  CatWISE, ~0.53 B LS DR11 and a few 10⁸ each for AllWISE, SkyMapper, VHS and
  VVVX matches: **≈ 0.45 TB (0.35–0.6 TB)**, not the 3–5 TB budgeted earlier.
  `/arc` had 634 TB free. Cache growth: none.

## Known data caveats

- **AllWISE motions.** The AllWISE motion fit is not astrometry: wired as
  proper motion it moved a source by 1,630″ between 2010 and 2016 (pilot v1).
  xmatcher ffa5e23 keeps `pmra`/`pmdec` as columns only; AllWISE positions stay
  at 2010.
- **LS DR11 duplicates.** Every Gaia source has an LS row at its Gaia position
  (`REF_CAT=G3`, separation 0.000″, match 100%), and 6–23% of Gaia sources have
  a second row at the same position (`sep_second` ≈ 0), from `RELEASE` 11010 and
  11011. The best match among exact duplicates is the lowest row; P2 must pick
  one release (or brick) per object.
- **HSC forced photometry** has deblend parents and children (and probably
  patch/tract overlaps) but no `isPrimary` flag: 52–63% of Gaia rows have ≥ 2
  rows within 1.5″ (`sep_second` ≈ 0.025″), and 39% of best matches are
  parents (`deblend_nChild > 0`). Fluxes are `instFlux`. P2 should prefer
  `deblend_nChild == 0` rows and calibrate the fluxes.
- **Euclid DR1 MER** partitions are `Npix=N/` directories of files (read as
  such; xmatcher 6d65857 also counts their footers). 16% of Gaia rows in 2/133
  have two MER rows within 1.5″.
- **Crowding** in the bulge is real: VVVX 13% crowded at 95% match.
- **Gaia XP** stays out of the master; join `/arc/projects/hats/gaia/dr3_xp_continuous`
  on `source_id` as a sidecar.
- **UKIDSS GCS** was not exercised by the pilot (no overlap); verify it on
  the full-run validator report.
- **Remote fallbacks** (`gaia_hats`, `sdss_hats`, `ps1_hats`, `catwise_hats`,
  `vhs_hats`, ...) are CDS/LSDB HTTPS copies; they are not needed for this
  master and must not be mirrored into `/arc/projects/hats`.
