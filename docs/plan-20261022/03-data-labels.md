# 03 — Data & labels plan input (agent `data-audit`, 2026-10-22)

Scope: label inventory (local HDF5 reads), gap analysis for the six requested label
families, cross-survey calibration status, all-Gaia input store, dust/PM/position
proposal, ranked acquisition plan. `verified` = read from a local file this session or
from a live publisher page (URL given); `producer-only` = repo/monitor claim not
re-read here; `needs verification` = not confirmed. Row counts are measurement rows
unless the column says unique stars. No code/config/ledger was modified.

## 1. Current label inventory (all paths under `experiments/xp/results-stellar-improvement/`)

| artifact (file) | rows / unique Gaia DR3 | labels present | errors | flags / quality | role splits | key caveat (verified) |
|---|---|---|---|---|---|---|
| `label_registry_v2/labels.h5` (116 MB) | 3,914,337 meas rows; FeH 3,421,262 rows / **1,509,545** unique stars; selected **737,001 unique** | FeH, MH (separate) | `uncertainty` per row | `quality_pass`; FeH qpass 762,135 rows | split 516,203/110,384/110,414 | selected = GALAH 550,851 + APOGEE 185,859 + Li2025 291; MH 493,075 rows, **0** quality-pass; no alpha/age/distance |
| `joint-native-preparation-v2/joint-native-labels.h5` (120 MB) | 756,836 rows/unique | Teff, logg, FeH (3 cols; order confirmed by `tests/test_joint_native_labels.py:304`) | per-target `uncertainty` | `targets/observed`, `quality_pass`, per-source raw flags | fit 20,216; fit_complete **19,299**; tune 2,226; cal 2,136; test **100,634**; ref-holdout 1,819; external 1,358 | observed+finite: Teff 736,710 / logg 717,777 / FeH 736,710; all-3 = 717,777; astrometry observed 673,345 but `distance_fit_eligible=0` |
| `enriched_feature_cohort_v1/enriched-feature-cohort.h5` (663 MB) | 756,836 rows; **673,345** complete-110-XP | labels via layer index (native/PIGS/Li) | `reported_feature_errors` (140) | `feature_available`, `feature_observed`, `input_source` | native 737,001 / PIGS 11,827 / Li 8,304 rows; strict_fit 19,859 / 222 / 190; native test 100,620 | 140 feature slots; **83,491 rows have no usable XP**, of which **25,113 are labelled**; layer counts are repeated measurements, not independent stars |
| `desi_labels_v1/desi-labels.h5` (960 MB) | 6,372,607 raw rows; eligible 2,977,019 rows; **2,976,271 unique selected** | FeH (raw + corrected), Teff, logg | `feh_err_native`, `teff_err_native`, `logg_err_native`, `temperature_correction_err_independent` | `success`, `primary`, `rvs_warn=0`, `RR_SPECTYPE=STAR`, SN_R≥10, 0<FEH_ERR<0.5, correction bounds | split 2,083,224/446,590/446,457 | **RVS only**; published temperature correction pinned (`feh_correct.py@10b6ab6`); sky median sep 0.0038", 447 IDs >1" rejected; random errors ≠ total calibrated |
| `pigs_labels_v1/pigs-labels.h5` (8 MB) | 13,235 spectroscopy rows; **11,827 unique**; distance 10,379 | FeH (+derived distance posteriors) | FERRE/ULySS totals (0.13 dex floor already inside) | `goodF/goodU`, `double=false`, dup resolved by pipeline | selected 11,827 | distances are StarHorse model posteriors, not geometric truth; FERRE/ULySS correlated, never averaged |
| `li2025-silver-v1/li2025-silver-labels.h5` (0.8 MB) | 8,440 raw rows; **8,304 unique** | FeH (CaT) | reported error + separate published 0.2 dex Ca/Fe allowance | `q_EW1`, method, `cat_index_flag`, SNR>20 cut | 8,304 silver | metal-poor CaT sample; 62 duplicate IDs are distinct observations/methods |
| `posterior_labels_v2/posterior-labels.h5` (15 MB) | APOGEE17 63,750 rows (age valid 63,146); GALAH 127,016 (127,016) | age (Gyr), distance (kpc) percentiles 5/16/50/84/95 | asymmetric percentiles | raw `AgeFlag`, `AgeInOut`, `InputFlag`, `OutputFlag` verbatim | n/a | StarHorse model-dependent; median age 6.57 (APO) / 5.64 (GALAH) Gyr; frac >10 Gyr 17.9% / 13.3%; **not independent truth** |
| `age_feature_cohort_v1/age-feature-cohort.h5` (24 MB) | **152,156** unique selected | age percentiles | asymmetric | strict_fit **4,440**; tune 22,741; identity_test 22,830; external_ref 37 | see left | 50/66 external seismic refs feature-available; 66 ref IDs banned from fit/tune; 4,860 cross-survey duplicate candidates |
| `distance_feature_cohort_v1/distance-feature-cohort.h5` (28 MB) | **153,725** unique selected | distance percentiles (kpc) | asymmetric | strict_fit **4,452**; broader 106,466; tune 22,867; identity 22,932 | ref union **99,574** banned fit+tune | Hipparcos reference 4,166 complete-XP candidates (159 non-positive parallax); convert `parallax_mas = 1/d_kpc`, never invert noisy π |
| `pristine_quality_view_v1/pristine-quality-view.h5` (84 MB) | 6,380,652 pristine rows | **no FeH label** (CaHK photometry only) | per-row CaHK error | `input_eligible` 4,331,240; `weak_candidate` 3,660,370 | native input-eligible 19,864 (100 FeH<−2, 26 <−3); strict 583 | photometric metallicity percentiles, **not spectroscopic truth**; CaHK already dereddened |
| `dataset_view_v2/dataset-view.h5` (47 MB) | 9,000,000 train + 1,000,000 validation index | label pointers | – | `label_holdout_split`, `selected_label_row` | fixed 9M/1M | index with external links, not a data copy |

**Verified tail counts (unique quality-pass stars, not rows):** registry selected FeH<−1 16,385 / <−2 **1,325** / <−3 **93**
(93 is 24 Li2025 CaT + GALAH/APOGEE); joint-native FeH<−2 **1,049 total, 145 in test, 25 in fit_complete, 0 below −3**;
DESI selected <−1 630,643 / <−2 **78,419** / <−3 **3,829**; PIGS <−2 2,069 / <−3 16; Li2025 <−2 8,288 / <−3 24.

**Union across registry+DESI+PIGS+Li: 3,726,162 unique FeH-labelled stars, of which 89,446 at FeH<−2** — i.e. the
metal-poor tail is *already on disk* in DESI and is excluded only by policy, not availability. Alpha labels: **0 rows anywhere**
(no quantity `alpha` in registry; DESI `desi-labels.h5` dropped `ALPHAFE`; see §2).

BRIEF quotes "141 test stars FeH<−2, 14 below −3" from the density review; I measure 145/<−2 and **0**/<−3 on the
`joint-native-v2` test mask. Reconcile the mask definition before any tail number is published.

## 2. Gap analysis for the six requested label families

| family | public status now (2026-10-22) | volume / rows | relevant columns | Gaia DR3 ID? | expected new unique labels (reasoning) | tail FeH<−2 yield | ingestion effort | main risk |
|---|---|---|---|---|---|---|---|---|
| **SDSS DR20** ("all surveys") | live: <https://www.sdss.org/dr20/> (DR20 paper arXiv:2607.26149). **No new APOGEE** — DR19 copied; APOGEE/ASPCAP still DR17-equivalent. New: **~1.5 M MWM optical BOSS stars**, Astra params from 11 pipelines (ASPCAP, BOSSNet, SLAM, ApogeeNet/2, SnowWhite, ESP-HS) | use Astra summary files (`astraAllStar*`), GB-scale, not spectra | Teff, logg, FeH + per-pipeline flags; abundances page incl. **BOSS-CLAM α/Fe VAC**; MineSweeper VAC vetted on halo stars | yes (SDSS-V targeting carries Gaia DR3; verify per-file) | O(1e5–1e6) new unique spectra not in APOGEE/GALAH; APOGEE part adds **0** new labels | O(1e3–1e4) plausibly (Halo program + MineSweeper; **needs verification** from file) | med: 11 pipelines → choose per target class, flags, no averaging, cross-survey offset | flag_bad misses ~<<1% FGKM but higher for compact binaries; ASPCAP cool-metal-poor local-minimum failures documented on the same page |
| **DESI** | **DR1 only** (19 Mar 2025), <https://data.desi.lbl.gov/doc/releases/> lists EDR+DR1 as latest → **no public DR2** | DR1 iron RVTAB 6,372,607 rows; MWS VACs: MWS catalog, BHB, SpecDis (~4 M), SPDist, stellar reddening | Teff, logg, FeH, **ALPHAFE (+_ERR) present in the collected FITS schema but not ingested**; `RVS_WARN`, `CHISQ_*`, `SN_R` | yes (join audited, median 0.0038") | +**~2.98 M** FeH already; +**~2.9 M alpha/Fe** for near-zero cost (re-read ARC FITS, extend dtype) | **+78,419** (3,829 <−3) — largest single tail asset | low for alpha re-ingest; medium for DR2 later | FeH needs published Teff/logg correction; DESI−APOGEE scatter 0.0746 vs quoted 0.0102 (§3); RVS-only selection biases hot/dwarf |
| **GALAH** | **DR4 is still latest public** (<https://www.galah-survey.org/> shows DR3/DR4 only; `/dr5/` → 404). GALAH 2 = HERMES+Veloce, Veloce calibration spectra being analysed; **no DR5 date published** | DR4 allstar 917,588 rows (9.2% already ingested as 550,851 quality-pass) | Teff, logg, FeH, 30+ element abundances, `flag_sp/flag_sp_fit/flag_fe_h` | yes | ~0 new; 366,737 DR4 rows are un-ingested feature/label candidates | +O(1e2) at best (GALAH is thin below −2) | low (re-run existing parser with relaxed flags) | **GALAH logg is parallax-adjusted** (documented in `configs/joint-native-preparation.yaml`); DR5 timing unknown |
| **ESO surveys** | ESO sites returned **503 (system upgrade)** today → status **needs verification**. Gaia-ESO DR5/iDR6 public status, 4MOST DR1 date, and ESO HARPS/UVES/GIRAFFE Phase-3 volumes all unverified | – | – | – | GES ~1e5 stars total, high-res UVES subset O(1e3), strongest for **benchmark** tail stars; 4MOST DR1 likely ≥2027 | O(1e2–1e3) but *benchmark-grade* | high (heterogeneous reductions, abundance scales) | access + proprietary periods; do not plan capacity on unverified dates |
| **Gaia RVS** | DR3 GSP-Spec public now (~5.6 M stars with Teff/logg/FeH/α and quality flags — **needs verification of exact DR3 row count**). **Gaia DR4 goes live 2 Dec 2026 ~12:00 CET** (<https://www.cosmos.esa.int/web/gaia/data-release-4>): `ap_rvs` **35,825,564** rows (GSP-Spec + FLAME ages), `ap_chem` 35,825,564, `ap_xp` **999,939,280**, `gaia_source` **2,000,033,276**, `xp_continuous_mean_spectrum` **1,824,229,813** | DR3 10 TB total; DR4 ~400 TB total, but only 3–4 small AP tables needed (GBs–100s GB) | Teff, logg, FeH, α/Fe, FLAME age/mass; `flags` per DR3 GSPSpec calibration pages | yes | DR3: +1e6-scale; **DR4: +3.6e7 with ages** — the single biggest label upgrade in the plan | DR3 O(1e4), DR4 O(1e5) (needs verification) | med now (DR3 flags/calibration), high for DR4 (new schema + source-id remap) | DR4 source list is independent: mapping needs `dr3_neighbourhood` (2,448,832,159 rows) — plan for a DR3↔DR4 crosswalk, not a join on `source_id` |
| **LAMOST DR12** | **live**: <https://www.lamost.org/dr12/> (DR12 v2.0, survey rows 2023-09-18→2024-06-03; v0/v1.0/v1.1/v2.0) | LRS 12,602,862 spectra / 12,233,809 stars; **AFGK params 8,376,135**; M giant/dwarf/subdwarf 985,962; **MRS 15,471,446 spectra, 3,224,901 star-parameter rows** | Teff, logg, FeH (+α for MRS), SNR/flags; LRS FeH scale known to compress the metal-poor end | yes in DR11+ (verify column name per catalogue) | +O(1e6) unique FeH, +O(1e6) Teff/logg; large overlap with Gaia XP footprint | O(1e3–1e4) after flag+scale cuts (**needs verification**) | med-high: LRS/MRS are different pipelines, differing flags; **must fit a LAMOST→native FeH offset before use** | LAMOST FeH scale bias at low metallicity; never train raw LAMOST FeH against native FeH without a calibrated offset layer |

Cross-family conclusion: **no new download is required to grow the metal-poor tail 60×** (DESI selected 1,325 → ~79 k at <−2);
the blocking work is ingestion + calibration, not acquisition. Alpha goes from 0 → ~2.9 M by re-reading the DESI FITS
already on ARC. Age stays model-dependent until DR4 FLAME + APOKASC-style references are in.

## 3. Cross-survey calibration status (verified from `survey_overlap_v1/`, `native-overlap-audit.*`)

| pair | pairs (test) | median Δ (dex) | robust scatter | median quoted pair σ | >3σ fraction | tail pairs (<−2) |
|---|---|---|---|---|---|---|
| GALAH − APOGEE | 14,954 (2,296) | −0.0037 | 0.0482 | 0.0588 | 1.8% | 10 train / 2 test (APOGEE<−2) |
| GALAH − DESI | 4,096 (605) | −0.0003 | 0.0730 | 0.0610 | 3.1% | 1 by GALAH, 0 by DESI |
| APOGEE − DESI | 2,061 (322) | +0.0024 | **0.0746** | **0.0102** | **66.8%** (66.5% incl. independent Teff term) | **0** |
| GALAH − PIGS | 7 (2) | +0.0269 | 0.163 | 0.176 | 0% | – |
| APOGEE − PIGS | 45 (7) | −0.0223 | 0.194 | 0.153 | 0% | – |
| PIGS − DESI | **0** | – | – | – | – | – |
| Li2025 − native | 244 selected IDs | +0.120 (mag method) / −0.108 (colour) | – | – | – | different cohorts, not a calibration |
| Pristine CaHK vs native | 19,864 input-eligible, 18,711 broader | pilot: CaHK adds 3.1–4.7% FeH MAE improvement; 13 test stars <−2, 4 <−3 | – | – | – | too few to claim tail gain |

DESI's quoted random error is **7× too small** relative to the paired scatter, and no APOGEE/DESI tail pair exists below −2.
Therefore: **no global calibration is currently defensible, and the metal-poor cross-survey scale is unconstrained.**

Proposed label model (to be validated, not assumed):
1. **Source-tagged layers**, never merged values: one row per (source, survey, quantity), with survey code, native flag
   bits, raw reported error, and a survey×regime offset/scale parameter. The registry already has this shape — extend it.
2. **Per-survey affine calibration in latent label space** `y_s = a_s + b_s·y + ε_s`, with `ε_s` inflation learned from
   overlap pairs (`native-overlap-pairs.npz`, `survey_overlap_v1/overlap-pairs.npz`) and *stratified by Teff/logg/regime*
   (`desi-conditional-diagnostics.json` already shows Teff-bin-dependent DESI offsets: −0.015 at 4200–5000 K vs +0.0002 at 5800–6600 K).
3. **Hierarchical / noisy-label objective**: survey-specific noise scale + shared intrinsic label; loss weights from the
   learned `ε_s`, not from quoted σ; PIGS 0.13 dex floor and Li 0.2 dex allowance kept as *separate* components.
4. **Tail release gate** — a FeH<−2 claim requires at least two *independent* metal-poor references agreeing to a stated
   tolerance: Pristine weak-candidate (photometric, no truth claim), Li2025 CaT (n=8,288, systematic ≲0.2 dex), DESI dark-time
   high-S/N subset, APOGEE/ASPCAP tail, Gaia DR4 GSP-Spec after Dec 2026, and globular-cluster members (not yet acquired).
   Until then: report tail numbers, do not calibrate away the offset.

## 4. All-Gaia input store

- **Gaia DR3 totals** (producer-only from BRIEF, consistent with DR4 page's DR3 = 10 TB): 1,811,709,771 sources; XP
  continuous-mean-spectrum subset ≈ **219 M (12%)**; RVS subset ~3.4e7. The 10 M benchmark is 9 M train + 1 M validation
  sampled without replacement proportional to shards, seed 42 (`docs/xp-benchmark-dataset.md`), and the 150 M training cache
  is "150 M distinct original TRAIN rows" (`docs/residual-huber-150m-20261020.md`), so both are **subsamples of the XP-rich
  12%**, never of all Gaia.
- **`data/add-gaia-features.py` writes `220M_pretrain_data.h5`** — the repo's own name for the XP-rich subset; treat 220 M
  as the DR3 ceiling for XP-conditioned work (producer-only).
- **Full-sky feature groups today** (`pipeline/registry.py`, `data/pretraining-partial-table-maker.py`): Gaia G/BP/RP
  (all-sky), WISE W1/W2 via CatWISE (all-sky), 2MASS (all-sky), PS1 (dec ≳ −30°), SDSS (legacy footprint ~1/3 sky),
  SkyMapper DR4 (southern), plus PARALLAX/EBV/pmra/pmdec/RA/DEC. **No feature group other than Gaia is all-sky**, so for
  most of the 1.8 B the model runs on Gaia-only + missing-mask tokens — that is a legitimate regime, not a failure.
- **Can stars without XP be scored? Yes, mechanically**: `eval_forward(..., drop_xp=True)` exists
  (`pipeline/steps.py:430`) and the checkpoint score *includes* an XP-off eval; missing orders are a mask, never median-filled.
  **But** the training mixture is XP-dominated: `synthetic_xp_masks` only mask stars that *have* XP, so a no-XP star is a
  distribution shift. Evidence that this is fixable: the enriched cohort already carries **83,491 no-XP rows of which 25,113
  are labelled** — a ready XP-off supervised pool, currently excluded by `feature_available`.
- **Drafter note — Gaia DR4 (2 Dec 2026) changes the requirement**: DR4 publishes `xp_continuous_mean_spectrum` for
  **1,824,229,813** rows (≈ all sources). "Stars without XP" is a **DR3-only transitional constraint**; plan a DR3 product
  now and a DR4 product in 2027 (new basis dimensions → `ColumnRegistry(n_bp, n_rp)` is already parameterized).
- **Storage estimate** (assumption-based, measure before committing): 140 float32 slots = 560 B/row → 150 M rows ≈ **84 GB**
  (matches `docs/joint-continuation-30e.md:129`) ⇒ 1.8 B rows ≈ **1.0 TB** dense. A no-XP compact schema (~26 photometry +
  6 astrometry float32 + int64 id ≈ 190 B/row) ⇒ ~**340 GB** raw, ~**120–170 GB** as zstd parquet. Note the legacy loader
  caches the whole table in RAM (117 GB for 10 M = 11.7 kB/row) — that path cannot be used above ~10 M rows regardless of disk.
- **Compute estimate**: per-star forward is dominated by the 110-order encoder (`OrderEncoder`, width ~256) +
  TabM ancillary trunk; O(1e7) FLOPs/star ⇒ 1.8 B stars ≈ 1.8e16 FLOP ≈ **0.5–2 GPU-hours at 5–10 TFLOP/s effective**.
  The catalogue is **I/O- and preprocessing-bound, not FLOP-bound**; per-healpix parallelism is the lever, not more FLOPs.
- **Partition scheme**: `StellarBatch` already carries `healpix` as a first-class field and `docs/pipeline.md` says the loop
  "iterates partitions. It does not concatenate a shard". Use **HEALPix nside=64** (50,331 cells; ~36 k stars/cell at 1.8 B)
  written as one parquet file per cell, rows sorted by `source_id`, with a partition manifest (count, sha256, schema,
  nside, `nested` scheme) and `source_id` range shards only as a secondary key. nside=32 is the cheaper fallback (12,582
  cells, ~144 k stars/cell). HATS is the natural interchange format but is **unwired** (`docs/pipeline.md` "Not in this
  commit") — do not block on it.
- **CANFAR facts (from `canfar-lab-workflow`; verify live with `df -h`, `canfar lab status --json`, `canfar info <id>`)**: `/scratch`
  is **session-only** (wiped at session end, non-desktop ceiling often ~200 GiB, operator-overridable); `${WORK}` is
  **ephemeral** code+env space (not `$HOME`); `/arc/projects/<group>` is shared persistent project storage; `/arc/home` is a
  **small** persistent quota for dotfiles only. Product: ~1 TB feature store + ~0.3–1 TB output catalogue **must** live on
  `/arc/projects/k-pop`, never `/scratch`. Actual project quota is **not stated anywhere in the repo/skill → human must
  confirm** before a full-sky build.

## 5. Dust / proper motions / positions

**Available today**: 2D SFD `E(B−V)` joined with `dustmaps.SFDQuery` in `data/pretraining-partial-table-maker.py:19-27`
(`DUSTMAPS_DATA_DIR` must be pre-fetched); PARALLAX + `e_parallax`, pmra/pmdec + errors, RA/DEC are declared in
`pipeline/registry.py`; the audit recorded EBV quantiles but **"raw_unit: not established by cohort metadata", "map_provenance:
not established"** and `safe_for_unqualified_physical_extinction_input: false` (`experiments/xp/monitor/audit_joint_native_readiness.py`).
Hipparcos reference block has `gaia_ruwe`, `number_of_gaia_neighbors`, `gaia_astrometric_params_solved`, epoch-corrected
separation; the 140-slot feature table has **no RUWE** and no per-star AV uncertainty.

**Missing for the distance goal**: 3D dust (Edenhofer/Green/Bayestar; none ingested, no package pinned in `pyproject.toml`),
per-star `A_V` uncertainty and validity, RUWE/`ipd_frac_multi_peak` at feature level, epoch propagation for high-PM stars,
and an accepted treatment of the parallax zero point.

**Why the "physical core" masks them** (`docs/stellar-prediction-final-design-20261017.md:29,33-47`): PARALLAX, EBV, pmra,
pmdec, RA, DEC are masked in **both** supervised and reconstruction calls because (a) the EBV column is a 2D
distance-independent map, not a stellar reddening, and its units/provenance failed audit; (b) parallax + kinematic/position
shortcuts are label leakage for a model that must generalise; (c) the inherited parent saw astrometry, so an input change
never "erases that history".

**Proposal (three tiers, each an ablation):**
1. **Context/prior only (default)**: 3D dust queried at the *predicted* distance, iterated 2–3 times, used as external
   context and for reporting — no gradient into the point head. Positions/PM only for healpix bookkeeping, crowding and
   spatial-error diagnostics.
2. **Joint likelihood**: distance `s` and stellar `A_V≥0` inferred jointly with the distance likelihood; `A_V` prior from
   a named 3D map with its published uncertainty and a per-star validity flag; extinction law stated explicitly. This is
   the tier that can legitimately claim "dust improves distant-star distance".
3. **Input-channel ablation (declared extra information)**: AV at Gaia-parallax distance as a 141st slot, flagged
   "measurement-assisted"; never mixed into the feature-only product.
**Required bins / references to prove gain**: `d > 3 kpc`, `|b| < 15°`, reddening quartiles, magnitude quartiles; Hipparcos
nearby cohort as the negative control (must not improve); eclipsing/GC/APOKASC distances for the distant control.
**What would falsify the gain**: (i) improvement vanishes when the same model is trained with the map shuffled within
sky patches; (ii) gain appears only where the map was built from the same stars/photometry (circularity); (iii) accuracy
improves near the Sun but not at `d>3 kpc`; (iv) per-bin calibration degrades (coverage 68/95% breaks) even as MAE drops;
(v) no improvement over a distance-only refit that *also* gets the map, i.e. the gain was information, not dust physics.

## 6. Ranked ingestion / acquisition plan

| # | dataset | action | size | new unique labels | tail <−2 | effort | gate before use |
|---|---|---|---|---|---|---|---|
| 1 | **DESI DR1 `ALPHAFE`/`ALPHAFE_ERR`** (FITS already on ARC, sha `eae3b318…`) | re-read, extend `desi-labels.h5` dtype | 0 new bytes | +~2.9 M alpha | – | S (hours) | publisher sha recheck, flag semantics, publish a new versioned file |
| 2 | **DESI selected FeH into the joint training pool** | policy change + calibration layer | 0 | +2.98 M FeH | **+78 k** | M | §3 calibration gate; keep source-tagged, do not merge values |
| 3 | **Gaia DR3 GSP-Spec (`ap_rvs`/`ap_chem`)** | fetch AP tables via TAP (GBs) | GBs | +1e6-scale Teff/logg/FeH/α | O(1e4) | M | DR3 GSPSpec logg/FeH calibration pages; flag subsetting |
| 4 | **LAMOST DR12 LRS AFGK (8,376,135 rows) + MRS (3,224,901)** | fetch catalogues + Gaia DR3 crossmatch | 10s GB | +O(1e6) FeH/Teff/logg | O(1e3–1e4) | M-H | LAMOST→native offset layer; flag semantics; never raw-concat |
| 5 | **SDSS DR20 Astra summary files + BOSS-CLAM/MineSweeper VACs** | fetch summary FITS | GBs | +O(1e5–1e6), +α/Fe | O(1e3–1e4) | M | DR20 paper + Astra flags; per-pipeline choice; no APOGEE gain |
| 6 | **Gaia DR4 (2 Dec 2026)**: `ap_rvs` 35.8 M, `ap_xp` 1.0 B, `ap_chem`, `xp_continuous_mean_spectrum` 1.82 B | schedule; build DR3↔DR4 crosswalk (`dr3_neighbourhood`) | 100s GB subset of a 400 TB release | +3.6e7 with ages; XP for ~all sources | O(1e5) | H | new schema, J2017.5 epoch, source-id remap; do not overwrite the DR3 product |
| 7 | **GALAH DR4 un-ingested rows** | re-run existing parser | 0 | +O(1e5) Teff/logg, few FeH | +O(1e2) | S | GALAH logg is parallax-conditioned |
| 8 | **ESO (Gaia-ESO DR5, HARPS/UVES/GIRAFFE, 4MOST)** | discovery when ESO is back online | unknown | benchmark tail stars | O(1e2–1e3) | H | access/proprietary; **do not plan capacity on unverified dates** |
| 9 | **Globular-cluster + open-cluster memberships** | construct | small | tail/age validation only | O(1e3) | M | membership + isochrone-free evidence standard |

**Audit gates that apply to every row of the table** (they already exist in this repo and must not be relaxed):
publisher sha256 (+ `expected_sha256` in `configs/stellar-catalog-sources.yaml`), exact int64 Gaia DR3 ID match with
sky agreement (≤1″; the DESI job rejected 447 IDs >1″), unit/G-normalisation audit (XP errors are raw units, divide by
`10**((8.5−G)/2.5)`), flag semantics read from the publisher ReadMe (Pristine point-source flags are −1/−2, not +1;
Li2025 `q_EW1` is a *method* flag, not bad-quality), **no naive averaging** of repeated/correlated measurements,
preserve every raw row/rejection, version the HDF5 and re-derive masks from the published file.

**Human decisions required**: (i) `/arc/projects/k-pop` quota for a ~1 TB feature store + ~0.3 TB outputs; (ii) approval to
run the DESI-pool policy change (2) — it is the largest scientific jump available and it changes every downstream number;
(iii) whether to schedule the DR4 product for Dec 2026/Jan 2027 (it supersedes parts of 3 and 5); (iv) ESO/4MOST access and
proprietary-data status; (v) acceptance that 88% of DR3 sources have no XP and no external labels, so the all-Gaia product
must ship with explicit per-source validity flags rather than a single accuracy claim.

## What this implies for the plan

- The metal-poor bottleneck is **not acquisition**: 79 k DESI tail stars (3.8 k below −3) sit on disk unused, 60× the native tail.
- Alpha is **free**: 2.9 M DESI `ALPHAFE` rows exist in an already-checksummed FITS; the ingestion simply didn't copy the column.
- The real work is **calibration + policy**, and the current evidence says the surveys do not yet share an abundance scale
  below −2 (no pairs, DESI σ 7× underquoted) — so the plan must buy a **tail-validation reference set** (GCs, Li2025,
  DR4 GSP-Spec) before any metal-poor accuracy claim.
- All-Gaia inference is **mechanically supported** (XP masking, `drop_xp`, healpix batches) and **scientifically unvalidated**
  for no-XP stars; run a labelled no-XP pilot (25,113 rows already exist) before the 1.8 B product.
- **Gaia DR4 on 2 Dec 2026** resets both requirements (XP for 1.82 B sources, 35.8 M GSP-Spec labels with ages); the
  plan should deliver a DR3 product now and budget a DR4 migration rather than over-engineer DR3-only coverage.
