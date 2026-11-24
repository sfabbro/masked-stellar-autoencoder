# MSA data reference: labels, full-Gaia inputs and data sets (2026-11-23)

Scope: the stellar-parameter programme (Teff, log g, [Fe/H], [α/Fe], distance, E(B−V), age for every Gaia
star, then Pensus). This page records where the data are, what has been checked, and the versioned definition
of every data set. The machine-readable copy of section 3 is [`datasets.yaml`](datasets.yaml).

Evidence used: ARC listings and file headers read on 2026-11-23 (VOSpace `nodes` listings, HATS `properties`,
FITS headers, `manifest.json` files), launch receipts in `experiments/xp/decision-wave-20261106/`, and the code
named in each section. ARC ignores HTTP `Range` requests, so Parquet and HDF5 footers could not be read without
downloading whole files. Statements that rest on such files are marked *(not verified)*. Counts marked *(est.)*
are planning estimates, not measurements.

Related pages: [architecture.md](architecture.md) (models), [qa.md](qa.md) (QA checks),
[experiment-history.md](experiment-history.md) (runs and the `msa_runs` loss).

---

## 0. Summary

| Item | Path | Status |
|---|---|---|
| Master spectroscopic labels | `/arc/projects/astroai/msa/runs/master_spectroscopic_labels_20261113_v1/` | Current and complete. 44,576,092 records, 19,255,342 stars (19,205,652 with a Gaia DR3 ID), 43 surveys and pipelines. No newer master exists. |
| Label census / raw sources | `runs/master_label_census_20261113_v2/`, `runs/master_label_sources_20261110_v1/`, `runs/master_survey_stage_20261113_v1/` | Inputs of the master. Complete. |
| Current training table | `/arc/projects/astroai/msa/runs/multisurvey_table_20261117_v5/` | 5,872,237 labelled XP stars (`finetune.fits`), 12,890,518 records, 6,362,706 unlabelled XP stars. |
| Gaia DR3, all sources | `/arc/projects/hats/gaia/dr3` (HATS) and `/arc/projects/k-pop/gaia/GaiaSource/` (3,386 HDF5) | 1,811,709,771 rows, all 152 columns. |
| Gaia DR3 XP coefficients | `/arc/projects/hats/gaia/dr3_xp_continuous` | 219,197,643 rows, on the same HATS partitions as `gaia/dr3`. |
| Gaia-keyed survey matches, all Gaia | `/arc/projects/k-pop/catalogues/adql_matches/` | 2MASS 469,001,402; Pan-STARRS1 951,450,474 IDs; SDSS 119,663,765; CatWISE 678,990,741 rows. |
| Survey HATS mirrors | `/arc/projects/hats/<survey>/<release>` | GALEX, SkyMapper, SDSS, PS1, LS DR11, DES, DELVE, 2MASS, UHS, VHS, UKIDSS, VIKING, VVVX, CatWISE, AllWISE, S-PLUS, J-PLUS, Pristine and others (section 2.4). |
| Pensus outer-joined crossmatch | `/arc/projects/hats/pensus/views/` | Empty. Inputs are staged; no match product yet. |
| A full-Gaia × full-wavelength table | — | Does not exist. Section 2.7 gives the build basis. |

---

## 1. Master label dataset

### 1.1 Current master: `master_spectroscopic_labels_20261113_v1`

Path: `/arc/projects/astroai/msa/runs/master_spectroscopic_labels_20261113_v1/` (7.12 GB, 65 files).

| File | Content |
|---|---|
| `records/<survey>.parquet` (43 files, 3.42 GB) | One row per native catalogue row, no quality cut. |
| `stars.parquet` (1.20 GB, sha256 `2fd54a0e…`) | One row per star. |
| `catalogues.json` (sha256 `c815d874…`) | Native file, HDU, release, URL, sha256 and ID definition per catalogue. |
| `manifest.json` | Per-survey counts, identity methods, metal-poor counts, output hashes. `status: complete`. |
| `work/` | Staging and cached Gaia positional candidates. |

Record schema (verified on `records/pigs_aat.parquet`): `survey, release, pipeline, tier, dependency_group,
catalogue, native_hdu, native_row, native_id, source_id, id_method, match_sep_arcsec, match_n_within, ra, dec,
teff, e_teff, logg, e_logg, fe_h, e_fe_h, fe_h_kind, fe_h_corrected, alpha, e_alpha, alpha_kind, snr, fit_ok,
flag_int, flag_text, star_id, v2_role, has_xp, has_xp_source, label_groups`. Column meanings are in
[`docs/master-spectroscopic-labels.md`](../master-spectroscopic-labels.md).

Provenance:
- Built by `scripts/build_master_spectroscopic_labels.py` at commit `23ca0168`, session `zeeir2pj`, 2026-11-13 10:27 UTC.
- Inputs: census `master_label_census_20261113_v2/observations.parquet` (0.58 GB), staged surveys
  `master_survey_stage_20261113_v1/` (LAMOST DR12 LRS and MRS, DESI DR1 `mwsall-pix-iron.fits`, APOGEE DR17
  allStar, BOSSNet DR20, RAVE DR6 MADERA, Gaia crossmatch XML), raw downloads `master_label_sources_20261110_v1/`,
  and `/arc/projects/k-pop/catalogues/` (GALAH DR4, GSP-Spec).
- `training_annotations: skipped`. The columns `v2_role`, `label_groups` and `has_xp` are empty, and
  `stars_with_xp_known = 0`. XP availability must be joined from Gaia (`has_xp_continuous`).

Totals:

| | Count |
|---|--:|
| Records | 44,576,092 |
| Stars | 19,255,342 |
| Stars with a Gaia DR3 ID | 19,205,652 |
| Stars without a Gaia ID | 49,690 (92,139 records) |
| Stars with fe_h_median < −1 / −2 / −3 / −4 (all pipelines) | 1,769,352 / 385,801 / 120,724 / 19,658 |
| High-res reference stars with [Fe/H] < −1 / −2 / −3 / −4 | 5,110 / 3,101 / 706 / 37 |

The all-pipeline metal-poor counts are dominated by low-S/N DESI SP and LAMOST tails; the high-res counts are
the real reference depth.

Records and stars per survey (from `manifest.json`; `fit_ok` is each survey's own failed-fit flag):

| Survey | Records | fit_ok | Stars |
|---|--:|--:|--:|
| LAMOST DR12 LRS (LASP) | 8,376,135 | 8,376,135 | 5,922,941 |
| DESI DR1 RVSpecFit | 6,372,607 | 4,309,404 | 4,920,596 |
| DESI DR1 SP (FERRE), same spectra | 6,372,607 | 3,659,040 | 4,920,596 |
| Gaia DR3 GSP-Spec | 5,591,594 | 5,591,594 | 5,591,594 |
| LAMOST DR12 MRS (LASP) | 3,224,901 | 3,222,602 | 1,511,213 |
| LAMOST DR12 MRS CNN, same spectra | 3,224,901 | 2,645,064 | 1,511,213 |
| SDSS DR20 BOSSNet | 1,803,883 | 1,674,982 | 1,802,289 |
| SDSS DR20 Astra SLAM | 1,803,883 | 756,043 | 1,802,290 |
| SDSS DR19 Astra AstroNN | 1,585,505 | 1,387,352 | 1,207,023 |
| SDSS DR19 Astra APOGEENet | 1,122,317 | 1,087,946 | 996,804 |
| SDSS DR19 Astra The Payne | 1,111,879 | 1,090,512 | 988,175 |
| SDSS DR19 Astra ASPCAP | 1,095,480 | 775,085 | 974,746 |
| GALAH DR4 | 917,588 | 663,075 | 917,581 |
| APOGEE DR17 | 733,901 | 691,591 | 656,955 |
| RAVE DR6 | 518,387 | 322,367 | 451,636 |
| SDSS DR12 SSPP | 386,337 | 372,334 | 385,825 |
| Gaia-ESO DR5.1 | 114,916 | 94,247 | 114,916 |
| PASTEL 2020 | 81,362 | 43,998 | 37,345 |
| S5 DR1 | 31,777 | 29,584 | 29,598 |
| MaStar params VAC v2 | 24,290 | 24,162 | 24,130 |
| AMBRE FEROS | 18,618 | 4,553 | 6,667 |
| SAGA (Milky Way) | 16,117 | 12,430 | 6,674 |
| PIGS AAT | 13,235 | 13,235 | 13,231 |
| LAMOST VMP (Li+2018) | 10,011 | 10,011 | 9,705 |
| LAMOST CaT (Li+2025) | 8,440 | 8,440 | 8,378 |
| JINAbase 2021 | 7,789 | 1,658 | 4,475 |
| Kirby+2010 dSph and high-res | 3,093 | 3,093 | 3,075 |
| Other metal-poor and high-res papers (15 tables) | 4,539 | 4,478 | 4,529 |

The 15 small tables are Pristine (Aguado+2019), Best & Brightest, Bensby+2014, LAMOST–Subaru (Li+2022),
Roederer+2014, RPA (Hansen+2018, Holmbeck+2020, Sakari+2018, Ezzeddine+2020), SEGUE EMP (Aoki+2013),
SkyMapper EMP (Da Costa+2019 t2/t3, Yong+2021) and Yong+2013 (program and literature).

Pipelines run on the same spectra share a `dependency_group` (APOGEE: ASPCAP DR17, DR19 ASPCAP, AstroNN,
The Payne, APOGEENet; DESI: RVSpecFit, SP; LAMOST MRS: LASP, CNN; LAMOST LRS: LASP, VMP, CaT). They are not
independent measurements.

### 1.2 Versions and their status

| Version | Location | Status |
|---|---|---|
| `master_spectroscopic_labels_20261111_v1` | `k-pop/msa_runs/…` | Lost (12–13 Nov). 29,307,497 records, 19,138,656 stars. This is the 19.1M figure still quoted in `docs/master-spectroscopic-labels.md`. |
| `…_20261111_v2` | `k-pop/msa_runs/…` | Lost. Wrong Bensby+2014 position epoch. |
| `…_20261111_v3` | `k-pop/msa_runs/…` | Lost. 44,576,092 records, 19,255,342 stars; carried `v2_role`, `label_groups`, `has_xp` from training source v7. |
| `…_20261113_v1` | `/arc/projects/astroai/msa/runs/` | **Current.** Rebuilt from public downloads after the loss. Totals identical to v3, without the training annotations. Moved from `k-pop` on 2026-11-20 (0 mismatches in `launcher/migration_20261120_verify.tsv`). |

Censuses: `master_label_census_20261110_v1`–`v3` and `20261111_v4`–`v6` were lost; `20261113_v1` and `v2` survive
(v2 is the input of the current master). Sidecars `master_label_sidecars_20261111_v1/v2` were lost. Listings of
`astroai/msa/runs`, `astroai/uspm`, `k-pop` (top level, `catalogues`, `data`) and `hats` show no other master label
product.

The 8M-star training table was cut from this master: `multisurvey_table_20261117_v5/manifest.json` records
8,000,140 chosen labelled Gaia stars, 5,872,237 with XP inputs and 2,128,095 without. The "of 19.1M" in older
notes is the v1 star count; the current master has 19,205,652 Gaia-identified stars.

### 1.3 Label products outside the master

The master holds spectroscopic Teff, log g, [Fe/H] and α only. The other targets live elsewhere:

| Target | Product | Path | Content |
|---|---|---|---|
| Age | Age audit records | `/arc/projects/astroai/msa/analysis/age_audit_20261122/` (`records_part0–5.parquet`) | ≈ 8.05M records for ≈ 3.6–3.75M stars from 37 catalogue variants (incl. HR23); FLAME kept separately. |
| Age | Age information sources | `/arc/projects/astroai/msa/analysis/age_labels_20261122/` | 21 files: benchmark clusters, WD+MS pairs, rotation, lithium, [C/N], seismology, NEST, activity (see its `README.md`). |
| Age | StarHorse ages | `/arc/projects/k-pop/catalogues/starhorse_ages/` | APOGEE17, GALAH DR3, LAMOST, RAVE, SDSS, GES, RVS tables. |
| E(B−V) | Reddening labels | `/arc/projects/astroai/msa/runs/reddening_labels_20261118_v6/` | `v5_reddening.npz`, `v1_reddening.npz`, intrinsic colours fitted on the 4,697,789 v5 train rows. |
| log g anchors | Seismology | `/arc/projects/k-pop/catalogues/` | `apokasc-3-2024.fits`, `apo_k2_2024.fits`, `apokasc-1-dwarfs.fits`, `kepler-legacy-dwarfs.fits`; loaded by `training/anchor_labels.py`. |
| Teff anchors | IRFM-scale Teff | computed by `training/irfm_teff.py` | 898,839 stars in v5. They are computed from photometry that is also a model input, so they are a consistency anchor, not an independent truth. |
| Benchmarks | Gaia FGK benchmarks v3 | `/arc/projects/k-pop/catalogues/gaia_fgk_benchmarks.parquet` (`GaiaDR3` rounded through float64: 107 of 192 ids not in DR3, 2 more point at a faint neighbour); exact ids in `/arc/projects/astroai/msa/catalogues/gaia_fgk_benchmarks_v3_dr3ids.parquet` (`scripts/fix_gbs_dr3_ids.py`) | 192 stars with a stored id, all recovered (63 with XP inputs in v5, built from the wrong ids). |
| Distance | Gaia DR3 parallax | `hats/gaia/dr3` | Label in the parallax-free variant, input in the parallax-conditioned variant. |
| Co-natal | Consistency targets | `runs/consistency_table_20261118/`, `analysis/consistency/` | 1,440,253 targets (El-Badry+2021 pairs, Hunt & Reffert 2023 members incl. globulars); 1,209,544 with XP inputs. |

### 1.4 Gaps in the label data

- **Stale document.** `docs/master-spectroscopic-labels.md` names `…20261111_v3` as current. The current product is
  `…20261113_v1`.
- **Dead native paths.** `catalogues.json` points staged files at `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/…`,
  which no longer exists. `read_native_rows` needs the prefix mapped to `/arc/projects/astroai/msa/runs/`.
  GALAH and GSP-Spec paths in `k-pop/catalogues` are still valid.
- **No XP flag.** `has_xp` is empty; join `has_xp_continuous` from `hats/gaia/dr3`.
- **No roles.** The master carries no split; the split is defined in section 3.
- **Sources not available:** H3 (no public catalogue), 4MOST (no release), Gaia DR4 GSP-Spec (2 Dec 2026),
  DESI DR2 (spring 2027), SAGA Local Group, AMBRE HARPS/UVES.
- **Unused at scale:** about 11.2M labelled stars outside the 8M cap and 2,128,095 labelled no-XP stars inside it.
- **Lost:** census v6 annotations, `loose:` and `chem:` label groups, training sources v6–v9, the StarHorse
  age/distance label sidecars, the 1-in-8 no-XP reconstruction pool. All are rebuildable from surviving inputs.
- **Ages, α and distance** are not in one long-format label table with the spectroscopic records. A master v2
  should add the age records and the anchors as further `survey` blocks with `tier = anchor | benchmark`.

---

## 2. Full-Gaia input catalogues

### 2.1 Gaia DR3 itself

| Product | Path | Rows | Notes |
|---|---|--:|---|
| `gaia_source`, HATS | `/arc/projects/hats/gaia/dr3` (491 GB) | 1,811,709,771 | All 152 columns; epoch J2016.0; order-6 partitions, max 2M rows; 10″ and 300″ margins; `source_id` index. Built from the 3,386 ESA csv.gz files checked against the ESA MD5 manifest (`properties`, 2026-10-26). Official-mirror status in `k-pop/pensus/inputs/gaia_lsdb_hats/SOURCE.json`. |
| `gaia_source`, HDF5 | `/arc/projects/k-pop/gaia/GaiaSource/GaiaSource_*.hdf5` (1.09 TB, 3,386 files) | 1,811,709,771 | Column arrays at file root, sliced by `source_id`; `source_ids_x_file_names.h5` (29 GB) maps IDs to files. Used by `build_gaia_unlabelled_pool.py`. |
| XP continuous, HATS | `/arc/projects/hats/gaia/dr3_xp_continuous` (2.73 TB) | 219,197,643 | 55 + 55 coefficients, errors, correlations; same partitions as `gaia/dr3`, so the join is partition-local. |
| XP continuous, raw | `/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum/` (3.61 TB) | 219,197,643 | ESA csv.gz. |
| RVS mean spectra | `/arc/projects/k-pop/spectra/gaia/dr3/rvs_mean_spectrum/` (20 GB) | ≈ 1.0M *(not verified)* | Not used by MSA. |

Gaia DR3 published coverage: G for ≈ 1.806B sources, BP for ≈ 1.542B, RP for ≈ 1.555B, 5- or 6-parameter
astrometry for ≈ 1.468B, radial velocities for ≈ 33.8M, GSP-Phot for ≈ 471M, GSP-Spec for ≈ 5.6M.

### 2.2 Gaia-keyed survey matches for all Gaia sources (`adql_matches`)

`/arc/projects/k-pop/catalogues/adql_matches/` holds Gaia-archive (DPAC) best-neighbour joins keyed by
`source_id`. Row counts are from the FITS headers. They are not limited to XP stars: 2MASS alone has 469M rows,
against 219M XP stars.

| File | Rows | Columns |
|---|--:|---|
| `twomass.fits` | 469,001,402 | `source_id, tmass_oid, ph_qual, j_m, j_msigcom, h_m, h_msigcom, ks_m, ks_msigcom`. **Mis-keyed, do not use** (2026-11-24): `tmass_oid` holds DR3 `clean_tmass_psc_xsc_oid`, but the photometry is that of the DR1 `tmass_original_valid` row with that `tmass_oid`, i.e. another 2MASS source at a similar declination (right only for the first rows near the south pole). The Andrae file's J/H/Ks are these values. Correct key: DR3 `original_ext_source_id` = DR1 `designation` (`data/twomass_gaia.py`). |
| `pans_gaia_ids.fits` | 951,450,474 | `source_id, original_ext_source_id` (all of `gaiadr3.panstarrs1_best_neighbour`) |
| `panstarrsdr1_matches/` (89 GB), `ps1_curated.h5` (66 GB) | — | PS1 g, r, i, z, y mean PSF mags and errors, `obj_info_flag`, `quality_flag` (query in `data/example_adql_match.py`) |
| `sdss.fits` | 119,663,765 | `source_id, objid, flags, flags_u…z, u, g, r, i, z, err_u…err_z` (SDSS DR13 best neighbour) |
| `sdss_curated.fits` | 49,277,912 | cleaned ugriz with errors |
| `catwise.fits` | 678,990,741 | `id2, source_id, w1mpro, w1mpro_pm, w1sigmpro, w1sigmpro_pm, w1snr, w1snr_pm`, same for W2 |
| `smssdr4_curated.h5` (33 GB), `skymapperdr2_adql_match_files/` | *(not verified)* | SkyMapper u, v, g, r, i, z; the DR4 master CSVs in `k-pop/catalogues/skymapperdr4/` (726 parts) carry Gaia DR3 IDs |
| `tmass_curated.h5`, `tmass_curated_0622.h5` | *(not verified)* | cleaned 2MASS |

The curated HDF5 files were the inputs of the XP training set. Their row counts could not be read remotely.
`experiment-history.md` §5.4 records `matched_no_xp: 0` for the training inputs, so the XP cut happened when the
training set was built. The raw FITS tables above still cover non-XP Gaia stars.

### 2.3 XP-only training inputs

| Product | Path | Rows | Content |
|---|---|--:|---|
| Andrae 2023 SSL set | `/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5` (475 GB) | ≈ 218M in 135 shards (`sslset<portion>_part<chunk>`) | W1, W2, G, BP, RP, `bp_1..55`, `rp_1..55` (flux-scaled by `10**((8.5−G)/2.5)`), SkyMapper uvgriz, SDSS ugriz, PS1 grizy, 2MASS JHKs, PARALLAX, EBV (SFD), pmra, pmdec, RA, DEC and their errors. XP stars only. A newer `sslset-realmags-full-062425.h5` (same size) exists; MSA tables read the `052725` file. |
| `merged_gaia_lite_xp/` | `/arc/projects/k-pop/catalogues/merged_gaia_lite_xp/` (2.15 TB, 3,386 Parquet) | *(not verified)* | `source+xpcoeffs<N>.parquet`: Gaia lite columns plus XP, per GaiaSource range. |
| `aydan_data_09_23/` | `/arc/projects/k-pop/catalogues/aydan_data_09_23/` (2.35 TB, 4,583 files) | *(not verified)* | Older XP tables. |

### 2.4 Survey catalogues on `/arc/projects/hats`

Rows and MOC sky fraction are from each tree's `properties`. Sky fraction is the MOC at the tree's order and can
overstate the real footprint. Each mirror holds the release's native columns, with errors and flags; the column
lists were not read here.

| Survey (release) | Path | Rows | Sky | Bands | Astrometry |
|---|---|--:|--:|---|---|
| GALEX GUVcat AIS GR6/7 | `hats/galex/guvcat_ais_gr67` | 82,992,062 | 0.995 | FUV, NUV | none |
| SkyMapper DR4 | `hats/skymapper/dr4` | 724,329,344 | 0.870 | u, v, g, r, i, z | mean epoch, no PM |
| SDSS photoPrimary DR13 | `hats/sdss/photoprimary_dr13` | 469,050,917 | 0.698 | u, g, r, i, z | per-band MJD |
| Pan-STARRS1 DR2 OTMO | `hats/ps1/otmo` (2.99 TB) | 10,560,724,292 | ≈ 0.75 | g, r, i, z, y | mean epoch; includes single-detection objects, filter on detections |
| Legacy Surveys DR11 | `hats/legacysurvey/dr11` | 3,967,618,454 | 0.810 | g, r, i, z, forced W1–W4 | Gaia-seeded positions |
| DES Y6 Gold | `hats/des/y6_gold` | 691,483,608 | 0.189 | g, r, i, z, Y | none |
| DELVE DR3 Gold | `hats/delve/dr3_gold` | 2,588,267,795 | 0.613 | g, r, i, z | none |
| VST ATLAS DR5 | `hats/atlas/dr5` | 150,198,243 | 0.190 | u, g, r, i, z | none |
| KiDS DR5 | `hats/kids/dr5` | 138,812,117 | 0.053 | u, g, r, i (+ VIKING Z Y J H Ks) | none |
| UNIONS CFIS DR6 | `hats/unions/cfis_DR6_u`, `cfis_DR6_r` | 377,545,437 (u) | 0.360 | u, r | none |
| HSC PDR3 forced | `hats/hsc/pdr3/forced_{g,r,i,z,y}` + 4 NB | ≈ 0.73–0.78B per band | 0.04 | g, r, i, z, y, NB | saturates near G ≈ 18 |
| Euclid DR1 wide MER | `hats/euclid/dr1/mer_catalogue_wide_survey/…-hats` | 888,152,165 | 0.075 | VIS, Y, J, H | saturates bright |
| S-PLUS DR4 | `hats/splus/dr4` | 80,155,547 | 0.296 | 12 bands (5 broad, 7 narrow) | none |
| J-PLUS DR3 | `hats/jplus/dr3` | 16,419,372 | 0.370 | 12 bands | none |
| J-PAS e2DR | `hats/jpas/e2dr` | 4,804,561 | 0.167 | 56 narrow bands | none |
| Pristine DR2 | `hats/pristine/dr2` | 25,224,342 | 0.633 | CaHK | none |
| 2MASS PSC | `hats/2mass/psc` | 470,992,970 | 1.000 | J, H, Ks | epoch per scan |
| UKIRT UHS DR3 | `hats/uhs/dr3` | 514,336,270 | 0.394 | J, K | none |
| VISTA VHS DR7 | `hats/vhs/dr7` | 1,496,104,720 | 0.493 | Y, J, H, Ks | per-band MJD |
| UKIDSS DR11 LAS / GCS / DXS / UDS | `hats/ukidss/dr11_{las,gcs,dxs,uds}` | 77,685,267 / 64,835,919 / 2,992,865 / 296,026 | 0.185 / 0.071 / — / — | Y, J, H, K | none |
| VISTA VIKING DR5 | `hats/viking/dr5` | 82,063,585 | 0.073 | Z, Y, J, H, Ks | none |
| VISTA VVVX DR1 | `hats/vvvx/dr1` | 886,574,093 | 0.051 | Z, Y, J, H, Ks | none |
| VISTA VMC DR7 | `hats/vmc/dr7` | 64,359,516 | 0.017 | Y, J, Ks | none |
| CatWISE2020 | `hats/catwise/2020` (0.90 TB) | 1,890,715,640 | 1.000 | W1, W2 | PM |
| AllWISE | `hats/allwise/catalog` | 747,634,026 | 1.000 | W1–W4 | PM; `ph_qual`, `cc_flags`, `ext_flg` |
| eROSITA eRASS3 | `hats/erosita/erass3` | 1,975,540 | 0.729 | 0.2–2.3 keV | activity QA only |

Also present: `hats/2mass/xsc`, `hats/clauds`, `hats/video/dr6`, `hats/lsst/dp1`, `hats/hubble`, `hats/lotss`,
UNIONS GAaP and PGM products, and `hats/zensus/20261019` (a 1,169,651,709-row redshift catalogue, not a
photometric input). Unpromoted caches sit in `/arc/projects/k-pop/pensus/inputs/_cache/` (GALEX 369 GB, PS1 OTMO,
DELVE, DES DR2).

### 2.5 Pensus status

- Inputs: 48 alias directories in `/arc/projects/k-pop/pensus/inputs/` (`*_hats/SOURCE.json`) point at the
  promoted mirrors in `/arc/projects/hats`. The SDSS alias `sdss_dr16_hats` resolves to `photoprimary_dr13`:
  SDSS imaging has not changed since DR13. `allwise_hats` is a secondary source for W3/W4; CatWISE is the W1/W2 spine.
  `splus_dr5_hats` has no HATS path yet.
- Outputs: `/arc/projects/hats/pensus/views/` and `sidecars/` are empty; `k-pop/pensus/tiles/pinned-match-*` are
  empty test runs. **No Pensus crossmatch product exists on 23 Nov.**

### 2.6 Coverage: band × catalogue

"Gaia stars with a counterpart" is measured where a Gaia-keyed match exists (bold). Otherwise it is a rough
planning estimate (*est.*, ±50 %): Gaia sources inside the footprint times a detection fraction. "In MSA today"
means the band is in the current input schema (`configs/finetune-multisurvey-20261121-v10.yaml`); every such band
is used only for XP stars.

| Band(s) | Catalogue | Gaia stars with a counterpart | In MSA today |
|---|---|--:|---|
| G, BP, RP | Gaia DR3 | ≈ 1.806B (G), ≈ 1.54B (BP/RP) | yes |
| Parallax, PM | Gaia DR3 | ≈ 1.468B | parallax, pmra, pmdec |
| XP 55 + 55 | Gaia DR3 XP | **219,197,643** | yes |
| RVS spectrum | Gaia DR3 | ≈ 1.0M | no |
| FUV, NUV | GALEX GUVcat AIS | NUV ≈ 35M, FUV ≈ 5M (*est.*) | no |
| u, v | SkyMapper DR4 | ≈ 0.3B (*est.*) | yes |
| g, r, i, z | SkyMapper DR4 | ≈ 0.7B (*est.*) | yes |
| u, g, r, i, z | SDSS DR13 | **119,663,765** (best neighbour) | yes |
| u, r | UNIONS CFIS DR6 | ≈ 40M (*est.*) | no |
| u, g, r, i, z | VST ATLAS DR5 | ≈ 35M (*est.*) | no |
| u, g, r, i | KiDS DR5 | ≈ 8M (*est.*) | no |
| 12 bands | S-PLUS DR4 / J-PLUS DR3 | ≈ 20M / ≈ 10M (*est.*) | no |
| CaHK | Pristine DR2 | ≈ 25M (*est.*) | no |
| g, r, i, z, y | Pan-STARRS1 DR2 | **951,450,474** (best-neighbour IDs) | yes |
| g, r, i, z (+W1–W4 forced) | Legacy Surveys DR11 | **532,539,684** (Data Lab 1.5″ table, not on ARC) | no |
| g, r, i, z, Y | DES Y6 Gold | ≈ 40M (*est.*) | no |
| g, r, i, z | DELVE DR3 Gold | ≈ 250M (*est.*) | no |
| VIS, Y, J, H | Euclid DR1 wide | ≈ 15M (*est.*) | no |
| J, H, Ks | 2MASS PSC | **469,001,402** | yes |
| J, K | UHS DR3 | ≈ 450M (*est.*) | no |
| Y, J, H, Ks | VHS DR7 | ≈ 600M (*est.*) | no |
| Y, J, H, K | UKIDSS LAS + GCS | ≈ 45M (*est.*) | no |
| Z, Y, J, H, Ks | VIKING DR5 | ≈ 8M (*est.*) | no |
| Z, Y, J, H, Ks | VVVX DR1 | ≈ 300M (*est.*) | no |
| W1, W2 | CatWISE2020 | **678,990,741** | yes (W1, W2) |
| W3, W4 | AllWISE | W3 ≈ 60M, W4 ≈ 15M (*est.*) | no |
| E(B−V) | SFD map | all | yes (EBV input) |

### 2.7 Basis for a full-Gaia, full-wavelength table

**Now, without a new crossmatch (MSA-IN v1a).** Anchor on `hats/gaia/dr3` (1,811,709,771 rows) and join on
`source_id`:
- `hats/gaia/dr3_xp_continuous` (partition-local join);
- the `adql_matches` tables: 2MASS, PS1 (IDs plus photometry), SDSS, CatWISE;
- SkyMapper DR4 Gaia IDs from the DR4 master;
- SFD E(B−V) at the Gaia position.

This reproduces the current input schema for every Gaia star, with XP missing where absent. It is enough to
predict all Gaia stars and to train on the 2.13M+ labelled no-XP stars with today's architecture.

**Full wavelength (MSA-IN v2, weeks).** Run the xmatcher `ray-anchor` engine with `hats/gaia/dr3` as anchor and
the mirrors of section 2.4 (GALEX, SkyMapper, SDSS, PS1, LS DR11, DES, DELVE, ATLAS, 2MASS, UHS, VHS, UKIDSS,
VIKING, VVVX, CatWISE, AllWISE, S-PLUS, J-PLUS, Pristine, CFIS). Use target epoch J2016.0 and record `sep`,
`n_within`, `crowded` and quality for each survey ([`gaia-master-data.md`](../gaia-master-data.md)). This is the
Gaia-anchored projection of Pensus. Use the Pensus view instead once it exists.

**Pensus.** Adds non-Gaia objects (beyond Gaia) and the outer join. Keep the same column contract.

**DR4 (2 Dec 2026).** Re-anchor on DR4 `gaia_source` and DR4 XP. Data-set keys carry over through the DR3→DR4
neighbourhood table (section 3.8).

**Missing for full UV to mid-IR coverage:**
- GALEX deeper than AIS (MIS) and any UV in the Galactic plane;
- u and Hα in the plane (VPHAS+, IPHAS not staged);
- UKIDSS GPS (northern plane NIR) not staged;
- Spitzer GLIMPSE (plane mid-IR) not staged;
- S-PLUS DR5 (pointer only);
- Gaia RVS spectra and epoch photometry not integrated;
- no full-Gaia master table of any kind yet.

---

## 3. Data-set definitions (split spec `msa-split-v1`)

### 3.1 Earlier definitions found in the code

| Definition | Code | Rule | Problem |
|---|---|---|---|
| v5 table train/valid/test (in use by v5–v10, S1–S3, XP forward model) | `training/finetune_data.py:58` `split_row_positions`; `_split_data` | `train_test_split(arange(n_kept), test_size=0.2, random_state=42)`, then 0.33 of the rest → 80 % / 13.4 % / 6.6 %; v5: 4,697,789 / 786,880 / 387,568 | Depends on row order and the metal-poor filter. Any rebuild reshuffles. It is a seeded row permutation, not a `source_id` hash. Valid is used for epoch choice and for calibration (S1 Neyman critical values; S2 conformal on half of split 1). |
| v1-table test exclusion | `scripts/build_stopgap_finetune_table.py:83` `test_ids_of`, `--exclude-test-of` | The v1 table's test stars are dropped from later tables | 129,927 stars. Common test for stopgap to v10. Not reproducible from `source_id`. |
| Frozen [Fe/H] core | `scripts/core_eval_rows.py`; `runs/rescore_20261119/rows.npy`, `rows_split.npz` | v5 rows with a record from the `fe_h_core` prefixes, with their v5 split | 39,073 rows; 7,740 held out. Bound to the v5 row order. |
| October roles (lost runs) | `scripts/prepare_expanded_six_source.py:28` `role_buckets`; `extend_full_raw_source.py:44` | splitmix mix of `source_id`, no salt, mod 100: < 75 fit, < 85 tune, < 92 cal, else devtest; benchmarks and protected globular members → external | Stable, but unsalted. Its devtest was inspected during development. |
| Group-aware split | `training/multisurvey_splits.py:14` `build_group_split` | SHA-256 of `[salt, component]` over union-find components of declared groups; fit/tune/cal/test | The hash is over the member list, so a new member can move a whole group. |
| Catalogue pipeline split | `pipeline/splits.py:22` `four_way_split` | Seeded permutation of unique `source_id`: train/select/calibrate/test 70/10/10/10 | Not stable when the ID set changes. |
| S3 split | `scripts/s3_distance_extinction.py:138` `_hash_unit`, `s3_split` | Keep the table split; hash new rows (murmur3 fmix64, 0.8 / 0.934) | Two different rules in one data set. |
| No-XP REC pool (lost) | `scripts/build_gaia_unlabelled_pool.py:44` | splitmix64(`source_id`) < rate × 2⁶⁴, `has_xp_continuous = False`, non-FIT stars excluded | The right idea; the product was lost. |
| Consistency set | `scripts/build_consistency_table.py` | El-Badry+2021 pairs and HR23 members, inputs only | 107,026 of the 1,209,544 predicted stars are in the training table, so their labels were seen. |

### 3.2 Principles

1. Membership is a pure function of a stable key. It never depends on file row order, on the label version or on
   the input release.
2. The four-way partition is defined for **every** object in the universe (Gaia DR3, later DR4 and Pensus),
   labelled or not. A star that receives its first label later (DR4 GSP-Spec, 4MOST, a new survey) is already in
   its partition, and reconstruction never saw a held-out star.
3. Co-natal groups are not split: clusters and wide binaries move as units.
4. A small set of benchmarks is reserved and never enters fitting, selection or calibration.
5. Held-out membership is append-only. A minor version can move objects from FIT into held-out roles and never
   the other way. Changing the salt or the fractions is a major version and retires the TEST set.
6. DEV and CAL are separate. Early stopping and model selection never use CAL; calibration never uses DEV.

### 3.3 Key and hash

**Key.** In order of precedence:
1. Gaia DR3 `source_id` (EDR3 IDs are identical).
2. For a Gaia DR4 source without a DR3 counterpart: the string `gdr4:<source_id>`.
3. For a Pensus object without Gaia: `pensus:<survey>:<native_id>`, using the highest-precedence survey in the
   object (`datasets.yaml: keys.pensus_precedence`). The Pensus object ID itself is not used, because it may
   change between Pensus releases.
4. For a labelled master star without Gaia: `lbl:<survey>:<native_id>`, the lexicographically smallest over the
   star's records.

**Hash.** Integer keys are used as unsigned 64-bit values. String keys map to
`int.from_bytes(sha256(key)[:8], "little") | 2**63`; the top bit is set, and Gaia IDs are positive, so the two
cannot collide.

```python
SALT64 = int.from_bytes(hashlib.sha256(b"msa-split-v1").digest()[:8], "little")  # 0xb6f6e5353645106b
def unit(k64):                       # numpy uint64 in, float64 in [0, 1) out
    z = (k64 ^ SALT64) + 0x9E3779B97F4A7C15                      # wrapping uint64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z ^= z >> 31
    return (z >> 11) * 2.0**-53
```

The function is vectorised splitmix64, as in `build_gaia_unlabelled_pool.py`, with a salt. Test vectors:

| Key | u | Partition |
|---|---|---|
| 4185355646116022528 | 0.662500593453 | FIT |
| 4398046511104 | 0.798229566688 | DEV |
| 4295806720 | 0.852028592613 | CAL |
| 4295806753 | 0.882257348434 | TEST |
| `grp:hr23:Melotte_22` | 0.440677168076 | FIT |

On 2×10⁷ synthetic Gaia-like IDs the realised fractions are 0.7501 / 0.0500 / 0.0799 / 0.1199.

### 3.4 Partitions and roles

**Partition** `P(key)`: FIT if u < 0.75, DEV if u < 0.80, CAL if u < 0.88, TEST otherwise.
**Group override:** a member of a co-natal group takes `P(group key)` (section 3.5).
**Reserved:** reserved QA objects get no partition role (section 3.5).

| Set | Definition | Used for | Never used for |
|---|---|---|---|
| **FIT** | labelled ∩ P = FIT ∖ reserved | supervised loss; label-model fitting (survey maps, common-scale maps, age consensus, intrinsic colours, IRFM relation); scalers; co-natal training terms on FIT groups | reporting held-out metrics |
| **DEV** | labelled ∩ P = DEV ∖ reserved | early stopping, checkpoint and hyperparameter choice, ablation ranking | calibration; final reporting |
| **CAL** | labelled ∩ P = CAL ∖ reserved | conformal and recalibration (CQR, Neyman critical values, coverage maps) | training, model selection |
| **TEST** | labelled ∩ P = TEST ∖ reserved | final, pre-registered metrics only | any decision. A look that changed a decision is logged and marks TEST as inspected for that model generation. |
| **REC** | all universe objects with P = FIT ∖ reserved (inputs only) | self-supervised reconstruction and pretraining; masked-modality training | — |
| **REC-DEV** | objects with P = DEV (inputs only) | reconstruction early stopping | training |
| **PRED** | the whole universe: Gaia DR3 1,811,709,771; then DR4; then Pensus | inference and the catalogue | — |
| **QA-\*** | section 3.6 | evaluation | as their partition allows |

Notes:
- **Labelled** means at least one usable record for the target under the label policy of the build. Labelled is
  per target (spectroscopic, age, E(B−V), parallax). The partition does not depend on it.
- REC excludes DEV, CAL and TEST objects, so TEST stars are input-fresh as well as label-fresh for any encoder
  pretrained on REC v1. The July 2026 checkpoint 26 was pretrained on the whole Andrae XP set. Models started
  from it are label-fresh on TEST but not input-fresh on XP stars. Report this with every result until a REC-v1
  encoder exists.
- **FIT folds** for cross-fitting small heads or maps: `fold = floor(u / 0.15)` (0–4) inside FIT.
- **Nested REC samples** use an independent salt: `unit(key, salt = "msa-rec-v1") < rate`, so a 1/8 sample is a
  subset of the 1/4 sample.
- A transductive release model (REC over all objects) may be trained after all decisions are frozen. Its TEST
  numbers must then be labelled transductive.

### 3.5 Groups and reserved objects

**Co-natal groups** (members inherit the group partition):

| Group | Catalogue on ARC | Membership rule | Group key | Size |
|---|---|---|---|---|
| Open clusters | Hunt & Reffert 2023, `/arc/projects/k-pop/catalogues/OpenClusters/members.fits` | p ≥ 0.7, as in the age audit | `grp:hr23:<Name>` | 7,167 clusters; 574,116 members at p > 0.5 |
| Wide binaries | El-Badry, Rix & Heintz 2021, `analysis/age_labels_20261122/wide_pairs_eb21.parquet` | `R_chance_align` < 0.1 | `grp:eb21:<min(source_id1, source_id2)>` | 1,256,400 pairs |

Precedence when a star is in several groups: cluster before binary; among clusters, the highest membership
probability, then the smaller key. Adding members to a group never changes its partition, because the key names
the group, not its members. Hunt & Reffert 2024 (cleaned bound-cluster classification) is not on ARC; when added,
it changes only membership, not keys.

**Reserved objects** (never FIT, DEV or CAL; excluded from REC):

| Reserved set | Source | Size | Reason |
|---|---|---|---|
| Gaia FGK benchmarks v3 | `k-pop/catalogues/gaia_fgk_benchmarks.parquet` | 192 stars | Independent Teff/log g scale. |
| Globular-cluster members | Vasiliev & Baumgardt 2021, `k-pop/catalogues/gc_gaia/` (+ `cluster_members_extra.parquet`, 9 GCs, 203,450 stars) | ≈ 170 GCs, members at p ≥ 0.5 (count at freeze) | Only old metal-poor co-natal anchors. Keeps the October `protected_gc` rule. |
| Age-benchmark clusters | `analysis/age_labels_20261122/cluster_priors.csv` | 35 clusters (LDB, eclipsing-binary, seismic, GC); members from HR23 at p ≥ 0.7 | Age validation. |
| Kepler LEGACY | `k-pop/catalogues/kepler-legacy-dwarfs.fits` | 66 stars | Seismic dwarf ages and log g. |
| WD+MS pairs | `analysis/age_labels_20261122/wdms_pairs.parquet`, `wd_ages_published.parquet` | 16,703 pairs (1,275 MS companions with label records) | Independent age physics. |

Reservation costs few labels except for globular clusters, where APOGEE observed many members (a few thousand
labelled giants between [Fe/H] −1 and −2.3). Releasing GCs to the group rule is the one open option; it needs a
major version.

### 3.6 QA sets

Each QA family is evaluated **per partition**. QA-dev means members with P = DEV (decisions allowed); QA-cal means
P = CAL (calibration checks); QA-test means P = TEST (final reporting). Reserved sets are report-only.

| QA set | Definition | Partition rule | Current size |
|---|---|---|---|
| QA-core ([Fe/H] core v1) | Stars with ≥ 1 `fit_ok`, non-grid-edge [Fe/H] record from the `fe_h_core` surveys (`finetune-multisurvey-20261121-v10.yaml: multisurvey.fe_h_core`), master `20261113_v1`; ID list frozen | per star | 39,073 v5 rows (7,740 held out); master-wide count at freeze |
| QA-mp (metal-poor references) | QA-core stars with [Fe/H] < −2, and metal-poor survey stars (PIGS, S5, LAMOST VMP/CaT, Pristine, SkyMapper EMP) | per star | high-res < −2: 3,101; < −3: 706; < −4: 37 |
| QA-seis (seismic log g and age) | APOKASC-3, APO-K2, APOKASC dwarfs | per star; FIT members act as anchors | 13,871 / 4,660 / 413 reference records |
| QA-irfm (Teff) | IRFM-scale Teff | per star; consistency check only | 898,839 |
| QA-gbs | Gaia FGK benchmarks v3 | reserved | 192 (63 with XP) |
| QA-oc (open clusters) | HR23 members, p ≥ 0.7 | per cluster | 7,167 clusters |
| QA-gc (globular clusters) | VB21 members | reserved | ≈ 170 GCs; 118 with ≥ 5 predicted v7 members |
| QA-wb (wide binaries) | EB21 pairs, `R_chance_align` < 0.1 | per pair | 1,256,400 pairs; 280,045 with both stars predicted in v7 |
| QA-age (age benchmarks) | `cluster_priors.csv` clusters, Kepler LEGACY, WD+MS pairs | reserved | 35 clusters, 66 stars, 16,703 pairs |
| QA-dist (distance) | Hipparcos parallaxes (independent instrument); HR23 cluster parallaxes | per star / per cluster | 99,525 Hipparcos IDs (earlier plan); 95 clusters in S3 |
| QA-ext (E(B−V)) | SFD at \|b\| > 60° (not independent); cluster E(B−V) | per star / per cluster | as in S3 |
| QA-loso (survey transfer) | leave-one-survey-out within TEST (`pipeline/splits.py: leave_one_survey_out`) | TEST only | per survey |
| QA-noxp | labelled TEST stars without XP, and TEST stars with XP hidden | TEST only | section 3.7 |

Strata pre-registered for every QA set (from [qa.md](qa.md) §5.2): G bins, [Fe/H] bins, Teff/log g regime, input
class (section 3.7), and labelled vs never-labelled companions for co-natal checks.

### 3.7 Labelled stars without XP, and input classes

- Labelled stars without XP are ordinary members of FIT, DEV, CAL and TEST by their key. They enter as
  photometry-only rows: the XP group is masked as missing, never filled.
- Minimum inputs for a supervised row: Gaia G plus at least one colour (BP−RP, or one external band). Below that
  the star stays in PRED, but its prediction is flagged.
- Input class, recorded per row and used as a stratum:
  - **X**: XP present;
  - **P**: no XP, at least two colours;
  - **G**: G only, or G plus astrometry.
- Stars with XP also get whole-XP dropout in training, so the XP-off path is learned on both populations. Metrics
  are always reported for X, P and X-with-XP-hidden separately. The no-XP population is fainter and is not the
  same distribution as hidden-XP stars.
- Sizes: 2,128,095 labelled no-XP stars inside the v5 8M cap. Over the whole master about 5M *(est.)*, using the
  v5 hit rate of 73 %.

### 3.8 Carry-over to DR4 and Pensus

- **Gaia DR4.** Map each DR4 source to DR3 through the DR3→DR4 neighbourhood table that DPAC publishes with DR4,
  using a unique best match with a small separation and \|ΔG\| cut, recorded as `key_route`.
  - A DR3 source split into several DR4 sources gives each child the DR3 key.
  - Several DR3 sources merged into one DR4 source take the most held-out of their partitions
    (TEST > CAL > DEV > FIT).
  - New DR4 sources use `gdr4:<source_id>`.
- **Pensus.** Gaia-matched objects use the Gaia key. Others use `pensus:<survey>:<native_id>` by the fixed survey
  precedence. Group and reserved lists are re-resolved through the same keys.
- **Label versions** (master v2, DR4 GSP-Spec, new surveys) change which objects are labelled, never their
  partition.

### 3.9 Sizes

Expected target sizes are the fractions applied to the universe. Exact counts come from the freeze.

| Set | Current (v5 lineage, 23 Nov) | Target, Gaia DR3 universe |
|---|--:|--:|
| FIT | 4,697,789 (v5 train) | ≈ 14.40M labelled (of 19,205,652 Gaia-identified master stars) |
| DEV | 786,880 (v5 valid, also used for calibration) | ≈ 0.96M |
| CAL | none dedicated | ≈ 1.54M |
| TEST | 387,568 (v5 test) + 129,927 (v1-table test) | ≈ 2.30M |
| REC | 6,362,706 XP stars (v5 `unlabelled.npz`, ≤ 5M used); July pretraining ≈ 218M XP rows | 1,358,782,328 (≈ 164.4M with XP) |
| REC-DEV | — | 90,585,489 |
| PRED | 1.21M (consistency) + benchmarks | 1,811,709,771 |
| High-res [Fe/H] < −2 per FIT / DEV / CAL / TEST | — | ≈ 2,326 / 155 / 248 / 372 |
| High-res [Fe/H] < −3 per FIT / DEV / CAL / TEST | — | ≈ 530 / 35 / 56 / 85 |

Reserved sets and group overrides move a few hundred thousand stars at most. DR4 and Pensus enlarge PRED and REC;
the fractions stay fixed.

### 3.10 Versioning, freeze and migration

- **Spec version:** `msa-split-v1` = salt, fractions, key routing, group catalogues with versions and checksums,
  reserved lists with checksums. Minor versions (v1.1, …) may add groups or reserved objects, and only move objects
  FIT → held-out. Major versions change the salt or fractions and retire TEST.
- **Freeze artefact** (proposed location `/arc/projects/astroai/msa/splits/msa-split-v1/`, not created):
  - `manifest.json` with the spec, input sha256s and counts per partition × input class × survey;
  - `overrides.parquet` (`key, group_key, partition, reserved_reason`; a few million rows);
  - `qa_sets.parquet` (`key, qa_set, group_key`).
  The base partition is recomputed from the key; it is never stored per row for 1.8B objects.
- **Code:** one module should replace `split_row_positions`, `four_way_split`, `role_buckets`, `s3_split` and
  `in_sample`. Every table build writes `partition`, `role`, `input_class` and `split_spec` columns and refuses a
  mismatched spec. This is not implemented yet.
- **Migration from the v5 lineage:**
  - keep three legacy evaluation sets for continuity: L-v1test (129,927), L-v5test (387,568) and L-core
    (7,740 held-out core stars);
  - compare a new model with v5–v10 only on TEST ∖ (v5 train ∪ v5 valid), which those models have not seen;
  - the reddening intrinsic colours and the S1–S3 heads were fitted on v5 train rows and must be refitted on FIT.

---

## 4. Open items

1. Freeze `msa-split-v1`: resolve HR23, EB21 and VB21 membership to keys, then write `overrides.parquet` and the
   counts (CPU job).
2. Build MSA-IN v1a (section 2.7) for all 1.81B Gaia sources, and add `has_xp` and input class to the master.
3. Update `docs/master-spectroscopic-labels.md` to `…20261113_v1`, and remap the dead `catalogues.json` paths.
4. Stage the missing plane surveys (VPHAS+, IPHAS, UKIDSS GPS, GLIMPSE) for Pensus, or record them as out of scope.
5. Pretrain a REC-v1 encoder so that TEST becomes input-fresh.
