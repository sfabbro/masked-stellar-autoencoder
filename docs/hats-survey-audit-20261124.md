# Audit of the community HATS trees in `/arc/projects/hats` (2026-11-24)

For the owner preparing xmatcher and Pensus. Every HATS catalogue under
`/arc/projects/hats` was audited read-only on 2026-11-24: 74 catalogue roots
(54 of them public), 5 HATS collections and 1 symbolic link, about 24.07 TB of the
project's 27.7 TB quota. Nothing was written under `/arc/projects/hats`. The
machine-readable results are in
`/arc/projects/astroai/msa/runs/hats_survey_audit_20261124/` (`vos:astroai/msa/runs/hats_survey_audit_20261124`).
[Appendix A](#appendix-a-method) describes the method and
[Appendix B](#appendix-b-script-sessions-and-outputs) lists the script, the sessions
and the output files.

Each finding has a source:

- **verified (all rows)**: from the footer statistics of every partition, or from a
  full scan of every row;
- **verified (sample)**: from the sampled partitions; the numbers given are per
  partition;
- **metadata**: from `properties`, `partition_info.csv`, `PENSUS_*.json` or the
  schema;
- **unverified**: the cause is inferred and not checked.

"Gaia ≥2" is the share of Gaia DR3 sources (J2016.0 positions) in a partition that
have two or more rows of the tree within 1.5″, xmatcher's default radius. "m_ge2" is
the same share among the Gaia sources that have at least one match.

**Affects** column: **X** = xmatcher (reading, pins, matching), **P** = Pensus users
(the community tree as published), **M** = MSA training (photometry inputs).

## 1. Summary

### 1.1 Top 10 problems by impact

1. **HSC PDR3 forced, broad bands (`hsc/pdr3/forced_{g,r,i,z,y}`): blocker for
   Gaia-anchored matching.** There is no primary flag (`detect_isPrimary`), no
   patch/tract-inner flag and no pixel flag in any of the five trees. Deblend parents,
   their children and patch/tract-overlap copies are all present. In the sampled
   partitions 30–76 % of rows have another row within 0.5″ and 9–15 % sit at the same
   position (<1 mas). In full-depth fields 87–95 % of the matched Gaia stars have
   two or more rows within 1.5″. The best match is a deblend parent for 26–44 % of the
   Gaia stars. `instFlux` is in coadd counts with a zero point of about 27, not nJy
   (verified). Affects X, P, M.
2. **ATLAS DR5 (`atlas/dr5`): blocker, corrupt partitions.** 10 of the 385 partition
   files cannot be read. Two are 0 bytes (`3/605`, `3/729`) and eight fail with
   "Deserializing page header failed" (`2/122`, `3/273`, `3/274`, `3/290`, `3/565`,
   `3/573`, `3/683`, `3/684`). That loses 6,613,768 rows (4.4 %), and
   `hats_nrows=150,198,243` overstates the 143,584,475 rows that can be read. Any full
   read of the tree fails. The five epoch columns `UMJD`…`ZMJD` are null in every row.
   Affects X, P, M.
3. **SDSS (`sdss/photoprimary_dr13`): `id` is not the SDSS objID.** It equals
   `THING_ID` in all 469,050,917 rows, and it is not unique (3 repeated values: 4710,
   5418 and 7849). Decoding it as a packed objID matches `RUN`, `FIELD` and `ID` in at
   most 0.003 % of the sampled rows. The tree has both `ID` (object number within a
   field, 1–9304) and `id`, two names that differ only in case. xmatcher's pin assumes
   that `id` is the objID. The key `RUN, CAMCOL, FIELD, ID` is unique across all rows
   (`RERUN` is 301 in every sampled row). Affects X, M.
4. **PS1 OTMO (`ps1/otmo`): single-detection objects and ambiguous Gaia matches.**
   7,160,529,807 of the 10,560,724,292 rows (67.8 %) have `nDetections = 1`, and
   805,690,947 (7.6 %) have 0 (verified, all rows). Single detections are only
   0.04–3 % of the best Gaia matches. Gaia ≥2 is 19–36 % in the sampled order-5/6 fields, with the second row about 1.2″ away. Many
   columns use −999 for missing values (`raMeanErr`, `epochMean`, `*MeanPSFMag`,
   `raStack`). Eight rows have RA just outside [0, 360) (verified, all rows). The tree
   has not been rebuilt: session `ykehpdqk` *pensus-ps1-import-arm* was still running.
   Affects X, M.
5. **Legacy Surveys DR11 (`legacysurvey/dr11`): rows in the north/south overlap
   appear twice.** 485 of 9,222 partitions hold both `RELEASE` 11010 (S) and 11011 (N).
   In 400 of them, 21.1 % of rows (37,324,926 of 176,991,268) have a twin with the
   other RELEASE within 0.1″, and 37.5 % within 0.5″ (verified, follow-up). Where the
   twins concentrate, 72 % of Gaia sources have two or more rows within 1.5″, and the
   second row has the other RELEASE in a median 97 % of cases. Extrapolated, about 22 M
   objects are stored twice. Partitions outside the overlap hold a single RELEASE, and
   there the Gaia ≥2 share of 1–13 % comes from real neighbours about 1″ away.
   Affects X, P, M.
6. **Euclid DR1 MER wide and Euclid Q1: ambiguous Gaia matches built into the MER
   catalogue.** m_ge2 is 16–28 % (wide 19–28 %, Q1 16–19 %, deep 9–26 %). The second
   row is in the same `tileid` (99.6–100 %), so these are not tile-overlap copies:
   39–48 % of the pairs differ in `spurious_flag` and 30–48 % in `vis_det`. Tree-wide,
   22.4 % of the wide rows have `vis_det=0`, 6.3 % have `spurious_flag=1` and 18.0 %
   have `det_quality_flag≠0`; 62.1 % pass all three cuts. The wide tree is the IRSA
   collection layout, with `Npix=N/` directory partitions and properties that lack
   `hats_version`, `hats_creation_date` and `hats_col_healpix`. Affects X, M.
7. **S-PLUS DR4 (`splus/dr4`): overlapping fields are not de-duplicated.** In the
   samples 0.8–11 % of rows have a twin within 0.5″, and every one of those pairs comes
   from two different `field`s (for example `SPLUS-n20s34`/`SPLUS-n20s35`,
   `MC0021`/`MC0029`). Tree-wide, 9.6 % of the 80,155,547 rows (7,700,226) have a twin
   from another field within 0.5″, and none from the same field (verified, all rows).
   Affects X, P, M.
8. **J-PLUS DR3 and J-PAS e2DR: numbers stored as strings, and the array order is not
   wavelength.** Every photometry and flag column is a space-separated text array
   (`"19.0042 19.7985 … 99.0000"`, `"NaN NaN …"`), and `TILE_ID`, `NUMBER` and
   `CLASS_STAR` are text too. The colour slopes against Gaia BP−RP show the 12 J-PLUS
   elements in CEFCA filter-ID order (rSDSS, gSDSS, iSDSS, zSDSS, uJAVA, J0378, J0395,
   J0410, J0430, J0515, J0660, J0861), not in wavelength order. In the J-PAS e2DR
   samples most of the 57 elements are `NaN` for most rows. Affects X, P, M.
9. **Rebuild and swap state.** DES Y6 Gold and UHS DR3 were rebuilt with RA wrapped
   into [0, 360): no out-of-range rows remain (verified, all rows), and the row counts
   per partition equal those of the backups. The swap changed DES's `obs_collection`
   to `y6_gold_rawrap`, so xmatcher's pin (`des_y6_gold`) now fails verification.
   DELVE DR3 still has 8 rows at RA = 360.0 (session `qlhgso49`
   *pensus-des-delve-import* was still running). The backups `des/y6_gold_prera_backup`
   (1.39 TB) and `uhs/dr3_prera_backup` (27 GB) still sit in the public tree with their
   RA = 360 rows. Affects X, P.
10. **No epoch, no quality flags, non-null sentinels in the VISTA/UKIRT trees** (UHS,
    VHS, VVVX, VMC, VIKING, VIDEO, UKIDSS). Every magnitude and error column uses
    −999999500 for missing values (verified, all rows). There is no epoch, no position
    error and no `ppErrBits`/`errBits`. VHS has J, H and Ks only (no Y or Z). The
    seamless primary selection is correct (`priorsec` is 0 or equal to `framesetid`,
    and no secondaries in any sample). Affects M (and X for epoch propagation).

The appendix also covers a problem outside the tree: the k-pop
`adql_matches/twomass.fits` pairs Gaia sources with the wrong 2MASS rows
([Appendix C](#appendix-c-related-k-pop-adql_matches-outside-the-hats-tree)).

### 1.2 Tree × severity

**Blocker**: breaks reading, breaks a pin, or gives wrong data or wrong pairs if the
tree is used as published. **Bias**: the tree works but biases matching or training
unless it is handled. **Cosmetic**: metadata, naming or provenance only.

| Tree | Blocker | Bias | Cosmetic | Affects |
|---|---|---|---|---|
| `gaia/dr3` | — | — | no sync record; description claims margins/index that are absent | X P |
| `gaia/dr3_xp_continuous` | no `_common_metadata`/`_metadata`/`properties` (readers that need them) | — | `_done/` markers left; no `hats_version` | X P M |
| `2mass/psc` | — | — | flags as text; `decl` | P |
| `2mass/xsc` | — | 4 columns only (no photometry) | 12 order-0 partitions; temporary host | P |
| `allwise/catalog` | — | motions are not astrometry (ratio to Gaia 1.55 in one sample) | no sync record | X M |
| `atlas/dr5` | 10 unreadable partitions (6.6 M rows) | epoch columns all null | `obs_collection=dr5`; temporary host | X P M |
| `catwise/2020` | — | — | `obs_collection=2020` | — |
| `clauds` | — | — | data files readable only by group `hats` while the flat copy is public; no provenance | P |
| `delve/dr3_gold` | — | 8 rows at RA = 360.0; −9999/99 sentinels | hats v0.1, no `hats_col_healpix`; rebuild pending | X P |
| `des/y6_gold` | xmatcher pin (`obs_collection`) broken by the swap | −99/99 sentinels | `obs_collection=y6_gold_rawrap`; null `Norder/Dir/Npix` columns | X P |
| `des/y6_gold_prera_backup` | — | 2 rows at RA = 360.0 | 1.39 TB duplicate in the public tree | P |
| `erosita/erass3` | — | — | — | — |
| `euclid/dr1/*` (IRSA) | — | m_ge2 9–28 % (spurious/NIR-only rows) | collection layout, `Npix=N/` dirs, missing keys; many all-null columns | X P M |
| `euclid/q1` | — | m_ge2 16–19 % | hats v0.1 | X |
| `galex/guvcat_ais_gr67` | — | `*_mag_auto`/`*_mag_aper_*` lack the zero point (offset 20.08/18.82); −999/−99 | 24 rows fewer than GUVcat (unverified) | X M |
| `hsc/pdr3/forced_{g,r,i,z,y}` | no primary/inner flags; m_ge2 up to 95 % | flux in counts (ZP≈27); no pixel flags; 20–46 partitions <100 rows per band | five separate trees, built Jun 2026 without a provenance record | X P M |
| `hsc/pdr3/forced_nb*` | — | — | labelled temporary ("Temp HATS only") | P |
| `hubble/hsc_v4` | — | 22–94 % of rows have a neighbour <0.5″; m_ge2 38–100 % | no provenance record | X |
| `jpas/e2dr`, `jplus/dr3` | numbers as strings | array order = filter ID; 99 sentinels | order-0 partition (J-PAS); temporary host | X P M |
| `kids/dr5` | — | — | text ids stored as binary | P |
| `legacysurvey/dr11` | N/S twins in the overlap | ambiguous Gaia matches | temporary host | X P M |
| `lotss/dr3` | — | — | duplicate `Source_Name` in samples; 3 order-0 partitions | P |
| `pristine/dr2` | — | 58,177 duplicated `source_id` pairs (0.23 %) | — | X M |
| `ps1/otmo` | — | ~75 % single detections; m_ge2 19–36 %; −999 sentinels; 8 rows out of range | hats v0.1 (hipscat 0.2.5); no sync record | X M |
| `sdss/photoprimary_dr13` | `id` = THING_ID, not objID; `id`/`ID` case clash | −9999 sentinels; no epoch | directory says DR13, properties DR16 | X M |
| `skymapper/dr4` | — | — | 2 order-0 partitions; no sync record | X |
| `splus/dr4` | — | field-overlap twins (9.6 % of rows) | order-0 partition; text `id` | X P M |
| `uhs/dr3` (+ backup) | — | sentinel −999999500; no epoch/flags | `obs_collection=dr3_cephwrap`; backup with RA = 360 kept | X P M |
| `ukidss/dr11_{las,gcs,dxs,uds}` | — | sentinel; no epoch/flags | DXS/UDS provenance says `gcsSource`; DXS order-0 partitions | P M |
| `vhs/dr7`, `vvvx/dr1`, `vmc/dr7`, `viking/dr5`, `video/dr6` | — | sentinel; no epoch/errors/flags; VHS no Y/Z | `obs_collection` collisions (`dr5`, `dr7`, `dr1`) | X M |
| `lsst/*` (private) | — | — | DP1 light-curve collections only; DP2 is a link outside the tree | X |
| `unions/*` (private) | `gaap_sp_ugriz_photoz_ext` is an unfinished build | — | second Euclid DR1 wide copy (1.11 B rows); `.claude/` settings file | P |
| `zensus/*` (private) | — | — | derived product in the community tree | P |
| tree level | — | — | `.local/`, `.herd-gidsync.state`, stale `registry/` snapshot; 87 % of quota | P |

## 2. Per-tree findings

Unless a tree's section says otherwise, its HATS structure is clean. The general
checks and their exceptions:

- `partition_info` matches the files present: no missing, extra or nested
  partitions, and every `Dir` value is right.
- Every footer can be read, except ATLAS.
- Each tree has one schema, except `unions/gaap_sp_ugriz_photoz_ext`.
- The footer row sums equal `hats_nrows`, except ATLAS, the unfinished `gaap_sp` build
  and the index trees (which have no HEALPix partitions).
- `_metadata` agrees with the footers, except ATLAS, `gaap_sp` and the XP tree, which
  has no `_metadata`.
- No coordinate is NaN in any sample or full scan.
- In all 347 sampled partitions every row lies in its partition's pixel and every
  `_healpix_29` value agrees with RA/Dec (verified, sample).

### 2.1 `gaia/dr3` (Gaia DR3 gaia_source, anchor)

- 1,811,709,771 rows, the ESA count; 2016 partitions, orders 2–6; 153 columns; HATS
  v1.0 (hats-import 0.11.2.dev1, built 2026-10-26). `hats_coordinate_epoch=J2016.0` is
  declared, and this is the only tree that declares an epoch. Verified (all rows):
  structure, row counts, coordinate ranges.
- Cosmetic: there is no `PENSUS_SYNC.json`. The only provenance is xmatcher's yaml
  comment ("byte copy of the LSDB tree"). `obs_description` promises "10 arcsec and
  300 arcsec margins and a source_id index", but neither exists in
  `/arc/projects/hats/gaia`. xmatcher's docs record 1,812,731,847 rows for the LSDB
  tree on 2026-10-22, so the local copy is not the current LSDB build (metadata).
- Fix: add a sync record (source URL and version). Either copy the margins and index,
  or remove the sentence from the description. **Affects** X (provenance), P.

### 2.2 `gaia/dr3_xp_continuous` (Pensus XP layout)

- 219,197,643 rows, the ESA XP count; the same 2016 partitions as `gaia/dr3`; `Npix=N/`
  directory partitions. In the sampled partition `2/46`, all 176,621 `source_id`s were
  found in the Gaia partition at a separation of exactly 0 (verified, sample).
  Coefficients are `list<double>[55]` with errors `[55]` and correlations `[1485]`.
- Possible blocker for generic readers (unverified with lsdb): there is no
  `dataset/_common_metadata`, no `dataset/_metadata` and no `properties` (only
  `hats.properties`). `hats_version`, `hats_col_healpix`, `hats_max_rows` and
  `moc_sky_fraction` are missing. A `_done/` directory with 3386 build markers is left
  at the root. The files carry dictionary-encoded `Norder/Dir/Npix` columns.
- Fix: write `_common_metadata` and `_metadata` (for example with
  `hats.io.write_parquet_metadata`), a `properties` copy and the missing keys, then
  remove `_done/`. **Affects** X, P, M (the XP sidecar of the v11 encoder).

### 2.3 `2mass/psc`

- 470,992,970 rows (the PSC count). Synced from `data.lsdb.io/hats/two_mass/two_mass`
  (`PENSUS_SYNC.json`, version `hats-http-6d401f246861`); hats 0.7.1 build of
  2025-11-24. The declination column is `decl` (declared in the properties). Close
  pairs (<0.5″, 0.02–0.6 % of rows) are distinct PSC sources with different
  designations, not duplicates. Gaia ≥2 ≤0.4 % (verified, sample).
- Cosmetic: the flags `rd_flg`, `bl_flg`, `cc_flg`, `ndet` and `a` are text, as in the
  original release. 99/999/9999 sentinels in `err_ang`, `*_snr`, `dist_edge_*`,
  `phi_opt` (sample). **Affects** P.

### 2.4 `2mass/xsc`

- 1,647,599 rows in 12 partitions, all at order 0. Only 4 columns (`designation`,
  `ra`, `dec`, `_healpix_29`): no photometry, no shape and no flags (metadata). The
  `PENSUS_BUILD.json` says `temporary_host: true`.
- Fix: rebuild with the XSC photometry columns, using `hats_max_rows` for a
  1.6 M-row tree (order 2–3). **Affects** P (unusable for SEDs).

### 2.5 `allwise/catalog`

- 747,634,026 rows, 299 columns, a byte copy of the LSDB tree (no sync record). Clean.
- Bias: `pmra/pmdec` are AllWISE apparent motions (mas/yr) with ±99/±999 sentinels.
  In one sample their median ratio to Gaia's proper motion was 1.55, which confirms
  xmatcher's choice not to wire them. 99.99 in `w?mlq` and ±99 in `rho12/23/34`.
  **Affects** X (already handled), M.

### 2.6 `atlas/dr5` (VST ATLAS)

- **Blocker (verified, all files).** 10 of 385 partition files cannot be read:
  - 0-byte files: `dataset/Norder=3/Dir=0/Npix=605.parquet` and
    `dataset/Norder=3/Dir=0/Npix=729.parquet`;
  - data pages fail with "Couldn't deserialize thrift … Deserializing page header
    failed": partitions `2/122`, `3/273`, `3/274`, `3/290`, `3/565`, `3/573`, `3/683`,
    `3/684`. Their footers can be read, but their column data cannot.

  The footers sum to 149,445,334 rows, 143,584,475 rows can be read, and
  `hats_nrows` and `_metadata` both say 150,198,243. The `PENSUS_BUILD.json` records
  `host: /arc/projects/skyai` and "Public ACL only after promote". The damage looks
  like a truncated or interrupted copy into the community tree (unverified).
- Bias: `UMJD, GMJD, RMJD, IMJD, ZMJD` are null in every row, so there is no epoch.
  `PRIMARY_SOURCE` is 1 in every row. 0.25–0.6 % of rows have a neighbour within 0.5″
  and Gaia ≥2 is 0.5–0.8 % (sample), which is acceptable.
- Cosmetic: `obs_collection=dr5` is the same as KiDS DR5 and VIKING DR5. `IAUNAME` is
  binary.
- Fix: copy the tree again from the skyai build (or rebuild), verify every partition
  by reading it, then re-run the QA. Until then, keep ATLAS out of xmatcher.
  **Affects** X, P, M.

### 2.7 `catwise/2020`

- 1,890,715,640 rows, the published count. Pensus build from the IPAC tiles,
  checked against each tile's `\Nsrc`. Clean. `PMRA/PMDec` are in arcsec/yr: their
  median ratio to Gaia (mas/yr) is 0.0011 (verified, sample), which matches xmatcher's
  comment. Gaia ≥2 ≤0.1 %. Sentinels: `w2sky` −99, `w?snr` 9999, `w?mpro_pm` −9.99.
- Cosmetic: `obs_collection=2020`. **Affects** —.

### 2.8 `clauds`

- 19,027,857 rows, 47 partitions; owner `changbi`; built 2026-07-20; no provenance
  file. Gaia matches have a median separation of 0.02–0.04″.
- Access (metadata): the directory is public, but `properties`, `partition_info.csv`
  and `dataset/` are readable only by group `hats` (anonymous GETs return 403). The
  flat `master_catalog.parquet` (2.4 GB) beside the tree is public. The HATS copy is
  therefore not usable by the community, while the flat copy is.
- Fix: make the ACLs consistent, add a provenance record, and consider moving the
  flat file out of the HATS tree. **Affects** P.

### 2.9 `delve/dr3_gold`

- 2,588,267,795 rows, 6477 partitions, orders 2–10; hats-import 0.6.0 (2025-06),
  `hats_version=v0.1`, no `hats_col_healpix`; the creation date escapes the colon as
  `18\:38UTC`.
- **Bias (verified, all rows):** 8 rows at RA = 360.0 in 7 partitions (`4/1264`,
  `5/4864`, `5/4288`, `5/4351`, `5/4147`, `5/4291`, `5/8354`×2), for example
  `COADD_OBJECT_ID` 8559743794 (RA 360.0, Dec 24.048227). There are no NaN coordinates.
  The tree has not been rebuilt: its properties are unchanged since 2025-06-24, and
  session `qlhgso49` *pensus-des-delve-import* was still running during the audit.
- Sentinels: `DNF_*` −9999, `MAG_AUTO_*` 99, `GAP_FLUX_ERR_*` up to 1e33. Gaia ≥2
  ≤0.4 % (sample).
- Fix: finish the RA-wrapped rebuild. **Affects** X, P.

### 2.10 `des/y6_gold` and `des/y6_gold_prera_backup`

- Rebuilt on 2026-11-24 at 02:33 from the live tree with RA wrapped into [0, 360). The
  full scan finds no out-of-range or NaN coordinate in 691,483,608 rows. The partition
  set and the row counts per partition equal the backup's (all 1633 partitions). The
  backup still has 2 rows at RA = 360.0 (`5/4351`: `COADD_OBJECT_ID` 1033925010;
  `5/4864`: 1034047618), verified by footer statistics and samples.
- **Blocker for xmatcher:** the swap changed `obs_collection` from `des_y6_gold` to
  `y6_gold_rawrap`. xmatcher pins `obs_collection: "des_y6_gold"`, so `verify_tree`
  now fails.
- Cosmetic: the build artefact name `y6_gold_rawrap` is in `obs_collection`, and
  `PENSUS_BUILD.json` still records `planned_hats_path …/y6_gold_rawrap` and
  `live_pin_until_swap`. The rebuilt files carry `Norder/Dir/Npix` columns that are
  null in every row. The 1.39 TB backup is still in the public tree. Sentinels:
  `WAVG_MAG_PSF_*` −99 and `MAG_*` 99.
- Fix: set `obs_collection=des_y6_gold`, or update the pin; move the backup out of
  `/arc/projects/hats`. **Affects** X, P.

### 2.11 `erosita/erass3`

- 1,975,540 rows; LSDB copy (2026-10-26). Clean. There are 2 order-0 partitions, as
  expected for a sparse X-ray catalogue. No action.

### 2.12 `euclid/dr1/*` (IRSA collections: wide, deep, deep photometry)

- These are byte copies of IRSA's HATS collections: `collection.properties`, the
  catalogue, a 10″ margin cache and an `object_id` index, with `md5sums.txt`,
  `linecounts.txt` and download scripts. Rows: wide 888,152,165 (IRSA README: "same
  as original"), deep 51,820,952, deep photometry 26,008,743. Clean structure. These
  are the only community trees with margins, and the only ones whose fields carry
  units and descriptions.
- Layout (cosmetic for HATS, but it broke xmatcher's estimator): the catalogue sits
  inside its collection directory
  (`…/mer_catalogue_wide_survey/euclid_dr1_reformatted_catalogs_mer_catalogue_wide_survey-hats`),
  and partitions are `Npix=N/part0.snappy.parquet` directories. The IRSA properties
  have `hats_builder=irsa-parquet`, no `hats_version`, no `hats_creation_date` and no
  `hats_col_healpix`, and give `hats_cols_sort` as a Python list literal.
  `_common_metadata` declares `Norder/Dir/Npix`, which the files do not hold.
  `partition_info.csv` adds `Dir,num_rows` (it agrees with the footers). The wide
  margin has 769 partitions against the catalogue's 768.
- **Bias (verified, sample):** 10–13 % of rows have a neighbour within 1″, and
  1.3–2.1 % within 0.5″. m_ge2 is 19–28 % (wide) and 9–26 % (deep and deep
  photometry). The second row is in the same `tileid` in 99.6–100 % of pairs. 39–48 %
  of the pairs differ in `spurious_flag` and 30–48 % in `vis_det` (wide). Tree-wide
  (wide; verified, all rows), `vis_det=0` in 199,125,435 rows
  (22.4 %), `spurious_flag=1` in 55,966,502 (6.3 %) and `det_quality_flag≠0` in
  160,144,348 (18.0 %). 551,309,183 rows (62.1 %) have `vis_det=1`, `spurious_flag=0`
  and `det_quality_flag=0`. Many
  columns (for example `flux_*_ext_lsst_*`, `binary_flag`, `blended_prob`) are null in
  every row. In the deep tree, `object_id` -616384011471271136 appears twice in
  partition `5/8583`.
- Duplicate copy: `unions/euclid_dr1_mer_catalogue_wide` (private) is a different
  build with 1,114,025,864 rows, Q1-style column names and hats v0.1 (2026-02).
- Fix: in xmatcher, resolve the collection root through `collection.properties`, and
  prefer `vis_det=1`, `spurious_flag=0` candidates. Keep the IRSA files as they are
  (md5-pinned). **Affects** X, P, M.

### 2.13 `euclid/q1`

- 29,767,806 rows; synced from LSDB (`hats-http-cc72052ad0d6`); hats v0.1
  (hats-import 0.4.4); no `hats_npix_suffix` or `hats_col_healpix`. m_ge2 is 16–19 %
  (sample), with the same MER behaviour as DR1. **Affects** X.

### 2.14 `galex/guvcat_ais_gr67`

- 82,992,062 rows. CDS's GALEX GR6/7 HATS (per xmatcher's docs) and GUVcat_AIS
  have 82,992,086, so 24 rows are missing (metadata; cause unverified, possibly 2 per
  input CSV).
- **Bias (verified, all rows from footer statistics):** `nuv_mag_auto` spans
  [−10.43, 4.32] and `nuv_mag` spans [9.65, 24.40]. That is an offset of exactly 20.08,
  the NUV zero point. `fuv_mag_auto` reaches 4.97 and `fuv_mag` reaches 23.79, an
  offset of 18.82, the FUV zero point. The `*_mag_auto` and `*_mag_aper_*` columns are
  therefore instrumental magnitudes without the zero point. Tree-wide, `nuv_mag − nuv_mag_auto` is 20.08 and `fuv_mag − fuv_mag_auto`
  is 18.82 at the 1st, 50th and 99th percentiles (1.66 M and 0.16 M sampled rows with
  valid magnitudes; verified, follow-up). Whether the MAST CSV
  ships them this way is unverified. Missing values: −999 in `fuv_mag`, `fuv_flux` and
  `fuv_magerr`, and −99/99 in the aperture and auto columns. 93 % of `fuv_mag` are
  −999 in one sample.
- Fix: document this, or add the zero points. **Affects** X (if mag_auto columns are
  projected), M.

### 2.15 `hsc/pdr3/forced_{g,r,i,z,y}`

- Five separate trees, one per band, built 2026-06-26 with hats-import 0.9.0. There
  is no `PENSUS_BUILD.json`, no `hats_col_healpix` and no combined tree. Row counts
  differ per band (g 783,961,389; r 779,192,787; i 757,603,782; z 771,110,298;
  y 731,099,731), and so do the partitions (2010–2574, orders 2/4–10). There are
  20–46 partitions with fewer than 100 rows per band, down to 1 row (`10/4481013`
  in g and r). The files carry `Norder/Dir/Npix` columns that are null in every row.
- Coordinates: `coord_ra`/`coord_dec` are in degrees. The footer statistics of every
  row group span [4.9e−13, 360) × [−7.35, 53.46], and every sampled row falls in its
  pixel in degrees (verified, all rows and sample).
- Schema: 242 columns. There is no `detect_isPrimary`, no `detect_isPatchInner` or
  `detect_isTractInner`, no `merge_*` and no `base_PixelFlags_*`. Only `parent` and
  `deblend_nChild` describe deblending.
- **Blocker (verified, sample of 7 partitions per band):**
  - 12–15 % of rows are deblend parents (`deblend_nChild>0`), 30–55 % are children
    and 30–70 % are isolated;
  - 30–76 % of rows have another row within 0.5″, and 9–15 % have a twin closer than
    1 mas (for example 50,980 of 464,523 rows in g `4/1649`);
  - in full-depth fields (g `6/18007`, i `6/27353`, r `6/27410`, i `7/104868`) the
    Gaia match rate is 97–99 %, Gaia ≥2 is 87–92 % and m_ge2 is 89–95 %. The second
    row sits within about 2×10⁻⁵″, and 69–86 % of these pairs are parent and child.
    The rest are leaf pairs whose `id>>42` (the tract under the usual HSC packing,
    unverified) often differs, that is patch/tract overlaps;
  - the best Gaia match is a parent for 26–44 %, a child for 31–57 % and an isolated
    object for 3–38 %. The task's 39 % of best matches being parents is within this
    range.
- Units (verified, sample): the median of G + 2.5 log₁₀(`base_PsfFlux_instFlux`) for
  17 < G < 20 is 25.5–26.0 (g), 26.7–27.0 (r), 27.2–27.4 (i) and 27.9 (y). These
  values fit counts on the HSC coadd zero point of 27 plus the band − G colour. The
  flux is not in nJy, which would give about 31.4. One i-band sample (`5/1036`) gave
  23.5.
- Fix: rebuild from the PDR3 `forced` tables with `isprimary`, the pixel flags and
  calibrated fluxes or magnitudes (nJy) in one multi-band tree. Until then, select
  `deblend_nChild==0` and de-duplicate by position and tract. Do not use these trees
  as published for a Gaia-anchored master. **Affects** X, P, M.

### 2.16 `hsc/pdr3/forced_nb{0387,0816,0921,1010}`

- 13.5 M, 14.5 M, 14.5 M and 3.1 M rows; 11 columns; Pensus builds of 2026-11-19.
  `PENSUS_BUILD.json` says "Temp HATS only — promote to hats after deep QA +
  headroom; no ACL here", yet the trees are in the public tree. `flux_psf` is in nJy:
  `mag_psf + 2.5 log₁₀(flux_psf) = 31.400` in every sample (verified). Duplicates are
  negligible (0.03–0.1 % of rows have a neighbour within 0.5″). `merge_peak` is the
  only flag. Mostly cosmetic. **Affects** P.

### 2.17 `hubble/hsc_v4` (Hubble Source Catalog v4)

- 139,162,343 rows, orders 1–10, 26 columns (no errors and no flags); no provenance
  record. **Bias (verified, sample):** 22–94 % of rows have another row within 0.5″
  (median spacing about 0.4″), and m_ge2 is 38–100 %. These are distinct `MatchID`s at
  nearly the same position, a known HSC property. The tree is unsuitable for
  one-to-one matching without a deblending policy. **Affects** X.

### 2.18 `jpas/e2dr` and `jplus/dr3`

- 4,804,561 and 16,419,372 rows. Built from CEFCA TAP `MagABDualObj`; both are
  temporary hosts (`/arc/projects/mlao`).
- **Blocker (verified, all rows by type and sample by content):** `TILE_ID`,
  `NUMBER`, `CLASS_STAR`, `FLAGS`, `MASK_FLAGS` and every `MAG_*`/`MAG_ERR_*` column
  are `string`. Photometry and flags are space-separated arrays. J-PLUS has 12
  elements per row, with 99.0000 for missing values (10.7–16.3 % of elements). J-PAS
  has 57 elements, most of them `NaN` in the e2DR samples. The `FLAGS` arrays hold
  `NaN` too.
- J-PLUS array order (verified, sample, 5 partitions): slopes of
  (element − G) against BP−RP of +0.2, +0.8, −0.35, −0.65, +1.6, +1.7, +1.6, +1.4,
  +1.3, +0.8, 0.0 and −0.6, and median offsets from G of +0.05, +0.6, −0.2, −0.3,
  +2.2, +2.0, +1.9, +1.2, +1.1, +0.5, −0.05 and −0.28. This is the CEFCA filter-ID
  order: rSDSS, gSDSS, iSDSS, zSDSS, uJAVA, J0378, J0395, J0410, J0430, J0515, J0660,
  J0861.
- J-PAS: element 0 is very blue (median offset +2.9), and the slopes fall from about
  +1.4 to −0.7 from element 3 to element 55. Element 56 has a slope of −0.39 and an
  offset of −0.2 to −0.25 (an iSDSS-like broad band?). The element-to-filter map is
  unverified.
- Fix: rebuild with numeric `list<float>` columns, or one column per filter named by
  filter, with nulls instead of 99/NaN. Record the filter order in the properties.
  `scripts/pensus/arc/fix_cefca_magab_parquet.py` fixes the header problem, not the
  types. J-PAS also has an order-0 partition (`0/4`, 817,190 rows). **Affects** X, P, M.

### 2.19 `kids/dr5`

- 138,812,117 rows; built from ESO catalogtiles; `id`, `KIDS_TILE` and `THELI_NAME`
  are binary (text stored as bytes). Duplicates are negligible (<0.01 % of rows within
  0.5″), and Gaia ≥2 is ≤1.6 %. Sentinels: `MAG_LIM_*` −99 and `MAG_AUTO` 99.
  Cosmetic. **Affects** P.

### 2.20 `legacysurvey/dr11`

- 3,967,618,454 rows from 2126 north and south sweeps, carried with `PHOTSYS`;
  `integrity_key` RELEASE·10¹²+BRICKID·10⁶+OBJID (metadata). `RELEASE` takes only
  11010 and 11011 (verified, all rows).
- **N/S twins (verified, follow-up):**
  - The footer statistics of `RELEASE` show 485 of the 9,222 partitions holding both
    11010 (S) and 11011 (N), mostly along the Dec ≈ 32° boundary (for example `4/2037`,
    `5/1239`–`5/1273`, `4/334`, `4/416`).
  - In 400 of them (176,991,268 rows), 21.1 % of rows have a twin with the other
    RELEASE within 0.1″ and 37.5 % within 0.5″. Per partition the share is 0–41 %
    (median 6 %): for example `5/1261` has 40.7 % of 519,143 rows and `4/2037` has
    26 %.
  - Extrapolated to the 485 partitions, that is about 45 M rows, or about 22 M objects
    stored twice (about 1.1 % of the tree).
  - In the order-6 sub-pixel with the most twins in each partition, 72 % of 2,143,117
    Gaia sources have two or more LS rows within 1.5″ (per partition 5–100 %, median
    67 %). In a median 97 % of these cases the second row has the other RELEASE. The
    6–23 % of Gaia positions reported earlier fits the edges of the overlap; in its
    core the share approaches 100 %.
  - Whether a twin pair shares the same Gaia `REF_ID` was not established: the audit
    looked for `REF_CAT='G2'`, and DR11 writes `G3`.
- Outside the overlap (verified, sample of 5): each partition holds a single
  RELEASE, and Gaia ≥2 of 1–13 % comes from second rows of the same RELEASE about
  0.6–1.1″ away (real neighbours). Gaia sources are fixed at their Gaia positions, so
  the match separation is 0.000″.
- Fix: keep one row per object in the overlap. The standard DR9+ rule assigns
  north (`PHOTSYS=N`) at Dec ≥ 32.375° in the NGC and south elsewhere, or uses the
  sweeps' `brick_primary`/`MASKBITS` rules. Alternatively, publish N and S as two
  trees. **Affects** X, P, M.

### 2.21 `lotss/dr3`

- 13,667,877 rows, 3 order-0 partitions. `Source_Name` is repeated in the samples (4
  rows in `0/7`, for example `ILTJ172245.73+033306.6`; 2 in `2/21`). This is minor
  for a radio catalogue. **Affects** P.

### 2.22 `pristine/dr2`

- 25,224,342 rows from Zenodo 21092750 (`Pristine_DR2_v2.fits`, md5 recorded).
- Coordinates (verified, sample): `RA/Dec` are the Gaia DR3 positions at J2016.0. The
  separation from Gaia for the same `source_id` is about 2×10⁻⁸″, and every
  `source_id` was found in the Gaia pixel. `RA_Pr/Dec_Pr` are the Pristine positions:
  their median separation from Gaia is 0.05–0.09″, and the 99th percentile is at most
  0.44″. `RA_Pr` reaches 360.0 (it is not the HATS
  column). xmatcher's assumptions are confirmed, and the tree can be joined on
  `source_id` instead of being matched by position.
- **Bias (verified, sample):** 0.17–0.68 % of rows repeat a `source_id` at an
  identical position (813, 1479, 699, 3936 and 525 rows in the five samples).
  Tree-wide (verified, all rows) there are 58,177 duplicate pairs (0.23 % of rows;
  25,166,165 distinct `source_id`). Every pair shares RA/Dec, the Gaia position. `CaHK_0`
  differs in 99.2 % of the pairs (it is identical in 480), and `FeH_Pristine` and `G_0`
  differ in about half. These are separate Pristine measurements matched to the same
  Gaia source, not copies. `G_0` has values down to −489.3 and `CaHK_0` down to −80.1
  (verified, all rows), which are unphysical. **Affects** X, M.

### 2.23 `ps1/otmo`

- 10,560,724,292 rows (MAST's ObjectThin+MeanObject HATS), 27,161 partitions, orders
  2/5/6/7; hipscat-import 0.2.5 converted with 0.4.3 (2024-12-11), `hats_version=v0.1`,
  no `hats_col_healpix` or `hats_npix_suffix`, no sync record.
- **Coordinates (verified, all rows):** 8 rows out of range in 8 partitions. Six have
  RA slightly negative (−6.6×10⁻⁷ to −5.6×10⁻⁶ deg) and two have RA ≥ 360
  (360.0000021 and 360.0), in `5/5116`, `5/3967`, `5/5056`, `6/3051`, `6/13663`,
  `6/17404`, `6/14199` and `6/20416`. Example: `objID` 152510000000013521. There is no
  NaN. `raStack` reaches 360.0.
- **Single detections (verified, all rows):** 7,160,529,807 rows (67.8 %) have
  `nDetections = 1` and 805,690,947 (7.6 %) have 0, so 75.4 % have at most one
  detection. Per sample 10–79 % (72–79 % in typical order-5/6 fields, 10 % in the
  Galactic-plane partition `7/115182`). Rows with
  `nDetections = 1` are 0.04–3.4 % of best Gaia matches, against 24–79 % of unmatched
  rows.
- **Gaia ≥2 = 19–36 %** in high-latitude samples (`5/5116`, `5/3967`, `5/5056`,
  `5/6780`, `6/5997`), with the second row at a median of 1.2″. This would flag most
  bright stars as ambiguous unless single detections are dropped.
- Units (verified, sample): `pmra/pmdec` are in mas/yr, with a median ratio to Gaia
  of 0.98–1.01 over 117–139 stars with |μ| > 30 mas/yr. They are finite for only
  1–79 % of rows (16 % tree-wide) and have absurd tails (±5.8×10⁸). `raMeanErr` is
  consistent with arcsec: the all-row median is 0.17″, and for matched G<19 stars the
  median is 0.6–3 mas. `epochMean` is an MJD (about 55,900–57,100). Both use −999 for
  missing values (about 2 % of rows).
- Fix: drop or flag `nDetections ≤ 1` (or apply MAST's `qualityFlag`) before matching,
  null the −999s, and finish the RA-wrapped rebuild. **Affects** X, M.

### 2.24 `sdss/photoprimary_dr13`

- 469,050,917 rows from the DR16 SAS `eboss/sweeps/dr13_final/301`
  `calibObj-*-stargal-primary` files (`PENSUS_BUILD.json`). These are the DR13
  calibration, which DR14–DR17 imaging reuse, so the data are "DR13 imaging as
  distributed with DR16". `obs_collection=dr16` and `planned_hats_path …/sdss/dr16`
  disagree with the directory name `photoprimary_dr13`. The `stargal` sweeps leave out
  some object types, so the count need not equal SkyServer's PhotoPrimary count
  (unverified).
- **`id` (verified, all rows):** `id` equals `THING_ID` in all 469,050,917 rows. Three
  values repeat (4710, 5418 and 7849), and the composite (`RUN`, `CAMCOL`, `FIELD`,
  `ID`) is unique across the tree. Decoding `id` as an objID matches
  `RUN`/`CAMCOL`/`FIELD`/`ID` in 0–7 % of sampled rows, so it is not the objID. `ID` is
  the object number within a field (1–9304). `RERUN` is binary (`b'301'`). The objID
  can be rebuilt as `(2<<59)|(301<<48)|(RUN<<32)|(CAMCOL<<29)|(FIELD<<16)|ID`
  (skyVersion 2, rerun 301; the firstField bit is unverified). The tree has two
  columns, `ID` and `id`, that differ only in case, which breaks case-insensitive
  tools (SQL engines, some DataFrame libraries, FITS export).
- Sentinels: `MODELMAG_*` −9999. The flux ranges (`PSFFLUX_r` from −199 to 1.7×10⁷)
  fit nanomaggies, as xmatcher assumes, but this was not checked against Gaia. There is
  no epoch column. One order-0 partition (`0/9`, 338,686 rows).
- Fix: rename `id` to `thing_id` (or drop it), add a real `objid`, and settle the
  release label. In xmatcher, use `THING_ID` or the composite key. **Affects** X, M.

### 2.25 `skymapper/dr4`

- 724,329,344 rows; a copy of the LIneA build (no sync record). Clean. `mean_epoch` is
  an MJD (56,730–59,474). 2 order-0 partitions (`0/0`, `0/1`, up to 1.73 M rows).
  Gaia ≥2 ≤0.2 %. **Affects** X (planner, order 0).

### 2.26 `splus/dr4`

- 80,155,547 rows from NOIRLab Data Lab `splus_dr4.dual` (the CDS public DR4 HATS has
  27,235,968 rows). `id` is text (`DR4_3_SPLUS-n20s23_0000102`).
- **Bias (verified, sample of 6):** 0.8–11 % of rows have a twin within 0.5″, and
  every such pair comes from different fields (`SPLUS-n20s34`/`SPLUS-n20s35`,
  `STRIPE82-0087`/`-0088`, `MC0021`/`MC0029`, `HYDRA-0130`/`-0138`, …). Gaia ≥2 is up
  to 6 %. Tree-wide (verified, all rows), 7,700,226 rows (9.6 %) have a twin from
  another field within 0.5″ (8,807,339, 11.0 %, within 1″), and none from the same
  field.
- `*_auto` use 99 for missing values; one order-0 partition (`0/10`, 798,258 rows).
- Fix: de-duplicate the field overlaps (keep the detection nearer its field centre,
  or the best-S/N one), or use a de-duplicated S-PLUS release. **Affects** X, P, M.

### 2.27 `uhs/dr3` and `uhs/dr3_prera_backup`

- Rebuilt on 2026-11-23 at 22:53 with RA wrapped into [0, 360): no out-of-range row
  in 514,336,270 (verified, full scan). The single RA = 360.0 row of the backup
  (`sourceid` 460073108063, Dec 53.9883731, partition `4/175`) is fixed. Partitions
  and row counts match the backup exactly. `obs_collection=dr3_cephwrap` is a build
  artefact. The backup (27 GB) is still public. xmatcher deliberately leaves this tree
  unpinned. See [2.30](#230-vistaukirt-trees-vhs-vvvx-vmc-viking-video-ukidss) for the
  columns. **Affects** X, P, M.

### 2.28 `ukidss/dr11_{las,gcs,dxs,uds}`

- Clean structure. `PENSUS_BUILD.json` for DXS and UDS says
  `WSA UKIDSSDR11PLUS gcsSource`, but the staging scripts queried `dxsSource` and
  `udsSource`, and the partitions lie on the DXS fields (XMM-LSS, Lockman, ELAIS-N1,
  SA22) and the UDS field (34.5, −5.1). The label is wrong and the data are right.
  DXS has 2 order-0 partitions. See 2.30. **Affects** P.

### 2.29 `lsst/*` (private, group `LSST_Can`)

- `lsst/dp1` holds three DP1 collections of nested light-curve tables
  (`object_collection`, `dia_object_collection` with 5″ margins and id indexes, and
  `object_photoz`), built in 2025 with hats v0.1 (lsdb 0.5.2 / hats-import 0.6.0).
  `object_photoz` lacks `hats_order`. `lsst/dp2` is a symbolic link to
  `/arc/projects/LSST/lsdb_data/dp2` (`object_collection`, `dia_object_collection`,
  `object_photoz`, `public-files`), outside this tree and not audited.
  `lsst/instructions.md` describes the USDF transfer (5.5 TB).
- For xmatcher, `XMATCHER_DP2_HATS` must point at the object table inside the
  `object_collection` under `/arc/projects/LSST/lsdb_data/dp2`. Access depends on LSST
  data rights. **Affects** X.

### 2.30 VISTA/UKIRT trees (VHS, VVVX, VMC, VIKING, VIDEO, UKIDSS)

- 11–17 columns each: `sourceid, ra, dec, mergedclass, priorsec, framesetid`, plus
  `*apermag3` and `*apermag3err` for the bands present. VHS DR7, VVVX DR1 and UHS DR3
  have J, H and Ks/K only. VMC has Y, J and Ks; VIKING and VIDEO have Z, Y, J, H and Ks.
- Missing values: −999999500 in every magnitude and error column (verified, all
  rows).
- No epoch (`mjd`), no position errors, and no `ppErrBits` or `errBits`. Saturated
  and artefact-flagged detections cannot be removed.
- The seamless primary selection is correct: in every sample `priorsec` is 0 or equal
  to `framesetid`, with no secondary rows. Duplicates are negligible (<0.5 % of rows
  within 0.5″), and Gaia ≥2 is 0.03–4 % (12.5 % in the VVVX bulge partition `7/171358`).
- `obs_collection` collides: VHS and VMC are both `dr7`, VIKING `dr5` matches ATLAS
  and KiDS, and VVVX is `dr1`.
- Fix: re-stage with `mjd`, `ppErrBits`/`errBits` and position errors per band, and
  null the sentinel. **Affects** M (and X for epochs).

### 2.31 `unions/*` (private, group `hats`)

- There are 10 HATS trees, built in 2025–2026 with hats-import 0.5.1 (v0.1):
  `cfis_DR6_{u,r}`, `cfis_DR6_{u,r}_processed`, `cfis_LSB_DR6_r`, `PGM`
  (2,793,645,382 rows), `gaap_ugriz_photoz_ext`, `ShapePipe/{cut_struct_v1.6.6,psf_v1.6a}`
  and `euclid_dr1_mer_catalogue_wide`.
- **Blocker:** `unions/gaap_sp_ugriz_photoz_ext` (311 GB, 637,129,635 rows) is an
  unfinished build. It has no `properties`, no `_common_metadata`, an empty
  `_metadata` (0 rows, none of the 1650 files referenced) and an `intermediate/`
  directory, and 20 of its files lack the `N_EPOCH` column.
- A `.claude/settings.local.json` agent settings file sits in `unions/` (not read).
  **Affects** P.

### 2.32 `zensus/20261019` (private)

- This is a derived product (the zensus full-sky redshift catalogue, 1,169,651,709
  rows, order 5, with a 5″ margin and a 33 GB flat copy), not a survey. One partition
  (`5/8323`) holds 1.58 M rows against `hats_max_rows=1,000,000`.
  `.astroai-lab/pythonpath` points at a path that does not exist
  (`/arc/projects/hats/zscrape/...`). Cosmetic. **Affects** P.

## 3. Cross-cutting issues

1. **Naming and identity.** `obs_collection` holds the release directory, not the
   survey: `dr5` (ATLAS, KiDS, VIKING), `dr7` (VHS, VMC), `dr3` (J-PLUS, LoTSS), `dr1`,
   `2020`, `dr4`, `dr16` (in a `dr13` directory). Two hold build artefacts:
   `y6_gold_rawrap` and `dr3_cephwrap`. Anything that keys on `obs_collection` (xmatcher
   pins, LSDB registries) gets collisions or breaks at each swap. Use
   `<survey>_<release>` (for example `atlas_dr5`) and keep it stable across rebuilds.
2. **Provenance.** There is no sync or build record for `gaia/dr3`,
   `allwise/catalog`, `skymapper/dr4`, `ps1/otmo`, `delve/dr3_gold`, the five HSC
   broad-band trees, `hubble/hsc_v4`, `lotss/dr3`, `erosita/erass3` or `clauds`. The
   DXS and UDS records name the wrong WSA table. Most `PENSUS_BUILD.json` files say
   `temporary_host: true` and "Public ACL only after promote", yet the trees are
   published. Update or remove those notes after promotion.
3. **Margins.** Only the Euclid DR1 collections (and the private LSST and zensus
   trees) have margin caches. A Gaia-anchored match at 1.5″ must read neighbouring
   partitions for every other tree.
4. **Partition orders.** There are order-0 partitions in `2mass/xsc` (12), `erosita`
   (2), `jpas` (1), `lotss` (3), `sdss` (1), `skymapper` (2), `splus` (1),
   `ukidss/dr11_dxs` (2), `lsst/dp1/object_photoz` (1) and `unions/ShapePipe/psf_v1.6a`
   (3), and orders up to 10 in HSC, DELVE, Hubble and LSST. A planner that overlaps
   partitions must handle order 0 against order 6 (Gaia). Many partitions are tiny:
   HSC has 20–46 per band with fewer than 100 rows, DES has `6/47393` with 2 rows, and
   VHS has 10 with fewer than 100.
5. **Missing-value conventions.** These differ by tree, and most are numbers rather
   than nulls: −999999500 (VSA/WSA), −999 (PS1, GALEX), −99/99 (DES, DELVE, KiDS,
   S-PLUS, J-PLUS, GALEX, 2MASS, AllWISE), −9999 (SDSS `MODELMAG`, DELVE `DNF_*`),
   99.99 (AllWISE, 2MASS), NaN in the J-PAS text, and real nulls elsewhere (Gaia,
   SkyMapper, Euclid, HSC). Footer min/max for every column is in `structure/*.json`
   (`column_stats`, `sentinel_like_min_max`).
6. **Epochs and motions.** Only Gaia declares an epoch (`hats_coordinate_epoch`). Per-row
   epochs exist only in 2MASS (`jdate`), AllWISE/CatWISE (MJD), SkyMapper
   (`mean_epoch`), PS1 (`epochMean`, −999 for missing) and LS (`REF_EPOCH`, 0 for
   non-Gaia sources). The ATLAS MJD columns are empty. None of the VISTA/UKIRT, DES,
   DELVE, KiDS, HSC, Euclid, S-PLUS, J-PLUS, J-PAS or SDSS trees has one. Proper
   motions: Gaia; CatWISE (arcsec/yr, verified); PS1 (mas/yr, unreliable); AllWISE
   (not astrometry); LS (copies of Gaia).
7. **Mixed HATS generations.** v0.1 trees (`delve`, the DES backup, `euclid/q1`,
   `ps1/otmo`, `unions/*`, `lsst/dp1`) lack `hats_col_healpix`, and some escape `:` in
   dates. The IRSA trees have no version. The `Norder/Dir/Npix` columns embedded in the
   files are null (DES rebuild, HSC) or dictionary-encoded (XP).
8. **Backups and duplicate copies in the public tree.** `des/y6_gold_prera_backup`
   (1.39 TB) and `uhs/dr3_prera_backup` still contain the RA = 360 rows, and the second
   Euclid DR1 wide copy is in `unions/`. The tree uses 24.07 of 27.7 TB (87 %).
9. **Tree-level clutter.** `.local/` (empty), `.herd-gidsync.state`, `pensus/` (empty
   `views/` and `sidecars/`, `docs/quota-probe.json`), `registry/hats_registry.json` (a
   2026-10-22 snapshot of the IVOA registry, not a HATS tree) and
   `unions/.claude/settings.local.json`.
10. **Missing products.**
    - There is no S-PLUS DR5 tree; xmatcher's `splus5_hats` is only a placeholder.
    - LSST DP2 is only a link to `/arc/projects/LSST`.
    - There is no GALEX MIS, VPHAS+, IPHAS, UKIDSS GPS or GLIMPSE.
    - HSC has no combined multi-band table with `isprimary`.
    - There is no LS DR11 tractor-level product apart from the sweeps.
    - The temporary or pending-QA trees are the HSC NB trees, ATLAS, KiDS, J-PLUS,
      J-PAS, S-PLUS, SDSS, VHS, VVVX, 2MASS XSC and the UKIDSS trees (per their
      `PENSUS_BUILD.json`).
11. **What a Gaia-anchored match should expect** (sampled m_ge2): about 0–1 % for
    2MASS, AllWISE, CatWISE, SkyMapper, SDSS and the VISTA trees; a few % for DES,
    DELVE, KiDS and ATLAS; 1–13 % for LS outside the overlap (median 67 % and up to 100 % in it); up to
    6 % for S-PLUS (9.6 % of its rows are field-overlap twins); 9–28 % for Euclid; 19–36 % for PS1; 70–95 % for HSC broad bands;
    and up to 100 % for Hubble HSC. Ambiguity flags will dominate the PS1, Euclid, HSC
    and Hubble HSC columns unless their cleaning cuts are applied first.

## Appendix A. Method

All access was read-only. Nothing was written, synced or `chmod`ed under
`/arc/projects/hats`. The audit script refuses an output path inside the audited
root, and every output went to `/arc/projects/astroai/msa`.

1. **Remote metadata (this machine, 2026-11-24 06:20–07:40 UTC).** Anonymous and
   proxy-certificate VOSpace listings of every directory down to each catalogue root
   (`/arc/nodes/projects/hats/...`), and GETs of the small files only: `properties`,
   `hats.properties`, `partition_info.csv`, `collection.properties`, `PENSUS_*.json`,
   `README.txt`, `md5sums.txt`, `linecounts.txt`, `dataset/_common_metadata`,
   `registry/hats_registry.json` and `lsst/instructions.md`. No data partition and no
   `_metadata` file was downloaded. These files gave the properties, partition-table
   and schema checks, and the comparison with xmatcher's pins
   (`src/xmatcher/xmatcher.yaml` at sfabbro/xmatcher `d651612`). Only the DES pin fails.
2. **CANFAR CPU job, `scripts/audit_hats_community.py`.** One headless session through
   the MSA launcher, step `hats-survey-audit` (16 cores, 96 GB,
   `images.canfar.net/astroai/base:latest`), reading the tree through the `/arc` mount:
   - *discovery*: a walk that follows no symlink;
   - *structure* (every catalogue): properties keys, the partition table against the
     files on disk, the footer of every partition file (rows, row groups, schema,
     per-column min/max/null statistics over all row groups), `_metadata`,
     `_common_metadata`, and margin/index siblings;
   - *samples* (per catalogue: the named Pensus QA partitions, partitions whose footer
     statistics put a coordinate out of range, the median-size partition of each order,
     then up to 5 random partitions within a 2.5 GB read budget): pixel membership of
     every row (with the radians hypothesis), `_healpix_29` consistency, duplicate ids
     within the partition, nearest-neighbour distances, a profile of every column in
     the first two sampled partitions (type, NaN/null, sentinels, numbers stored as
     text), and the Gaia DR3-anchored count of rows within 1.5″ of each Gaia source
     (in one order-6 sub-pixel when the partition order is below 6), with per-survey
     diagnostics;
   - *fullscan*: every row's coordinates in each catalogue below 2×10⁸ rows, plus UHS,
     DES, DELVE and PS1;
   - *compare*: rows per partition of each rebuilt tree against its backup.
3. **Follow-up CANFAR job** (`--stages followup`): tree-wide counts for LS DR11
   (every partition whose footer statistics hold two RELEASE values, up to 400 read),
   PS1 `nDetections`, Pristine `source_id`, SDSS `id`, S-PLUS twins, GALEX zero points
   and the Euclid wide flags.
4. **Limits.**
   - Duplicate ids are checked within each partition only.
   - The main pass's nearest-neighbour pair tallies can miss some exact-duplicate
     pairs: with two points at distance 0, the KD-tree may return the point itself as
     the second neighbour. The neighbour fractions and the Gaia counts are not affected,
     and the follow-up uses a neighbour search that skips the point itself.
   - The Gaia check uses J2016.0 positions without propagation to the survey epoch.
   - Trees that Pensus was rebuilding (PS1: session `ykehpdqk`; DELVE: `qlhgso49`) were
     read as they stood.
   - No row data were read from `lsst/dp2` (outside the tree).

## Appendix B. Script, sessions and outputs

- Script: `scripts/audit_hats_community.py`, tested by
  `tests/test_audit_hats_community.py` (a synthetic tree with a missing partition, an
  out-of-range RA, a duplicate id, a positional twin, a sentinel, numbers as text and
  north/south twins, plus the refusal to write under the root). Launcher steps
  `hats-survey-audit` and `hats-survey-audit-followup` in
  `scripts/launch_master_label_training_chain.py`.
- Main audit: session **`umd2ni2t`** (attempt 2; attempt 1 was never created during a
  CANFAR outage), MSA commit `3f1ab3c`, run
  `hats_survey_audit_20261124`, 07:38–08:15 UTC. Receipts:
  `experiments/xp/decision-wave-20261106/hats_survey_audit_20261124{,-attempt-2}.json`.
- Follow-up: session **`dlec1oky`** (attempt 2; attempt 1 was never created during an
  outage), MSA commit `cd6e4ed`, run `hats_survey_audit_20261124_followup`, 08:42–08:57
  UTC. Receipts: `experiments/xp/decision-wave-20261106/hats_survey_audit_20261124_followup{,-attempt-2}.json`.
- Outputs in `/arc/projects/astroai/msa/runs/hats_survey_audit_20261124/`:
  - `discovery.json`: the tree walk, links and access;
  - `structure/<tree>.json`: properties, the partition table against the files,
    footer statistics of every column, `_metadata` and `_common_metadata`;
  - `partitions/<tree>_partitions.csv`: rows and RA/Dec ranges of every partition;
  - `samples/<tree>.json`: the sampled-partition checks and Gaia multiplicity;
  - `fullscan/<tree>.json`: the per-row coordinate scans;
  - `compare.json`: rebuilt trees against their backups;
  - `summary.json`: one line per tree;
  - `followup/followup.json`: the tree-wide follow-up counts;
  - the session log, `hats_survey_audit_20261124.session.log`.

## Appendix C. Related: k-pop `adql_matches` (outside the HATS tree)

`/arc/projects/k-pop/catalogues/adql_matches/twomass.fits` (469,001,402 rows)
pairs Gaia sources with the wrong 2MASS rows. Its `tmass_oid` holds the DR3
`clean_tmass_psc_xsc_oid`, but its photometry is that of the DR1
`tmass_original_valid` row with that number, which is another 2MASS source at a
similar declination. The correlation of Ks with G is 0.002–0.016 instead of 0.95
for the correct designation join. This is documented in `docs/programme/data.md`
and `docs/programme/README.md`, and `build_photometry_v1a.py` refuses the file. The
2023 `tmass_adql_match_files` behind the Andrae-file J/H/Ks share the fault.
Whether the PS1, SDSS and CatWISE `adql_matches` tables are mis-keyed the same way
was not checked by this audit (another agent is checking). This HATS audit did not
re-verify any of this. Use the community 2MASS PSC tree (2.3) with a Gaia-anchored
match, or the designation join of `data/twomass_gaia.py`.
