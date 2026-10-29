# Requested survey-union inventory — 28 October 2026

Scope: source-only repository inspection, public primary survey documentation and
read-only project storage metadata. Each staged FITS header was inspected from at
most the first 512 KiB; ARC ignored the HTTP Range request, so the stream was
closed after the bounded prefix. Header matches do **not** certify whole-file
hashes, publication provenance, unique stars, feature coverage, usable labels,
model training or absence of other copies. The local header evidence is
[the recorded header inventory](../../experiments/xp/survey-union-headers-20261028.json) and its reproducer is
[the bounded header audit](../../scripts/audit_survey_fits_headers.py).

## Existing multi-survey reader and known source

The completed Gaia GSP-Spec overlay is a **union** within the old FIT/feature
base of APOGEE DR17, DESI DR1 RVSpecFit, Gaia DR3 GSP-Spec and GALAH DR4; it
does not cover all requested surveys or all external feature rows. It stores
Gaia source-specific M/H and alpha as extras, but the present three-target
reader does **not** consume those extras as alpha or Fe/H training targets.
The precise completed overlay receipt is
`experiments/xp/multisurvey-gaia-gspspec-overlay-completion-20261028.json`.
A full source, target-availability and feature union is still needed.

## High-priority staged catalogues outside that four-source union

- **APOGEE DR19 / SDSS-V and BOSS DR19:** staged
  `/arc/projects/k-pop/catalogues/astraMWMLite-0.6.0.fits.gz`,
  895,352,521 compressed bytes; bounded FITS header: 1,365,769 rows, 332
  columns, `gaia_dr3_source_id` stored as FITS int64, native `teff`,
  `logg`, `fe_h`, `m_h_atm`, `alpha_m_atm`, errors and flags. The
  [SDSS DR19 guide](https://www.sdss.org/dr19/data_access/get_data/) names this
  exact file as a mixed-pipeline recommended-parameter compilation of APOGEE
  1.3, APOGEE DR17 and BOSS spectra. Its
  [selection logic](https://www.sdss.org/dr19/mwm/astra/pipelines-in-astra/bestparams/)
  is population- and pipeline-dependent and even uses Gaia color/absolute
  magnitude for some strata. Do not treat Gaia-derived `zgr_*` entries as new
  independent spectroscopic truth, nor count DR17/DR19 reductions of the same
  spectrum as independent observations. The locally staged file has not yet
  been compared with the publisher checksum or checked for valid ID uniqueness.
  [SDSS DR20](https://sdss.org/dr20/mwm/) states APOGEE products are copies of
  DR19; DR20 adds newer BOSS spectra/pipelines, which this DR19 file does not
  establish are staged.

- **RAVE DR6 final:** staged files:
  `/arc/projects/k-pop/catalogues/rave_dr6_madera.fits` (45,630,720 B;
  header 518,387 observation rows), `rave_dr6_x_gaiaedr3.fits`
  (289,759,680 B; header 517,357 rows), `rave_dr6_cnn.fits` (64,854,720 B;
  header 426,574 rows), `rave_dr6_gaia_bdasp.fits` (337,924,800 B), and
  `rave_dr6_bdasp.fits` (61,358,400 B). The MADERA observation table has
  `rave_obs_id`, Teff, logg, **M/H**, errors, SNR and `algo_conv_madera`;
  [M/H is not direct Fe/H](https://www.rave-survey.org/metadata/ravedr6/dr6_madera/).
  The Gaia-EDR3 crossmatch has exact `rave_obs_id` and `source_id`; its
  `source_id` FITS column is `K` with `TNULL=9223372036854775807`, a
  **positive** null sentinel. The CNN table has Fe/H, alpha/M, Mg/Fe, Si/Fe,
  errors and flags, but its `source_id` is explicitly **Gaia DR2** per
  [RAVE's schema](https://www.rave-survey.org/metadata/ravedr6/dr6_cnn/).
  Join by observation ID to the published EDR3 crossmatch and explicitly handle
  nulls, repeated spectra and ambiguous crossmatches; never relabel the CNN DR2
  ID as DR3. EDR3 source IDs use the Gaia DR3 source identity system, but the
  association itself still needs checking. The
  [RAVE DR6 release](https://www.rave-survey.org/metadata/ravedr6/) also
  provides GAUGUIN Fe/H and alpha/Fe and its uncertainties in a separate
  [table](https://www.rave-survey.org/metadata/ravedr6/dr6_gauguin_madera/),
  not identified in this limited top-level staging listing. BDASP uses Gaia
  parallax; model the dependency and do not call it an independent spectroscopic
  gravity anchor. Repeated RAVE reductions are not independent sources.

- **Gaia-ESO DR5.1:** staged
  `/arc/projects/k-pop/catalogues/ges_dr5.1.fits` (301,576,320 B);
  bounded FITS header has 114,916 rows, 350 columns, `OBJECT`, RA,
  DECLINATION, `TEFF/E_TEFF`, `LOGG/E_LOGG`, `FEH/E_FEH` and
  `MG1/SI1/CA1/TI1` plus corresponding `E_*` fields; no Gaia-ID field
  appeared in the inspected header. The count matches
  [ESO DR5.1](https://www.eso.org/qi/catalog/show/411), but the local full hash
  and exact provenance are unverified. Retain each element and ionization
  definition rather than inventing a composite alpha. A published
  [Gaia-ESO/Gaia DR3 comparison](https://doi.org/10.1051/0004-6361/202450357)
  used a 2-arcsec cone, reporting 114,864 matches and discarding 52 ambiguous
  stars; these are the authors' numbers, not a certification of our staged
  crossmatch. Reproduce a versioned, proper-motion/ambiguity-aware match.

- **LAMOST DR8 v2 fallback:** `/arc/projects/k-pop/catalogues/lamost-param/`
  contains `dr8_v2_LRS_stellar.fits` (3,302,118,720 B; header 6,684,413
  observation rows) and `dr8_v2_MRS_stellar.fits` (725,909,760 B; header
  1,230,307 rows), with `obsid`, `gaia_source_id` as `25A` text and
  native temperature/gravity/metallicity. LRS contains `feh`, `alpha_m`
  and errors; MRS contains `feh_lasp`, `alpha_m_lasp`, `alpha_m_cnn`
  and several element ratios. Parse IDs exactly as integers, never via float.
  Current [DR8 v2 LRS](https://www.lamost.org/dr8/v2.0/doc/lr-data-production-description)
  and [MRS](https://www.lamost.org/dr8/v2.0/doc/mr-data-production-description)
  descriptions say Gaia DR3, whereas DR8 v1 used DR2. The staged filenames
  suggest v2 but need publisher-hash/version confirmation. Related
  `catalogues/calsurvey/lamost_dr8_lrs_mrs_filtered.parquet` is already
  filtered and should not be confused with the raw union.

- **Metal-poor targeted sources:** staged
  `catalogues/li_et_al_lamost_vmps.fits` (77,760 B) and
  `catalogues/li_et_al_x_gaiaids.fits` (63,360 B); the repo also records a
  Li et al. LAMOST DR10 metal-poor source in
  `configs/stellar-improvement-plan.yaml`. Their release, quality and exact
  Gaia-ID linkage have not been independently checked here. Preserve selection
  and overlap provenance; metal-poor targeted spectra can add domain support
  but are not a representative abundance truth sample.

## Requested public sources not established as staged or consumed

- **LAMOST DR12 v2.0** is marked public by the
  [release listing](https://www.lamost.org/lmusers/) and is newer than the
  staged DR8 files. Its [LRS parameter schema](https://www.lamost.org/dr12/v2.0/table/stellar)
  contains `gaia_source_id` as text for Gaia DR3, `teff/logg/feh/alpha_m`
  and errors. The [official documentation](https://www.lamost.org/dr12/v2.0/doc/lr-data-production-description)
  says the Gaia association used a 3-arcsec crossmatch: preserve uncertainty,
  offsets and multiplicity, not just an integer key. The publisher states that
  the quantity named `feh` is a metallicity proxy not rigorously identical
  to GALAH iron, and that MRS CNN outputs currently lack uncertainty estimates
  ([release notes](https://www.lamost.org/dr12/v2.0/doc/release_note)).
  Treat LASP alpha/M as its native quantity and errors as explicitly unknown
  where absent. The LRS AFGK count 8,376,135 is catalogue entries, not an
  independently verified unique-star count. An anonymous FTP-link attempt
  redirected to a login page, but this does not establish whether the user has
  access through another service or that the full catalogue is unavailable.

- **SDSS DR20 BOSS/SpecObj:** [SDSS ALLSPEC](https://www.sdss.org/dr20/data_access/allspec/)
  connects DR20 BOSS `spAll` and DR17 `SpecObjAll` using `specobjid`.
  A spectrum identifier is not itself a stellar-parameter label: join to
  appropriate analysis pipelines, retain release/instrument/pipeline identities
  and crossmatch SDSS IDs to Gaia DR3. DR20
  [Astra](https://www.sdss.org/dr20/mwm/data/parameters/) recommends different
  pipelines by star type and lists BOSS-CLAM/MIN​ESweeper products for
  metallicity/alpha regimes. Check their input/training dependencies and source
  flags. No DR20 BOSS label catalogue was established as staged in this audit.

- **MaStar:** SDSS recommends the
  [v2 parameter value-added catalogue](https://www.sdss4.org/dr17/mastar/mastar-stellar-parameters/),
  with `TEFF_MED`, `LOGG_MED`, `FEH_MED`, `ALPHA_MED`, errors and four
  method-level measurements; the
  [per-star data model](https://data.sdss.org/datamodel/files/MANGA_SPECTRO_MASTAR/DRPVER/MPROCVER/vac/parameters/VER/mastar-goodstars-DRPVER-MPROCVER-params-VER.html)
  lists 24,290 unique MaNGAID records and RA/DEC but no Gaia ID. A verified
  DR3 crossmatch remains needed. `FEH_CAL_MED` is APOGEE-calibrated and the
  four methods share MaStar spectra, so do not multiply their likelihoods as
  independent; keep original and calibrated estimates distinct.

- **DESI:** the public
  [DR1 stellar VAC](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/)
  is currently the latest linked release and is already staged as
  `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/desi_collection_v1/mwsall-pix-iron.fits`.
  It lists 6,372,607 fitted coadded spectra, 4,920,596 unique Gaia sources and
  4,586,247 unique spectroscopically classified DESI stars; these populations
  overlap and are not billions of unique labelled stars. DESI says
  [DR2 is planned public Spring 2027](https://help.desi.lbl.gov/index.php?qa=179&qa_1=availability-of-desi-dr2-y3-bao-data-split-by-sky-region).
  RVS and FERRE/SP fits share spectra; do not treat them as independent. Keep
  alpha below the source's low-metallicity validity domain as a labelled
  diagnostic rather than fabricating reliable truth.

## Recommended next actions

1. Ingest the already staged DR19 astraMWMLite, RAVE DR6 tables and Gaia-ESO
   DR5.1 via separate, source-only adapters, preserving every raw record,
   per-field quality and native error status. Verify full source digests, exact
   release origin, duplicate handling, and Gaia-ID match/null semantics.
2. Prioritize DR12 v2 LRS/MRS. The user requires at least DR11; do not use
   the staged DR8 as a fallback. Preserve each release and pipeline identity,
   exact Gaia DR3 identifiers, source-specific metallicity and alpha quantities,
   repeat spectra, provenance and minimally filtered raw rows.
3. Add DR20 BOSS and MaStar v2 after identifying their actual public product
   and DR3 crossmatch. Keep multiple pipelines on the same spectra linked as
   dependent; retain valid partial stars instead of imposing intersection cuts.
4. Produce identity-level full-survey **union** counts by target, survey, role,
   feature availability, metallicity and quality. Keep unknown outside-base
   ancestry and cache nonmatches explicitly unknown.
5. Extend the likelihood to fit native M/H, Fe/H, alpha/M, alpha/Fe and
   individual elemental observables through documented source-dependent maps.
   Learn relative transformations only where overlap/anchors support them;
   track nuisance/measurement covariance and distinguish physical calibration
   from a chosen consensus gauge. Derived algebraic quantities must not be
   multiplied as independent observations of their input measurements.
