# Age consensus: one age scale learned from many catalogues (design note, 2026-11-22)

Status: data audit plus design. Nothing is trained or launched. The audit is reproduced by
`scripts/age_label_audit.py` (stages `ingest`, `gaia`, `compare`, `plots`); its tables and
plots are in `docs/age-audit-20261122/`. This note builds on
`design-20261122-data-driven-inference.md` (ground rules, products A/B/C, kinematics
circularity), `plan-20261022/09-age-data-and-systematics.md`,
`plan-20261022/11-anchors-a1-a2-a3.md`, and the anchored survey maps used for [Fe/H]
(`training/anchored_survey_map.py`, `anchored_multisurvey.py`, `common_scale_labels.py`).

Units: `lage = log10(age / Gyr)`. Differences are in dex (0.1 dex = 26 %). The training
code (`multi_teacher_age.py`) uses `tau = ln(age / Gyr) = 2.303 lage`. NMAD = 1.4826 × the
median absolute deviation. "Inflation" = NMAD of a pair difference divided by the median
of the two quoted errors added in quadrature: above 1, the quoted errors are too small;
below 1, the two catalogues share inputs and their errors are correlated.

## 0. Summary

- **Data.** 14 publications were ingested and matched to Gaia DR3: 13 star-level age
  catalogues plus the HR23 cluster catalogue, in 37 catalogue variants. In all there are
  8.05M age records for **3.61M distinct stars**. Without Gaia FLAME: 2.76M stars, of which
  385k have ages from ≥ 2 different catalogue families and 16.5k from ≥ 3. A family
  counts once: all StarHorse inputs are one, all Willett files are one.
- **Precision of the best methods**, from pairs of catalogues measuring the same stars:
  - seismic vs seismic, Kepler giants: APOKASC-3 − Willett 2026, **−0.021 dex, NMAD 0.065**
    (n = 7,644);
  - isochrone vs isochrone, LAMOST subgiants: StarHorse − Xiang 2024, **−0.008, NMAD 0.088**
    (n = 200,301);
  - two GALAH pipelines: GALAH DR4 − StarHorse(GALAH DR3), **+0.026, NMAD 0.052**
    (n = 100,679).
- **Systematics are regime-specific, not global:**
  - APOKASC-2 − APOKASC-3 is −0.10 dex for red-clump stars and +0.005 for RGB stars. The
    same pipeline family moved its RC scale by 26 % between versions.
  - FLAME DR3 has a 0.4 dex trend with [Fe/H] (−0.27 below −1, +0.14 above +0.1). Its
    giants are unusable: APOKASC-3 − FLAME = +0.89 dex, Spearman 0.19.
  - Every field-star pipeline puts main-sequence members of young clusters at 2–5 Gyr:
    +1.0 to +1.4 dex for clusters below 1 Gyr. These values are priors, not measurements.
- **Cluster ages are not a clean anchor.** The Hunt & Reffert (2023) neural-network ages
  are too young for old clusters:
  - NGC 6791: 3.8 Gyr, against 8.2 (APOKASC-3), 9.0 (Willett) and 8.6 (Leung) for its
    giants;
  - M67: 1.7 Gyr, against 4.7–6.2 from the turn-off pipelines.
- **Co-natal precision checks.** Within-cluster and within-binary scatter shows that the
  quoted errors are 1.5–5× too small (StarHorse, FLAME). A data-driven estimator
  (Leung 2023) is tighter than its quoted errors (ratio 0.7).
- **Problems in the releases** (all handled in the script):
  - the Willett 2026 `GAIADR3_ID` column is constant;
  - the Willett K2-GALAH `age_Dnu` is copied from K2-APOGEE in every one of 4,163 rows;
  - `AGE_DW` in the APOKASC-3 VAC equals Chaplin et al. 2014 exactly;
  - the AMBRE file carries Gaia DR2 IDs only.
- **Bridging is the weakest link.** The giant (seismic) regime and the turn-off/subgiant
  (isochrone) regime share almost no stars. Only these connect them:
  - seismic dwarfs (Chaplin 2014, 110–307 overlaps; LEGACY, 38);
  - GALAH DR4, which runs one pipeline across both regimes;
  - two young Kepler clusters;
  - FLAME, whose giants are unusable.
- **Recommendation.**
  - Gauge: APOKASC-3 as the identity anchor for giants.
  - Co-natal equality constraints (clusters, wide binaries) as likelihood terms.
  - Xiang 2024 as a second, declared scale variant for subgiants. The difference between
    the two anchored solutions is published as the scale systematic.
  - The user chooses the anchor (§4).
- **First experiment (C0).** Fit the consensus from the labels alone: per-source maps,
  noise, censoring and co-natal terms, with no network. This is CPU work taking minutes on
  the audit's `records.parquet`.

## 1. What "better accuracy without a stellar model" can and cannot mean

Age is not observed. For a field star, every absolute age passes through a physical model
somewhere:

- isochrones or tracks for turn-off and subgiant stars;
- seismic scaling relations plus a grid for giants;
- cooling models for white dwarfs;
- a cluster sequence dated with isochrones for gyrochronology;
- seismic training sets for chemical clocks and machine-learning estimators.

Our ground rules forbid stellar or Galactic models *inside our inference*. They allow
model-based catalogues as noisy, method-biased measurements. Within those rules:

**What a consensus can improve.**

1. *Dependence on one pipeline.* Each catalogue's grid, input-parameter scale, priors and
   selection leave a distortion that is a smooth function of age, metallicity and
   evolutionary state. Where catalogues overlap, the relative distortions are
   identifiable, and a consensus that models them is no longer tied to any one
   pipeline's choices. Examples from §2:
   - the APOKASC RC scale moved 0.10 dex between versions;
   - FLAME assumes solar metallicity;
   - StarHorse is 0.05–0.10 dex older on APOGEE inputs than on Gaia RVS inputs for the
     same stars.
2. *Prior domination.* A posterior median in a regime with no information is the prior
   mean. Treating it as a measurement creates false precision. A measurement model with
   information-dependent slopes (§3.4) detects this and down-weights it: the young
   clusters in §2.3 are the clearest case.
3. *Cross-regime consistency.* Members of a cluster and components of a wide binary share
   an age. That equality needs no model. It is the only fully model-independent age
   information we have. It constrains relative ages, across regimes and across
   catalogues, and it measures precision directly.
4. *Error calibration.* Repeated measurements of one star and co-natal replicates fix
   each catalogue's real noise (its inflation). The consensus intervals are then honest
   where the quoted errors are not.

**What it cannot do.**

1. *Absolute scale.* The zero point and slope of the age scale come from whatever we
   declare as the anchor (gauge, §3.6). The consensus is "age on the APOKASC-3 scale" (or
   another named scale), never truth.
   - For giants, age scales roughly as M^-3 at fixed composition, so a 3 % error in the
     seismic mass scale is about 0.04 dex in age.
   - The anchor's systematics pass straight into every star.
2. *Errors shared by all catalogues.* Common input physics (mixing length, diffusion,
   α-element treatment, RGB mass loss) moves all methods together and cancels in every
   pairwise difference. Only methods with different physics break this degeneracy:
   eclipsing binaries, white-dwarf cooling, lithium depletion, nucleocosmochronometry.
   They are used for validation (§4, option E).
3. *Information the inputs lack.*
   - Main-sequence dwarfs carry almost no age information in XP, photometry and
     parallax. Their predictive age distributions must stay broad.
   - Giants need a mass. XP measures it only through log g, L and Teff, so giant ages from
     XP will be far less precise than seismic ages: expected ≥ 0.15 dex, against the
     0.045 dex seismic ceiling.
   - The gain is in accuracy and calibration, not in precision beyond the data.

## 2. Data audit

### 2.1 Inventory

Counts of *good* records (catalogue quality cuts as in the script) with a Gaia DR3 ID.
"Class" counts use the catalogue's own state where it has one (seismic RGB/RC, StarHorse
MSTO/SGB, FLAME evolstage), else Teff/log g. Full table, with method, models and ID route
per source: `docs/age-audit-20261122/inventory.csv`.

| source (arc path or origin) | method; models and priors | good / with DR3 ID | classes | median age, max | median σ (dex) | notes |
|---|---|---:|---|---|---:|---|
| StarHorse 2023 (`k-pop/catalogues/starhorse_ages/`) | Bayesian isochrone, spectro-photometric-astrometric; PARSEC, Galactic priors | LAMOST-LRS 1,194,769; RVS 650,141; LAMOST-MRS 143,081; GALAH DR3 124,228; RAVE 67,291; APOGEE 61,184; SEGUE 54,714; GES 10,094 | MSTO/SGB only | 4.2–10.5 Gyr; capped at 13.72 | 0.03–0.11 | Queiroz+2023; flags `OutputFlag`, `AgeInOut`; sky fraction 0.33–1.0 |
| APOKASC-3 (`apokasc-3-2024.fits`, SDSS VAC) | seismic scaling masses (10 pipelines) → grid age | 12,056 | RGB 6.7k / RC 5.4k | 4.9; up to 60 Gyr (3.3 % > 13.8) | 0.045 | Kepler field (1.4 % of sky); gold + silver |
| APOKASC-2 (same VAC) | seismic masses → grid age | 6,479 | RGB/RC | 4.5 | 0.075 | Pinsonneault+2018 |
| APO-K2 (`apo_k2_2024*.fits`) | seismic K2 GAP + APOGEE, grid | 4,661 (K2) + 3,209 (Kepler) | giants | 8.3; up to 74 Gyr (16 % > 13.8) | 0.14 / 0.05 | Schonhut-Stasik+2024; its Kepler table tracks APOKASC-2 closely (NMAD 0.030), so it is not independent |
| Willett 2026 (Zenodo 20428851) | PARAM seismic, Δν or L variants | Kepler 7,873; K2-APOGEE BHM 6,095 / COR 5,751; K2-GALAH (L only) 2,222; TESS 1,313 | RGB/RC (Kepler) | 5.6–9.1; capped at about 19.5 | 0.06–0.14 | Gaia ID column constant (bug); matched through KIC (100 % of Kepler), EPIC/2MASS (71–72 % of K2-APOGEE), sobject (94 % of K2-GALAH), 2MASS (44 % of TESS) |
| LEGACY (`kepler-legacy-dwarfs.fits`) | individual-frequency modelling, 6 pipelines | 66 × 6 | dwarfs | 2.8–3.5 | 0.02–0.06 | Silva Aguirre+2017 |
| Chaplin 2014 (`apokasc-1-dwarfs.fits`) | seismic dwarfs/subgiants, grids | 414 | 110 dwarfs, 277 subgiants | 5.3 | 0.061 | identical to VAC `AGE_DW` |
| Xiang, Rix & Liu 2024 (China-VO 10.12149/101467; **downloaded**, 89 MB) | Bayesian isochrone fit of subgiants; Y² isochrones with α | 320,026 | subgiants | 7.4; capped at 20 | 0.033 | the most precise field set; [Fe/H] 5th percentile −0.96 |
| GALAH DR4 (`galah_dr4_allstar_240705.fits`) | isochrone-based age from the DR4 pipeline | 662,922 (`flag_sp == 0`) | 268k dwarfs, 158k subgiants, 237k giants | 6.5 | none quoted | no per-star age error in the main table |
| Leung 2023 (`nn_latent_age_dr17.csv.gz`) | data-driven: spectra → NN, trained on Kepler seismic (APOKASC) ages | 142,257 (129,603 with DR3) | no state in file | 4.8 | 0.12 | APOGEE_ID → 2MASS → DR3; soft floor (1 % below 0.82 Gyr) |
| Martig 2016 (`martig-kepler-preds.fits`) | data-driven [C/N] (The Cannon on APOKASC) | 52,286 (51,876 with DR3) | giants | 4.7; capped at 13 | none | 2.9 % of rows at the cap |
| AMBRE (`ambre-ages.fits`) | isochrone | 366 | | 8.3 | 0.078 | Gaia **DR2** IDs; not cross-matched |
| Gaia DR3 FLAME (archive, queried) | L, Teff → BaSTI **solar-metallicity** tracks; GSP-Phot inputs | 2,811,442 of 4.5M catalogue stars and targets | all | 4.1; capped at 13.5 | 0.062 | 128.6M FLAME ages exist in DR3 |
| Gaia DR3 FLAME-spec (archive) | same, GSP-Spec inputs | 1,099,569 | all | 3.3 | 0.10 | 2.44M in DR3 |
| Hunt & Reffert 2023 (`OpenClusters/`) | cluster CMD ages from a Bayesian NN trained on PARSEC simulations | 7,167 clusters; 574,116 members (p > 0.5) | | | logAge 16/84 | members via `consistency_targets_20261118.npz` (p ≥ 0.7 used) |

The master-label census (`master_labels.py`, `master_label_census_20261113_v2`) holds no
age column: our master labels are Teff, log g, [Fe/H] and [α/Fe] only. Ages entered the
project only through the StarHorse cohorts (`age_feature_cohort_v1`) and the APOKASC-3 /
Willett anchor tables (`build_age_anchor_labels.py`). The October run directories
(`apokasc3_collection_v1`, `age_teacher_anchors_v*`) were not migrated to
`/arc/projects/astroai/msa/runs/` on 2026-11-20. The source catalogues in
`k-pop/catalogues` are intact.

**On arc but not ingested** (no Gaia DR3 route without a further cross-match):

- `lamost-kepler-ages.fits`: 1,348,645 rows of LAMOST RGB/RC neural-network ages, keyed
  by `obsid`; publication not yet confirmed.
- `rave_dr6_bdasp.fits`: 494,695 rows keyed by `rave_obs_id`; the RAVE inputs are already
  covered by StarHorse-RAVE.
- `gc_gaia/`: Vasiliev & Baumgardt globular-cluster members, but no ages.

**Public, not on arc** (not downloaded unless noted; sizes approximate):

| catalogue | content | use |
|---|---|---|
| Cantat-Gaudin et al. 2020, A&A 640, A1 (VizieR J/A+A/640/A1) | about 2,000 clusters with neural-network ages trained on literature isochrone ages; < 1 MB | second cluster-age scale. Today CDS reset connections and the CfA mirror's database was down |
| Dias et al. 2021, MNRAS 504, 356 | about 1,700 clusters, PARSEC isochrone fits; < 1 MB | third cluster-age scale (same blocker) |
| VandenBerg et al. 2013, ApJ 775, 134 | 55 globular-cluster ages | the only old, metal-poor co-natal anchors |
| Gaia DR4 FLAME (2 Dec 2026) | about 1B (`ap_xp`) and 35.8M (`ap_rvs`) | a new source; check whether DR4 keeps the solar-metallicity assumption |
| Anders et al. 2023, A&A 678, A158 | XGBoost APOGEE DR17 giant ages, 178,825 | data-driven, seismically calibrated (shares the APOKASC lineage) |
| Stone-Martinez et al. 2024 (CatBoost, chemical clocks) | about 125k APOGEE giants | same lineage |
| Sanders & Das 2018, MNRAS 481, 4093 | about 3M isochrone ages from spectroscopic surveys | an older isochrone scale |
| Mackereth et al. 2019 (astroNN) | APOGEE Bayesian-NN ages trained on APOKASC-2 | same lineage as Leung/Martig |
| Wu et al. 2019; Wang et al. 2023 | LAMOST RGB/RC ML ages, 0.6–1M | lineage: LAMOST-Kepler seismic |
| El-Badry et al. 2021 white-dwarf + main-sequence wide pairs (the parent table is in the arc consistency tarball) | white-dwarf cooling ages need cooling models plus an initial–final mass relation; published age sets have hundreds of pairs | independent physics for dwarfs (option E) |
| DEBCat (Southworth), cluster eclipsing binaries | a few hundred detached eclipsing binaries; host unreachable today | dynamical masses; validation |
| Gyrochronology (McQuillan+2014: 34,030 periods; Santos+2021) | rotation periods; ages need a calibration tied to clusters | young and intermediate FGK dwarfs only |
| TESS CVZ and K2 GAP seismic samples (Hon+2021; Zinn+2022) | ν_max and Δν without ready ages | a derivation job, out of scope |

### 2.2 Inter-catalogue systematics

All pairs with ≥ 30 shared stars: `pairwise_summary.csv`. Binned by age, [Fe/H], Teff,
log g and evolutionary class: `pairwise_bins_n100.csv`. Plots:
`pairwise_hexbin.png`, `offset_vs_age_feh.png` and `pair_matrix.png` (NMAD and median
offset for 21 representative sources).

| pair (A − B) | n | median (dex) | NMAD (dex) | inflation | dependence |
|---|---:|---:|---:|---:|---|
| APOKASC-3 − Willett-Kepler (Δν) | 7,644 | −0.021 | 0.065 | 0.80 | RGB −0.011 / 0.048; RC −0.032 / 0.080; −0.057 at 12–14 Gyr, +0.047 above 14 Gyr; flat in [Fe/H] down to −1 (only 14 stars below) |
| APOKASC-3 − Willett-Kepler (L) | 7,781 | −0.042 | 0.107 | 1.35 | Δν and L variants of one pipeline differ by NMAD 0.11 |
| APOKASC-2 − APOKASC-3 | 6,240 | −0.024 | 0.078 | 0.88 | **RC −0.103 / 0.051**, RGB +0.005 / 0.036 |
| APO-K2-Kepler − APOKASC-2 | 3,199 | +0.010 | 0.030 | 0.35 | not independent (it tracks APOKASC-2 almost one to one) |
| APO-K2 − Willett-K2 BHM | 2,989 | −0.005 | 0.120 | 0.69 | K2 giants; +0.09 above 14 Gyr |
| Willett-K2 BHM − COR (two seismic pipelines) | 4,902 | −0.018 | 0.099 | 0.61 | with L: NMAD 0.025, so the L constraint dominates |
| APOKASC-3 − Leung 2023 | 5,516 | −0.064 | 0.106 | 1.12 | −0.11 below 4 Gyr (inflation 1.5–2.1); RC NMAD 0.14 vs RGB 0.09 |
| APOKASC-3 − Martig 2016 | 5,398 | +0.011 | 0.144 | | −0.10 below 2 Gyr, +0.06 at 4–10 Gyr: compressed toward the mean |
| StarHorse-LAMOST − Xiang 2024 | 200,301 | −0.008 | 0.088 | 1.31 | flat at 1–12 Gyr (±0.03); −0.11 above 14 Gyr (cap); −0.06 at [Fe/H] > +0.1 |
| GALAH DR4 − StarHorse GALAH DR3 | 100,679 | +0.026 | 0.052 | | +0.003 at 1–2 Gyr rising to +0.050 at 12–14 Gyr |
| StarHorse APOGEE − StarHorse RVS (same pipeline, different spectra) | 25,075 | +0.072 | 0.118 | 1.32 | +0.18 at Teff > 6500 K; input scales leak into age |
| FLAME − StarHorse RVS | 517,148 | −0.022 | 0.161 | 1.49 | **[Fe/H] < −1: −0.27; −1 to −0.5: −0.16; > +0.1: +0.14** (solar-metallicity tracks); MSTO NMAD 0.18, SGB 0.08 |
| APOKASC-3 − FLAME | 5,593 | +0.889 | 0.457 | 6.2 | RGB +0.84, RC +0.93; FLAME giant ages unusable |
| APOKASC-3 − FLAME-spec | 2,097 | +0.228 | 0.254 | 2.8 | better, still biased |
| Chaplin14 − Xiang 2024 (seismic vs isochrone, subgiants) | 110 | +0.065 | 0.102 | 1.80 | +0.03 at 2–4 Gyr, +0.09 at 4–8 Gyr |
| Chaplin14 − StarHorse APOGEE | 307 | +0.031 | 0.110 | 1.63 | |
| LEGACY (any pipeline) − StarHorse APOGEE / RVS | 38–39 | −0.10 to −0.21 | 0.09–0.16 | 1.6–3.6 | StarHorse older; the October "+1.7 to +2.3 Gyr" bias |
| LEGACY pipeline vs pipeline | 66 | −0.06 to +0.07 | 0.04–0.10 | 0.5–1.8 | model-physics spread at fixed data |
| Leung 2023 vs Xiang 2024 or StarHorse-LAMOST (subgiants) | 607 / 542 | +0.03 / 0.00 | 0.34 / 0.39 | | Spearman 0.15–0.18: Leung is outside its training domain |

Reading the table:

- **The precision floor of seismology** for giants is about 0.045 dex per catalogue
  (APOKASC-3 vs Willett, NMAD 0.065 with shared inputs). The scatter of the oldest bin
  is twice that.
- **Model-physics spread at fixed data** (APOKASC-2 → 3 for RC; Δν vs L; the six LEGACY
  pipelines) is 0.03–0.10 dex in offset. It is regime-dependent (RC vs RGB). Any map
  must therefore depend on evolutionary state.
- **Input-scale leakage.** The same pipeline (StarHorse) on different spectra differs by
  0.04–0.07 dex in median and 0.10–0.13 in NMAD, with a strong Teff trend. Isochrone
  ages inherit the Teff/[Fe/H] scale of their inputs. This is the age analogue of the
  survey maps we already fit for [Fe/H].
- **Data-driven ages inherit their training scale** (Leung −0.06 vs APOKASC-3, below its
  APOKASC-2 parent's −0.09) and compress toward the training mean (Martig). They fail
  outside their domain (Leung on subgiants). They are *not* independent of APOKASC for
  the consensus, and must be grouped with APOKASC in the correlation model (§3.3).
- **Truncation and capping differ by source.** Upper edges: StarHorse 13.72 Gyr, Martig
  13.0, FLAME 13.5, Willett about 19.5, Xiang 20.0; APOKASC-3 and APO-K2 are uncapped
  (3–16 % above 13.8 Gyr). These are the oldest stars, the regime the science cares
  about, so censoring must be modelled (§3.4).

### 2.3 Co-natal tests (clusters and wide binaries)

Members of Hunt & Reffert 2023 clusters with p ≥ 0.7 and ≥ 3 stars per catalogue:

- per cluster: `cluster_members_by_catalogue.csv`;
- summary: `cluster_summary.csv`;
- named clusters: `cluster_benchmarks.csv`;
- plots: `cluster_consistency.png`, `cluster_spread.png`.

Wide binaries: El-Badry+2021 MSMS pairs, R_chance < 0.1 (`binary_summary.csv`).

**Benchmark clusters.** Median member age in Gyr (number of members). Approximate
literature ages from eclipsing binaries or lithium depletion are for orientation only and
are used in no computation: NGC 6791 ≈ 8, NGC 6819 ≈ 2.5, M67 ≈ 4, NGC 188 ≈ 6–7,
NGC 6811 ≈ 1.0, Praesepe ≈ 0.7, Pleiades ≈ 0.12 Gyr.

| cluster | HR23 | APOKASC-3 | Willett-K | Leung | StarHorse (LAMOST / APOGEE / GES) | GALAH DR4 | FLAME |
|---|---:|---|---|---|---|---|---|
| NGC 6791 | 3.80 | 8.23 (29) | 8.97 (27) | 8.56 (20) | | | 6.15 (179) |
| NGC 6819 | 1.68 | 2.14 (13) | 2.23 (11) | 2.41 (12) | | | 2.91 (272) |
| M67 (NGC 2682) | 1.69 | | | 3.42 (17) | 5.19 (17) / 4.69 (60) / 6.21 (16) | 5.47 (107) | 6.45 (276) |
| NGC 188 | 2.63 | | | 5.22 (16) | | | 8.41 (252) |
| NGC 6811 | 1.08 | 0.62 (5) | 0.60 (4) | 0.91 (5) | 1.82 (21) | | 1.97 (79) |
| NGC 6866 | 0.63 | 0.52 (3) | | 0.76 (4) | 1.33 (22) | | 3.38 (151) |
| Praesepe (NGC 2632) | 0.35 | | | | 1.87 (5, APOGEE) | 3.22 (67) | 5.33 (170) |
| Pleiades (Melotte 22) | 0.12 | | | | 2.47 (9, APOGEE) | 3.69 (34) | 3.49 (127) |

| test | result |
|---|---|
| HR23 cluster ages | too young for old clusters, by 0.3–0.4 dex (NGC 6791, M67, NGC 188) |
| seismic cluster means | agree with each other within 0.04 dex (NGC 6791, 6819, 6811); APO-K2-Kepler is +0.11 for NGC 6791 |
| field pipelines on young clusters (HR23 < 1 Gyr, CMD class ≥ 0.5) | StarHorse-LAMOST +1.04 dex (146 clusters), GALAH DR4 +1.39 (46), FLAME +1.07 (4,786): the posterior returns the prior for main-sequence dwarfs |
| old clusters (≥ 1 Gyr, CMD class ≥ 0.5), catalogue − HR23 | StarHorse-LAMOST +0.08 (16), StarHorse-GES +0.28 (15), GALAH DR4 +0.26 (13), Leung +0.09 (18), FLAME −0.04 (233); every number includes HR23's own bias |
| within-cluster NMAD / median quoted σ (old / young / all clusters) | StarHorse-LAMOST 2.9 / 4.0 / 3.7, StarHorse-APOGEE 2.3 / 4.5 / 3.3, FLAME 3.1 / 2.6 / 2.4, APOKASC-3 2.0 (3 old clusters), Leung 0.85 / 0.36 / 0.63 |
| wide binaries, per-star scatter (NMAD/√2) | FLAME 0.30 dex (101,294 pairs; inflation 1.05), FLAME-spec 0.22 (7,649), StarHorse-RVS 0.14 (756; inflation 1.49), StarHorse-LAMOST 0.084 (251; 1.78), GALAH DR4 0.13 (142) |

What the co-natal tests say:

1. Quoted errors are 1.5–5× too small for the isochrone pipelines and FLAME. Inflation
   must be learned per source and per regime.
2. Prior-dominated records (young main-sequence members) are the dominant failure. They
   *look* like measurements, with quoted σ of 0.04–0.1 dex.
3. A cluster's age is itself method-dependent. Co-natal **equality** is reliable; a
   cluster **value** is just one more source.
4. The co-natal bridge between giants and turn-off stars currently rests on NGC 6811 and
   NGC 6866, both young and in the prior-dominated regime of the turn-off pipelines. More
   bridges are needed: M67 and NGC 188 giants with seismic or data-driven ages (K2 C5,
   C16 and C18 for M67), turn-off ages in NGC 6791 and 6819, and giant + dwarf wide pairs.
   The MSMS selection used here excludes evolved components.

## 3. The consensus measurement model

> **Rule (user, 22 Nov):** no label is ground truth; every label is fitted.
> - **Anchors** (APOKASC-3; Xiang in the variant) are identity-mapped, with fitted noise.
> - **Every other source** enters through its fitted map and noise.
> - **Clusters** are fitted latents with weak priors from their least model-dependent
>   method.
> - **Wide binaries** are equality constraints on fitted latents.
> - **Observables that are not ages** (ln P_rot, Li equivalent width, activity indices)
>   enter through their own fitted maps in age.

### 3.1 Latent and coordinates

- Each star *i* has a latent age τ_i = ln(age/Gyr), as in `multi_teacher_age.py`. All
  maps are written in τ; reports are in dex.
- The model conditions on a small set of covariates x_i: [Fe/H], evolutionary class,
  Teff and log g. These come from our own common-scale labels where available, else from
  the catalogue. They only parameterise the maps, never the latent.

### 3.2 Measurement maps (one per source, tiered by overlap)

Source *s* (a catalogue variant, e.g. StarHorse-LAMOST, Willett-Kepler-Δν) reports

    y_is = τ_i + δ_s(τ_i, x_i) + ε_is,
    δ_s = b_s + (a_s − 1)(τ_i − τ_0) + c_s · φ(τ_i, x_i),

as in the [Fe/H] survey maps (`AnchoredSurveyMaps`).

- **Tiers by usable overlap.** Overlap counts only stars shared with sources already
  connected to the anchor, and must cover the range of τ, [Fe/H] and class that the
  term describes:
  - identity: the anchor;
  - offset: ≥ 30 overlap stars;
  - offset + slope + class: ≥ 1,000, spanning ≥ 0.6 dex in age;
  - flexible: ≥ 20,000.
  - The audit tables give these counts directly. For example, StarHorse-LAMOST has
    200k overlaps with Xiang, but only about 300 seismic overlaps, through Chaplin.
- **State and metallicity dependence is required,** because the measured systematics are
  regime-specific: APOKASC-2 RC vs RGB; FLAME vs [Fe/H]; StarHorse vs Teff. The class
  enters as separate offset and slope per class (MS/MSTO, SGB, RGB, RC); [Fe/H] enters
  linearly, plus the flexible basis at the top tier.
- **Monotonicity in τ.** The slope a_s is kept above a floor, as `min_slope` does for
  [Fe/H]. Exception: §3.4 lets the slope reach zero in prior-dominated regimes.
- **Torchregress.** `RelativeSurveySplineHead` and `CommonScaleSplineHead`
  (`torchregress.calibration.multisource`) already implement monotone spline maps of a
  common scale. They are the natural flexible tier.

### 3.3 Noise and shared inputs

- **Noise model.** ε_is ~ Student-t_ν with scale² = κ_s,c² σ_is² + ω_s,c², where c is the
  evolutionary class. κ is a learned inflation of the quoted error; ω is a learned floor
  per source and class (it covers GALAH DR4 and Martig, which quote none).
- **Starting values from the audit:** κ of 1.5–4 for the isochrone pipelines and FLAME;
  about 1 for the seismic catalogues.
- **Correlated groups.** Sources that share inputs have correlated errors (inflation < 1):
  - Kepler light curves + APOGEE spectra: APOKASC-2/3, Willett-Kepler, APO-K2-Kepler,
    and Leung/Martig, which were trained on them;
  - the K2 GAP light curves: APO-K2, Willett-K2;
  - the same spectra fed to two isochrone codes (StarHorse-LAMOST and Xiang;
    StarHorse-GALAH and GALAH DR4).
- **How the groups enter.** Each group gets a shared nuisance u_ig ~ N(0, η_g²) added to
  every member's measurement. Without it, the model would count 3 correlated seismic ages
  as 3 independent measurements. The audit's inflation < 1 values (0.35–0.8) are the
  evidence.
- **Torchregress.** `MarginalizedMultiSourceGaussianNLLLoss` already accepts a full
  per-object measurement covariance. The shared-nuisance block is just its off-diagonal.

### 3.4 Posteriors as labels: prior pull, information, censoring

- **Prior pull.** Most catalogues report posterior medians under their own priors. In
  regimes without information the reported value is pulled to the prior, so the map
  slope a_s → 0 there.
- **Information-aware slopes.** The maps therefore let the slope depend on an information
  proxy: the reported σ relative to the catalogue's prior width, and the class. They
  also let the noise floor grow there. A record whose slope is near zero then carries
  almost no weight, which is the right behaviour.
- **What identifies those regimes:** the young-cluster test of §2.3, the seismic dwarfs,
  and the subgiants of Xiang 2024. Without young co-natal groups the model cannot learn
  that the turn-off pipelines are blind below about 1–2 Gyr for dwarfs.
- **Simpler fallback for the first experiment:** exclude StarHorse, GALAH DR4 and FLAME
  records of stars on the main sequence (class MS, or FLAME evolstage < 360), and say so
  in the manifest.
- **Censoring.** Values within 1 % of a source's cap (§2.2) are right-censored, with
  likelihood P(y ≥ cap). Leung's soft floor near 0.8 Gyr is compression (a map slope),
  not censoring.
  `torchregress.losses.CensoredGaussianNLLLoss` covers the Gaussian case. A Student-t
  version with an uploaded per-object variance is the missing generic piece.
- **No cap on τ.** We impose no "age < 13.8 Gyr" constraint, which would be a
  cosmological prior. The fraction of the consensus above 13.8 Gyr is reported as a scale
  diagnostic.

### 3.5 Co-natal terms

- **Clusters.** τ_i = T_c + ζ_i, with ζ_i ~ N(0, s_c²), where s_c is small: 0.01 dex,
  allowing for a short star-formation duration. The cluster age T_c is a free latent per
  cluster.
- **Membership.** Each member enters as a mixture: p_i · co-natal + (1 − p_i) · field, using
  the published membership probability p_i (HR23 `Prob`).
- **Wide binaries.** τ_1 = τ_2 as the same two-member group. Contamination comes in
  through El-Badry's R_chance.
- **Cluster catalogue ages** (HR23 `logAge50`; later Cantat-Gaudin and Dias) enter as
  one more source measuring T_c, with their own map. HR23's old-cluster bias (§2.3)
  needs a slope tier there.
- **Estimation.** Group latents are integrated in closed form for Gaussian pieces, and by
  quadrature or EM otherwise.
- `cluster_coeval_loss` and `LearnableClusterAges` in `multi_teacher_age.py` are the head
  version of this term. The consensus fit (C0) needs the per-star latent version, with
  membership mixtures.

### 3.6 Gauge

- **Why a gauge.** Only relative distortions are identified. Any monotone reparametrisation
  τ → g(τ) can be absorbed by all the maps. The gauge is fixed by identity-mapping one
  anchor source, or one per regime (options in §4).
- **The [Fe/H] lesson applies:** free zero-sum offsets over a reference set were not
  identifiable there, and absorbed a head bias. Anchors are identity-mapped, not
  "centred".
- **An anchor covering one regime fixes only that regime.** Other regimes are tied to it
  only through overlaps and co-natal groups. With today's bridges (§2.3) the subgiant
  scale on an APOKASC-3 gauge is uncertain at about 0.05–0.1 dex, and must be reported
  as such.

### 3.7 Identifiability checks (built into C0)

- **Overlap graph:** sources as nodes, shared stars and co-natal links as edges, split by
  class. Every source must connect to the anchor through edges with enough stars in the
  τ and [Fe/H] range its tier claims. Otherwise its tier is lowered.
- **Simulation-based check:** draw latent ages; generate catalogues with the fitted maps,
  noise, censoring and the real overlap pattern; refit; recover maps and κ.
- **Prior sensitivity:** the population density p(τ | cell) used to integrate the latents
  of single-measurement stars is estimated nonparametrically (NPMLE on a τ grid per cell,
  design note C2). It is never assumed.

## 4. Anchor options (gauge); the user decides

| option | anchor (identity) | pros | cons, from the audit |
|---|---|---|---|
| **A. Seismic giants** (recommended) | APOKASC-3 (gold + silver) | most precise: 0.045 dex per star; realistic quoted σ (inflation 0.8 against Willett with shared inputs); 12k stars with native Gaia IDs; best connected (Willett 7.6k, APOKASC-2 6.2k, Leung 5.5k, Martig 5.4k, APO-K2 3k); cluster means match Willett within 0.04 dex | Kepler field only (1.4 % of sky); few stars below [Fe/H] −1 (14–25 overlaps); no dwarfs or turn-off stars; 3.3 % above 13.8 Gyr; scaling-relation mass scale with model corrections; RC scale moved 0.10 dex between versions |
| B. Cluster scale | curated cluster ages (eclipsing-binary or lithium-depletion dated: NGC 6791, 6819, 188, M67, Pleiades, Hyades/Praesepe...) | one age for many stars in both regimes; all-sky | HR23 ages fail as an anchor (0.3–0.4 dex too young when old); a curated set is about 10–20 clusters, mostly solar [Fe/H]; still isochrone-based; membership contamination |
| C. Isochrone subgiants | Xiang 2024 | 320k stars; realistic precision (NMAD 0.053 against GALAH DR4); the regime with the most age information per star; [Fe/H] to −1; LAMOST sky (0.55) | Y² physics; subgiants only; not connected to giants except through Chaplin (+0.065 dex, n = 110), GALAH DR4 and clusters; 6 % above 13.8 Gyr |
| D. Weighted consensus of a declared reference set (e.g. all seismic catalogues), zero-sum offsets on fixed overlap stars | | uses all seismic information | shared Kepler/APOGEE inputs: averaging does not cancel their common bias; zero-sum gauges failed for [Fe/H]; only identifiable on the overlap stars |
| E. Independent physics: eclipsing binaries, white-dwarf cooling (WD + MS pairs), lithium depletion, Th/U chronometry | | breaks the shared-physics degeneracy | tens to hundreds of objects; not yet on arc; validation, not anchor |

**Recommendation: A with C as a declared scale variant.**

- **Primary solution:** APOKASC-3 identity for giants. Chaplin14 and LEGACY as
  offset-tier sources (not identity: they disagree with each other by 0.06–0.19 dex).
  All isochrone pipelines are mapped, and co-natal groups are used as constraints.
- **Second solution:** the same model with Xiang 2024 as identity in the subgiant regime
  and APOKASC-3 mapped.
- **The difference** between the two, per regime, is the published `Var_scale` term of the
  systematic decomposition in `multi_teacher_age.py`.
- **If the user prefers one scale everywhere,** A alone is the choice. Its subgiant
  uncertainty is then stated, not hidden.

## 5. Predictions for about 2 billion Gaia stars

Three products, as in the data-driven inference note:

- **A (measurement on the consensus scale)** for every labelled star: invert its maps,
  τ̂ = f_s⁻¹(y), with variance J⁻¹(κ²σ² + ω²)J⁻ᵀ, and combine its sources and co-natal
  information. This mirrors `common_scale_labels.py`. It feeds validation and
  deconvolution, and needs no population prior.
- **B (predictive distribution)** for every Gaia star: a frozen-latent head, like S2
  (`scripts/s2_distributional_heads.py`).
  - Inputs: the v10 MSA latent (XP + photometry).
  - Plus a parallax measurement term (as in S1-plx; luminosity is the key age information
    for subgiants).
  - Plus [α/Fe] information, already in the latent; extra photometry where present.
  - Output: a one-dimensional density in τ. Heads: a mixture of 4 Student-t, and a
    rational-quadratic spline flow (torchregress `student_t_mixture`,
    `NormalizingFlowLoss`), each as a K = 5 ensemble.
- **Training loss of B** is the consensus measurement model with the maps frozen from C0:

      L_i = −log ∫ p_θ(τ | x_i) · Π_s p(y_is | τ; map_s, κ_s, ω_s, censoring) · Π_groups (...) dτ

  - It uses Gauss–Hermite or quadrature on τ, as `joint_teacher_age_loss` does.
  - Intervals therefore cover the latent consensus age, not a catalogue's label.
  - It is product B, with a stated reference population: the labelled stars reweighted to
    the parent catalogue by propensity, as in design note §5.
- **Honest width where age is unconstrained.**
  - For main-sequence dwarfs, p_θ(τ | x) should equal the population's age distribution
    for similar stars. That is the correct predictive and must be labelled
    population-dependent.
  - This works only if prior-dominated records are not fed in as precise labels (§3.4).
    Otherwise the head learns false precision. The young-cluster test is the guard.
  - Each star gets an informativeness flag: the ratio of predictive width to the width of
    its cell's population distribution.
- **Calibration.**
  - Split-conformal level sets of −log p (`ConformalizedDensity`).
  - Mondrian strata: evolutionary class (predicted) × G bin × [Fe/H] bin. Coverage is of
    the consensus measurement predictive.
  - Propensity-weighted conformal gives parent-population coverage.
  - The scale systematic (A vs C anchor) is reported separately, not folded into coverage.
- **Kinematics.**
  - The age–velocity relation is strong information for dwarfs, but circular for Milky
    Way structure.
  - B0 uses intrinsic data only. B1 adds reduced proper motion, v_t and (v_R, v_φ, v_z)
    where available, for targeting and as a prediction-powered-inference predictor only.
  - Population studies of age use C1 (rectified B1) or C2 (deconvolution of A), never raw
    B1.
- **Scale of the run.**
  - DR3 XP covers about 220M stars, so the latent exists for them now. DR4 (2 Dec 2026)
    brings XP for about 1.8B.
  - For stars without XP, B is a photometry + astrometry head: broad except for giants
    and turn-off stars with good parallaxes.
  - Summaries (16/50/84 %, informativeness, P(age > 10 Gyr)) at ≥ 10⁵ stars/s per GPU.
    The head is tiny, so densities are recomputed on demand.

## 6. Validation plan and QA plots

**Validation:**

1. **Held-out methods.** Refit C0 without one source family: APOKASC-2, Willett, APO-K2,
   Leung/Martig, Xiang, GALAH DR4. Then predict that source through its map, refitting
   the map on half its stars and testing on the other half. Report bias and NMAD per
   class and [Fe/H] bin. This tests whether the consensus transfers rather than memorises.
2. **Held-out clusters**, assigned by cluster ID. Report the dispersion of held-out
   member predictions against predicted widths (z-scores). Report the agreement between
   the cluster mean and other catalogues' members. Report young-cluster recovery: does B
   give a broad, population-like predictive for young main-sequence members rather than
   3 Gyr ± 0.1?
3. **Wide binaries, held out.** The distribution of (τ₁ − τ₂)/√(σ₁² + σ₂²) should be
   unit normal. This is the only test available for dwarfs. Report it per class and
   G bin.
4. **Seismic stars as the precision ceiling.** APOKASC-3 gold, LEGACY, Chaplin: B's width
   and error against the 0.045 dex seismic floor.
5. **Old and metal-poor slices**, with n: [Fe/H] < −1 and age > 10 Gyr. Below 20 stars a
   slice is descriptive only. Also the fraction of predictions above 13.8 Gyr.

**QA plots the user will see:**

- (now) pairwise hexbins, offsets against age and [Fe/H], the pair matrix, the cluster
  co-natal panels and the spread histograms (`docs/age-audit-20261122/`);
- (C0) the fitted map δ_s(τ) per source and class, with bands;
- (C0) κ and ω per source;
- (C0) consensus minus each catalogue, against age, per class;
- (C0) the overlap graph coloured by tier;
- (C0) the latent cluster ages against HR23 and against the benchmarks;
- (C1) coverage curves per Mondrian stratum;
- (C1) a Kiel diagram coloured by predictive width (informativeness);
- (C1) held-out binary z-score histograms;
- (C1) the cluster sequence: member predictions against colour;
- (C1) old and metal-poor tail tables;
- (C1) age maps in (R, z), from C1/C2 only, never raw B.

## 7. Staged plan and code

| stage | what | inputs | runtime | gate |
|---|---|---|---|---|
| **C0** consensus from labels (first experiment) | per-source maps (tiers), κ/ω, censoring, correlated groups, cluster + binary latents; APOKASC-3 gauge plus the Xiang variant; held-out-method folds | audit `records.parquet` (8.0M records), HR23 members, El-Badry pairs | CPU, about 10–30 min per fit (sparse, full-batch L-BFGS on about 3.6M star latents; star latents profiled in closed form per star) | maps recover synthetic truth; held-out-method bias ≤ 0.03 dex per class; cluster z-scores about N(0, 1) |
| C0b | information-aware slopes, MS exclusions | same | CPU | young-cluster members no longer "measured" at 3 Gyr |
| C1 | 1-D distributional heads on frozen v10 latents (B0); EIV against the C0 measurement model | consensus records with latents (first count the overlap of age stars with the S1/S2 table) | 1 GPU slice, about 20–40 min for K = 5 | beats single-teacher (StarHorse-only) heads in held-out log score; coverage within ±0.03 per stratum |
| C1b | + parallax measurement term; + B1 kinematic variant | | similar | subgiant NMAD against Xiang/seismic |
| C2 | calibration; inference on 220M DR3 XP stars; DR4 later | | about 1–2 h GPU | |

**Code to write:**

- `src/masked_stellar_autoencoder/training/age_consensus.py`. Overlap-graph tiers,
  information-aware maps, the correlated-group covariance and the co-natal group
  likelihood, reusing:
  - `multi_teacher_age.TeacherCalibration` (affine, MLP, stage-affine links);
  - `joint_teacher_age_loss` (shared-latent quadrature);
  - `cluster_coeval_loss`;
  - `systematic_variance_decomposition`.
  `multi_teacher_age.py` (2.5k lines) has never been run on real data, because A1 was
  blocked on the Willett cross-match. This audit removes that blocker.
- `scripts/fit_age_consensus.py`: the C0 driver, folds and QA plots.
- An `age` arm for `scripts/s2_distributional_heads.py`: a one-dimensional τ target with
  the consensus likelihood.
- Tests: synthetic catalogues with known maps, censoring, shared noise and clusters.

**Upstream to `torchregress`** (generic regression maths, per the project rules):

- a Student-t (and censored-t) version of `MarginalizedMultiSourceGaussianNLLLoss` for a
  low-dimensional latent;
- a group (random-effect) shared-latent likelihood with membership mixtures;
- censored likelihoods with per-object variance;
- a Mondrian wrapper for `ConformalizedDensity`, if S2 has not already added one.

**Kept in MSA:** the catalogue readers, ID routes, tier rules, the class and [Fe/H]
design, and the experiments.

## 8. Decisions for the user

1. **Anchor:** A (APOKASC-3), B (curated clusters), C (Xiang 2024), or A + C as
   variants (recommended).
2. **FLAME DR3:**
   - include only dwarfs and subgiants, with a [Fe/H]-dependent map (recommended), or
     exclude it;
   - FLAME giants are excluded in any case;
   - DR4 FLAME is a new source once its priors are documented.
3. **Data-driven catalogues** (Leung, Martig, later Anders/astroNN): include them as an
   APOKASC-lineage correlated group (recommended), or keep them for validation only.
4. **Prior-dominated records:** exclude main-sequence records from the isochrone
   pipelines for C0 (recommended), or rely on information-aware maps from the start.
5. **Acquisitions worth doing** (all small):
   - Cantat-Gaudin 2020 and Dias 2021 cluster ages, and VandenBerg 2013 globular clusters
     (once CDS is reachable);
   - a curated eclipsing-binary and lithium-depletion cluster list;
   - giant + dwarf wide pairs from the full El-Badry table;
   - a LAMOST obsid → Gaia match, to bring in `lamost-kepler-ages.fits`.
6. **Ages above 13.8 Gyr:** report them without capping (recommended), or cap the product
   with an explicit cosmological prior. The latter would be a physics assumption.

## Appendix: reproducing the audit

```bash
# RAW mirrors /arc/projects/k-pop/catalogues (+ willett2026/, Xiang 2024 FITS,
# consistency_targets_20261118.npz from /arc/projects/astroai/msa/runs/).
export PYTHONNOUSERSITE=1
pixi run python scripts/age_label_audit.py ingest  --raw RAW --out OUT
pixi run python scripts/age_label_audit.py gaia    --raw RAW --out OUT   # Gaia archive TAP
pixi run python scripts/age_label_audit.py ingest  --raw RAW --out OUT   # resolve 2MASS, add FLAME
pixi run python scripts/age_label_audit.py compare --raw RAW --out OUT
pixi run python scripts/age_label_audit.py plots   --out OUT --plots docs/age-audit-20261122
pixi run pytest tests/test_age_label_audit.py -q
```

- **Local runs** read column subsets of StarHorse RVS and LAMOST-LRS and of GALAH DR4,
  with the same file names (disk limits). The readers use only those columns.
- **Gaia archive (2026-11-22):**
  - 158,818 2MASS designations resolved through `tmass_psc_xsc_best_neighbour`; only
    single-neighbour matches are kept;
  - FLAME and FLAME-spec queried for 4,525,412 source IDs (all catalogue stars plus
    co-natal targets), with 2.81M FLAME ages returned;
  - astroquery's asynchronous jobs returned exactly 50,000 rows; the script sets MAXREC
    explicitly;
  - the archive goes offline on 25 November for DR4.
- **Records** `sha256` and counts are in `docs/age-audit-20261122/compare_manifest.json`.
