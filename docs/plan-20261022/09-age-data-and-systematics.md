# Age data refresh and correction learning

**Prepared:** 2026-10-22. Research note for the age component of
[TRAINING_PLAN.md](TRAINING_PLAN.md) §6 / [08-losses-age-distance-alpha.md](08-losses-age-distance-alpha.md) §2.
No model or job was launched from this note; it defines the data acquisition, the
systematics model and the experiments to add next.

---

## 0. What the current age labels actually are (verified)

| Item | Current state | Implication |
|---|---|---|
| Training labels | StarHorse posterior ages from `/arc/.../catalogues/starhorse_ages/apogee17.fits` (63,750 rows) and `galahdr3.fits` (127,016 rows); `posterior_labels_v1/v2` built 2026-10-15/16 | Based on **APOGEE DR17 + GALAH DR3** (≈2021–22 compilations), i.e. the "1–2-year-old" ARC compilation the user flagged |
| Age flags | `AgeFlag` ∈ {MSTO, SGB} only (48,293 + 15,457 rows) | StarHorse ages used for **main-sequence turn-off / subgiant** stars; **no giants** in the current age training set |
| Usable cohort | 152,156 selected; **4,440 strict fit**, 105,953 broader fit; 22,830 identity test | Small and regime-limited |
| External check | 50 LEGACY seismic stars: MAE 1.72–2.36 Gyr, **bias +1.68…+2.34 Gyr** across six correlated pipelines | Already a red flag for teacher scale/systematics |
| APOKASC3 | 15,808 giants (12,303 with recommended age), ages from Kepler seismology + APOGEE DR17; local `apokasc3_collection_v1` | **No direct overlap with the current cohort** because it is RGB/RC, not MSTO/SGB (verified: 19 raw and 18 selected overlaps). It is a new **giant** label source, not an external check of the current model |
| In the 10M benchmark | APOKASC3: only **672 / 15,808** IDs in train (0 in validation), because the 10M benchmark is a ~4.5% sample of XP stars | Giant-age labels must be trained on the **150M cache** (or full catalogue), not the 10M benchmark |

**Teacher-dependence is directly measurable.** Matching APOKASC3 and the new
Willett+2026 Kepler catalogue on **10,081 shared KIC stars** (both asteroseismic, different
pipelines/priors):

| Comparison | n | median Δ(APOKASC3 − Willett) | NMAD |
|---|---:|---:|---:|
| `age_Dnu` | 9,608 | −0.22 Gyr | 0.73 Gyr |
| `age_L` | 9,560 | −0.39 Gyr | 1.25 Gyr |
| age 0–4 Gyr | 3,710 | −0.09 | 0.38 |
| age 4–8 Gyr | 3,923 | −0.49 | 1.03 |
| age 8–12 Gyr | 1,630 | −0.35 | 1.30 |
| age 12–20 Gyr | 266 | +0.84 | **3.32** |
| FeH < −1 | 35 | −1.17 | **2.09** |

This is exactly the "non-reported systematics / model- and prior-dependent" problem: two
modern asteroseismic analyses of the same stars differ by up to several Gyr, worst for the
oldest and most metal-poor. No single catalogue can be treated as truth; a correction /
calibration layer is required.

---

## 1. New data researched (with availability)

### 1.1 APOKASC-3 — already on disk, unused for giants

- **Reference:** Pinsonneault et al. 2025, ApJS 276, 69; local `apokasc3_collection_v1/table4.dat`.
- **Content:** 15,808 Kepler evolved stars, 12,303 with recommended ages; 10,036 gold;
  median fractional age uncertainty 11.1%; 10 independent seismic methods; APOGEE DR17
  spectroscopy; asymmetric RGB/RC ages and flags; Gaia DR3 IDs.
- **Use:** new **giant** age labels (RGB/RC/AGB), not independent truth. In the existing
  enriched cohort 8,895 IDs are present; in the 10M benchmark only 672 → train on the 150M
  cache or full catalogue.
- **Caveats:** Kepler field only (north, limited sky); seismic/model/prior dependence;
  RGB vs RC age systematics; selection function not uniform.

### 1.2 Willett et al. 2026 — 17,000 Kepler/K2/TESS asteroseismic ages (new)

- **Source:** "Asteroseismic ages for 17,000 stars in Kepler, K2 and TESS", Zenodo record
  **20428851**, v2, 2026, 44.3 MB, files `cat_Kepler_APG.csv`, `cat_K2_APG_{BHM,COR}.csv`,
  `cat_K2_GAL_BHM.csv`, `cat_TESS_APG.csv`, column PDF.
- **Content:** homogeneous PARAM analysis of >17,000 red giants; Gaia DR3 + APOGEE DR17 +
  GALAH DR3 constraints; `age_Dnu`/`age_16_Dnu`/`age_84_Dnu` and `age_L`/`age_16_L`/`age_84_L`
  (Dnu vs luminosity constraint variants), mass, radius, distance, flags
  (`warn_low_numax`, `warn_high_Dnu`, `warn_high_RUWE`, `warn_plss_err`, `warn_dustmaps`,
  `warn_nve_Av`, `RelAge_Dnu`, `RelAge_L`), plus orbital actions.
- **Why it matters:** adds **TESS and K2** sky coverage beyond Kepler; a second modern
  seismic scale that can be compared with APOKASC3 on shared stars (above).
- **Verified SHA256 of the downloaded files:** Kepler `5e6f04e3…`, K2 APG BHM `d546118d…`,
  K2 APG COR `54c5edc1…`, K2 GAL BHM `6535cb2f…`, TESS `61170dba…`, PDF `0755424d…`.
- **Data issue found:** in the downloaded CSV the `GAIADR3_ID` column is **constant**
  (`1636148068921376768`) for every row of every file, while `KIC_ID`/`TESS_ID`/`APOGEE_ID`
  vary. This is almost certainly a release/writer bug or a placeholder; **do not ingest the
  Gaia ID column blindly**. Crossmatch via `KIC_ID` (Kepler), `TESS_ID` (TIC), `APOGEE_ID`
  (APOGEE allStar / 2MASS) or `ASPCAP_ID`, and record the crossmatch provenance. Flag to the
  authors/check for a corrected v3 before production ingestion.
- **Use:** new giant age labels for Kepler/K2/TESS; second teacher for the hierarchical model;
  candidate independent evaluation layer for APOKASC3 (same stars, different pipeline).

### 1.3 TESS red-giant samples and TESS chemical clocks

- A TESS **Continuous Viewing Zone** sample (~19,000 red giants) with asteroseismic
  parameters exists (identified in the earlier data audit; reference needs re-verification
  before ingestion). `ν_max`/`Δν` + Gaia/spectroscopy can be turned into PARAM/BASTA ages,
  adding southern-sky giants; this is a **derivation job**, not a ready age catalogue.
- TESS chemical-clock calibration: Pakštienė et al. 2026, A&A 708, A250 — 218 giants with
  TESS asteroseismic ages and NLTE [C/N], [Y/Mg]; small but a useful TESS-scale anchor.
- **Use:** southern/young/old coverage; weak-label calibration; not a stand-alone truth
  source.

### 1.4 Chemical clocks for giants — mass-produced weak ages

- **Casali et al. 2026 (MNRAS, arXiv 2609.21928):** [C/N]–age–[Fe/H] calibration from
  **9,122 Kepler giants** with asteroseismic ages + APOGEE DR17; calibrated separately for
  LRGB, URGB and RC; typical precision ~30% for 2–10 Gyr; validated on K2/TESS/open
  clusters; **degrades for URGB/RC, [Fe/H] < −1, <2 Gyr and >10 Gyr**.
- **Reach:** [C/N] (and [Fe/H]) exists for millions of APOGEE/GALAH/LAMOST giants, so this is
  the only route to *bulk* giant ages — but it is a **weak, calibrated proxy**, not an
  independent seismic age, and it is least reliable exactly where the human goal is hardest
  (metal-poor/old).
- Other chemical clocks ([Y/Mg], [Y/Al], [Ba/Al], [Ce/Mg], [Zr/Ti]) are region- and
  metallicity-dependent; **[Ce/Mg] is explicitly not universal** (Dubay et al. 2026). Use
  only with per-regime calibration and as a cross-check, never as truth.
- **Usage in training:** auxiliary weak-label head on APOGEE/GALAH giants, with
  `label_kind="chemical_clock"`, calibrated errors inflated by the published scatter;
  excluded from metal-poor training claims; never used as an evaluation reference.

### 1.5 Open and globular clusters — coeval anchors

- **Hunt & Reffert Gaia DR3 open-cluster catalogue**: 2,384 clusters (2024 version used in
  recent kinematics) / 5,646 in later analyses; ages, memberships, and a published
  **selection function** (Hunt et al. 2026, A&A 706, A341). Coeval populations provide
  **one age for many stars** and bridge stellar regimes (turn-off + giants) — the key to
  connecting the MSTO/SGB StarHorse scale to the giant seismic scale.
- **Globular clusters:** literature ages and Gaia DR3 memberships; small n but irreplaceable
  for the oldest/most metal-poor regime. Use as anchor/evaluation, not bulk training.
- **Caveats:** membership contamination, age–metallicity–reddening degeneracy, selection
  function, isochrone-model dependence; cluster ages are also model-dependent. Use as
  anchors with cluster-level random effects and selection weights.

### 1.6 Gaia DR4 — the step change (2 Dec 2026)

- Announcement: DR4 live **2026-12-02 ~12:00 CET**; content table (last updated
  2026-09-14) includes:
  - `gaia_source`: 2,000,033,276 sources;
  - `ap_rvs`: 35,825,564 rows from GSP-Spec + **FLAME** (Final Luminosity, Age and Mass
    Estimator);
  - `ap_xp`: 999,939,280 rows with GSP-Phot + **FLAME**;
  - `xp_continuous_mean_spectrum`: 1,824,229,813 (22 TB);
  - `dr3_neighbourhood`: 2,448,832,159-row DR3↔DR4 crosswalk (29 GB) — **source IDs are
    explicitly not stable across releases**;
  - total volume ~400 TB.
- **Why it matters:** FLAME ages for ~0.35–1B stars, with published priors/model dependence;
  not truth, but a homogeneous all-sky prior/teacher covering regimes the current labels
  cannot. Plan ingestion as a **new teacher layer** after release, with the DR4 priors
  documented and the DR3↔DR4 crosswalk used for eval continuity.
- **Caution:** FLAME is a model/prior-based estimator; treating it as ground truth would
  reproduce exactly the systematics the human wants corrected.

### 1.7 Surveys without public age catalogues yet

- **GALAH DR4** (~917,000 spectra; 2026) exists, but no public general age VAC was found;
  the 2026 binary paper produces isochrone ages for ~12,911 candidate binaries only
  (selection-biased, not a field-age catalogue).
- **WEAVE** and **4MOST**: no public stellar-age data release yet; they will become
  additional teachers/anchors after DR1.
- **APOKASC-4**: not found; APOKASC-3 remains the latest APOKASC release.

---

## 2. Correction learning design ("learn the systematics while training/predicting")

**Target definition.** Do not call any catalogue truth. Define an **age scale** anchored to
a named reference (recommended: a seismic ensemble = APOKASC3 + Willett/Kepler/TESS + K2,
with clusters as external anchors). Publish ages *on that scale* plus a systematic envelope;
a change of anchor is a new versioned product.

### 2.1 Multi-teacher hierarchical model (train-time)

For star `i`, latent age `t_i` (in `τ=ln(Gyr)`), teacher `s` with measurement `y_{is}`:

```
t_i ~ p_pop(τ | x_i; ψ)                              # learned population prior / flow
y_{is} | t_i, x_i ~ N( a_s(x_i) + b_s(x_i)·t_i ,     # teacher bias + scale
                       σ_s²(x_i) + κ_s² σ_quoted,is² )  # within-teacher + inflation
```

- `x_i` includes Teff, logg, FeH, α, mass/radius (seismic where available), evolutionary
  stage and survey; `a_s,b_s` are smooth (e.g. small MLPs or affine in stage/FeH bins),
  not constants.
- `κ_s` fitted from overlap scatter (e.g. APOKASC3 vs Willett: NMAD 0.73–1.25 Gyr).
- **Identifiability:** teachers must overlap, or be bridged by clusters. Where overlap is
  absent (MSTO/SGB StarHorse vs giant seismic), use **cluster coevality**: for cluster `c`
  with age `A_c` and members `i`, add
  `L_cluster = Σ_i [ (τ̂_i − τ_c)² / (σ_i² + σ_c²) ]` with latent cluster age and a
  cluster random effect; or anchor `a_s,b_s` with cluster ages.
- **Training:** amortized network predicts `p(τ|x)`, while teacher heads model `y_{is}`;
  latent `t_i` integrated by Gauss–Hermite/MC; partial labels; teacher weights/variances
  learned. This is a regression analogue of multi-annotator (Dawid–Skene) models.
- **Selection:** each teacher has its own selection function (parallax quality, Kepler
  field, magnitude); optional importance weights to the target all-Gaia population; report
  weighted and unweighted.

### 2.2 Meta-correction layer (predict-time + few-shot)

A small **calibration module** on top of the frozen base age model:

```
τ_corrected = τ_base + g_φ(τ_base, x, teacher/target-scale)
```

- `g_φ` monotone (e.g. affine plus smooth residual) and conditioned on Teff/logg/FeH/α/
  stage; fit on anchor stars (seismic, clusters, or new survey overlaps) with a small
  calibration set.
- When a new survey (GALAH DR4/WEAVE/4MOST/DR4 FLAME) appears, fit only `g_φ` — no full
  retrain — which is the practical "correction learning as we predict" mechanism.
- At inference, the user can choose the target scale (e.g. `apokasc3`, `willett_tess`,
  `cluster`, `dr4_flame`) and receive the corrected quantiles plus the systematic spread.

### 2.3 Uncertainty decomposition (must be reported)

```
Var(age|x) = Var_model(age|x) + Var_within_teacher + Var_between_teacher + Var_scale
```

- Conformal calibration **within a teacher** does not cover teacher/scale systematics; the
  between-teacher variance (from the hierarchical model and the anchor ensemble) is reported
  as a **systematic envelope**, not as coverage of truth.
- Report `PIT`/coverage per teacher scale and the cross-teacher age dispersion; no claim of
  physical-truth coverage.

### 2.4 Physics/coeval consistency (regularizers, not hard truth)

- Cluster coeval loss and age–mass–logg consistency via seismic mass/radius;
- Age–metallicity–α population prior (weak, learned from the data; no fixed Galactic
  chemical-evolution model as truth);
- Isochrone consistency as a diagnostic, not a training target.

### 2.5 Evaluation protocol

1. **Leave-one-teacher-out**: train without APOKASC3, evaluate its age scale; train without
   Willett/TESS, evaluate; report bias/NMAD per regime.
2. **Cluster holdout**: train without cluster `c`, predict members, compare to cluster age;
   report cluster-level scatter and age-scale offset.
3. **Old/metal-poor slice**: FeH < −2 and age > 10 Gyr with n; no claim if the anchor n is
   below the Poisson limit.
4. **Age-scale shift test**: apply the learned correction from scale A to scale B; report
   offset/scatter; a good correction should reproduce the measured APOKASC3−Willett offsets
   (−0.2 to −0.4 Gyr median, growing with age/FeH).
5. Always report "model/prior-dependent age, not truth"; never average the six LEGACY
   pipelines into a ground truth.

---

## 3. Drift/selection risks

| Risk | Evidence / mitigation |
|---|---|
| Teacher correlation | APOKASC3, Willett and FLAME share stellar models/spectroscopy; not independent. Use leave-one-teacher-out and cluster anchors; report correlation. |
| Age-scale ambiguity | No absolute scale; fix a named anchor and version it; do not mix scales in one table. |
| Cluster selection/membership | Use published selection functions and membership probabilities; cluster-level random effects; report contamination. |
| Old/metal-poor scarcity | APOKASC3/Willett FeH<−1 overlap is tiny (35 stars in the APOKASC3−Willett match); clusters/GCs are the only anchors; no strong tail claim. |
| Release bugs | Willett `GAIADR3_ID` constant → crossmatch required; DR4 source IDs not stable → use `dr3_neighbourhood`. |
| FLAME prior dependence | Treat DR4 FLAME as a teacher/prior, not truth; ingest its flags/priors and validate on clusters. |
| Label leakage | Anchor IDs (APOKASC3/Willett/clusters/LEGACY) held out of fit/tune/cal; one global source-ID split. |

---

## 4. Experiments to add (after the current W1/W2 jobs)

| ID | Experiment | Data | Head/loss | Budget | Gate |
|---|---|---|---|---|---|
| **A0** | Age data audit + crossmatch | APOKASC3 (local), Willett (44 MB), clusters, current StarHorse | CPU only | CPU hours | SHA/columns/ID crossmatch report; teacher overlap matrix; no labels on test |
| **A1** | Multi-teacher giant-age training | StarHorse MSTO/SGB + APOKASC3 + Willett; 150M cache (giant coverage) | hierarchical multi-teacher + cluster coeval; log-age quantiles | 20M–150M presentations | leave-one-teacher-out; cluster holdout; bias/NMAD |
| **A2** | Meta-correction layer | base age model + APOKASC3/Willett/cluster anchors | affine/smooth monotone correction; few-shot | head-only adaptation | correction reproduces measured teacher offsets; no regression on anchors |
| **A3** | Cluster anchors | Hunt & Reffert clusters + GC memberships/ages | coeval loss / anchor calibration | CPU + head-only | cluster-level scatter and selection report |
| **A4** | [C/N] weak-label giant ages | APOGEE/GALAH giants with [C/N], [Fe/H] | weak-label NLL with inflated errors; LRGB first | head-only on 150M | calibration vs seismic on overlap; exclude FeH<−1 claims |
| **A5** | Uncertainty/systematics envelope | multi-teacher model | between-teacher variance + conformal within teacher | analysis | report decomposition; no truth-coverage claim |
| **A6** | Gaia DR4 FLAME ingestion | `ap_rvs` 35.8M + `ap_xp` 1B; crosswalk | new teacher layer + meta-correction | CPU/GPU 2027 | DR4 schema/prior audit; DR3 continuity via `dr3_neighbourhood` |

A1/A2 are the core; A4 is the only bulk route but weak and not metal-poor; A6 is the long-term
step change.

---

## 5. Immediate actions

1. **Ingest APOKASC3 as a new label layer** (local; 8,895 enriched/672 benchmark; use the
   150M cache for giant training), with its flags and asymmetric errors, and keep it out of
   any "truth" field.
2. **Acquire Willett 2026** (44 MB, SHA recorded above), fix/crossmatch the `GAIADR3_ID`
   column via KIC/TESS/APOGEE IDs, and audit its flags before training.
3. **Build the teacher/overlap matrix** (StarHorse MSTO/SGB vs APOKASC3 vs Willett vs clusters)
   — this A0 CPU step determines whether a bridge exists without clusters.
4. **Add cluster anchors** (Hunt & Reffert memberships/ages + GCs) as the bridge between
   MSTO/SGB and giant scales.
5. **Implement the multi-teacher hierarchical loss + meta-correction module** (A1/A2) in the
   age head; keep the current StarHorse-only model as the control.
6. **Schedule A0 now** (CPU, no GPU) and A1 after the current W1/W2 screen; **DR4 ingestion**
   after 2 Dec 2026.
7. Report age as **model/prior-dependent on a named scale**; never as truth.
