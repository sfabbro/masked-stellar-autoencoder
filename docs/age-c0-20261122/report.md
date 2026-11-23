# C0: age consensus from the labels alone (2026-11-22)

Status: first fit of the consensus measurement model of `docs/design-20261122-age-consensus.md` (§7, stage C0). There is no network and no stellar or Galactic model inside the inference.
- Code: `src/masked_stellar_autoencoder/training/age_consensus.py` (model and fit), `scripts/age_consensus_c0.py` (data, configurations, folds, tables, plots), `tests/test_age_consensus_c0.py`.
- Units: the latent is t = log10(age/Gyr). Offsets b are the map value minus t at 4 Gyr (t = 0.6); A is the slope minus 1; β is the [Fe/H] coefficient. All in dex.

## 1. Summary
- **Scale of the fit.** 7,456,035 good records with a DR3 ID cover 3,608,454 stars. After the exclusions in §3, 1.67M representative records from the fitted catalogues remain. 271,190 stars (543,837 records) have at least two independent constraints and fix the maps and the noise. Every star with a fitted record (1,163,699) gets a consensus.
- **Within each regime, the relative maps are well determined.**
  - Held-out residual NMAD after the map: 0.035–0.077 dex for Kepler seismic catalogues, 0.046–0.10 for isochrone catalogues, 0.11–0.17 for seismic dwarfs.
  - Mapped bias ≤ 0.006 dex, except LEGACY (n = 31) and Chaplin.
  - Validation-only catalogues against the consensus: Leung 2023 z-NMAD 1.11; Martig 2016 NMAD 0.150 → 0.119 after its map; GALAH DR4 giants 0.207 → 0.162.
- **Noise.**
  - Xiang's quoted errors are 1.8× too small; StarHorse-APOGEE's 1.7–2.0×.
  - The isochrone pipelines share a per-star age error, η = 0.123 ± 0.002 dex, learned from co-natal and seismic constraints. It cannot be averaged away and floors the consensus σ of isochrone-only stars (median 0.127 dex).
- **Cross-regime offset (APOKASC-3 giants vs Xiang subgiants): −0.026 ± 0.002 dex statistical, but not robust.**
  - Bridge variants range from −0.04 to −0.01; GALAH continuity gives −0.13.
  - The Xiang-anchored solution implies +0.25 to +0.27 for giants instead of −0.026.
  - Leaving out one subgiant catalogue moves the zero point by −0.27 to +0.36.
  - The scale systematic between the two anchors is therefore at least 0.25 dex and is set by the bridge, not by the anchors.
- **Young clusters remain prior-dominated.** Fitted latents: Pleiades 2.75 Gyr, Praesepe 2.1 Gyr, NGC 6866 1.6 Gyr. Old clusters look reasonable: NGC 6791 8.2 Gyr, M67 5.2, NGC 188 5.4, NGC 6819 2.1.

## 2. How each source enters (no label is truth)

| source | role | likelihood |
|---|---|---|
| APOKASC-3 (gold + silver) | gauge anchor (A) | identity map with **fitted noise**: Student-t (ν = 5), scale² = κ²σ_quoted² + ω² per class (RGB, RC) |
| Xiang, Rix & Liu 2024 | anchor of C; offset map in A | identity (C) or offset (A); fitted noise and isochrone shared error in both |
| APOKASC-2, APO-K2-Kepler | APOKASC shared-input group | one record per star and group (APOKASC-3 first); the rest are validation records |
| APO-K2 (K2) | mapped | map and noise |
| Willett 2026 (Kepler, K2 BHM/COR, K2-GALAH, TESS; Δν and L) | mapped | one map per variant shared across missions (one pipeline, one grid); one record per star (Δν first) |
| LEGACY (6 pipelines) | mapped (offset) | BASTA in the fit; the other five are validation records |
| Chaplin 2014 | mapped (offset); identity in variant V_seismic_dwarfs | map and noise |
| GALAH DR4 subgiants | mapped | map and noise (no quoted error: ω only) |
| GALAH DR4 giants | validation (main); in the fit in variants | see §4 |
| StarHorse 2023 (8 inputs) | mapped, one map per input survey | one record per star across surveys (APOGEE, GALAH, GES, LAMOST-MRS, LAMOST-LRS, SDSS, RAVE, RVS) |
| isochrone family (StarHorse, Xiang, GALAH DR4) | shared age error | y = f(t + u), u ~ N(0, η²) per star; cancels inside the family, so learned from independent constraints only |
| HR23 open and globular cluster members (p > 0.5) | co-natal | cluster age = **fitted latent** (flat prior); t_i ~ N(T_g, 0.01²) through a membership mixture (prior = HR23 p, responsibility against a field density) |
| El-Badry+21 MSMS wide binaries (R < 0.1) | co-natal | equality of two fitted latents (0.01 dex spread), prior 0.9 |
| HR23 and literature cluster ages | not in the fit | shown beside the fitted latents for comparison only |
| Leung 2023, Martig 2016 | validation only | map fitted on half their stars against the consensus; residuals on the other half |
| Gaia FLAME / FLAME-spec | excluded | kept for a later test |

## 3. Data and decisions
- **Main-sequence records excluded.** One rule for the isochrone pipelines: the catalogue's own log g ≥ 4.1 (the audit's dwarf class). StarHorse "MSTO" reaches log g 4.33. Excluded records:

  | catalogue | records excluded |
  |---|---:|
  | StarHorse LAMOST-LRS | 677,432 |
  | StarHorse RVS | 295,991 |
  | GALAH DR4 | 267,687 |
  | StarHorse LAMOST-MRS | 91,249 |
  | StarHorse GALAH | 57,148 |
  | StarHorse SDSS | 38,006 |
  | StarHorse APOGEE | 28,305 |
  | StarHorse RAVE | 25,974 |
  | Xiang | 14,484 |
  | StarHorse GES | 4,810 |

- **FLAME** (3.91M records) excluded.
- **Censoring.** Records within 1 % of a cap are right-censored: StarHorse 13.72 Gyr, GALAH DR4 14.79, Willett 19.5, Xiang 20. There is no 13.8 Gyr cap; 4.0 % of the consensus lies above 13.8 Gyr.
- **Shared inputs.** Correlated catalogues enter through one representative record per star and group, as in the [Fe/H] `spectra_groups`. The other records are validation records. On top of that, the isochrone family carries a shared age error.

## 4. Model and estimation
- **Measurement maps by tier.** One map per catalogue (Willett: per variant).
  - Tiers by independent overlap: identity; offset (≥ 30 stars); linear plus per-class offset and slope plus [Fe/H] (≥ 1,000 stars spanning ≥ 0.6 dex in age); flexible (≥ 20,000 stars), adding u², x·u and x².
  - Polynomial and [Fe/H] terms are clamped to the data range of each map.
  - Components linked by fewer than 1,000 shared stars get a reference map limited to an offset. This applies to Xiang in gauge A and to APOKASC-3 in gauge C, because the thin bridges cannot support slope or [Fe/H] terms.
- **Estimation.**
  - Maps: profile (errors-in-variables) objective, with latents, shared errors and group ages re-solved exactly at every evaluation (envelope gradient, verified by finite differences); L-BFGS.
  - Noise (κ, ω, η): restricted likelihood (latents and group ages integrated under flat priors).
  - Record-level robustness: Student-t EM weights computed against each star's own records.
  - Censoring: EM imputation.
  - Co-natal: responsibilities against a field density.
- **Per-star consensus.** Flat prior, with censored likelihoods; a star with only capped records gets a lower limit.
- **Extending the model.** A new label type, including non-age observables, is a new `MapFamily` (the `MonotoneMaps` example is tested) or a new source of an existing family.

## 5. Identifiability
### 5.1 Structure
- APOKASC-3 shares stars with no non-Kepler catalogue except about 19 stars.
- In gauge A the components are:
  - the seismic component {APOKASC-3, Willett Δν (all missions), APO-K2};
  - the subgiant component {Xiang, StarHorse LAMOST-LRS, LAMOST-MRS, RVS, APOGEE and GALAH, GALAH DR4 subgiants};
  - small own components with offset references (Chaplin, LEGACY, StarHorse GES/RAVE/SDSS, APOKASC-2-only, Willett-L).
- The seismic–subgiant bridge is co-natal only: 1,893 member stars in six clusters (M67, NGC 6791, 6819, 6811, 6866, UBC 143).

### 5.2 Bridge variants
Xiang minus the APOKASC-3 latent, [Fe/H] = 0, in dex (`bridge_variants.csv`):

| variant | 1 Gyr | 2 Gyr | 4 Gyr | 8 Gyr | 12.6 Gyr |
|---|---|---|---|---|---|
| **main** (co-natal + Willett mission continuity) | −0.026 | −0.026 | −0.026 ± 0.002 | −0.026 | −0.026 |
| co-natal only | −0.037 | −0.037 | −0.037 ± 0.002 | −0.037 | −0.037 |
| seismic dwarfs (Chaplin identity) | −0.036 | −0.036 | −0.036 ± 0.002 | −0.036 | −0.036 |
| GALAH giants in the fit (warm start) | −0.024 | −0.024 | −0.024 ± 0.002 | −0.024 | −0.024 |
| affine bridge | +0.039 | +0.016 | −0.007 ± 0.003 | −0.029 | −0.045 |
| GALAH continuity (one GALAH map for giants and subgiants) | −0.19 | −0.16 | −0.125 ± 0.004 | −0.14 | −0.17 |
| continuity only (no co-natal terms; not converged) | +0.21 | +0.11 | +0.01 | −0.10 | −0.18 |

With GALAH giants in the fit and started from scratch, an earlier fit drove Xiang to −0.39. Their map, calibrated on K2 giants, extrapolates badly to globular-cluster giants (NGC 6397, 6656, 6752, 5904). This is why they are validation records in the main fit.

### 5.3 Gauge C and leave-one-out
**Gauge C (Xiang anchor)**
- APOKASC-3 maps at −0.272 ± 0.013. Willett Δν (−0.263) and APO-K2 (−0.264) follow.
- Refitted from the gauge-A solution, it returns to −0.251 with essentially equal likelihood (`gauge_mode_check.json`: −2 ln L_REML −1,241,864 vs −1,240,535).
- The two anchored fits are not exact reparametrisations of each other:
  - Gauge A rejects the prior-dominated turn-off members of NGC 6811 (responsibility < 0.5 for about half of its members) and keeps NGC 6791 at 8 Gyr.
  - Gauge C accepts them and shifts the giants about 0.27 dex older, which puts NGC 6791 at about 15 Gyr.
  - The membership mixture's field density is evaluated in each gauge's own latent frame.

**Leave-one-out.** The raw bias of a held-out subgiant catalogue against the refitted consensus shows how far the subgiant zero point moved:

| held out | raw bias vs refitted consensus | main-fit map offset |
|---|---:|---:|
| GALAH DR4 subgiants | −0.41 | −0.05 |
| StarHorse-GALAH | +0.19 | −0.08 |
| Xiang | +0.10 | −0.03 |
| StarHorse-LAMOST | −0.11 | −0.02 |

### 5.4 Conclusion
The relative maps inside each regime are robust. The zero point that ties subgiants to the giant anchor is uncertain at the ±0.3 dex level.

## 6. Maps (gauge A; `maps_A.csv`)

| map | tier | n constrained | b | A | β | class terms (Δb, ΔA) | flexible |
|---|---|---:|---|---|---|---|---|
| APOKASC-3 | identity | 7,850 | | | | | |
| Willett Δν (Kepler + K2 + TESS) | linear | 10,971 | +0.010 ± 0.001 | +0.005 ± 0.002 | −0.037 ± 0.003 | | |
| Willett L (L-only stars) | offset | 362 | +0.284 ± 0.010 | | | | |
| APO-K2 (K2) | linear | 3,420 | −0.061 ± 0.005 | +0.253 ± 0.013 | −0.010 ± 0.009 | | |
| APOKASC-2 (A2-only stars) | offset | 87 | −0.053 ± 0.006 | | | | |
| Chaplin 2014 | offset | 301 | −0.002 ± 0.008 | | | | |
| LEGACY (BASTA) | offset | 55 | −0.086 ± 0.017 | | | | |
| Xiang 2024 | offset (bridge) | 196,769 | −0.026 ± 0.002 | | | | |
| GALAH DR4 subgiants | flex | 65,032 | −0.048 ± 0.002 | +0.100 ± 0.004 | −0.272 ± 0.002 | | u² −0.15, xu +0.15, x² −0.04 |
| StarHorse LAMOST-LRS | flex | 174,900 | −0.024 ± 0.002 | −0.036 ± 0.002 | −0.184 ± 0.002 | MSTO −0.016, +0.183 | u² −0.34, xu +0.16, x² −0.04 |
| StarHorse GALAH | flex | 48,428 | −0.080 ± 0.002 | +0.136 ± 0.004 | −0.368 ± 0.003 | SGB +0.007, +0.001 | u² −0.10, xu +0.27, x² −0.10 |
| StarHorse RVS | linear | 14,765 | −0.086 ± 0.002 | +0.014 ± 0.004 | −0.235 ± 0.004 | MSTO −0.063, +0.152 | |
| StarHorse APOGEE | linear | 9,788 | −0.045 ± 0.002 | +0.064 ± 0.004 | −0.160 ± 0.003 | MSTO +0.004, −0.007 | |
| StarHorse LAMOST-MRS | linear | 9,283 | −0.106 ± 0.003 | +0.291 ± 0.004 | +0.021 ± 0.004 | | |
| StarHorse RAVE / SDSS / GES | offset | 832 / 534 / 460 | −0.074 / −0.074 / −0.045 | | | | |

The gauge-C maps of the subgiant catalogues equal these plus about 0.026. The giant catalogues are −0.26 to −0.32 (`maps_C.csv`).

## 7. Noise (gauge A; `noise_A.csv`)

| unit | median quoted σ | κ | ω | scale at median σ | effective inflation |
|---|---:|---|---|---:|---:|
| APOKASC-3 RGB / RC | 0.051 / 0.040 | 0.32 ± 0.02 / 0.35 ± 0.03 | 0.005 / 0.014 | 0.017 / 0.019 | 0.34 / 0.49 |
| Willett-Kepler RGB / RC | 0.058 / 0.078 | 0.64 / 0.52 | 0.005 / 0.062 | 0.037 / 0.074 | 0.64 / 0.94 |
| Willett K2 BHM / COR | 0.106 / 0.112 | 0.47 / 0.23 | 0.082 / 0.074 | 0.095 / 0.078 | 0.90 / 0.70 |
| APO-K2 | 0.137 | 0.33 | 0.005 | 0.046 | 0.34 |
| Chaplin SGB | 0.052 | 0.30 | 0.012 | 0.020 | 0.38 |
| LEGACY | 0.059 | 0.63 | 0.080 | 0.088 | 1.49 |
| Xiang | 0.032 | 1.00 ± 0.01 | 0.046 | 0.056 | 1.76 |
| StarHorse APOGEE MSTO / SGB | 0.028 / 0.025 | 0.73 / 1.15 | 0.054 / 0.032 | 0.057 / 0.043 | 2.04 / 1.70 |
| StarHorse LAMOST-LRS SGB / MSTO | 0.072 / 0.039 | 0.70 / 0.20 (bound) | 0.054 / 0.005 | 0.074 / 0.009 | 1.02 / 0.24 |
| StarHorse RVS MSTO / SGB | 0.085 / 0.048 | 0.33 / 0.45 | 0.070 / 0.049 | 0.075 / 0.054 | 0.89 / 1.13 |
| StarHorse GALAH MSTO / SGB | 0.060 / 0.040 | 0.39 / 0.20 (bound) | 0.005 | 0.024 / 0.009 | 0.40 / 0.24 |
| GALAH DR4 subgiants | none | | 0.026 | 0.026 | |
| **isochrone shared age error η** | | | **0.123 ± 0.002** | | |

Units at the bounds (StarHorse-LAMOST MSTO, StarHorse-GALAH, StarHorse-SDSS) are catalogues that share spectra with another fitted catalogue: their agreement is not independent. APOKASC-3's scale is probably underestimated for the same reason (light curves and spectra shared with Willett). No seismic shared error was identifiable.

## 8. Leave-one-catalogue-out (`loo_predictive.csv`, `c0_loo.png`)
Refit without the catalogue; map refitted on half its stars against that consensus; residuals on the other half (dex).

| held out | n test | raw bias / NMAD | mapped bias / NMAD | z-NMAD |
|---|---:|---|---|---:|
| APO-K2-Kepler | 1,534 | +0.018 / 0.049 | −0.001 / 0.035 | 0.86 |
| APOKASC-2 (RGB / RC) | 1,797 / 1,359 | | +0.009 / 0.034; −0.097 / 0.052 | 0.81 / 0.43 |
| Willett-Kepler (RGB / RC) | 1,620 / 2,147 | | −0.009 / 0.046; +0.014 / 0.080 | (APOKASC-3 noise unidentified) |
| APO-K2 | 1,675 | +0.008 / 0.125 | +0.030 / 0.198 | 0.50 |
| Chaplin | 146 | +0.008 / 0.114 | +0.026 / 0.114 | 0.75 |
| LEGACY (6 pipelines) | 31 | −0.07 to −0.23 / 0.14–0.17 | −0.05 to +0.04 / 0.14–0.17 | 1.3–1.7 |
| Xiang | 97,548 | +0.103 / 0.090 | −0.001 / 0.090 | 0.56 |
| GALAH DR4 subgiants | 32,302 | −0.413 / 0.053 | −0.001 / 0.046 | 0.24 |
| StarHorse LAMOST-LRS | 105,283 | −0.111 / 0.094 | −0.001 / 0.095 | 0.67 |
| StarHorse RVS | 47,467 | −0.088 / 0.098 | +0.004 / 0.097 | 0.58 |
| StarHorse GALAH | 26,446 | +0.191 / 0.061 | +0.000 / 0.044 | 0.29 |
| StarHorse APOGEE | 9,789 | −0.027 / 0.080 | −0.002 / 0.073 | 0.53 |
| StarHorse LAMOST-MRS | 12,973 | −0.066 / 0.094 | +0.006 / 0.086 | 0.52 |
| StarHorse RAVE / SDSS | 8,950 / 682 | −0.10 / 0.090; −0.09 / 0.069 | +0.001 / 0.090; +0.003 / 0.069 | 0.65 / 0.49 |

Reading the table:
- The raw biases of the subgiant catalogues measure how far the zero point moved in each refit (§5.3).
- z-NMAD < 1 for isochrone catalogues is expected: the shared error is common to the held-out record and to the consensus from other isochrone catalogues.
- With Willett held out, the K2 block disconnects (no K2 predictions) and APOKASC-3's noise becomes unidentifiable.

## 9. Validation-only records (`validation_records.csv`)
Residuals against the gauge-A records-only consensus (dex):

| catalogue | n test | raw bias / NMAD | mapped bias / NMAD | z-NMAD |
|---|---:|---|---|---:|
| Leung 2023 | 6,877 | −0.007 / 0.173 | +0.001 / 0.169 | 1.11 |
| Martig 2016 | 3,052 | −0.015 / 0.150 | +0.007 / 0.119 | 1.11 |
| GALAH DR4 giants | 1,902 | −0.006 / 0.207 | +0.016 / 0.162 | 0.99 |
| APOKASC-2 (on APOKASC-3 stars, same group) | 3,117 | −0.022 / 0.078 | −0.000 / 0.038 | 0.96 |
| Willett-Kepler L | 3,888 | +0.041 / 0.107 | +0.000 / 0.102 | 1.10 |
| Willett K2 COR | 1,772 | +0.036 / 0.096 | +0.000 / 0.091 | 0.92 |
| StarHorse secondary surveys (same group) | 360–40,208 | −0.04 to −0.11 / 0.07–0.10 | ≈ 0 / 0.04–0.10 | 0.28–0.68 |

## 10. Co-natal consistency (`clusters_A.csv`, `c0_clusters.png`)
- **Member spread** over 80 clusters with ≥ 5 members: raw catalogue values 0.093 dex; mapped records-only consensus 0.081 (0.070 for members with responsibility > 0.5).
- **Member z-scores** against the other members: NMAD 0.76 for open clusters (1,526 members) and 0.65 for wide binaries (643). The consensus σ is conservative for co-natal stars.
- **Fitted cluster latents** (gauge A; HR23 and literature ages are comparisons, not inputs):

  | cluster | members (resp. > 0.5) | fitted latent (age) | raw member median | HR23 | literature |
  |---|---|---|---|---|---|
  | NGC 6791 | 58 (38) | 0.912 ± 0.004 (8.2 Gyr) | 0.927 | 3.8 | ≈ 8 |
  | M67 | 115 (110) | 0.713 ± 0.012 (5.2) | 0.661 | 1.7 | ≈ 4 |
  | NGC 188 | 5 | 0.731 ± 0.062 (5.4) | 0.778 | 2.6 | ≈ 6.5 |
  | NGC 6819 | 34 (22) | 0.322 ± 0.004 (2.1) | 0.310 | 1.7 | ≈ 2.5 |
  | NGC 6811 | 25 (11) | −0.077 ± 0.010 (0.84) | −0.002 | 1.1 | ≈ 1.0 |
  | NGC 6866 | 18 (15) | 0.209 ± 0.032 (1.6) | 0.010 | 0.63 | |
  | Pleiades | 8 | 0.439 ± 0.052 (2.75) | 0.385 | 0.12 | ≈ 0.12 |
  | Praesepe | 2 | 0.323 ± 0.10 (2.1) | 0.247 | 0.35 | ≈ 0.7 |
  | NGC 6752 (globular) | 26 | 1.112 ± 0.026 (12.9) | 1.090 | | |

- The young clusters remain prior-dominated, as the audit found.

## 11. Scale systematic between the anchors (`c0_scale_difference.png`)
t(Xiang gauge) − t(APOKASC-3 gauge) is −0.026 for MSTO, SGB and MS stars and +0.272 for RGB, RC and K2 giants. It is flat in age and [Fe/H]. Started from the APOKASC-3 solution, the Xiang gauge gives −0.026 and +0.251. In both cases the difference is set by the inconsistent co-natal bridge (§5.3), not by a genuine age or [Fe/H] dependence of the scale.

## 12. What failed, and what to fix next (C0b)
1. **Cross-regime zero point.** Fix the gauge dependence of the membership mixture (field density per component, or fixed memberships). Down-weight or exclude prior-dominated turn-off ages in the bridge. Add bridges: benchmark clusters, evolved + dwarf wide pairs, white-dwarf + main-sequence pairs.
2. **Shared-spectra pairs.** StarHorse with Xiang and StarHorse with GALAH DR4 need their own shared effect, or one-record grouping.
3. **Information-aware maps for turn-off stars** (design §3.4); young-cluster members are still "measured" at 2–3 Gyr.
4. **Bugs found and fixed in this session:**
   - The profile gradient ignored the dependence of c on the slopes, so the optimiser stalled early. This invalidated the first WIP snapshot tables; a finite-difference test now guards it.
   - The shared error was first written in measurement space, where it was spuriously identifiable; it now enters the age as f(t + u).
   - Student-t weights were judged against the co-natal latent, which caused a runaway.
   - The `predict_heldout` unpacking crash.
5. **Statistical errors** come from the Hessian at fixed noise and responsibilities; some are NaN (clamped terms). Willett TESS noise is unconstrained.
6. **Convergence.** The continuity-only variant (no co-natal terms) stopped at its 12-iteration cap. The Xiang leave-one-out refit ended at a step of 0.004.

## 13. Reproduction
```bash
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4
pixi run python scripts/age_consensus_c0.py run   --records OUT/records.parquet --targets consistency_targets_20261118.npz --work WORK --docs docs/age-c0-20261122
pixi run python scripts/age_consensus_c0.py plots --records OUT/records.parquet --targets consistency_targets_20261118.npz --clusters OpenClusters/clusters.fits --work WORK --docs docs/age-c0-20261122
pixi run pytest tests/test_age_consensus_c0.py -q
```
Runtime: 6,532 s for `run` on 4 CPU threads (two gauges, six variants, 13 leave-one-out refits), plus about 3 min for `plots`. Large outputs go to WORK: `consensus_stars.parquet` (both gauges, number and names of contributing catalogues per star) and `records_c0_A.parquet` (per-record C1 input). The gauge mode check was run with a scratch script (`mode_check.py`, not committed).
