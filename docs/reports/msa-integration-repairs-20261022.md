# Integration repairs — 2026-10-22

## Implemented

- **Conformal validity:** calibration-selected gamma is explicitly empirical, not finite-sample exact. `fit_with_tuning` selects on disjoint tuning identities and calibrates the frozen score on separate calibration objects. Fixed-gamma arithmetic and conditional-on-exchangeability validity remain explicit.
- **Age likelihood:** new `joint_teacher_age_loss` integrates the shared Gaussian latent once, including off-diagonal teacher covariance. `MultiTeacherAgeHead.loss(objective="joint")` exposes it; the historical composite default retains its numerical meaning and is labelled correctly. Joint teacher weighting/quality constraints are explicit rather than silently inventing a likelihood.
- **Comparisons:** ordered source-ID/shape checks, unique catalogue joins, identical finite heldout subsets for both systems, and no clipped unattainable conformal rank. [M/H] versus [Fe/H] remains a target-proxy comparison. Similar MAEs alone no longer imply statistical indistinguishability.
- **Actual anchor bug fixed:** APOKASC3 `E_AgeRGB/RC` are upper uncertainty offsets and `e_AgeRGB/RC` are signed lower offsets, according to the local published ReadMe. Old v1 treated them as bounds in reverse order. Versioned v2 computes median plus offsets, masks sentinels and invalid/nonpositive bounds: 12,301 valid triplets among 15,808 objects. Old artifacts retained.
- **Runner integration:** bounded age and signed-parallax distance pilots now call the actual heads, with frozen-input/source/units/SHA contracts, FIT-only gradients and TUNE-only selection. Synthetic tests prove TEST target perturbations cannot influence selected predictions or history.
- **Native pilot:** source-bound parent86 photometry+XP embeddings for 384 real objects (256 FIT,64 TUNE,64 TEST); 102 StarHorse and282 APOKASC labels, zero shared labelled objects in this bounded sample. Two joint-age head updates completed and selected state saved. This checks native plumbing, **not** cross-domain age-scale identifiability or scientific improvement. All legacy seismic reference identities excluded. Teacher features are unused under constant affine links.
- **Density repair tooling:** versioned full-role FP64 fixed-checkpoint rescoring entrypoint, immutable source/artifact/library/implementation policy and exclusive new outputs. This prepares corrected evaluation without SGD or replacing failed original outputs.

## Evidence and limits

- Focused integration/statistical repair tests: 67 passed before additional rescoring/crossmatch tests.
- Current torchregress TARFlow/public-API/CDF/density tests: 23 passed. Substantial unrelated uncommitted library work was preserved; this is not a full-library release certificate or automatic MSA repin.
- Readiness audit: `experiments/xp/results-stellar-improvement/target-integration-audit-20261022-v1/readiness.json`.
- Native pilot: `experiments/xp/results-stellar-improvement/joint-age-native-pilot-v1/manifest.json` and bound preparation manifests.
- Required full-suite results and final publication are recorded in the durable progress log after completion.

## Gates that remain real, not waived

1. Three APOKASC objects occur in the old age tuning cohort. Redefine family/object holdouts before claiming an independent anchor improvement; the small native pilot is not that evaluation.
2. Willett raw Gaia IDs are constant. Only identity matches established from external catalogue identifiers may be used; unmatched K2/TESS objects cannot be imputed.
3. No alpha product is ready until native source/units/flags, overlap calibration, heldout comparison and coverage evidence exist. The interface remains disabled by default.
4. New distance training requires actual signed Gaia parallaxes, quoted errors and zero-point/effective-error provenance. Hipparcos reference geometry alone is not training parallax provenance. GH convergence and independent anchors remain required.
5. Full-role corrected density results require the versioned rescoring and independent audit; the completed 384-row H100 certificate cannot retroactively clear all failed densities.

No new CANFAR training wave, catalogue acquisition or accepted experiment repin is part of these repairs.

Existing-data identity recovery: versioned Willett Kepler KIC→Gaia crossmatch recovered10,081/10,169 rows from APOKASC3;88 Kepler and32,883 nonKepler rows remain unmatched. Exact64-bit IDs/inputSHAs preserved. Shared seismic inputs mean correlated teacher errors, not independent truth. New crossmatch5testsPASS; rescoring-policy15testsPASS.

Final verification: **798 tests passed, 4 skipped**, repository hooks passed; implementation published to origin at `d95311d`. No active pilot/test process remains. Full-role density rescoring is prepared, not yet executed; blocked native-target gates above remain explicit.
