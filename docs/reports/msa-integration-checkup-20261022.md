# Integration checkup — 2026-10-22

Scope: local code/evidence review at MSA `9026eb3`, torchregress `b797a02`. No training launched, accepted experiment changed, or concurrent library edits modified. This is a bounded review, not a certificate of the entire library or pipeline.

## Verification

`pixi run --frozen --no-install pytest tests/test_alpha_heads.py tests/test_distance_heads.py tests/test_multi_teacher_age.py tests/test_theory_conformal.py -q`: **70 passed**. Synthetic contracts passing do not establish native-data validity or end-to-end integration.

## Findings

### High: adaptive gamma selection does not retain ordinary split-conformal exactness

`src/masked_stellar_autoencoder/reporting/theory.py`, `TheoryScaledScore.fit`: gamma is chosen using out-of-fold scores on calibration labels, then the final threshold is computed on those same labels. The result unconditionally records `finite_sample_exact=True` (line 583). Avoiding test labels alone does not establish the exchangeability of calibration and future scores after this selection. The order statistic is exact arithmetic; the nominal coverage guarantee is not thereby established.

Before a coverage claim: choose gamma on a separate tuning split and freeze it before final calibration, or use a method with a proved selection adjustment. Keep current adaptive results explicitly empirical. Add tests distinguishing a fixed/tune-selected score from calibration-selected score metadata.

### High: teacher objective is a composite marginal likelihood, not a shared-latent joint likelihood

`src/masked_stellar_autoencoder/training/multi_teacher_age.py`, lines 1324–1335 and 1409–1440: each teacher is integrated separately, then its mean NLL is summed. This computes a weighted composite objective based on `product_s integral p(t|x) p(y_s|t) dt`. A joint model sharing a latent age requires `integral p(t|x) product_s p(y_s|t) dt`. These differ when latent uncertainty is nonzero; the joint Gaussian covariance contains off-diagonal terms `b_s b_r Var(t)` which the composite score omits.

The composite objective can be deliberate and useful, but its descriptions should say so. If the intended A1 model requires joint evidence from overlapping teachers, implement a per-star joint integral/covariance with sparse-teacher masks and separately declare teacher balancing. Add an overlapping-two-teacher analytic joint-Gaussian regression test. Existing one-teacher convolution tests do not test this property. Shared systematic errors among seismic teachers are an additional scientific assumption.

### Medium: new catalogue comparisons need stronger pairing and target contracts

`scripts/compare_released_andrae.py` and `scripts/compare_released_srivastava.py` load `test_mean` and associate it with cohort IDs without the source-ID alignment check already present in `compare_released_gspphot.py`. Add shape/order/ID or equivalent artifact-binding checks before reusing these scripts with another prediction file. No current mismatch is demonstrated.

Andrae's heldout subsection reports its heldout metrics, but does not report MSA metrics on the identical heldout subset. Add those for a heldout head-to-head comparison. Both Andrae and GSP-Phot [M/H] are being compared to native [Fe/H]; retain the explicit proxy caveat and check alpha-dependent offsets before interpreting the metal-poor ranking as identical-target performance. Similar MAEs on the Srivastava overlap alone do not establish statistical indistinguishability; use paired uncertainty estimates for a difference claim.

## Actual integration status

- **Alpha:** `alpha_heads.py` explicitly declares `interface_only`, is disabled by default and requires five evidence gates. It distinguishes APOGEE alpha/M, GALAH alpha/Fe and RVS alphafe. This is a sound contract, not a trained/validated alpha product.
- **Age:** the multi-teacher machinery, named-scale gauge, cluster/coeval and monotone correction components exist and pass focused tests. A1–A3 are still a plan. Current production `run_native_age.py` does not invoke the new module. Willett ID crossmatching remains incomplete; APOKASC3 has only 672 objects in the 10M benchmark, and the current StarHorse/seismic domains differ. Finish identity/overlap, family-level holdouts and actual runner integration before claiming new age prediction capability.
- **Distance:** the new head uses log distance in kpc and signed parallax in mas, with likelihood convolution rather than parallax inversion; focused tests pass. Current `run_distance_value.py` does not invoke `distance_heads.py`. Distinguish older distance-value experiments from the new latent-distance/parallax likelihood track. Require native units, zero-point/error provenance, quadrature convergence and heldout distance-anchor evaluation before launch/results claims.
- **torchregress:** latest committed work adds tokenized TARFlow NFM architecture and its argument documentation. The working tree also contains substantial API removals/moves and new CDF/density metrics. Those uncommitted changes were preserved, not certified by MSA tests. Pin a tested commit for integration; run its public-API and focused numerical tests separately. Existing MSA production pins do not automatically adopt this new architecture.
- **Density numerical repair:** the bounded actual H100 FP64 certificate passed three existing NSF checkpoints; this does not replace full-role corrected scoring or clear old failed reports.

## Recommended order

1. Correct the statistical claims/objective distinction above before A1 or theoretical coverage publication.
2. Add comparison identity guards and matched heldout metrics; retain measured-label and target-definition caveats.
3. Finish anchor crossmatches and runner wiring for one small age pilot; keep alpha and new distance products marked unrun until their actual data gates pass.
4. Complete versioned full-role FP64 density rescoring before ranking new distribution methods.
5. Review/pin the finished torchregress API changes and test TARFlow separately; avoid silently changing existing experiment pins.
