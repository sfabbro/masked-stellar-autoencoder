# Corrected conformal controls: native-coordinate evaluation

The original three validation reports were **scientifically invalid**: standardized saved means and variances were used directly against native Teff/logg/FeH labels. An independent arithmetic replay initially reproduced the same error; a subsequent units audit identified it. Original artifacts and failed audit attempts remain preserved.

The repaired runner converts means as `center + scale × mean`, variances as `scale² × variance`, and family dispersion after converting each member to native units. All family members must have the identical SHA-bound cohort and label normalizers. Calibration inputs and interval widths now use K/dex/dex consistently. A regression uses nonzero centers and nonunit scales.

These are saved-prediction evaluation controls with **zero training updates and no model forwards**. They are not new model-density certificates. The 2,046 original calibration objects are split deterministically into 1,023 gamma-tuning and 1,023 final-calibration objects. The test population has 97,695 objects, including 141 with measured FeH < −2. The adaptive control reusing calibration labels remains an empirical diagnostic, not an ordinary split-conformal guarantee.

## FeH marginal 90% intervals

| Model | Protocol | Population | Covered / n | Coverage | Mean width (dex) |
|---|---|---|---:|---:|---:|
| gaussian_family | fixed_gamma0 | all | 87182 / 97695 | 89.2390% | 0.385766 |
| gaussian_family | fixed_gamma0 | FeH_below_minus2_posthoc | 90 / 141 | 63.8298% | 1.746225 |
| gaussian_family | independent_tuning | all | 86867 / 97695 | 88.9165% | 0.388784 |
| gaussian_family | independent_tuning | FeH_below_minus2_posthoc | 83 / 141 | 58.8652% | 1.628006 |
| long_gaussian | fixed_gamma0 | all | 86330 / 97695 | 88.3669% | 0.345629 |
| long_gaussian | fixed_gamma0 | FeH_below_minus2_posthoc | 120 / 141 | 85.1064% | 1.246266 |
| long_gaussian | independent_tuning | all | 86330 / 97695 | 88.3669% | 0.345629 |
| long_gaussian | independent_tuning | FeH_below_minus2_posthoc | 120 / 141 | 85.1064% | 1.246266 |
| maf | fixed_gamma0 | all | 88297 / 97695 | 90.3803% | 0.427571 |
| maf | fixed_gamma0 | FeH_below_minus2_posthoc | 47 / 141 | 33.3333% | 1.425873 |
| maf | independent_tuning | all | 88297 / 97695 | 90.3803% | 0.427571 |
| maf | independent_tuning | FeH_below_minus2_posthoc | 47 / 141 | 33.3333% | 1.425873 |

This table uses global scores; the complete CSV also contains survey-Mondrian controls, other labels, empirical gamma selection and Bonferroni joint coverage. Tail membership is a post-hoc measured-label evaluation subgroup, not a label-based calibration stratum. Exchangeability and independently fitted score assumptions remain necessary; neither marginal nor survey calibration guarantees arbitrary tail or physical-truth coverage.

The old 9–14 dex tail widths and apparent 96–99% tail coverage were mixed-coordinate artifacts and must not be interpreted scientifically. Corrected results continue to show substantial tail undercoverage for several controls. Model uncertainty is intrinsic predictive scale; quoted target measurement errors are not added by this evaluation.

## Artifacts and verification

- [Corrected independent report](../../experiments/xp/results-stellar-improvement/validation-conformal-native-coordinate-correction-20261023/independent-report.json)
- [All verified conditional metrics](../../experiments/xp/results-stellar-improvement/validation-conformal-native-coordinate-correction-20261023/conformal-metrics.csv)
- [Coverage figure](../../experiments/xp/results-stellar-improvement/validation-conformal-native-coordinate-correction-20261023/conformal-coverage.png)
- [Width figure](../../experiments/xp/results-stellar-improvement/validation-conformal-native-coordinate-correction-20261023/conformal-width.png)
- Verifier: `experiments/xp/monitor/verify_validation_evaluations.py` independently reconstructs role identities, family sample dispersion, the gamma grid/folds, rank thresholds, coverage counts and widths from source-bound saved arrays. The production conformal helper is not called by this verifier.
- Test evidence: seven focused conformal-runner tests and three independent-auditor analytic tests passed; Ruff passed. Whole-repository checks are performed by the coordinating agent.

Native measured labels, reused tests, single training seeds and possible unsupervised overlap limit scientific interpretation.
