# Density prediction failures and remaining multi-target experiments

Status: 2026-10-22. This inventory records numerical verification failures separately from training failures. Most affected models passed separate point-mean and variance checks. Original failed reports remain authoritative until a versioned repair is independently verified.

## Failed density or distribution-related checks

| Experiment | Failed check / current limitation |
|---|---|
| Gaussian NLL, seed 43 | Intrinsic standardized log density: 1 of 97,695 test stars. |
| Noise-aware Gaussian + reconstruction, 10M | Intrinsic standardized log density: 19 test stars. |
| Noise-aware Gaussian, no reconstruction, 10M | Intrinsic standardized log density: 51 test stars. |
| Noise-aware Gaussian + reconstruction, 270M; tune-selected at 99M | Intrinsic standardized density: 767 test stars. Calibration failures: 14 standardized intrinsic, 5 native intrinsic, and 1 standardized observed; native observed calibration density passed. |
| Original clean NSF, recovered v2 | Pilot failed on 1 calibration standardized-density row. Full-role verification not completed. |
| NSF, NLL density objective | Pilot failed on 1 standardized-density row. Full-role verification not completed. |
| NSF, pure contrastive objective | Pilot failed on 1 standardized-density row. Full-role verification not completed. |
| NSF, NLL + contrastive objective | Standardized density failed on 131 test and 3 calibration stars. Separate full-role mean, variance, native-density and conformal checks passed. |
| BPF | Standardized density failed on 1 test star. Separate actual identity-tail and ignored depth/width bugs were fixed in torchregress; the corrected map requires versioned evaluation. |
| Noise-aware NSF | Intrinsic standardized density failed on 578 test and 15 calibration stars; native intrinsic failed on 1 test star. Observed standardized density failed on 8 test and 1 calibration stars; native observed density passed. |
| 3D forward inversion | Strict CPU solution replay failed on 29,162 test and 623 calibration label rows. Some gradient/Fisher and conformal-threshold checks also failed. This is a plugin inversion with local-stationarity diagnostics, not a verified global posterior. |
| Earlier scalar ForwardInversion/FlowEB screen | Numerical integration did not converge. No verified FlowEB result. |
| Older classifier NRE / contrastive-flow formulations | Strict grid/argmax replay failures. Superseded formulations remain unverified. |

MAF and cINN passed full independent joint-density, point-moment and conformal checks. This does not establish that they have the best point predictions or metal-poor coverage.

All original checks retain their declared tolerances. A change of log-density units can affect relative-tolerance acceptance; passing a native-unit check does not erase a failed standardized-unit check.

## Repairs and remaining checks on existing experiments

- [ ] Complete the bounded FP64 CUDA certificate, `density_fp64_precision_certificate_v1`, CANFAR ID `wetmolvn`. Last observed state: queued. It compares three immutable NSF checkpoints against independent CPU calculations, with zero training updates.
- [ ] After the certificate passes, produce versioned full-test/calibration rescoring for affected Gaussian and NSF models. A bounded 128-row certificate alone does not clear full-role failures.
- [ ] Independently verify corrected outputs and recompute global and metal-poor metrics, preserving original failed reports and outputs.
- [ ] Evaluate the corrected BPF tail map with explicit library-version provenance. Retrain where the requested architecture differs from the architecture actually trained; do not retroactively relabel old scores.
- [ ] Repair forward-inversion replay/stationarity and conformal discrepancies before ranking it as verified.
- [ ] Complete missing raw benchmark reconstruction-source and CUDA training-mask provenance checks where applicable.
- [ ] Complete independent flow-TTT checks before making flow-specific test-time adaptation claims.
- [ ] Resolve older weighted conformal provenance gaps: calibration weights/scores and domain-classifier state/seed were not saved. Existing subset geometry checks do not verify weighted coverage.

## Multi-target distribution families not yet tested on MSA

These are capability gaps, not completed experiments or measured results.

- [ ] SLS.
- [ ] SLSConformal.
- [ ] Mixture density networks.
- [ ] Full-covariance Gaussian NLL.
- [ ] Low-rank Gaussian NLL.
- [ ] Quantile models.
- [ ] Expectile models.
- [ ] Normalizing Flow Matching variants using corrected implementations.
- [ ] Conditional flow-matching variants using corrected implementations.

## Missing matched comparisons and distribution metrics

- [ ] Clean versus noise-aware NSF after corrected scoring, holding other policies fixed.
- [ ] Fully verified NLL versus pure contrastive versus NLL + contrastive, including global and Mondrian density-conformal coverage.
- [ ] Distributional ensembles scored as predictive mixtures. The completed three-seed Gaussian ensemble report covers point predictions and residual conformal intervals, not a complete mixture-density comparison.
- [ ] Predictive-sample joint metrics, including energy and variogram scores.
- [ ] Marginal CRPS, PIT and quantile diagnostics for joint models where their samples/CDFs support these calculations. Do not substitute Gaussian proxies for flow distributions.
- [ ] Density-region size alongside coverage, with the estimator and numerical integration method explicitly stated.
- [ ] Replicated seeds for promising systems, using reserved tuning data for selection.

## Reporting requirements

For both point predictions and predictive distributions, report population-wide and **FeH < -2** results separately. Keep distinct cohorts separate: native multi-target tests contain 97,695 stars and 141 metal-poor objects; historical scalar tests contain 100,620 stars and 185 metal-poor objects.

Report point errors, proper distribution scores, calibration and region/interval size separately. Density level sets are not rectangular intervals. Native survey labels are noisy; their quoted diagonal errors are not a physical-truth covariance model or input-EIV correction. Single-seed results, reused test comparisons and unsupervised overlap remain limitations.

## Existing review

- [Rendered density review](msa-density-review-20261022.html)
- [Durable experiment progress](../../experiments/xp/stellar-improvement-progress.md)

This document lists required work; it does not claim new experiments have been launched or missing methods have produced results.
