# Full label-union common-reference QA

Requested 7 November 2026 (Perth). CANFAR jobs have been submitted; these plots are not yet results from a completed large fit.

The union is keyed by exact Gaia `int64` source IDs. It retains every observed target and each survey measurement on overlapping stars. The union itself does not define a common physical truth. For each own-target monotone survey observation map `y_s = h_s(theta) + noise`, the plotted common observation is `h_s^{-1}(y_s)`. This coordinate change retains measurement noise, source errors and possible outliers; it is not Bayesian denoising. No average across surveys is silently substituted for labels.

The scope is the complete source-v5 registry: all available spectroscopic measurements in reference-only plots, and every usable-input star in prediction products (10,427,497 across FIT/TUNE/CAL/DEV/external). Spectroscopic-union counts are measured explicitly rather than conflated with the teacher-augmented registry. All-count Kiel histograms/contours include the full in-range population; uniformly sampled scatter overlays are identified as such. Display limits do not trim residual statistics, and tail counts are exported.

The producer shares the exact immutable snapshot held by the existing first-checkpoint QA job. It waits for that checkpoint, so comparisons use the same encoder, NSF and survey maps. It does not interfere with or select the running large training.

- GPU producer: `full_label_union_inference_20261107_v1`, session `i9qzn8xg`; 128 whole-vector draws per usable star, nine six-target marginal quantiles, conditional medians. Reserved globular members additionally receive 1,024 whole-vector draws each. These are Monte Carlo quantiles, not exact marginals or a newly calibrated density.
- CPU report: `full_label_union_report_20261107_v1`, session `if263n5h`; no model loading or training.
- Full survey-colored Kiel panels: all mapped labels, mapped labels with usable inputs, predictions on identical stars; each split plus pooled data. An additional density plot counts unique spectroscopic-union stars once.
- All six targets: mapped reference versus predicted scatter, residual versus reference and full-count median residual trends, residual distributions, matched label/prediction population distributions, per-source and FIT/TUNE/CAL/DEV statistics. Age/distance posterior teachers are explicitly distinguished from spectroscopic measurements.
- Same-star inter-survey differences before/after mapping, including DEV NMAD matrices, all-split statistics and overlap counts. Tighter agreement alone does not prove physical calibration.
- Up to eight globular clusters, membership chosen independently of model predictions: mapped measurement distributions, MSA median-star distributions, matched same-survey MSA distributions, and equal-star posterior mixtures from genuine draws. No new-stage FIT members are allowed into the reserved cluster QA.
- Complete-release exact-ID joins to Andrae (174,922,161 stars) and GaiaNet (219,196,764 rows), compared on all matched spectroscopic-union prediction stars, by split. Andrae known training membership is separately reported. Andrae [M/H] versus MSA [Fe/H] is a proxy comparison; GaiaNet ancestry is unknown. Catalogue agreement is not independent physical truth.

Raw 90% interval/reference inclusion is explicitly labelled: these intervals do not include survey measurement noise or a CAL correction, so inclusion is not a latent coverage certificate. FIT has optimization exposure, TUNE selection exposure and DEV development exposure.

Expected durable products:

- [Inference status](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/full_label_union_inference_20261107_v1/manifest.json)
- [Report status](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/full_label_union_report_20261107_v1/manifest.json)
- [Plot report when complete](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/full_label_union_report_20261107_v1/report.md)
- `predictions.h5`: exact source/role binding, six medians and quantiles, cluster draws.
- `mapped-survey-observations.h5`: common and original measurement coordinates, masks and identities.
- `residual-statistics.csv`, `same-star-survey-agreement.csv`, PNG figures and SHA-bound manifests.

Verification: 11 tests passed, including all-target inverse/forward map round trips (knots and extrapolation), rejection of missing/invalid log coordinates, synthetic end-to-end reporting across all splits and a reserved cluster, existing source-map equivalence, and the documented pipeline checks. No real model was executed locally.
