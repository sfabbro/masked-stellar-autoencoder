# MSA multi-target summary — 23 October 2026

All new small fits use the same native three-label cohort: **97,695 test stars, including 141 with measured FeH < −2**, parent86, photometry+XP. Numerical full-role verification and complete training-pipeline verification are separate. Scores below passed the former; objective/stream/decoder audits remain pending. No model forwards or retraining were performed for this summary.

## Matched 10M, seed45, reconstruction weight .03

NLL is the joint native-unit negative log likelihood in K/dex/dex, natural log per object; lower is better. Point errors are conditional-mean MAEs.

| Model | Global MAE: Teff K / logg dex / FeH dex | Metal-poor MAE: Teff / logg / FeH | Joint NLL global / metal-poor |
|---|---:|---:|---:|
| Gaussian | 72.32 / 0.1431 / 0.0963 | 161.86 / 0.3653 / 0.8078 | 4.7199 / 10.2937 |
| MDN4 | 68.77 / 0.1406 / 0.0948 | 172.68 / 0.3593 / 0.7380 | 4.6543 / 9.9769 |
| NSF / NLL | 75.02 / 0.1512 / 0.1070 | 268.47 / 0.4864 / 0.7515 | 4.1698 / 9.3882 |
| NSF / NLL +0.1 contrastive | 70.81 / 0.1429 / 0.1010 | 227.15 / 0.4473 / 0.7319 | 4.1060 / 8.7592 |
| NSF / NLL +1.0 contrastive | 72.20 / 0.1437 / 0.1004 | 234.86 / 0.4422 / 0.7645 | 4.1109 / 8.9344 |
| NSF / pure contrastive | 144.55 / 0.2100 / 0.1843 | 985.29 / 0.5848 / 2.0011 | 5.6669 / 28.5446 |

NLL+0.1 contrastive improves both global and metal-poor density score versus matched NLL45; it also improves all three mean MAEs in that comparison. Pure contrastive is substantially worse in both seeds45/46. The matched seed46 control now also passed numerical checks: its NLL is4.2218 global/9.2598 tail; mixed.1 is4.2679/9.0668 and mixed1 is4.1618/9.4463. Thus mixed.1 improves tail NLL in both seeds but global NLL and mean errors do not improve uniformly; mixed1 also has seed-dependent tradeoffs. This is promising evidence, not a universal win. MDN4 is a competitive point predictor; a density-score improvement is a different criterion.

## Earlier experiments and missing families

- **MAF / cINN:** full independent density/point/CP checks passed. Earlier joint NLL global/tail: MAF4.2675/9.9754, cINN4.2932/9.8366 versus matched Gaussian42 4.6948/10.6277. Their mean prediction errors worsened. These are earlier seed42/settings, not matched to the new45/46 table.
- **BPF:** point/CP report exists; old standardized-density check failed. Published tail/factory semantic repairs require a separate map/version evaluation. Old-map FP64 rescoring is not evidence for the repaired map.
- **Long noise-aware Gaussian + recon:** strongest established point result in the earlier programme: global MAE58.83K/.11997/.07583; tail161.34K/.19984/.39312. It used270M presentations and tune-selected99M, not the10M budget. Old density replay failure remains separate from verified points/CP.
- **Ensemble:** fixed three-seed Gaussian mean improves points slightly; no complete predictive-mixture density comparison.
- **Forward inversion / FlowEB:** training or bounded code paths exist, but old scalar integration and3D strict replay failures remain. No verified global posterior result.
- **TTT:** completed long-Gaussian subset check shows tiny mixed changes; clean prediction remains primary. Its512 adapted test objects contain no FeH<-2 stars. Flow-specific TTT remains unverified.
- **SLS / SLSConformal:** NOT RUN on MSA. No scores exist.
- **Full-covariance / low-rank Gaussian, native CRPS-trained heads, quantile/expectile, corrected NFM/CFM and TARFlow:** no completed matched native MSA result in this ledger. Library availability is not an experiment.
- **Age:** six384-object frozen-embedding joint/composite pilots completed, with state/role/selection checks. Zero shared-teacher overlap; code-path evidence only, not age accuracy or identifiability.
- **Alpha and distance:** integration/preparation exists, but source-label/quality provenance gates remain; no validated new predictive result.

## Calibration and unresolved metrics

The conformal standardized/native-coordinate bug is repaired; original invalid reports remain archived. Corrected fixed-gamma long-Gaussian FeH90% coverage is88.37% global and85.11% tail, with mean widths.3456/1.2463dex. Global calibration does not guarantee metal-poor or physical-truth coverage. Mean-residual intervals and joint density level sets are different objects.

Predictive-sample energy/variogram scores, flow marginal CRPS/PIT, and density-region volume are still missing where no validated samples/CDF/volume estimator were saved; Gaussian moment proxies are not substituted for flows. Native quoted diagonal label errors are not input-EIV correction or physical-truth covariance. Reused tests, few seeds and unsupervised overlap limit conclusions.

## Full data

- [All24 new fits, both populations; pending cells explicit](../../experiments/xp/results-stellar-improvement/multitarget-summary-20261023/new-wave-metrics.csv)
- [Source-bound JSON](../../experiments/xp/results-stellar-improvement/multitarget-summary-20261023/new-wave-metrics.json)
- [Earlier full metric review](msa-density-review-20261022.html)
- [Corrected conformal report](msa-conformal-native-coordinate-correction-20261023.md)
