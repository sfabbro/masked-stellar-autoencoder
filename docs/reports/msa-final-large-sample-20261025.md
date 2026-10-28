# Final large-sample six-target review

**Follow-up:** [The log g diagnosis](msa-logg-failure-diagnosis-20261025.md) finds major structured failures hidden by global averages. The present recipe should not be treated as reliable across the full gravity range.

The four trained recipes and their expanded scoring/reporting jobs finished. The [complete report with all plots and metric tables](msa-final-large-sample-20261025/index.html) is the main review artifact.

Gaussian gives the best global XP MAE for log g, Fe/H, age and distance; noise-aware NSF is best for Teff and alpha among these four recipes. Mixed NLL + 0.1 contrastive improves metal-poor XP Fe/H MAE from 0.2286 to 0.2198 dex versus matched NSF but worsens global Fe/H/alpha errors. Retain Gaussian and NSF as the two useful candidates; mixed remains a tail-focused alternative, not an automatic promotion.

The uncertainty distributions are not ready for unrestricted scientific use: nominal intrinsic 90% intervals under-cover reported references, alpha generalization is weak, and observed age/distance PIT is unavailable with unknown quoted measurement errors. Intrinsic prediction versus noisy-reference coverage is not itself a latent-truth calibration test. The report separates these estimands.

The evaluation contains 4,412,982 feature-eligible identities, with primary per-label counts 688,406/670,065/688,406/1,707,545/161,359/163,887. Official no-XP and unknown-publication photometry are separated. Metal-poor references adequately support native-three-label diagnostics, while alpha/age/distance have only 4/1/1 primary tail references across modalities. Age/distance are catalogue-posterior labels; external APOKASC ages include nonphysical source tails and remain a separate diagnostic.

All requested six-label prediction/reference panels, metal-poor residual zooms, available PITs and ten-star/three-model corner galleries are linked in the report. The common corner axes enclose every original saved draw; no models were replayed or samples generated. Factorized heads do not identify a physical six-dimensional copula.

The original report-only read-only-mask failures remain preserved. CPU-only recoveries completed from immutable predictions. Verification includes source/input/checkpoint bindings, hash/CRC preservation, saved-corner coordinate checks and independent same-object Andrae array arithmetic; it does not certify the full expanded raw-array/neural/source/SGD pipeline. Full metric CSV/JSON and limitations are included.

Monitoring and task-owned background work have stopped. The scheduled monitor was deleted; no active task-owned CANFAR jobs or local dispatchers, collectors or report servers remain. Unrelated user sessions/processes were left untouched. [Shutdown evidence](msa-final-large-sample-20261025/shutdown.json).
