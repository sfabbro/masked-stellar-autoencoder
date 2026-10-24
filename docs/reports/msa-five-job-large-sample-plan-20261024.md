# Four-head large-sample comparison

This wave has a hard budget of **five CANFAR headless submissions total**: one CPU source-preparation job and four GPU training/evaluation jobs. Failed or ambiguous submissions consume that budget; they are not automatically replaced. No native model execution runs locally.

## Matched training

All four heads start from the completed large reconstruction model, checkpoint `c544548ba203f249e11509d0e0f9832df0e8366e6f7b0c7a639460752061a2f9`. Each receives three passes over the same existing **150 million distinct original TRAIN objects**, or **450 million reconstruction presentations**. This is not a repetition of the smaller 9M benchmark. Heads and optimizers start fresh. Reconstruction weight is fixed at 0.03, with the same learning rates, batches and modality masking across arms.

The arms are an ordered three-quantile pinball baseline, a noise-aware diagonal Gaussian, a noise-aware scalar NSF product, and the same NSF with NLL plus 0.1 contrastive loss. The latter convolves quoted label errors in both positive and negative contexts. Unknown measurement errors are not claimed to be known zero, and catalogue-posterior widths are not substituted for measurement errors. Conformal calibration is post-fit, separate from the contrastive training objective.

A shared whole-XP masking policy on FIT objects teaches the model to predict from photometry when XP is absent. Natural XP availability and artificial masking are archived separately. Model selection uses a common TUNE point criterion available for every head; TEST labels never select a checkpoint or recipe.

## Test and reference evidence

Existing FIT, TUNE and CAL identities are retained. The large existing sparse TEST cohort is supplemented only with eligible held-out labelled objects, with source-qualified labels, quality masks and ancestry checks. Genuine no-published-XP status requires the official Gaia availability flag, absent XP coefficients and usable measured photometry; a missing staged feature row alone is insufficient evidence.

The existing GALAH Gaia crossmatch provides availability and photometry without acquiring a new catalogue. Added TEST objects are checked against prior direct supervised exposure. Reconstruction-sample membership is recorded separately: unsupervised exposure is disclosed rather than confused with direct target-label exposure.

The released Andrae XGBoost catalogue contains stars with published XP. Its common-object comparison therefore applies to the XP subset; the genuine no-XP subgroup is evaluated against available survey references. Andrae [M/H] versus reference [Fe/H] is explicitly a proxy comparison, and its training-sample membership is retained. Teff/logg/metallicity comparison does not create Andrae references for alpha, age or distance.

The six coordinates are Teff, logg, [Fe/H], alpha, age and distance. Age/distance references are catalogue posteriors, not identified physical truth or raw-parallax likelihoods. Sparse labels remain missing. In particular, missing metal-poor alpha/age/distance references are shown as empty panels/counts rather than imputed validation.

## Required report

Each job must retain selected checkpoints, source and policy bindings, role predictions, scores, intervals and figure receipts. Saved-array comparisons will show per-target counts and errors for the global, metal-poor, natural-XP and genuine-no-XP groups, with source strata and paired Andrae overlap.

For all six coordinates, reports include predicted-versus-reference panels and residual-versus-reference panels zoomed on the metal-poor subset. Distribution heads additionally provide native predictive CDF/PIT diagnostics and proper distribution scores. PIT summaries conditional on measured metallicity are descriptive; the conditioning changes the uniformity null.

Representative model-blind stars are selected by label and brightness strata, using common objects and axes across heads. Full six-coordinate corner plots use actual predictive draws and show only available references. Scalar NSF and diagonal Gaussian heads imply conditional coordinate independence; these plots do not identify an empirical six-dimensional copula or a physical posterior.

The three-quantile baseline defines three ordered marginal quantiles, not a normalized PDF or full joint sampler. It receives point, pinball, interval and conformal comparisons; its PIT and full corner entries are explicitly unsupported. No Gaussian proxy will be invented to fill them.

Artificially masked XP CAL/TEST rows are a separate paired stress test. They do not establish exchangeability or conformal guarantees for the genuinely no-XP population.

## Acceptance state

Source preparation, actual no-XP counts, runtime gates and submission IDs will be recorded after their corresponding CANFAR results exist. This design document does not claim a launched job, a passed production gate or a scientific winner.

Primary reference: [Andrae, Rix and Chandra (2023), Robust Data-driven Metallicities for 175 Million Stars from Gaia XP Spectra](https://arxiv.org/abs/2302.02611).

## Launch and source results

All five submissions were accepted once at source `7552253630fe22bcd680a91ee9fa658893218d43`, with torchregress `6bd8b93b35fe4fe216319870e72e94eb78429106`. Full source CI passed **994 tests, 4 skipped**, and origin advertised the exact commit before submission. The single gated launcher finished without ambiguous IDs; it must not be restarted.

- CPU preparation: `uein5h8u`, actual source preparation complete.
- Three-quantile baseline: `yr4vfjl3`.
- Noise-aware Gaussian: `wzo3q1yg`.
- Noise-aware scalar NSF: `qrf5ggn3`.
- Noise-aware scalar NSF plus 0.1 contrastive: `xqvrpnd5`.

All four GPU sessions were actually Running with the pinned source and H100 NVL MIG 1g.12GB allocation observed. Startup is not a certificate of completed SGD, three epochs, or final prediction accuracy.

The prepared input SHA is `01bb0b412d81859d42fabbce4c4e64cf2092e77ee06df33d6caafae87d6a5882`. TEST now contains **104,385 distinct stars**, adding 7,335 eligible held-out objects. Per-target TEST reference counts are **103,956 Teff; 101,191 logg; 103,956 [Fe/H]; 431 alpha; 2,048 age; 2,048 distance**. These are sparse reference counts, not 104,385 complete six-label stars.

There are **4,739 genuinely no-published-XP TEST stars**: all have Teff/[Fe/H] references, and 4,570 have logg. None have eligible alpha/age/distance references. The metal-poor corner-selection pools contain 144 XP stars and 10 no-XP stars; the latter is a small descriptive subgroup, not strong evidence of tail calibration. Four TEST stars have all six eligible references.

The exact same-object Andrae TEST overlap is **94,187**, including **68,538 known not to belong to its released training sample**. No Andrae six-label truth is inferred. FIT/TUNE/CAL bytes were retained, and direct held-out FeH exposure in the audited large-model FeH auxiliary stream is zero. Cross-target exposure and unsupervised 150M membership remain explicit strata; earlier parent history is not fully certified.

Runtime manifests, source evidence and the completed launch state are preserved under `experiments/xp/results-stellar-improvement/five-job-large-wave-20261024/`. Final results and figures remain pending the four actual training/evaluation completions and saved-array/visual checks.
