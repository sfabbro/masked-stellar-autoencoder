# Proposed release architecture: stellar prediction and measurement refinement

Prepared 2026-10-17 UTC. This is the recommended target design, not an implemented or validated joint pipeline and not a new launch authorization. Six accepted epoch-72 FeH pilots have finished; their independent CPU verification is progressing. They select loss/head candidates for FeH, not the final multi-target model. Existing main training and accepted provenance remain unchanged.

## Decision

Update after the completed six-arm independent comparison: residual-MLP Huber is the conservative point-head candidate (overall MAE0.08537557 versus plain0.08680719 dex); RnC has lower overall bias but slightly higher MAE. BatchEnsemble did not beat residual under this one budget/seed. Gaussian CRPS is preferable to Gaussian NLL on population scores, but its nominal95% intervals cover only70.81% of the185 reported FeH<-2 labels. Do not treat either Gaussian as a calibrated physical PDF. All six worsen the common original-validation reconstruction/spectral diagnostics, supporting the matched auxiliary check rather than assuming reconstruction weight0.01 fully protects the encoder. These observations refine the proposal; they do not authorize another launch. See `../experiments/xp/results-stellar-improvement/prediction-comparison-e72/verified-comparison/comparison.md`.

Build one shared spectrophotometric encoder with reconstruction regularization and a joint Teff/logg/FeH/distance prediction module. Alpha and age are optional modules, disabled until their native labels and validation support them. Deliver both an estimate from spectra/photometry alone and a separately named measurement-assisted estimate. Always retain the original measurement and its provenance.

Do not launch another broad grid. After verifying the six pilots, the remaining essential work is one bounded joint-system comparison and independent uncertainty/refinement validation. A flow must earn its additional complexity against a joint Gaussian control. A second seed is for finalists only. No new survey acquisition is required for this round.

## Exact data routing

### Core encoder: observations, availability and uncertainty

Use the following 134 existing observables, with their real availability masks:

- Gaia XP: bp_1 through bp_55 and rp_1 through rp_55 (110 coefficients).
- Gaia photometry: G, BP, RP.
- WISE: W1, W2.
- SkyMapper: U_SMSS, V_SMSS, G_SMSS, R_SMSS, I_SMSS, Z_SMSS.
- SDSS: U_SDSS, G_SDSS, R_SDSS, I_SDSS, Z_SDSS.
- PS1: G_PS1, R_PS1, I_PS1, Z_PS1, Y_PS1.
- 2MASS: J, H, KS.

Preserve the parent's documented XP G-normalization, photometry transform, feature order and scaler; verify units on actual original records. Do not mix raw and normalized XP or silently change photometric systems. Keep observed apparent photometry; extinction treatment must agree with that convention. Missing values are never zero-valued measurements.

For the first compatible continuation, preserve the 140 input slots and the parent's observed/missing convention. The remaining six slots PARALLAX, EBV, pmra, pmdec, RA, DEC are masked in the proposed physical core, in BOTH supervised and reconstruction calls. This differs from the current six pilots: their supervised call retains EBV, while their reconstruction call can see all 140 fields. The inherited parent was trained with astrometry; this input change does not erase that history or certify unseen pretraining.

Keep native per-band uncertainties and full available XP covariance in a separate observation-noise record. Use them for quality checks, noise perturbation/integration and uncertainty validation. Do not append hundreds of unvalidated error/correlation columns to the pretrained encoder or fabricate missing errors. A later uncertainty-context adapter is a separately tested extension, not a requirement for first release. Prediction of physical parameters should not depend on survey label precision or spectroscopic label flags through an encoder shortcut.

### EBV and extinction

Do not treat the current EBV column as exact, distance-specific stellar reddening. Audit its map, units, extinction law and 2D/3D meaning first. For the first release exclude it from the core point predictor; carry the audited map value, uncertainty/validity and sky query provenance as external context and report performance by reddening. This is a conservative explicit choice, not evidence that dust information is useless.

An optional dust-assisted branch can later infer nonnegative stellar reddening jointly with distance, with an explicitly declared dust constraint. It needs a separate validation comparison and must advertise that additional information. Do not apply a full-column 2D map as an exact nearby-star extinction correction, or deredden an already dereddened quantity again. Dust/extinction degeneracies must appear in uncertainties, not be hidden by an exact input assumption.

### Positions and proper motions

RA/DEC in ICRS with reference epoch: source association, epoch propagation, dust lookup, crowding/isolation, Gaia zero-point metadata, spatial validation and spatially correlated-error diagnostics ONLY. Galactic coordinates/HEALPix are derived bookkeeping, not extra point-head inputs.

pmra/pmdec in mas/yr, their errors/correlations and reference epoch: association/propagation and quality diagnostics ONLY. No kinematic distance relation, velocity-population prior or Galactic membership shortcut in the primary model. Radial velocity is likewise not a primary input.

### Parallax and existing parameter measurements

Raw signed Gaia parallax in mas, its formal uncertainty and audited quality/zero-point metadata enter the distance observation likelihood and the optional measurement-assisted inference once. They are never a core encoder input. Retain zero/negative observations with valid errors; no positive-parallax or parallax-SNR selection for fitting distance.

Existing Teff/logg/FeH/alpha measurements, errors, covariance when available, survey identity and target-quality flags enter supervision and optional refinement, not the feature-only head. Preserve every source layer separately. An assisted estimate excludes that object's measurement from predictor fitting through cross-fitting or a held-out prediction; it must not reuse it as independent evidence twice.

### Head inputs and outputs

Both the reconstruction decoder and prediction module take the encoder's 256-dimensional latent. No direct RA/DEC/PM/parallax or spectroscopic-label concatenation enters either primary prediction module.

The reconstruction decoder keeps its 136-channel checkpoint-compatible shape initially. Score only the 134 photometry/XP outputs above; PARALLAX and EBV channels are inactive and are not released predictions. No position or PM decoder exists. Reconstruction is a training regularizer, not an additional source of independent evidence for refinement.

The primary prediction module represents the joint event vector (log(Teff/K), logg[cgs dex], [Fe/H][dex], log(d/kpc)). Preserve joint dependence. Add [alpha/Fe] only after native definitions, quality, errors and cross-survey scale are audited; APOGEE alpha/M cannot simply be renamed alpha/Fe. A separately flagged age/Gyr module is initially off. No predicted ages or alphas should appear as if those modules were trained.

Use the existing residual encoder; select a compact MLP/residual/BatchEnsemble conditioner using verified six-arm results and cost. No immediate TabR retrieval database or wholesale backbone replacement. Our BatchEnsemble pilot is TabM-style, not official TabM. A density output can use that conditioner without changing the encoder into a different architecture.

## Losses and statistical interpretation

### Primary physical-observation loss

Fit a conditional latent distribution q_theta(u | x) by frequentist maximum likelihood. A joint Gaussian with positive-definite covariance is the reference; a conditional neural spline flow is the flexible candidate. The flow is a density model, not automatically Bayesian. Include transformation Jacobians when evaluating densities in physical units.

Train against actual measurements, not just their point estimates treated as exact physical truth:

    L_obs = -log integral q_theta(u | x) p_s(m_observed | h(u), R_s, quality) du

Here h maps log-temperature to Kelvin and log-distance to parallax, retains logg and abundance units, and applies explicitly audited survey calibration conventions. For distance s=log(d/kpc), h_pi(s)=exp(-s) mas plus the declared Gaia zero point. R_s denotes the native measurement-error model; its quoted random errors are not automatically total calibrated errors. Unknown covariances/systematics remain explicit limitations or separately identifiable nuisance terms, not invented universal floors.

This formulation allows negative measured parallaxes without requiring negative physical distances. For a point-distance control it reduces, up to constants, to:

    0.5 * ((pi_observed - z - exp(-s))**2 / sigma_pi**2 + log(sigma_pi**2))

Numerically verify quadrature/sampling, gradients and the narrow-error limit independently before training a noise-convolved flow. Do not silently replace this integral by a flow NLL on StarHorse distance medians, subtract variances after fitting, or call observed-label CRPS a score of deconvolved physical truth.

Missing targets contribute a marginal likelihood over observed dimensions, not imputed labels. A joint Gaussian supports subvector marginals directly. The current torchregress flow API rejects partially masked target vectors: first use a declared complete-core-target cohort for its matched comparison; sparse targets may train separate marginal heads. Do not score an arbitrary filled target vector. A general partially observed flow requires tested marginalization before use.

Survey observation models remain source-tagged; do not average GALAH/APOGEE/PIGS/Li/DESI into a universal truth. Absolute scale is only as good as independently established calibration. Correlated repeat labels from one star are not independent stars. Existing native catalogues are the first supervision; additional silver labels are optional following their measured population tradeoffs.

### Reconstruction, spectral auxiliary and representation loss

Candidate total objective:

    L = L_obs + 0.01 * (L_masked_MAE + 0.1 * L_spectral)

The two weights are starting values supported by earlier separate pilots, not established optima for the new joint objective. Keep their documented scored-feature/hidden-XP normalization and check actual-parent gradient magnitudes. Compare ordinary reconstruction against the spectral addition with matched inputs and streams. Do not add a new photometric auxiliary by default: previous evidence was mixed.

Huber is the robust point-prediction control. Gaussian CRPS/NLL from the six pilots can select useful observed-label baselines; they do not yet implement the physical-observation likelihood above. Score predictive distributions in observation space after convolution, and separately score latent estimates on appropriate independent references. CRPS evaluates marginals; joint likelihood and joint diagnostics are needed for dependency structure. Quantile intervals remain a simpler fallback if a reliable full density is not justified.

No default extra RnC/RaC term: include it only if the verified pilot improves relevant validation without damaging precision or tails. The current arm uses continuous single-target FeH RnC, not Chang's adaptive-bin multi-target recipe. A future adaptive-bin multi-target objective is an additional experiment, not a component we can claim tested. Never interpret ensemble member spread alone as calibrated uncertainty.

Normalize supervised losses over actually observed eligible data, reconstruction over scored features, and accumulated microbatches using their actual denominators. Keep source/tier sampling explicit; changing population weights changes the estimand. Use fit-only scaling and exclude tuning/calibration/final references from every fitted transform.

## Two prediction products

1. **Feature-only:** joint spectrophotometric predictive distribution and marginal medians, 68%/95% intervals, covariance or joint samples where validated; distance and corresponding physical parallax reported consistently. No random feature masking at prediction, dropout off, genuine missingness retained. Full PDFs require validating tails and multimodality; a Gaussian control cannot claim to resolve separate dwarf/giant modes.
2. **Measurement-assisted:** original measurements plus feature-only prediction, source/error provenance, consistency test and calibrated refinement policy. Retain an unchanged-measurement fallback for precise labels, conflicting evidence, unsupported populations, binaries/poor fits and missing calibration. An unchanged fallback prevents an algorithmic change, not unknown original systematic errors.

For a simple independent Gaussian scalar case, the precision-weighted combination is a useful control; validate independence, bias and covariance first. Cross-fitting prevents direct self-training but does not remove correlated survey systematics. Never combine a Gaia-conditioned prediction with the same Gaia parallax as if independent.

If a learned conditional density is multiplied by a measurement likelihood and normalized, it acts as a population prior for that refinement. Do not advertise that operation as prior-free frequentist distance inference merely because the network was maximum-likelihood trained. For the requested non-Bayesian assisted distance product, use an explicitly specified joint measurement/predictive-error model, profile likelihood and calibrated bootstrap confidence sets, including bias/covariance/selection assumptions. Keep the original parallax and its confidence set available. A normalized profile likelihood is not a posterior PDF. Low information can give an unbounded distance confidence set; do not force a finite upper distance limit for cosmetic intervals.

The feature-only learned predictive PDF is itself population-dependent. Avoiding a hand-written Galactic prior does not eliminate training-selection or regularization assumptions. It is acceptable to publish predictive PDFs for the feature-only product and confidence sets for the frequentist assisted product; these are different statistical objects and must be named accordingly.

## Meaning of improving precision

The goal is calibrated lower error, not simply narrower reported error bars. We cannot guarantee lower error for every star. Precise independent spectroscopy/nearby geometry should dominate an assisted estimate, and fallback can preserve it exactly. Weak measurements can improve substantially when independent spectrophotometric information is informative and calibrated; the gain is not guaranteed in degenerate/OOD regimes or for alpha/age that the inputs barely constrain.

Define quality regimes before evaluation. In the high-precision regime require paired non-inferiority in error/bias AND interval coverage, with a scientific tolerance specified from the reference accuracy. If the evidence is inconclusive, retain the original measurement for that regime. In lower-precision regimes require a measured accuracy gain at satisfactory coverage, reporting tails and failures. Do not choose a numerical tolerance after seeing the final test result.

Use separate source-ID fit, tune, calibration and locked final-test roles; reserve spatial/survey holdouts and all declared external references. The previously inspected 100620-star set cannot become a fresh untouched final test through renaming. Unsupervised pretraining overlap is separately disclosed. Conformal calibration can validate prediction of future observed labels under exchangeability; it does not certify latent physical-parameter coverage or per-star conditional coverage. Use independent references/repeats and the explicit observation model for physical uncertainty claims.

For distance evaluate raw signed-parallax residuals, zero point, uncertainty coverage and outliers across magnitude, color, reddening, stellar class and precision; independent geometric references establish improvement over Gaia. Do not bin distance only by inverted noisy observed parallax. The existing nearby Hipparcos cohort does not settle distant-star accuracy. Without appropriate distant references, distant performance remains unvalidated rather than inferred from agreement with Gaia-conditioned catalogue distances.

## Bounded remaining experiments and implementation gates

- Finish existing six independent verifications and a paired report; choose a conservative conditioner/loss candidate. No resubmission.
- Materialize native Teff/logg/error/flag/unit associations from already acquired catalogues. Audit optional alpha separately. Construct the joint cohort and immutable roles; no fabricated joint-label count.
- Build/test the clean-input joint observation-likelihood runner, actual-input noise integration, counterfactual no-copy tests, native covariance handling, accumulated gradients and full encoder/decoder/head reloads. These tests precede launch.
- One staged bounded comparison: joint Gaussian baseline versus joint flow, same source/parent/roles/10M replay budget/optimizer and ordinary reconstruction; then ordinary versus spectral auxiliary for the selected candidate. This is at most three initial runs if the flow passes its integration gate; otherwise two Gaussian reconstruction arms. Do not simultaneously vary source merge, balancing, LR and input policy. Use one common parent for this comparison, chosen and hashed at its eventual launch.
- Repeat only the selected configuration with one second seed if the initial gains justify it. Validate precise/noisy measurement refinement and intervals on independent roles; no automatic alpha/age/distance acquisition wave. Age can train later with the core frozen initially, so questionable age teachers cannot move the core before benefit is established.

Start from the conservative effective batch32768, microbatch16384, encoder LR near1.25e-6 and warmed head LR1e-3, subject to actual gradient checks and tuning. The earlier sweep does not establish a universal LR optimum. Save initial predictions and evaluations every three epochs in a longer accepted continuation; for 10M-presentation pilots retain presentation-based snapshots instead of falsely calling one replay pass three epochs. Select on tuning data, calibrate once, report the locked test once.

## Sources and local evidence

- [Six-arm specification](prediction-comparison-10m-20261016.md), accepted ledger and latest durable progress: actual execution and scope.
- [Earlier continuation discussion](continuation-decision-20261016.md): reconstruction/LR evidence and preparation limitations.
- Hogg, Eilers & Rix, [Spectrophotometric parallaxes with linear models](https://arxiv.org/abs/1810.09468): raw-parallax fitting precedent for a particular red-giant sample, not evidence of our achieved accuracy.
- Durkan et al., [Neural Spline Flows](https://arxiv.org/abs/1906.04032): flexible density construction, not a guarantee of physical calibration.
- Gibbs, Cherian & Candes, [Conformal Prediction With Conditional Guarantees](https://arxiv.org/abs/2305.12616): limits and restricted forms of conditional coverage.
