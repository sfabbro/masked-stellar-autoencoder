# Frozen-latent CNF feature screen

The inspected harness `cnf_parity` at04bb08e/a2a39f8 implements the binary density-shaping objective of Elsharkawy & Kahn (2025), approximating the paper's gated/quadratic architecture with zuko NSF. This is distinct from torchregress ContrastiveFlowLoss (conditional InfoNCE/NRE with flow scores) and from MSA's existing classifier NRE-B. This experiment must not inherit either old estimator's name or interpret their grid likelihoods interchangeably.

## Continuous FeH adaptation

The target task remains point regression of measured native FeH. Define two *training* regimes from existing strict-fit labels only: signal FeH<-2dex; background FeH>=-2dex. Learn two unconditional densities of fixed, fit-only PCA8 projections of the frozen epoch72 encoder. The PCA is shared by NLL/CNF controls and fitted after a fit-only latent standardization. Preserve its components, centering and signs; no test labels or test-latent statistics may fit preprocessing.

For the signal flow use `-mean(log p_s(signal)) + 2*mean(log p_s(background))`. For the background flow reverse the batches and use coefficient0.5. Clamp the *scalar batch objective* from below at-50 exactly as the harness recipe. This is a density-shaping objective, not conditional maximum likelihood; the guard can yield zero-gradient updates and its clamp fraction must be recorded. Sampling with replacement is necessary for the sparse signal regime; both class populations/counts and actual ordered source-ID streams must be recorded. This does not create additional unique metal-poor labels.

Freeze both flows before training the continuous head. Append `Phi_s = sigmoid(clamp(log p_s,-30,30))` and `Phi_b` to all256 standardized original latents, fit the appended-feature means/scales only on strict fits, then train the same residual-MSE point head. Inference uses no labels, bins or test-target values. Do not turn the binary ratio into a continuous FeH posterior. Density-shaping features provide a possible useful representation for regression, not an assertion of identifiable physical metallicity or nuisance invariance.

## Matched controls and proposed budget

- Raw residual-MSE head on256 standardized latent channels plus two fixed zero columns (same258input parameter count).
- NLL-flow feature augmentation: same PCA8, two NSF flows, optimizer, class sampling, updates and appended-feature scaling; ordinary positive-class NLL only.
- CNF-flow feature augmentation: same stages and models, shaped flow losses above.

Proposed screen:1000flow updates per flow atbatch256 and3000point updates atbatch256, with the raw control using3000point updates. Fixed seed, fit-only roles and no encoder updates. These are supervised/frozen-head budgets, NOT10Mreconstruction presentations and NOT continuation of parent Adam moments. Serialize each flow, point head, full preprocessing, regime counts, RNG, update counters, source stream digests and immutable source/library IDs. Main pretraining remains stopped; no parent/checkpoint mutations.

At high latent dimension absolute densities can be so small that the harness's logp[-30,30] clamp makes Phi constant. This is why the flows use a declared PCA8 projection while the continuous head retains all256channels. Record clipping fractions and rawPhi ranges/stds; reject constant or numerically negligible Phi columns before point-head fitting rather than silently inventing a feature variance. Standardizing tiny-but-resolved Phi is allowed using fit statistics and must be audited for test outliers.

## Gates and metrics

Actual-parent/native-input smoke, class-support check, finite loss/gradient checks, complete serialized reloads (including projection/Phi scalers), inference label-permutation invariance and source-role bans precede production. Independent CPU replay checks both densities/Phi transforms and every final point prediction with declared tolerances; no MLE grid is used. Compare population/source/metal-poor FeH MAE/bias and per-bin support. The185below-2test stars and14below-3test stars do not establish broad rare-tail accuracy. No new acquisitions, calibrated physical-PDF, paper numerical-replication or attenuation-cured claim.

Helper API: `CNFRegressionFeatures(signal_flow,background_flow).partition(fit_labels_dex)`, `density_shaping_loss(flow,pos,neg,contrastive_c=2/.5/None)`, `.freeze_flows()`, `.forward(all_latents,density_features=PCA8_latents)`. Distribution factories must return one finite log-probability per star. State dict contains both flows and the physical threshold; runner owns preprocessing/pointhead state.
