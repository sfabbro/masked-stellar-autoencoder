# torchregress update review — 2026-10-18

Reviewed local astroai/torchregress HEAD f9f71f2 against accepted MSA pin66d52fe and astroai/torchregress-harness HEAD a2a39f8. No dependency changes or submissions made. Existing completed runs retain immutable pins.

## New implementation

VIR adds VIRLoss and VIRRegressor: probabilistic latent representation, neighbouring-label statistics smoothing, density-reweighted NIG/Student-t output, predicted-bin two-pass inference. It is a discriminative VIR core, not the paper's complete reconstruction ELBO and not an EIV measurement-error likelihood. BalancedMSE changes are attribution/docs only: existing loss is inverse bin-frequency weighting, not Ren et al. noise-adaptive BMC. Uncommitted conformal documentation and Claude outputs left untouched.

Harness a2a39f8 reports single-seed GPU observations, not independently reverified here. California housing MSE RMSE .600, VIR .602; worst-group MAE .850 versus .800 and tail RMSE1.226 versus1.152. GroupDRO has better worst-group .611 and tail1.144 but worse population RMSE .718. Thus the prose claim that VIR is best on worst-region/tail is not supported literally by its table; the useful result is its population/tail compromise. Synthetic stress VIR tailRMSE1.071 is worse than MSE1.033. No universal advantage or calibrated physical uncertainty claim follows.

## Bounded candidate experiment set

Use existing immutable parent72/cohort/role bans and same reconstruction/label budgets for comparability, not latest stopped-pretrain checkpoint. No survey acquisition. Prioritise a frozen-encoder head screen before costly encoder updates; all screened arms must share that frozen regime, and a matched residual-MSE head must be included.

1. Matched residual-MSE head control.
2. Unweighted NIG baseline: same probabilistic head family and regularizer, unit density weights, no representation recolouring. This separates a distribution-family change from imbalance correction.
3. Density-reweighted NIG (VIRLoss) without feature recolouring: fit density/bins on fit labels only. Must compare inference using predicted labels, never heldout true bins.
4. Full discriminative VIR with count-shrunk smoothing: fit-only feature statistics; predicted-bin inference. Same probabilistic architecture/optimization as baseline; no silent use of standalone VIRRegressor.fit that replaces the accepted optimizer/replay policy. Test-label permutation invariance is a required leakage check.

Before launch: preregister physical bin edges/bandwidth/shrinkage, inspect actual strict-fit tail counts, pin published dependency, actual-parent/native-input smoke and reload, checkpoint all bin/statistical state and RNG. The standalone fit loop is not a drop-in continuation runner; integration needs an explicit adapter and tests. These are proposed arms, not accepted jobs.

Report population and source-separated conditional MAE/bias, below-2/below-3 support, per-survey metrics, NLL/CRPS and interval coverage versus measured native labels. Below-2 test185, below-3 test14 limit conclusions; neither a latent-metallicity calibration nor attenuation cure can be established. Distinguish imbalance correction from XP dilution bias.

Do not prioritise another CNF architecture run from current harness evidence: latest method reproduction approximates the paper architecture and has no toy rotation advantage over DNN. New classifierNRE MSA verification failed an initial adjacent-grid argmax check and requires diagnosis before a fresh claim.

## Existing comparison state

Training complete for all four mechanismv2 runs. MSE/correctedscore/DRO full independent prediction/decoder replay passed; correctedscore objective audit and conditional report/visual gates unfinished. ClassifierNRE replay failed, not ongoing. SIMEX17model replay passed but paired scientific report unfinished. No active comparison process is established by this review; avoid describing unfinished work as running.
