# Proposed coherent continuation, 2026-10-16

Status: proposal for discussion, not a launch record. Existing accepted IDs remain authoritative. No additional session was submitted for this proposal.

## What the evidence supports

- Matched epoch-47 spectral auxiliaries at weight 0.1 improved XP coefficient MAE in all three seeds (mean 0.614%) and spectral error (mean 2.714%). Adding the photometric auxiliary did not replicate uniformly. Removing dropout and spectral weight 0.3 were not improvements under tested settings. See `experiments/xp/results-followup-e47/comparison.md`.
- Thirteen paired epoch-57 LR/batch pilots favor conservative LR. Best observed XP MAE was 0.750220 at effective batch 32768 and fixed LR 1.2318160523280266e-6, versus 0.750835 for batch32768/LR2.4636321046560533e-6. This small single-seed difference is not a universal optimum. Smaller batches were not clearly better; higher LR worsened the tested outcomes. Microbatch controls memory, not optimizer update cadence.
- Native FeH fine-tuning improved the held-out native labels. Reconstruction weight 0.01 improved population MAE 0.0799 to 0.0780 dex and the 184-star FeH<-2 subset 0.3743 to 0.3619 dex. Balancing improved the tail further but changed the objective. These are downstream target results, not a proof about all stellar populations.
- Age fine-tuning improved agreement with conditioned model labels and correlated seismic references; absolute systematic offsets remain. Age should initially be disabled in the shared training objective.
- The completed distance pilot learned StarHorse log-distance labels. Its nearby Hipparcos reference performance was poor. It did not establish improvement or failure at large distances, nor did it test the raw-parallax likelihood proposed below.

## Parent and optimization

Fresh ARC metrics at 2026-10-16T19:19:15Z show main run rrrc2ihf within epoch72; latest completed epoch is71 at17:02:26Z. Its logged next LR is9.99618372939158e-6 following the epoch70 restart. A checkpoint73/74 has not been verified. Select the latest fully saved, hash-verified checkpoint at submission, without changing the main run or placing an idle GPU on standby.

Use one common parent and two matched arms, with identical data, source-ID partitions, row-attached masks, supervised/replay streams, steps/presentations, initial heads and optimizer policy. Effective reconstruction batch32768 and the existing microbatch16384 are the first settings. A fixed encoder LR near1.25e-6 is the conservative candidate; retain the parent's Adam moments and explicitly replace its cosine LR policy in both arms. Newly added heads need their own warmed-up optimizer groups; their LR is not inferred from the reconstruction sweep.

## Objective and scope

The preferred eventual joint continuation has a shared encoder/decoder and masked-label heads for Teff, logg, native FeH, and positive distance/inverse-distance. Age remains an optional branch with zero loss weight initially. Existing heads and separate pilot runners do not yet constitute a tested production joint runner.

Arm A: supervised parameter losses plus ordinary masked-MAE reconstruction replay. Arm B: the same objective and streams plus the established spectral auxiliary. These are matched joint-training arms; Arm A is not a pure unsupervised-MAE run once supervised labels are introduced. For supervised replay, reconstruction weight0.01 is a starting candidate under the previously tested normalization, and spectral weight0.1 is relative to its reconstruction term. Loss normalizations and gradient contributions must be checked before treating these weights as transferable. Missing labels contribute no supervised loss. No new labels are imputed or averaged across surveys.

Keep the current native-label sources as the primary supervision. Treat PIGS/Li as separately tagged optional additions pending the already accepted enriched-value result. DESI needs further uncertainty/scale treatment; Shi matches are not certified; photometric FeH is not a spectroscopic target. New survey acquisition is deferred as a proposed scope reduction for this round, rather than a prerequisite for continuation.

## Prediction policy

Training uses artificial masking and reconstruction. Catalogue prediction uses eval mode, all permitted actually observed inputs, genuine missingness masks, and no random feature hiding. Target columns and aliases are always excluded. Fixed artificial masks remain useful for separate reconstruction and missing-modality diagnostics. Random-mask averaging is not automatically a calibrated uncertainty estimate.

Primary distance inference is spectrophotometric: exclude measured parallax and its aliases from the distance input, and keep sky/proper-motion shortcuts excluded. A separately reported Gaia-assisted product may combine calibrated independent spectrophotometric and astrometric measurements with a frequentist likelihood. Do not input Gaia parallax to the distance predictor and then combine that same observation a second time. Existing pretraining may already encode astrometry of held-out sources; this must remain a limitation of independence claims.

## Non-Bayesian distance objective

Let s=log(d/kpc), d=exp(s) kpc, and q=exp(-s) mas. Optimize the raw measurement likelihood with mean q+z, where z is a declared Gaia zero-point model, preserving every finite signed parallax and valid formal error:

    L_pi = 0.5 * ((pi_observed - z - q)**2 / variance + log(variance))

No log/inversion/sign cut is applied to observed parallax. Initial variance is the measurement variance; any additional scatter or covariance requires explicit calibration, not an invented floor. This is frequentist likelihood fitting and introduces no explicit Galactic distance prior. The representation, training selection and regularization still impose learned population assumptions. Weak parallaxes cannot identify precise finite distances without informative features or additional assumptions; confidence sets can be very broad or unbounded. Do not use a numerical distance cap as if it were measured information.

A primary precedent is Hogg, Eilers & Rix (2018), https://arxiv.org/abs/1810.09468, which predicts positive spectrophotometric parallaxes while training on untransformed measured parallaxes and uses leave-out fitting. Its red-giant results do not establish performance for our data or architecture.

## Evaluation every three epochs

Propose a bounded nine-epoch comparison, with stored predictions at0/3/6/9 and epoch checkpoints. Fixed source-ID fitting, tuning, calibration and final-test roles; exclude all benchmark validation and declared external references from supervised fitting/tuning. Choose settings on tuning data, not repeated final-test inspection.

At each snapshot save actual paired IDs, predictions, masks, labels/uncertainties, source provenance and checkpoint hashes. Monitor stellar residuals by survey, Teff/logg, FeH bins and metal-poor selection completeness/purity; reconstruction by fixed mask regime and spectral wavelength; distance in raw parallax measurement space by magnitude, uncertainty and declared stellar regime. Avoid selecting distance bins solely by noisy positive observed parallaxes. Independent references are needed to compare accuracy against Gaia itself; Gaia cannot be scored against the same observation used as its supposed independent truth. Catalogue-conditioned distance agreement is reported separately.

The launch gate is a tested joint runner, verified parent/source inputs, full head/encoder reload checks, raw-parallax negative/zero edge-case and no-copy tests, and a small actual-input integration run. No new scientific result or submission is claimed here.


## Local preparation status after the discussion

A standalone raw-signed-parallax likelihood primitive now exists in `src/masked_stellar_autoencoder/training/parallax_likelihood.py`; it is not connected to a runner. Ten fresh unit/gradient tests pass. A constant-prediction numerical check on all4,166 preserved Hipparcos measurements, retaining159nonpositive values, matches independent NumPy objective and analytic gradients. This is not a trained-distance result or full actual-input model integration. The legacy fine-tune data helper imputes errors and row-splits; the proposed native joint path must instead retain explicit missing-label/error masks and source-ID partitions. Targeted lint passes; repository-wide lint currently reports501errors in unrelated existing local files. No new source commit, push or CANFAR submission was made.


The next actual-data readiness audit confirms that the current native sidecar materializes FeH only: Teff/logg and their native errors/target quality must still be projected from the existing original survey files. All672,007native complete-XP IDs have finite materialized Gaia parallaxes and positive formal errors, including6,115nonpositive parallaxes. Those are mathematical availability counts, not a complete Gaia quality/zero-point audit. Holding all99,574external reference IDs out of fit/tune leaves19,808strict native candidate fits. These counts do not alter the accepted FeH comparison or establish a joint training cohort. See `experiments/xp/results-stellar-improvement/joint-continuation-preparation-v1/native-joint-input-readiness.json`.


A standalone candidate spectrophotometric input policy now retains only named photometry/XP and optional EBV, excluding unknown columns by default. Together with the signed-parallax primitive its tests pass14cases. All31,550archived actual safe inputs match the completed parallax-free policy exactly, including a withheld-value/observed-bit counterfactual. This is input-policy evidence only: no latest-checkpoint encoder replay or full joint-runner integration has been performed. The production launch gates above remain outstanding.


Joint loss composition now has a synthetic numerical regression: ragged microbatches (including no usable labels) reproduce whole-batch losses and shared-parameter gradients for both MAE and spectral arms. Parallax local means require usable-label-count weighting against the full optimizer batch; reconstruction retains its separate scored-feature denominator. The existing spectral auxiliary is a hidden-XP-count-weighted per-star fractional flux error divided by the overall reconstruction scored-feature count. Its weight0.1 must retain that convention. Independent NumPy integration and flux-unit invariance pass. All17local preparation tests pass; actual-parent gradient balance and full runner integration remain outstanding.
