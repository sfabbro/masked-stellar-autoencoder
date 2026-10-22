# Matched native three-target validation wave — 2026-10-22

## Purpose and scope

Twenty-four independent GPU fits extend the existing completed W1/W2 experiments with new seeds and a matched reconstruction-weight contrast. Each predicts the same quality-bound native Teff, logg and FeH, from photometry plus XP only, starting from preserved parent 86 with immutable torchregress 1d735c6. These are new tests, not validated scientific results.

The current production cohort and complete-case density runner support **three** native targets. The alpha, named-scale multi-teacher age and signed-parallax distance work require separate source/mask/error/units gates. This wave does not invent their labels or claim a six-target joint density. Age/distance prepared-embedding pilots are separate integration checks.

## Predeclared matrix

For each of the six combinations below, run seed 45 with reconstruction weights 0.01 and 0.03, and seed 46 with weight 0.03:

- Diagonal Gaussian, clean-label likelihood.
- Diagonal Gaussian, quoted-label-error-integrated likelihood.
- Four-component MDN, clean-label likelihood.
- Four-component MDN, quoted-label-error-integrated likelihood.
- NSF, clean-label NLL.
- NSF, quoted-label-error-integrated NLL.

All 24 IDs are enumerated in `configs/multilabel-validation-wave-20261022.yaml`. Noise-aware means integration over quoted label errors; it does not mean XP input-EIV. MDN uses 4 components and NSF uses its supported fixed 256-point moment estimator. TTT and library-TTA are disabled to avoid adaptation cost and confounding. Existing seed 42 results and Gaussian 43/44 fits are reused rather than retrained. No old run ID or same-seed identical objective/settings is resubmitted.

## Budget and matching

Each independent job scans 10M nominal existing benchmarkTRAIN reconstruction rows, the existing 9M followed by the necessary repeat. Total nominal budget is 240M presentations across 24 jobs, not 240M distinct stars or a new catalogue. Parent 86 SHA, native sidecar and independent native-quality gate, cohort role split, scaler, supervised schedule, 500 warmup steps, reconstruction optimizer, head optimizer and 256-point moment policy remain matched. Each arm must archive actual update/presentation counters, not infer them from nominal budget.

The seed 45 within-family 0.01-versus-0.03 contrast tests reconstruction strength. Seed45/46 at 0.03 test seed variation against existing matched seed 42; comparison across different library versions or head implementations must declare those differences. No test-based winner selection or automatic additional wave is authorized by this configuration.

## Runtime gates and evaluation

Before SGD require published source/library pins, actualCUDA, parent checkpoint SHA and strict load, original 128 XP/error association audits, native row/role/source/quality evidence, and the actual 128-context/256-node flow-memory preflight. Local schema/head construction tests do not replace these production gates.

Use isolated FP64 encoder/head evaluation with the original saved quadrature nodes. Preserve all paired inputs/source IDs, scaler/split/streams, snapshots, predictions, checkpoints and checksum receipts. After independent model/objective/decoder/stream checks, report native point metrics, actual predictive joint NLL/proper scores where available, marginal versus simultaneous residual conformal intervals, and per-survey counts. Report both global and the same measured native FeH<-2 population. Do not substitute Gaussian scores for flows or claim physical-truth coverage from noisy native labels.

The existing mean-residual conformal path is retained. The additional joint-density conformal branch is enabled only for the six new clean contrastive arms; all 18 base arms retain the original disabled setting. Quantile q50-versus-mean comparisons use existing saved quantile outputs postfit; they require no extra training job. Unsupported predictive samples/CDF metrics remain explicitly unavailable.

## Ensembles and additional targets

After successful independent verification, fixed predeclared equal mixtures can reuse matched existing seed 42 plus new 45/46 at reconstruction 0.03. No extra training job is needed. All three checkpoints remain necessary at inference; no aggregate encoder exists. Distribution metrics must score the predictive mixture, not fit a Gaussian to averaged moments. Preserve the distinction between predictive disagreement and calibrated uncertainty.

Before an age/distance/alpha six-target extension, finish their independent source associations, availability masks and unit/error semantics. Do not use complete native three-target rows as evidence that these other targets exist. Prepared-embedding age/distance pilots are scoped integration tests and do not certify shared six-target training.

## Clean NSF contrastive extension

Six additional matched reconstruction 0.03 jobs test pure context InfoNCE, NLL plus 0.1 InfoNCE, and NLL plus 1.0 InfoNCE, each at seeds 45 and 46. The base NLL seeds 45/46 are their matched controls. One cyclic source-distinct FIT-minibatch context negative, both gradient branches, temperature 1/margin 0 and division by three targets match the previously tested policy. No noise-aware contrastive composition is allowed: the current public head explicitly rejects it.

These six arms additionally archive global and FIT-G-tertile Mondrian joint-density conformal calibration, using disjoint calibration roles; these are joint level sets, not rectangular coordinate intervals or physical-truth confidence. TTT remains disabled. The matrix now totals 24 training jobs and 240M nominal presentations; the separate 27 evaluation-only jobs are defined by the root launch plan, totaling 51 jobs without claiming 510M training presentations.

Human steering: contrastive flow has potential in external tests. Added pure InfoNCE and NLL+0.1/1.0 InfoNCE at seeds45/46, each matched to existingNLL controls/recon.03. Per-arm density-level-set CP enabled without changing the original18-arm policy. This is torchregress ContrastiveFlowLoss likelihood-ratio/ranking, not the distinct earlier binary-feature CNF density-shaping experiment. Core ContrastiveFlowLoss/conformal code unchanged between pinned1d735 and currentlibraryHEAD; later BPFfactory fixes are separate. Noise-aware contrastive composition remains rejected explicitly rather than bypassing the measurement wrapper. Fullwave51 jobs:24x10M fits+15precisionrescoring+3conformal+6agepilot+3catalogue;240M newMSAreplay.
