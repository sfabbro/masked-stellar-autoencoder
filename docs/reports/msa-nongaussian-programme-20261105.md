# Non-Gaussian multi-survey programme — 5 November 2026

Status: prepared, not yet a model result. This continues the user's request to
go beyond a single Gaussian and deliver a morning report after actual runs.

Four matched heads predict a complete **joint** (Teff, logg, Fe/H) vector:
an autoregressive rational-quadratic neural spline flow, a Gaussian-copula
spline residual flow, a four-component full-covariance mixture, and a Gaussian
residual control. The unrestricted autoregressive flow is not a collection of
independent scalar PDFs. Both flows have 16 bins and a FIT-only spline bound
derived with torchregress's tail-bound helper from integration-reference
coordinate extents. The autoregressive flow has three transforms with 128-wide
two-layer conditioners. Exact parameter counts are recorded by the producer.

The completed fine-tuned Gaussian checkpoint `efd57528…c26d` supplies a fixed
feature-only encoder, coordinate transform and relative-survey observation map.
Its optimizer is not resumed. The native parent/scaler/source contracts and
full checkpoint digests are checked before encoding. A uniform 65,536-star FIT
cache and 2,048-star TUNE cache retain exact Gaia IDs, source masks, native
measurement errors, natural XP availability and provenance bindings. Context
normalization uses FIT only. CAL/devtest/external groups are not opened in this
first screen. All heads see the same shuffled 64,000-star stream, at 1,000
updates, batch 64 and head LR 0.0003. No reconstruction or encoder SGD is used
in this density-family comparison. It is not a new full-catalogue training run.

For each star let the reference coordinates be `z = mu_ref(x) + L_ref(x) u`.
The fixed affine survey model has a Gaussian measurement likelihood for every
available source record, with all records stacked for the same physical star.
For an arbitrary learned joint density q, the identity is

```
p_q(y_all_sources | x)
  = p_0(y_all_sources | x)
    * E_{p_0(u | y_all_sources,x)}[q(u|x) / Normal(u;0,I)].
```

The Gaussian reference is an integration proposal; it does not force q to be
Gaussian. The Gaussian and mixture controls have exact analytic noisy-label
likelihoods. Flow training uses scrambled Sobol importance integration (256
particles; 1,024 for evaluation). No weights are clipped. Missing labels are
integrated rather than filled. Complete intrinsic predictions and joint draws
come from q conditioned on features only, **not** from the label-conditioned
integration proposal. Physical-coordinate Jacobians are retained implicitly
by the observation whitening and reference likelihood; a unit-change regression
checks that the density shifts correctly.

Before fitting and at candidate snapshots, both flows must meet median relative
ESS >=0.1, minimum >=0.01, independent-replica log-density gap p95 <=0.1 and
particle-doubling gap p95 <=0.1. The fixed 128-star numerical panel includes
source/missingness patterns, hot/cold/gravity/metallicity extremes and staged
XP absence. Raw per-star diagnostics and maximum gaps are preserved. A fitted
checkpoint is eligible for TUNE selection only if it passes the predefined
gate; final selected states face an independent, larger-particle check. A p95
gate is not an all-star guarantee. Numerical failures remain failures and are
never converted into scientific wins. Selection uses only existing TUNE roles.

Outputs include SHA-bound head checkpoints and their context transforms,
joint samples, marginal intervals, native-source PIT and noisy-label predictions,
likelihoods, training logs, source/exposure hashes and numeric-gate evidence.
Both environment locks now pin torchregress 0.3.0 at `2777b0de…d2a`; historical
runs retain their original source commits and locks. The project GPU environment is sufficient
for this task, which needs no GaiaXPy spectral-flux projection. No dependencies
are installed into a home `.local` directory.

The existing [100 K, 0.1 dex, 0.1 dex] survey floors remain illustrative; native
catalog covariance is explicitly assumed diagonal. The inherited observation
map is a fixed first-order approximation, not a calibrated physical scale.
Source systematics, parent exposure, input-noise context and independent physical
reference calibration remain limitations. The output will not call a smooth PDF
or a successful numerical gate proof of physical calibration.

Local preparation checks: 1,304 tests passed, 4 skipped on the updated library;
the ten focused math/lifecycle checks also pass after switching the NSF to the
library flow factory and FIT-derived bounds. Checks include independent SciPy
Gaussian integration, full correlated noise, missing/empty observations,
gradients, measurement-unit Jacobians, a genuinely non-Gaussian mixture integral,
and a complete synthetic four-head save/reload/predict lifecycle. Two tiny HDF5
tests now supply synthetic sufficient disk space, with a separate explicit
low-space rejection test; the production space guard is unchanged. A local
duplicate environment solve ran out of space, so only that incomplete generated
environment was removed and the already installed project environment reused.
The XP lock reuses the canonical library metadata from the freshly solved main
lock, leaves its platform packages unchanged, and passed Pixi's offline
`--check --no-install` validation. No fresh local XP installation is claimed.

After the first screen, eligible finalists may receive a second-seed or
tail-focused matched follow-up and a separate calibration-role evaluation.
The morning report must distinguish actual completed results, failures,
numerical eligibility and remaining scientific limits. No recurring automation
or unbounded grid is part of this programme.

Overnight refinement decision: the first 100/300-update checks exposed flow
integration error at 256 particles. The copula passed at 100 but missed the
replica-gap gate at 300; the unrestricted NSF missed both early gates. The
follow-up retains the exact original cache (`99f9e869…c5ed`), FIT stream,
initialization seed, architecture, optimizer and 1,000-update budget, but uses
1,024 training particles and 4,096 for checkpoint gates/scoring. Acceptance
thresholds stay fixed. Every candidate snapshot is now saved, including those
that fail a numerical gate. No encoder or survey-map fitting is repeated.
This is a numerical refinement experiment, not an independent training seed.

Evaluation-only CAL support copies the selected head's FIT normalization and
strict-loads a fresh instance under its original architecture. It fits no
parameters and does not reselect checkpoints. CAL diagnostics include native
ensemble CRPS and an independent, doubled-particle likelihood check on all
scored stars for flow heads. Full local suite before this follow-up: 1,314
passed, 4 skipped; focused tests also check fresh-instance reload and unbiased
ensemble CRPS against its independent pairwise definition.
