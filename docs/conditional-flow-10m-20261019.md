# Conditional observed-FeH density comparison

Four matched, seed42, epoch72-parent, 10M reconstruction-presentation arms:
scalar affine MAF control; conditional NSF8; conditional NSF16; NSF8 with frozen
encoder. Other objective and sampling settings follow targeted-10m-20261019.yaml.
Every arm minimizes the exact change-of-variables conditional density NLL of
reported standardized native FeH. These are direct inverse conditional models,
not the failed forward-XP quadrature experiment, empirical Bayes deconvolution,
or calibrated physical-truth posteriors. No label-error convolution is included.

The scalar affine MAF is mathematically a conditional Gaussian, even with three
composed transforms. NSF8/16 supply flexible monotone scalar transforms. All
start at standard normal by zeroing each conditional hypernetwork final layer.
The initial mean/logvariance gate is absolute2e-6, accounting for float32 spline
roundoff; initial density and CDF are separately tested against standard normal.
The encoder/context determines density parameters; exact NLL provides encoder
gradients after warmup for the three trainable-encoder variants.

NSF tails are identity. A fixed bound is computed once, before head optimizer
construction, from full FIT standardized labels with pinned torchregress's
recommended_tail_bound (1.25 maxabs, clipped5..25). It never reads test labels.
The fit bound and number outside it are retained. This avoids the known fixed5
problem where valid extreme FIT labels receive no spline gradient. It does not
promise useful extrapolation beyond the fitted bound.

Point outputs contain deterministic conditional mean and logvariance, not
Gaussian distribution parameters. Scalar MAF moments are analytic via its affine
inverse. NSF moments use64 equal panels ×8 Gauss-Legendre nodes in base-normal
space on[-bound,bound], plus analytic identity-tail moments. All100620 test rows
also retain128-panel mean/variance for independent convergence assessment.
Density log score, transform-based CDF, and quantiles(.025,.16,.5,.84,.975) are
saved independently. Quantiles/CDF do not depend on moment quadrature. A later
report must not claim converged moment metrics when refinement fails declared
5e-5 relative+absolute tolerance; preserve the strict failure rather than
silently increasing tolerance. Independent CPU checkpoint replay remains
required before scientific ranking. Conditional moments must not be passed to
the legacy Gaussian scoring helper.

Local component evidence:8 focused tests cover identity, inverse Jacobian,
density normalization, CDF/quantile inversion, reload, encoder gradient,
fit-tail gradient, and independent SciPy adaptive quadrature. Actual parent72+
actual cohort128fit/test, three-head-step smoke-v2 passed all three head strict
reloads;64/128-panel mean differences0/1.04e-7/2.83e-7 for affine/NSF8/NSF16.
This is not the original mounted128 XP certificate, GPU/resume production proof,
or a scientific result. Runner integration and production original-unit gates
remain required.

The previous forward-XP p84fpd54 integral failure is preserved. Its actual trained
checkpoint FIT128 meanNLL/XP changes0.59403→0.48229→0.40856→0.33970→0.25547
for GH16/32/64/128/256. This is serious nonconvergence of a single local importance
proposal, not a reason to relax the gate. No unchanged forward-EIV retry is
proposed here. No torchregress source change is justified by this MSA integration
failure.
