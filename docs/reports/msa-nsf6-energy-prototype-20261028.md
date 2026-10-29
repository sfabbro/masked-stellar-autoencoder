# Masked energy score prototype

This is a synthetic-only prototype for a future full six-target NSF experiment.
It does not load stellar models or weights, modify the repository, submit jobs,
or clear the historical sparse-label NLL/integration failures.

`src/masked_stellar_autoencoder/training/multisurvey_masked_energy.py` scores [sample, star, coordinate] whole-vector draws after
grouping stars by the actual observed-coordinate mask. It uses fixed, positive
per-coordinate scales, an off-diagonal energy-score U-statistic, explicit
fixed positive uncertainties or unknown-error floors, and a supplied advancing
`torch.Generator` for diagonal Gaussian measurement noise. Weighted mean is
the self-normalized sum, so its use with a survey sampler must match that
sampler's declared target population. No nonnegativity clamp/gate is used.
For beta=1, the triangle inequality makes the exact-arithmetic U-statistic
nonnegative; floating-point rounding can differ.

The NumPy pairwise reference, all 63 nonempty six-target masks, coordinate
permutation, weighted reductions, known/unknown uncertainty guards, seeded
noise, and finite gradients at exact zero were checked. An independent analytic
normal-CRPS check includes Gaussian measurement convolution. A toy shared-latent
observation-map parameter and a randomly initialized actual `JointSixHead`
NSF both received finite nonzero gradients; the unused Gaussian proposal was
excluded from the toy optimizer. These are not CANFAR CUDA or scientific checks.

Validation, 2026-10-28:

```
env -u PYTHONPATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .pixi/envs/default/bin/python -m pytest \
  tests/test_multisurvey_masked_energy.py -q -p no:cacheprovider
# 10 passed in 10.49s
```

Ruff check and format passed for both files. The test imported the installed
torchregress revision `82ca199`; the parent independently checked that the
current `origin/main` merge `d126db7` has the same source tree.

Before a real pilot:

1. Bind
   scales, masks, source IDs, split, uncertainty/floor policy, and sampler
   weights to the experiment contract. Save and advance the NSF sample and
   observational-noise RNG states on every update/resume.
2. Add a genuine multi-survey source-record reader and source-specific
   observation maps. Each record for a star must be simulated from the same
   six-dimensional latent draw. Prevent duplicate/shared-spectrum records
   being treated as independent and model their noise covariances where needed.
   This prototype only accepts an explicitly declared diagonal error model; it
   neither constructs nor verifies survey maps, dependencies or physical truth.
3. Keep the old importance-estimated marginal likelihood gate unchanged.
   This sample score is not an observed-data NLL. A new flow-only gradient
   guard must not require a gradient for the unused Gaussian proposal; verify
   the real CUDA head, encoder and reconstruction gradients separately.
4. Evaluate selected checkpoints on the same predeclared TUNE/held-out stars
   with prediction/reference plots, PIT/coverage, proper per-label scores,
   energy and dependence-sensitive variogram diagnostics, and actual joint
   samples/corners. Sparse observed pairs and selection effects still limit
   what correlations and physical survey scales can be identified.
