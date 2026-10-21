# Matched joint-density and conformal contrastive-flow screen

This screen compares the current torchregress density APIs with the earlier
native Teff/logg/FeH experiments. It does not change the running 150M training.

Three GPU fits use parent86, identical photometry+XP inputs, native quality masks,
FIT/tune/calibration/test roles, seed42, 10M benchmark reconstruction scans,
reconstruction weight0.03, 500 head-warmup updates and 306 joint updates:

- NSF conditional negative log likelihood (new-library matched control).
- NSF context-contrastive InfoNCE alone.
- NSF negative log likelihood plus 0.1 context-contrastive InfoNCE.

All joint objectives are divided by three to retain the previous per-label
scale. Context negatives are one fixed cyclic derangement of the same FIT-only
minibatch, sampled without replacement; duplicate source IDs are rejected.
Temperature1 and margin0 are fixed. Positive and negative context branches both
retain encoder gradients. These are conditional three-label normalized flows,
not binary density features, classifier NRE, or contrastive density shaping.
Pure ranking need not produce a useful conditional density despite the normalized
flow architecture. No noise-aware wrapper is composed around the contrastive
loss: that wrapper would integrate log_prob and bypass its contrastive forward.

Each checkpoint receives the previous mean-residual marginal/Bonferroni CP
baseline and, without further fitting, global and Mondrian joint density level
sets at90/95%. Mondrian boundaries are apparent-G tertiles fitted only on FIT.
Actual calibration labels determine score thresholds, never training or group
boundaries; test labels only measure outcomes. Full scores, role IDs, uniform
weights, boundaries, memberships and thresholds are archived. Independent NumPy
finite-sample order statistics must agree exactly with both library wrappers.
A joint level set is not a rectangular credible interval or a physical-truth
confidence region. No per-axis density interval helper is used for three labels.

TTT is disabled to isolate density training and calibration; its previous fixed
policy did not establish a general gain. Point estimates retain exactly256
antithetic Sobol inverse nodes and chunk16, with a CUDA128-context preflight
before SGD. Original128XP/source and native quality gates remain mandatory.
The earlier scalar contrastive and forward-inversion failures remain failures.

Primary evaluation: native joint NLL (include the fitted-label Jacobian),
per-label point MAE, FeH<-2 diagnostic tail, survey and observable-group
coverage for joint level sets, alongside the existing marginal CP boxes.
Use the earlier verified MAF and Gaussian controls with explicit library/budget
scope; the matched new NSF controls library drift. Density calibration cannot
improve a point estimate. Independent functional model replay and calibration
arithmetic, paired CSV/JSON/PNG/PDF and visual inspection precede rankings.
One seed, reused test data, measured native labels, diagonal quoted errors and
unsupervised overlap limit interpretation. Group calibration requires suitable
exchangeability and is not unrestricted conditional or physical-truth coverage.
