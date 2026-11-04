# Multi-survey distribution pilot — 4 November 2026

The first bounded fit uses the current three-survey atmospheric likelihood:
one full-covariance Gaussian over Teff, logg and Fe/H, conditioned on the MSA
latent, with source-specific relative spline maps and native measurement errors.
The Gaussian is the supported reference, not a claim that it beats every
torchregress distribution. The Gaussian-copula spline head remains a candidate;
its six-target single-reference likelihood cannot be substituted for the
multi-record survey likelihood without a tested noise-integration objective.

Two matched arms compare a frozen encoder with fine-tuning at 1.25e-6. Both
start from the completed 4.5B-presentation checkpoint (SHA c544548b…), use
1000 updates of 64 source-associated FIT stars, head LR 0.001 and the same seed,
initialization, source union and sampled identities. Reconstruction is off to
isolate supervised adaptation. This is a short pilot, not a complete pass over
the 1.7M-star FIT union. The sampler is uniform over source-associated FIT stars;
some associated rows can have no retained target and contribute zero likelihood.

The immutable existing FIT/TUNE roles are preserved. A deterministic 1024-star
TUNE panel is scored before and after training; CAL, devtest and external roles
are not opened. This is an inspected development comparison, not an untouched
test. The 4.5B parent has historical Fe/H supervision and incomplete exposure
independence. No claim of fresh-object validation is made.

Outputs retain canonical means and Cholesky covariance factors; ten stars have
8192 coupled draws each. Separate record-level arrays retain exact int64 Gaia
IDs, survey/target identities, observed means/variances, references and PITs.
Joint observation NLL, native-unit MAE/bias, 68%/90% observed-label coverage and
PIT histograms are reported per survey and target. The final full checkpoint
is SHA-bound and reloaded before final inference. Predictions use eval mode,
natural availability and no extra XP hiding.

Survey error floors [100 K, 0.1 dex, 0.1 dex] are inherited illustrative numerical
choices, not calibrated errors. Catalog covariance is explicitly diagonal;
the shared latent predictive covariance is full. The relative survey spline
likelihood uses first-order linearization. Its canonical scale is a relative
gauge without independent physical anchors. Neither good PITs under these
floors nor attractive corners would establish physical calibration. Alpha,
age, distance and Gaia RVS are outside this three-target pilot.

Launch and completion receipts will distinguish prepared, accepted, running,
failed and completed states. No scheduled automation or retry dispatcher is
part of this experiment.

Local verification: the full locked-environment suite recorded 1290 passed,
4 skipped and two failures. The loopback-socket test then passed with the
required sandbox permission. The remaining pre-existing registry test expects
seven ignored archival JSON paths absent from this checkout. No missing evidence
was fabricated or registry contract weakened. The 27 focused distribution,
trainer, preflight, model and pipeline checks and three new saved-array/runner/
launcher checks passed; repository lint, formatting of new Python files, shell
syntax and the two-arm launcher dry-run passed. This is not a fully green suite.
