# Multi-survey distribution pilot — 4 November 2026

Both pilots completed 1000 updates and passed producer full-checkpoint reload.
The [results, distributions and calibration plots](msa-multisurvey-distribution-20261104/index.md)
compare 1024 identical TUNE stars after exposure to 62,827 identical unique FIT
identities. Fine-tuning improves joint observation NLL from 5.6981 to 5.4843;
the paired difference is -0.2138 (star-bootstrap 95% interval -0.2582 to -0.1687).
Prefer the fine-tuned Gaussian as the current **baseline**, not a winner over
all distribution families or a calibrated physical catalogue.

Nominal 90% observed intervals vary by survey/target: fine-tuned DESI Fe/H is
88.3%, GALAH Fe/H 98.0%, APOGEE Teff 100% (n=79). For the 29 source references
with Fe/H<-1.5, fine-tuned MAE is 0.520 dex and mean 90% interval width is
2.148 dex. The seven >=8000 K references have 2786 K MAE and mean observed
90% interval width 10,315 K. These small panels expose failures and excessive
width; they do not certify population tail accuracy. Attractive joint samples
do not resolve these source-scale, error-model and extrapolation limitations.

The next usable baseline checkpoint is
`/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/multisurvey_gaussian_finetune_20261104_v1/final.pth`,
SHA256 `efd575285789928d2e445d7c11429a7809a35a12e6be29424bd26c48e154c26d`.
The frozen checkpoint and all source pins are in the results summary. All four
prediction NPZ hashes match producer receipts; CRC, exact integer-ID pairing,
unique FIT exposure, independent PIT and saved joint-draw moment checks passed.
See [verification and completion evidence](msa-multisurvey-distribution-20261104/verification.json).
Both task-owned CANFAR sessions are completed; local collectors/watchers have
ended and no recurring automation was created. ARC transfer-negotiation errors
were recovered through direct reads of remaining small files, with no model
rerun. Results were assembled on 5 November in Australia/Perth.

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
The three-target Gaussian heads are freshly initialized: initial TUNE scores
describe an untrained new head, not the parent's inherited Fe/H predictor.

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
The quicklook does not establish calibration by XP availability, metal-poor
tail, spatial holdout or genuinely independent physical references.

Launch and completion receipts will distinguish prepared, accepted, running,
failed and completed states. No scheduled automation or retry dispatcher is
part of this experiment.

Local verification: the first full locked-environment suite recorded 1290
passed, 4 skipped and two failures: sandbox loopback restrictions and missing
historical registry evidence. Four original ARC JSON files restored the seven
missing registry references, with bounded collection, JSON parsing and local
hash receipts in `experiments/xp/survey-registry-evidence-restoration-20261104.json`.
No evidence was fabricated and no test contract weakened. With local loopback
access, the final full suite passed **1293 tests, 4 skipped**, including the new
closed-form CRPS check. These ignored metadata files remain local, so a fresh
checkout still needs its referenced evidence. Repository lint, formatting of
new Python files, shell syntax and the two-arm launcher dry-run passed.

Both distinct runs were accepted once at immutable MSA
`0517c90880f97358bb9e352580300ca3ef7f286c`, published to the origin fork on
`codex/multisurvey-distribution-20261104`. Frozen session `cauxrwil` and fine-tuned
session `chgpndj3` both reached actual Running, installed the locked environment
and obtained H100 NVL MIG 1g.12gb devices. Their first manifests report running;
this does not yet establish optimizer progress or successful fitting. Live quota
at submission was 33 TB with 13.157 TB available. Exact launch receipts are in
`experiments/xp/multisurvey-distribution-submitted-20261104.json`.

The current local torchregress release is 0.3.0 (`2777b0de88121768434725ef227893cd35595d2a`,
4 November). It batches the multi-source likelihood and missing-pattern copula
calculations. The accepted pilots retain their immutable older library pin.
A scoped float64 synthetic comparison against the current likelihood passed
for missing/empty labels and full correlated noise: maximum value difference
7.11e-15 and gradient difference 3.55e-15. This is mathematical parity on that
panel, not an actual-pilot speed benchmark or validation of the entire release.
Receipt: `experiments/xp/torchregress-030-multisurvey-parity-20261104.json`.
