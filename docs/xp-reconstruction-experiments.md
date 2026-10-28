# Bounded XP reconstruction experiments

Seven independent CANFAR headless jobs continue the same immutable epoch-27
checkpoint. Each uses 10M training rows, 1M fixed validation rows, and 20M
training presentations. Samples are bounded sequential prefixes of each shard,
with seed 42 and the original training/validation split. The scaler, model,
AdamW moments, optimizer batch (32768), microbatch (16384), and checkpoint LR
are shared. This is a controlled continuation comparison; the LR is held fixed.

| Experiment | Change from unchanged control |
|---|---|
| unchanged_control | Current stable embedding, MAE, whole-XP mask 0.5, other-feature mask 0.2 |
| mixed_coefficients | Whole XP hidden for 25% of stars; random half of coefficients hidden for another 50% |
| mixed_blocks | Same mixture, with a contiguous half-XP block hidden |
| mixed_bp_rp | Same mixture, hiding either BP or RP |
| xp_mse_control | XP Gaussian objective with fixed variance 0.1 and zero target noise; other targets retain MAE |
| torchregress_noisy_xp | Same fixed floor plus reported XP target-error variance, using torchregress NoisyTargetGaussianNLL |
| signal_weighted_xp | XP MAE weighted by a training-only estimate of signal variance fraction, floored at 0.05 |

The three masking experiments preserve the control's hidden XP coefficient
budget. The XP-MSE control separates the effect of Gaussian residual scoring
from the use of target uncertainties. Reliability is estimated on 100K
evenly spaced training rows and frozen. It is a heuristic, not a measured
source-specific Gaia relevance-basis cutoff.

The uncertainty experiment uses diagonal coefficient errors divided by the
saved scaler scales. Coefficients and their errors share the preprocessing
builder's G=8.5 normalization. Full covariance matrices and uncertainty in the
normalizing G magnitude are unavailable in the prepared table. Missing or
invalid errors fall back to the fixed residual variance. No predictive variance
head is added; this experiment does not estimate learned uncertainties.

## Readouts

Every evaluation uses the same fixed masks and full 1M-row validation sample:
current common mask, all XP visible, all XP hidden, half-coefficient random
mask, contiguous block mask, and BP/RP channel mask. Visible outputs are
excluded from the MAE score. The leading metrics are **all reconstructed-column
MAE** and **XP-only MAE**, in the shared RobustScaler units. The four input-only
columns are excluded because the decoder does not predict them. JSONL also
records each column's MAE and valid target count at initialization and every
5M presentations, including the final 20M result.

Residual JSON records median/zero-baseline skill, XP coefficient order groups
(1–10, 11–30, 31–55), magnitude bins, and coefficient |value|/sigma bins.
Those coefficient SNR bins are not source-spectrum SNR estimates.

GaiaXPy 2.1.2 BP v375wi/RP v142r calibration and merge matrices project a
deterministic 10K-row validation snapshot into G-normalized spectra. Report
integrated absolute residual divided by integrated absolute target flux and
wavelength MAE. Partial-mask reconstruction preserves measured visible
coefficients. The saved calibration matrices, wavelength grid, version and
hash identify this diagnostic. All 55 bases are retained; covariance-based
spectral error bars are not inferred.

Every result saves a full experiment-only checkpoint and verifies exact
forward prediction equality after reloading it. Manifests store the parent
checkpoint hash, source revision, package lock, sampling policy and validation
ID. Compare ordinary MAE, never the different optimized training objectives.
Only completed runs with identical parent and validation IDs are eligible for
a final comparison. The first suite uses one seed; small differences require
replication before changing production training.

## Launch and artifacts

Run `scripts/launch_xp_experiments.py` in the repository's Pixi `monitor`
environment with `--checkpoint`, `--git-ref`, `--git-sha`, `--output-root`, and
`--record`. `--dry-run` writes the seven planned jobs without submitting them.
Existing launch records are refused to prevent duplicate submissions.

Each job requests 4 CPUs, 64 GiB RAM and one GPU, installs the separate
`experiments/xp/pixi.toml` environment under `/scratch`, and writes to ARC.
`<output-root>/<run-id>.session.log` captures entrypoint, dependency installation
and training output. Each `<output-root>/<run-id>/manifest.json` tracks the job;
its `<run-id>/metrics.jsonl`, residual JSON/NPZ, checkpoint and verification JSON
hold the experiment results. The main 100-epoch pre-training job continues.
