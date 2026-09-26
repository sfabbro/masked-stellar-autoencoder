# Stellar-parameter pipeline

Photometric stellar parameters for Gaia sources, including stars with no XP spectrum and stars missing most external surveys. DR3 XP (55 BP + 55 RP signed Hermite coefficients) is the first table, not the schema. A DR4 file is a new `n_bp` / `n_rp`.

The original trainer lives in `masked_stellar_autoencoder.training` (`pretrain_msa`, `finetune_msa`, `models.model`). Its CLI commands are `pixi run python -m masked_stellar_autoencoder.training.pretrain_msa` and `pixi run python -m masked_stellar_autoencoder.training.finetune_msa`. This catalogue pipeline is a separate path under `masked_stellar_autoencoder.pipeline`.

## Run

```bash
pixi run pytest tests/test_pipeline.py -q
pixi run python -m masked_stellar_autoencoder.pipeline.train \
  --config configs/pipeline.yaml --data /path/to/table.h5
```

The CLI reads a named HDF table and counts stars. It does not fit scalers, pretrain, fine-tune, or write a checkpoint. Losses and metrics come from torchregress. If that package is not installed, the pipeline adds `~/src/astroai/torchregress/src` (or `$WORK/astroai/torchregress/src` on CANFAR). zuko is optional. Without it the joint head is the low-rank Gaussian.

`configs/pipeline.yaml` rejects an old config that still has `feature_cols`.

## What a batch is

Training sees a `StellarBatch`: `source_id`, healpix, survey, `G`, and five groups (`xp_bp`, `xp_rp`, `photometry`, `astrometry`, `labels`). Each group has values, a missing mask, and either diagonal errors or a covariance. BP and RP covariances stay separate. A new survey is a row in `pipeline/registry.py`.

The loop iterates partitions. It does not concatenate a shard. Scaler statistics are a prefix of the training iterator, capped at `reservoir_rows` (1e6). Shuffle partitions before that pass.

## XP scaling

Tag `pogson_div_v1` in `pipeline/normalize.py`.

1. One Pogson factor per star, `F = 10**(-0.4*(G - G_ref))` with `G_ref = 8.5`. Divide coefficients and both covariances by that factor. `G` itself stays a magnitude feature. A missing `G` leaves the XP orders missing.
2. Per-order median and IQR on the train reservoir, then `asinh`.
3. Diagonal Jacobian `dz/dc = (1/F) * (1/s) / sqrt(1+u^2)`, applied as `Sigma_z = J Sigma J^T` separately for BP and RP.

`flux_state: raw` does step 1. `flux_state: already_scaled` skips it, so a table that was already divided is not divided again. Confirm that flag on one real file before a long run.

## Missing data

Missing is a mask, never a median fill. An order is missing when it is past `n_relevant_bases`, the star has no XP, `G` is missing in the raw flux state, the published error or variance is non-positive, or the scaled value is not finite. A non-positive `Teff` is a missing label (`log10` is not applied to a floor). Astrometry enters as signal-to-noise clipped to ±10. A non-positive astrometric error is missing, not an infinite signal-to-noise.

The parallax *label* is `asinh` of milliarcseconds. The parallax *input* is the clipped signal-to-noise. RA and DEC are not features.

## Training contract

Artificial masks apply only to stars that have XP: a contiguous span along order, or a full drop of BP, RP, or both. The reconstruction loss scores the orders that exist and were hidden, under the measurement covariance `Sigma_z`. There is no second noise draw. Present ancillary columns are hidden with probability 0.1 so the missing bit is learned on stars that have the survey. Bands that were already missing are not reconstruction targets.

Fine-tuning is quantile pinball at 0.16 / 0.5 / 0.84, a crossover penalty, a small reconstruction term (`recon_weight`), and the joint negative log-likelihood. Rare [Fe/H] uses inverse-histogram weights after scaling. The encoder is frozen for `freeze_epochs`, then trained at `encoder_lr`. Its AdamW weight decay is 0.

The checkpoint score is the NMAD of [Fe/H] (factor 1.4826 once) on the select split, in the metal-poor bin (`[Fe/H] < -1`) plus the XP-off eval forward. Both forwards are eval mode with the artificial mask off. A bin with fewer than two stars scores as infinite. Global Teff error does not choose the model. The joint term is logged and does not enter that score. Its gradient does reach the shared encoder.

CQR is fit per label on the calibrate split, in scaled space, with `alpha = 0.32` for the 0.16–0.84 interval. Coverage is quoted after the inverse transform. Splits are on `source_id`: train, select, calibrate, test. Leave-one-survey-out raises if a held-out source is still in train. The pretrain holdout is healpix. `regime_nll` reports span-mask NLL and full-XP-drop NLL on that holdout, in eval, with no optimizer step.

The flow ships only when its joint coverage on held-out stars is at least `joint_coverage_minimum` (0.9). Otherwise the published joint head is the low-rank Gaussian. Catalogue inference stays the quantiles.

## Not in this commit

HATS as a reader, a survey-selection shift, binned leave-one-survey-out tables, and torchregress residual plots are unwired. The functions they would call are in `pipeline/steps.py`.
