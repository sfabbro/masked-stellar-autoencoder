# XP photometric consistency experiments

Five matched epoch-47 continuations use the verified 9M training / 1M validation benchmark, 20M presentations, inherited fixed AdamW learning rate, batch 32768/microbatch 512, seed 42 and identical masks/row order. The benchmark and production pretraining are unchanged.

- `unchanged_control`: ordinary MAE.
- `spectral_auxiliary_mae`: ordinary MAE plus 0.1 relative integrated spectral residual.
- `observed_photometry_mae`: ordinary MAE plus 0.01 observed-photometry Huber consistency.
- `photometry_spectral_mae`: both observed-photometry and spectral auxiliary terms.
- `anchored_decoder_consistency_mae`: observed-photometry consistency plus 0.002 detached decoder consistency on hidden photometric columns.

The fixed nine-band linear matrices are derived from torchgaiaxp's raw SDSS, PanSTARRS1 and Sky_Mapper g/r/i assets. Bundled `experiments/xp/photometry-projection.npz` retains all 55 BP/RP bases. No standardised additive offsets or colour corrections are applied. Optical edge bands and infrared surveys are excluded.

For each star, use visible observed XP coefficients and hidden predicted coefficients, undo RobustScaler, and project to filter flux. Convert observed survey magnitudes to flux divided by `10**((8.5-G)/2.5)` so both are in the benchmark's G-normalised units. True G is a loss-target normalisation; it never replaces a masked network input and this term supplies no gradient to the G prediction.

Fit each band's median synthetic-minus-observed flux bias and 1.4826*MAD residual scatter using 100K evenly spaced training-only stars. Divide flux by the band's median positive observed normalised flux before calibration; floor scatter at 0.02 of that reference. Require at least 1,000 valid training matches per band. Freeze these values; never fit them on validation or reconstructed coefficients. The floor includes observational scatter and is deliberately conservative when adding reported magnitude uncertainty in quadrature.

Observed consistency scores Huber(delta=1) of bias-corrected flux residual divided by the quadrature floor and the observed flux's propagated magnitude uncertainty. Average eligible bands per star, multiply by its scored hidden-XP count, and divide by the full optimizer batch's scored-entry count. This keeps the auxiliary coefficient independent of microbatching and band coverage. Missing/invalid reported uncertainties fall back to the frozen floor. Nonpositive observed flux or incomplete original XP is ineligible. Negative predicted flux remains penalised in flux space, rather than silently disappearing behind a logarithm.

Decoder consistency uses predicted survey magnitudes converted with the same target G normalisation, but detaches that target branch. It scores only artificially hidden finite survey measurements; observed-photometry consistency and ordinary MAE remain independent anchors. This is a one-way consistency regularizer rather than a free two-branch cycle that can drift jointly.

A CPU-only preflight fits and saves the training calibration and observed-XP versus measured-photometry baseline before GPU submission. Every GPU experiment independently repeats the identical training-only calibration and saves its checksum. Shared diagnostics include per-column/all-column/XP MAE on 1M validation stars, BP/RP order groups, spectra on deterministic 10K validation rows, and per-band synthetic versus measured flux/magnitude residuals separated by hidden/visible survey inputs. Raw magnitude diagnostics retain survey bias; calibrated flux diagnostics remove only the frozen training offset. Count nonpositive synthetic flux explicitly.

Compare only complete 20M-presentation runs with identical parent checksum, benchmark IDs, validation IDs/masks, scaler, projection/calibration hashes, steps and initial predictions. Preserve the calibration, logs, residual arrays and checkpoint reload probes. Small gains require longer or repeated-seed confirmation; calibration residuals can reflect variability and imperfect crossmatches as well as XP error.
