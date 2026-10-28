# XP representation and latent-refinement pilots

`run_xp_representation_probes.py` is a read-only CANFAR headless executable. It
never writes the benchmark or saved pretraining checkpoints. Each output must be
new; an existing manifest prevents a duplicate run.

## Genuine stellar-label probes

The default label source is the established fine-tuning FITS file
`/arc/projects/k-pop/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits`.
Targets are `teff`, `logg`, `fe_h`, evaluated in kelvin and dex. Each checkpoint
uses its original feature order and frozen epoch-27 reference scaler, checked
against the immutable 10M benchmark. The FITS reader follows the established
reader's column/null conventions. An independent matched-source audit checks
all 110 XP coefficients and G for at least 128 FITS/benchmark stars, distinguishes
raw from G-normalized coefficients, and fails on ambiguous or inconsistent units.
No assumed normalization can silently enter the experiment. The audit streams
benchmark prefixes until enough overlap is found, avoiding a whole-file scan
when overlaps are plentiful.

Source-ID hashing gives 70% training, 15% tuning, 15% final test membership;
duplicate source IDs are reduced to the first catalogue row. Labels, uncertainties,
direct label aliases, RA/DEC and proper motions cannot enter the encoder.
The labelled split is independent of supervised training, but is **not certified
unseen in unsupervised pretraining**. This is a paired representation-quality
pilot, not a claim of wholly new-source generalization. `--split sky` instead
uses DEC >= 30 degrees as final test; the remaining rows are source-ID partitioned
into fitting/tuning. Sky mode requires valid DEC. No test measurements select
regularization, checkpoint or steps.

The encoder stays frozen for ridge and nonlinear-head probes. Ridge latent
standardization and target normalization fit only training rows; alpha is selected
from 0.01, 1, 100, 10000 on tuning rows. The same shuffled fitting rows and label
budgets (1k, 10k, 100k) are used for every checkpoint. A width-128 nonlinear head
trains for 1000 fixed steps with LR 1e-3. A separate encoder+head fine-tune uses
200 fixed steps, LR 1e-6 for the encoder and 1e-3 for the head, new AdamW state and
the same seed/batches. All adaptation starts from an independent model copy; no
head or encoder updates can contaminate the other budgets/tasks. Common adaptation
architecture/default dropout is used across checkpoints, intentionally avoiding
confounding representation comparison with inherited experiment dropout settings.

The default capped sample is 100k fitting, 20k tuning, 20k test rows. Missing
labels are filtered independently per target. Final source IDs and prediction
arrays are preserved in `probe_predictions.npz`; `results.json` includes physical
MAE/RMSE/bias, low-label results and sky/label-error subdivisions. Identity and sky
splits require separate jobs and outputs.

Example, using a verified saved final checkpoint (replace expected checksum):

```bash
pixi run --as-is python scripts/run_xp_representation_probes.py \
  --mode probes \
  --checkpoint /arc/projects/k-pop/msa_runs/xp-photometry-e47-20261013/photometry_spectral_mae/photometry_spectral_mae/checkpoint.pth \
  --checkpoint-sha256 EXPECTED_FINAL_CHECKPOINT_SHA256 \
  --fits-xp-units auto --split identity \
  --output /arc/projects/k-pop/msa_runs/xp-representation-20261014/photometry_spectral_identity
```

## Per-star frozen-decoder latent refinement

This is inference refinement, not encoder pretraining. Only the per-star latent
is optimized; encoder and decoder parameters remain frozen in eval mode. The
initial latent is obtained with the exact heldout mask. Every reconstruction
feature that is visible and finite contributes equally to per-star scaled MAE;
heldout truth never enters the objective. A quadratic prior penalizes departure
from the initial latent, divided by latent standard deviations fitted on 4096
benchmark training stars (floor 0.05 in latent units). The fixed final step is
scored; no validation-based stopping or selection occurs.

Full-XP masking hides all 110 coefficients and 20% of other input features.
Partial-XP masking hides 25% of coefficients and 20% of other features. Finite
hidden entries alone are scored. The default is 20k validation stars, 30 Adam
steps, LR 0.01 and prior weight 0.01. Samples are bounded prefixes of the already
sampled benchmark; train/validation sampled source-ID overlap is rejected.

```bash
pixi run --as-is python scripts/run_xp_representation_probes.py \
  --mode latent_refinement \
  --checkpoint /arc/projects/k-pop/msa_runs/xp-photometry-e47-20261013/photometry_spectral_mae/photometry_spectral_mae/checkpoint.pth \
  --checkpoint-sha256 EXPECTED_FINAL_CHECKPOINT_SHA256 \
  --refinement-mask full_xp --refinement-steps 30 --prior-weight 0.01 \
  --output /arc/projects/k-pop/msa_runs/xp-representation-20261014/photometry_spectral_refine_full
```

Run full/partial masking and chosen prior weights as independent headless jobs;
compare the same source IDs and masks within each masking policy. A gain here
supports inference refinement only; it does not establish a better encoder.

## Verification

```bash
pixi run --as-is pytest tests/test_xp_representation_probes.py -q
```

Tests verify duplicate-safe splits, target/alias removal, train-only ridge
scaling against an independent linear target, raw/normalized unit discrimination,
and exact heldout-value/gradient exclusion during latent refinement.
