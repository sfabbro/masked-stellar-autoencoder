# Full-FIT common-reference training — 5 November 2026

Authorized continuation of the pilot: train the common stellar distribution,
shared encoder, masked reconstruction decoder and three relative survey heads
jointly on every primary-input-eligible FIT star. There is no FIT sample cap.

The warm start combines the prior fine-tuned reference encoder/decoder and
survey maps with the selected four-component full-covariance stellar mixture.
The Gaussian coordinate-head parameters remain fixed as a parameterization;
the encoder, mixture and survey maps learn. The physical predictive density
remains a complete non-Gaussian mixture. Survey offsets retain the library's
exact per-target zero-sum gauge, defining a relative common reference across
APOGEE, DESI DR1 RVS raw and GALAH. This does not establish an absolute physical
calibration.

Three full shuffled FIT passes use batch 1,024, microbatch 128. Supervised
predictions use naturally available inputs. A separate masked forward hides
50% of available XP coefficients and 20% of other reconstruction-prefix
features; its MAE reconstruction term has weight 0.03. Both losses update the
same encoder. Decoder LR is 1e-5, encoder LR 1.25e-6, mixture LR 3e-4 and survey
map LR 1e-4; survey-map regularization has weight 1e-4. Source error floors and
the first-order survey observation approximation remain explicit and unchanged.

Verified role-specific records are cached in RAM; the original IDs, target
masks, errors, provenance, repeat resolution and attachment bindings are kept.
The caller verifies each FIT batch against that role-restricted reader. Exact
unique IDs are shuffled without replacement each epoch, including the final
partial batch. All epoch identity streams and optimizer checkpoints are saved.
This is data-scale expansion on one GPU allocation, not a multi-GPU claim.

Only a fixed 16,384-star TUNE panel selects among epoch checkpoints. Devtest is
opened after the chosen checkpoint is fixed. Final natural-input inference
exports every primary-input-eligible star in FIT, TUNE, CAL, devtest, external
and excluded roles, retaining role and source identity. Each HDF5 contains
mixture weights, physical component means and Cholesky factors, exact common
mean/covariance and native survey records. Missing inputs remain missing.

The job produces matched native/common Kiel diagrams for complete FIT and
devtest cohorts, plus common-reference diagrams divided by predicted
metallicity. A narrower predictive-mean sequence is a diagnostic, not evidence
on its own of physical accuracy or calibrated joint uncertainty. The inherited
checkpoint's historical exposure remains a limitation even for devtest.

Native CUDA checks require nonzero finite gradients in encoder, decoder,
mixture and relative survey maps on the first two updates. Progress records
report reconstruction loss, label likelihood, masked counts, source exposure
and gradient norms. There is no Gaussian fallback and no new stellar inference
on the local laptop. Pixi/GPU environments and code are pinned at submission.
