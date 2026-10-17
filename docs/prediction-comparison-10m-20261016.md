# Prediction-loss and head comparison, 16 October 2026

The human authorized 10 million **reconstruction-row presentations per arm**,
using the same existing labelled subset. This is not 10 million independent
spectroscopic labels. Six separate native-FeH pilots are implemented; submission
and completion are recorded only in the accepted-job ledger.

## Existing losses

The main MSA run uses masked MAE reconstruction, not quantile supervision. The
recent verified native FeH and age adaptations used standardized-target Huber;
the distance/enriched frozen comparisons used ridge heads. The older general
fine-tuning implementation supports pinball quantiles (0.16/0.5/0.84), and the
separate `pipeline` model has quantile/joint heads, but its entrypoint currently
reads data rather than performing a production training run. These paths must
not be described as one already trained joint model.

## Bounded comparison

All arms use actual completed epoch72 checkpoint SHA
`9de961ce92f0e69838aed174fce6c47179639e23816baf61a45e63d614ff66ec`.
Checkpoint72 completed at20:03:07UTC; epoch73 was underway in the actual metrics
response. No checkpoint73/74 is claimed.

The six comparisons are:

- Plain MLP with Huber, the matched point-prediction control.
- The same MLP with Gaussian NLL, a distribution-family control.
- The same MLP with analytic Gaussian CRPS, changing the proper score.
- Plain MLP with Huber plus0.1 normalized-latent Rank-N-Contrast.
- Residual MLP with Huber, changing head architecture.
- Four-member BatchEnsemble with Huber, a **TabM-style** head, not official TabM.

The last distinction is deliberate: `torchregress` provides BatchEnsemble
layers, not an exact native implementation of official TabM or TabR. TabR's
retrieval machinery would require its own train-only index and memory/latency
design. The TabM paper supports it as a strong general baseline, but neither
that paper nor our Chang single-seed comparison selects a stellar winner.

Head parameter counts at256latent/128hidden dimensions are33,154plain,
41,762residual and35,210BatchEnsemble. These are comparable small heads,
not exactly parameter- or FLOP-matched models; all receive identical data and
update counts. Ensemble members receive their own loss, with member means
averaged for point predictions. Their spread is not certified uncertainty.

## Shared data, optimization and budget

Use the already audited enriched feature artifact's **native layer only**:
19,859strict benchmark-train/source-ID-split0 fits and100,620native split2 tests.
Native values, errors, flags and original rows remain authoritative in the
hashed HDF5. No PIGS/Li/DESI labels are added to these six comparisons. Input
photometry, all110XP coefficients and optional EBV follow the tested allowlist;
parallax, sky, proper motion, labels and unknown aliases are withheld from
supervised prediction. Inference uses all permitted actual observations, eval
mode and no artificial hiding.

Reconstruction replay uses all9M original benchmark **training** rows, followed
by a1M training-prefix repeat:10M presentations,9M unique reconstruction rows.
The1M benchmark validation partition is never replayed. Parent encoder/decoder
and Adam moments are restored identically. Fixed model LR1.25e-6 replaces the
cosine schedule explicitly; the head uses AdamW1e-3 and500common head-only
warmup updates. Reconstruction optimizer batch32768/microbatch16384, masking
XP0.5/other0.2, reconstruction weight0.01 under scored-feature normalization.
No spectral training term is added in this wave: holding reconstruction fixed
isolates prediction losses/heads; spectral reconstruction remains a diagnostic.

Each of306joint updates averages eight256-star supervised batches before the
update. This adds626,688supervised presentations plus128,000warmup
presentations. These754,688label presentations repeatedly sample the19,859fit
stars; they are additional to the10Mreconstruction presentations. This budget
differs from older native-label pilots, so compare the six new matched arms
against one another. The supervised, replay-row and reconstruction-mask streams
are hashed and must match across arms. Dropout seeds are reset per branch.

Two later residual-head arms reuse this parent, cohort, budget and stream:
`prediction_e72_residual_lds_huber_10m_v1` applies torchregress LDS weights to
Huber, with kernel width 2 in that library's 100-bin histogram units, fitted
once on all standardized native fit labels. `prediction_e72_residual_nll_labelerr_10m_v1`
adds the quoted native FeH variance to the predicted Gaussian variance in
standardized units. The completed residual Huber arm is their control and is
not resubmitted. Quoted errors are not a total calibrated uncertainty, and LDS
changes the population estimand. `prediction_e72_contrastive_flow_10m_v1` is
the same budget with a different head: an 8-d LayerNorm summary of the latent,
trained by torchregress `ContrastiveFlowLoss` as p(summary | FeH). Negatives
are fixed rolls of the 256-star label batch, shifts 1, 2, 4, 8, 16, 32, 64 and
128, so the shared row stream is not advanced. The reported point is a 201-point
grid maximum of that likelihood over the fit-label range plus 0.5 standardized
units on each side. It is not q(FeH | latent), not a posterior that multiplies
by the training-label frequency, and not a Gaussian, so this arm has no
coverage. Its initial predictions are not the zero head used by the other arms.

Three residual-Huber arms then ask whether XP measurement error, rather than
the label distribution, attenuates FeH. They use the reported coefficient
errors divided by the parent scale, one coefficient at a time. Photometry and
EBV are unchanged, and the 72-epoch pretraining is not repeated.
`prediction_e72_rc_shared_xp_10m_v1` shrinks every star by the fit-star rms
reliability. `prediction_e72_rc_per_star_xp_10m_v1` uses each star's own error
with that same signal variance. `prediction_e72_simex_xp_10m_v1` retrains at
the library's lambda grid, four simulations for every positive lambda. The
reported point is not torchregress's quadratic extrapolation: on the classical
curve 2/(2+lambda) that quadratic lands at 1.49 rather than 2, so each star is
extrapolated to lambda=-1 with a/(c+lambda). `NoisyTargetGaussianNLL` is the
quoted-error arm already completed. Functional errors-in-variables, input-noise
augmentation, pseudo-labels and orthogonal distance regression are not run.

RnC uses256-star contrast pools and L2-normalized latents, temperature2.
Accumulating eight losses does not create a2048-star negative pool. This is a
single-view auxiliary, not a claim to reproduce Chang's two independently
corrupted input views. Chang's manuscript used CE plus equal-bin CDF loss plus
RnC-N; that CDF loss is not exact CRPS in the physical target coordinate.

## Distribution interpretation and evaluation

Gaussian CRPS and NLL initially describe the **reported native FeH label**
distribution. They do not deconvolve measurement noise, resolve survey
systematics or supply latent physical-truth errors. The same Gaussian family
and train-only target scaling let the two scores be compared directly. No
arbitrary error floor or silent log-variance clipping is introduced; numerical
overflow/underflow fails. Huber's unused variance channel must not be used for
uncertainty reports.

Store predictions initially, after head warmup, and near3M/6M/9M/10Mreplay
presentations, recording the actual completed-batch counts. Final diagnostics
include point MAE/bias/RMSE, FeH tails, Gaussian CRPS/NLL in dex and68/95percent
coverage for the distribution-trained heads. Fixed10,000original-validation
rows/masks retain paired initial/final reconstruction and wavelength-space
spectral diagnostics. All test head/member outputs and validation decoder
outputs must reload exactly on the same device. Checkpoint optimizers are
preserved; saving alone does not certify resumed optimization.

Native test labels do not enter supervised updates or selection. Some stars
can have appeared in unsupervised pretraining/reconstruction replay; source-ID
label holdout does not imply unseen-star pretraining. No seed-independent
winner or SOTA claim follows from this first matched single-seed wave.

## Joint neural spline flow

A conditional NSF is a reasonable subsequent distribution model for Teff/logg/
nativeFeH, with age off initially. First materialize their native values,
units, errors and quality flags from the **already acquired** original surveys.
The current audited sidecar is FeH-only. `torchregress` joint-flow NLL rejects
feature-wise partial target masks; compare the flow and a joint Gaussian on
the same complete-target cohort, rather than silently filling missing labels.
Evaluate marginal CRPS **and** joint NLL/energy score; marginal CRPS alone cannot
assess dependence. A flow does not by itself correct target systematics or
introduce a valid raw-parallax measurement model. The proposed non-Bayesian
signed-parallax distance objective remains a separate integration step.

No extra survey acquisition is needed for this wave. The already accepted
enriched frozen comparison remains independently monitored without duplication.
