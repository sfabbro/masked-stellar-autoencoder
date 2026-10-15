# Epoch57 learning-rate and effective-batch experiments

Nine separately submitted headless GPU experiments isolate standard-MAE optimization using `configs/pretrain.xp-lr-batch-e57.yaml`. All start from immutable epoch57 checkpoint SHA256 `9b033ae63a3b87c93dcfeffd989172f7888d029f77143e6cf315cf7a032ab2ca`, whose inherited LR is2.4636321046560533e-6. Do not change main pretraining.

The matrix is effective optimizer batches32768/8192/2048 × LR factors0.5/1/2, giving absolute LRs1.2318160523280267e-6,2.4636321046560533e-6,4.927264209312107e-6. Preserve Adam moments/steps and group settings; hold each LR fixed, without advancing the saved parent cosine scheduler. Microbatch512, seed42, dropout0.1, original masking ratios, original scaler and source-ID matched9M/1M benchmark remain common. Standard MAE is unchanged; spectral and photometry quantities are evaluation diagnostics.

Every run sees exactly20M training presentations: twice9M plus2M. Pass-boundary partial batches give612/2443/9767 optimizer updates respectively. Compare final checkpoints at equal presentations; also report quality versus updates and time. Smaller batches inherently have more updates and different Adam memory measured in examples. Do not call update-matched points data-budget matched.

`mask_chunk_rows:512` is opt-in: masks are generated in fixed512-row chunks before gradient accumulation, so changing effective batch boundaries preserves actual masks. The shuffled cache row stream stays identical. Final `training_pairing.json` and metrics record cumulative SHA256 over shuffled little-endian uint64 training-cache row indices and row-major bool input masks. Require exact final row/mask hash agreement across all9 runs, identical initial evaluation arrays and source/scaler identity, plus checkpoint reload probes. Old experiment configurations retain their original whole-effective-batch mask policy.

Loss normalization uses the hidden finite feature count over each actual optimizer batch; all microbatch losses use that common denominator. Partial pass tails are retained. Weight updates use norm clipping1.0 as before. Existing interval metrics include gradient norm, clipping fraction, current LR, actual effective batch, training seconds and rows/s. No per-step parameter-difference tensors are saved, avoiding an extra full model copy/update overhead.

On the measured prior micro512 pilots,20M training compute was22–25min, while full1M evaluations made total wall time approximately70min. Provision1.25–2GPUh per run excluding queues/setup/cache, revise after initial measured throughput. End-to-end timing is part of the result, not a guaranteed ETA.

Compare all-column/XP MAE, count-weighted BP/RP orders1–10/11–30/31–55, integrated spectral error, calibrated hidden/visible photometric agreement and downstream frozen representation probes. Seed42 alone is exploratory; confirm the chosen setting and unchanged control with independent training seeds before changing large-training policy.

An additional wider-LR grid uses factors4 and12 (9.854528418624213e-6 and2.956358525587264e-5) at batches8192 and2048. These four runs retain identical budgets, masks, source checkpoint, Adam state and all scientific checks. The first nine accepted runs remain immutable. Require finite objectives and report divergence rather than relaxing checks.
