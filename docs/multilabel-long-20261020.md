# One longer three-label density candidate

Human authorized one separate prediction job while the150M residual-Huber
pretraining job continues. `multi_gaussian_noise_recon_30e_v1` uses the tested
parent86 encoder/decoder, three-output diagonal Gaussian NLL with analytic
convolution of native quoted target errors, and masked reconstruction weight.03.
The targets are Teff, logg and FeH. Native error covariances between targets are
unknown; independent quoted errors are an approximation, not physical truth.
This candidate is chosen for stable analytic noise integration. There is no
established multilabel winner yet: earlier joint test outputs lack a completed
independent full comparison, and noise-aware/TTT superiority has not been shown.
Flow convolution and covariance learning are not silently substituted here.

OneGPU,8CPU,64GiB;30 passes over the same9M benchmarkTRAIN rows =270M
reconstruction presentations, not150M distinct stars. The separate150M job is
unchanged. Native19299fit/2118tune/2046calibration/97695test rows and original
quality/source/128XP gates are reused. Fresh matched encoder and head Adam states,
encoder LR1.25e-6/head1e-3,500headwarmup updates,32768effective/4096micro,
eight256-label batches per update. Prediction and reconstruction both withhold
astrometry/EBV/target aliases, retaining photometry and110XP.

Full model/head/bothAdam/RNG checkpoints every9M pass, with checksums and exact
state reload verification. All30 checkpoints retained. Selection uses the lowest
mean observed-label NLL per coordinate on the reserved tuning role only, with
first minimum retained on ties. No test/calibration-based selection or early
stopping. Selected state is restored for final test/calibration/TTT export; the
last270M training checkpoint remains separately preserved. Training budget and
selected-model counters are reported separately. Conformal calibration uses the
disjoint calibration role; marginal versus simultaneous boxes remain distinct.

Clean model predictions are primary. The existing three-step label-free
LayerNorm reconstruction refinement is archived as a secondary comparison on a
fixed512test/256calibration subset, with reset per object and calibration after
refinement. It is not selected on final test, nor declared a default improvement.
Current TTT independent audit reached all768refinements but failed a final
matched-clean byte-equality comparison against a differently batched baseline;
its original failure and diagnostics remain visible. This new run does not
silently promote that gate to passed. No calibrated physical-truth or best-model
claim follows from acceptance, producer reloads or reported native-label scores.
