# Remediation experiments — 2026-10-26

Authorized by the user: execute the label/input/logg/joint-head remediation plan and launch the necessary experiments; a genuinely coupled six-target NSF is mandatory. No scheduled monitor is recreated.

## First immutable submission batch

1. `remediation_inputs_v1`: CPU source/normalization/row-association audit of the selected failing stars, with separate photometry/XP/native-label and astrometry readiness.
2. `remediation_source_v1`: CPU uncapped sparse development source. Rebuild native quality directly from GALAH/APOGEE; preserve independent alpha/age/distance availability, original benchmark masks and source scales; expose measured-parallax input separately. Previously inspected TEST identities are consumed as development, never described as fresh unseen TEST.
3. `remediation_joint6_preflight_v1`: CUDA-only synthetic correlated six-dimensional Gaussian/MDN/NSF fits and numerical admission checks. The NSF uses one six-dimensional transform and whole-vector sampling. Arbitrary partial/noisy NSF likelihood is a finite-QMC importance estimate with explicit proposal correction; replica/refinement/ESS gates are separate from the historical strict numerical tests.

The three first attempts are independent. Training pilots require actual successful source/audit/head preflights, exact input/source/library/checkpoint bindings and an immutable published runner. No GPU pod waits indefinitely for missing prerequisites. Failed and ambiguous attempts are retained; absence/time does not authorize resubmission.

## Controlled supervised pilots

Huber point baseline, full-covariance Gaussian, full-covariance MDN and coupled six-target NSF, each with frozen encoder/no reconstruction, fine-tuned encoder/no reconstruction, and fine-tuned encoder with the matched reconstruction term. Additional Gaussian/NSF arms expose verified signed measured parallax on the same parent feature scaler. The parallax input experiment does not assume an independent parallax likelihood for parallax-conditioned survey labels.

Use expanded per-target FIT pools, no complete-case intersection or cohort caps. A finite pilot update budget is a sampled supervised pilot, not an assertion of full epochs over all labels. Log actual unique identities, per-target/source/gravity coverage, finite gradients and separate objective contributions. Reconstruction and supervised presentations are reported separately. Natural-XP inference uses evaluation mode with no random XP masking; genuine no-XP and deliberately masked-XP diagnostics remain separate.

Selection uses common development TUNE evidence, never a fresh-TEST claim. Required outputs include six label predictions/residuals, gravity-region/source/modality comparisons and matched Andrae comparisons; available PIT/proper scores, native whole-vector corner draws on common axes, and explicit missing-label/reference limitations. There is no PDF/PIT fabricated for a point baseline. Correlations must come from the fitted joint distribution; arbitrary artificial correlations are forbidden.

## Sequential follow-on gates

Only after pilot input/numerical/coverage results: source-scale and alternative-survey supervision controls; gradients/reconstruction/spectral and ordered-label ranking/CRPS/contrastive objective ablations with one change at a time; joint calibration evaluated on genuinely observed blocks. New public catalogue candidates require provenance/quality/overlap audit before becoming supervision. StarHorse posterior medians remain catalogue-reference targets, widths are not measurement errors. A six-target joint model does not itself certify a physical six-dimensional posterior or identify correlations absent from the observed label overlap.

Keep the original failures and all historical evidence. Do not rerun completed old training, TTT, density or reporting jobs. Native MSA execution is CANFAR headless only. Scheduled tasks remain off; new jobs are first-attempt submissions from immutable tested source.
