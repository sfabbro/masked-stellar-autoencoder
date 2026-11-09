# Matched seismic supervision experiments

Launched 7 November 2026, approximately 15:50 Perth. The existing large training remains a separate baseline. No experimental result has yet established whether seismic supervision, a learned seismic map or a different noise assumption is preferable.

## Questions and controlled treatments

All arms start from the earlier pooled NSF checkpoint (`c59ea40759bfe8f16e0f5553fdb6f7e99796fd2d146eac6f452d151daf0adff1`), before the explicit APOKASC mapping/supervision branch. Existing spectroscopic calibration and ancestor dependencies may remain: this is not a claim of a wholly seismic-independent training ancestry.

All arms use the exact expanded source-v5 inputs and native label pools (source SHA `a117ebc33bfd5d06a771eab6db1d4242d01e87b826b55c9dc8dc459da9ad3ec2`). Source maps initially equal identity. All controlled arms use APOGEE logg as the same explicit reference convention. This convention identifies the common coordinates; it is not an unbiased physical anchor. Consequently these comparisons are not pure replicas of the running production model, which instead uses a seismic identity anchor with a free APOGEE gravity map.

| Arm | Explicit seismic supervision | Seismic map | Additional independent scatter |
|---|---:|---|---:|
| off | 0 | Identity, zero seismic loss | 0 |
| identity | Full | Identity | 0 |
| affine | Full | Learned offset and positive slope | 0 |
| identity_floor | Full | Identity | 0.02 dex |
| affine_floor | Full | Learned offset and positive slope | 0.02 dex |
| identity_lowweight | 0.1 × full | Identity | 0 |
| context | Full | Positive-slope gravity map with temperature/iron-dependent offset | 0 |

The affine/context maps are constrained towards identity with explicit regularization. Context maps preserve conditional invertibility in gravity; all other coordinates pass through unchanged. The source-wide offset is a learned calibration parameter, distinct from a per-star random error floor. These are regularized point estimates, not a posterior over instrument systematic uncertainty. The 0.02 dex extra-scatter arms are sensitivity controls, not measured noise estimates.

## Matching and training budget

Each arm runs 1,500 updates, using the full admitted source pools rather than a small fixed training subset. This is a fixed-budget pilot, not a complete FIT epoch. Each step has 256 uniformly sampled source observation records, 32 seismic FIT records and 16/8/8 StarHorse auxiliary records. The same exact source/star schedule, model initialization and random seeds are used across treatments. Off and low-weight arms still compute the seismic draws/noise and keep the common denominator of 320, so removing seismic loss does not silently increase native-survey weights. Exact source/star batch hashes and reconstruction-block hashes must match before paired interpretation.

Two joint-training seeds each run all seven treatments: encoder, decoder, NSF and maps, with MAE reconstruction weight 0.03 over random blocks of the full 150M cache. A frozen-encoder suite runs off/identity/affine at the first seed, separating fixed-representation head/map adaptation from joint adaptation (one frozen seed is provisional). There are **17 trained treatment endpoints**, plus identical initialization readouts.

- `seismic_supervision_joint_s1_20261107_v1`: session `lffjg57w`.
- `seismic_supervision_joint_s2_20261107_v1`: session `mx7dv2bq`.
- `seismic_supervision_frozen_s1_20261107_v1`: session `ck66z83m`.
- Incremental CPU comparison `seismic_supervision_comparison_20261107_v1`: receipt records its session.

## Evaluation

Terminal readouts are fixed by budget, rather than separately choosing favorable checkpoints. Shared fixed TUNE/DEV native-label panels, all eligible seismic TUNE/DEV matches, all 12,017 matched TESS DEV references, reserved benchmarks and globular members are evaluated. CAL remains closed. Source-projected scores are separate from common predictions: a learned seismic map can improve catalogue agreement without certifying physical common gravity.

The incremental report includes native nonseismic TUNE CRPS, common APOKASC/TESS gravity residuals, interval widths and quoted-noise reference inclusion, gravity-dependent residual trends, matched-star MAE differences with star-bootstrap intervals, benchmark errors and cluster metallicity medians/NMAD. Bootstrap intervals are conditional on the finite training seeds and omit shared source systematics. Lower scatter alone is not an accuracy criterion. Cross-seed consistency and effects on the other targets matter alongside gravity.

[Live comparison report](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/seismic_supervision_comparison_20261107_v1/report.md) · [Comparison status](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/seismic_supervision_comparison_20261107_v1/manifest.json)

## Verification

The native CUDA readout preflight passed all seven treatments, two gradient steps each, including common and source-native prediction export. Exact batch and reconstruction hashes agreed. The initial launch failed before optimization because its bundle omitted the new scoring module; the corrected immutable bundle is preserved as `seismic_supervision_readout_preflight_20261107_v2`, session `rnq82i2b`. All full suites require its successful terminal status and use the same tested source hashes.

Local synthetic tests cover invertibility, initialization identity, unchanged non-gravity coordinates, matched schedules/loss normalization and the incremental paired report; existing source-map, FIT-record and pipeline checks also passed (14 checks in the combined run). No real model was run locally.

The three GPU suites have reached actual optimization. Both joint seeds show nonzero encoder/head/map gradients; the frozen control shows zero encoder gradient and nonzero head/map gradients, as intended. The comparison reporter was restarted as `seismic_supervision_comparison_recovery_20261107_v1` (session `ysjvul0o`) after fixing its handling of nine benchmark rows without Gaia IDs. Nullable integer parsing preserves exact IDs; missing-ID references are counted and excluded from crossmatches. The pre-readout failed report manifest is archived; the GPU experiments were unaffected. The updated synthetic reporting check also covers a missing-ID row and cluster membership parsing.
