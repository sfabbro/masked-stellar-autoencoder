# BRIEF v2 — training-architecture plan (2026-10-22, follow-up)

Read this with `BRIEF.md` (repo facts still valid). This v2 scope supersedes the
program-level plan for the current round.

## What the human asked for now (verbatim intent)

> "For now the plan is about designing the best training architecture. I will focus
> in parallel to get the best labels needed (SDSS, ESO, Gaia, LAMOST, DESI and GALAH
> all types of surveys). Use for now the data you have access to. On ARC we have
> enough quota (on /arc/projects/k-pop) or else I will ask more. Basically I want the
> best training for now for our goals, designing losses and inputs, statistically
> correct prediction heads for distance, ages, alpha and metrics. Also the plan should
> include as many smaller (10-20M stars) as necessary, but could also make experiments
> using our latest checkpoint of the big training."

Also: check `docs/reports/msa-density-failures-and-missing-experiments-20261022.md`
(73 lines) — every checkbox there must appear in the campaign or be explicitly
retired/deferred with a reason.

## Constraints / decisions implied

- **Labels are out of scope for this round**: the human runs SDSS/ESO/Gaia/LAMOST/
  DESI/GALAH ingestion in parallel. Design label-source hooks, but do **not** block on
  new labels. Use the existing native3 cohort and available sidecars.
- **Quota is not a blocker**: `/arc/projects/k-pop` is adequate (request more only if a
  concrete measurement says so).
- **Smaller experiments**: 10–20M-star runs (10M benchmark = 9M train + 1M val; the
  150M cache can be sub-sampled to 10M/20M unique rows). Plus fine-tunes from the
  **latest big-training checkpoint** (`residual_huber_e86_150m_30e_v1`, local states
  50M/1.35B/2.25B; final ~4.5B expected 24 Oct; parent86 also local).
- **Focus**: losses, inputs, statistically correct prediction heads for distance, age,
  alpha, and the metrics/calibration methodology. Not label acquisition, not the
  all-Gaia product release (still the eventual target, but this round is model design).
- **Evidence culture unchanged**: verified vs producer-only; no invented numbers; no
  tolerance relaxation; immutable outputs; locked/tuning roles; float64 scoring.

## Verified facts for this round (2026-10-22 ~16:00 UTC)

- HEAD `0476628`; CI 572 passed / 4 skipped. New doc untracked
  `docs/reports/msa-density-failures-and-missing-experiments-20261022.md`.
- `wetmolvn` (bounded FP64 CUDA certificate, 3 NSF checkpoints ×128 rows) is
  **Completed**, output dir local only has `session-api-first.log`; full results not
  yet fetched/verified. It does **not** clear full-role failures.
- `dd27e718` still Running, ~2.3B/4.5B presentations; latest local full state
  `checkpoint-002250000000.pth` (SHA from receipt; 410,367,373 B). ETA 24 Oct.
- Native3 cohort: 19,299 fit / 2,118 tune / 2,046 cal / 97,695 test / **141** FeH<−2.
  Scalar FeH cohort: 100,620 test / 185 FeH<−2. Do not mix.
- Existing 10M screen runner `scripts/run_multilabel_screen.py` supports only
  `kind in {huber, gaussian, flow}`, noise-aware diag Gaussian, NSF/contrastive NSF,
  TTT, CP. **Not wired**: full-cov Gaussian, low-rank Gaussian, MDN, quantile,
  expectile, SLS, SLSConformal, NFM, CFM. `training/joint_prediction.py` already has a
  full-covariance `ResidualJointHead` + signed-parallax NLL (distance math verified:
  `exp(-log d)` mas, no 1000× bug). `pipeline/steps.py` imports torchregress
  `MultivariateGaussianLoss`, `LowRankGaussianLoss`, `MultiQuantileLoss`, CQR.
- torchregress local HEAD `0e77eee` = pin `1d735c6` + `4e81a72`, `2db6fa1`, `8973bcd`
  (numerically relevant; `8973bcd` is the recommended bump for BPF identity tails and
  honoured capacity). All missing families exist in the library exports:
  `MDNLoss/MixtureDensityLoss/create_mdn_loss`, `MultivariateGaussianLoss`,
  `LowRankGaussianLoss`, `QuantileLoss/MultiQuantileLoss`, `ExpectileLoss/
  MultiExpectileLoss`, `SLSLoss`, `SLSConformal`, `FlowMatching`/NFM/CFM modules.
- Existing data assets usable now (no new labels): `enriched-feature-cohort.h5`
  (756,836 rows; 673,345 complete-XP). **Correction (07 §1c): the 83,491 "no-XP" rows are
  label-only** — 0/140 features, 0 astrometry, 0 role membership; the earlier "25,113
  labelled no-XP rows" claim does not reproduce. All XP-off evidence is synthetic masking.
  `joint-native-labels.h5` (3-label complete-case roles), 10M benchmark (ARC),
  150M cache (ARC, ~87 GB), `apokasc3_collection_v1/table4.dat` (seismic logg/age),
  `distance_feature_cohort_v1` / `age_feature_cohort_v1`, Hipparcos reference.
- CANFAR resources used so far: `dd27e718` 8 CPU / 137 GB / 1×H100 MIG 1g.12GB
  (expiry ~3 Nov). Typical short arms: 270M presentations ≈ 4.6 h; a 10M screen
  arm is small (307 updates) but staging/startup dominates. New sessions can request
  1 GPU; state whether 12/24/40 GB is needed per head family.

## Working rules for agents

- Read-only investigation; only write your own `docs/plan-20261022/0N-*.md`.
- No code/config/test/ledger edits, no job submissions, no git push.
- Local CPU only; use `pixi run`; no pip installs; no multi-GB downloads.
- Cite `file:line` and artifact paths. Separate verified / producer-only / proposed.
- Timebox: orient ≤10 min, write ≤25 min, then message the Lead and stop.
- Deliverable: ≤220 lines, tables/formulas over prose, ending with
  "what the Lead must put in the plan".
